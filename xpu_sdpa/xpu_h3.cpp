// MiniMax-H3 transformer blocks on an Intel XPU through libtorch.
//
// sd.cpp keeps the embeddings, token refiner, time embedding, RoPE table and final layer; it hands the
// packed hidden state to xpu_h3_blocks() once per denoising step, and this library runs all
// transformer blocks with oneDNN bf16 matmuls and SDPA (SYCL kernels with 256 GRFs), which are several
// times faster on Arc Pro B70 than the Vulkan path.
//
// The int8 block weights (~19.5 GB) are resident on the device. sd.cpp calls xpu_h3_init() when it
// creates the H3 runner, before the text encoder runs, so Vulkan (budgeting against live free VRAM)
// fits the text encoder and VAE into what is left.
//
// Weights come straight from the ComfyUI int8 "convrot" checkpoint (SD_XPU_H3_WEIGHTS): per-row int8
// rotated weights W_rot = W R with R the 256-group normalized Hadamard. Linears compute
// y = (x R) (Q s)^T with full-precision activations; Q*s is expanded to bf16 one layer at a time.
//
// Block math mirrors src/model/diffusion/minimax_h3.hpp TransformerBlock::forward.

#include <ATen/ATen.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/core/Device.h>
#include <c10/core/CachingDeviceAllocator.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const c10::Device kXpu(c10::DeviceType::XPU, 0);

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end   = 0;
};

struct Linear {
    at::Tensor q;  // int8 [out, in] (rotated)
    at::Tensor s;  // f32  [out, 1]
    int64_t group = 256;
};

struct Block {
    at::Tensor norm1, norm2, q_norm, k_norm;  // f32
    Linear qkv, out, fc1, fc2;
    at::Tensor ada_w, ada_b;  // f32 [6*3*hidden, tdim], [6*3*hidden]
};

std::vector<Block> g_blocks;
at::Tensor g_hadamard;  // bf16 [256, 256]
bool g_loaded = false;

at::ScalarType to_scalar(const std::string& dtype) {
    if (dtype == "I8") return at::kChar;
    if (dtype == "U8") return at::kByte;
    if (dtype == "F32") return at::kFloat;
    if (dtype == "F16") return at::kHalf;
    if (dtype == "BF16") return at::kBFloat16;
    throw std::runtime_error("unsupported dtype " + dtype);
}

std::map<std::string, TensorInfo> read_header(std::ifstream& f, uint64_t* base) {
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    std::string header(n, '\0');
    f.read(header.data(), static_cast<std::streamsize>(n));
    *base = 8 + n;
    std::map<std::string, TensorInfo> out;
    static const std::regex entry(
        R"re("([^"]+)":\{"dtype":"([A-Z0-9]+)","shape":\[([0-9,]*)\],"data_offsets":\[([0-9]+),([0-9]+)\]\})re");
    for (auto it = std::sregex_iterator(header.begin(), header.end(), entry); it != std::sregex_iterator(); ++it) {
        TensorInfo t;
        t.dtype = (*it)[2];
        std::string dims = (*it)[3];
        size_t pos = 0;
        while (pos < dims.size()) {
            size_t comma = dims.find(',', pos);
            t.shape.push_back(std::stoll(dims.substr(pos, comma - pos)));
            pos = comma == std::string::npos ? dims.size() : comma + 1;
        }
        t.begin       = std::stoull((*it)[4]);
        t.end         = std::stoull((*it)[5]);
        out[(*it)[1]] = t;
    }
    return out;
}

at::Tensor load_tensor(std::ifstream& f, uint64_t base, const std::map<std::string, TensorInfo>& index,
                       const std::string& name, at::ScalarType target, bool keep_on_host = false) {
    auto it = index.find(name);
    if (it == index.end()) {
        throw std::runtime_error("missing tensor " + name);
    }
    const auto& info = it->second;
    auto host = at::empty(info.shape, at::TensorOptions().dtype(to_scalar(info.dtype)).pinned_memory(true));
    f.seekg(static_cast<std::streamoff>(base + info.begin));
    f.read(static_cast<char*>(host.data_ptr()), static_cast<std::streamsize>(info.end - info.begin));
    if (keep_on_host) {
        return host;  // pinned; streamed to the device per block
    }
    return host.to(kXpu).to(target);
}

std::string read_comfy_quant(std::ifstream& f, uint64_t base, const std::map<std::string, TensorInfo>& index,
                             const std::string& name) {
    auto it = index.find(name);
    if (it == index.end()) {
        return "";
    }
    std::string s(it->second.end - it->second.begin, '\0');
    f.seekg(static_cast<std::streamoff>(base + it->second.begin));
    f.read(s.data(), static_cast<std::streamsize>(s.size()));
    return s;
}

Linear load_linear(std::ifstream& f, uint64_t base, const std::map<std::string, TensorInfo>& index,
                   const std::string& prefix) {
    const std::string cfg = read_comfy_quant(f, base, index, prefix + ".comfy_quant");
    if (cfg.find("int8_tensorwise") == std::string::npos) {
        throw std::runtime_error(prefix + ": expected int8_tensorwise convrot weights");
    }
    Linear l;
    std::smatch m;
    if (std::regex_search(cfg, m, std::regex(R"("convrot_groupsize":\s*([0-9]+))"))) {
        l.group = std::stoll(m[1]);
    }
    if (cfg.find("\"convrot\": true") == std::string::npos && cfg.find("\"convrot\":true") == std::string::npos) {
        l.group = 0;
    }
    l.q = load_tensor(f, base, index, prefix + ".weight", at::kChar);
    l.s = load_tensor(f, base, index, prefix + ".weight_scale", at::kFloat).reshape({-1, 1});
    return l;
}

at::Tensor hadamard(int64_t n) {
    auto h4 = at::tensor({1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1., 1.}, at::kDouble).view({4, 4}) / 2.0;
    auto h  = at::ones({1, 1}, at::kDouble);
    while (h.size(0) < n) {
        h = at::kron(h, h4);
    }
    return h.to(kXpu).to(at::kBFloat16);
}

void load_weights(const char* path, int verbose) {
    const auto t0 = std::chrono::steady_clock::now();
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(std::string("cannot open ") + path);
    }
    uint64_t base = 0;
    auto index    = read_header(f, &base);
    int n_blocks  = 0;
    while (index.count("blocks." + std::to_string(n_blocks) + ".norm1.weight")) {
        ++n_blocks;
    }
    if (n_blocks == 0) {
        throw std::runtime_error("no transformer blocks found");
    }
    g_blocks.resize(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        Block& b            = g_blocks[i];
        b.norm1             = load_tensor(f, base, index, p + "norm1.weight", at::kFloat);
        b.norm2             = load_tensor(f, base, index, p + "norm2.weight", at::kFloat);
        b.q_norm            = load_tensor(f, base, index, p + "attn.q_norm.weight", at::kFloat);
        b.k_norm            = load_tensor(f, base, index, p + "attn.k_norm.weight", at::kFloat);
        b.qkv               = load_linear(f, base, index, p + "attn.qkv_proj");
        b.out               = load_linear(f, base, index, p + "attn.out_proj");
        b.fc1               = load_linear(f, base, index, p + "mlp.fc1");
        b.fc2               = load_linear(f, base, index, p + "mlp.fc2");
        b.ada_w             = load_tensor(f, base, index, p + "adaln_proj.linear.weight", at::kFloat);
        b.ada_b             = load_tensor(f, base, index, p + "adaln_proj.linear.bias", at::kFloat);
    }
    g_hadamard = hadamard(256);
    // drop the load-time conversion scratch so Vulkan sees it as free
    c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
    g_loaded = true;
    if (verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_h3] loaded %d blocks from %s in %.1f s\n", n_blocks, path, s);
    }
}

// y = (x R) (Q s)^T for x [N, in] bf16; returns bf16 [N, out]
at::Tensor linear(const at::Tensor& x, const Linear& l) {
    at::Tensor xr = x;
    if (l.group > 0) {
        const int64_t n = x.size(0), k = x.size(1);
        xr = at::matmul(x.view({n, k / l.group, l.group}), g_hadamard).view({n, k});
    }
    auto q = l.q.is_cpu() ? l.q.to(kXpu, /*non_blocking=*/true) : l.q;
    auto w = (q.to(at::kFloat) * l.s).to(at::kBFloat16);
    return at::matmul(xr, w.t());
}

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& w, double eps) {
    return x * at::rsqrt(x.pow(2).mean(-1, true) + eps) * w;
}

struct Segment {
    int64_t start, end, row;
};

}  // namespace

extern "C" int xpu_h3_init(int verbose) {
    try {
        if (!g_loaded) {
            const char* path = std::getenv("SD_XPU_H3_WEIGHTS");
            if (path == nullptr) {
                throw std::runtime_error("SD_XPU_H3_WEIGHTS is not set");
            }
            load_weights(path, verbose);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3] init error: %s\n", e.what());
        return 1;
    }
}

extern "C" void xpu_h3_release() {
    if (!g_loaded) {
        return;
    }
    g_blocks.clear();
    g_hadamard.reset();
    g_loaded = false;
    c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
}

extern "C" int xpu_h3_blocks(const float* x_host,
                             int64_t rows,
                             int64_t hidden,
                             const float* temb_host,
                             int64_t trows,
                             int64_t tdim,
                             const float* pe_host,
                             int64_t half_rot,
                             const int64_t* segs,
                             int64_t n_segs,
                             int64_t heads,
                             int64_t head_dim,
                             float* out_host,
                             int verbose) {
    try {
        using clock   = std::chrono::steady_clock;
        const auto t0 = clock::now();
        if (!g_loaded) {
            const char* path = std::getenv("SD_XPU_H3_WEIGHTS");
            if (path == nullptr) {
                throw std::runtime_error("SD_XPU_H3_WEIGHTS is not set");
            }
            load_weights(path, verbose);
        }
        static const int64_t chunk = [] {
            const char* v = std::getenv("SD_XPU_H3_CHUNK");
            return v ? std::atoll(v) : int64_t(16384);
        }();
        const double eps = 1e-5;
        auto fopt        = at::TensorOptions().dtype(at::kFloat);

        auto x    = at::from_blob(const_cast<float*>(x_host), {rows, hidden}, fopt).pin_memory().to(kXpu);
        auto temb = at::from_blob(const_cast<float*>(temb_host), {trows, tdim}, fopt).to(kXpu);
        auto pe   = at::from_blob(const_cast<float*>(pe_host), {rows, half_rot, 2, 2}, fopt).to(kXpu);
        auto cosv = pe.select(3, 0).select(2, 0).unsqueeze(1).to(at::kBFloat16);  // [rows, 1, half]
        auto sinv = pe.select(3, 0).select(2, 1).unsqueeze(1).to(at::kBFloat16);  // pe[r][1][0] = sin
        std::vector<Segment> segments(static_cast<size_t>(n_segs));
        for (int64_t i = 0; i < n_segs; ++i) {
            segments[i] = {segs[3 * i], segs[3 * i + 1], segs[3 * i + 2]};
        }

        const int64_t inner = heads * head_dim;
        const int64_t rot   = half_rot * 2;

        // apply fn(seg, a, b) to every segment's overlap with rows [s, e)
        auto for_segments = [&](int64_t s, int64_t e, auto&& fn) {
            for (const auto& seg : segments) {
                const int64_t a = std::max(s, seg.start), b = std::min(e, seg.end);
                if (a < b) {
                    fn(seg, a, b);
                }
            }
        };
        // modulated, normalized rows [s, e) as bf16
        auto norm_mod = [&](const at::Tensor& w, const at::Tensor& mods, int64_t s, int64_t e, int shift_i, int scale_i) {
            auto h = rms_norm(x.slice(0, s, e), w, eps);
            for_segments(s, e, [&](const Segment& seg, int64_t a, int64_t b) {
                auto part = h.slice(0, a - s, b - s);
                auto m    = mods[seg.row];
                part.mul_(1 + m[scale_i]).add_(m[shift_i]);
            });
            return h.to(at::kBFloat16);
        };
        auto gated_add = [&](const at::Tensor& update, const at::Tensor& mods, int64_t s, int64_t e, int gate_i) {
            for_segments(s, e, [&](const Segment& seg, int64_t a, int64_t b) {
                x.slice(0, a, b).add_(update.slice(0, a - s, b - s).to(at::kFloat) * mods[seg.row][gate_i]);
            });
        };
        auto rope = [&](const at::Tensor& t, int64_t s, int64_t e) {  // t [n, heads, head_dim] bf16
            auto c  = cosv.slice(0, s, e);
            auto sn = sinv.slice(0, s, e);
            auto a  = t.slice(2, 0, half_rot);
            auto b  = t.slice(2, half_rot, rot);
            return at::cat({a * c - b * sn, a * sn + b * c, t.slice(2, rot, head_dim)}, 2);
        };

        // per-step buffers, reused by every block; only attention needs full-length q/k/v
        auto bopt = at::TensorOptions().dtype(at::kBFloat16).device(kXpu);
        auto q    = at::empty({rows, heads, head_dim}, bopt);
        auto k    = at::empty({rows, heads, head_dim}, bopt);
        auto v    = at::empty({rows, heads, head_dim}, bopt);

        for (const auto& blk : g_blocks) {
            auto mods = (at::matmul(temb, blk.ada_w.t()) + blk.ada_b).view({trows * 3, 6, hidden});

            for (int64_t s = 0; s < rows; s += chunk) {
                const int64_t e = std::min(rows, s + chunk);
                auto qkv        = linear(norm_mod(blk.norm1, mods, s, e, 0, 1), blk.qkv);
                auto qc         = qkv.slice(1, 0, inner).view({e - s, heads, head_dim});
                auto kc         = qkv.slice(1, inner, 2 * inner).view({e - s, heads, head_dim});
                q.slice(0, s, e).copy_(rope(rms_norm(qc.to(at::kFloat), blk.q_norm, eps).to(at::kBFloat16), s, e));
                k.slice(0, s, e).copy_(rope(rms_norm(kc.to(at::kFloat), blk.k_norm, eps).to(at::kBFloat16), s, e));
                v.slice(0, s, e).copy_(qkv.slice(1, 2 * inner, 3 * inner).view({e - s, heads, head_dim}));
            }
            auto o = at::scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                                      v.transpose(0, 1).unsqueeze(0));  // [1, H, rows, D]
            for (int64_t s = 0; s < rows; s += chunk) {
                const int64_t e = std::min(rows, s + chunk);
                auto a          = o.squeeze(0).slice(1, s, e).transpose(0, 1).reshape({e - s, inner});
                gated_add(linear(a, blk.out), mods, s, e, 2);
            }
            o.reset();

            for (int64_t s = 0; s < rows; s += chunk) {
                const int64_t e = std::min(rows, s + chunk);
                auto uv         = linear(norm_mod(blk.norm2, mods, s, e, 3, 4), blk.fc1);
                const int64_t f = uv.size(1) / 2;
                auto act        = at::silu(uv.slice(1, 0, f)) * uv.slice(1, f, 2 * f);
                uv.reset();
                gated_add(linear(act, blk.fc2), mods, s, e, 5);
            }
        }
        q.reset();
        k.reset();
        v.reset();

        auto host_out = x.to(c10::Device(c10::DeviceType::CPU));
        std::memcpy(out_host, host_out.data_ptr<float>(), static_cast<size_t>(rows * hidden) * sizeof(float));
        // keep only the weights resident between steps so the Vulkan side has room (VAE decode)
        x.reset();
        cosv.reset();
        sinv.reset();
        pe.reset();
        c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
        if (verbose) {
            const double s = std::chrono::duration<double>(clock::now() - t0).count();
            std::fprintf(stderr, "[xpu_h3] %zu blocks, rows=%lld, segments=%lld: %.2f s\n", g_blocks.size(),
                         (long long)rows, (long long)n_segs, s);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3] error: %s\n", e.what());
        return 1;
    }
}
