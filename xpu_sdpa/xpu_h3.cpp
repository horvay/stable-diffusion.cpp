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
// SD_XPU_H3_INT8=1 (default off; changes outputs slightly): W8A8 like xpu_qwen21. The checkpoint's int8 weights
// are used exactly as stored. Each linear's input is rotated by R, quantized per row to int8 and multiplied with
// oneDNN's s8 x s8 matmul (scales fused, bf16 out). LoRAs (SD_XPU_H3_LORA) are not merged (requantizing the base
// weights would add ~20x more error than the LoRA's own effect): A R is quantized per row and appended to the
// weight as extra output rows, so the same int8 GEMM also yields x A^T, and y += (x A^T) B^T in bf16. The rotate + quantize is
// fused into the op producing the input (xpu_h3_kernels.hpp); SD_XPU_H3_INT8_UNFUSED=1 uses separate torch
// rotate + quantize passes instead. Needs the icpx build (build_icpx.sh: DPC++ + SYCL oneDNN).
//
// Block math mirrors src/model/diffusion/minimax_h3.hpp TransformerBlock::forward.

#include <ATen/ATen.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/core/Device.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include "xpu_h3_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
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
    at::Tensor la, lb;      // optional LoRA: bf16 [r, in] and [out, r] (strength folded into lb)
    int64_t lora_rows = 0;  // int8 mode: q / s carry r extra rows = int8(A R) for the LoRA side path
};

struct Block {
    at::Tensor norm1, norm2, q_norm, k_norm;  // f32
    Linear qkv, out, fc1, fc2;
    at::Tensor ada_w, ada_b;  // f32 [6*3*hidden, tdim], [6*3*hidden]
};

std::vector<Block> g_blocks;
at::Tensor g_hadamard;  // bf16 [256, 256]
bool g_loaded = false;
bool g_fused  = false;  // int8 with every linear rotated by 256-groups: fused SYCL kernels
const bool g_int8 = [] {
    const char* v = std::getenv("SD_XPU_H3_INT8");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}();

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

const at::Tensor& hadamard_f32(int64_t n) {  // entries are +-2^-k, so this equals the bf16 one exactly
    static std::map<int64_t, at::Tensor> cache;
    auto& h = cache[n];
    if (!h.defined()) {
        h = hadamard(n).to(at::kFloat);
    }
    return h;
}

// SD_XPU_H3_LORA="a.safetensors[@strength],b.safetensors[@strength]": PEFT/ai-toolkit LoRAs on the block
// linears (diffusion_model.blocks.N.{attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2}.lora_{A,B}.weight).
// Applied as a low-rank side path on the unrotated input; several LoRAs are concatenated along the rank.
void load_loras(int verbose) {
    const char* spec = std::getenv("SD_XPU_H3_LORA");
    if (spec == nullptr || *spec == '\0') {
        return;
    }
    std::stringstream list(spec);
    std::string item;
    while (std::getline(list, item, ',')) {
        float strength = 1.f;
        const size_t at = item.rfind('@');
        if (at != std::string::npos) {
            strength = std::stof(item.substr(at + 1));
            item     = item.substr(0, at);
        }
        std::ifstream f(item, std::ios::binary);
        if (!f) {
            throw std::runtime_error("cannot open LoRA " + item);
        }
        uint64_t base = 0;
        auto index    = read_header(f, &base);
        int applied   = 0;
        for (size_t i = 0; i < g_blocks.size(); ++i) {
            const std::pair<const char*, Linear*> targets[] = {{"attn.qkv_proj", &g_blocks[i].qkv},
                                                                {"attn.out_proj", &g_blocks[i].out},
                                                                {"mlp.fc1", &g_blocks[i].fc1},
                                                                {"mlp.fc2", &g_blocks[i].fc2}};
            for (const auto& [name, lin] : targets) {
                std::string key;
                for (const char* prefix : {"diffusion_model.", "transformer.", "model.diffusion_model.", ""}) {
                    std::string k = std::string(prefix) + "blocks." + std::to_string(i) + "." + name;
                    if (index.count(k + ".lora_A.weight")) {
                        key = k;
                        break;
                    }
                }
                if (key.empty()) {
                    continue;
                }
                auto a      = load_tensor(f, base, index, key + ".lora_A.weight", at::kFloat);
                auto b      = load_tensor(f, base, index, key + ".lora_B.weight", at::kFloat);
                float scale = strength;
                if (index.count(key + ".alpha")) {
                    scale *= load_tensor(f, base, index, key + ".alpha", at::kFloat).item<float>() / static_cast<float>(a.size(0));
                }
                b = b * scale;
                if (!g_int8) {  // int8 mode quantizes A R itself (append_lora_rows_int8)
                    a = a.to(at::kBFloat16);
                    b = b.to(at::kBFloat16);
                }
                lin->la = lin->la.defined() ? at::cat({lin->la, a}, 0) : a;
                lin->lb = lin->lb.defined() ? at::cat({lin->lb, b}, 1) : b;
                ++applied;
            }
        }
        if (verbose) {
            std::fprintf(stderr, "[xpu_h3] LoRA %s @ %.2f on %d linears\n", item.c_str(), strength, applied);
        }
        if (applied == 0) {
            throw std::runtime_error("LoRA " + item + " matched no block linears");
        }
    }
}

// int8 mode: append int8(A R) (per-row scales) to the weight as extra output rows; the base rows stay as stored.
void append_lora_rows_int8(Linear& l) {
    if (!l.la.defined()) {
        return;
    }
    auto ar = l.la;  // f32 [r, in], unrotated input side
    if (l.group > 0) {
        const int64_t r = ar.size(0), k = ar.size(1);
        ar = at::matmul(ar.reshape({r, k / l.group, l.group}), hadamard_f32(l.group)).view({r, k});
    }
    auto sa     = ar.abs().amax(1, true).div_(127).clamp_min_(1e-12);  // [r, 1]
    auto qa     = at::round(ar / sa).to(at::kChar);
    l.lora_rows = ar.size(0);
    l.q         = at::cat({l.q, qa}, 0).contiguous();
    l.s         = at::cat({l.s, sa}, 0).contiguous();  // [out + r, 1]
    l.lb        = l.lb.to(at::kBFloat16);              // [out, r]
    l.la.reset();
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
    load_loras(verbose);
    if (g_int8) {
        for (auto& b : g_blocks) {
            for (Linear* l : {&b.qkv, &b.out, &b.fc1, &b.fc2}) {
                append_lora_rows_int8(*l);
            }
            c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
        }
    }
    g_fused = g_int8 && std::getenv("SD_XPU_H3_INT8_UNFUSED") == nullptr;
    for (const auto& b : g_blocks) {
        for (const Linear* l : {&b.qkv, &b.out, &b.fc1, &b.fc2}) {
            g_fused = g_fused && l->group == 256;
        }
    }
    // drop the load-time conversion scratch so Vulkan sees it as free
    c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
    g_loaded = true;
    if (verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_h3] loaded %d blocks (%s) from %s in %.1f s\n", n_blocks,
                     g_fused ? "int8 W8A8, fused kernels" : g_int8 ? "int8 W8A8" : "int8 weights, bf16 compute", path, s);
    }
}

// ---- int8 W8A8 (SD_XPU_H3_INT8): per-row quantize + oneDNN s8 x s8 -> bf16 matmul with fused scales ----

// xq[m, :] = round(x[m, :] / s[m]), s[m] = amax(|x[m, :]|) / 127; one work-group per row (x bf16, already rotated).
void quantize_rows(sycl::queue& q, const at::Tensor& x, at::Tensor& xq, at::Tensor& sx) {
    constexpr int WG = 256;
    const int64_t M = x.size(0), K = x.size(1);
    const auto* xp  = reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(x.data_ptr());
    auto* qp        = xq.data_ptr<int8_t>();
    auto* sp        = sx.data_ptr<float>();
    q.parallel_for(sycl::nd_range<1>(M * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t row = it.get_group(0);
        const int lid     = it.get_local_id(0);
        const auto* xr    = xp + row * K;
        float amax        = 0.f;
        for (int64_t k = lid; k < K; k += WG) {
            amax = sycl::fmax(amax, sycl::fabs(static_cast<float>(xr[k])));
        }
        amax              = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        const float scale = amax > 0.f ? amax / 127.f : 1.f;
        const float inv   = 1.f / scale;
        for (int64_t k = lid; k < K; k += WG) {
            qp[row * K + k] = static_cast<int8_t>(sycl::rint(static_cast<float>(xr[k]) * inv));
        }
        if (lid == 0) {
            sp[row] = scale;
        }
    });
}

struct Dnnl {
    dnnl::engine eng;
    dnnl::stream strm;
    std::map<std::tuple<int64_t, int64_t, int64_t>, dnnl::matmul> prims;  // (M, K, N)
};

Dnnl& dnnl_state() {
    static Dnnl d = [] {
        Dnnl s;
        auto& q = c10::xpu::getCurrentXPUStream().queue();  // torch's in-order queue keeps op ordering
        s.eng   = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        s.strm  = dnnl::sycl_interop::make_stream(s.eng, q);
        return s;
    }();
    return d;
}

// y = (xq * sx) (q * s)^T as bf16 [M, N]; xq int8 [M, K] rotated like the weights, sx f32 [M].
at::Tensor matmul_int8(const at::Tensor& xq, const at::Tensor& sx, const Linear& l) {
    using namespace dnnl;
    auto& d         = dnnl_state();
    const int64_t M = xq.size(0), K = xq.size(1), N = l.q.size(0);
    auto y          = at::empty({M, N}, xq.options().dtype(at::kBFloat16));
    memory::desc src_md({M, K}, memory::data_type::s8, memory::format_tag::ab);
    memory::desc wei_md({K, N}, memory::data_type::s8, memory::format_tag::ba);  // q is [N][K] row-major
    memory::desc dst_md({M, N}, memory::data_type::bf16, memory::format_tag::ab);
    memory::desc sx_md({M}, memory::data_type::f32, memory::format_tag::a);
    memory::desc sw_md({N}, memory::data_type::f32, memory::format_tag::a);
    auto& mm = d.prims[{M, K, N}];
    if (!mm) {
        primitive_attr attr;
        attr.set_scales_mask(DNNL_ARG_SRC, 1 << 0);      // per row of x
        attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);  // per output column
        mm = matmul(matmul::primitive_desc(d.eng, src_md, wei_md, dst_md, attr));
    }
    auto mem = [&](const memory::desc& md, void* p) {
        return sycl_interop::make_memory(md, d.eng, sycl_interop::memory_kind::usm, p);
    };
    mm.execute(d.strm, {{DNNL_ARG_SRC, mem(src_md, xq.data_ptr())},
                        {DNNL_ARG_WEIGHTS, mem(wei_md, l.q.data_ptr())},
                        {DNNL_ARG_DST, mem(dst_md, y.data_ptr())},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, mem(sx_md, sx.data_ptr())},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, mem(sw_md, l.s.data_ptr())}});
    return y;
}

// int8 matmul of a block linear, plus its LoRA side path when the weight carries LoRA rows
at::Tensor matmul_int8_lora(const at::Tensor& xq, const at::Tensor& sx, const Linear& l) {
    auto y = matmul_int8(xq, sx, l);
    if (l.lora_rows == 0) {
        return y;
    }
    const int64_t out = l.q.size(0) - l.lora_rows;
    return at::addmm(y.slice(1, 0, out), y.slice(1, out, out + l.lora_rows), l.lb.t());  // new contiguous [M, out]
}

// unfused int8 linear: x [N, in] bf16 -> rotate (torch) -> quantize rows (SYCL) -> int8 matmul
at::Tensor linear_int8(const at::Tensor& x, const Linear& l) {
    at::Tensor xr = x;
    if (l.group > 0) {
        const int64_t n = x.size(0), k = x.size(1);
        xr = at::matmul(x.reshape({n, k / l.group, l.group}), g_hadamard).view({n, k});
    }
    xr      = xr.contiguous();
    auto xq = at::empty(xr.sizes(), xr.options().dtype(at::kChar));
    auto sx = at::empty({xr.size(0)}, xr.options().dtype(at::kFloat));
    quantize_rows(c10::xpu::getCurrentXPUStream().queue(), xr, xq, sx);
    return matmul_int8_lora(xq, sx, l);
}

// y = (x R) (Q s)^T for x [N, in] bf16; returns bf16 [N, out]
at::Tensor linear(const at::Tensor& x, const Linear& l) {
    if (g_int8) {
        return linear_int8(x, l);
    }
    at::Tensor xr = x;
    if (l.group > 0) {
        const int64_t n = x.size(0), k = x.size(1);
        xr = at::matmul(x.view({n, k / l.group, l.group}), g_hadamard).view({n, k});
    }
    auto q = l.q.is_cpu() ? l.q.to(kXpu, /*non_blocking=*/true) : l.q;
    auto w = (q.to(at::kFloat) * l.s).to(at::kBFloat16);
    auto y = at::matmul(xr, w.t());
    if (l.la.defined()) {
        y.add_(at::matmul(at::matmul(x, l.la.t()), l.lb.t()));
    }
    return y;
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

        // Fused int8: per-token modulation row for the SYCL norm kernel (segments must not overlap; else unfused).
        bool fused = g_fused;
        at::Tensor mod_row;
        if (fused) {
            std::vector<int32_t> rows_mod(static_cast<size_t>(rows), -1);
            for (const auto& seg : segments) {
                for (int64_t r = std::max<int64_t>(seg.start, 0); r < std::min(seg.end, rows) && fused; ++r) {
                    fused         = rows_mod[r] == -1;
                    rows_mod[r]   = static_cast<int32_t>(seg.row);
                }
            }
            mod_row = at::from_blob(rows_mod.data(), {rows}, at::TensorOptions().dtype(at::kInt)).to(kXpu);
        }
        auto& queue  = c10::xpu::getCurrentXPUStream().queue();
        auto i8      = at::TensorOptions().dtype(at::kChar).device(kXpu);
        auto f32dev  = at::TensorOptions().dtype(at::kFloat).device(kXpu);
        // x rows [s, e) -> RMSNorm * w -> modulation (shift_i, scale_i) -> R -> int8, then the int8 linear
        auto norm_linear_fused = [&](const at::Tensor& w, const at::Tensor& mods, int64_t s, int64_t e, int shift_i, int scale_i,
                                     const Linear& l) {
            auto xq = at::empty({e - s, hidden}, i8);
            auto sx = at::empty({e - s}, f32dev);
            xh3::rms_mod_rot_quant(queue, x.data_ptr<float>() + s * hidden, w.data_ptr<float>(), mods.data_ptr<float>(),
                                   mod_row.data_ptr<int32_t>() + s, shift_i, scale_i, e - s, hidden, static_cast<float>(eps),
                                   xq.data_ptr<int8_t>(), sx.data_ptr<float>());
            return matmul_int8_lora(xq, sx, l);
        };
        auto bf = [](const at::Tensor& t) { return reinterpret_cast<const xh3::bf16*>(t.data_ptr()); };

        for (const auto& blk : g_blocks) {
            auto mods = (at::matmul(temb, blk.ada_w.t()) + blk.ada_b).view({trows * 3, 6, hidden}).contiguous();

            for (int64_t s = 0; s < rows; s += chunk) {
                const int64_t e = std::min(rows, s + chunk);
                auto qkv        = fused ? norm_linear_fused(blk.norm1, mods, s, e, 0, 1, blk.qkv)
                                        : linear(norm_mod(blk.norm1, mods, s, e, 0, 1), blk.qkv);
                auto qc         = qkv.slice(1, 0, inner).view({e - s, heads, head_dim});
                auto kc         = qkv.slice(1, inner, 2 * inner).view({e - s, heads, head_dim});
                q.slice(0, s, e).copy_(rope(rms_norm(qc.to(at::kFloat), blk.q_norm, eps).to(at::kBFloat16), s, e));
                k.slice(0, s, e).copy_(rope(rms_norm(kc.to(at::kFloat), blk.k_norm, eps).to(at::kBFloat16), s, e));
                v.slice(0, s, e).copy_(qkv.slice(1, 2 * inner, 3 * inner).view({e - s, heads, head_dim}));
            }
            auto o = at::scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                                      v.transpose(0, 1).unsqueeze(0));  // [1, H, rows, D]
            if (fused) {
                o = o.contiguous();
            }
            for (int64_t s = 0; s < rows; s += chunk) {
                const int64_t e = std::min(rows, s + chunk);
                if (fused) {
                    auto xq = at::empty({e - s, inner}, i8);
                    auto sx = at::empty({e - s}, f32dev);
                    xh3::attn_out_rot_quant(queue, bf(o), rows, heads, head_dim, s, e - s, xq.data_ptr<int8_t>(), sx.data_ptr<float>());
                    gated_add(matmul_int8_lora(xq, sx, blk.out), mods, s, e, 2);
                    continue;
                }
                auto a          = o.squeeze(0).slice(1, s, e).transpose(0, 1).reshape({e - s, inner});
                gated_add(linear(a, blk.out), mods, s, e, 2);
            }
            o.reset();

            for (int64_t s = 0; s < rows; s += chunk) {
                const int64_t e = std::min(rows, s + chunk);
                if (fused) {
                    auto uv         = norm_linear_fused(blk.norm2, mods, s, e, 3, 4, blk.fc1);
                    const int64_t f = uv.size(1) / 2;
                    auto xq         = at::empty({e - s, f}, i8);
                    auto sx         = at::empty({e - s}, f32dev);
                    xh3::swiglu_rot_quant(queue, bf(uv), e - s, f, xq.data_ptr<int8_t>(), sx.data_ptr<float>());
                    uv.reset();
                    gated_add(matmul_int8_lora(xq, sx, blk.fc2), mods, s, e, 5);
                    continue;
                }
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
