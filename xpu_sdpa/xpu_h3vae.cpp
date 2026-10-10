// MiniMax-H3 video VAE decoder on an Intel XPU through libtorch.
//
// sd.cpp decodes H3 video in temporal windows of 7 latent frames, each split into 256 px tiles with 0.25
// overlap (MiniMaxH3VideoVAERunner::decode). With SD_XPU_H3VAE_LIB set, every tile's decode graph
// (post_quant_conv, x_embedder, 36 transformer blocks, norm_out, proj_out, unpatchify, pixel
// de-normalization) runs here instead of on Vulkan. The tiling, window plan and temporal blending stay in
// sd.cpp, so the numerics of the tile layout are unchanged.
//
// The decoder weights (~4.8 GB, SD_XPU_H3VAE_WEIGHTS = the fp16 VAE safetensors) load on the first tile and
// stay resident on the device across tiles and windows until xpu_h3vae_release(). release(keep_host = 1)
// moves them to pinned host memory first, so the next load is a ~0.3 s upload instead of a file read
// (used by the multi-shot runner, which hands the GPU back to the transformer blocks between shots).
//
// SD_XPU_H3VAE_DTYPE=bf16 (default) | fp16 | f32: dtype of the big linears and attention. Residual stream,
// norms, x_embedder, post_quant_conv and proj_out (f32 in sd.cpp too) stay f32.
// SD_XPU_H3VAE_DEVICE=xpu (default) | cpu: cpu is only for reference checks without a GPU.
//
// Block math mirrors src/model/vae/minimax_h3_vae.hpp (Decoder / DecoderBlock / DecoderAttention /
// DecoderFeedForward):
//   x += attn(RMSNorm(x) * norm1) * scale1;  x += ff(RMSNorm(x) * norm2) * scale2
//   attn: to_qkv -> per head [q | k | v] (64 each); RMSNorm(q), RMSNorm(k) over head_dim without weight;
//         split-half RoPE on the first 2 * half_rot dims; full (unmasked) SDPA, scale 1/sqrt(64); to_out
//   ff:   w1 -> [gate | up]; w2(silu(gate) * up)
// Tokens: the tile's latent patches (t, h, w order, w fastest), 4 register tokens, 1 zero token.

#include <ATen/ATen.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDim       = 2048;
constexpr int64_t kHeads     = 32;
constexpr int64_t kHeadDim   = 64;
constexpr int64_t kFfInner   = 8192;
constexpr int64_t kLatentC   = 24;
constexpr int64_t kRegisters = 4;
constexpr int64_t kPatch     = 16;
constexpr int64_t kPatchT    = 4;
constexpr int64_t kOutC      = 3;
constexpr double kEps        = 1e-5;

const c10::Device g_dev = [] {
    const char* v = std::getenv("SD_XPU_H3VAE_DEVICE");
    return (v != nullptr && std::strcmp(v, "cpu") == 0) ? c10::Device(c10::DeviceType::CPU) : c10::Device(c10::DeviceType::XPU, 0);
}();
const bool g_on_xpu = g_dev.is_xpu();

const at::ScalarType g_cdt = [] {
    const char* v = std::getenv("SD_XPU_H3VAE_DTYPE");
    if (v == nullptr || std::strcmp(v, "bf16") == 0) {
        return at::kBFloat16;
    }
    if (std::strcmp(v, "fp16") == 0 || std::strcmp(v, "f16") == 0) {
        return at::kHalf;
    }
    if (std::strcmp(v, "f32") == 0 || std::strcmp(v, "fp32") == 0) {
        return at::kFloat;
    }
    std::fprintf(stderr, "[xpu_h3vae] unknown SD_XPU_H3VAE_DTYPE=%s, using bf16\n", v);
    return at::kBFloat16;
}();

void empty_cache() {
    if (g_on_xpu) {
        c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
    }
}

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end   = 0;
};

at::ScalarType to_scalar(const std::string& dtype) {
    if (dtype == "F32") return at::kFloat;
    if (dtype == "F16") return at::kHalf;
    if (dtype == "BF16") return at::kBFloat16;
    throw std::runtime_error("unsupported dtype " + dtype);
}

struct SafeTensors {
    std::ifstream f;
    uint64_t base = 0;
    std::map<std::string, TensorInfo> index;
    std::string path;

    explicit SafeTensors(const std::string& p)
        : f(p, std::ios::binary), path(p) {
        if (!f) {
            throw std::runtime_error("cannot open " + p);
        }
        uint64_t n = 0;
        f.read(reinterpret_cast<char*>(&n), 8);
        std::string header(n, '\0');
        f.read(header.data(), static_cast<std::streamsize>(n));
        base = 8 + n;
        static const std::regex entry(
            R"re("([^"]+)":\s*\{\s*"dtype":\s*"([A-Z0-9]+)",\s*"shape":\s*\[([0-9, ]*)\],\s*"data_offsets":\s*\[([0-9]+),\s*([0-9]+)\]\s*\})re");
        for (auto it = std::sregex_iterator(header.begin(), header.end(), entry); it != std::sregex_iterator(); ++it) {
            TensorInfo t;
            t.dtype = (*it)[2];
            std::stringstream ds((*it)[3].str());
            std::string d;
            while (std::getline(ds, d, ',')) {
                if (d.find_first_not_of(' ') != std::string::npos) {
                    t.shape.push_back(std::stoll(d));
                }
            }
            t.begin         = std::stoull((*it)[4]);
            t.end           = std::stoull((*it)[5]);
            index[(*it)[1]] = t;
        }
    }

    // host tensor as stored (pinned when the target is the XPU)
    at::Tensor read(const std::string& name) {
        auto it = index.find(name);
        if (it == index.end()) {
            throw std::runtime_error("missing tensor " + name + " in " + path);
        }
        const auto& info = it->second;
        auto host        = at::empty(info.shape, at::TensorOptions().dtype(to_scalar(info.dtype)).pinned_memory(g_on_xpu));
        if (static_cast<uint64_t>(host.nbytes()) != info.end - info.begin) {
            throw std::runtime_error("size mismatch for " + name);
        }
        f.seekg(static_cast<std::streamoff>(base + info.begin));
        f.read(static_cast<char*>(host.data_ptr()), static_cast<std::streamsize>(info.end - info.begin));
        if (!f) {
            throw std::runtime_error("short read for " + name);
        }
        return host;
    }
};

struct Lin {
    at::Tensor w, b;  // w [out, in], b [out]
};

struct Block {
    at::Tensor norm1, norm2, scale1, scale2;  // f32 [kDim]
    Lin qkv, out, w1, w2;                     // compute dtype
};

struct Model {
    at::Tensor pq_w, pq_b;    // post_quant_conv as a [24, 24] linear, f32
    Lin embed;                // x_embedder [2048, 24], f32
    at::Tensor registers;     // [4, 2048] f32
    std::vector<Block> blocks;
    at::Tensor norm_out_w, norm_out_b;  // f32
    Lin proj_out;                       // [3072, 2048], f32 (force_f32 in sd.cpp)
    at::Tensor pixel_mean, pixel_std;   // [3, 1, 1, 1] f32
};

Model g_model;       // on g_dev while g_loaded
Model g_host_model;  // pinned host copy (same dtypes) after release(keep_host = 1)
bool g_loaded      = false;
bool g_has_host    = false;
bool g_load_failed = false;  // sticky: sd.cpp then decodes every tile on Vulkan instead of retrying the load

// every tensor of a Model, in a fixed order, for moving the whole model between devices
std::vector<at::Tensor*> model_tensors(Model& m) {
    std::vector<at::Tensor*> out = {&m.pq_w, &m.pq_b, &m.embed.w, &m.embed.b, &m.registers,
                                    &m.norm_out_w, &m.norm_out_b, &m.proj_out.w, &m.proj_out.b,
                                    &m.pixel_mean, &m.pixel_std};
    for (auto& b : m.blocks) {
        for (at::Tensor* t : {&b.norm1, &b.norm2, &b.scale1, &b.scale2, &b.qkv.w, &b.qkv.b, &b.out.w, &b.out.b, &b.w1.w,
                              &b.w1.b, &b.w2.w, &b.w2.b}) {
            out.push_back(t);
        }
    }
    return out;
}

// Accept plain ComfyUI names and the usual prefixes (first_stage_model., vae.).
std::string find_prefix(const SafeTensors& st) {
    for (const char* p : {"", "first_stage_model.", "vae.", "model.", "decoder_model."}) {
        if (st.index.count(std::string(p) + "decoder.transformer_blocks.0.attn.to_qkv.weight")) {
            return p;
        }
    }
    throw std::runtime_error("no H3 video VAE decoder weights in " + st.path);
}

void load_from_file(int verbose) {
    const char* path = std::getenv("SD_XPU_H3VAE_WEIGHTS");
    if (path == nullptr || *path == '\0') {
        throw std::runtime_error("SD_XPU_H3VAE_WEIGHTS is not set");
    }
    const auto t0 = std::chrono::steady_clock::now();
    SafeTensors st(path);
    const std::string p = find_prefix(st);
    auto dev            = [&](const std::string& name, at::ScalarType type) { return st.read(p + name).to(g_dev).to(type); };
    auto lin            = [&](const std::string& name, at::ScalarType type) {
        return Lin{dev(name + ".weight", type), dev(name + ".bias", type)};
    };
    Model m;
    m.pq_w      = dev("post_quant_conv.weight", at::kFloat).reshape({kLatentC, kLatentC});
    m.pq_b      = dev("post_quant_conv.bias", at::kFloat);
    m.embed     = lin("decoder.x_embedder", at::kFloat);
    m.registers = dev("decoder.register_tokens", at::kFloat).reshape({kRegisters, kDim});
    int n_blocks = 0;
    while (st.index.count(p + "decoder.transformer_blocks." + std::to_string(n_blocks) + ".attn.to_qkv.weight")) {
        ++n_blocks;
    }
    m.blocks.resize(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        const std::string b = "decoder.transformer_blocks." + std::to_string(i) + ".";
        Block& blk          = m.blocks[i];
        blk.norm1           = dev(b + "norm1.weight", at::kFloat);
        blk.norm2           = dev(b + "norm2.weight", at::kFloat);
        blk.scale1          = dev(b + "scale1", at::kFloat).reshape({kDim});
        blk.scale2          = dev(b + "scale2", at::kFloat).reshape({kDim});
        blk.qkv             = lin(b + "attn.to_qkv", g_cdt);
        blk.out             = lin(b + "attn.to_out", g_cdt);
        blk.w1              = lin(b + "ff.w1", g_cdt);
        blk.w2              = lin(b + "ff.w2", g_cdt);
        if (blk.qkv.w.size(0) != 3 * kDim || blk.w1.w.size(0) != 2 * kFfInner || blk.w2.w.size(1) != kFfInner) {
            throw std::runtime_error("unexpected H3 VAE decoder block shapes");
        }
    }
    m.norm_out_w = dev("decoder.norm_out.weight", at::kFloat);
    m.norm_out_b = dev("decoder.norm_out.bias", at::kFloat);
    m.proj_out   = lin("decoder.proj_out", at::kFloat);
    if (m.proj_out.w.size(0) != kOutC * kPatchT * kPatch * kPatch) {
        throw std::runtime_error("unexpected H3 VAE proj_out shape");
    }
    auto fopt    = at::TensorOptions().dtype(at::kFloat);
    m.pixel_mean = at::tensor({0.485f, 0.456f, 0.406f}, fopt).view({kOutC, 1, 1, 1}).to(g_dev);
    m.pixel_std  = at::tensor({0.229f, 0.224f, 0.225f}, fopt).view({kOutC, 1, 1, 1}).to(g_dev);
    g_model      = std::move(m);
    empty_cache();
    if (verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_h3vae] loaded %d decoder blocks (%s) from %s in %.1f s\n", n_blocks,
                     g_cdt == at::kBFloat16 ? "bf16" : g_cdt == at::kHalf ? "fp16" : "f32", path, s);
    }
}

void load_now(int verbose) {
    if (g_has_host) {
        const auto t0 = std::chrono::steady_clock::now();
        Model m       = g_host_model;  // copies the handles; each one is replaced by a device copy below
        auto src      = model_tensors(g_host_model);
        auto dst      = model_tensors(m);
        for (size_t i = 0; i < src.size(); ++i) {
            *dst[i] = src[i]->to(g_dev, /*non_blocking=*/true);
        }
        g_model = std::move(m);
        if (verbose) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[xpu_h3vae] uploaded decoder weights from host in %.2f s\n", s);
        }
    } else {
        load_from_file(verbose);
    }
}

void ensure_loaded(int verbose) {
    if (g_loaded) {
        return;
    }
    if (g_load_failed) {
        throw std::runtime_error("decoder weights failed to load earlier");
    }
    try {
        load_now(verbose);
    } catch (...) {
        g_load_failed = true;
        g_model       = Model();
        empty_cache();
        throw;
    }
    g_loaded = true;
}

at::Tensor rms_norm(const at::Tensor& x, double eps) {  // x f32, over the last dim
    return x * at::rsqrt(x.pow(2).mean(-1, true) + eps);
}

}  // namespace

// Load the decoder weights now (otherwise the first decode does it). 0 on success.
extern "C" int xpu_h3vae_init(int verbose) {
    try {
        ensure_loaded(verbose);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3vae] init error: %s\n", e.what());
        return 1;
    }
}

// Free the device copy of the weights. keep_host = 1 keeps (or makes) a pinned host copy for a fast reload.
extern "C" void xpu_h3vae_release(int keep_host) {
    try {
        if (g_loaded && keep_host && !g_has_host) {
            Model h  = g_model;
            auto src = model_tensors(g_model);
            auto dst = model_tensors(h);
            for (size_t i = 0; i < src.size(); ++i) {
                auto pinned = at::empty(src[i]->sizes(), src[i]->options().device(c10::Device(c10::DeviceType::CPU)).pinned_memory(g_on_xpu));
                pinned.copy_(*src[i]);
                *dst[i] = pinned;
            }
            g_host_model = std::move(h);
            g_has_host   = true;
        }
        if (!keep_host) {
            g_host_model  = Model();
            g_has_host    = false;
            g_load_failed = false;
        }
        if (g_loaded) {
            g_model  = Model();
            g_loaded = false;
            empty_cache();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3vae] release error: %s\n", e.what());
        g_model  = Model();
        g_loaded = false;
        empty_cache();
    }
}

// Decode one latent tile.
//   z_host   [C = 24][T][H][W] f32 (sd.cpp tensor [W, H, T, 24]): VAE-space latents (before post_quant_conv)
//   pe_host  [T*H*W + 5][half_rot][2][2] f32: the RoPE table sd.cpp builds for the tile (build_rope)
//   out_host [3][T*4][H*16][W*16] f32: pixels in [0, 1] (sd.cpp tensor [W*16, H*16, T*4, 3])
extern "C" int xpu_h3vae_decode(const float* z_host,
                                int64_t W,
                                int64_t H,
                                int64_t T,
                                int64_t C,
                                const float* pe_host,
                                int64_t pe_rows,
                                int64_t half_rot,
                                float* out_host,
                                int verbose) {
    try {
        using clock   = std::chrono::steady_clock;
        const auto t0 = clock::now();
        ensure_loaded(verbose);
        const Model& m  = g_model;
        const int64_t P = W * H * T;
        const int64_t N = P + kRegisters + 1;
        if (C != kLatentC || pe_rows != N || half_rot * 2 > kHeadDim) {
            throw std::runtime_error("bad tile arguments");
        }
        auto fopt = at::TensorOptions().dtype(at::kFloat);
        auto host = [&](const float* p, at::IntArrayRef shape) {
            auto t = at::from_blob(const_cast<float*>(p), shape, fopt);
            return g_on_xpu ? t.pin_memory().to(g_dev) : t.clone();
        };

        // post_quant_conv (1x1x1) and x_embedder per token
        auto z  = host(z_host, {C, P}).t();                       // [P, 24]
        auto zq = at::addmm(m.pq_b, z, m.pq_w.t());               // [P, 24]
        auto hp = at::addmm(m.embed.b, zq, m.embed.w.t());        // [P, 2048]
        auto x  = at::cat({hp, m.registers, at::zeros({1, kDim}, hp.options())}, 0).contiguous();  // [N, 2048] f32
        z.reset();
        zq.reset();
        hp.reset();

        // RoPE table: pe[n][i] = [[cos, -sin], [sin, cos]]
        auto pe   = host(pe_host, {N, half_rot, 2, 2});
        auto cosv = pe.select(3, 0).select(2, 0).unsqueeze(1).contiguous();  // [N, 1, half]
        auto sinv = pe.select(3, 0).select(2, 1).unsqueeze(1).contiguous();
        pe.reset();
        const int64_t rot = half_rot * 2;
        auto rope         = [&](const at::Tensor& t) {  // t [N, heads, 64] f32
            auto a = t.slice(2, 0, half_rot);
            auto b = t.slice(2, half_rot, rot);
            return at::cat({a * cosv - b * sinv, a * sinv + b * cosv, t.slice(2, rot, kHeadDim)}, 2);
        };

        for (const auto& blk : m.blocks) {
            {
                auto h   = (rms_norm(x, kEps) * blk.norm1).to(g_cdt);
                auto qkv = at::linear(h, blk.qkv.w, blk.qkv.b).view({N, kHeads, 3, kHeadDim});  // per head [q | k | v]
                h.reset();
                auto q = rope(rms_norm(qkv.select(2, 0).to(at::kFloat), kEps)).to(g_cdt);
                auto k = rope(rms_norm(qkv.select(2, 1).to(at::kFloat), kEps)).to(g_cdt);
                auto v = qkv.select(2, 2).contiguous();
                auto o = at::scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                                          v.transpose(0, 1).unsqueeze(0));  // [1, heads, N, 64]
                q.reset();
                k.reset();
                v.reset();
                qkv.reset();
                auto a = o.squeeze(0).transpose(0, 1).reshape({N, kDim});
                o.reset();
                x.addcmul_(at::linear(a, blk.out.w, blk.out.b).to(at::kFloat), blk.scale1);
            }
            {
                auto h  = (rms_norm(x, kEps) * blk.norm2).to(g_cdt);
                auto gu = at::linear(h, blk.w1.w, blk.w1.b);  // [N, gate | up]
                h.reset();
                auto act = at::silu(gu.slice(1, 0, kFfInner)) * gu.slice(1, kFfInner, 2 * kFfInner);
                gu.reset();
                x.addcmul_(at::linear(act, blk.w2.w, blk.w2.b).to(at::kFloat), blk.scale2);
            }
        }
        cosv.reset();
        sinv.reset();

        auto y = at::layer_norm(x.slice(0, 0, P), {kDim}, m.norm_out_w, m.norm_out_b, kEps);
        x.reset();
        y = at::addmm(m.proj_out.b, y, m.proj_out.w.t());  // [P, 3 * 4 * 16 * 16]
        // unpatchify: feature = ((c * 4 + pt) * 16 + ph) * 16 + pw, token = (t * H + h) * W + w
        auto pix = y.view({T, H, W, kOutC, kPatchT, kPatch, kPatch})
                       .permute({3, 0, 4, 1, 5, 2, 6})
                       .reshape({kOutC, T * kPatchT, H * kPatch, W * kPatch});
        y.reset();
        pix = (pix * m.pixel_std + m.pixel_mean).clamp_(0.f, 1.f).contiguous();
        auto out = pix.to(c10::Device(c10::DeviceType::CPU));
        std::memcpy(out_host, out.data_ptr<float>(), static_cast<size_t>(out.numel()) * sizeof(float));
        pix.reset();
        if (verbose) {
            const double s = std::chrono::duration<double>(clock::now() - t0).count();
            std::fprintf(stderr, "[xpu_h3vae] tile %lldx%lldx%lld latent (%lld tokens): %.3f s\n", (long long)W, (long long)H,
                         (long long)T, (long long)N, s);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3vae] decode error: %s\n", e.what());
        return 1;
    }
}
