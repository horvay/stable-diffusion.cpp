// MiniMax-H3 video VAE decoder on an Intel XPU, native SYCL + oneDNN (no libtorch).
//
// sd.cpp decodes H3 video in temporal windows of 7 latent frames, each split into 256 px tiles with 0.25
// overlap (MiniMaxH3VideoVAERunner::decode). With SD_XPU_H3VAE_LIB set, every tile's decode graph
// (post_quant_conv, x_embedder, 36 transformer blocks, norm_out, proj_out, unpatchify, pixel
// de-normalization) runs here instead of on Vulkan. The tiling, window plan and temporal blending stay in
// sd.cpp, so the numerics of the tile layout are unchanged.
//
// The decoder weights (~4.8 GB, SD_XPU_H3VAE_WEIGHTS = the fp16 VAE safetensors) load on the first tile and
// stay resident on the device across tiles and windows until xpu_h3vae_release(). release(keep_host = 1)
// moves them to pinned host memory first, so the next load is a fast upload instead of a file read
// (used by the multi-shot runner, which hands the GPU back to the transformer blocks between shots).
// Per-tile buffers come from the bounded pool (xpu_native.hpp) and are handed back to the driver after each tile.
//
// SD_XPU_H3VAE_DTYPE=bf16 (default) | fp16 | f32: dtype of the big linears and attention. Residual stream,
// norms, x_embedder, post_quant_conv and proj_out (f32 in sd.cpp too) stay f32.
// SD_XPU_H3VAE_SDPA=sycl forces the SYCL attention kernel (default: oneDNN Graph fused SDPA, SYCL fallback).
// SD_XPU_H3VAE_DEVICE=cpu (the old torch CPU reference mode) is not available in the native build.
// SD_XPU_H3VAE_SELFTEST=<dir>: xpu_h3vae_init() decodes the tile in <dir> and compares with <dir>/ref_out.f32.
//
// Block math mirrors src/model/vae/minimax_h3_vae.hpp (Decoder / DecoderBlock / DecoderAttention /
// DecoderFeedForward) and the previous libtorch implementation op by op:
//   x += attn(RMSNorm(x) * norm1) * scale1;  x += ff(RMSNorm(x) * norm2) * scale2
//   attn: to_qkv -> per head [q | k | v] (64 each); RMSNorm(q), RMSNorm(k) over head_dim without weight;
//         split-half RoPE on the first 2 * half_rot dims; full (unmasked) SDPA, scale 1/sqrt(64); to_out
//   ff:   w1 -> [gate | up]; w2(silu(gate) * up)
// Tokens: the tile's latent patches (t, h, w order, w fastest), 4 register tokens, 1 zero token.

#include "xpu_native.hpp"

namespace {

using xn::bf16;
using xn::Buf;
using xn::DType;
using xn::f16;
using xn::Q;

constexpr int64_t kDim       = 2048;
constexpr int64_t kHeads     = 32;
constexpr int64_t kHeadDim   = 64;
constexpr int64_t kFfInner   = 8192;
constexpr int64_t kLatentC   = 24;
constexpr int64_t kRegisters = 4;
constexpr int64_t kPatch     = 16;
constexpr int64_t kPatchT    = 4;
constexpr int64_t kOutC      = 3;
constexpr float kEps         = 1e-5f;

const DType g_cdt = [] {
    const char* v = std::getenv("SD_XPU_H3VAE_DTYPE");
    if (v == nullptr || std::strcmp(v, "bf16") == 0) {
        return DType::bf16;
    }
    if (std::strcmp(v, "fp16") == 0 || std::strcmp(v, "f16") == 0) {
        return DType::f16;
    }
    if (std::strcmp(v, "f32") == 0 || std::strcmp(v, "fp32") == 0) {
        return DType::f32;
    }
    std::fprintf(stderr, "[xpu_h3vae] unknown SD_XPU_H3VAE_DTYPE=%s, using bf16\n", v);
    return DType::bf16;
}();

const bool g_cpu_requested = [] {
    const char* v = std::getenv("SD_XPU_H3VAE_DEVICE");
    return v != nullptr && std::strcmp(v, "cpu") == 0;
}();

const bool g_graph_sdpa = [] {
    const char* v = std::getenv("SD_XPU_H3VAE_SDPA");
    return !(v != nullptr && std::strcmp(v, "sycl") == 0);
}();

const int64_t g_max_blocks = xn::env_i64("SD_XPU_MAX_BLOCKS", -1);

struct Lin {
    Buf w, b;  // w [out, in], b [out]
    int64_t out = 0, in = 0;
};

struct Block {
    Buf norm1, norm2, scale1, scale2;  // f32 [kDim]
    Lin qkv, out, w1, w2;              // compute dtype
};

struct Model {
    Buf pq_w, pq_b;  // post_quant_conv as a [24, 24] linear, f32
    Lin embed;       // x_embedder [2048, 24], f32
    Buf registers;   // [4, 2048] f32
    std::vector<Block> blocks;
    Buf norm_out_w, norm_out_b;  // f32
    Lin proj_out;                // [3072, 2048], f32 (force_f32 in sd.cpp)
};

std::vector<Buf*> model_bufs(Model& m) {
    std::vector<Buf*> out = {&m.pq_w, &m.pq_b, &m.embed.w, &m.embed.b, &m.registers, &m.norm_out_w, &m.norm_out_b, &m.proj_out.w, &m.proj_out.b};
    for (auto& b : m.blocks) {
        for (Buf* t : {&b.norm1, &b.norm2, &b.scale1, &b.scale2, &b.qkv.w, &b.qkv.b, &b.out.w, &b.out.b, &b.w1.w, &b.w1.b, &b.w2.w, &b.w2.b}) {
            out.push_back(t);
        }
    }
    return out;
}

struct HostBlob {
    void* p  = nullptr;
    size_t n = 0;
    bool pinned = false;
};

struct State {
    Model model;                  // on the device while loaded
    std::vector<HostBlob> host;   // host copy of every model buffer (model_bufs order) after release(keep_host = 1)
    Model shape;                  // dims of the host copy (buffers empty)
    bool loaded      = false;
    bool has_host    = false;
    bool load_failed = false;  // sticky: sd.cpp then decodes every tile on Vulkan instead of retrying the load
};

State& S() {
    static auto* s = new State();  // leaked: no device frees after the SYCL runtime is gone at process exit
    return *s;
}

void free_host_copy() {
    for (auto& h : S().host) {
        if (h.p != nullptr) {
            if (h.pinned) {
                sycl::free(h.p, Q());
            } else {
                std::free(h.p);
            }
        }
    }
    S().host.clear();
    S().has_host = false;
}

std::string find_prefix(const xn::SafeTensors& st) {
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
    xn::SafeTensors st(path);
    const std::string p = find_prefix(st);
    auto dev            = [&](const std::string& name, DType type) { return xn::load_tensor(st, p + name, type); };
    auto lin            = [&](const std::string& name, DType type) {
        Lin l;
        std::vector<int64_t> shape;
        l.w = xn::load_tensor(st, p + name + ".weight", type, true, &shape);
        l.b = xn::load_tensor(st, p + name + ".bias", type);
        if (shape.size() != 2) {
            throw std::runtime_error(name + ": expected a 2-D weight");
        }
        l.out = shape[0];
        l.in  = shape[1];
        return l;
    };
    Model m;
    m.pq_w = dev("post_quant_conv.weight", DType::f32);
    if (m.pq_w.bytes() != static_cast<size_t>(kLatentC * kLatentC * 4)) {
        throw std::runtime_error("unexpected H3 VAE post_quant_conv shape");
    }
    m.pq_b      = dev("post_quant_conv.bias", DType::f32);
    m.embed     = lin("decoder.x_embedder", DType::f32);
    m.registers = dev("decoder.register_tokens", DType::f32);
    if (m.embed.out != kDim || m.embed.in != kLatentC || m.registers.bytes() != static_cast<size_t>(kRegisters * kDim * 4)) {
        throw std::runtime_error("unexpected H3 VAE x_embedder / register shapes");
    }
    int n_blocks = 0;
    while (st.index.count(p + "decoder.transformer_blocks." + std::to_string(n_blocks) + ".attn.to_qkv.weight")) {
        ++n_blocks;
    }
    m.blocks.resize(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        const std::string b = "decoder.transformer_blocks." + std::to_string(i) + ".";
        Block& blk          = m.blocks[i];
        blk.norm1           = dev(b + "norm1.weight", DType::f32);
        blk.norm2           = dev(b + "norm2.weight", DType::f32);
        blk.scale1          = dev(b + "scale1", DType::f32);
        blk.scale2          = dev(b + "scale2", DType::f32);
        blk.qkv             = lin(b + "attn.to_qkv", g_cdt);
        blk.out             = lin(b + "attn.to_out", g_cdt);
        blk.w1              = lin(b + "ff.w1", g_cdt);
        blk.w2              = lin(b + "ff.w2", g_cdt);
        if (blk.qkv.out != 3 * kDim || blk.qkv.in != kDim || blk.out.out != kDim || blk.w1.out != 2 * kFfInner || blk.w2.in != kFfInner ||
            blk.scale1.bytes() != static_cast<size_t>(kDim * 4)) {
            throw std::runtime_error("unexpected H3 VAE decoder block shapes");
        }
    }
    m.norm_out_w = dev("decoder.norm_out.weight", DType::f32);
    m.norm_out_b = dev("decoder.norm_out.bias", DType::f32);
    m.proj_out   = lin("decoder.proj_out", DType::f32);
    if (m.proj_out.out != kOutC * kPatchT * kPatch * kPatch || m.proj_out.in != kDim) {
        throw std::runtime_error("unexpected H3 VAE proj_out shape");
    }
    S().model = std::move(m);
    Q().wait();
    xn::pool().trim();
    if (verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_h3vae] loaded %d decoder blocks (%s) from %s in %.1f s\n", n_blocks, xn::dname(g_cdt), path, s);
    }
}

// rebuild the device model from the host copy (same dims as S().shape)
void load_from_host(int verbose) {
    const auto t0 = std::chrono::steady_clock::now();
    Model m;
    m.blocks.resize(S().shape.blocks.size());
    m.embed.out = S().shape.embed.out, m.embed.in = S().shape.embed.in;
    m.proj_out.out = S().shape.proj_out.out, m.proj_out.in = S().shape.proj_out.in;
    for (size_t i = 0; i < m.blocks.size(); ++i) {
        const Block& sb = S().shape.blocks[i];
        Block& b        = m.blocks[i];
        b.qkv.out = sb.qkv.out, b.qkv.in = sb.qkv.in, b.out.out = sb.out.out, b.out.in = sb.out.in;
        b.w1.out = sb.w1.out, b.w1.in = sb.w1.in, b.w2.out = sb.w2.out, b.w2.in = sb.w2.in;
    }
    auto dst = model_bufs(m);
    if (dst.size() != S().host.size()) {
        throw std::runtime_error("host copy does not match the model layout");
    }
    for (size_t i = 0; i < dst.size(); ++i) {
        *dst[i] = xn::resident(S().host[i].n);
        if (S().host[i].pinned) {
            Q().memcpy(dst[i]->get(), S().host[i].p, S().host[i].n);
        } else {
            xn::h2d(dst[i]->get(), S().host[i].p, S().host[i].n);
        }
    }
    Q().wait();
    S().model = std::move(m);
    if (verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_h3vae] uploaded decoder weights from host in %.2f s\n", s);
    }
}

void ensure_loaded(int verbose) {
    if (S().loaded) {
        return;
    }
    if (g_cpu_requested) {
        throw std::runtime_error("SD_XPU_H3VAE_DEVICE=cpu is not supported by the native (non-torch) build");
    }
    if (S().load_failed) {
        throw std::runtime_error("decoder weights failed to load earlier");
    }
    try {
        if (S().has_host) {
            load_from_host(verbose);
        } else {
            load_from_file(verbose);
        }
    } catch (...) {
        S().load_failed = true;
        S().model       = Model();
        xn::release_all();
        throw;
    }
    S().loaded = true;
}

// ---- kernels ----

// out [n, H] (T) = (x * rsqrt(mean(x^2) + eps)) * w   (x f32 rows; w f32, or none)
template <class T>
void rms_norm(const float* x, const float* w, T* out, int64_t n, int64_t H, float eps) {
    constexpr int WG = 256;
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(n * WG), WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int lid   = static_cast<int>(it.get_local_id(0));
        const float* xr = x + r * H;
        float ss        = 0.f;
        for (int64_t c = lid; c < H; c += WG) {
            ss += xr[c] * xr[c];
        }
        const float rstd = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / static_cast<float>(H) + eps);
        for (int64_t c = lid; c < H; c += WG) {
            const float a = xr[c] * rstd;
            out[r * H + c] = static_cast<T>(w != nullptr ? a * w[c] : a);
        }
    });
}

// out [n, H] f32 = LayerNorm(x) * w + b (biased variance, f32)
void layer_norm(const float* x, const float* w, const float* b, float* out, int64_t n, int64_t H, float eps) {
    constexpr int WG = 256;
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(n * WG), WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int lid   = static_cast<int>(it.get_local_id(0));
        const float* xr = x + r * H;
        float s         = 0.f;
        for (int64_t c = lid; c < H; c += WG) {
            s += xr[c];
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / static_cast<float>(H);
        float v          = 0.f;
        for (int64_t c = lid; c < H; c += WG) {
            const float d = xr[c] - mean;
            v += d * d;
        }
        const float rstd = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) / static_cast<float>(H) + eps);
        for (int64_t c = lid; c < H; c += WG) {
            const float t = (xr[c] - mean) * rstd;
            const float u = t * w[c];
            out[r * H + c] = u + b[c];
        }
    });
}

// qkv [N, heads, 3, 64] (T) -> q, k [N, heads, 64] (T): RMSNorm over 64 (no weight, f32) -> split-half RoPE (f32 cos /
// sin, a * c - b * s, a * s + b * c) on the first 2 * half dims. One work-group of 64 per (token, q|k, head).
template <class T>
void qk_norm_rope(const T* qkv, const float* cosv, const float* sinv, int64_t half, int64_t N, T* q_out, T* k_out) {
    const int64_t rot = 2 * half;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(kHeadDim), h);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(N * 2 * kHeads * kHeadDim), kHeadDim), [=](sycl::nd_item<1> it) {
            const int64_t g    = it.get_group(0);
            const int64_t n    = g / (2 * kHeads);
            const int which    = static_cast<int>((g / kHeads) % 2);
            const int64_t head = g % kHeads;
            const int d        = static_cast<int>(it.get_local_id(0));
            const float xv     = static_cast<float>(qkv[n * 3 * kDim + head * 3 * kHeadDim + which * kHeadDim + d]);
            const float ss     = sycl::reduce_over_group(it.get_group(), xv * xv, sycl::plus<float>());
            const float rr     = sycl::rsqrt(ss / static_cast<float>(kHeadDim) + kEps);
            const float nv     = xv * rr;
            sh[d]              = nv;
            sycl::group_barrier(it.get_group());
            float o = nv;
            if (d < half) {
                const float c = cosv[n * half + d], s = sinv[n * half + d];
                const float t1 = sh[d] * c;
                const float t2 = sh[d + half] * s;
                o              = t1 - t2;
            } else if (d < rot) {
                const int64_t j = d - half;
                const float c = cosv[n * half + j], s = sinv[n * half + j];
                const float t1 = sh[j] * s;
                const float t2 = sh[d] * c;
                o              = t1 + t2;
            }
            (which == 0 ? q_out : k_out)[n * kDim + head * kHeadDim + d] = static_cast<T>(o);
        });
    });
}

// x [n, H] f32 += float(y) * s   (addcmul_)
template <class T>
void addcmul(float* x, const T* y, const float* s, int64_t n, int64_t H) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n * H)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        const float t   = static_cast<float>(y[i]) * s[i % H];
        x[i]            = x[i] + t;
    });
}

// y [P, 3 * 4 * 16 * 16] f32 -> pixels [3][T * 4][H * 16][W * 16] in [0, 1]:
// feature = ((c * 4 + pt) * 16 + ph) * 16 + pw, token = (t * H + h) * W + w; pix * std + mean, clamped.
void unpatchify(const float* y, int64_t T_, int64_t H_, int64_t W_, float* out) {
    const int64_t OT = T_ * kPatchT, OH = H_ * kPatch, OW = W_ * kPatch;
    const int64_t feat = kOutC * kPatchT * kPatch * kPatch;
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(kOutC * OT * OH * OW)), [=](sycl::id<1> id) {
        const int64_t i  = id[0];
        const int64_t ww = i % OW, hh = (i / OW) % OH, tt = (i / (OW * OH)) % OT, c = i / (OW * OH * OT);
        const int64_t t = tt / kPatchT, pt = tt % kPatchT, h = hh / kPatch, ph = hh % kPatch, w = ww / kPatch, pw = ww % kPatch;
        const int64_t token = (t * H_ + h) * W_ + w;
        const int64_t f     = ((c * kPatchT + pt) * kPatch + ph) * kPatch + pw;
        const float stdv    = c == 0 ? 0.229f : c == 1 ? 0.224f : 0.225f;
        const float meanv   = c == 0 ? 0.485f : c == 1 ? 0.456f : 0.406f;
        const float a       = y[token * feat + f] * stdv;
        const float v       = a + meanv;
        out[i]              = sycl::fmin(sycl::fmax(v, 0.f), 1.f);
    });
}

// y [n, out] = x [n, in] W^T + b (dtype dt everywhere, f32 accumulation)
void linear(DType dt, const void* x, int64_t n, const Lin& l, void* y) {
    xn::MatmulArgs a;
    a.M = n, a.K = l.in, a.N = l.out;
    a.xt = dt, a.x = x, a.wt = dt, a.w = l.w.get(), a.yt = dt, a.y = y, a.bt = dt, a.bias = l.b.get();
    xn::matmul(a);
}

template <class T>
void decode_blocks(const Model& m, float* x, int64_t N, const float* cosv, const float* sinv, int64_t half, int verbose) {
    const DType dt         = g_cdt;
    const size_t n_run     = g_max_blocks >= 0 ? std::min<size_t>(m.blocks.size(), static_cast<size_t>(g_max_blocks)) : m.blocks.size();
    const float attn_scale = 1.f / std::sqrt(static_cast<float>(kHeadDim));
    for (size_t bi = 0; bi < n_run; ++bi) {
        const Block& blk = m.blocks[bi];
        {
            Buf h = xn::scratch_n<T>(N * kDim);
            rms_norm(x, blk.norm1.as<float>(), h.as<T>(), N, kDim, kEps);
            Buf qkv = xn::scratch_n<T>(N * 3 * kDim);  // per head [q | k | v]
            linear(dt, h.get(), N, blk.qkv, qkv.get());
            h.reset();
            Buf q = xn::scratch_n<T>(N * kDim), k = xn::scratch_n<T>(N * kDim);
            qk_norm_rope(qkv.as<T>(), cosv, sinv, half, N, q.as<T>(), k.as<T>());
            Buf o = xn::scratch_n<T>(N * kDim);  // [heads, N, 64]
            xn::attention(dt, q.get(), kHeadDim, kDim, k.get(), kHeadDim, kDim, qkv.as<T>() + 2 * kHeadDim, 3 * kHeadDim, 3 * kDim, kHeads, N, N,
                          kHeadDim, attn_scale, -1, o.get(), g_graph_sdpa, verbose, "xpu_h3vae");
            q.reset();
            k.reset();
            qkv.reset();
            Buf a = xn::scratch_n<T>(N * kDim);
            xn::heads_to_tokens(o.as<T>(), kHeads, N, kHeadDim, 0, N, a.as<T>());
            o.reset();
            Buf y = xn::scratch_n<T>(N * kDim);
            linear(dt, a.get(), N, blk.out, y.get());
            addcmul(x, y.as<T>(), blk.scale1.as<float>(), N, kDim);
        }
        {
            Buf h = xn::scratch_n<T>(N * kDim);
            rms_norm(x, blk.norm2.as<float>(), h.as<T>(), N, kDim, kEps);
            Buf gu = xn::scratch_n<T>(N * 2 * kFfInner);  // [N, gate | up]
            linear(dt, h.get(), N, blk.w1, gu.get());
            h.reset();
            Buf act = xn::scratch_n<T>(N * kFfInner);
            xn::silu_mul(gu.as<T>(), act.as<T>(), N, kFfInner);
            gu.reset();
            Buf y = xn::scratch_n<T>(N * kDim);
            linear(dt, act.get(), N, blk.w2, y.get());
            addcmul(x, y.as<T>(), blk.scale2.as<float>(), N, kDim);
        }
    }
}

void decode(const float* z_host, int64_t W, int64_t H, int64_t T_, int64_t C, const float* pe_host, int64_t pe_rows, int64_t half_rot,
            float* out_host, int verbose) {
    using clock   = std::chrono::steady_clock;
    const auto t0 = clock::now();
    ensure_loaded(verbose);
    const Model& m  = S().model;
    const int64_t P = W * H * T_;
    const int64_t N = P + kRegisters + 1;
    if (C != kLatentC || pe_rows != N || half_rot * 2 > kHeadDim) {
        throw std::runtime_error("bad tile arguments");
    }
    const int64_t n_out = kOutC * T_ * kPatchT * H * kPatch * W * kPatch;
    {
        // post_quant_conv (1x1x1) and x_embedder per token
        Buf z = xn::scratch_n<float>(C * P);  // [24, P]
        xn::h2d(z.get(), z_host, static_cast<size_t>(C * P) * 4);
        Buf zq = xn::scratch_n<float>(P * kLatentC);
        {
            xn::MatmulArgs a;  // zq [P, 24] = z^T pq_w^T + pq_b
            a.M = P, a.K = kLatentC, a.N = kLatentC;
            a.xt = DType::f32, a.x = z.get(), a.x_rs = 1, a.x_cs = P;
            a.wt = DType::f32, a.w = m.pq_w.get(), a.yt = DType::f32, a.y = zq.get(), a.bt = DType::f32, a.bias = m.pq_b.get();
            xn::matmul(a);
        }
        Buf x = xn::scratch_n<float>(N * kDim);  // [N, 2048] f32: patches, registers, zero token
        linear(DType::f32, zq.get(), P, m.embed, x.get());
        Q().memcpy(x.as<float>() + P * kDim, m.registers.get(), static_cast<size_t>(kRegisters * kDim) * 4);
        Q().memset(x.as<float>() + (P + kRegisters) * kDim, 0, static_cast<size_t>(kDim) * 4);
        z.reset();
        zq.reset();

        // RoPE table: pe[n][i] = [[cos, -sin], [sin, cos]]
        Buf cosv = xn::scratch_n<float>(N * half_rot), sinv = xn::scratch_n<float>(N * half_rot);
        {
            std::vector<float> c(static_cast<size_t>(N * half_rot)), s(c.size());
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t i = 0; i < half_rot; ++i) {
                    c[n * half_rot + i] = pe_host[(n * half_rot + i) * 4 + 0];
                    s[n * half_rot + i] = pe_host[(n * half_rot + i) * 4 + 2];
                }
            }
            xn::h2d(cosv.get(), c.data(), c.size() * 4);
            xn::h2d(sinv.get(), s.data(), s.size() * 4);
        }
        switch (g_cdt) {
            case DType::bf16: decode_blocks<bf16>(m, x.as<float>(), N, cosv.as<float>(), sinv.as<float>(), half_rot, verbose); break;
            case DType::f16: decode_blocks<f16>(m, x.as<float>(), N, cosv.as<float>(), sinv.as<float>(), half_rot, verbose); break;
            default: decode_blocks<float>(m, x.as<float>(), N, cosv.as<float>(), sinv.as<float>(), half_rot, verbose); break;
        }
        cosv.reset();
        sinv.reset();

        Buf y = xn::scratch_n<float>(P * kDim);
        layer_norm(x.as<float>(), m.norm_out_w.as<float>(), m.norm_out_b.as<float>(), y.as<float>(), P, kDim, kEps);
        x.reset();
        Buf yp = xn::scratch_n<float>(P * m.proj_out.out);  // [P, 3 * 4 * 16 * 16]
        linear(DType::f32, y.get(), P, m.proj_out, yp.get());
        y.reset();
        Buf pix = xn::scratch_n<float>(n_out);
        unpatchify(yp.as<float>(), T_, H, W, pix.as<float>());
        yp.reset();
        xn::d2h(out_host, pix.get(), static_cast<size_t>(n_out) * 4);
        if (xn::pool().verbose()) {
            xn::pool().report("xpu_h3vae");
        }
    }
    xn::pool().trim();
    xn::pool().reset_peak();
    if (verbose) {
        const double s = std::chrono::duration<double>(clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_h3vae] tile %lldx%lldx%lld latent (%lld tokens): %.3f s\n", (long long)W, (long long)H, (long long)T_, (long long)N,
                     s);
    }
}

int selftest(const std::string& dir, int verbose) {
    auto meta          = xn::read_meta(dir);
    const int64_t W    = static_cast<int64_t>(meta.at("W"));
    const int64_t H    = static_cast<int64_t>(meta.at("H"));
    const int64_t T_   = static_cast<int64_t>(meta.at("T"));
    const int64_t C    = static_cast<int64_t>(meta.at("C"));
    const int64_t half = static_cast<int64_t>(meta.at("half_rot"));
    const int64_t rows = static_cast<int64_t>(meta.at("pe_rows"));
    auto z             = xn::read_vec<float>(dir + "/z.f32");
    auto pe            = xn::read_vec<float>(dir + "/pe.f32");
    auto ref           = xn::read_vec<float>(dir + "/ref_out.f32");
    if (static_cast<int64_t>(z.size()) != C * T_ * H * W || static_cast<int64_t>(pe.size()) != rows * half * 4) {
        throw std::runtime_error("selftest: inconsistent input files in " + dir);
    }
    std::vector<float> out(static_cast<size_t>(kOutC * T_ * kPatchT * H * kPatch * W * kPatch));
    const auto t0 = std::chrono::steady_clock::now();
    decode(z.data(), W, H, T_, C, pe.data(), rows, half, out.data(), verbose);
    const int64_t nb = static_cast<int64_t>(S().model.blocks.size());
    std::fprintf(stderr, "[xpu_h3vae] selftest: tile %lldx%lldx%lld through %lld block(s), %s, %.2f s\n", (long long)W, (long long)H, (long long)T_,
                 (long long)(g_max_blocks >= 0 ? std::min<int64_t>(g_max_blocks, nb) : nb), xn::dname(g_cdt),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    xn::write_file(dir + "/native_out.f32", out.data(), out.size() * 4);
    return xn::compare_report("xpu_h3vae", out, ref, xn::env_f64("SD_XPU_SELFTEST_TOL", 0.02));
}

}  // namespace

// Load the decoder weights now (otherwise the first decode does it). 0 on success.
XN_EXPORT int xpu_h3vae_init(int verbose) {
    try {
        ensure_loaded(verbose);
        if (const char* dir = std::getenv("SD_XPU_H3VAE_SELFTEST"); dir != nullptr && *dir != '\0') {
            return selftest(dir, verbose);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3vae] init error: %s\n", e.what());
        return 1;
    }
}

// Free the device copy of the weights. keep_host = 1 keeps (or makes) a pinned host copy for a fast reload.
XN_EXPORT void xpu_h3vae_release(int keep_host) {
    try {
        if (S().loaded && keep_host && !S().has_host) {
            auto src = model_bufs(S().model);
            std::vector<HostBlob> host;
            for (Buf* b : src) {
                HostBlob h;
                h.n = b->bytes();
                h.p = sycl::malloc_host(h.n, Q());
                h.pinned = h.p != nullptr;
                if (!h.pinned) {
                    h.p = std::malloc(h.n);
                    if (h.p == nullptr) {
                        throw std::runtime_error("cannot allocate the host copy of the decoder weights");
                    }
                }
                if (h.pinned) {
                    Q().memcpy(h.p, b->get(), h.n);
                } else {
                    xn::d2h(h.p, b->get(), h.n);
                }
                host.push_back(h);
            }
            Q().wait();
            S().host = std::move(host);
            // remember the dims for load_from_host
            Model shape;
            shape.blocks.resize(S().model.blocks.size());
            shape.embed.out = S().model.embed.out, shape.embed.in = S().model.embed.in;
            shape.proj_out.out = S().model.proj_out.out, shape.proj_out.in = S().model.proj_out.in;
            for (size_t i = 0; i < shape.blocks.size(); ++i) {
                const Block& b = S().model.blocks[i];
                Block& d       = shape.blocks[i];
                d.qkv.out = b.qkv.out, d.qkv.in = b.qkv.in, d.out.out = b.out.out, d.out.in = b.out.in;
                d.w1.out = b.w1.out, d.w1.in = b.w1.in, d.w2.out = b.w2.out, d.w2.in = b.w2.in;
            }
            S().shape    = std::move(shape);
            S().has_host = true;
        }
        if (!keep_host) {
            free_host_copy();
            S().load_failed = false;
        }
        if (S().loaded) {
            S().model  = Model();
            S().loaded = false;
        }
        xn::release_all();
        if (xn::g_rt_started && xn::pool().verbose()) {
            xn::pool().report("xpu_h3vae release");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3vae] release error: %s\n", e.what());
        try {
            S().model  = Model();
            S().loaded = false;
            xn::release_all();
        } catch (...) {
        }
    }
}

// Decode one latent tile.
//   z_host   [C = 24][T][H][W] f32 (sd.cpp tensor [W, H, T, 24]): VAE-space latents (before post_quant_conv)
//   pe_host  [T*H*W + 5][half_rot][2][2] f32: the RoPE table sd.cpp builds for the tile (build_rope)
//   out_host [3][T*4][H*16][W*16] f32: pixels in [0, 1] (sd.cpp tensor [W*16, H*16, T*4, 3])
XN_EXPORT int xpu_h3vae_decode(const float* z_host,
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
        decode(z_host, W, H, T, C, pe_host, pe_rows, half_rot, out_host, verbose);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3vae] decode error: %s\n", e.what());
        xn::trim_after_error(e);
        return 1;
    }
}
