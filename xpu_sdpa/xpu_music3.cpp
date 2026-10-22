// MiniMax Music 3 text-to-music on an Intel XPU (Arc Pro B70), native SYCL + oneDNN: no torch, no ggml.
//
// Same three stages as ComfyUI's MiniMax Music 3 (comfy/ldm/minimax_music), same checkpoints:
//   1. AR    Qwen3-8B-style LM (int8 convrot) writes one audio frame per step (25 frames/s): c0 from the pruned LM
//            head, codebooks 1..7 from a 4-layer depth decoder, all with CFG against a prompt-blanked batch row.
//            Each frame's hidden states (LM + 7 depth outputs, 8 x 4096) condition the DiT.
//   2. DiT   36-block flow transformer (int8 convrot) over the latent [128, T] (86.13 latents/s), euler/simple
//            sampling with CFG against a zero condition; windows of 689 latents with a 344 hop past 8 s.
//   3. DAV   conv decoder (weight-normed convs, snake activations), 512 samples per latent, stereo 44.1 kHz,
//            then ComfyUI's VAEDecodeAudio loudness guard (divide by max(1, 5 std)).
// Int8 linears are W8A8 like the other XPU libraries here: the bf16 input is rotated by the 256-group Hadamard,
// quantized per row, and run through oneDNN's int8 matmul with the row and weight scales fused. Attention is
// oneDNN Graph SDPA (DiT) or SYCL kernels (causal prefill, split-K decode, the depth decoder).
//
// Usage: xpu_music3 --caption-file F [--lyrics-file F] --out song.wav [--seconds 120] [--seed N] [--steps 30]
//                   [--cfg 1.7] [--ar-cfg 1.5] [--top-k 50]
//        xpu_music3 --selftest {dit,cond,dav,ar} DIR   (compare with ref_music3.py dumps)

#include "xpu_native.hpp"

#include "core/rng_mt19937.hpp"
#include "tokenizers/hf_tokenizer.h"
#include "xpu_music3_kernels.hpp"

#include <filesystem>
#include <random>

namespace {

using xn::bf16;
using xn::Buf;
using xn::DType;
using xn::Q;

// The weights live in the model store (~/models/music/minimax-music3); --te / --dit / --dav override each file.
const std::string kModels = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/models/music/minimax-music3/";
int g_verbose            = 1;

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#define LOG(...) (std::fprintf(stderr, "[xpu_music3] " __VA_ARGS__), std::fputc('\n', stderr))

// ---- weights ----

// int8 convrot linear: q [out, in] rotated by the 256-group Hadamard, f32 row scales, optional f32 bias.
struct Lin {
    Buf q, s, bias;
    int64_t out = 0, in = 0, group = 0;
};

Lin load_lin(xn::SafeTensors& st, const std::string& prefix, bool bias = false) {
    Lin l;
    const auto& wi = st.info(prefix + ".weight");
    l.out          = wi.shape.at(0);
    l.in           = wi.shape.at(1);
    const std::string cfg = st.str(prefix + ".comfy_quant");
    if (cfg.find("int8_tensorwise") == std::string::npos || wi.dtype != "I8") {
        throw std::runtime_error(prefix + ": expected an int8_tensorwise weight");
    }
    l.group = cfg.find("\"convrot\": true") != std::string::npos ? 256 : 0;
    l.q     = xn::load_tensor(st, prefix + ".weight", DType::s8, true);
    l.s     = xn::load_tensor(st, prefix + ".weight_scale", DType::f32, true);
    if (static_cast<int64_t>(l.s.bytes() / 4) != l.out) {
        throw std::runtime_error(prefix + ".weight_scale: expected one scale per row");
    }
    if (bias) {
        l.bias = xn::load_tensor(st, prefix + ".bias", DType::f32, true);
    }
    return l;
}

// Dense weight [out, in] (bf16 or f32 on the device), optional f32 bias.
struct Dense {
    Buf w, bias;
    DType wt    = DType::bf16;
    int64_t out = 0, in = 0;
};

Dense load_dense(xn::SafeTensors& st, const std::string& name, DType wt, const std::string& bias_name = "") {
    Dense d;
    const auto& wi = st.info(name);
    d.out          = wi.shape.at(0);
    d.in           = wi.numel() / d.out;  // 1x1 / k-tap conv weights flatten to [out, in * k]
    d.wt           = wt;
    d.w            = xn::load_tensor(st, name, wt, true);
    if (!bias_name.empty()) {
        d.bias = xn::load_tensor(st, bias_name, DType::f32, true);
    }
    return d;
}

Buf load_f32(xn::SafeTensors& st, const std::string& name) { return xn::load_tensor(st, name, DType::f32, true); }

// ---- linears ----

// y [M, out] (yt) = x [M, in] (bf16) W^T (+ bias): rotate by R, quantize rows to int8, oneDNN int8 matmul.
void lin(const bf16* x, int64_t M, const Lin& l, void* y, DType yt = DType::bf16, bool accumulate = false) {
    Buf xr;
    const bf16* src = x;
    if (l.group > 0) {
        xr = xn::scratch_n<bf16>(M * l.in);
        xn::rotate_groups(x, xr.as<bf16>(), M, l.in, l.group);
        src = xr.as<bf16>();
    }
    Buf xq = xn::scratch(static_cast<size_t>(M * l.in));
    Buf sx = xn::scratch_n<float>(M);
    xn::quantize_rows(src, M, l.in, xq.as<int8_t>(), sx.as<float>());
    xn::MatmulArgs a;
    a.M = M, a.K = l.in, a.N = l.out;
    a.xt = DType::s8, a.x = xq.get(), a.wt = DType::s8, a.w = l.q.get(), a.yt = yt, a.y = y;
    a.x_scales = sx.as<float>(), a.w_scales = l.s.as<float>();
    a.bias       = l.bias.empty() ? nullptr : l.bias.get();
    a.accumulate = accumulate;
    xn::matmul(a);
}

// y [M, out] = x [M, in] W^T (+ bias); x dtype = weight dtype. y_ld < 0: dense.
void dense(const void* x, int64_t M, const Dense& d, void* y, DType yt, bool accumulate = false, int64_t y_ld = -1, int64_t x_rs = -1) {
    xn::MatmulArgs a;
    a.M = M, a.K = d.in, a.N = d.out;
    a.xt = d.wt, a.x = x, a.x_rs = x_rs, a.wt = d.wt, a.w = d.w.get(), a.yt = yt, a.y = y, a.y_ld = y_ld;
    a.bias       = d.bias.empty() ? nullptr : d.bias.get();
    a.accumulate = accumulate;
    xn::matmul(a);
}

// ============================================================================================================
// AR: Qwen3-8B-style LM + depth decoder
// ============================================================================================================

constexpr int kHid = 4096, kNQ = 32, kNKV = 8, kHD = 128, kFF = 12288;
constexpr int kDecHeads = 16, kDecHD = 256, kDecFF = 6144, kCodebooks = 8, kAudioVocab = 1024, kC0Vocab = 16384;
constexpr int kAudioCfgId = 151654, kAudioOffset = 151675;
constexpr float kEps = 1e-6f, kTheta = 1000000.f;

struct LmLayer {
    Buf in_norm, post_norm, q_norm, k_norm;
    Lin qkv, o, gate_up, down;
};

struct DecLayer {
    Buf in_norm, post_norm;
    Lin qkv, o, gate_up, down;
};

struct AR {
    std::unique_ptr<xn::SafeTensors> st;  // kept open for the prefill embedding rows (read on the host)
    std::vector<LmLayer> layers;
    Buf norm;
    Dense lm_head;                 // [16385, 4096] bf16: 0 = stop, 1 + c = audio code c
    Buf emb_audio, emb_extra;      // bf16 [16384, 4096], [7 * 1024, 4096]
    Dense proj;                    // depth decoder input projection, bf16 [4096, 4096]
    Buf pos_emb;                   // f32 [16, 4096]
    std::vector<DecLayer> dec;
    Buf dec_norm;
    Dense heads;                   // the 7 audio heads stacked: bf16 [7 * 1024, 4096]
    std::string tokenizer_json;
};

AR load_ar(const std::string& path) {
    const double t0 = now_s();
    AR ar;
    ar.st   = std::make_unique<xn::SafeTensors>(path);
    auto& st = *ar.st;
    if (!st.has("model.embed_tokens_prefill.weight") || !st.has("model.lm_head_pruned.weight")) {
        throw std::runtime_error("expected the pruned MiniMax Music 3 text encoder (embed_tokens_prefill, lm_head_pruned)");
    }
    for (int i = 0;; ++i) {
        const std::string m = "model.layers." + std::to_string(i) + ".";
        if (!st.has(m + "self_attn.qkv_proj.weight")) {
            break;
        }
        LmLayer L;
        L.in_norm   = load_f32(st, m + "input_layernorm.weight");
        L.post_norm = load_f32(st, m + "post_attention_layernorm.weight");
        L.q_norm    = load_f32(st, m + "self_attn.q_norm.weight");
        L.k_norm    = load_f32(st, m + "self_attn.k_norm.weight");
        L.qkv       = load_lin(st, m + "self_attn.qkv_proj");
        L.o         = load_lin(st, m + "self_attn.o_proj");
        L.gate_up   = load_lin(st, m + "mlp.gate_up_proj");
        L.down      = load_lin(st, m + "mlp.down_proj");
        ar.layers.push_back(std::move(L));
    }
    ar.norm      = load_f32(st, "model.norm.weight");
    ar.lm_head   = load_dense(st, "model.lm_head_pruned.weight", DType::bf16);
    ar.emb_audio = xn::load_tensor(st, "model.embed_tokens_audio.weight", DType::bf16, true);
    ar.emb_extra = xn::load_tensor(st, "model.audio_extra_embedding.weight", DType::bf16, true);
    ar.proj      = load_dense(st, "model.audio_decoder.projection.weight", DType::bf16);
    ar.pos_emb   = load_f32(st, "model.audio_decoder.pos_embedding.weight");
    for (int i = 0;; ++i) {
        const std::string m = "model.audio_decoder.layers." + std::to_string(i) + ".";
        if (!st.has(m + "self_attn.qkv_proj.weight")) {
            break;
        }
        DecLayer L;
        L.in_norm   = load_f32(st, m + "input_layernorm.weight");
        L.post_norm = load_f32(st, m + "post_attention_layernorm.weight");
        L.qkv       = load_lin(st, m + "self_attn.qkv_proj");
        L.o         = load_lin(st, m + "self_attn.o_proj");
        L.gate_up   = load_lin(st, m + "mlp.gate_up_proj");
        L.down      = load_lin(st, m + "mlp.down_proj");
        ar.dec.push_back(std::move(L));
    }
    ar.dec_norm  = load_f32(st, "model.audio_decoder.norm.weight");
    ar.heads.wt  = DType::bf16;
    ar.heads.in  = kHid;
    ar.heads.out = (kCodebooks - 1) * kAudioVocab;
    ar.heads.w   = xn::resident(static_cast<size_t>(ar.heads.out * kHid) * 2);
    for (int i = 0; i < kCodebooks - 1; ++i) {
        const auto& t = st.info("model.audio_decoder.audio_heads." + std::to_string(i) + ".weight");
        st.upload_raw(t, ar.heads.w.as<bf16>() + static_cast<int64_t>(i) * kAudioVocab * kHid);
    }
    {
        auto b            = st.bytes_of("tokenizer_json");
        ar.tokenizer_json = std::string(b.begin(), b.end());
    }
    Q().wait();
    xn::pool().trim();
    LOG("AR: %zu LM layers + %zu depth layers loaded in %.1f s, %.0f MiB resident", ar.layers.size(), ar.dec.size(), now_s() - t0,
        xn::pool().resident() / 1048576.0);
    return ar;
}

// KV caches: LM (p, b, g, :) for p < cap; depth decoder (p, b, h, :) for p < 8.
struct ArState {
    int64_t B = 2, cap = 0, pos = 0;
    std::vector<Buf> k, v;  // per LM layer: bf16 [cap, B, kNKV, kHD]
    std::vector<Buf> dk, dv;  // per depth layer: bf16 [kCodebooks, B, kDecHeads, kDecHD]
};

// Run S new tokens per batch row (rows b * S + i, embeddings bf16 [B * S, 4096]) at positions st.pos ...;
// writes the final-normed hidden of each row's last token to last [B, 4096] (bf16).
void lm_forward(const AR& ar, ArState& s, const bf16* emb, int64_t S, bf16* last) {
    const int64_t B = s.B, R = B * S;
    if (s.pos + S > s.cap) {
        throw std::runtime_error("AR: KV cache full");
    }
    Buf x = xn::scratch_n<float>(R * kHid);
    xn::convert(emb, x.as<float>(), R * kHid);
    const float scale = 1.f / std::sqrt(static_cast<float>(kHD));
    for (size_t li = 0; li < ar.layers.size(); ++li) {
        const LmLayer& L = ar.layers[li];
        Buf h            = xn::scratch_n<bf16>(R * kHid);
        m3k::rms_norm(x.as<float>(), kHid, L.in_norm.as<float>(), h.as<bf16>(), R, kHid, kEps);
        Buf qkv = xn::scratch_n<bf16>(R * L.qkv.out);
        lin(h.as<bf16>(), R, L.qkv, qkv.get());
        Buf q = xn::scratch_n<bf16>(R * kNQ * kHD);
        m3k::qkv_scatter<kHD>(qkv.as<bf16>(), B, S, kNQ, kNKV, s.pos, L.q_norm.as<float>(), L.k_norm.as<float>(), kTheta, kEps, q.as<bf16>(),
                              s.k[li].as<bf16>(), s.v[li].as<bf16>());
        qkv.reset();
        Buf o = xn::scratch_n<bf16>(R * kNQ * kHD);
        const int64_t ks = B * kNKV * kHD;
        if (S == 1) {
            m3k::decode_attn<kNQ / kNKV>(q.as<bf16>(), s.k[li].as<bf16>(), s.v[li].as<bf16>(), ks, B * kNQ, s.pos + 1, scale, o.as<bf16>());
            // o [B * 32, 128] is already token-major [B, 4096]
        } else {
            Buf oh = xn::scratch_n<bf16>(R * kNQ * kHD);
            m3k::flash_attn_gqa<kHD / 16>(q.as<bf16>(), S * kHD, kHD, s.k[li].as<bf16>(), kHD, ks, s.v[li].as<bf16>(), kHD, ks, B * kNQ, kNQ / kNKV, S,
                                          s.pos + S, scale, s.pos, oh.as<bf16>());
            for (int64_t b = 0; b < B; ++b) {
                xn::heads_to_tokens(oh.as<bf16>() + b * kNQ * S * kHD, kNQ, S, kHD, 0, S, o.as<bf16>() + b * S * kHid);
            }
        }
        q.reset();
        lin(o.as<bf16>(), R, L.o, x.get(), DType::f32, true);
        m3k::rms_norm(x.as<float>(), kHid, L.post_norm.as<float>(), h.as<bf16>(), R, kHid, kEps);
        Buf gu = xn::scratch_n<bf16>(R * 2 * kFF);
        lin(h.as<bf16>(), R, L.gate_up, gu.get());
        Buf g = xn::scratch_n<bf16>(R * kFF);
        xn::silu_mul(gu.as<bf16>(), g.as<bf16>(), R, kFF);
        gu.reset();
        lin(g.as<bf16>(), R, L.down, x.get(), DType::f32, true);
    }
    m3k::rms_norm(x.as<float>() + (S - 1) * kHid, S * kHid, ar.norm.as<float>(), last, B, kHid, kEps);
    s.pos += S;
}

// Depth decoder over n new (already projected) tokens per row at positions p0 ...; in bf16 [B * n, 4096].
// out bf16 [B, 4096] = final-normed output of each row's last token.
void dec_forward(const AR& ar, ArState& s, const bf16* in, int64_t n, int64_t p0, bf16* out) {
    const int64_t B = s.B, R = B * n;
    Buf x           = xn::scratch_n<float>(R * kHid);
    {
        const float* pe = ar.pos_emb.as<float>();
        float* xp       = x.as<float>();
        Q().parallel_for(sycl::range<1>(static_cast<size_t>(R * kHid)), [=](sycl::id<1> id) {
            const int64_t i = id[0];
            const int64_t r = i / kHid, c = i % kHid;
            xp[i]           = static_cast<float>(in[i]) + pe[(p0 + r % n) * kHid + c];
        });
    }
    const float scale = 1.f / std::sqrt(static_cast<float>(kDecHD));
    for (size_t li = 0; li < ar.dec.size(); ++li) {
        const DecLayer& L = ar.dec[li];
        Buf h             = xn::scratch_n<bf16>(R * kHid);
        m3k::rms_norm(x.as<float>(), kHid, L.in_norm.as<float>(), h.as<bf16>(), R, kHid, kEps);
        Buf qkv = xn::scratch_n<bf16>(R * L.qkv.out);
        lin(h.as<bf16>(), R, L.qkv, qkv.get());
        Buf q = xn::scratch_n<bf16>(R * kHid);
        m3k::qkv_scatter<kDecHD>(qkv.as<bf16>(), B, n, kDecHeads, kDecHeads, p0, nullptr, nullptr, 0.f, kEps, q.as<bf16>(), s.dk[li].as<bf16>(),
                                 s.dv[li].as<bf16>());
        qkv.reset();
        Buf oh            = xn::scratch_n<bf16>(R * kHid);
        const int64_t ks  = B * kDecHeads * kDecHD;
        m3k::flash_attn_gqa<kDecHD / 16>(q.as<bf16>(), n * kDecHD, kDecHD, s.dk[li].as<bf16>(), kDecHD, ks, s.dv[li].as<bf16>(), kDecHD, ks,
                                         B * kDecHeads, 1, n, p0 + n, scale, p0, oh.as<bf16>());
        Buf o = xn::scratch_n<bf16>(R * kHid);
        for (int64_t b = 0; b < B; ++b) {
            xn::heads_to_tokens(oh.as<bf16>() + b * kDecHeads * n * kDecHD, kDecHeads, n, kDecHD, 0, n, o.as<bf16>() + b * n * kHid);
        }
        lin(o.as<bf16>(), R, L.o, x.get(), DType::f32, true);
        m3k::rms_norm(x.as<float>(), kHid, L.post_norm.as<float>(), h.as<bf16>(), R, kHid, kEps);
        Buf gu = xn::scratch_n<bf16>(R * 2 * kDecFF);
        lin(h.as<bf16>(), R, L.gate_up, gu.get());
        Buf g = xn::scratch_n<bf16>(R * kDecFF);
        xn::silu_mul(gu.as<bf16>(), g.as<bf16>(), R, kDecFF);
        gu.reset();
        lin(g.as<bf16>(), R, L.down, x.get(), DType::f32, true);
    }
    m3k::rms_norm(x.as<float>() + (n - 1) * kHid, n * kHid, ar.dec_norm.as<float>(), out, B, kHid, kEps);
}

// rows[r] = table[ids[r]] (bf16 rows of width W)
void gather(const bf16* table, const std::vector<int32_t>& ids, int64_t W, bf16* rows) {
    for (size_t r = 0; r < ids.size(); ++r) {
        Q().memcpy(rows + static_cast<int64_t>(r) * W, table + static_cast<int64_t>(ids[r]) * W, static_cast<size_t>(W) * 2);
    }
}

// Top-k sampling as in comfy/ldm/minimax_music/ar.py. c0: guided = u + (c - u) cfg over the indices whose
// conditioned logit reaches the conditioned top-k threshold, then top-k of guided; depth: top-k of guided.
struct Sampler {
    std::mt19937_64 rng;
    explicit Sampler(uint64_t seed) : rng(seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) {}

    int sample(const float* cond, const float* unc, int64_t n, float cfg, int top_k, bool cond_mask) {
        std::vector<std::pair<float, int>> cand;
        cand.reserve(static_cast<size_t>(n));
        float thr_c = -INFINITY;
        if (cond_mask) {
            std::vector<float> c(cond, cond + n);
            std::nth_element(c.begin(), c.begin() + (top_k - 1), c.end(), std::greater<float>());
            thr_c = c[top_k - 1];
        }
        for (int64_t i = 0; i < n; ++i) {
            if (cond_mask && cond[i] < thr_c) {
                continue;
            }
            float g = unc[i] + (cond[i] - unc[i]) * cfg;
            if (!std::isfinite(g)) {
                g = std::isnan(g) ? -1e9f : (g > 0 ? 1e9f : -1e9f);
            }
            cand.emplace_back(g, static_cast<int>(i));
        }
        const size_t k = std::min<size_t>(static_cast<size_t>(top_k), cand.size());
        std::nth_element(cand.begin(), cand.begin() + (k - 1), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        const float thr = cand[k - 1].first;
        std::vector<std::pair<float, int>> top;
        for (const auto& c : cand) {
            if (c.first >= thr) {
                top.push_back(c);
            }
        }
        float m = -INFINITY;
        for (const auto& c : top) {
            m = std::max(m, c.first);
        }
        double sum = 0;
        std::vector<double> p(top.size());
        for (size_t i = 0; i < top.size(); ++i) {
            p[i] = std::exp(static_cast<double>(top[i].first - m));
            sum += p[i];
        }
        double u = std::uniform_real_distribution<double>(0.0, sum)(rng);
        for (size_t i = 0; i < top.size(); ++i) {
            u -= p[i];
            if (u <= 0) {
                return top[i].second;
            }
        }
        return top.back().second;
    }
};

struct ArOptions {
    float cfg   = 1.5f;
    int top_k   = 50;
    uint64_t seed = 0;
    int64_t max_frames = 3000;
    const std::vector<int32_t>* forced = nullptr;  // selftest: [frames, 8] codes to use instead of sampling
    std::vector<float>* logits0        = nullptr;  // selftest: c0 logits [frames, 2, 16385]
    bool store_all                     = false;    // selftest: keep frame 0's hidden too
};

// y [M, rows] f32 = x [M, 4096] (bf16) * w[rows, 4096]^T for a weight slice
void dense_slice(const bf16* x, int64_t M, const bf16* w, int64_t rows, int64_t in, float* y) {
    xn::MatmulArgs a;
    a.M = M, a.K = in, a.N = rows;
    a.xt = DType::bf16, a.x = x, a.wt = DType::bf16, a.w = w, a.yt = DType::f32, a.y = y;
    xn::matmul(a);
}

// Prompt ids -> hidden frames bf16 [F, 8 * 4096] on the device (F in *frames_out). Mirrors MiniMaxMusic3AR.generate:
// frame 0's hidden is dropped, the frame that samples <|audio_end|> ends the song, at most max_frames frames.
Buf ar_generate(AR& ar, const std::vector<int32_t>& ids, const ArOptions& opt, int64_t* frames_out) {
    const int64_t P     = static_cast<int64_t>(ids.size());
    const int64_t limit = std::min<int64_t>(opt.max_frames, 9000);
    ArState s;
    s.B   = 2;
    s.cap = P + limit + 2;
    for (size_t i = 0; i < ar.layers.size(); ++i) {
        s.k.push_back(xn::scratch_n<bf16>(s.cap * s.B * kNKV * kHD));
        s.v.push_back(xn::scratch_n<bf16>(s.cap * s.B * kNKV * kHD));
    }
    for (size_t i = 0; i < ar.dec.size(); ++i) {
        s.dk.push_back(xn::scratch_n<bf16>(kCodebooks * s.B * kHid));
        s.dv.push_back(xn::scratch_n<bf16>(kCodebooks * s.B * kHid));
    }
    // prefill: row 0 = prompt, row 1 = the prompt with [1, P - 2) blanked to <|audio_cfg|>
    const double t0 = now_s();
    Buf last        = xn::scratch_n<bf16>(2 * kHid);
    {
        const auto& t = ar.st->info("model.embed_tokens_prefill.weight");
        std::vector<uint16_t> host(static_cast<size_t>(2 * P * kHid));
        for (int64_t b = 0; b < 2; ++b) {
            for (int64_t i = 0; i < P; ++i) {
                const int32_t id = (b == 1 && i >= 1 && i < P - 2) ? kAudioCfgId : ids[i];
                ar.st->read_into(t, static_cast<uint64_t>(id) * kHid * 2, host.data() + (b * P + i) * kHid, kHid * 2);
            }
        }
        Buf emb = xn::scratch_n<bf16>(2 * P * kHid);
        Q().memcpy(emb.get(), host.data(), host.size() * 2).wait();
        lm_forward(ar, s, emb.as<bf16>(), P, last.as<bf16>());
    }
    Q().wait();
    LOG("AR prefill: %lld prompt tokens x 2 rows in %.2f s", (long long)P, now_s() - t0);

    const int64_t keep = opt.store_all ? limit + 1 : std::max<int64_t>(limit, 1);
    Buf hidden         = xn::scratch_n<bf16>(keep * kCodebooks * kHid);
    Sampler smp(opt.seed);
    const int64_t NV = ar.lm_head.out;
    Buf logits       = xn::scratch_n<float>(2 * NV);
    Buf hl           = xn::scratch_n<float>(2 * kAudioVocab);
    Buf seq          = xn::scratch_n<bf16>(4 * kHid);  // depth decoder input rows b * n + i
    Buf ps           = xn::scratch_n<bf16>(4 * kHid);
    Buf dout         = xn::scratch_n<bf16>(2 * kHid);
    Buf fb           = xn::scratch_n<bf16>(2 * kHid);
    std::vector<float> lh(static_cast<size_t>(2 * NV)), dh(2 * kAudioVocab);
    int64_t stored  = 0;
    const double t1 = now_s();
    double last_log = t1;
    for (int64_t f = 0; f <= limit; ++f) {
        dense(last.get(), 2, ar.lm_head, logits.get(), DType::f32);
        Q().memcpy(lh.data(), logits.get(), lh.size() * 4).wait();
        if (opt.logits0 != nullptr) {
            opt.logits0->insert(opt.logits0->end(), lh.begin(), lh.end());
        }
        const int64_t nforced = opt.forced ? static_cast<int64_t>(opt.forced->size()) / kCodebooks : 0;
        if (opt.forced != nullptr && f >= nforced) {
            break;
        }
        const int code0 = opt.forced ? (*opt.forced)[f * kCodebooks] + 1 : smp.sample(lh.data(), lh.data() + NV, NV, opt.cfg, opt.top_k, true);
        if (code0 == 0) {
            break;  // <|audio_end|>
        }
        int32_t codes[kCodebooks];
        codes[0]  = code0 - 1;
        const bool keep_frame = f > 0 || opt.store_all;
        bf16* hf  = hidden.as<bf16>() + stored * kCodebooks * kHid;
        if (keep_frame) {
            Q().memcpy(hf, last.get(), kHid * 2);  // row 0 (conditioned)
        }
        // depth decoder: positions 0, 1 = proj(hidden), proj(emb_audio[c0]); then proj(extra[code_i]) per codebook
        for (int64_t b = 0; b < 2; ++b) {
            Q().memcpy(seq.as<bf16>() + (b * 2) * kHid, last.as<bf16>() + b * kHid, kHid * 2);
            Q().memcpy(seq.as<bf16>() + (b * 2 + 1) * kHid, ar.emb_audio.as<bf16>() + static_cast<int64_t>(codes[0]) * kHid, kHid * 2);
        }
        dense(seq.get(), 4, ar.proj, ps.get(), DType::bf16);
        int64_t n = 2, p0 = 0;
        for (int i = 1; i < kCodebooks; ++i) {
            dec_forward(ar, s, ps.as<bf16>(), n, p0, dout.as<bf16>());
            p0 += n;
            if (keep_frame) {
                Q().memcpy(hf + i * kHid, dout.get(), kHid * 2);
            }
            dense_slice(dout.as<bf16>(), 2, ar.heads.w.as<bf16>() + static_cast<int64_t>(i - 1) * kAudioVocab * kHid, kAudioVocab, kHid, hl.as<float>());
            Q().memcpy(dh.data(), hl.get(), dh.size() * 4).wait();
            codes[i] = opt.forced ? (*opt.forced)[f * kCodebooks + i] : smp.sample(dh.data(), dh.data() + kAudioVocab, kAudioVocab, opt.cfg, opt.top_k, false);
            if (i < kCodebooks - 1) {
                const bf16* e = ar.emb_extra.as<bf16>() + static_cast<int64_t>(codes[i] + (i - 1) * kAudioVocab) * kHid;
                Q().memcpy(seq.get(), e, kHid * 2);
                Q().memcpy(seq.as<bf16>() + kHid, e, kHid * 2);
                dense(seq.get(), 2, ar.proj, ps.get(), DType::bf16);
                n = 1;
            }
        }
        if (keep_frame) {
            ++stored;
        }
        if (stored >= limit && !opt.store_all) {
            break;
        }
        // feedback: (emb_audio[c0] + sum_i extra[code_i + (i - 1) 1024]) / sqrt(8), same for both rows
        {
            struct Idx {
                int32_t r[kCodebooks];
            } idx;
            idx.r[0] = codes[0];
            for (int i = 1; i < kCodebooks; ++i) {
                idx.r[i] = codes[i] + (i - 1) * kAudioVocab;
            }
            const bf16* ea = ar.emb_audio.as<bf16>();
            const bf16* ee = ar.emb_extra.as<bf16>();
            bf16* out      = fb.as<bf16>();
            const float sc = 1.f / std::sqrt(static_cast<float>(kCodebooks));
            Q().parallel_for(sycl::range<1>(kHid), [=](sycl::id<1> id) {
                const int64_t c = id[0];
                float v         = static_cast<float>(ea[static_cast<int64_t>(idx.r[0]) * kHid + c]);
                for (int i = 1; i < kCodebooks; ++i) {
                    v += static_cast<float>(ee[static_cast<int64_t>(idx.r[i]) * kHid + c]);
                }
                out[c] = out[kHid + c] = static_cast<bf16>(v * sc);
            });
        }
        lm_forward(ar, s, fb.as<bf16>(), 1, last.as<bf16>());
        if (g_verbose && now_s() - last_log > 10.0) {
            last_log = now_s();
            LOG("AR: %lld frames (%.1f s of audio), %.1f frames/s", (long long)stored, stored / 25.0, (f + 1) / (now_s() - t1));
        }
    }
    Q().wait();
    LOG("AR: %lld frames (%.2f s of audio) in %.1f s", (long long)stored, stored / 25.0, now_s() - t1);
    *frames_out = stored;
    return hidden;
}

// ============================================================================================================
// DiT
// ============================================================================================================

constexpr int kDim = 2048, kDitHeads = 32, kDitHD = 64, kDitFF = 8192, kLat = 128, kCondCh = 2048, kInCh = 2304;

int64_t latent_length(int64_t frames) { return std::max<int64_t>(1, static_cast<int64_t>(frames * 44100.0 / 24000.0 * 960.0 / 512.0)); }

struct DitBlock {
    Buf ln1_g, ln1_b, ln2_g, ln2_b;
    Lin qkv, out, ff_in, ff_out;
};

struct DiT {
    std::vector<DitBlock> blocks;
    Dense pre, proj_in, proj_out, post, cond_conv;  // f32
    Buf inv_freq;                                   // f32 [16]
    std::vector<float> tf_w, te0_w, te0_b, te2_w, te2_b;
    float layer_w[8] = {};  // softmax(cond_layer_logits) * cond_layer_scale
};

DiT load_dit(const std::string& path) {
    const double t0 = now_s();
    xn::SafeTensors st(path);
    DiT d;
    const std::string T = "diffusion_transformer.transformer.";
    for (int i = 0;; ++i) {
        const std::string m = T + "layers." + std::to_string(i) + ".";
        if (!st.has(m + "self_attn.to_qkv.weight")) {
            break;
        }
        DitBlock b;
        b.ln1_g  = load_f32(st, m + "pre_norm.gamma");
        b.ln1_b  = load_f32(st, m + "pre_norm.beta");
        b.ln2_g  = load_f32(st, m + "ff_norm.gamma");
        b.ln2_b  = load_f32(st, m + "ff_norm.beta");
        b.qkv    = load_lin(st, m + "self_attn.to_qkv");
        b.out    = load_lin(st, m + "self_attn.to_out");
        b.ff_in  = load_lin(st, m + "ff.ff.0.proj", true);
        b.ff_out = load_lin(st, m + "ff.ff.2", true);
        d.blocks.push_back(std::move(b));
    }
    d.pre       = load_dense(st, "diffusion_transformer.preprocess_conv.weight", DType::f32);
    d.proj_in   = load_dense(st, T + "project_in.weight", DType::f32);
    d.proj_out  = load_dense(st, T + "project_out.weight", DType::f32);
    d.post      = load_dense(st, "diffusion_transformer.postprocess_conv.weight", DType::f32);
    d.cond_conv = load_dense(st, "latent_conditioners.0.weight", DType::f32, "latent_conditioners.0.bias");
    d.inv_freq  = load_f32(st, T + "rotary_pos_emb.inv_freq");
    d.tf_w      = st.host_f32("diffusion_transformer.timestep_features.weight");
    d.te0_w     = st.host_f32("diffusion_transformer.to_timestep_embed.0.weight");
    d.te0_b     = st.host_f32("diffusion_transformer.to_timestep_embed.0.bias");
    d.te2_w     = st.host_f32("diffusion_transformer.to_timestep_embed.2.weight");
    d.te2_b     = st.host_f32("diffusion_transformer.to_timestep_embed.2.bias");
    auto logits = st.host_f32("cond_layer_logits");
    const float scale = st.host_f32("cond_layer_scale").at(0);
    float m = -INFINITY, sum = 0.f;
    for (float v : logits) {
        m = std::max(m, v);
    }
    for (int i = 0; i < 8; ++i) {
        d.layer_w[i] = std::exp(logits.at(i) - m);
        sum += d.layer_w[i];
    }
    for (float& w : d.layer_w) {
        w = w / sum * scale;
    }
    Q().wait();
    xn::pool().trim();
    LOG("DiT: %zu blocks loaded in %.1f s, %.0f MiB resident", d.blocks.size(), now_s() - t0, xn::pool().resident() / 1048576.0);
    return d;
}

// Timestep embedding (host, f32): [cos(2 pi t w), sin(2 pi t w)] -> Linear -> SiLU -> Linear.
std::vector<float> timestep_embed(const DiT& d, float t) {
    const int nf = static_cast<int>(d.tf_w.size());  // 128
    std::vector<float> feat(2 * nf), h(kDim), out(kDim);
    for (int i = 0; i < nf; ++i) {
        const float f = static_cast<float>(2.0 * M_PI) * t * d.tf_w[i];
        feat[i]       = std::cos(f);
        feat[nf + i]  = std::sin(f);
    }
    for (int o = 0; o < kDim; ++o) {
        float a = d.te0_b[o];
        for (int i = 0; i < 2 * nf; ++i) {
            a += d.te0_w[static_cast<size_t>(o) * 2 * nf + i] * feat[i];
        }
        h[o] = a / (1.f + std::exp(-a));
    }
    for (int o = 0; o < kDim; ++o) {
        float a = d.te2_b[o];
        for (int i = 0; i < kDim; ++i) {
            a += d.te2_w[static_cast<size_t>(o) * kDim + i] * h[i];
        }
        out[o] = a;
    }
    return out;
}

// DiffusionTransformer on one window: x [T, 128], cond [T, 2048] (token-major f32 device, cond null = zeros),
// temb [2048] -> y [T, 128] = the transformer output (postprocess conv applied, NOT negated).
void dit_forward(const DiT& d, const float* x, const float* cond, const std::vector<float>& temb, int64_t T, float* y) {
    const int64_t L = T + 1;
    Buf F           = xn::scratch_n<float>(T * kInCh);
    {
        float* Fp = F.as<float>();
        Q().parallel_for(sycl::range<1>(static_cast<size_t>(T * kInCh)), [=](sycl::id<1> id) {
            const int64_t i = id[0];
            const int64_t t = i / kInCh, c = i % kInCh;
            Fp[i]           = c < kLat ? x[t * kLat + c] : c < 2 * kLat ? 0.f : (cond != nullptr ? cond[t * kCondCh + c - 2 * kLat] : 0.f);
        });
    }
    Buf G = xn::scratch_n<float>(T * kInCh);
    Q().memcpy(G.get(), F.get(), F.bytes());
    dense(F.get(), T, d.pre, G.get(), DType::f32, true);  // G = F W^T + F
    F.reset();
    Buf X = xn::scratch_n<float>(L * kDim);
    Q().memcpy(X.get(), temb.data(), kDim * 4).wait();
    dense(G.get(), T, d.proj_in, X.as<float>() + kDim, DType::f32);
    G.reset();
    Buf h  = xn::scratch_n<bf16>(L * kDim);
    Buf qk = xn::scratch_n<bf16>(L * 3 * kDim);
    Buf q  = xn::scratch_n<bf16>(L * kDim), k = xn::scratch_n<bf16>(L * kDim), v = xn::scratch_n<bf16>(L * kDim);
    Buf oh = xn::scratch_n<bf16>(L * kDim), o = xn::scratch_n<bf16>(L * kDim);
    Buf vg = xn::scratch_n<bf16>(L * 2 * kDitFF), g = xn::scratch_n<bf16>(L * kDitFF);
    const float scale = 1.f / std::sqrt(static_cast<float>(kDitHD));
    for (const DitBlock& b : d.blocks) {
        m3k::layer_norm(X.as<float>(), b.ln1_g.as<float>(), b.ln1_b.as<float>(), h.as<bf16>(), L, kDim, 1e-5f);
        lin(h.as<bf16>(), L, b.qkv, qk.get());
        m3k::dit_qkv_rope(qk.as<bf16>(), d.inv_freq.as<float>(), L, kDitHeads, q.as<bf16>(), k.as<bf16>(), v.as<bf16>());
        xn::attention(DType::bf16, q.get(), L * kDitHD, kDitHD, k.get(), L * kDitHD, kDitHD, v.get(), L * kDitHD, kDitHD, kDitHeads, L, L, kDitHD,
                      scale, -1, oh.get(), true, g_verbose > 1, "xpu_music3");
        xn::heads_to_tokens(oh.as<bf16>(), kDitHeads, L, kDitHD, 0, L, o.as<bf16>());
        lin(o.as<bf16>(), L, b.out, X.get(), DType::f32, true);
        m3k::layer_norm(X.as<float>(), b.ln2_g.as<float>(), b.ln2_b.as<float>(), h.as<bf16>(), L, kDim, 1e-5f);
        lin(h.as<bf16>(), L, b.ff_in, vg.get());
        m3k::glu_value_gate(vg.as<bf16>(), g.as<bf16>(), L, kDitFF);
        lin(g.as<bf16>(), L, b.ff_out, X.get(), DType::f32, true);
    }
    Buf O = xn::scratch_n<float>(T * kLat);
    dense(X.as<float>() + kDim, T, d.proj_out, O.get(), DType::f32);
    Q().memcpy(y, O.get(), O.bytes());
    dense(O.get(), T, d.post, y, DType::f32, true);  // y = O W^T + O
}

// aligned_condition: hidden bf16 [F, 8 * 4096] -> layer-weighted sum -> Conv1d(4096 -> 2048, k3, pad 1) ->
// nearest resize to Tl latents. Returns f32 [Tl, 2048] (token-major) on the device.
Buf make_condition(const DiT& d, const bf16* hidden, int64_t F, int64_t Tl) {
    Buf X = xn::scratch_n<float>(F * kHid * 3);  // im2col: X[f, i * 3 + k] = wsum[f + k - 1, i]
    {
        struct W8 {
            float w[8];
        } lw;
        std::copy(d.layer_w, d.layer_w + 8, lw.w);
        float* Xp = X.as<float>();
        Q().parallel_for(sycl::range<2>(static_cast<size_t>(F), kHid), [=](sycl::id<2> id) {
            const int64_t f = id[0], i = id[1];
            float s         = 0.f;
            for (int l = 0; l < 8; ++l) {
                s += lw.w[l] * static_cast<float>(hidden[f * 8 * kHid + l * kHid + i]);
            }
            for (int k = 0; k < 3; ++k) {
                const int64_t ff = f - k + 1;  // wsum[f] lands in row f - k + 1 at tap k
                if (ff >= 0 && ff < F) {
                    Xp[ff * kHid * 3 + i * 3 + k] = s;
                }
            }
            if (f == 0) {
                Xp[i * 3 + 0] = 0.f;  // row 0, tap 0 reads wsum[-1]
            }
            if (f == F - 1) {
                Xp[(F - 1) * kHid * 3 + i * 3 + 2] = 0.f;  // last row, tap 2 reads wsum[F]
            }
        });
    }
    Buf Y = xn::scratch_n<float>(F * kCondCh);
    dense(X.get(), F, d.cond_conv, Y.get(), DType::f32);
    X.reset();
    Buf out = xn::scratch_n<float>(Tl * kCondCh);
    {
        const float* Yp = Y.as<float>();
        float* op       = out.as<float>();
        const float sc  = static_cast<float>(F) / static_cast<float>(Tl);
        Q().parallel_for(sycl::range<2>(static_cast<size_t>(Tl), kCondCh), [=](sycl::id<2> id) {
            const int64_t t = id[0], c = id[1];
            int64_t src;
            if (Tl == F) {
                src = t;
            } else if (Tl == 2 * F) {
                src = t >> 1;
            } else {
                src = sycl::min(static_cast<int64_t>(sycl::floor(static_cast<float>(t) * sc)), F - 1);
            }
            op[t * kCondCh + c] = Yp[src * kCondCh + c];
        });
    }
    Q().wait();
    return out;
}

// Model output (comfy MiniMaxMusic3DiT.forward, = -transformer) for the whole latent, windows past 689 latents.
void model_out(const DiT& d, const std::vector<float>& x, const float* cond, const std::vector<float>& temb, int64_t Tl, std::vector<float>& out) {
    const int64_t window = latent_length(200), hop = latent_length(100);
    out.assign(static_cast<size_t>(Tl * kLat), 0.f);
    std::vector<float> cnt(static_cast<size_t>(Tl), 0.f), tmp;
    Buf xd = xn::scratch_n<float>(std::min(Tl, window) * kLat);
    Buf yd = xn::scratch_n<float>(std::min(Tl, window) * kLat);
    for (int64_t start = 0;; start += hop) {
        const int64_t end = std::min(start + window, Tl), n = end - start;
        Q().memcpy(xd.get(), x.data() + start * kLat, static_cast<size_t>(n * kLat) * 4).wait();
        dit_forward(d, xd.as<float>(), cond ? cond + start * kCondCh : nullptr, temb, n, yd.as<float>());
        tmp.resize(static_cast<size_t>(n * kLat));
        Q().memcpy(tmp.data(), yd.get(), tmp.size() * 4).wait();
        for (int64_t i = 0; i < n * kLat; ++i) {
            out[start * kLat + i] -= tmp[i];
        }
        for (int64_t t = start; t < end; ++t) {
            cnt[t] += 1.f;
        }
        if (end == Tl || Tl <= window) {
            break;
        }
    }
    for (int64_t t = 0; t < Tl; ++t) {
        for (int c = 0; c < kLat; ++c) {
            out[t * kLat + c] /= cnt[t];
        }
    }
}

// euler / simple flow sampling with CFG against a zero condition. Returns the latent channel-major [128, Tl].
std::vector<float> dit_sample(const DiT& d, const float* cond, int64_t Tl, int steps, float cfg, uint64_t seed) {
    std::vector<float> x(static_cast<size_t>(Tl * kLat));
    {
        MT19937RNG rng;
        rng.manual_seed(seed);
        auto n = rng.randn(static_cast<uint32_t>(kLat * Tl));  // torch.randn([1, 128, Tl]) on the CPU generator
        for (int c = 0; c < kLat; ++c) {
            for (int64_t t = 0; t < Tl; ++t) {
                x[t * kLat + c] = n[c * Tl + t];
            }
        }
    }
    std::vector<float> sig(static_cast<size_t>(steps + 1));
    const double ss = 1000.0 / steps;
    for (int i = 0; i < steps; ++i) {
        sig[i] = static_cast<float>(1000 - static_cast<int64_t>(i * ss)) / 1000.f;
    }
    sig[steps] = 0.f;
    std::vector<float> oc, ou;
    const double t0 = now_s();
    for (int i = 0; i < steps; ++i) {
        const float s = sig[i], sn = sig[i + 1];
        const auto temb = timestep_embed(d, 1.f - s);
        model_out(d, x, cond, temb, Tl, oc);
        if (cfg != 1.f) {
            model_out(d, x, nullptr, temb, Tl, ou);
        }
        for (size_t j = 0; j < x.size(); ++j) {
            const float dc  = x[j] - oc[j] * s;
            const float den = cfg != 1.f ? (x[j] - ou[j] * s) + (dc - (x[j] - ou[j] * s)) * cfg : dc;
            const float dd  = (x[j] - den) / s;
            x[j]            = x[j] + dd * (sn - s);
        }
        if (g_verbose) {
            std::fprintf(stderr, "\r[xpu_music3] DiT step %d/%d (%.1f s)", i + 1, steps, now_s() - t0);
        }
    }
    if (g_verbose) {
        std::fputc('\n', stderr);
    }
    std::vector<float> lat(x.size());
    for (int64_t t = 0; t < Tl; ++t) {
        for (int c = 0; c < kLat; ++c) {
            lat[c * Tl + t] = x[t * kLat + c];
        }
    }
    return lat;
}

// ============================================================================================================
// DAV decoder (oneDNN 1-D convolutions, f32)
// ============================================================================================================

struct Conv {
    Buf w, b;
    int64_t O = 0, I = 0, K = 0, stride = 1, pad = 0, dil = 1;
    bool transpose = false;
};

// weight_norm (dim 0): w = g[d0] * v / ||v[d0, ...]||
Conv load_conv(xn::SafeTensors& st, const std::string& p, int64_t stride, int64_t pad, int64_t dil, bool transpose) {
    Conv c;
    c.stride = stride, c.pad = pad, c.dil = dil, c.transpose = transpose;
    std::vector<float> w;
    std::vector<int64_t> shape;
    if (st.has(p + ".weight_v")) {
        w          = st.host_f32(p + ".weight_v");
        auto g     = st.host_f32(p + ".weight_g");
        shape      = st.info(p + ".weight_v").shape;
        const int64_t d0 = shape[0], rest = static_cast<int64_t>(w.size()) / d0;
        for (int64_t i = 0; i < d0; ++i) {
            double n2 = 0;
            for (int64_t j = 0; j < rest; ++j) {
                n2 += static_cast<double>(w[i * rest + j]) * w[i * rest + j];
            }
            const float f = g[i] / static_cast<float>(std::sqrt(n2));
            for (int64_t j = 0; j < rest; ++j) {
                w[i * rest + j] *= f;
            }
        }
    } else {
        w     = st.host_f32(p + ".weight");
        shape = st.info(p + ".weight").shape;
    }
    c.K = shape.at(2);
    c.O = transpose ? shape[1] : shape[0];
    c.I = transpose ? shape[0] : shape[1];
    c.w = xn::resident(w.size() * 4);
    Q().memcpy(c.w.get(), w.data(), w.size() * 4).wait();
    auto b = st.host_f32(p + ".bias");
    c.b    = xn::resident(b.size() * 4);
    Q().memcpy(c.b.get(), b.data(), b.size() * 4).wait();
    return c;
}

int64_t conv_out_len(const Conv& c, int64_t W) {
    return c.transpose ? (W - 1) * c.stride - 2 * c.pad + c.K : (W + 2 * c.pad - c.dil * (c.K - 1) - 1) / c.stride + 1;
}

// y [N, O, OW] = conv(x [N, I, W]) (+ bias)
void conv_run(const Conv& c, const float* x, int64_t N, int64_t W, float* y) {
    using namespace dnnl;
    using tag        = memory::format_tag;
    const int64_t OW = conv_out_len(c, W);
    struct Ent {
        primitive prim;
        memory::desc src, wei, bias, dst, scr;
    };
    static std::map<std::string, std::shared_ptr<Ent>> cache;
    char key[200];
    std::snprintf(key, sizeof(key), "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d", (long long)N, (long long)W, (long long)c.O, (long long)c.I,
                  (long long)c.K, (long long)c.stride, (long long)c.pad, (long long)c.dil, (int)c.transpose);
    auto& e = cache[key];
    if (!e) {
        e      = std::make_shared<Ent>();
        e->src = memory::desc({N, c.I, W}, memory::data_type::f32, tag::ncw);
        e->dst = memory::desc({N, c.O, OW}, memory::data_type::f32, tag::ncw);
        e->bias = memory::desc({c.O}, memory::data_type::f32, tag::x);
        e->wei  = memory::desc({c.O, c.I, c.K}, memory::data_type::f32, c.transpose ? tag::bac : tag::abc);
        primitive_attr attr;
        attr.set_scratchpad_mode(scratchpad_mode::user);
        if (c.transpose) {
            auto pd = deconvolution_forward::primitive_desc(xn::rt().eng, prop_kind::forward_inference, algorithm::deconvolution_direct, e->src, e->wei,
                                                            e->bias, e->dst, {c.stride}, {c.dil - 1}, {c.pad}, {c.pad}, attr);
            e->scr  = pd.scratchpad_desc();
            e->prim = deconvolution_forward(pd);
        } else {
            auto pd = convolution_forward::primitive_desc(xn::rt().eng, prop_kind::forward_inference, algorithm::convolution_direct, e->src, e->wei,
                                                          e->bias, e->dst, {c.stride}, {c.dil - 1}, {c.pad}, {c.pad}, attr);
            e->scr  = pd.scratchpad_desc();
            e->prim = convolution_forward(pd);
        }
    }
    auto mem = [&](const memory::desc& md, const void* p) {
        return sycl_interop::make_memory(md, xn::rt().eng, sycl_interop::memory_kind::usm, const_cast<void*>(p));
    };
    std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, mem(e->src, x)}, {DNNL_ARG_WEIGHTS, mem(e->wei, c.w.get())},
                                            {DNNL_ARG_BIAS, mem(e->bias, c.b.get())}, {DNNL_ARG_DST, mem(e->dst, y)}};
    Buf scr;
    if (e->scr.get_size() > 0) {
        scr                       = xn::scratch(e->scr.get_size());
        args[DNNL_ARG_SCRATCHPAD] = mem(e->scr, scr.get());
    }
    e->prim.execute(xn::rt().strm, args);
}

struct DavRU {
    Buf a1, a2;
    Conv c7, c1;
};
struct DavBlock {
    Buf alpha;
    Conv up;
    DavRU ru[3];
};
struct DAV {
    Conv in_proj, conv_in, conv_out;
    DavBlock blocks[4];
    Buf alpha_out;
};

DAV load_dav(const std::string& path) {
    xn::SafeTensors st(path);
    DAV d;
    d.in_proj         = load_conv(st, "dec_in_proj", 1, 0, 1, false);
    d.conv_in         = load_conv(st, "decoder.model.0", 1, 3, 1, false);
    const int strides[4] = {8, 8, 4, 2};
    for (int b = 0; b < 4; ++b) {
        const std::string m = "decoder.model." + std::to_string(b + 1) + ".block.";
        DavBlock& B         = d.blocks[b];
        B.alpha             = load_f32(st, m + "0.alpha");
        B.up                = load_conv(st, m + "1", strides[b], (strides[b] + 1) / 2, 1, true);
        const int dil[3]    = {1, 3, 9};
        for (int r = 0; r < 3; ++r) {
            const std::string u = m + std::to_string(r + 2) + ".block.";
            B.ru[r].a1          = load_f32(st, u + "0.alpha");
            B.ru[r].c7          = load_conv(st, u + "1", 1, 3 * dil[r], dil[r], false);
            B.ru[r].a2          = load_f32(st, u + "2.alpha");
            B.ru[r].c1          = load_conv(st, u + "3", 1, 0, 1, false);
        }
    }
    d.alpha_out = load_f32(st, "decoder.model.5.alpha");
    d.conv_out  = load_conv(st, "decoder.model.6", 1, 3, 1, false);
    Q().wait();
    return d;
}

// z [128, T] (channel-major, host) -> waveform [2, T * 512] (host), tiled over time (receptive field < 16 latents).
std::vector<float> dav_decode(const DAV& d, const std::vector<float>& z, int64_t T) {
    const int64_t core = 512, halo = 16, up = 512;
    std::vector<float> wav(static_cast<size_t>(2 * T * up));
    for (int64_t t0 = 0; t0 < T; t0 += core) {
        const int64_t a0 = std::max<int64_t>(0, t0 - halo), a1 = std::min(T, t0 + core + halo), n = a1 - a0;
        std::vector<float> zt(static_cast<size_t>(2 * 64 * n));
        for (int64_t c = 0; c < 128; ++c) {
            std::copy(z.begin() + c * T + a0, z.begin() + c * T + a1, zt.begin() + c * n);  // [2, 64, n] = channels b * 64 + c
        }
        Buf x = xn::scratch_n<float>(2 * 64 * n);
        Q().memcpy(x.get(), zt.data(), zt.size() * 4).wait();
        int64_t W = n, C = 64;
        auto step = [&](const Conv& c) {
            const int64_t OW = conv_out_len(c, W);
            Buf y            = xn::scratch_n<float>(2 * c.O * OW);
            conv_run(c, x.as<float>(), 2, W, y.as<float>());
            x = std::move(y);
            W = OW;
            C = c.O;
        };
        step(d.in_proj);
        step(d.conv_in);
        for (const DavBlock& B : d.blocks) {
            m3k::snake(x.as<float>(), B.alpha.as<float>(), x.as<float>(), 2, C, W);
            step(B.up);
            for (const DavRU& r : B.ru) {
                Buf t1 = xn::scratch_n<float>(2 * C * W), t2 = xn::scratch_n<float>(2 * C * W);
                m3k::snake(x.as<float>(), r.a1.as<float>(), t1.as<float>(), 2, C, W);
                conv_run(r.c7, t1.as<float>(), 2, W, t2.as<float>());
                m3k::snake(t2.as<float>(), r.a2.as<float>(), t2.as<float>(), 2, C, W);
                conv_run(r.c1, t2.as<float>(), 2, W, t1.as<float>());
                xn::add_inplace(x.as<float>(), t1.as<float>(), 2 * C * W);
            }
        }
        m3k::snake(x.as<float>(), d.alpha_out.as<float>(), x.as<float>(), 2, C, W);
        step(d.conv_out);
        {
            float* p = x.as<float>();
            Q().parallel_for(sycl::range<1>(static_cast<size_t>(2 * W)), [=](sycl::id<1> i) { p[i] = sycl::tanh(p[i]); });
        }
        std::vector<float> out(static_cast<size_t>(2 * W));
        Q().memcpy(out.data(), x.get(), out.size() * 4).wait();
        const int64_t keep0 = (t0 - a0) * up, keepn = (std::min(T, t0 + core) - t0) * up;
        for (int ch = 0; ch < 2; ++ch) {
            std::copy(out.begin() + ch * W + keep0, out.begin() + ch * W + keep0 + keepn, wav.begin() + ch * T * up + t0 * up);
        }
    }
    return wav;
}

// ============================================================================================================
// prompt, wav, CLI
// ============================================================================================================

std::string replace_all(std::string s, const std::string& a, const std::string& b) {
    for (size_t p = 0; (p = s.find(a, p)) != std::string::npos; p += b.size()) {
        s.replace(p, a.size(), b);
    }
    return s;
}

// comfy/ldm/minimax_music/prompt.py clean_caption
std::string clean_caption(const std::string& caption) {
    std::string text;
    {
        static const std::regex tag(R"(<\|([^|]*)\|>)");
        std::string rest = caption;
        std::smatch m;
        while (std::regex_search(rest, m, tag)) {
            text += m.prefix().str();
            std::string inner = m[1].str();
            const size_t b = inner.find_first_not_of(" \t\n\r"), e = inner.find_last_not_of(" \t\n\r");
            inner          = b == std::string::npos ? "" : inner.substr(b, e - b + 1);
            const size_t sp = inner.find_first_of(" \t\n\r");
            if (sp != std::string::npos) {
                const size_t r = inner.find_first_not_of(" \t\n\r", sp);
                inner          = inner.substr(0, sp) + " is " + inner.substr(r);
            }
            text += inner;
            rest = m.suffix().str();
        }
        text += rest;
    }
    std::stringstream in(text);
    std::string line, joined;
    bool first = true;
    static const std::regex heading(R"(^\s{0,3}#{1,6}\s+)"), bullet(R"(^\s*[*+-]\s+)"), bold(R"(\*\*([^*]+)\*\*)"), rule(R"(^\s*[-*_]{3,}\s*$)");
    while (std::getline(in, line)) {
        line = std::regex_replace(line, heading, "", std::regex_constants::format_first_only);
        line = std::regex_replace(line, bullet, "", std::regex_constants::format_first_only);
        while (line.find("**") != std::string::npos) {
            const std::string u = std::regex_replace(line, bold, "$1");
            if (u == line) {
                break;
            }
            line = u;
        }
        // (?<!\*)\*([^*\n]+)\*(?!\*) -> \1
        std::string o;
        for (size_t i = 0; i < line.size();) {
            if (line[i] == '*' && (i == 0 || line[i - 1] != '*')) {
                const size_t j = line.find('*', i + 1);
                if (j != std::string::npos && j > i + 1 && (j + 1 >= line.size() || line[j + 1] != '*')) {
                    o += line.substr(i + 1, j - i - 1);
                    i = j + 1;
                    continue;
                }
            }
            o += line[i++];
        }
        line = o;
        const size_t e = line.find_last_not_of(" \t\r\f\v");
        line           = e == std::string::npos ? "" : line.substr(0, e + 1);
        if (std::regex_match(line, rule)) {
            line.clear();
        }
        joined += (first ? "" : "\n") + line;
        first = false;
    }
    joined = replace_all(replace_all(joined, "\xE2\x80\xA2 ", ""), "    ", "");
    static const std::regex blank(R"(\n{2,})");
    return std::regex_replace(joined, blank, "\n");
}

// normalize_lyrics: section tags lower-cased on their own lines, " ^ " -> newline, "[start]" first.
std::string normalize_lyrics(const std::string& lyrics) {
    static const std::regex tag(R"(\s*(\[[^\]]+\])\s*)");
    std::vector<std::string> parts;
    std::string rest = lyrics;
    std::smatch m;
    while (std::regex_search(rest, m, tag)) {
        parts.push_back(m.prefix().str());
        std::string t = m[1].str();
        std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        parts.push_back(t);
        rest = m.suffix().str();
    }
    parts.push_back(rest);
    std::string text;
    bool first = true;
    for (const auto& p : parts) {
        if (p.empty()) {
            continue;
        }
        text += (first ? "" : "\n") + p;
        first = false;
    }
    return "[start]\n" + replace_all(text, " ^ ", "\n");
}

std::vector<int32_t> tokenize(const AR& ar, const std::string& caption, const std::string& lyrics) {
    const std::string prompt = "<|im_start|><|caption_start|>" + clean_caption(caption) + "<|caption_end|><|lyrics_start|>" + normalize_lyrics(lyrics) +
                               "<|lyrics_end|><|im_end|><|audio_start|>";
    const char* cache = std::getenv("XDG_CACHE_HOME");
    const std::filesystem::path dir =
        std::filesystem::path(cache ? cache : std::string(std::getenv("HOME")) + "/.cache") / "xpu_music3";
    std::filesystem::create_directories(dir);
    const auto path = (dir / "tokenizer.json").string();
    if (!std::filesystem::exists(path) || std::filesystem::file_size(path) != ar.tokenizer_json.size()) {
        xn::write_file(path + ".tmp", ar.tokenizer_json.data(), ar.tokenizer_json.size());
        std::filesystem::rename(path + ".tmp", path);
    }
    HFTokenizer tok(path);
    std::vector<int> ids;
    std::string err;
    if (!tok.encode(prompt, ids, nullptr, &err)) {
        throw std::runtime_error("tokenizer: " + err);
    }
    if (ids.size() > 5000) {
        throw std::runtime_error("prompt has " + std::to_string(ids.size()) + " tokens; the maximum is 5000");
    }
    return std::vector<int32_t>(ids.begin(), ids.end());
}

void write_wav(const std::string& path, const std::vector<float>& wav, int64_t n, int sr) {
    std::vector<int16_t> pcm(static_cast<size_t>(2 * n));
    for (int64_t i = 0; i < n; ++i) {
        for (int c = 0; c < 2; ++c) {
            const float v      = std::clamp(wav[c * n + i], -1.f, 1.f);
            pcm[2 * i + c]     = static_cast<int16_t>(std::lrint(v * 32767.f));
        }
    }
    std::ofstream f(path + ".tmp", std::ios::binary);
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    const uint32_t data = static_cast<uint32_t>(pcm.size() * 2);
    f.write("RIFF", 4), u32(36 + data), f.write("WAVE", 4), f.write("fmt ", 4), u32(16), u16(1), u16(2), u32(sr), u32(sr * 4), u16(4), u16(16);
    f.write("data", 4), u32(data);
    f.write(reinterpret_cast<const char*>(pcm.data()), data);
    f.close();
    std::filesystem::rename(path + ".tmp", path);
}

std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class T>
std::vector<T> read_raw(const std::string& path) {
    return xn::read_vec<T>(path);
}

int selftest(const std::string& what, const std::string& dir, const std::string& te, const std::string& dit, const std::string& dav) {
    auto meta = xn::read_meta(dir);
    if (what == "dit") {
        DiT d          = load_dit(dit);
        const int64_t T = static_cast<int64_t>(meta["T"]);
        auto x = read_raw<float>(dir + "/x.f32"), c = read_raw<float>(dir + "/cond.f32"), ref = read_raw<float>(dir + "/out.f32");
        std::vector<float> xt(T * kLat), ct(T * kCondCh);
        for (int64_t t = 0; t < T; ++t) {
            for (int i = 0; i < kLat; ++i) xt[t * kLat + i] = x[i * T + t];
            for (int i = 0; i < kCondCh; ++i) ct[t * kCondCh + i] = c[i * T + t];
        }
        Buf xd = xn::scratch_n<float>(T * kLat), cd = xn::scratch_n<float>(T * kCondCh), yd = xn::scratch_n<float>(T * kLat);
        Q().memcpy(xd.get(), xt.data(), xt.size() * 4);
        Q().memcpy(cd.get(), ct.data(), ct.size() * 4).wait();
        dit_forward(d, xd.as<float>(), cd.as<float>(), timestep_embed(d, static_cast<float>(meta["t"])), T, yd.as<float>());
        std::vector<float> y(T * kLat), got(T * kLat);
        Q().memcpy(y.data(), yd.get(), y.size() * 4).wait();
        for (int64_t t = 0; t < T; ++t) for (int i = 0; i < kLat; ++i) got[i * T + t] = y[t * kLat + i];
        return xn::compare_report("dit", got, ref, 3e-2);
    }
    if (what == "cond") {
        DiT d           = load_dit(dit);
        const int64_t F = static_cast<int64_t>(meta["F"]), Lt = static_cast<int64_t>(meta["L"]);
        auto h = read_raw<float>(dir + "/hidden.f32"), ref = read_raw<float>(dir + "/cond.f32");
        Buf hf = xn::scratch_n<float>(h.size()), hb = xn::scratch_n<bf16>(h.size());
        Q().memcpy(hf.get(), h.data(), h.size() * 4).wait();
        xn::convert(hf.as<float>(), hb.as<bf16>(), static_cast<int64_t>(h.size()));
        Buf c = make_condition(d, hb.as<bf16>(), F, Lt);
        std::vector<float> ct(Lt * kCondCh), got(Lt * kCondCh);
        Q().memcpy(ct.data(), c.get(), ct.size() * 4).wait();
        for (int64_t t = 0; t < Lt; ++t) for (int i = 0; i < kCondCh; ++i) got[i * Lt + t] = ct[t * kCondCh + i];
        return xn::compare_report("cond", got, ref, 1e-2);
    }
    if (what == "dav") {
        DAV d           = load_dav(dav);
        const int64_t T = static_cast<int64_t>(meta["T"]);
        auto z = read_raw<float>(dir + "/z.f32"), ref = read_raw<float>(dir + "/wav.f32");
        auto got = dav_decode(d, z, T);
        return xn::compare_report("dav", got, ref, 1e-3);
    }
    if (what == "ar") {
        AR ar     = load_ar(te);
        auto ids  = read_raw<int32_t>(dir + "/ids.i32");
        auto codes = read_raw<int32_t>(dir + "/codes.i32");
        auto ref_h = read_raw<float>(dir + "/hidden.f32"), ref_l = read_raw<float>(dir + "/logits0.f32");
        std::vector<float> lg;
        ArOptions o;
        o.forced     = &codes;
        o.logits0    = &lg;
        o.store_all  = true;
        o.max_frames = static_cast<int64_t>(codes.size() / kCodebooks);
        int64_t F    = 0;
        Buf h        = ar_generate(ar, ids, o, &F);
        std::vector<float> got(static_cast<size_t>(F * kCodebooks * kHid));
        {
            Buf hf = xn::scratch_n<float>(static_cast<int64_t>(got.size()));
            xn::convert(h.as<bf16>(), hf.as<float>(), static_cast<int64_t>(got.size()));
            Q().memcpy(got.data(), hf.get(), got.size() * 4).wait();
        }
        lg.resize(std::min(lg.size(), ref_l.size()));
        int rc = xn::compare_report("ar logits", lg, ref_l, 5e-2);
        rc |= xn::compare_report("ar hidden", got, ref_h, 5e-2);
        for (int64_t f = 0; f < F; ++f) {
            std::vector<float> a(got.begin() + f * kCodebooks * kHid, got.begin() + (f + 1) * kCodebooks * kHid);
            std::vector<float> b(ref_h.begin() + f * kCodebooks * kHid, ref_h.begin() + (f + 1) * kCodebooks * kHid);
            xn::compare_report(("frame " + std::to_string(f)).c_str(), a, b, 5e-2);
        }
        return rc;
    }
    throw std::runtime_error("selftest: unknown stage " + what);
}

}  // namespace

int main(int argc, char** argv) {
    std::string caption, lyrics, out, test, test_dir;
    std::string te  = kModels + "minimax_music3_text_encoder_pruned_int8_convrot.safetensors";
    std::string dit = kModels + "minimax_music3_dit_int8_convrot.safetensors";
    std::string dav = kModels + "minimax_music3_dav.safetensors";
    double seconds  = 120;
    uint64_t seed   = 0;
    int steps = 30, top_k = 50;
    float cfg = 1.7f, ar_cfg = 1.5f;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next           = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for " + a);
                }
                return argv[++i];
            };
            if (a == "--caption") caption = next();
            else if (a == "--caption-file") caption = read_text(next());
            else if (a == "--lyrics") lyrics = next();
            else if (a == "--lyrics-file") lyrics = read_text(next());
            else if (a == "--out" || a == "-o") out = next();
            else if (a == "--seconds") seconds = std::stod(next());
            else if (a == "--seed") seed = std::stoull(next());
            else if (a == "--steps") steps = std::stoi(next());
            else if (a == "--cfg") cfg = std::stof(next());
            else if (a == "--ar-cfg") ar_cfg = std::stof(next());
            else if (a == "--top-k") top_k = std::stoi(next());
            else if (a == "--te") te = next();
            else if (a == "--dit") dit = next();
            else if (a == "--dav") dav = next();
            else if (a == "--quiet") g_verbose = 0;
            else if (a == "--selftest") { test = next(); test_dir = next(); }
            else throw std::runtime_error("unknown option " + a);
        }
        if (!test.empty()) {
            return selftest(test, test_dir, te, dit, dav);
        }
        if (caption.empty() || out.empty()) {
            std::fprintf(stderr, "usage: xpu_music3 --caption-file F [--lyrics-file F] --out song.wav [--seconds 120] [--seed N] [--steps 30] [--cfg 1.7]\n");
            return 2;
        }
        const double t0 = now_s();
        const int64_t max_frames = std::min<int64_t>(9000, std::max<int64_t>(1, std::llround(seconds * 25.0)));
        int64_t F = 0;
        Buf cond;
        int64_t Tl = 0;
        {
            AR ar     = load_ar(te);
            auto ids  = tokenize(ar, caption, lyrics);
            LOG("prompt: %zu tokens, up to %lld frames (%.1f s)", ids.size(), (long long)max_frames, max_frames / 25.0);
            ArOptions o;
            o.cfg        = ar_cfg;
            o.top_k      = top_k;
            o.seed       = seed;
            o.max_frames = max_frames;
            Buf hidden   = ar_generate(ar, ids, o, &F);
            if (F == 0) {
                throw std::runtime_error("the AR stage produced no audio frames");
            }
            Tl = latent_length(F);
            ar = AR();  // free the LM before the DiT loads
            Q().wait();
            xn::release_all();
            DiT d = load_dit(dit);
            cond  = make_condition(d, hidden.as<bf16>(), F, Tl);
            hidden.reset();
            auto lat = dit_sample(d, cond.as<float>(), Tl, steps, cfg, seed);
            cond.reset();
            d = DiT();
            xn::release_all();
            DAV v    = load_dav(dav);
            auto wav = dav_decode(v, lat, Tl);
            const int64_t n = Tl * 512;
            double sum = 0, sum2 = 0;
            for (float s : wav) {
                sum += s;
            }
            const double mean = sum / wav.size();
            for (float s : wav) {
                sum2 += (s - mean) * (s - mean);
            }
            const double sd  = std::sqrt(sum2 / (wav.size() - 1));
            const double div = std::max(1.0, 5.0 * sd);
            for (float& s : wav) {
                s = static_cast<float>(s / div);
            }
            write_wav(out, wav, n, 44100);
            LOG("wrote %s: %.2f s of stereo 44.1 kHz audio in %.1f s total", out.c_str(), n / 44100.0, now_s() - t0);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_music3] error: %s\n", e.what());
        return 1;
    }
}
