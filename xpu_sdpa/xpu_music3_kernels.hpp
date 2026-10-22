// SYCL kernels for xpu_music3.cpp (MiniMax Music 3 on an Intel XPU).
//
//   layer_norm / rms_norm       f32 rows -> normalized bf16 rows (inputs of the int8 linears)
//   add_bf16                    f32 residual += bf16
//   glu_value_gate              DiT feed-forward: value * silu(gate), value columns first
//   dit_qkv_rope                DiT qkv rows -> q, k, v [heads, L, 64] with RoPE on the first 32 dims of each head
//   qkv_scatter                 AR / depth-decoder qkv rows -> per-head RMS norm + RoPE -> q [B*NQ, S, D] and the KV cache
//   flash_attn_gqa              causal prefill attention with grouped KV heads (xn::flash_attention_sycl + a head group)
//   decode_attn                 one query token against a long KV cache: split-K partials, then a combine pass
//   snake                       DAV activation x + sin^2(a x) / (a + 1e-9)
#pragma once

#include "xpu_native.hpp"

namespace m3k {

using xn::bf16;
using xn::Q;

constexpr int kWg = 256;

inline float bf16_bits(uint32_t b) { return sycl::bit_cast<float>(b << 16); }

// out[r, :] = bf16((x[r, :] - mean) / sqrt(var + eps) * g + b)  (torch layer_norm: biased variance)
inline void layer_norm(const float* x, const float* g, const float* b, bf16* out, int64_t rows, int64_t H, float eps) {
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows * kWg), kWg), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int lid   = static_cast<int>(it.get_local_id(0));
        const float* xr = x + r * H;
        float s         = 0.f;
        for (int64_t c = lid; c < H; c += kWg) {
            s += xr[c];
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / static_cast<float>(H);
        float v          = 0.f;
        for (int64_t c = lid; c < H; c += kWg) {
            const float d = xr[c] - mean;
            v += d * d;
        }
        const float rstd = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) / static_cast<float>(H) + eps);
        for (int64_t c = lid; c < H; c += kWg) {
            out[r * H + c] = static_cast<bf16>((xr[c] - mean) * rstd * g[c] + b[c]);
        }
    });
}

// out[r, :] = bf16(x[r * x_ld + :] / sqrt(mean(x^2) + eps) * w)
inline void rms_norm(const float* x, int64_t x_ld, const float* w, bf16* out, int64_t rows, int64_t H, float eps) {
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows * kWg), kWg), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int lid   = static_cast<int>(it.get_local_id(0));
        const float* xr = x + r * x_ld;
        float s         = 0.f;
        for (int64_t c = lid; c < H; c += kWg) {
            s += xr[c] * xr[c];
        }
        const float rs = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / static_cast<float>(H) + eps);
        for (int64_t c = lid; c < H; c += kWg) {
            out[r * H + c] = static_cast<bf16>(xr[c] * rs * w[c]);
        }
    });
}

inline void add_bf16(float* x, const bf16* y, int64_t n) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) { x[i] += static_cast<float>(y[i]); });
}

// vg [n, 2f] (value | gate) -> out [n, f] = value * silu(gate)
inline void glu_value_gate(const bf16* vg, bf16* out, int64_t n, int64_t f) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n * f)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        const int64_t r = i / f, c = i % f;
        const float g   = static_cast<float>(vg[r * 2 * f + f + c]);
        out[i]          = static_cast<bf16>(static_cast<float>(vg[r * 2 * f + c]) * (g / (1.f + sycl::exp(-g))));
    });
}

// DiT: qkv [L, 3 * H * 64] -> q, k, v [H, L, 64]; q and k get RoPE on dims [0, 32) of each head (rotate-half pairs
// d, d + 16), angle = pos * inv_freq[d % 16] with inv_freq the checkpoint's 16 frequencies (f32).
inline void dit_qkv_rope(const bf16* qkv, const float* inv_freq, int64_t L, int64_t H, bf16* q, bf16* k, bf16* v) {
    constexpr int D = 64, R = 32, HALF = 16;
    Q().parallel_for(sycl::range<2>(static_cast<size_t>(L * 3 * H), static_cast<size_t>(D)), [=](sycl::id<2> id) {
        const int64_t row = id[0] / (3 * H);
        const int64_t hh  = id[0] % (3 * H);  // 0..3H-1: q heads, k heads, v heads
        const int d       = static_cast<int>(id[1]);
        const int which   = static_cast<int>(hh / H);
        const int64_t h   = hh % H;
        const bf16* src   = qkv + row * 3 * H * D + hh * D;
        float val         = static_cast<float>(src[d]);
        if (which < 2 && d < R) {
            const int f       = d % HALF;
            const float ang   = static_cast<float>(row) * inv_freq[f];
            const float c     = sycl::cos(ang), s = sycl::sin(ang);
            const float other = static_cast<float>(src[d < HALF ? d + HALF : d - HALF]);
            val               = d < HALF ? val * c - other * s : val * c + other * s;
        }
        bf16* dst                 = which == 0 ? q : which == 1 ? k : v;
        dst[(h * L + row) * D + d] = static_cast<bf16>(val);
    });
}

// qkv [B * S, (NQ + 2 NKV) * D] (row b * S + i at position pos0 + i) ->
//   q_out [B * NQ, S, D]; K / V cache: (p, b, g, :) at kc + ((p * B + b) * NKV + g) * D.
// qn / kn (f32 [D]): per-head RMS norm (eps) of q / k before RoPE (null: none). theta > 0: rotate-half RoPE over D.
template <int D>
inline void qkv_scatter(const bf16* qkv, int64_t B, int64_t S, int NQ, int NKV, int64_t pos0, const float* qn, const float* kn, float theta,
                        float eps, bf16* q_out, bf16* kc, bf16* vc) {
    constexpr int EPL = D / 16;
    const int NH      = NQ + 2 * NKV;
    const int64_t n   = B * S * NH;  // one sub-group per (row, head)
    constexpr int SGS = 8;
    const int64_t wgs = (n + SGS - 1) / SGS;
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(wgs * SGS * 16), SGS * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg          = it.get_sub_group();
        const int lane   = static_cast<int>(sg.get_local_linear_id());
        const int64_t id = it.get_group(0) * SGS + sg.get_group_linear_id();
        if (id >= n) {
            return;  // uniform across the sub-group
        }
        const int64_t row = id / NH;
        const int hd      = static_cast<int>(id % NH);
        const int64_t b = row / S, i = row % S, pos = pos0 + i;
        const bf16* src   = qkv + row * NH * D + hd * D;
        float v[EPL];
        for (int j = 0; j < EPL; ++j) {
            v[j] = static_cast<float>(src[j * 16 + lane]);
        }
        const bool is_q = hd < NQ, is_k = hd >= NQ && hd < NQ + NKV;
        const float* nw = is_q ? qn : is_k ? kn : nullptr;
        if (nw != nullptr) {
            float ss = 0.f;
            for (int j = 0; j < EPL; ++j) {
                ss += v[j] * v[j];
            }
            const float r = sycl::rsqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / D + eps);
            for (int j = 0; j < EPL; ++j) {
                v[j] = v[j] * r * nw[j * 16 + lane];
            }
        }
        if ((is_q || is_k) && theta > 0.f) {
            for (int j = 0; j < EPL / 2; ++j) {
                const int f       = j * 16 + lane;  // < D / 2
                const float inv   = 1.f / sycl::pow(theta, static_cast<float>(2 * f) / static_cast<float>(D));
                const float ang   = static_cast<float>(pos) * inv;
                const float c     = sycl::cos(ang), s = sycl::sin(ang);
                const float a     = v[j], bb = v[j + EPL / 2];
                v[j]              = a * c - bb * s;
                v[j + EPL / 2]    = bb * c + a * s;
            }
        }
        bf16* dst;
        if (is_q) {
            dst = q_out + ((b * NQ + hd) * S + i) * D;
        } else if (is_k) {
            dst = kc + ((pos * B + b) * NKV + (hd - NQ)) * D;
        } else {
            dst = vc + ((pos * B + b) * NKV + (hd - NQ - NKV)) * D;
        }
        for (int j = 0; j < EPL; ++j) {
            dst[j * 16 + lane] = static_cast<bf16>(v[j]);
        }
    });
}

// xn::flash_attention_sycl with grouped KV heads: query head hh reads KV head hh / grp.
template <int DPL>
inline void flash_attn_gqa(const bf16* q, int64_t qh, int64_t qs, const bf16* k, int64_t kh, int64_t ks, const bf16* v, int64_t vh, int64_t vs,
                           int64_t H, int64_t grp, int64_t Sq, int64_t Skv, float scale, int64_t causal_q0, bf16* out) {
    constexpr int D  = DPL * 16;
    constexpr int NQ = 8;
    constexpr int KT = 16;
    const int64_t qblocks = (Sq + NQ - 1) / NQ;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qsm(sycl::range<1>(NQ * D), h);
        sycl::local_accessor<float, 1> ksm(sycl::range<1>(KT * (D + 1)), h);
        sycl::local_accessor<float, 1> vsm(sycl::range<1>(KT * D), h);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(H * qblocks * NQ * 16), NQ * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t g  = it.get_group(0);
            const int64_t hh = g / qblocks;
            const int64_t kv = hh / grp;
            const int64_t i0 = (g % qblocks) * NQ;
            auto sg          = it.get_sub_group();
            const int lane   = static_cast<int>(sg.get_local_linear_id());
            const int sgi    = static_cast<int>(sg.get_group_linear_id());
            const int lid    = static_cast<int>(it.get_local_id(0));
            const int64_t i  = i0 + sgi;
            const bool valid = i < Sq;
            for (int e = lid; e < NQ * D; e += NQ * 16) {
                const int64_t r = i0 + e / D;
                qsm[e]          = r < Sq ? static_cast<float>(q[hh * qh + r * qs + e % D]) : 0.f;
            }
            int64_t kend = Skv;
            if (causal_q0 >= 0) {
                kend = sycl::min(Skv, causal_q0 + sycl::min(Sq, i0 + NQ));
            }
            const int64_t my_lim = causal_q0 >= 0 ? causal_q0 + i : Skv - 1;
            float m = -INFINITY, l = 0.f;
            float acc[DPL];
            for (int d = 0; d < DPL; ++d) {
                acc[d] = 0.f;
            }
            for (int64_t j0 = 0; j0 < kend; j0 += KT) {
                sycl::group_barrier(it.get_group());
                for (int e = lid; e < KT * D; e += NQ * 16) {
                    const int64_t j                 = j0 + e / D;
                    const bool in                   = j < Skv;
                    ksm[(e / D) * (D + 1) + e % D] = in ? static_cast<float>(k[kv * kh + j * ks + e % D]) : 0.f;
                    vsm[e]                          = in ? static_cast<float>(v[kv * vh + j * vs + e % D]) : 0.f;
                }
                sycl::group_barrier(it.get_group());
                const int64_t j = j0 + lane;
                float s         = 0.f;
                for (int d = 0; d < D; ++d) {
                    s += qsm[sgi * D + d] * ksm[lane * (D + 1) + d];
                }
                s                = s * scale;
                const bool vis   = valid && j < Skv && j <= my_lim;
                s                = vis ? s : -INFINITY;
                const float mt   = sycl::reduce_over_group(sg, s, sycl::maximum<float>());
                const float mnew = sycl::fmax(m, mt);
                if (mnew == -INFINITY) {
                    continue;
                }
                const float p     = vis ? sycl::exp(s - mnew) : 0.f;
                const float alpha = m == -INFINITY ? 0.f : sycl::exp(m - mnew);
                l                 = l * alpha + sycl::reduce_over_group(sg, p, sycl::plus<float>());
                for (int d = 0; d < DPL; ++d) {
                    acc[d] *= alpha;
                }
                for (int jj = 0; jj < KT; ++jj) {
                    const float pj = sycl::select_from_group(sg, p, jj);
                    for (int d = 0; d < DPL; ++d) {
                        acc[d] += pj * vsm[jj * D + d * 16 + lane];
                    }
                }
                m = mnew;
            }
            if (valid) {
                const float inv = l > 0.f ? 1.f / l : 0.f;
                bf16* o         = out + (hh * Sq + i) * D;
                for (int d = 0; d < DPL; ++d) {
                    o[d * 16 + lane] = static_cast<bf16>(acc[d] * inv);
                }
            }
        });
    });
}

// One query token per head against `len` cached keys (decode step), D = 128, GRP query heads per KV head.
// q [NH, D] dense; key j of KV head kvh at kc + j * ks + kvh * D (likewise vc); out [NH, D].
// Pass 1: one work-group per (KV head, chunk of CH keys) writes (max, sum, unnormalized acc[D]) per query head;
// pass 2 merges the chunks.
template <int GRP>
inline void decode_attn(const bf16* q, const bf16* kc, const bf16* vc, int64_t ks, int64_t NH, int64_t len, float scale, bf16* out) {
    constexpr int D = 128, CH = 256, WG = 256, NSG = WG / 16;
    static_assert(GRP * 64 == WG, "PV pass maps 64 threads (2 dims each) per query head");
    const int64_t NKV    = NH / GRP;
    const int64_t nsplit = (len + CH - 1) / CH;
    xn::Buf part         = xn::scratch_n<float>(NH * nsplit * (D + 2));
    float* pp            = part.as<float>();
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qsm(sycl::range<1>(GRP * D), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>(GRP * CH), h);
        sycl::local_accessor<float, 1> stat(sycl::range<1>(2 * GRP), h);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(NKV * nsplit * WG), WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t g   = it.get_group(0);
            const int64_t kvh = g / nsplit, sp = g % nsplit;
            const int64_t j0  = sp * CH;
            const int jn      = static_cast<int>(sycl::min<int64_t>(CH, len - j0));
            const int lid     = static_cast<int>(it.get_local_id(0));
            auto sg           = it.get_sub_group();
            const int lane    = static_cast<int>(sg.get_local_linear_id());
            const int sgi     = static_cast<int>(sg.get_group_linear_id());
            for (int e = lid; e < GRP * D; e += WG) {
                qsm[e] = static_cast<float>(q[(kvh * GRP) * D + e]);
            }
            sycl::group_barrier(it.get_group());
            // scores: each sub-group takes keys sgi, sgi + 16, ...; lane holds dims [8 lane, 8 lane + 8)
            for (int jj = sgi; jj < jn; jj += NSG) {
                const sycl::uint4 raw = *reinterpret_cast<const sycl::uint4*>(kc + (j0 + jj) * ks + kvh * D + lane * 8);
                float kv[8]           = {bf16_bits(raw.x() & 0xffffu), bf16_bits(raw.x() >> 16), bf16_bits(raw.y() & 0xffffu), bf16_bits(raw.y() >> 16),
                                         bf16_bits(raw.z() & 0xffffu), bf16_bits(raw.z() >> 16), bf16_bits(raw.w() & 0xffffu), bf16_bits(raw.w() >> 16)};
                for (int hq = 0; hq < GRP; ++hq) {
                    float s = 0.f;
                    for (int t = 0; t < 8; ++t) {
                        s += qsm[hq * D + lane * 8 + t] * kv[t];
                    }
                    s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                    if (lane == 0) {
                        sc[hq * CH + jj] = s * scale;
                    }
                }
            }
            sycl::group_barrier(it.get_group());
            // softmax statistics: sub-group hq handles query head hq
            if (sgi < GRP) {
                float m = -INFINITY;
                for (int jj = lane; jj < jn; jj += 16) {
                    m = sycl::fmax(m, sc[sgi * CH + jj]);
                }
                m       = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
                float l = 0.f;
                for (int jj = lane; jj < jn; jj += 16) {
                    const float p    = sycl::exp(sc[sgi * CH + jj] - m);
                    sc[sgi * CH + jj] = p;
                    l += p;
                }
                l = sycl::reduce_over_group(sg, l, sycl::plus<float>());
                if (lane == 0) {
                    stat[2 * sgi]     = m;
                    stat[2 * sgi + 1] = l;
                }
            }
            sycl::group_barrier(it.get_group());
            // PV: 64 threads per query head, 2 dims each
            const int hq   = lid / 64;
            const int d0   = (lid % 64) * 2;
            float a0 = 0.f, a1 = 0.f;
            const bf16* vb = vc + j0 * ks + kvh * D + d0;
            for (int jj = 0; jj < jn; ++jj) {
                const uint32_t raw = *reinterpret_cast<const uint32_t*>(vb + jj * ks);
                const float p      = sc[hq * CH + jj];
                a0 += p * bf16_bits(raw & 0xffffu);
                a1 += p * bf16_bits(raw >> 16);
            }
            float* dst = pp + ((kvh * GRP + hq) * nsplit + sp) * (D + 2);
            dst[d0]     = a0;
            dst[d0 + 1] = a1;
            if (d0 == 0) {
                dst[D]     = stat[2 * hq];
                dst[D + 1] = stat[2 * hq + 1];
            }
        });
    });
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(NH * D), D), [=](sycl::nd_item<1> it) {
        const int64_t hh = it.get_group(0);
        const int d      = static_cast<int>(it.get_local_id(0));
        const float* src = pp + hh * nsplit * (D + 2);
        float M          = -INFINITY;
        for (int64_t s = 0; s < nsplit; ++s) {
            M = sycl::fmax(M, src[s * (D + 2) + D]);
        }
        float L = 0.f, acc = 0.f;
        for (int64_t s = 0; s < nsplit; ++s) {
            const float w = sycl::exp(src[s * (D + 2) + D] - M);
            L += src[s * (D + 2) + D + 1] * w;
            acc += src[s * (D + 2) + d] * w;
        }
        out[hh * D + d] = static_cast<bf16>(acc / L);
    });
}

// x [N, C, W] -> x + sin(a x)^2 / (a + 1e-9), a per channel
inline void snake(const float* x, const float* alpha, float* y, int64_t N, int64_t C, int64_t W) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(N * C * W)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        const float a   = alpha[(i / W) % C];
        const float v   = x[i];
        const float s   = sycl::sin(a * v);
        y[i]            = v + (1.f / (a + 1e-9f)) * (s * s);
    });
}

}  // namespace m3k
