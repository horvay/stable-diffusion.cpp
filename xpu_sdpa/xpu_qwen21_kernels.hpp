// Fused SYCL kernels for the Qwen Image 2.1 XPU blocks (xpu_qwen21.cpp, int8 mode).
//
// Every int8 linear needs its bf16 input rotated by R (256-group normalized Hadamard, kron of h4 / 2 with
// h4[a][b] = a + b == 3 ? -1 : 1) and quantized per row. These kernels fold that into the op that produces
// the input, so each activation is read once and written once as int8 + a per-row f32 scale:
//   ln_mod_rot_quant      x += y * gate (optional); LayerNorm(x) * scale -> R -> int8       (attn / mlp input)
//   qk_norm_rope          RMSNorm(q|k) * w -> RoPE -> bf16 [heads, rows, D]                  (SDPA input)
//   attn_out_rot_quant    SDPA output [heads, L, D] -> token-major -> R -> int8              (attn out input)
//   swiglu_rot_quant      up * silu(gate) -> R -> int8                                        (mlp down input)
// One work-group handles one row; its 16 sub-groups of 16 lanes each own G consecutive 256-wide groups. Lane l
// holds elements j * 16 + l (j = 0..15) of a group, so every load/store instruction of a sub-group touches 16
// consecutive elements; the Hadamard's two high base-4 digits (j) are in-lane and the two low ones (l) are
// sub-group shuffles (the transform is the same h4 on every digit, so digit order does not matter).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>

namespace xq21 {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kSg  = 16;         // lanes per sub-group
constexpr int kSgs = 16;         // sub-groups per work-group
constexpr int kWg  = kSg * kSgs;  // one row per work-group

// One 4-point stage: y_b = sum_a h4[a][b] x_a = (x0 + x1 + x2 + x3) - 2 x_{3-b}.
inline void h4(float& a, float& b, float& c, float& d) {
    const float s = a + b + c + d;
    const float t0 = a, t1 = b, t2 = c, t3 = d;
    a = s - 2 * t3;
    b = s - 2 * t2;
    c = s - 2 * t1;
    d = s - 2 * t0;
}

// Normalized 256-point transform of the group whose element e = j * 16 + lane lives in v[j] of `lane`.
inline void hadamard256(sycl::sub_group sg, float (&v)[16]) {
    for (int hi = 0; hi < 4; ++hi) {  // digit 0 (j % 4)
        h4(v[hi * 4], v[hi * 4 + 1], v[hi * 4 + 2], v[hi * 4 + 3]);
    }
    for (int lo = 0; lo < 4; ++lo) {  // digit 1 (j / 4)
        h4(v[lo], v[4 + lo], v[8 + lo], v[12 + lo]);
    }
    // Cross-lane digits with uniform XOR shuffles (cheap register moves): the lane with digit b reads the
    // partner with digit b ^ m, whose h4 coefficient is -1 iff (b ^ m) + b == 3.
    const int lane = static_cast<int>(sg.get_local_linear_id());
    for (int shift = 0; shift <= 2; shift += 2) {  // lane digit 0, then lane digit 1
        const int b = (lane >> shift) & 3;
        float sign[4];
        for (int m = 1; m < 4; ++m) {
            sign[m] = ((b ^ m) + b == 3) ? -1.f : 1.f;
        }
        const float self = (b + b == 3) ? -1.f : 1.f;  // never -1 (b + b is even)
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float x = v[j];
            float y       = self * x;
            for (int m = 1; m < 4; ++m) {
                y += sign[m] * sycl::permute_group_by_xor(sg, x, m << shift);
            }
            v[j] = y;
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        v[j] *= 1.f / 16;
    }
}

// Rotate the lane's G groups, then quantize the whole row with one scale and store it.
template <int G>
inline void rot_quant_store(sycl::nd_item<1> it, float (&v)[G][16], int8_t* q_row, float* s_out, int64_t first_group) {
    auto sg = it.get_sub_group();
    for (int g = 0; g < G; ++g) {
        hadamard256(sg, v[g]);
    }
    float amax = 0.f;
    for (int g = 0; g < G; ++g) {
        for (int j = 0; j < 16; ++j) {
            amax = sycl::fmax(amax, sycl::fabs(v[g][j]));
        }
    }
    amax              = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    const float inv   = 1.f / scale;
    const int lane    = static_cast<int>(sg.get_local_linear_id());
    for (int g = 0; g < G; ++g) {
        int8_t* dst = q_row + (first_group + g) * 256 + lane;
        for (int j = 0; j < 16; ++j) {
            dst[j * 16] = static_cast<int8_t>(sycl::rint(v[g][j] * inv));
        }
    }
    if (it.get_local_id(0) == 0) {
        *s_out = scale;
    }
}

inline int groups_per_subgroup(int64_t width) {
    if (width % (256 * kSgs) != 0) {
        throw std::runtime_error("xq21: row width must be a multiple of 4096");
    }
    return static_cast<int>(width / (256 * kSgs));
}

#define XQ21_DISPATCH_G(G_RUNTIME, ...)                                  \
    switch (G_RUNTIME) {                                                 \
        case 1: { constexpr int G = 1; __VA_ARGS__; } break;              \
        case 2: { constexpr int G = 2; __VA_ARGS__; } break;              \
        case 3: { constexpr int G = 3; __VA_ARGS__; } break;              \
        case 4: { constexpr int G = 4; __VA_ARGS__; } break;              \
        default: throw std::runtime_error("xq21: unsupported row width"); \
    }

// x [rows, H] f32 (updated in place when y != nullptr: x += y * gate[sel]); then
// h = LayerNorm(x) * scale[sel] -> R -> int8 xq [rows, H], sx [rows]. sel = row < prefix ? 1 : 0.
inline void ln_mod_rot_quant(sycl::queue& q, float* x, const bf16* y, const float* gate, const float* scale,
                             int64_t prefix, int64_t rows, int64_t H, float eps, int8_t* xq, float* sx) {
    const int g_rt = groups_per_subgroup(H);
    XQ21_DISPATCH_G(g_rt, q.parallel_for(sycl::nd_range<1>(rows * kWg, kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        const int64_t row = it.get_group(0);
        auto sg           = it.get_sub_group();
        const int lane    = static_cast<int>(sg.get_local_linear_id());
        const int64_t g0  = static_cast<int64_t>(sg.get_group_linear_id()) * G;
        const int sel     = row < prefix ? 1 : 0;
        float v[G][16];
        float sum = 0.f;
        for (int g = 0; g < G; ++g) {
            const int64_t c0 = (g0 + g) * 256 + lane;
            float* xp        = x + row * H + c0;
            for (int j = 0; j < 16; ++j) {
                float val = xp[j * 16];
                if (y != nullptr) {
                    val += static_cast<float>(y[row * H + c0 + j * 16]) * gate[sel * H + c0 + j * 16];
                    xp[j * 16] = val;
                }
                v[g][j] = val;
                sum += val;
            }
        }
        const float mean = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>()) / H;
        float var        = 0.f;
        for (int g = 0; g < G; ++g) {
            for (int j = 0; j < 16; ++j) {
                const float d = v[g][j] - mean;
                var += d * d;
            }
        }
        const float rstd = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), var, sycl::plus<float>()) / H + eps);
        for (int g = 0; g < G; ++g) {
            const int64_t c0 = (g0 + g) * 256 + lane;
            for (int j = 0; j < 16; ++j) {
                v[g][j] = (v[g][j] - mean) * rstd * scale[sel * H + c0 + j * 16];
            }
        }
        rot_quant_store<G>(it, v, xq + row * H, sx + row, g0);
    }));
}

// qkv [rows, 3 * heads * D] bf16 (q | k | v). Writes q_out, k_out [heads, rows, D] bf16:
// RMSNorm over D with weights w_qk [2][D], then RoPE on adjacent pairs with pe [rows, D/2, 2, 2] f32
// ([[cos, -sin], [sin, cos]]). D must be 128: one sub-group per (row, q|k, head); lane l takes pairs l + 16 p.
inline void qk_norm_rope(sycl::queue& q, const bf16* qkv, const float* w_qk, const float* pe, int64_t rows, int64_t heads,
                         int64_t D, float eps, bf16* q_out, bf16* k_out) {
    if (D != 128) {
        throw std::runtime_error("xq21: qk_norm_rope needs head_dim 128");
    }
    const int64_t inner = heads * D;
    const int64_t n_sg  = rows * 2 * heads;
    q.parallel_for(sycl::nd_range<1>(n_sg * kSg, kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        auto sg            = it.get_sub_group();
        const int64_t id   = it.get_global_id(0) / kSg;
        const int lane     = static_cast<int>(sg.get_local_linear_id());
        const int64_t row  = id / (2 * heads);
        const int64_t rem  = id % (2 * heads);
        const int which    = static_cast<int>(rem / heads);
        const int64_t head = rem % heads;
        const bf16* src    = qkv + row * 3 * inner + which * inner + head * 128;
        float v[8];
        float ss = 0.f;
        for (int p = 0; p < 4; ++p) {
            const int i  = p * 16 + lane;  // pair index
            v[2 * p]     = static_cast<float>(src[2 * i]);
            v[2 * p + 1] = static_cast<float>(src[2 * i + 1]);
            ss += v[2 * p] * v[2 * p] + v[2 * p + 1] * v[2 * p + 1];
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / 128.f + eps);
        bf16* dst       = (which == 0 ? q_out : k_out) + (head * rows + row) * 128;
        const float* w  = w_qk + which * 128;
        for (int p = 0; p < 4; ++p) {
            const int i    = p * 16 + lane;
            const float* f = pe + (row * 64 + i) * 4;  // [[cos, -sin], [sin, cos]]
            const float c = f[0], s = f[2];
            const float a  = v[2 * p] * inv * w[2 * i];
            const float b  = v[2 * p + 1] * inv * w[2 * i + 1];
            dst[2 * i]     = bf16(a * c - b * s);
            dst[2 * i + 1] = bf16(a * s + b * c);
        }
    });
}

// SDPA output o [heads, L, D] bf16 (contiguous) for tokens [row0, row0 + L) -> xq rows, rotated + int8.
inline void attn_out_rot_quant(sycl::queue& q, const bf16* o, int64_t L, int64_t heads, int64_t D, int64_t row0, int8_t* xq,
                               float* sx) {
    const int64_t inner = heads * D;
    const int g_rt      = groups_per_subgroup(inner);
    XQ21_DISPATCH_G(g_rt, q.parallel_for(sycl::nd_range<1>(L * kWg, kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        const int64_t t  = it.get_group(0);
        auto sg          = it.get_sub_group();
        const int lane   = static_cast<int>(sg.get_local_linear_id());
        const int64_t g0 = static_cast<int64_t>(sg.get_group_linear_id()) * G;
        float v[G][16];
        for (int g = 0; g < G; ++g) {
            for (int j = 0; j < 16; ++j) {
                const int64_t f = (g0 + g) * 256 + j * 16 + lane;  // feature = head * D + d
                v[g][j]         = static_cast<float>(o[((f / D) * L + t) * D + f % D]);
            }
        }
        rot_quant_store<G>(it, v, xq + (row0 + t) * inner, sx + row0 + t, g0);
    }));
}

// gu [rows, 2 * I] bf16 (gate | up) -> act = up * silu(gate) -> R -> int8 xq [rows, I], sx [rows].
inline void swiglu_rot_quant(sycl::queue& q, const bf16* gu, int64_t rows, int64_t I, int8_t* xq, float* sx) {
    const int g_rt = groups_per_subgroup(I);
    XQ21_DISPATCH_G(g_rt, q.parallel_for(sycl::nd_range<1>(rows * kWg, kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        const int64_t row = it.get_group(0);
        auto sg           = it.get_sub_group();
        const int lane    = static_cast<int>(sg.get_local_linear_id());
        const int64_t g0  = static_cast<int64_t>(sg.get_group_linear_id()) * G;
        float v[G][16];
        for (int g = 0; g < G; ++g) {
            const bf16* gp = gu + row * 2 * I + (g0 + g) * 256 + lane;
            for (int j = 0; j < 16; ++j) {
                const float z = static_cast<float>(gp[j * 16]);
                v[g][j]       = static_cast<float>(gp[I + j * 16]) * (z / (1.f + sycl::exp(-z)));
            }
        }
        rot_quant_store<G>(it, v, xq + row * I, sx + row, g0);
    }));
}

}  // namespace xq21
