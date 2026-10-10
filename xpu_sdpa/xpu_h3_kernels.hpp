// Fused SYCL kernels for the MiniMax-H3 XPU blocks in int8 W8A8 mode (xpu_h3.cpp, SD_XPU_H3_INT8=1).
//
// Same idea as xpu_qwen21_kernels.hpp (whose Hadamard / rotate+quantize helpers these reuse): each int8 linear's
// input is produced already rotated by R (256-group normalized Hadamard) and quantized per row, so the
// activation is read once and written once as int8 + a per-row f32 scale:
//   rms_mod_rot_quant   RMSNorm(x) * w, then * (1 + scale[seg]) + shift[seg]  -> R -> int8   (qkv / fc1 input)
//   attn_out_rot_quant  SDPA output [heads, rows, D] -> token-major -> R -> int8               (out_proj input)
//   swiglu_rot_quant    silu(gate) * up -> R -> int8                                           (fc2 input)
// H3 row widths are not multiples of 4096 (hidden 5376 = 21 groups of 256, attention 7168 = 28, MLP 14336 = 56),
// so a work-group has ng / G sub-groups of 16 lanes (one row per work-group) and each sub-group owns G
// consecutive 256-wide groups; G is the smallest of 1..4 that keeps the work-group within 1024 work-items.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>

#include "xpu_qwen21_kernels.hpp"

namespace xh3 {

using bf16           = sycl::ext::oneapi::bfloat16;
constexpr int kSg    = xq21::kSg;  // 16 lanes per sub-group
constexpr int kMaxWg = 1024;

struct Shape {
    int G;     // 256-wide groups per sub-group
    int n_sg;  // sub-groups per work-group
};

inline Shape row_shape(int64_t width) {
    if (width % 256 != 0) {
        throw std::runtime_error("xh3: row width must be a multiple of 256");
    }
    const int64_t ng = width / 256;
    for (int G = 1; G <= 4; ++G) {
        if (ng % G == 0 && (ng / G) * kSg <= kMaxWg) {
            return {G, static_cast<int>(ng / G)};
        }
    }
    throw std::runtime_error("xh3: unsupported row width");
}

#define XH3_DISPATCH_G(G_RUNTIME, ...)                                   \
    switch (G_RUNTIME) {                                                 \
        case 1: { constexpr int G = 1; __VA_ARGS__; } break;              \
        case 2: { constexpr int G = 2; __VA_ARGS__; } break;              \
        case 3: { constexpr int G = 3; __VA_ARGS__; } break;              \
        case 4: { constexpr int G = 4; __VA_ARGS__; } break;              \
        default: throw std::runtime_error("xh3: unsupported row width"); \
    }

// x [n, H] f32 (rows of the residual stream). mod_row [n] int32: modulation row of each token (-1 = none).
// mods [*, 6, H] f32: per modulation row, 6 vectors (shift1, scale1, gate1, shift2, scale2, gate2).
// h = RMSNorm(x) * w; with mod_row >= 0: h = h * (1 + mods[r][scale_i]) + mods[r][shift_i]; then R -> int8 xq, sx.
inline void rms_mod_rot_quant(sycl::queue& q, const float* x, const float* w, const float* mods, const int32_t* mod_row,
                              int shift_i, int scale_i, int64_t n, int64_t H, float eps, int8_t* xq, float* sx) {
    const Shape sh = row_shape(H);
    const int wg   = sh.n_sg * kSg;
    XH3_DISPATCH_G(sh.G, q.parallel_for(sycl::nd_range<1>(n * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        const int64_t row = it.get_group(0);
        auto sg           = it.get_sub_group();
        const int lane    = static_cast<int>(sg.get_local_linear_id());
        const int64_t g0  = static_cast<int64_t>(sg.get_group_linear_id()) * G;
        const float* xr   = x + row * H;
        float v[G][16];
        float ss = 0.f;
        for (int g = 0; g < G; ++g) {
            const int64_t c0 = (g0 + g) * 256 + lane;
            for (int j = 0; j < 16; ++j) {
                const float val = xr[c0 + j * 16];
                v[g][j]         = val;
                ss += val * val;
            }
        }
        const float rstd = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / H + eps);
        const int32_t mr = mod_row[row];
        const float* sc  = mr >= 0 ? mods + (static_cast<int64_t>(mr) * 6 + scale_i) * H : nullptr;
        const float* sh_ = mr >= 0 ? mods + (static_cast<int64_t>(mr) * 6 + shift_i) * H : nullptr;
        for (int g = 0; g < G; ++g) {
            const int64_t c0 = (g0 + g) * 256 + lane;
            for (int j = 0; j < 16; ++j) {
                const int64_t c = c0 + j * 16;
                float h         = v[g][j] * rstd * w[c];
                if (mr >= 0) {
                    h = h * (1.f + sc[c]) + sh_[c];
                }
                v[g][j] = h;
            }
        }
        xq21::rot_quant_store<G>(it, v, xq + row * H, sx + row, g0);
    }));
}

// o [heads, L, D] bf16 (contiguous SDPA output over all L tokens) -> rows [t0, t0 + n) token-major, rotated, int8.
inline void attn_out_rot_quant(sycl::queue& q, const bf16* o, int64_t L, int64_t heads, int64_t D, int64_t t0, int64_t n,
                               int8_t* xq, float* sx) {
    const int64_t inner = heads * D;
    const Shape sh      = row_shape(inner);
    const int wg        = sh.n_sg * kSg;
    XH3_DISPATCH_G(sh.G, q.parallel_for(sycl::nd_range<1>(n * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        const int64_t r  = it.get_group(0);
        const int64_t t  = t0 + r;
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
        xq21::rot_quant_store<G>(it, v, xq + r * inner, sx + r, g0);
    }));
}

// gu [n, 2 * I] bf16 (gate | up) -> silu(gate) * up -> R -> int8 xq [n, I], sx [n].
inline void swiglu_rot_quant(sycl::queue& q, const bf16* gu, int64_t n, int64_t I, int8_t* xq, float* sx) {
    const Shape sh = row_shape(I);
    const int wg   = sh.n_sg * kSg;
    XH3_DISPATCH_G(sh.G, q.parallel_for(sycl::nd_range<1>(n * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        const int64_t row = it.get_group(0);
        auto sg           = it.get_sub_group();
        const int lane    = static_cast<int>(sg.get_local_linear_id());
        const int64_t g0  = static_cast<int64_t>(sg.get_group_linear_id()) * G;
        float v[G][16];
        for (int g = 0; g < G; ++g) {
            const bf16* gp = gu + row * 2 * I + (g0 + g) * 256 + lane;
            for (int j = 0; j < 16; ++j) {
                const float z = static_cast<float>(gp[j * 16]);
                v[g][j]       = (z / (1.f + sycl::exp(-z))) * static_cast<float>(gp[I + j * 16]);
            }
        }
        xq21::rot_quant_store<G>(it, v, xq + row * I, sx + row, g0);
    }));
}

}  // namespace xh3
