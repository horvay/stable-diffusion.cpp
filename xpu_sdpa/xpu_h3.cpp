// MiniMax-H3 transformer blocks on an Intel XPU, native SYCL + oneDNN (no libtorch).
//
// sd.cpp keeps the embeddings, token refiner, time embedding, RoPE table and final layer; it hands the
// packed hidden state to xpu_h3_blocks() once per denoising step, and this library runs all
// transformer blocks with oneDNN bf16 / int8 matmuls, oneDNN Graph fused SDPA and SYCL kernels.
//
// The int8 block weights (~19.5 GB) are resident on the device (direct USM allocations, freed by
// xpu_h3_release()). Every per-step buffer comes from the bounded pool in xpu_native.hpp and is handed back to
// the driver at the end of each xpu_h3_blocks() call, so nothing but the weights stays allocated between steps,
// and xpu_h3_release() leaves the device with no memory held by this library.
//
// Weights come straight from the ComfyUI int8 "convrot" checkpoint (SD_XPU_H3_WEIGHTS): per-row int8
// rotated weights W_rot = W R with R the 256-group normalized Hadamard. Linears compute
// y = (x R) (Q s)^T with full-precision activations; Q*s is expanded to bf16 one layer at a time.
//
// SD_XPU_H3_INT8=1 (default off; changes outputs slightly): W8A8 like xpu_qwen21. The checkpoint's int8 weights
// are used exactly as stored. Each linear's input is rotated by R, quantized per row to int8 and multiplied with
// oneDNN's s8 x s8 matmul (scales fused, bf16 out). LoRAs (SD_XPU_H3_LORA) are not merged (requantizing the base
// weights would add ~20x more error than the LoRA's own effect): A R is quantized per row and appended to the
// weight as extra output rows, so the same int8 GEMM also yields x A^T, and y += (x A^T) B^T in bf16. The rotate +
// quantize is fused into the op producing the input (xpu_h3_kernels.hpp); SD_XPU_H3_INT8_UNFUSED=1 uses separate
// rotate + quantize passes instead.
//
// Attention: oneDNN Graph fused SDPA (the kernel torch's XPU SDPA used), compiled once per shape, with a SYCL
// flash-attention kernel as fallback. SD_XPU_H3_SDPA=sycl forces the SYCL kernel (any other value: graph).
// The first graph call per sequence length is checked against the SYCL kernel on the first 256 query rows
// (relative RMS printed; above SD_XPU_H3_SDPA_TOL, default 0.02, the library switches to the SYCL kernel);
// SD_XPU_H3_SDPA_CHECK=0 skips the check.
//
// SD_XPU_H3_TIMING=1 prints a per-call (per denoising step) breakdown: attention, the four GEMM groups (qkv, out,
// fc1, fc2) and everything else, with TFLOPS/TOPS. Each timed region is bracketed by queue waits.
// SD_XPU_H3_TIMING=2 also prints one line per block.
//
// SD_XPU_H3_SELFTEST=<dir>: xpu_h3_init() runs the inputs in <dir> (written by xpu_selftest gen) through the blocks
// and compares with <dir>/ref_out.f32 (written by the torch library), printing RMS / max error (see SELFTEST.md).
//
// Block math mirrors src/model/diffusion/minimax_h3.hpp TransformerBlock::forward and the previous libtorch
// implementation op by op (same dtypes, same per-op bf16 rounding, f32 accumulation).

#include "xpu_native.hpp"

#include "xpu_h3_kernels.hpp"

#include <array>
#include <set>

namespace {

using xn::bf16;
using xn::Buf;
using xn::DType;
using xn::Q;

struct Linear {
    Buf q;  // int8 [rows, in] (rotated); rows = out (+ lora_rows in int8 mode)
    Buf s;  // f32  [rows]
    int64_t out = 0, in = 0, rows = 0;
    int64_t group = 256;
    Buf la, lb;  // bf16 mode LoRA: bf16 [r, in] and [out, r] (strength folded into lb); int8 mode: lb only
    int64_t rank      = 0;
    int64_t lora_rows = 0;  // int8 mode: q / s carry r extra rows = int8(A R) for the LoRA side path
};

struct Block {
    Buf norm1, norm2, q_norm, k_norm;  // f32
    Linear qkv, out, fc1, fc2;
    Buf ada_w, ada_b;  // f32 [6*3*hidden, tdim], [6*3*hidden]
    int64_t ada_n = 0, ada_k = 0;
};

struct State {
    std::vector<Block> blocks;
    bool loaded = false;
    bool fused  = false;
};

State& S() {
    static auto* s = new State();  // leaked: no device frees after the SYCL runtime is gone at process exit
    return *s;
}

const bool g_int8 = [] {
    const char* v = std::getenv("SD_XPU_H3_INT8");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}();

int env_int(const char* name, int def) { return static_cast<int>(xn::env_i64(name, def)); }

// ---- attention backend (SD_XPU_H3_SDPA=graph|sycl) ----

bool g_sdpa_graph = [] {
    const char* v = std::getenv("SD_XPU_H3_SDPA");
    return !(v != nullptr && std::strcmp(v, "sycl") == 0);
}();
const bool g_sdpa_check = env_int("SD_XPU_H3_SDPA_CHECK", 1) != 0;
const double g_sdpa_tol = xn::env_f64("SD_XPU_H3_SDPA_TOL", 0.02);
std::set<int64_t> g_sdpa_checked;

std::vector<float> to_host_f32(const bf16* dev, int64_t n) {
    std::vector<uint16_t> raw(static_cast<size_t>(n));
    Q().memcpy(raw.data(), dev, static_cast<size_t>(n) * 2).wait();
    std::vector<float> out(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        out[i] = xn::bf16_bits_to_f32(raw[i]);
    }
    return out;
}

double rel_rms(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += static_cast<double>(b[i]) * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

// q, k, v [rows, H, D] bf16 token-major -> o [H, rows, D] bf16 dense
void attention(const bf16* q, const bf16* k, const bf16* v, int64_t rows, int64_t H, int64_t D, bf16* o, int verbose) {
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    const int64_t hs = D, ss = H * D;  // head stride, token stride
    if (g_sdpa_graph) {
        if (xn::graph_sdpa().run(DType::bf16, q, hs, ss, k, hs, ss, v, hs, ss, H, rows, rows, D, scale, o, verbose, "xpu_h3")) {
            if (g_sdpa_check && g_sdpa_checked.insert(rows).second) {
                const int64_t n = std::min<int64_t>(256, rows);
                Buf ref         = xn::scratch_n<bf16>(H * n * D);
                Buf got         = xn::scratch_n<bf16>(H * n * D);
                xn::attention_sycl(q, hs, ss, k, hs, ss, v, hs, ss, H, n, rows, D, scale, -1, ref.as<bf16>());
                xn::copy2d(o, rows * D, got.as<bf16>(), H, n * D);  // the same query rows of the graph output
                const double e  = rel_rms(to_host_f32(got.as<bf16>(), H * n * D), to_host_f32(ref.as<bf16>(), H * n * D));
                const bool fail = !(e <= g_sdpa_tol);
                std::fprintf(stderr, "[xpu_h3] graph SDPA check S=%lld (%lld query rows): relative RMS vs SYCL kernel %.2e%s\n", (long long)rows,
                             (long long)n, e, fail ? " -- too large, switching to the SYCL kernel" : "");
                if (fail) {
                    g_sdpa_graph = false;
                    xn::attention_sycl(q, hs, ss, k, hs, ss, v, hs, ss, H, rows, rows, D, scale, -1, o);
                }
            }
            return;
        }
        g_sdpa_graph = false;  // this shape cannot run as one fused kernel: stay on the SYCL kernel
    }
    xn::attention_sycl(q, hs, ss, k, hs, ss, v, hs, ss, H, rows, rows, D, scale, -1, o);
}

// ---- per-step timing breakdown (SD_XPU_H3_TIMING=1|2) ----

const int g_timing = env_int("SD_XPU_H3_TIMING", 0);

enum TimeCat { T_ATTN, T_QKV, T_OUT, T_FC1, T_FC2, T_N };
const char* const kTimeName[T_N] = {"attn", "qkv", "out", "fc1", "fc2"};

struct StepTimes {
    double sec[T_N]   = {};
    double flops[T_N] = {};
};
StepTimes g_times;

template <class F>
auto timed(int cat, double flops, F&& fn) -> decltype(fn()) {
    if (!g_timing) {
        return fn();
    }
    Q().wait();
    const auto t0 = std::chrono::steady_clock::now();
    auto r        = fn();
    Q().wait();
    g_times.sec[cat] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    g_times.flops[cat] += flops;
    return r;
}

// ---- loading ----

// host x [r, k] -> x R per group (f32, in place); same transform as xn::rotate_groups
void rotate_host(std::vector<float>& x, int64_t r, int64_t k, int64_t g) {
    for (int64_t row = 0; row < r; ++row) {
        for (int64_t g0 = 0; g0 < k; g0 += g) {
            float* p = x.data() + row * k + g0;
            for (int64_t stride = 1; stride < g; stride *= 4) {
                for (int64_t t = 0; t < g / 4; ++t) {
                    const int64_t b = (t / stride) * stride * 4 + (t % stride);
                    const float a0 = p[b], a1 = p[b + stride], a2 = p[b + 2 * stride], a3 = p[b + 3 * stride];
                    const float sum   = a0 + a1 + a2 + a3;
                    p[b]              = (sum - 2.f * a3) * 0.5f;
                    p[b + stride]     = (sum - 2.f * a2) * 0.5f;
                    p[b + 2 * stride] = (sum - 2.f * a1) * 0.5f;
                    p[b + 3 * stride] = (sum - 2.f * a0) * 0.5f;
                }
            }
        }
    }
}

Buf upload_bf16(const std::vector<float>& v) {
    std::vector<uint16_t> b(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        b[i] = xn::f32_to_bf16_bits(v[i]);
    }
    Buf d = xn::resident(b.size() * 2);
    xn::h2d(d.get(), b.data(), b.size() * 2);
    return d;
}

Linear load_linear(xn::SafeTensors& st, const std::string& prefix) {
    const std::string cfg = st.str(prefix + ".comfy_quant");
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
    std::vector<int64_t> shape;
    l.q = xn::load_tensor(st, prefix + ".weight", DType::s8, true, &shape);
    if (shape.size() != 2) {
        throw std::runtime_error(prefix + ".weight: expected a 2-D weight");
    }
    l.out = l.rows = shape[0];
    l.in           = shape[1];
    const auto& si = st.info(prefix + ".weight_scale");
    if (si.numel() == l.out) {
        l.s = xn::load_tensor(st, prefix + ".weight_scale", DType::f32);
    } else if (si.numel() == 1) {  // one scale for the whole tensor: broadcast per row
        auto s1 = st.host_f32(prefix + ".weight_scale");
        std::vector<float> s(static_cast<size_t>(l.out), s1[0]);
        l.s = xn::resident(s.size() * 4);
        xn::h2d(l.s.get(), s.data(), s.size() * 4);
    } else {
        throw std::runtime_error(prefix + ".weight_scale: unexpected size");
    }
    return l;
}

// SD_XPU_H3_LORA="a.safetensors[@strength],b.safetensors[@strength]": PEFT/ai-toolkit LoRAs on the block
// linears (diffusion_model.blocks.N.{attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2}.lora_{A,B}.weight).
// Applied as a low-rank side path on the unrotated input; several LoRAs are concatenated along the rank.
struct HostLora {
    std::vector<float> a;  // [r, in]
    std::vector<float> b;  // [out, r]
    int64_t r = 0;
};

void load_loras(std::vector<std::array<HostLora, 4>>& host, int verbose) {
    const char* spec = std::getenv("SD_XPU_H3_LORA");
    if (spec == nullptr || *spec == '\0') {
        return;
    }
    auto& blocks = S().blocks;
    std::stringstream list(spec);
    std::string item;
    while (std::getline(list, item, ',')) {
        float strength  = 1.f;
        const size_t at = item.rfind('@');
        if (at != std::string::npos) {
            strength = std::stof(item.substr(at + 1));
            item     = item.substr(0, at);
        }
        xn::SafeTensors st(item);
        int applied = 0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            const char* names[4]  = {"attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"};
            const Linear* lins[4] = {&blocks[i].qkv, &blocks[i].out, &blocks[i].fc1, &blocks[i].fc2};
            for (int t = 0; t < 4; ++t) {
                std::string key;
                for (const char* prefix : {"diffusion_model.", "transformer.", "model.diffusion_model.", ""}) {
                    std::string k = std::string(prefix) + "blocks." + std::to_string(i) + "." + names[t];
                    if (st.has(k + ".lora_A.weight")) {
                        key = k;
                        break;
                    }
                }
                if (key.empty()) {
                    continue;
                }
                auto a           = st.host_f32(key + ".lora_A.weight");
                auto b           = st.host_f32(key + ".lora_B.weight");
                const int64_t r  = st.info(key + ".lora_A.weight").shape.at(0);
                const int64_t in = lins[t]->in, out = lins[t]->out;
                if (static_cast<int64_t>(a.size()) != r * in || static_cast<int64_t>(b.size()) != out * r) {
                    throw std::runtime_error("LoRA " + item + ": shape mismatch at " + key);
                }
                float scale = strength;
                if (st.has(key + ".alpha")) {
                    scale *= st.host_f32(key + ".alpha")[0] / static_cast<float>(r);
                }
                for (auto& x : b) {
                    x *= scale;
                }
                HostLora& h = host[i][t];
                // concatenate along the rank: a rows appended, b columns appended
                h.a.insert(h.a.end(), a.begin(), a.end());
                std::vector<float> nb(static_cast<size_t>(out * (h.r + r)));
                for (int64_t o = 0; o < out; ++o) {
                    std::copy(h.b.begin() + o * h.r, h.b.begin() + (o + 1) * h.r, nb.begin() + o * (h.r + r));
                    std::copy(b.begin() + o * r, b.begin() + (o + 1) * r, nb.begin() + o * (h.r + r) + h.r);
                }
                h.b = std::move(nb);
                h.r += r;
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

// bf16 mode: LoRA side path as bf16 A [r, in], B [out, r]
void attach_lora_bf16(Linear& l, const HostLora& h) {
    if (h.r == 0) {
        return;
    }
    l.la   = upload_bf16(h.a);
    l.lb   = upload_bf16(h.b);
    l.rank = h.r;
}

// int8 mode: append int8(A R) (per-row scales) to the weight as extra output rows; the base rows stay as stored.
void append_lora_rows_int8(Linear& l, const HostLora& h) {
    if (h.r == 0) {
        return;
    }
    const int64_t r = h.r, k = l.in;
    std::vector<float> ar = h.a;  // f32 [r, in], unrotated input side
    if (l.group > 0) {
        rotate_host(ar, r, k, l.group);
    }
    std::vector<int8_t> qa(static_cast<size_t>(r * k));
    std::vector<float> sa(static_cast<size_t>(r));
    for (int64_t i = 0; i < r; ++i) {
        float amax = 0.f;
        for (int64_t j = 0; j < k; ++j) {
            amax = std::max(amax, std::fabs(ar[i * k + j]));
        }
        sa[i] = std::max(amax / 127.f, 1e-12f);
        for (int64_t j = 0; j < k; ++j) {
            qa[i * k + j] = static_cast<int8_t>(std::nearbyint(ar[i * k + j] / sa[i]));
        }
    }
    Buf q = xn::resident(static_cast<size_t>((l.out + r) * k));
    Buf s = xn::resident(static_cast<size_t>(l.out + r) * 4);
    Q().memcpy(q.get(), l.q.get(), static_cast<size_t>(l.out * k));
    Q().memcpy(s.get(), l.s.get(), static_cast<size_t>(l.out) * 4);
    Q().wait();
    xn::h2d(q.as<int8_t>() + l.out * k, qa.data(), qa.size());
    xn::h2d(s.as<float>() + l.out, sa.data(), sa.size() * 4);
    l.q         = std::move(q);
    l.s         = std::move(s);
    l.lora_rows = r;
    l.rows      = l.out + r;
    l.lb        = upload_bf16(h.b);  // [out, r]
    l.rank      = r;
}

void load_weights(const char* path, int verbose) {
    const auto t0 = std::chrono::steady_clock::now();
    xn::SafeTensors st(path);
    int n_blocks = 0;
    while (st.has("blocks." + std::to_string(n_blocks) + ".norm1.weight")) {
        ++n_blocks;
    }
    if (n_blocks == 0) {
        throw std::runtime_error("no transformer blocks found");
    }
    auto& blocks = S().blocks;
    blocks.clear();
    blocks.resize(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        Block& b            = blocks[i];
        b.norm1             = xn::load_tensor(st, p + "norm1.weight", DType::f32);
        b.norm2             = xn::load_tensor(st, p + "norm2.weight", DType::f32);
        b.q_norm            = xn::load_tensor(st, p + "attn.q_norm.weight", DType::f32);
        b.k_norm            = xn::load_tensor(st, p + "attn.k_norm.weight", DType::f32);
        b.qkv               = load_linear(st, p + "attn.qkv_proj");
        b.out               = load_linear(st, p + "attn.out_proj");
        b.fc1               = load_linear(st, p + "mlp.fc1");
        b.fc2               = load_linear(st, p + "mlp.fc2");
        std::vector<int64_t> shape;
        b.ada_w = xn::load_tensor(st, p + "adaln_proj.linear.weight", DType::f32, true, &shape);
        b.ada_n = shape.at(0);
        b.ada_k = shape.at(1);
        b.ada_b = xn::load_tensor(st, p + "adaln_proj.linear.bias", DType::f32);
    }
    std::vector<std::array<HostLora, 4>> loras(static_cast<size_t>(n_blocks));
    load_loras(loras, verbose);
    for (int i = 0; i < n_blocks; ++i) {
        Linear* lins[4] = {&blocks[i].qkv, &blocks[i].out, &blocks[i].fc1, &blocks[i].fc2};
        for (int t = 0; t < 4; ++t) {
            if (g_int8) {
                append_lora_rows_int8(*lins[t], loras[i][t]);
            } else {
                attach_lora_bf16(*lins[t], loras[i][t]);
            }
        }
    }
    bool fused = g_int8 && std::getenv("SD_XPU_H3_INT8_UNFUSED") == nullptr;
    for (const auto& b : blocks) {
        for (const Linear* l : {&b.qkv, &b.out, &b.fc1, &b.fc2}) {
            fused = fused && l->group == 256;
        }
    }
    S().fused = fused;
    Q().wait();
    xn::pool().trim();  // drop the load-time conversion scratch so Vulkan sees it as free
    S().loaded = true;
    if (verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_h3] loaded %d blocks (%s) from %s in %.1f s, %.0f MiB resident\n", n_blocks,
                     fused ? "int8 W8A8, fused kernels" : g_int8 ? "int8 W8A8" : "int8 weights, bf16 compute", path, s,
                     xn::pool().resident() / 1048576.0);
    }
}

void ensure_loaded(int verbose) {
    if (S().loaded) {
        return;
    }
    const char* path = std::getenv("SD_XPU_H3_WEIGHTS");
    if (path == nullptr) {
        throw std::runtime_error("SD_XPU_H3_WEIGHTS is not set");
    }
    try {
        load_weights(path, verbose);
    } catch (...) {
        S().blocks.clear();
        S().loaded = false;
        xn::release_all();
        throw;
    }
}

// ---- linears ----

int g_cat = T_QKV;  // timing category of the linears being run (SD_XPU_H3_TIMING)

// y = (xq * sx) (q * s)^T as bf16 [M, out]; xq int8 [M, K] rotated like the weights, sx f32 [M]. With LoRA rows,
// y = y_base + y_lora B^T (one rounding, like at::addmm).
Buf matmul_int8_lora(const Buf& xq, const Buf& sx, int64_t M, const Linear& l) {
    const double flops = 2.0 * M * l.in * l.rows;
    return timed(g_cat, flops, [&] {
        Buf y = xn::scratch_n<bf16>(M * l.rows);
        xn::MatmulArgs a;
        a.M = M, a.K = l.in, a.N = l.rows;
        a.xt = DType::s8, a.x = xq.get();
        a.wt = DType::s8, a.w = l.q.get();
        a.yt = DType::bf16, a.y = y.get();
        a.x_scales = sx.as<float>(), a.w_scales = l.s.as<float>();
        xn::matmul(a);
        if (l.lora_rows == 0) {
            return y;
        }
        Buf z = xn::scratch_n<bf16>(M * l.out);
        xn::copy2d(y.as<bf16>(), l.rows, z.as<bf16>(), M, l.out);
        xn::MatmulArgs b;
        b.M = M, b.K = l.lora_rows, b.N = l.out;
        b.xt = DType::bf16, b.x = y.as<bf16>() + l.out, b.x_rs = l.rows;
        b.wt = DType::bf16, b.w = l.lb.get();
        b.yt = DType::bf16, b.y = z.get();
        b.accumulate = true;
        xn::matmul(b);
        return z;
    });
}

// x [n, in] bf16 -> rotated copy (or nothing when the linear has no rotation)
Buf rotate_input(const bf16* x, int64_t n, const Linear& l) {
    Buf xr;
    if (l.group > 0) {
        xr = xn::scratch_n<bf16>(n * l.in);
        xn::rotate_groups(x, xr.as<bf16>(), n, l.in, l.group);
    }
    return xr;
}

// unfused int8 linear: x [n, in] bf16 -> rotate -> quantize rows -> int8 matmul
Buf linear_int8(const bf16* x, int64_t n, const Linear& l) {
    Buf xr          = rotate_input(x, n, l);
    const bf16* src = xr.empty() ? x : xr.as<bf16>();
    Buf xq          = xn::scratch(static_cast<size_t>(n * l.in));
    Buf sx          = xn::scratch_n<float>(n);
    xn::quantize_rows(src, n, l.in, xq.as<int8_t>(), sx.as<float>());
    xr.reset();
    return matmul_int8_lora(xq, sx, n, l);
}

// y = (x R) (Q s)^T for x [n, in] bf16; returns bf16 [n, out]
Buf linear(const bf16* x, int64_t n, const Linear& l) {
    if (g_int8) {
        return linear_int8(x, n, l);
    }
    const double flops = 2.0 * n * l.in * l.out;
    return timed(g_cat, flops, [&] {  // bf16 mode: rotation and weight expansion count as GEMM time
        Buf xr = rotate_input(x, n, l);
        Buf w  = xn::scratch_n<bf16>(l.out * l.in);
        xn::dequant_rows(l.q.as<int8_t>(), l.s.as<float>(), w.as<bf16>(), l.out, l.in);
        Buf y = xn::scratch_n<bf16>(n * l.out);
        xn::MatmulArgs a;
        a.M = n, a.K = l.in, a.N = l.out;
        a.xt = DType::bf16, a.x = xr.empty() ? static_cast<const void*>(x) : xr.get();
        a.wt = DType::bf16, a.w = w.get();
        a.yt = DType::bf16, a.y = y.get();
        xn::matmul(a);
        xr.reset();
        w.reset();
        if (l.rank > 0) {  // y += (x A^T) B^T, each product rounded to bf16 like the torch path
            Buf t = xn::scratch_n<bf16>(n * l.rank);
            xn::MatmulArgs b;
            b.M = n, b.K = l.in, b.N = l.rank;
            b.xt = DType::bf16, b.x = x, b.wt = DType::bf16, b.w = l.la.get(), b.yt = DType::bf16, b.y = t.get();
            xn::matmul(b);
            Buf t2 = xn::scratch_n<bf16>(n * l.out);
            xn::MatmulArgs c;
            c.M = n, c.K = l.rank, c.N = l.out;
            c.xt = DType::bf16, c.x = t.get(), c.wt = DType::bf16, c.w = l.lb.get(), c.yt = DType::bf16, c.y = t2.get();
            xn::matmul(c);
            xn::add_inplace(y.as<bf16>(), t2.as<bf16>(), n * l.out);
        }
        return y;
    });
}

// ---- block kernels ----

struct SegDev {
    int64_t start, end, row;
};

// h [n, H] bf16 = modulate(RMSNorm(x[s + r]) * w): (x * rsqrt(mean(x^2) + eps)) * w, then for every segment covering
// the row (in order): h = h * (1 + mods[row][scale_i]) + mods[row][shift_i]; all f32, rounded to bf16 at the end.
void norm_mod(const float* x, const float* w, const float* mods, const SegDev* segs, int64_t n_segs, int shift_i, int scale_i, int64_t s, int64_t n,
              int64_t H, float eps, bf16* h) {
    constexpr int WG = 256;
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(n * WG), WG), [=](sycl::nd_item<1> it) {
        const int64_t r   = it.get_group(0);
        const int64_t row = s + r;
        const int lid     = static_cast<int>(it.get_local_id(0));
        const float* xr   = x + row * H;
        float ss          = 0.f;
        for (int64_t c = lid; c < H; c += WG) {
            const float v = xr[c];
            ss += v * v;
        }
        const float mean = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / static_cast<float>(H);
        const float rstd = sycl::rsqrt(mean + eps);
        for (int64_t c = lid; c < H; c += WG) {
            const float a = xr[c] * rstd;
            float v       = a * w[c];
            for (int64_t i = 0; i < n_segs; ++i) {
                if (row >= segs[i].start && row < segs[i].end) {
                    const float* m = mods + segs[i].row * 6 * H;
                    const float f  = 1.f + m[scale_i * H + c];
                    const float t  = v * f;
                    v              = t + m[shift_i * H + c];
                }
            }
            h[r * H + c] = bf16(v);
        }
    });
}

// x[s + r] += float(u[r]) * mods[row][gate_i] for every segment covering the row (f32 product, then the add)
void gated_add(float* x, const bf16* u, const float* mods, const SegDev* segs, int64_t n_segs, int gate_i, int64_t s, int64_t n, int64_t H) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n * H)), [=](sycl::id<1> id) {
        const int64_t i   = id[0];
        const int64_t row = s + i / H, c = i % H;
        float acc         = x[row * H + c];
        const float uv    = static_cast<float>(u[i]);
        for (int64_t k = 0; k < n_segs; ++k) {
            if (row >= segs[k].start && row < segs[k].end) {
                const float t = uv * mods[(segs[k].row * 6 + gate_i) * H + c];
                acc           = acc + t;
            }
        }
        x[row * H + c] = acc;
    });
}

// qkv [n, 3 * inner] bf16 (rows s .. s + n): q / k -> RMSNorm over D (f32, weights q_norm / k_norm) -> bf16 ->
// split-half RoPE on the first 2 * half dims with bf16 cos / sin and per-op bf16 rounding (a * c - b * s,
// a * s + b * c) -> q_out / k_out [rows, H, D] at row s. One work-group of D work-items per (row, q|k, head).
void qk_norm_rope(const bf16* qkv, const float* wq, const float* wk, const bf16* cosv, const bf16* sinv, int64_t half, int64_t s, int64_t n,
                  int64_t H, int64_t D, float eps, bf16* q_out, bf16* k_out) {
    const int64_t inner = H * D;
    const int64_t rot   = 2 * half;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(static_cast<size_t>(D)), h);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(n * 2 * H * D), static_cast<size_t>(D)), [=](sycl::nd_item<1> it) {
            const int64_t g    = it.get_group(0);
            const int64_t r    = g / (2 * H);
            const int which    = static_cast<int>((g / H) % 2);
            const int64_t head = g % H;
            const int d        = static_cast<int>(it.get_local_id(0));
            const float xv     = static_cast<float>(qkv[r * 3 * inner + which * inner + head * D + d]);
            const float ss     = sycl::reduce_over_group(it.get_group(), xv * xv, sycl::plus<float>());
            const float rr     = sycl::rsqrt(ss / static_cast<float>(D) + eps);
            const float* w     = which == 0 ? wq : wk;
            const float a0     = xv * rr;
            const bf16 nv      = bf16(a0 * w[d]);
            sh[d]              = static_cast<float>(nv);
            sycl::group_barrier(it.get_group());
            const int64_t row = s + r;
            bf16 o            = nv;
            if (d < half) {
                const float c = static_cast<float>(cosv[row * half + d]), sn = static_cast<float>(sinv[row * half + d]);
                const bf16 t1 = bf16(sh[d] * c);
                const bf16 t2 = bf16(sh[d + half] * sn);
                o             = bf16(static_cast<float>(t1) - static_cast<float>(t2));
            } else if (d < rot) {
                const int64_t j = d - half;
                const float c = static_cast<float>(cosv[row * half + j]), sn = static_cast<float>(sinv[row * half + j]);
                const bf16 t1 = bf16(sh[j] * sn);
                const bf16 t2 = bf16(sh[d] * c);
                o             = bf16(static_cast<float>(t1) + static_cast<float>(t2));
            }
            (which == 0 ? q_out : k_out)[row * inner + head * D + d] = o;
        });
    });
}

const int64_t g_max_blocks = xn::env_i64("SD_XPU_MAX_BLOCKS", -1);

// The whole step. All device buffers are pooled and released (trimmed) at the end.
void run_blocks(const float* x_host, int64_t rows, int64_t hidden, const float* temb_host, int64_t trows, int64_t tdim, const float* pe_host,
                int64_t half_rot, const int64_t* segs, int64_t n_segs, int64_t heads, int64_t head_dim, float* out_host, int verbose) {
    using clock = std::chrono::steady_clock;
    ensure_loaded(verbose);
    auto& blocks = S().blocks;
    g_times      = StepTimes{};
    if (g_timing) {
        Q().wait();
    }
    const auto t0              = clock::now();
    static const int64_t chunk = xn::env_i64("SD_XPU_H3_CHUNK", 16384);
    const float eps            = 1e-5f;
    const int64_t inner        = heads * head_dim;
    const int64_t rot          = half_rot * 2;
    if (rot > head_dim || head_dim > 1024) {
        throw std::runtime_error("bad rope / head_dim");
    }
    const Block& b0 = blocks.at(0);
    if (b0.ada_k != tdim || b0.ada_n != 18 * hidden || b0.qkv.out != 3 * inner || b0.qkv.in != hidden) {
        throw std::runtime_error("input shapes do not match the loaded weights");
    }

    {  // scope: every device buffer of the step is released before the trim below
        Buf x = xn::scratch_n<float>(rows * hidden);
        xn::h2d(x.get(), x_host, static_cast<size_t>(rows * hidden) * 4);
        Buf temb = xn::scratch_n<float>(trows * tdim);
        xn::h2d(temb.get(), temb_host, static_cast<size_t>(trows * tdim) * 4);
        // RoPE: pe[r][i] = [[cos, -sin], [sin, cos]] -> bf16 cos / sin [rows, half]
        Buf cosv = xn::scratch_n<bf16>(rows * half_rot), sinv = xn::scratch_n<bf16>(rows * half_rot);
        {
            std::vector<uint16_t> c(static_cast<size_t>(rows * half_rot)), s(c.size());
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t i = 0; i < half_rot; ++i) {
                    const float* f      = pe_host + (r * half_rot + i) * 4;
                    c[r * half_rot + i] = xn::f32_to_bf16_bits(f[0]);
                    s[r * half_rot + i] = xn::f32_to_bf16_bits(f[2]);
                }
            }
            xn::h2d(cosv.get(), c.data(), c.size() * 2);
            xn::h2d(sinv.get(), s.data(), s.size() * 2);
        }
        std::vector<SegDev> seg_host(static_cast<size_t>(n_segs));
        for (int64_t i = 0; i < n_segs; ++i) {
            seg_host[i] = {segs[3 * i], segs[3 * i + 1], segs[3 * i + 2]};
            if (seg_host[i].row < 0 || seg_host[i].row >= trows * 3) {
                throw std::runtime_error("segment modulation row out of range");
            }
        }
        Buf seg_dev = xn::scratch(std::max<size_t>(1, seg_host.size()) * sizeof(SegDev));
        if (!seg_host.empty()) {
            Q().memcpy(seg_dev.get(), seg_host.data(), seg_host.size() * sizeof(SegDev)).wait();
        }
        const SegDev* sd = seg_dev.as<SegDev>();

        Buf q = xn::scratch_n<bf16>(rows * inner), k = xn::scratch_n<bf16>(rows * inner), v = xn::scratch_n<bf16>(rows * inner);

        // Fused int8: per-token modulation row for the SYCL norm kernel (segments must not overlap; else unfused).
        bool fused = S().fused;
        Buf mod_row;
        if (fused) {
            std::vector<int32_t> rows_mod(static_cast<size_t>(rows), -1);
            for (const auto& seg : seg_host) {
                for (int64_t r = std::max<int64_t>(seg.start, 0); r < std::min(seg.end, rows) && fused; ++r) {
                    fused       = rows_mod[r] == -1;
                    rows_mod[r] = static_cast<int32_t>(seg.row);
                }
            }
            mod_row = xn::scratch_n<int32_t>(rows);
            Q().memcpy(mod_row.get(), rows_mod.data(), rows_mod.size() * 4).wait();
        }
        auto& queue            = Q();
        auto norm_linear_fused = [&](const Buf& w, const Buf& mods, int64_t s, int64_t e, int shift_i, int scale_i, const Linear& l) {
            Buf xq = xn::scratch(static_cast<size_t>((e - s) * hidden));
            Buf sx = xn::scratch_n<float>(e - s);
            xh3::rms_mod_rot_quant(queue, x.as<float>() + s * hidden, w.as<float>(), mods.as<float>(), mod_row.as<int32_t>() + s, shift_i, scale_i,
                                   e - s, hidden, eps, xq.as<int8_t>(), sx.as<float>());
            return matmul_int8_lora(xq, sx, e - s, l);
        };

        const double attn_flops = 4.0 * static_cast<double>(rows) * static_cast<double>(rows) * static_cast<double>(inner);
        const size_t n_run      = g_max_blocks >= 0 ? std::min<size_t>(blocks.size(), static_cast<size_t>(g_max_blocks)) : blocks.size();
        for (size_t bi = 0; bi < n_run; ++bi) {
            const Block& blk       = blocks[bi];
            const StepTimes before = g_times;
            const auto block_start = clock::now();
            Buf mods               = xn::scratch_n<float>(trows * blk.ada_n);  // [trows * 3, 6, hidden]
            {
                xn::MatmulArgs a;
                a.M = trows, a.K = tdim, a.N = blk.ada_n;
                a.xt = DType::f32, a.x = temb.get(), a.wt = DType::f32, a.w = blk.ada_w.get();
                a.yt = DType::f32, a.y = mods.get(), a.bt = DType::f32, a.bias = blk.ada_b.get();
                xn::matmul(a);
            }

            g_cat = T_QKV;
            for (int64_t s = 0; s < rows; s += chunk) {
                const int64_t e = std::min(rows, s + chunk);
                Buf qkv;
                if (fused) {
                    qkv = norm_linear_fused(blk.norm1, mods, s, e, 0, 1, blk.qkv);
                } else {
                    Buf h = xn::scratch_n<bf16>((e - s) * hidden);
                    norm_mod(x.as<float>(), blk.norm1.as<float>(), mods.as<float>(), sd, n_segs, 0, 1, s, e - s, hidden, eps, h.as<bf16>());
                    qkv = linear(h.as<bf16>(), e - s, blk.qkv);
                }
                qk_norm_rope(qkv.as<bf16>(), blk.q_norm.as<float>(), blk.k_norm.as<float>(), cosv.as<bf16>(), sinv.as<bf16>(), half_rot, s, e - s,
                             heads, head_dim, eps, q.as<bf16>(), k.as<bf16>());
                xn::copy2d(qkv.as<bf16>() + 2 * inner, 3 * inner, v.as<bf16>() + s * inner, e - s, inner);
            }
            Buf o = xn::scratch_n<bf16>(rows * inner);  // [heads, rows, D]
            timed(T_ATTN, attn_flops, [&] {
                attention(q.as<bf16>(), k.as<bf16>(), v.as<bf16>(), rows, heads, head_dim, o.as<bf16>(), verbose);
                return 0;
            });
            g_cat = T_OUT;
            for (int64_t s = 0; s < rows; s += chunk) {
                const int64_t e = std::min(rows, s + chunk);
                Buf y;
                if (fused) {
                    Buf xq = xn::scratch(static_cast<size_t>((e - s) * inner));
                    Buf sx = xn::scratch_n<float>(e - s);
                    xh3::attn_out_rot_quant(queue, o.as<bf16>(), rows, heads, head_dim, s, e - s, xq.as<int8_t>(), sx.as<float>());
                    y = matmul_int8_lora(xq, sx, e - s, blk.out);
                } else {
                    Buf a = xn::scratch_n<bf16>((e - s) * inner);
                    xn::heads_to_tokens(o.as<bf16>(), heads, rows, head_dim, s, e - s, a.as<bf16>());
                    y = linear(a.as<bf16>(), e - s, blk.out);
                }
                gated_add(x.as<float>(), y.as<bf16>(), mods.as<float>(), sd, n_segs, 2, s, e - s, hidden);
            }
            o.reset();

            for (int64_t s = 0; s < rows; s += chunk) {
                const int64_t e = std::min(rows, s + chunk);
                Buf y;
                if (fused) {
                    g_cat           = T_FC1;
                    Buf uv          = norm_linear_fused(blk.norm2, mods, s, e, 3, 4, blk.fc1);
                    const int64_t f = blk.fc1.out / 2;
                    Buf xq          = xn::scratch(static_cast<size_t>((e - s) * f));
                    Buf sx          = xn::scratch_n<float>(e - s);
                    xh3::swiglu_rot_quant(queue, uv.as<bf16>(), e - s, f, xq.as<int8_t>(), sx.as<float>());
                    uv.reset();
                    g_cat = T_FC2;
                    y     = matmul_int8_lora(xq, sx, e - s, blk.fc2);
                } else {
                    g_cat = T_FC1;
                    Buf h = xn::scratch_n<bf16>((e - s) * hidden);
                    norm_mod(x.as<float>(), blk.norm2.as<float>(), mods.as<float>(), sd, n_segs, 3, 4, s, e - s, hidden, eps, h.as<bf16>());
                    Buf uv = linear(h.as<bf16>(), e - s, blk.fc1);
                    h.reset();
                    const int64_t f = blk.fc1.out / 2;
                    Buf act         = xn::scratch_n<bf16>((e - s) * f);
                    xn::silu_mul(uv.as<bf16>(), act.as<bf16>(), e - s, f);
                    uv.reset();
                    g_cat = T_FC2;
                    y     = linear(act.as<bf16>(), e - s, blk.fc2);
                }
                gated_add(x.as<float>(), y.as<bf16>(), mods.as<float>(), sd, n_segs, 5, s, e - s, hidden);
            }
            if (g_timing >= 2) {
                queue.wait();
                const double total = std::chrono::duration<double>(clock::now() - block_start).count();
                double accounted   = 0;
                std::string line;
                char buf[64];
                for (int c = 0; c < T_N; ++c) {
                    const double t = g_times.sec[c] - before.sec[c];
                    accounted += t;
                    std::snprintf(buf, sizeof(buf), " %s %.1f", kTimeName[c], t * 1e3);
                    line += buf;
                }
                std::fprintf(stderr, "[xpu_h3] block %2zu: %.1f ms =%s other %.1f (ms)\n", bi, total * 1e3, line.c_str(), (total - accounted) * 1e3);
            }
        }
        xn::d2h(out_host, x.get(), static_cast<size_t>(rows * hidden) * 4);
        if (xn::pool().verbose()) {
            xn::pool().report("xpu_h3");
        }
    }
    // keep only the weights resident between steps so the Vulkan side has room (VAE decode)
    xn::pool().trim();
    xn::pool().reset_peak();
    if (verbose || g_timing) {
        const double s = std::chrono::duration<double>(clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_h3] %zu blocks, rows=%lld, segments=%lld: %.2f s\n", blocks.size(), (long long)rows, (long long)n_segs, s);
        if (g_timing) {
            double accounted = 0;
            std::string line;
            char buf[96];
            for (int c = 0; c < T_N; ++c) {
                const double t = g_times.sec[c];
                accounted += t;
                std::snprintf(buf, sizeof(buf), " | %s %.2f s (%.0f%%, %.0f T%s)", kTimeName[c], t, 100.0 * t / s,
                              t > 0 ? g_times.flops[c] / t / 1e12 : 0.0, c == T_ATTN ? "FLOPS" : (g_int8 ? "OPS" : "FLOPS"));
                line += buf;
            }
            std::fprintf(stderr, "[xpu_h3] timing (%s SDPA)%s | other %.2f s (%.0f%%)\n", g_sdpa_graph ? "graph" : "sycl", line.c_str(), s - accounted,
                         100.0 * (s - accounted) / s);
        }
    }
}

// SD_XPU_H3_SELFTEST=<dir>: run the reference inputs and compare with <dir>/ref_out.f32
int selftest(const std::string& dir, int verbose) {
    auto m               = xn::read_meta(dir);
    const int64_t rows   = static_cast<int64_t>(m.at("rows"));
    const int64_t hidden = static_cast<int64_t>(m.at("hidden"));
    const int64_t trows  = static_cast<int64_t>(m.at("trows"));
    const int64_t tdim   = static_cast<int64_t>(m.at("tdim"));
    const int64_t half   = static_cast<int64_t>(m.at("half_rot"));
    const int64_t nsegs  = static_cast<int64_t>(m.at("n_segs"));
    const int64_t heads  = static_cast<int64_t>(m.at("heads"));
    const int64_t hd     = static_cast<int64_t>(m.at("head_dim"));
    auto x               = xn::read_vec<float>(dir + "/x.f32");
    auto temb            = xn::read_vec<float>(dir + "/temb.f32");
    auto pe              = xn::read_vec<float>(dir + "/pe.f32");
    auto segs            = xn::read_vec<int64_t>(dir + "/segs.i64");
    auto ref             = xn::read_vec<float>(dir + "/ref_out.f32");
    if (static_cast<int64_t>(x.size()) != rows * hidden || static_cast<int64_t>(segs.size()) != 3 * nsegs ||
        static_cast<int64_t>(temb.size()) != trows * tdim || static_cast<int64_t>(pe.size()) != rows * half * 4) {
        throw std::runtime_error("selftest: inconsistent input files in " + dir);
    }
    std::vector<float> out(x.size());
    const auto t0 = std::chrono::steady_clock::now();
    run_blocks(x.data(), rows, hidden, temb.data(), trows, tdim, pe.data(), half, segs.data(), nsegs, heads, hd, out.data(), verbose);
    const int64_t nb = static_cast<int64_t>(S().blocks.size());
    std::fprintf(stderr, "[xpu_h3] selftest: %lld rows through %lld block(s), %s, %.2f s\n", (long long)rows,
                 (long long)(g_max_blocks >= 0 ? std::min<int64_t>(g_max_blocks, nb) : nb), S().fused ? "int8 fused" : g_int8 ? "int8" : "bf16",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    xn::write_file(dir + "/native_out.f32", out.data(), out.size() * 4);
    return xn::compare_report("xpu_h3", out, ref, xn::env_f64("SD_XPU_SELFTEST_TOL", 0.02));
}

}  // namespace

XN_EXPORT int xpu_h3_init(int verbose) {
    try {
        ensure_loaded(verbose);
        if (const char* dir = std::getenv("SD_XPU_H3_SELFTEST"); dir != nullptr && *dir != '\0') {
            return selftest(dir, verbose);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3] init error: %s\n", e.what());
        return 1;
    }
}

XN_EXPORT void xpu_h3_release() {
    try {
        if (!S().loaded) {
            return;
        }
        S().blocks.clear();
        S().loaded = false;
        g_sdpa_checked.clear();
        xn::release_all();
        if (xn::g_rt_started && xn::pool().verbose()) {
            xn::pool().report("xpu_h3 release");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3] release error: %s\n", e.what());
    }
}

XN_EXPORT int xpu_h3_blocks(const float* x_host,
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
        run_blocks(x_host, rows, hidden, temb_host, trows, tdim, pe_host, half_rot, segs, n_segs, heads, head_dim, out_host, verbose);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_h3] error: %s\n", e.what());
        try {
            if (xn::g_rt_started) {
                xn::pool().trim();
            }
        } catch (...) {
        }
        return 1;
    }
}
