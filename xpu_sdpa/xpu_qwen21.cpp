// Qwen Image 2.1 transformer blocks on an Intel XPU, native SYCL + oneDNN (no libtorch).
//
// sd.cpp keeps the text projection, image input, time embedding / modulation, RoPE table and final layer
// on Vulkan; it hands the joint sequence (prefix tokens + target image tokens) to xpu_qwen21_blocks() once
// per model call, and this library runs all transformer blocks with oneDNN int8 / bf16 matmuls, oneDNN Graph
// fused SDPA (SYCL flash-attention kernel for the causal prefix segments) and SYCL kernels.
//
// Weights come from the ComfyUI int8 "convrot" checkpoint (SD_XPU_QWEN21_WEIGHTS): per-row int8 rotated
// weights W_rot = W R with R the 256-group normalized Hadamard and a per-row f32 scale s. LoRAs
// (SD_XPU_QWEN21_LORA) are merged at load (on the device). sd.cpp skips the Vulkan copy of the block weights.
// Weights are resident USM allocations freed by xpu_qwen21_release(); every per-call buffer comes from the
// bounded pool (xpu_native.hpp) and is handed back to the driver at the end of each call.
//
// Default (int8, W8A8): the checkpoint's int8 weights stay resident as-is (~7 GB; layers touched by a LoRA
// are merged in the rotated domain and requantized per row). A linear rotates its bf16 input by R, quantizes
// each row to int8 with a SYCL kernel, and runs oneDNN's int8 matmul with the per-row activation and
// per-column weight scales fused into the bf16 output.
// SD_XPU_QWEN21_INT8=0: weights are expanded to dense bf16 W = (Q s) R^T (~14 GB) and linears are plain
// bf16 GEMMs. SD_XPU_QWEN21_UNFUSED=1: int8 without the fused norm/rotate/quantize kernels.
// SD_XPU_QWEN21_SDPA=sycl forces the SYCL attention kernel for every segment.
// SD_XPU_QWEN21_SELFTEST=<dir>: xpu_qwen21_init() runs the inputs in <dir> and compares with <dir>/ref_out.f32.
//
// Block math mirrors src/model/diffusion/qwen_image_2_1.hpp QwenImage21TransformerBlock::forward:
// modulation rows are [target (t), prefix (t = 0)] x [norm1 scale, attn gate, norm2 scale, mlp gate].

#include "xpu_native.hpp"

#include "xpu_qwen21_kernels.hpp"

namespace {

using xn::bf16;
using xn::Buf;
using xn::DType;
using xn::Q;

// One linear: bf16 dense w (bf16 mode), or int8 rotated q with per-row scale s (int8 mode).
struct Lin {
    Buf w;     // bf16 [out, in]
    Buf q, s;  // int8 [out, in] (rotated when group > 0), f32 [out]
    int64_t out = 0, in = 0;
    int64_t group = 0;  // Hadamard group size of the rotation, 0 = none
};

struct Block {
    Buf norm_qk;                  // f32 [2, head_dim]: norm_q, norm_k
    Lin qkv, out, gate_up, down;  // qkv = to_q | to_k | to_v, gate_up = gate | up (gate rows first)
};

struct State {
    std::vector<Block> blocks;
    int64_t head_dim = 0;
    bool loaded      = false;
    bool fused       = false;  // int8 with every linear rotated by 256-groups: fused SYCL kernels (xpu_qwen21_kernels.hpp)
};

State& S() {
    static auto* s = new State();  // leaked: no device frees after the SYCL runtime is gone at process exit
    return *s;
}

const bool g_int8 = [] {
    const char* v = std::getenv("SD_XPU_QWEN21_INT8");
    return v == nullptr || std::strcmp(v, "0") != 0;
}();

const bool g_graph_sdpa = [] {
    const char* v = std::getenv("SD_XPU_QWEN21_SDPA");
    return !(v != nullptr && std::strcmp(v, "sycl") == 0);
}();

const int64_t g_max_blocks = xn::env_i64("SD_XPU_MAX_BLOCKS", -1);

// ---- device helpers for loading ----

// q[r, :] = round(w[r, :] / s[r]), s[r] = max(amax(|w[r, :]|) / 127, 1e-12)   (torch: amax.div_(127).clamp_min_, round(w / s))
void requant_rows(const float* w, int64_t rows, int64_t k, int8_t* q, float* s) {
    constexpr int WG = 256;
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows * WG), WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int lid   = static_cast<int>(it.get_local_id(0));
        float amax      = 0.f;
        for (int64_t c = lid; c < k; c += WG) {
            amax = sycl::fmax(amax, sycl::fabs(w[r * k + c]));
        }
        amax           = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());
        const float sc = sycl::fmax(amax / 127.f, 1e-12f);
        for (int64_t c = lid; c < k; c += WG) {
            q[r * k + c] = static_cast<int8_t>(sycl::rint(w[r * k + c] / sc));
        }
        if (lid == 0) {
            s[r] = sc;
        }
    });
}

void scale_inplace(float* x, int64_t n, float a) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) { x[i] = x[i] * a; });
}

// A checkpoint linear as stored, loaded into q / s at row offset `row0` of the destination buffers:
// int8 q [rows, in], f32 s [rows], rotation group (0 = none). Plain float weights are quantized per row here (no rotation).
struct PartInfo {
    std::string prefix;
    bool quant    = false;
    int64_t group = 0;
    int64_t rows = 0, in = 0;
};

PartInfo part_info(xn::SafeTensors& st, const std::string& prefix) {
    PartInfo p;
    p.prefix              = prefix;
    const std::string cfg = st.str(prefix + ".comfy_quant");
    const auto& wi        = st.info(prefix + ".weight");
    if (wi.shape.size() != 2) {
        throw std::runtime_error(prefix + ".weight: expected a 2-D weight");
    }
    p.rows = wi.shape[0];
    p.in   = wi.shape[1];
    if (cfg.empty()) {
        return p;
    }
    if (cfg.find("int8_tensorwise") == std::string::npos) {
        throw std::runtime_error(prefix + ": unsupported comfy_quant " + cfg);
    }
    p.quant = true;
    std::smatch m;
    if (std::regex_search(cfg, std::regex(R"("convrot":\s*true)"))) {
        p.group = 256;
        if (std::regex_search(cfg, m, std::regex(R"("convrot_groupsize":\s*([0-9]+))"))) {
            p.group = std::stoll(m[1]);
        }
    }
    return p;
}

void load_part(xn::SafeTensors& st, const PartInfo& p, int8_t* q, float* s) {
    if (!p.quant) {
        Buf w = xn::load_tensor(st, p.prefix + ".weight", DType::f32, false);
        requant_rows(w.as<float>(), p.rows, p.in, q, s);
        Q().wait();
        return;
    }
    Buf qt = xn::load_tensor(st, p.prefix + ".weight", DType::s8, false);
    Q().memcpy(q, qt.get(), static_cast<size_t>(p.rows * p.in));
    const auto& si = st.info(p.prefix + ".weight_scale");
    if (si.numel() == p.rows) {
        Buf sb = xn::load_tensor(st, p.prefix + ".weight_scale", DType::f32, false);
        Q().memcpy(s, sb.get(), static_cast<size_t>(p.rows) * 4);
        Q().wait();
    } else if (si.numel() == 1) {
        auto s1 = st.host_f32(p.prefix + ".weight_scale");
        std::vector<float> sv(static_cast<size_t>(p.rows), s1[0]);
        Q().memcpy(s, sv.data(), sv.size() * 4).wait();
    } else {
        throw std::runtime_error(p.prefix + ".weight_scale: unexpected size");
    }
}

// LoRA keys: PEFT (lora_A/lora_B, optionally ".default"), or kohya (lora_down/lora_up with "_" paths).
struct LoraFile {
    std::unique_ptr<xn::SafeTensors> st;
    float strength = 1.f;
    int applied    = 0;
};

bool find_lora(const xn::SafeTensors& st, const std::string& module, std::string* a, std::string* b, std::string* alpha) {
    for (const char* prefix : {"diffusion_model.", "transformer.", "model.diffusion_model.", "base_model.model.", ""}) {
        const std::string k = std::string(prefix) + module;
        for (const char* a_suffix : {".lora_A.weight", ".lora_A.default.weight"}) {
            if (st.has(k + a_suffix)) {
                std::string b_suffix(a_suffix);
                b_suffix.replace(b_suffix.find("lora_A"), 6, "lora_B");
                *a     = k + a_suffix;
                *b     = k + b_suffix;
                *alpha = k + ".alpha";
                return true;
            }
        }
    }
    std::string kohya = "lora_unet_" + module;
    std::replace(kohya.begin(), kohya.end(), '.', '_');
    if (st.has(kohya + ".lora_down.weight")) {
        *a     = kohya + ".lora_down.weight";
        *b     = kohya + ".lora_up.weight";
        *alpha = kohya + ".alpha";
        return true;
    }
    return false;
}

std::vector<LoraFile> open_loras(const char* spec) {
    std::vector<LoraFile> out;
    std::stringstream list(spec ? spec : "");
    std::string item;
    while (std::getline(list, item, ',')) {
        if (item.empty()) {
            continue;
        }
        LoraFile lf;
        const size_t at = item.rfind('@');
        if (at != std::string::npos) {
            lf.strength = std::stof(item.substr(at + 1));
            item        = item.substr(0, at);
        }
        lf.st = std::make_unique<xn::SafeTensors>(item);
        out.push_back(std::move(lf));
    }
    return out;
}

struct LoraTarget {
    std::string module;
    int64_t row_begin;
};

bool has_lora(const std::vector<LoraFile>& loras, const std::vector<LoraTarget>& targets) {
    std::string a, b, alpha;
    for (const auto& lf : loras) {
        for (const auto& t : targets) {
            if (find_lora(*lf.st, t.module, &a, &b, &alpha)) {
                return true;
            }
        }
    }
    return false;
}

// w_rot[rows] += (strength * alpha/r * B A) R for every LoRA on the target (w_rot f32 [total, in], rotated domain).
void merge_loras(std::vector<LoraFile>& loras, float* w_rot, int64_t total, int64_t in, int64_t group, const LoraTarget& target) {
    for (auto& lf : loras) {
        std::string a, b, alpha;
        if (!find_lora(*lf.st, target.module, &a, &b, &alpha)) {
            continue;
        }
        std::vector<int64_t> ash, bsh;
        Buf la = xn::load_tensor(*lf.st, a, DType::f32, false, &ash);  // [r, in]
        Buf lb = xn::load_tensor(*lf.st, b, DType::f32, false, &bsh);  // [out_t, r]
        if (ash.size() != 2 || bsh.size() != 2 || ash[1] != in || bsh[1] != ash[0] || target.row_begin + bsh[0] > total) {
            throw std::runtime_error("LoRA " + lf.st->path + ": shape mismatch for " + target.module);
        }
        const int64_t r = ash[0], rows_t = bsh[0];
        float scale     = lf.strength;
        if (lf.st->has(alpha)) {
            scale *= lf.st->host_f32(alpha)[0] / static_cast<float>(r);
        }
        Buf delta = xn::scratch_n<float>(rows_t * in);
        xn::MatmulArgs m;  // delta = B A (f32)
        m.M = rows_t, m.K = r, m.N = in;
        m.xt = DType::f32, m.x = lb.get(), m.wt = DType::f32, m.w = la.get(), m.w_nk = false;
        m.yt = DType::f32, m.y = delta.get();
        xn::matmul(m);
        scale_inplace(delta.as<float>(), rows_t * in, scale);
        if (group > 0) {
            xn::rotate_groups(delta.as<float>(), delta.as<float>(), rows_t, in, group);
        }
        xn::add_inplace(w_rot + target.row_begin * in, delta.as<float>(), rows_t * in);
        Q().wait();
        ++lf.applied;
    }
}

// Fuse checkpoint linears along the output rows, merge LoRAs, and lay the result out for the active mode.
Lin make_linear(xn::SafeTensors& st, const std::vector<std::string>& prefixes, std::vector<LoraFile>& loras, const std::vector<LoraTarget>& targets) {
    std::vector<PartInfo> parts;
    for (const auto& p : prefixes) {
        parts.push_back(part_info(st, p));
    }
    Lin l;
    l.group = parts[0].group;
    l.in    = parts[0].in;
    for (const auto& p : parts) {
        if (p.group != l.group || p.in != l.in) {
            throw std::runtime_error("cannot fuse linears with different rotation groups or input widths");
        }
        l.out += p.rows;
    }
    const bool lora = has_lora(loras, targets);
    if (g_int8 && !lora) {  // the checkpoint's own int8 weights, untouched
        l.q        = xn::resident(static_cast<size_t>(l.out * l.in));
        l.s        = xn::resident(static_cast<size_t>(l.out) * 4);
        int64_t r0 = 0;
        for (const auto& p : parts) {
            load_part(st, p, l.q.as<int8_t>() + r0 * l.in, l.s.as<float>() + r0);
            r0 += p.rows;
        }
        return l;
    }
    // w_rot = q * s (f32), LoRA deltas added in the rotated domain
    Buf w_rot = xn::scratch_n<float>(l.out * l.in);
    {
        Buf q      = xn::scratch(static_cast<size_t>(l.out * l.in));
        Buf s      = xn::scratch_n<float>(l.out);
        int64_t r0 = 0;
        for (const auto& p : parts) {
            load_part(st, p, q.as<int8_t>() + r0 * l.in, s.as<float>() + r0);
            r0 += p.rows;
        }
        xn::dequant_rows(q.as<int8_t>(), s.as<float>(), w_rot.as<float>(), l.out, l.in);
        Q().wait();
    }
    for (const auto& t : targets) {
        merge_loras(loras, w_rot.as<float>(), l.out, l.in, l.group, t);
    }
    if (g_int8) {
        l.q = xn::resident(static_cast<size_t>(l.out * l.in));
        l.s = xn::resident(static_cast<size_t>(l.out) * 4);
        requant_rows(w_rot.as<float>(), l.out, l.in, l.q.as<int8_t>(), l.s.as<float>());
    } else {
        l.w = xn::resident(static_cast<size_t>(l.out * l.in) * 2);
        if (l.group > 0) {
            xn::rotate_groups(w_rot.as<float>(), l.w.as<bf16>(), l.out, l.in, l.group);  // W = W_rot R^T (R symmetric)
        } else {
            xn::convert(w_rot.as<float>(), l.w.as<bf16>(), l.out * l.in);
        }
    }
    Q().wait();
    return l;
}

// ---- linears ----

// y = (xq * sx) (q * s)^T as bf16 [M, out], xq int8 [M, K] already rotated like the weights.
Buf matmul_int8(const int8_t* xq, const float* sx, int64_t M, const Lin& l) {
    Buf y = xn::scratch_n<bf16>(M * l.out);
    xn::MatmulArgs a;
    a.M = M, a.K = l.in, a.N = l.out;
    a.xt = DType::s8, a.x = xq, a.wt = DType::s8, a.w = l.q.get(), a.yt = DType::bf16, a.y = y.get();
    a.x_scales = sx, a.w_scales = l.s.as<float>();
    xn::matmul(a);
    return y;
}

Buf linear(const bf16* x, int64_t M, const Lin& l) {
    if (g_int8) {
        Buf xr;
        if (l.group > 0) {
            xr = xn::scratch_n<bf16>(M * l.in);
            xn::rotate_groups(x, xr.as<bf16>(), M, l.in, l.group);
        }
        Buf xq = xn::scratch(static_cast<size_t>(M * l.in));
        Buf sx = xn::scratch_n<float>(M);
        xn::quantize_rows(xr.empty() ? x : xr.as<bf16>(), M, l.in, xq.as<int8_t>(), sx.as<float>());
        xr.reset();
        return matmul_int8(xq.as<int8_t>(), sx.as<float>(), M, l);
    }
    Buf y = xn::scratch_n<bf16>(M * l.out);
    xn::MatmulArgs a;
    a.M = M, a.K = l.in, a.N = l.out;
    a.xt = DType::bf16, a.x = x, a.wt = DType::bf16, a.w = l.w.get(), a.yt = DType::bf16, a.y = y.get();
    xn::matmul(a);
    return y;
}

void load_weights(const char* path, int verbose) {
    const auto t0 = std::chrono::steady_clock::now();
    xn::SafeTensors st(path);
    auto loras   = open_loras(std::getenv("SD_XPU_QWEN21_LORA"));
    int n_blocks = 0;
    while (st.has("transformer_blocks." + std::to_string(n_blocks) + ".attn.to_q.weight")) {
        ++n_blocks;
    }
    if (n_blocks == 0) {
        throw std::runtime_error("no transformer blocks found");
    }
    auto& blocks = S().blocks;
    blocks.clear();
    blocks.resize(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        const std::string m = "transformer_blocks." + std::to_string(i) + ".";
        Block& b            = blocks[i];
        {
            Buf nq = xn::load_tensor(st, m + "attn.norm_q.weight", DType::f32, false);
            Buf nk = xn::load_tensor(st, m + "attn.norm_k.weight", DType::f32, false);
            if (nq.bytes() != nk.bytes()) {
                throw std::runtime_error("norm_q / norm_k size mismatch");
            }
            S().head_dim = static_cast<int64_t>(nq.bytes() / 4);
            b.norm_qk    = xn::resident(nq.bytes() * 2);
            Q().memcpy(b.norm_qk.get(), nq.get(), nq.bytes());
            Q().memcpy(b.norm_qk.as<char>() + nq.bytes(), nk.get(), nk.bytes());
            Q().wait();
        }
        const int64_t inner = st.info(m + "attn.to_q.weight").shape.at(0);
        b.qkv               = make_linear(st, {m + "attn.to_q", m + "attn.to_k", m + "attn.to_v"}, loras,
                                          {{m + "attn.to_q", 0}, {m + "attn.to_k", inner}, {m + "attn.to_v", 2 * inner}});
        b.out               = make_linear(st, {m + "attn.to_out.0"}, loras, {{m + "attn.to_out.0", 0}});
        const int64_t ffn   = st.info(m + "img_mlp.gate_up.weight").shape.at(0) / 2;
        b.gate_up           = make_linear(st, {m + "img_mlp.gate_up"}, loras,
                                          {{m + "img_mlp.gate_up", 0}, {m + "img_mlp.gate_layer", 0}, {m + "img_mlp.proj", ffn}});
        b.down              = make_linear(st, {m + "img_mlp.out"}, loras, {{m + "img_mlp.out", 0}});
        xn::pool().trim();  // keep the f32 staging from piling up while the weights grow
    }
    for (const auto& lf : loras) {
        if (verbose) {
            std::fprintf(stderr, "[xpu_qwen21] LoRA %s @ %.2f merged into %d linears\n", lf.st->path.c_str(), lf.strength, lf.applied);
        }
        if (lf.applied == 0) {
            throw std::runtime_error("LoRA " + lf.st->path + " matched no block linears");
        }
    }
    bool fused = g_int8 && std::getenv("SD_XPU_QWEN21_UNFUSED") == nullptr;
    for (const auto& b : blocks) {
        for (const Lin* l : {&b.qkv, &b.out, &b.gate_up, &b.down}) {
            fused = fused && l->group == 256;
        }
    }
    S().fused = fused;
    Q().wait();
    xn::pool().trim();
    S().loaded = true;
    if (verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_qwen21] loaded %d blocks (%s) from %s in %.1f s, %.0f MiB resident\n", n_blocks,
                     fused ? "int8 W8A8, fused kernels" : g_int8 ? "int8 W8A8" : "bf16", path, s, xn::pool().resident() / 1048576.0);
    }
}

std::mutex g_load_mutex;  // held for the whole load (and for release), so callers wait for an in-flight prefetch

void ensure_loaded(int verbose) {
    std::lock_guard<std::mutex> lock(g_load_mutex);
    if (S().loaded) {
        return;
    }
    const char* path = std::getenv("SD_XPU_QWEN21_WEIGHTS");
    if (path == nullptr) {
        throw std::runtime_error("SD_XPU_QWEN21_WEIGHTS is not set");
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

// ---- block kernels ----

// h [rows, H] bf16 = LayerNorm(x) * scale[sel] (no bias, biased variance, f32), sel = row < prefix ? 1 : 0
void norm_mod(const float* x, const float* scale, int64_t prefix, int64_t rows, int64_t H, float eps, bf16* h) {
    constexpr int WG = 256;
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows * WG), WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int lid   = static_cast<int>(it.get_local_id(0));
        const float* xr = x + r * H;
        const float* w  = scale + (r < prefix ? H : 0);
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
            h[r * H + c]  = bf16(t * w[c]);
        }
    });
}

// x += float(y) * gate[sel] (addcmul_), sel = row < prefix ? 1 : 0
void gated_add(float* x, const bf16* y, const float* gate, int64_t prefix, int64_t rows, int64_t H) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(rows * H)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        const int64_t r = i / H, c = i % H;
        const float t   = static_cast<float>(y[i]) * gate[(r < prefix ? H : 0) + c];
        x[i]            = x[i] + t;
    });
}

struct Segment {
    int64_t start, end;
    bool causal;
};

void run_blocks(const float* x_host, int64_t rows, int64_t hidden, const float* mods_host, int64_t prefix_len, const float* pe_host,
                const int64_t* segs, int64_t n_segs, int64_t heads, int64_t head_dim, float* out_host, int verbose) {
    using clock   = std::chrono::steady_clock;
    const auto t0 = clock::now();
    ensure_loaded(verbose);
    auto& blocks        = S().blocks;
    const float eps     = 1e-6f;
    const int64_t inner = heads * head_dim;
    const int64_t P     = prefix_len;
    if (head_dim != S().head_dim || blocks.at(0).qkv.out != 3 * inner || blocks[0].qkv.in != hidden) {
        throw std::runtime_error("input shapes do not match the loaded weights");
    }
    std::vector<Segment> segments(static_cast<size_t>(n_segs));
    for (int64_t i = 0; i < n_segs; ++i) {
        segments[i] = {segs[3 * i], segs[3 * i + 1], segs[3 * i + 2] != 0};
        if (segments[i].start < 0 || segments[i].end > rows || segments[i].start >= segments[i].end) {
            throw std::runtime_error("bad attention segment");
        }
    }
    const float attn_scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    static const bool timing = std::getenv("SD_XPU_QWEN21_TIMING") != nullptr;
    auto mark                = [&](const char* what, int i) {
        if (timing) {
            Q().wait();
            std::fprintf(stderr, "[xpu_qwen21] block %d %s: %.3f s\n", i, what, std::chrono::duration<double>(clock::now() - t0).count());
        }
    };
    const size_t n_run = g_max_blocks >= 0 ? std::min<size_t>(blocks.size(), static_cast<size_t>(g_max_blocks)) : blocks.size();
    {
        Buf x = xn::scratch_n<float>(rows * hidden);
        xn::h2d(x.get(), x_host, static_cast<size_t>(rows * hidden) * 4);
        // modulation [2, 4, hidden] -> scale1 = m0 + 1, gate1 = tanh(m1), scale2 = m2 + 1, gate2 = tanh(m3), each [2, hidden]
        Buf s1 = xn::scratch_n<float>(2 * hidden), g1 = xn::scratch_n<float>(2 * hidden);
        Buf s2 = xn::scratch_n<float>(2 * hidden), g2 = xn::scratch_n<float>(2 * hidden);
        {
            std::vector<float> hs1(2 * hidden), hg1(2 * hidden), hs2(2 * hidden), hg2(2 * hidden);
            for (int64_t r = 0; r < 2; ++r) {
                const float* m = mods_host + r * 4 * hidden;
                for (int64_t c = 0; c < hidden; ++c) {
                    hs1[r * hidden + c] = m[c] + 1.f;
                    hg1[r * hidden + c] = std::tanh(m[hidden + c]);
                    hs2[r * hidden + c] = m[2 * hidden + c] + 1.f;
                    hg2[r * hidden + c] = std::tanh(m[3 * hidden + c]);
                }
            }
            xn::h2d(s1.get(), hs1.data(), hs1.size() * 4);
            xn::h2d(g1.get(), hg1.data(), hg1.size() * 4);
            xn::h2d(s2.get(), hs2.data(), hs2.size() * 4);
            xn::h2d(g2.get(), hg2.data(), hg2.size() * 4);
        }
        Buf pe = xn::scratch_n<float>(rows * head_dim * 2);  // [rows, D/2, 2, 2]
        xn::h2d(pe.get(), pe_host, static_cast<size_t>(rows * head_dim * 2) * 4);
        Buf qo = xn::scratch_n<bf16>(heads * rows * head_dim), ko = xn::scratch_n<bf16>(heads * rows * head_dim);  // [heads, rows, D]
        auto& tq = Q();

        // attention of every segment: q rows [start, end) attend to keys [0, end); out(o, seg) consumes [H, len, D]
        auto attend = [&](const bf16* qkv, auto&& consume) {
            for (const auto& seg : segments) {
                const int64_t len = seg.end - seg.start;
                Buf o             = xn::scratch_n<bf16>(heads * len * head_dim);
                xn::attention(DType::bf16, qo.as<bf16>() + seg.start * head_dim, rows * head_dim, head_dim, ko.get(), rows * head_dim, head_dim,
                              qkv + 2 * inner, head_dim, 3 * inner, heads, len, seg.end, head_dim, attn_scale, seg.causal ? seg.start : -1, o.get(),
                              g_graph_sdpa, verbose, "xpu_qwen21");
                consume(o.as<bf16>(), seg);
            }
        };
        mark("inputs", -1);
        if (S().fused) {
            // Fused int8 path: each activation is produced already rotated + quantized for the next int8 GEMM.
            const int64_t ffn = blocks[0].gate_up.out / 2;
            Buf xq            = xn::scratch(static_cast<size_t>(rows * hidden));
            Buf xq_f          = xn::scratch(static_cast<size_t>(rows * ffn));
            Buf sx            = xn::scratch_n<float>(rows);
            float* xp         = x.as<float>();
            Buf y;  // previous block's MLP output, added (gated) by the next norm
            const float* yg = nullptr;
            for (size_t bi = 0; bi < n_run; ++bi) {
                const Block& blk = blocks[bi];
                xq21::ln_mod_rot_quant(tq, xp, y.empty() ? nullptr : y.as<bf16>(), yg, s1.as<float>(), P, rows, hidden, eps, xq.as<int8_t>(),
                                       sx.as<float>());
                Buf qkv = matmul_int8(xq.as<int8_t>(), sx.as<float>(), rows, blk.qkv);
                xq21::qk_norm_rope(tq, qkv.as<bf16>(), blk.norm_qk.as<float>(), pe.as<float>(), rows, heads, head_dim, eps, qo.as<bf16>(),
                                   ko.as<bf16>());
                mark("qkv+rope", static_cast<int>(bi));
                attend(qkv.as<bf16>(), [&](const bf16* o, const Segment& seg) {
                    xq21::attn_out_rot_quant(tq, o, seg.end - seg.start, heads, head_dim, seg.start, xq.as<int8_t>(), sx.as<float>());
                });
                qkv.reset();
                mark("sdpa", static_cast<int>(bi));
                y = matmul_int8(xq.as<int8_t>(), sx.as<float>(), rows, blk.out);
                xq21::ln_mod_rot_quant(tq, xp, y.as<bf16>(), g1.as<float>(), s2.as<float>(), P, rows, hidden, eps, xq.as<int8_t>(), sx.as<float>());
                Buf gu = matmul_int8(xq.as<int8_t>(), sx.as<float>(), rows, blk.gate_up);
                xq21::swiglu_rot_quant(tq, gu.as<bf16>(), rows, ffn, xq_f.as<int8_t>(), sx.as<float>());
                gu.reset();
                y  = matmul_int8(xq_f.as<int8_t>(), sx.as<float>(), rows, blk.down);
                yg = g2.as<float>();
                mark("done", static_cast<int>(bi));
            }
            if (!y.empty()) {
                gated_add(xp, y.as<bf16>(), g2.as<float>(), P, rows, hidden);
            }
        } else {
            Buf h = xn::scratch_n<bf16>(rows * hidden);
            for (size_t bi = 0; bi < n_run; ++bi) {  // unfused path (bf16, or int8 without fusion)
                const Block& blk = blocks[bi];
                norm_mod(x.as<float>(), s1.as<float>(), P, rows, hidden, eps, h.as<bf16>());
                Buf qkv = linear(h.as<bf16>(), rows, blk.qkv);
                mark("qkv", static_cast<int>(bi));
                xq21::qk_norm_rope(tq, qkv.as<bf16>(), blk.norm_qk.as<float>(), pe.as<float>(), rows, heads, head_dim, eps, qo.as<bf16>(),
                                   ko.as<bf16>());
                mark("qk norm+rope", static_cast<int>(bi));
                Buf a = xn::scratch_n<bf16>(rows * inner);  // token-major attention output
                attend(qkv.as<bf16>(), [&](const bf16* o, const Segment& seg) {
                    const int64_t len = seg.end - seg.start;
                    xn::heads_to_tokens(o, heads, len, head_dim, 0, len, a.as<bf16>() + seg.start * inner);
                });
                qkv.reset();
                mark("sdpa", static_cast<int>(bi));
                Buf y = linear(a.as<bf16>(), rows, blk.out);
                a.reset();
                gated_add(x.as<float>(), y.as<bf16>(), g1.as<float>(), P, rows, hidden);
                y.reset();
                mark("out", static_cast<int>(bi));
                norm_mod(x.as<float>(), s2.as<float>(), P, rows, hidden, eps, h.as<bf16>());
                Buf gu          = linear(h.as<bf16>(), rows, blk.gate_up);
                const int64_t f = blk.gate_up.out / 2;
                Buf act         = xn::scratch_n<bf16>(rows * f);
                xn::silu_mul(gu.as<bf16>(), act.as<bf16>(), rows, f);
                gu.reset();
                mark("gate_up+swiglu", static_cast<int>(bi));
                y = linear(act.as<bf16>(), rows, blk.down);
                gated_add(x.as<float>(), y.as<bf16>(), g2.as<float>(), P, rows, hidden);
                mark("done", static_cast<int>(bi));
            }
        }
        xn::d2h(out_host, x.get(), static_cast<size_t>(rows * hidden) * 4);
        if (xn::pool().verbose()) {
            xn::pool().report("xpu_qwen21");
        }
    }
    // keep only the weights resident between calls so the Vulkan side has room (VAE decode)
    xn::pool().trim();
    xn::pool().reset_peak();
    if (verbose) {
        const double s = std::chrono::duration<double>(clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_qwen21] %zu blocks, rows=%lld (prefix %lld), segments=%lld: %.2f s\n", blocks.size(), (long long)rows, (long long)P,
                     (long long)n_segs, s);
    }
}

int selftest(const std::string& dir, int verbose) {
    auto m               = xn::read_meta(dir);
    const int64_t rows   = static_cast<int64_t>(m.at("rows"));
    const int64_t hidden = static_cast<int64_t>(m.at("hidden"));
    const int64_t prefix = static_cast<int64_t>(m.at("prefix_len"));
    const int64_t nsegs  = static_cast<int64_t>(m.at("n_segs"));
    const int64_t heads  = static_cast<int64_t>(m.at("heads"));
    const int64_t hd     = static_cast<int64_t>(m.at("head_dim"));
    auto x               = xn::read_vec<float>(dir + "/x.f32");
    auto mods            = xn::read_vec<float>(dir + "/mods.f32");
    auto pe              = xn::read_vec<float>(dir + "/pe.f32");
    auto segs            = xn::read_vec<int64_t>(dir + "/segs.i64");
    auto ref             = xn::read_vec<float>(dir + "/ref_out.f32");
    if (static_cast<int64_t>(x.size()) != rows * hidden || static_cast<int64_t>(mods.size()) != 8 * hidden ||
        static_cast<int64_t>(pe.size()) != rows * hd * 2 || static_cast<int64_t>(segs.size()) != 3 * nsegs) {
        throw std::runtime_error("selftest: inconsistent input files in " + dir);
    }
    std::vector<float> out(x.size());
    const auto t0 = std::chrono::steady_clock::now();
    run_blocks(x.data(), rows, hidden, mods.data(), prefix, pe.data(), segs.data(), nsegs, heads, hd, out.data(), verbose);
    const int64_t nb = static_cast<int64_t>(S().blocks.size());
    std::fprintf(stderr, "[xpu_qwen21] selftest: %lld rows (prefix %lld) through %lld block(s), %s, %.2f s\n", (long long)rows, (long long)prefix,
                 (long long)(g_max_blocks >= 0 ? std::min<int64_t>(g_max_blocks, nb) : nb), S().fused ? "int8 fused" : g_int8 ? "int8" : "bf16",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    xn::write_file(dir + "/native_out.f32", out.data(), out.size() * 4);
    return xn::compare_report("xpu_qwen21", out, ref, xn::env_f64("SD_XPU_SELFTEST_TOL", 0.02));
}

}  // namespace

XN_EXPORT int xpu_qwen21_init(int verbose) {
    try {
        ensure_loaded(verbose);
        if (const char* dir = std::getenv("SD_XPU_QWEN21_SELFTEST"); dir != nullptr && *dir != '\0') {
            return selftest(dir, verbose);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_qwen21] init error: %s\n", e.what());
        return 1;
    }
}

// Start loading the block weights in the background (sd.cpp calls this when it creates the model, so the
// load overlaps its own weight loading and text encoding); the first xpu_qwen21_blocks() call waits for it
// on the load mutex, and retries (reporting the error) if the prefetch failed.
XN_EXPORT void xpu_qwen21_prefetch(int verbose) {
    static std::once_flag once;
    std::call_once(once, [verbose] {
        std::thread([verbose] {
            try {
                ensure_loaded(verbose);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[xpu_qwen21] prefetch error: %s\n", e.what());
            }
        }).detach();
    });
}

XN_EXPORT void xpu_qwen21_release() {
    try {
        std::lock_guard<std::mutex> lock(g_load_mutex);
        if (!S().loaded) {
            return;
        }
        S().blocks.clear();
        S().loaded = false;
        xn::release_all();
        if (xn::g_rt_started && xn::pool().verbose()) {
            xn::pool().report("xpu_qwen21 release");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_qwen21] release error: %s\n", e.what());
    }
}

// x_host [rows, hidden]: prefix tokens (rows < prefix_len) then target image tokens.
// mods_host [2, 4 * hidden]: row 0 = target (t), row 1 = prefix (t = 0); chunks = norm1 scale, attn gate,
// norm2 scale, mlp gate. pe_host [rows, head_dim / 2, 2, 2] = [[cos, -sin], [sin, cos]] on adjacent pairs.
// segs [n_segs, 3] = start, end, causal; each segment's queries attend to keys [0, end), causal ones with
// the in-segment future masked.
XN_EXPORT int xpu_qwen21_blocks(const float* x_host,
                                int64_t rows,
                                int64_t hidden,
                                const float* mods_host,
                                int64_t prefix_len,
                                const float* pe_host,
                                const int64_t* segs,
                                int64_t n_segs,
                                int64_t heads,
                                int64_t head_dim,
                                float* out_host,
                                int verbose) {
    try {
        run_blocks(x_host, rows, hidden, mods_host, prefix_len, pe_host, segs, n_segs, heads, head_dim, out_host, verbose);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_qwen21] error: %s\n", e.what());
        try {
            if (xn::g_rt_started) {
                xn::pool().trim();
            }
        } catch (...) {
        }
        return 1;
    }
}
