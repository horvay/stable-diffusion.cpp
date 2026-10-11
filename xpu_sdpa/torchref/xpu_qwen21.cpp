// Self-test reference build of the original libtorch implementation (torchref/, see ../SELFTEST.md): identical to
// the production source except that SD_XPU_MAX_BLOCKS=N runs only the first N transformer blocks.
// Qwen Image 2.1 transformer blocks on an Intel XPU through libtorch.
//
// sd.cpp keeps the text projection, image input, time embedding / modulation, RoPE table and final layer
// on Vulkan; it hands the joint sequence (prefix tokens + target image tokens) to xpu_qwen21_blocks() once
// per model call, and this library runs all transformer blocks with oneDNN bf16 matmuls and SDPA on the
// Level Zero stack, which is several times faster on Arc Pro B70 than the Vulkan path.
//
// Weights come from the ComfyUI int8 "convrot" checkpoint (SD_XPU_QWEN21_WEIGHTS): per-row int8 rotated
// weights W_rot = W R with R the 256-group normalized Hadamard and a per-row f32 scale s. LoRAs
// (SD_XPU_QWEN21_LORA) are merged at load. sd.cpp skips the Vulkan copy of the block weights.
//
// Default (int8, W8A8): the checkpoint's int8 weights stay resident as-is (~7 GB; layers touched by a LoRA
// are merged in the rotated domain and requantized per row). A linear rotates its bf16 input by R, quantizes
// each row to int8 with a SYCL kernel, and runs oneDNN's int8 matmul (XMX DPAS, ~280 TOPS on Arc Pro B70)
// with the per-row activation and per-column weight scales fused into the bf16 output.
// SD_XPU_QWEN21_INT8=0: weights are expanded to dense bf16 W = (Q s) R^T (~14 GB) and linears are plain
// bf16 GEMMs (~150 TFLOPS). Build with build_icpx.sh (needs DPC++ matching torch's SYCL runtime + oneDNN).
//
// Block math mirrors src/model/diffusion/qwen_image_2_1.hpp QwenImage21TransformerBlock::forward:
// modulation rows are [target (t), prefix (t = 0)] x [norm1 scale, attn gate, norm2 scale, mlp gate].

#include <ATen/ATen.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/xpu/XPUStream.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include "xpu_qwen21_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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

// One linear: bf16 dense w (bf16 mode), or int8 rotated q with per-row scale s (int8 mode).
struct Lin {
    at::Tensor w;       // bf16 [out, in]
    at::Tensor q, s;    // int8 [out, in] (rotated when group > 0), f32 [out]
    int64_t group = 0;  // Hadamard group size of the rotation, 0 = none
};

struct Block {
    at::Tensor norm_qk;           // f32 [2, 1, head_dim]: norm_q, norm_k
    Lin qkv, out, gate_up, down;  // qkv = to_q | to_k | to_v, gate_up = gate | up (gate rows first)
};

std::vector<Block> g_blocks;
bool g_loaded = false;
bool g_fused  = false;  // int8 with every linear rotated by 256-groups: fused SYCL kernels (xpu_qwen21_kernels.hpp)
const bool g_int8 = [] {
    const char* v = std::getenv("SD_XPU_QWEN21_INT8");
    return v == nullptr || std::strcmp(v, "0") != 0;
}();

at::ScalarType to_scalar(const std::string& dtype) {
    if (dtype == "I8") return at::kChar;
    if (dtype == "U8") return at::kByte;
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

    bool has(const std::string& name) const { return index.count(name) != 0; }

    at::Tensor load(const std::string& name, at::ScalarType target) {
        auto it = index.find(name);
        if (it == index.end()) {
            throw std::runtime_error("missing tensor " + name + " in " + path);
        }
        const auto& info = it->second;
        auto host        = at::empty(info.shape, at::TensorOptions().dtype(to_scalar(info.dtype)).pinned_memory(true));
        f.seekg(static_cast<std::streamoff>(base + info.begin));
        f.read(static_cast<char*>(host.data_ptr()), static_cast<std::streamsize>(info.end - info.begin));
        return host.to(kXpu).to(target);
    }

    std::string bytes(const std::string& name) {
        auto it = index.find(name);
        if (it == index.end()) {
            return "";
        }
        std::string s(it->second.end - it->second.begin, '\0');
        f.seekg(static_cast<std::streamoff>(base + it->second.begin));
        f.read(s.data(), static_cast<std::streamsize>(s.size()));
        return s;
    }
};

at::Tensor hadamard(int64_t n) {  // normalized, symmetric: R = R^T = R^-1
    auto h4 = at::tensor({1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1., 1.}, at::kDouble).view({4, 4}) / 2.0;
    auto h  = at::ones({1, 1}, at::kDouble);
    while (h.size(0) < n) {
        h = at::kron(h, h4);
    }
    return h.to(kXpu).to(at::kFloat);
}

const at::Tensor& rotation(int64_t group, at::ScalarType type) {
    static std::map<std::pair<int64_t, int>, at::Tensor> cache;
    auto& r = cache[{group, static_cast<int>(type)}];
    if (!r.defined()) {
        r = hadamard(group).to(type);  // entries are +-2^-k: exact in bf16
    }
    return r;
}

// x [rows, k] -> x R per group of `group` columns (R symmetric, R^-1 = R).
at::Tensor rotate(const at::Tensor& x, int64_t group) {
    const int64_t n = x.size(0), k = x.size(1);
    return at::matmul(x.reshape({n, k / group, group}), rotation(group, x.scalar_type())).view({n, k});
}

// A checkpoint linear as stored: int8 q [out, in], f32 s [out], rotation group (0 = none). Plain float
// weights are quantized per row here (no rotation).
Lin load_quant(SafeTensors& st, const std::string& prefix) {
    Lin l;
    const std::string cfg = st.bytes(prefix + ".comfy_quant");
    if (cfg.empty()) {
        auto w = st.load(prefix + ".weight", at::kFloat);
        l.s    = w.abs().amax(1).div_(127).clamp_min_(1e-12);
        l.q    = at::round(w / l.s.unsqueeze(1)).to(at::kChar);
        return l;
    }
    if (cfg.find("int8_tensorwise") == std::string::npos) {
        throw std::runtime_error(prefix + ": unsupported comfy_quant " + cfg);
    }
    l.q = st.load(prefix + ".weight", at::kChar);
    l.s = st.load(prefix + ".weight_scale", at::kFloat).reshape({-1});
    std::smatch m;
    if (std::regex_search(cfg, std::regex(R"("convrot":\s*true)"))) {
        l.group = 256;
        if (std::regex_search(cfg, m, std::regex(R"("convrot_groupsize":\s*([0-9]+))"))) {
            l.group = std::stoll(m[1]);
        }
    }
    return l;
}

// LoRA keys: PEFT (lora_A/lora_B, optionally ".default"), or kohya (lora_down/lora_up with "_" paths).
struct LoraFile {
    std::unique_ptr<SafeTensors> st;
    float strength = 1.f;
    int applied    = 0;
};

bool find_lora(const SafeTensors& st, const std::string& module, std::string* a, std::string* b, std::string* alpha) {
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
        lf.st = std::make_unique<SafeTensors>(item);
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

// w_rot[rows] += (strength * alpha/r * B A) R for every LoRA on the target (w_rot is in the rotated domain).
void merge_loras(std::vector<LoraFile>& loras, at::Tensor& w_rot, int64_t group, const LoraTarget& target) {
    for (auto& lf : loras) {
        std::string a, b, alpha;
        if (!find_lora(*lf.st, target.module, &a, &b, &alpha)) {
            continue;
        }
        auto la     = lf.st->load(a, at::kFloat);
        auto lb     = lf.st->load(b, at::kFloat);
        float scale = lf.strength;
        if (lf.st->has(alpha)) {
            scale *= lf.st->load(alpha, at::kFloat).item<float>() / static_cast<float>(la.size(0));
        }
        auto delta = at::matmul(lb, la).mul_(scale);
        if (group > 0) {
            delta = rotate(delta, group);
        }
        w_rot.slice(0, target.row_begin, target.row_begin + lb.size(0)).add_(delta);
        ++lf.applied;
    }
}

// Fuse checkpoint linears along the output rows, merge LoRAs, and lay the result out for the active mode.
Lin make_linear(const std::vector<Lin>& parts, std::vector<LoraFile>& loras, const std::vector<LoraTarget>& targets) {
    Lin l;
    l.group = parts[0].group;
    std::vector<at::Tensor> qs, ss;
    for (const auto& p : parts) {
        if (p.group != l.group) {
            throw std::runtime_error("cannot fuse linears with different rotation groups");
        }
        qs.push_back(p.q);
        ss.push_back(p.s);
    }
    l.q = qs.size() == 1 ? qs[0] : at::cat(qs, 0);
    l.s = ss.size() == 1 ? ss[0] : at::cat(ss, 0);
    if (g_int8 && !has_lora(loras, targets)) {
        return l;  // the checkpoint's own int8 weights, untouched
    }
    auto w_rot = l.q.to(at::kFloat).mul_(l.s.unsqueeze(1));
    for (const auto& t : targets) {
        merge_loras(loras, w_rot, l.group, t);
    }
    if (g_int8) {
        l.s = w_rot.abs().amax(1).div_(127).clamp_min_(1e-12);
        l.q = at::round(w_rot.div_(l.s.unsqueeze(1))).to(at::kChar);
    } else {
        l.w = (l.group > 0 ? rotate(w_rot, l.group) : w_rot).to(at::kBFloat16);
        l.q.reset();
        l.s.reset();
    }
    return l;
}

// ---- int8 linear: per-row quantize (SYCL) + oneDNN s8 x s8 -> bf16 matmul with fused scales ----

// xq[m, :] = round(x[m, :] / s[m]), s[m] = amax(|x[m, :]|) / 127; one work-group per row.
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

// y = (xq * sx) (q * s)^T as bf16, xq int8 [M, K] already rotated like the weights.
at::Tensor matmul_int8(const at::Tensor& xq, const at::Tensor& sx, const Lin& l) {
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

at::Tensor linear_int8(const at::Tensor& x, const Lin& l) {
    auto xr = (l.group > 0 ? rotate(x, l.group) : x).contiguous();
    auto xq = at::empty(xr.sizes(), xr.options().dtype(at::kChar));
    auto sx = at::empty({xr.size(0)}, xr.options().dtype(at::kFloat));
    quantize_rows(c10::xpu::getCurrentXPUStream().queue(), xr, xq, sx);
    return matmul_int8(xq, sx, l);
}

at::Tensor linear(const at::Tensor& x, const Lin& l) {
    return g_int8 ? linear_int8(x, l) : at::matmul(x, l.w.t());
}

void load_weights(const char* path, int verbose) {
    const auto t0 = std::chrono::steady_clock::now();
    SafeTensors st(path);
    auto loras   = open_loras(std::getenv("SD_XPU_QWEN21_LORA"));
    int n_blocks = 0;
    while (st.has("transformer_blocks." + std::to_string(n_blocks) + ".attn.to_q.weight")) {
        ++n_blocks;
    }
    if (n_blocks == 0) {
        throw std::runtime_error("no transformer blocks found");
    }
    g_blocks.resize(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        const std::string m = "transformer_blocks." + std::to_string(i) + ".";
        Block& b            = g_blocks[i];
        b.norm_qk           = at::stack({st.load(m + "attn.norm_q.weight", at::kFloat), st.load(m + "attn.norm_k.weight", at::kFloat)})
                        .unsqueeze(1);

        auto q              = load_quant(st, m + "attn.to_q");
        const int64_t inner = q.q.size(0);
        b.qkv               = make_linear({q, load_quant(st, m + "attn.to_k"), load_quant(st, m + "attn.to_v")}, loras,
                                          {{m + "attn.to_q", 0}, {m + "attn.to_k", inner}, {m + "attn.to_v", 2 * inner}});
        b.out               = make_linear({load_quant(st, m + "attn.to_out.0")}, loras, {{m + "attn.to_out.0", 0}});
        auto gu             = load_quant(st, m + "img_mlp.gate_up");
        const int64_t ffn   = gu.q.size(0) / 2;
        b.gate_up           = make_linear({gu}, loras,
                                          {{m + "img_mlp.gate_up", 0}, {m + "img_mlp.gate_layer", 0}, {m + "img_mlp.proj", ffn}});
        b.down              = make_linear({load_quant(st, m + "img_mlp.out")}, loras, {{m + "img_mlp.out", 0}});
        // keep the f32 staging from piling up while the bf16 weights grow
        c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
    }
    for (const auto& lf : loras) {
        if (verbose) {
            std::fprintf(stderr, "[xpu_qwen21] LoRA %s @ %.2f merged into %d linears\n", lf.st->path.c_str(), lf.strength, lf.applied);
        }
        if (lf.applied == 0) {
            throw std::runtime_error("LoRA " + lf.st->path + " matched no block linears");
        }
    }
    c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
    g_fused = g_int8 && std::getenv("SD_XPU_QWEN21_UNFUSED") == nullptr;
    for (const auto& b : g_blocks) {
        for (const Lin* l : {&b.qkv, &b.out, &b.gate_up, &b.down}) {
            g_fused = g_fused && l->group == 256;
        }
    }
    g_loaded = true;
    if (verbose) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[xpu_qwen21] loaded %d blocks (%s) from %s in %.1f s\n", n_blocks,
                     g_fused ? "int8 W8A8, fused kernels" : g_int8 ? "int8 W8A8" : "bf16", path, s);
    }
}

std::mutex g_load_mutex;  // held for the whole load, so callers wait for an in-flight prefetch

void ensure_loaded(int verbose) {
    std::lock_guard<std::mutex> lock(g_load_mutex);
    if (g_loaded) {
        return;
    }
    const char* path = std::getenv("SD_XPU_QWEN21_WEIGHTS");
    if (path == nullptr) {
        throw std::runtime_error("SD_XPU_QWEN21_WEIGHTS is not set");
    }
    load_weights(path, verbose);
}

struct Segment {
    int64_t start, end;
    bool causal;
};

}  // namespace

static const size_t g_max_blocks = [] {
    const char* v = std::getenv("SD_XPU_MAX_BLOCKS");
    return v != nullptr && *v != '\0' && std::atoll(v) >= 0 ? static_cast<size_t>(std::atoll(v)) : static_cast<size_t>(-1);
}();

extern "C" int xpu_qwen21_init(int verbose) {
    try {
        ensure_loaded(verbose);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_qwen21] init error: %s\n", e.what());
        return 1;
    }
}

// Start loading the block weights in the background (sd.cpp calls this when it creates the model, so the
// load overlaps its own weight loading and text encoding); the first xpu_qwen21_blocks() call waits for it
// on the load mutex, and retries (reporting the error) if the prefetch failed.
extern "C" void xpu_qwen21_prefetch(int verbose) {
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

extern "C" void xpu_qwen21_release() {
    if (!g_loaded) {
        return;
    }
    g_blocks.clear();
    g_loaded = false;
    c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
}

// x_host [rows, hidden]: prefix tokens (rows < prefix_len) then target image tokens.
// mods_host [2, 4 * hidden]: row 0 = target (t), row 1 = prefix (t = 0); chunks = norm1 scale, attn gate,
// norm2 scale, mlp gate. pe_host [rows, head_dim / 2, 2, 2] = [[cos, -sin], [sin, cos]] on adjacent pairs.
// segs [n_segs, 3] = start, end, causal; each segment's queries attend to keys [0, end), causal ones with
// the in-segment future masked.
extern "C" int xpu_qwen21_blocks(const float* x_host,
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
        using clock   = std::chrono::steady_clock;
        const auto t0 = clock::now();
        ensure_loaded(verbose);
        const double eps = 1e-6;
        auto fopt        = at::TensorOptions().dtype(at::kFloat);

        auto x    = at::from_blob(const_cast<float*>(x_host), {rows, hidden}, fopt).pin_memory().to(kXpu);
        auto mods = at::from_blob(const_cast<float*>(mods_host), {2, 4, hidden}, fopt).to(kXpu);
        auto pe   = at::from_blob(const_cast<float*>(pe_host), {rows, head_dim / 2, 2, 2}, fopt).pin_memory().to(kXpu);
        // pe[r][i] = [[cos, -sin], [sin, cos]] -> e^{i theta} per adjacent (even, odd) pair
        auto freqs = at::complex(pe.select(3, 0).select(2, 0), pe.select(3, 0).select(2, 1)).unsqueeze(1);  // [rows, 1, D/2]
        pe.reset();
        std::vector<Segment> segments(static_cast<size_t>(n_segs));
        for (int64_t i = 0; i < n_segs; ++i) {
            segments[i] = {segs[3 * i], segs[3 * i + 1], segs[3 * i + 2] != 0};
        }
        const int64_t inner = heads * head_dim;
        const int64_t P     = prefix_len;
        // row ranges with their modulation row: prefix uses row 1 (t = 0), target row 0
        std::vector<std::pair<std::pair<int64_t, int64_t>, int>> ranges;
        if (P > 0) {
            ranges.push_back({{0, P}, 1});
        }
        ranges.push_back({{P, rows}, 0});
        auto scale1 = mods.select(1, 0) + 1;      // [2, hidden]
        auto gate1  = at::tanh(mods.select(1, 1));
        auto scale2 = mods.select(1, 2) + 1;
        auto gate2  = at::tanh(mods.select(1, 3));
        auto bopt   = at::TensorOptions().dtype(at::kBFloat16).device(kXpu);
        auto h      = at::empty({rows, hidden}, bopt);

        auto norm_mod = [&](const at::Tensor& scale) {  // h = LayerNorm(x) * scale[row], as bf16
            for (const auto& [range, r] : ranges) {
                h.slice(0, range.first, range.second)
                    .copy_(at::layer_norm(x.slice(0, range.first, range.second), {hidden}, scale[r], {}, eps));
            }
            return h;
        };
        auto gated_add = [&](const at::Tensor& y, const at::Tensor& gate) {  // x += y * gate[row]
            for (const auto& [range, r] : ranges) {
                x.slice(0, range.first, range.second).addcmul_(y.slice(0, range.first, range.second), gate[r]);
            }
        };
        // RMSNorm + RoPE on q and k together: [rows, 2, heads, D] bf16 -> bf16; the norm is one f32 reduction
        // over the bf16 input instead of an f32 copy plus pow / mean / rsqrt passes.
        auto freqs_qk     = freqs.unsqueeze(1);  // [rows, 1, 1, D/2]
        auto qk_norm_rope = [&](const at::Tensor& t, const at::Tensor& w) {
            auto inv = at::rsqrt(at::linalg_vector_norm(t, 2, {-1}, true, at::kFloat).square_().div_(head_dim).add_(eps));
            auto n   = t * inv * w;
            auto c   = at::view_as_complex(n.view({rows, 2, heads, head_dim / 2, 2})) * freqs_qk;
            return at::view_as_real(c).view({rows, 2, heads, head_dim}).to(at::kBFloat16);
        };

        static const bool timing = std::getenv("SD_XPU_QWEN21_TIMING") != nullptr;
        auto& tq                 = c10::xpu::getCurrentXPUStream().queue();
        auto mark                = [&](const char* what, int i) {
            if (timing) {
                tq.wait();
                std::fprintf(stderr, "[xpu_qwen21] block %d %s: %.3f s\n", i, what, std::chrono::duration<double>(clock::now() - t0).count());
            }
        };
        mark("inputs", -1);
        // Fused int8 path: each activation is produced already rotated + quantized for the next int8 GEMM.
        if (g_fused) {
            const int64_t ffn = g_blocks[0].gate_up.q.size(0) / 2;
            auto pe_dev       = at::from_blob(const_cast<float*>(pe_host), {rows, head_dim / 2, 2, 2}, fopt).pin_memory().to(kXpu);
            auto s1 = scale1.contiguous(), s2 = scale2.contiguous(), g1 = gate1.contiguous(), g2 = gate2.contiguous();
            auto i8   = x.options().dtype(at::kChar);
            auto xq   = at::empty({rows, hidden}, i8);
            auto xq_f = at::empty({rows, ffn}, i8);
            auto sx   = at::empty({rows}, fopt.device(kXpu));
            auto qo   = at::empty({heads, rows, head_dim}, bopt);
            auto ko   = at::empty({heads, rows, head_dim}, bopt);
            auto* xp  = x.data_ptr<float>();
            auto bf   = [](const at::Tensor& t) { return reinterpret_cast<const xq21::bf16*>(t.data_ptr()); };
            at::Tensor y;          // previous block's MLP output, added (gated) by the next norm
            const float* yg = nullptr;
            int bi          = 0;
            for (const auto& blk : g_blocks) {
                if (static_cast<size_t>(bi) >= g_max_blocks) {
                    break;
                }
                xq21::ln_mod_rot_quant(tq, xp, y.defined() ? bf(y) : nullptr, yg, s1.data_ptr<float>(), P, rows, hidden, eps,
                                       xq.data_ptr<int8_t>(), sx.data_ptr<float>());
                auto qkv = matmul_int8(xq, sx, blk.qkv);
                xq21::qk_norm_rope(tq, bf(qkv), blk.norm_qk.data_ptr<float>(), pe_dev.data_ptr<float>(), rows, heads, head_dim, eps,
                                   reinterpret_cast<xq21::bf16*>(qo.data_ptr()), reinterpret_cast<xq21::bf16*>(ko.data_ptr()));
                auto qh = qo.unsqueeze(0), kh = ko.unsqueeze(0);  // [1, H, rows, D]
                auto vh = qkv.slice(1, 2 * inner, 3 * inner).view({rows, heads, head_dim}).transpose(0, 1).unsqueeze(0);
                mark("qkv+rope", bi);
                for (const auto& seg : segments) {
                    auto sqh = qh.slice(2, seg.start, seg.end);
                    auto sk  = kh.slice(2, 0, seg.end);
                    auto sv  = vh.slice(2, 0, seg.end);
                    at::Tensor o;
                    if (seg.causal && seg.start == 0) {
                        o = at::scaled_dot_product_attention(sqh, sk, sv, {}, 0.0, /*is_causal=*/true);
                    } else if (seg.causal) {
                        auto lopt = at::TensorOptions().dtype(at::kLong).device(kXpu);
                        auto mask = at::arange(0, seg.end, lopt).unsqueeze(0) <= at::arange(seg.start, seg.end, lopt).unsqueeze(1);
                        o         = at::scaled_dot_product_attention(sqh, sk, sv, mask);
                    } else {
                        o = at::scaled_dot_product_attention(sqh, sk, sv);
                    }
                    o = o.contiguous();
                    xq21::attn_out_rot_quant(tq, bf(o), seg.end - seg.start, heads, head_dim, seg.start, xq.data_ptr<int8_t>(),
                                             sx.data_ptr<float>());
                }
                qkv.reset();
                mark("sdpa", bi);
                y = matmul_int8(xq, sx, blk.out);
                xq21::ln_mod_rot_quant(tq, xp, bf(y), g1.data_ptr<float>(), s2.data_ptr<float>(), P, rows, hidden, eps,
                                       xq.data_ptr<int8_t>(), sx.data_ptr<float>());
                auto gu = matmul_int8(xq, sx, blk.gate_up);
                xq21::swiglu_rot_quant(tq, bf(gu), rows, ffn, xq_f.data_ptr<int8_t>(), sx.data_ptr<float>());
                gu.reset();
                y  = matmul_int8(xq_f, sx, blk.down);
                yg = g2.data_ptr<float>();
                mark("done", bi++);
            }
            gated_add(y, gate2);
        }
        int bi = 0;
        static const std::vector<Block> kNone;
        for (const auto& blk : g_fused ? kNone : g_blocks) {  // unfused path (bf16, or int8 without fusion)
            if (static_cast<size_t>(bi) >= g_max_blocks) {
                break;
            }
            auto qkv = linear(norm_mod(scale1), blk.qkv);
            mark("qkv", bi);
            auto qk  = qk_norm_rope(qkv.slice(1, 0, 2 * inner).view({rows, 2, heads, head_dim}), blk.norm_qk);
            auto qh  = qk.select(1, 0).transpose(0, 1).unsqueeze(0);  // [1, H, rows, D]
            auto kh  = qk.select(1, 1).transpose(0, 1).unsqueeze(0);
            auto vh  = qkv.slice(1, 2 * inner, 3 * inner).view({rows, heads, head_dim}).transpose(0, 1).unsqueeze(0);
            mark("qk norm+rope", bi);
            std::vector<at::Tensor> outs;
            for (const auto& seg : segments) {
                auto sq = qh.slice(2, seg.start, seg.end);
                auto sk = kh.slice(2, 0, seg.end);
                auto sv = vh.slice(2, 0, seg.end);
                if (seg.causal && seg.start == 0) {
                    outs.push_back(at::scaled_dot_product_attention(sq, sk, sv, {}, 0.0, /*is_causal=*/true));
                } else if (seg.causal) {
                    // key j is visible to query i (absolute positions) unless start <= i < j
                    auto lopt = at::TensorOptions().dtype(at::kLong).device(kXpu);
                    auto mask = at::arange(0, seg.end, lopt).unsqueeze(0) <= at::arange(seg.start, seg.end, lopt).unsqueeze(1);
                    outs.push_back(at::scaled_dot_product_attention(sq, sk, sv, mask));
                } else {
                    outs.push_back(at::scaled_dot_product_attention(sq, sk, sv));
                }
            }
            qk.reset();
            qkv.reset();
            auto o = (outs.size() == 1 ? outs[0] : at::cat(outs, 2)).squeeze(0).transpose(0, 1).reshape({rows, inner});
            outs.clear();
            mark("sdpa", bi);
            gated_add(linear(o, blk.out), gate1);
            mark("out", bi);
            o.reset();

            auto gu         = linear(norm_mod(scale2), blk.gate_up);
            const int64_t f = gu.size(1) / 2;
            auto act        = gu.slice(1, f, 2 * f) * at::silu(gu.slice(1, 0, f));
            gu.reset();
            mark("gate_up+swiglu", bi);
            gated_add(linear(act, blk.down), gate2);
            mark("done", bi++);
        }

        auto host_out = x.to(c10::Device(c10::DeviceType::CPU));
        std::memcpy(out_host, host_out.data_ptr<float>(), static_cast<size_t>(rows * hidden) * sizeof(float));
        // keep only the weights resident between calls so the Vulkan side has room (VAE decode)
        x.reset();
        h.reset();
        freqs.reset();
        freqs_qk.reset();
        mods.reset();
        c10::getDeviceAllocator(c10::DeviceType::XPU)->emptyCache();
        if (verbose) {
            const double s = std::chrono::duration<double>(clock::now() - t0).count();
            std::fprintf(stderr, "[xpu_qwen21] %zu blocks, rows=%lld (prefix %lld), segments=%lld: %.2f s\n", g_blocks.size(),
                         (long long)rows, (long long)P, (long long)n_segs, s);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_qwen21] error: %s\n", e.what());
        return 1;
    }
}
