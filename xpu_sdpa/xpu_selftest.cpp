// Self-test driver for the XPU block libraries (host-only: no SYCL, oneDNN or torch at link time; the libraries
// are dlopened). See SELFTEST.md.
//
//   xpu_selftest gen   h3|h3vae|qwen21 <dir> [options]   write deterministic random inputs + meta.txt to <dir>
//                                                       (reads the weights' safetensors header for the model shapes)
//   xpu_selftest ref   h3|h3vae|qwen21 <lib.so> <dir>    run <lib.so> (the libtorch build) on <dir>'s inputs and write
//                                                       <dir>/ref_out.f32
//   xpu_selftest check h3|h3vae|qwen21 <lib.so> <dir>    run the native <lib.so> self-test (sets SD_XPU_*_SELFTEST=<dir>
//                                                       and calls its init), exit status 0 = PASS
//
// gen options: h3:     rows=1024 trows=1 half_rot=32       (needs SD_XPU_H3_WEIGHTS)
//              h3vae:  W=16 H=16 T=2 half_rot=16
//              qwen21: rows=1280 prefix=256                (needs SD_XPU_QWEN21_WEIGHTS)
// Each step must run in its own process (the libtorch and native builds bring different SYCL runtimes, and
// the H3 weights do not fit on the device twice).

#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::map<std::string, std::vector<int64_t>> header_shapes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    std::string header(n, '\0');
    f.read(header.data(), static_cast<std::streamsize>(n));
    static const std::regex entry(R"re("([^"]+)":\s*\{\s*"dtype":\s*"[A-Z0-9]+",\s*"shape":\s*\[([0-9, ]*)\])re");
    std::map<std::string, std::vector<int64_t>> out;
    for (auto it = std::sregex_iterator(header.begin(), header.end(), entry); it != std::sregex_iterator(); ++it) {
        std::vector<int64_t> shape;
        std::string dims = (*it)[2];
        size_t pos       = 0;
        while (pos < dims.size()) {
            size_t comma    = dims.find(',', pos);
            std::string tok = dims.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (tok.find_first_not_of(' ') != std::string::npos) {
                shape.push_back(std::stoll(tok));
            }
            pos = comma == std::string::npos ? dims.size() : comma + 1;
        }
        out[(*it)[1]] = shape;
    }
    return out;
}

const std::vector<int64_t>& need(const std::map<std::string, std::vector<int64_t>>& h, const std::string& name) {
    auto it = h.find(name);
    if (it == h.end()) {
        throw std::runtime_error("weights have no tensor " + name);
    }
    return it->second;
}

std::map<std::string, double> parse_opts(int argc, char** argv, int first, std::map<std::string, double> defaults) {
    for (int i = first; i < argc; ++i) {
        std::string a = argv[i];
        auto eq       = a.find('=');
        if (eq == std::string::npos || !defaults.count(a.substr(0, eq))) {
            throw std::runtime_error("unknown option " + a);
        }
        defaults[a.substr(0, eq)] = std::atof(a.c_str() + eq + 1);
    }
    return defaults;
}

template <class T>
void write_vec(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    if (!f) {
        throw std::runtime_error("cannot write " + path);
    }
}

template <class T>
std::vector<T> read_vec(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<T> v(b.size() / sizeof(T));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
    return v;
}

std::map<std::string, double> read_meta(const std::string& dir) {
    std::ifstream f(dir + "/meta.txt");
    if (!f) {
        throw std::runtime_error("cannot open " + dir + "/meta.txt");
    }
    std::map<std::string, double> m;
    std::string k;
    double v;
    while (f >> k >> v) {
        m[k] = v;
    }
    return m;
}

void write_meta(const std::string& dir, const std::map<std::string, double>& m) {
    std::ofstream f(dir + "/meta.txt");
    for (const auto& [k, v] : m) {
        f << k << " " << static_cast<int64_t>(v) << "\n";
    }
}

std::mt19937 g_rng(1234);

std::vector<float> randn(size_t n, float sd) {
    std::normal_distribution<float> d(0.f, sd);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = d(g_rng);
    }
    return v;
}

// RoPE table [rows][half][2][2] = [[cos, -sin], [sin, cos]] with angle (row + 1) * 10000^(-i / half)
std::vector<float> rope_table(int64_t rows, int64_t half) {
    std::vector<float> pe(static_cast<size_t>(rows * half * 4));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < half; ++i) {
            const double a = static_cast<double>(r % 977 + 1) * std::pow(10000.0, -static_cast<double>(i) / static_cast<double>(half));
            float* p       = pe.data() + (r * half + i) * 4;
            p[0]           = static_cast<float>(std::cos(a));
            p[1]           = static_cast<float>(-std::sin(a));
            p[2]           = static_cast<float>(std::sin(a));
            p[3]           = static_cast<float>(std::cos(a));
        }
    }
    return pe;
}

std::string env_or_throw(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        throw std::runtime_error(std::string(name) + " is not set");
    }
    return v;
}

void gen(const std::string& model, const std::string& dir, int argc, char** argv) {
    std::filesystem::create_directories(dir);
    if (model == "h3") {
        auto o              = parse_opts(argc, argv, 4, {{"rows", 1024}, {"trows", 1}, {"half_rot", 32}});
        auto h              = header_shapes(env_or_throw("SD_XPU_H3_WEIGHTS"));
        const int64_t hid   = need(h, "blocks.0.norm1.weight").at(0);
        const auto& ada     = need(h, "blocks.0.adaln_proj.linear.weight");
        const int64_t hd    = need(h, "blocks.0.attn.q_norm.weight").at(0);
        const int64_t inner = need(h, "blocks.0.attn.qkv_proj.weight").at(0) / 3;
        const int64_t rows = static_cast<int64_t>(o["rows"]), trows = static_cast<int64_t>(o["trows"]), half = static_cast<int64_t>(o["half_rot"]);
        // two segments with different modulation rows
        std::vector<int64_t> segs = {0, rows / 2, 0, rows / 2, rows, std::min<int64_t>(1, trows * 3 - 1)};
        write_vec(dir + "/x.f32", randn(static_cast<size_t>(rows * hid), 1.f));
        write_vec(dir + "/temb.f32", randn(static_cast<size_t>(trows * ada.at(1)), 0.5f));
        write_vec(dir + "/pe.f32", rope_table(rows, half));
        write_vec(dir + "/segs.i64", segs);
        write_meta(dir, {{"rows", rows},
                         {"hidden", hid},
                         {"trows", trows},
                         {"tdim", ada.at(1)},
                         {"half_rot", half},
                         {"n_segs", 2},
                         {"heads", inner / hd},
                         {"head_dim", hd}});
    } else if (model == "h3vae") {
        auto o          = parse_opts(argc, argv, 4, {{"W", 16}, {"H", 16}, {"T", 2}, {"half_rot", 16}});
        const int64_t W = static_cast<int64_t>(o["W"]), H = static_cast<int64_t>(o["H"]), T = static_cast<int64_t>(o["T"]);
        const int64_t half = static_cast<int64_t>(o["half_rot"]);
        const int64_t N    = W * H * T + 5;
        write_vec(dir + "/z.f32", randn(static_cast<size_t>(24 * W * H * T), 1.f));
        write_vec(dir + "/pe.f32", rope_table(N, half));
        write_meta(dir, {{"W", W}, {"H", H}, {"T", T}, {"C", 24}, {"half_rot", half}, {"pe_rows", N}});
    } else if (model == "qwen21") {
        auto o              = parse_opts(argc, argv, 4, {{"rows", 1280}, {"prefix", 256}});
        auto h              = header_shapes(env_or_throw("SD_XPU_QWEN21_WEIGHTS"));
        const auto& wq      = need(h, "transformer_blocks.0.attn.to_q.weight");
        const int64_t hd    = need(h, "transformer_blocks.0.attn.norm_q.weight").at(0);
        const int64_t inner = wq.at(0), hid = wq.at(1);
        const int64_t rows = static_cast<int64_t>(o["rows"]), P = static_cast<int64_t>(o["prefix"]);
        if (P < 2 || P >= rows) {
            throw std::runtime_error("need 2 <= prefix < rows");
        }
        // two causal prefix segments (the second one exercises the offset mask) and the non-causal image segment
        std::vector<int64_t> segs = {0, P / 2, 1, P / 2, P, 1, P, rows, 0};
        write_vec(dir + "/x.f32", randn(static_cast<size_t>(rows * hid), 1.f));
        write_vec(dir + "/mods.f32", randn(static_cast<size_t>(2 * 4 * hid), 0.3f));
        write_vec(dir + "/pe.f32", rope_table(rows, hd / 2));
        write_vec(dir + "/segs.i64", segs);
        write_meta(dir, {{"rows", rows}, {"hidden", hid}, {"prefix_len", P}, {"n_segs", 3}, {"heads", inner / hd}, {"head_dim", hd}});
    } else {
        throw std::runtime_error("unknown model " + model);
    }
    std::fprintf(stderr, "wrote %s inputs to %s\n", model.c_str(), dir.c_str());
}

void* open_lib(const std::string& path) {
    void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    }
    return lib;
}

template <class F>
F sym(void* lib, const char* name) {
    auto f = reinterpret_cast<F>(dlsym(lib, name));
    if (f == nullptr) {
        throw std::runtime_error(std::string("missing symbol ") + name);
    }
    return f;
}

using H3BlocksFn  = int (*)(const float*, int64_t, int64_t, const float*, int64_t, int64_t, const float*, int64_t, const int64_t*, int64_t, int64_t,
                           int64_t, float*, int);
using VaeDecodeFn = int (*)(const float*, int64_t, int64_t, int64_t, int64_t, const float*, int64_t, int64_t, float*, int);
using QwenBlocksFn = int (*)(const float*, int64_t, int64_t, const float*, int64_t, const float*, const int64_t*, int64_t, int64_t, int64_t, float*,
                             int);

int ref(const std::string& model, const std::string& lib_path, const std::string& dir) {
    auto m    = read_meta(dir);
    void* lib = open_lib(lib_path);
    auto i64  = [&](const char* k) { return static_cast<int64_t>(m.at(k)); };
    std::vector<float> out;
    int rc = 1;
    if (model == "h3") {
        auto x = read_vec<float>(dir + "/x.f32"), temb = read_vec<float>(dir + "/temb.f32"), pe = read_vec<float>(dir + "/pe.f32");
        auto segs = read_vec<int64_t>(dir + "/segs.i64");
        out.resize(x.size());
        rc = sym<H3BlocksFn>(lib, "xpu_h3_blocks")(x.data(), i64("rows"), i64("hidden"), temb.data(), i64("trows"), i64("tdim"), pe.data(),
                                                    i64("half_rot"), segs.data(), i64("n_segs"), i64("heads"), i64("head_dim"), out.data(), 1);
    } else if (model == "h3vae") {
        auto z = read_vec<float>(dir + "/z.f32"), pe = read_vec<float>(dir + "/pe.f32");
        out.resize(static_cast<size_t>(3 * i64("T") * 4 * i64("H") * 16 * i64("W") * 16));
        rc = sym<VaeDecodeFn>(lib, "xpu_h3vae_decode")(z.data(), i64("W"), i64("H"), i64("T"), i64("C"), pe.data(), i64("pe_rows"), i64("half_rot"),
                                                        out.data(), 1);
    } else if (model == "qwen21") {
        auto x = read_vec<float>(dir + "/x.f32"), mods = read_vec<float>(dir + "/mods.f32"), pe = read_vec<float>(dir + "/pe.f32");
        auto segs = read_vec<int64_t>(dir + "/segs.i64");
        out.resize(x.size());
        rc = sym<QwenBlocksFn>(lib, "xpu_qwen21_blocks")(x.data(), i64("rows"), i64("hidden"), mods.data(), i64("prefix_len"), pe.data(), segs.data(),
                                                          i64("n_segs"), i64("heads"), i64("head_dim"), out.data(), 1);
    } else {
        throw std::runtime_error("unknown model " + model);
    }
    if (rc != 0) {
        std::fprintf(stderr, "%s returned %d\n", lib_path.c_str(), rc);
        return rc;
    }
    write_vec(dir + "/ref_out.f32", out);
    std::fprintf(stderr, "wrote %s/ref_out.f32 (%zu values)\n", dir.c_str(), out.size());
    return 0;
}

int check(const std::string& model, const std::string& lib_path, const std::string& dir) {
    const char* env  = model == "h3" ? "SD_XPU_H3_SELFTEST" : model == "h3vae" ? "SD_XPU_H3VAE_SELFTEST" : "SD_XPU_QWEN21_SELFTEST";
    const char* init = model == "h3" ? "xpu_h3_init" : model == "h3vae" ? "xpu_h3vae_init" : "xpu_qwen21_init";
    if (model != "h3" && model != "h3vae" && model != "qwen21") {
        throw std::runtime_error("unknown model " + model);
    }
    setenv(env, dir.c_str(), 1);
    void* lib = open_lib(lib_path);
    return sym<int (*)(int)>(lib, init)(1);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 4 && std::strcmp(argv[1], "gen") == 0) {
            gen(argv[2], argv[3], argc, argv);
            return 0;
        }
        if (argc == 5 && std::strcmp(argv[1], "ref") == 0) {
            return ref(argv[2], argv[3], argv[4]);
        }
        if (argc == 5 && std::strcmp(argv[1], "check") == 0) {
            const int rc = check(argv[2], argv[3], argv[4]);
            std::fprintf(stderr, "selftest %s: %s\n", argv[2], rc == 0 ? "PASS" : "FAIL");
            return rc;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xpu_selftest: %s\n", e.what());
        return 2;
    }
    std::fprintf(stderr,
                 "usage: xpu_selftest gen h3|h3vae|qwen21 <dir> [opt=value ...]\n"
                 "       xpu_selftest ref h3|h3vae|qwen21 <torch-lib.so> <dir>\n"
                 "       xpu_selftest check h3|h3vae|qwen21 <native-lib.so> <dir>\n");
    return 2;
}
