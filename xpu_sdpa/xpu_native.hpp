// Native Intel XPU runtime shared by libxpu_h3, libxpu_h3vae and libxpu_qwen21: SYCL (DPC++) + oneDNN
// (primitives and the Graph API) on Level Zero, with no libtorch / ATen / c10.
//
//   Runtime    one in-order sycl::queue on the first Level Zero GPU, a oneDNN engine/stream on that queue and a
//              oneDNN Graph engine whose allocator is the pool below. Created on first use, so loading a
//              library touches no GPU state.
//   Pool       bounded caching allocator for SYCL USM device memory (see class Pool). Every scratch buffer,
//              oneDNN scratchpad and oneDNN Graph allocation goes through it; resident weights are separate
//              direct allocations. trim() hands every cached block back to the driver, release_all() also
//              drops the oneDNN primitive / compiled-partition caches.
//   Buf        move-only owning handle of one device allocation.
//   matmul()   oneDNN matmul (f32 / bf16 / f16 / s8 x s8 with per-row and per-column scales), bias and
//              accumulate (sum post-op), strided source; primitives cached per shape, user-mode scratchpad.
//   attention  oneDNN Graph fused SDPA (MatMul -> Multiply(scale) -> SoftMax -> MatMul, the same partition
//              torch's XPU SDPA lowers to) with a SYCL flash-attention kernel as fallback (and for causal masks).
//   kernels    small SYCL elementwise helpers (casts, Hadamard rotation, SwiGLU, head transposes, ...).
//   io         safetensors reader, chunked host<->device copies through two bounded pinned staging buffers.
//
// Environment (all libraries):
//   SD_XPU_POOL_CACHE_MB  (default 1024)  max bytes of freed blocks the pool keeps cached for reuse; above it,
//                                         blocks go straight back to the driver. (4096 left an HD ControlNet H3 job
//                                         at 31+ of 32 GiB; past the card the driver moves buffers to system memory
//                                         and the job crawls. 1024 costs no measurable speed, 2026-10-18.)
//   SD_XPU_POOL_LIMIT_MB  (default 0=off) hard cap on live + cached pool bytes; an allocation over it first
//                                         trims the cache, then fails with a clear error instead of a driver OOM.
//   SD_XPU_POOL_VERBOSE=1                 print pool statistics on trim.
//   SD_XPU_MAX_BLOCKS=N                   debug / self-test: run only the first N transformer blocks.
#pragma once

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define XN_EXPORT extern "C" __attribute__((visibility("default")))

namespace xn {

using bf16 = sycl::ext::oneapi::bfloat16;
using f16  = sycl::half;

enum class DType { f32, bf16, f16, s8, u8, s32 };

inline size_t dsize(DType t) {
    switch (t) {
        case DType::f32: return 4;
        case DType::bf16: return 2;
        case DType::f16: return 2;
        case DType::s8: return 1;
        case DType::u8: return 1;
        case DType::s32: return 4;
    }
    return 0;
}

inline const char* dname(DType t) {
    switch (t) {
        case DType::f32: return "f32";
        case DType::bf16: return "bf16";
        case DType::f16: return "f16";
        case DType::s8: return "s8";
        case DType::u8: return "u8";
        case DType::s32: return "s32";
    }
    return "?";
}

inline DType dtype_of(const std::string& st) {  // safetensors dtype string
    if (st == "F32") return DType::f32;
    if (st == "BF16") return DType::bf16;
    if (st == "F16") return DType::f16;
    if (st == "I8") return DType::s8;
    if (st == "U8") return DType::u8;
    if (st == "I32") return DType::s32;
    throw std::runtime_error("unsupported safetensors dtype " + st);
}

inline int64_t env_i64(const char* name, int64_t def) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' ? std::atoll(v) : def;
}

inline double env_f64(const char* name, double def) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' ? std::atof(v) : def;
}

// ---- host-side float conversions (round to nearest even, like the device casts) ----

inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return 0x7fc0;  // NaN
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

inline float bf16_bits_to_f32(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp        = (h >> 10) & 0x1fu;
    uint32_t mant       = h & 0x3ffu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {  // subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            u = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7f800000u | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- runtime ----

class Pool;

struct Runtime {
    sycl::queue q;
    dnnl::engine eng;    // primitives
    dnnl::stream strm;   // on q
    dnnl::engine geng;   // graph (allocator = pool)
    dnnl::stream gstrm;  // on q
    std::unique_ptr<Pool> pool;
};

Runtime& rt();
inline sycl::queue& Q() { return rt().q; }
inline std::atomic<bool> g_rt_started{false};  // set once rt() has created the queue (release paths check it)

// Bounded caching allocator for device USM memory.
//   - alloc(n) rounds n up (512 B below 1 MiB, 2 MiB above) and reuses a cached free block of the same rounded
//     size class (at most 1/8 larger), else calls sycl::malloc_device. On failure it trims the cache and retries.
//   - free(p) returns the block to the cache. Work on the single in-order queue is ordered, so a block freed
//     after enqueueing its last use can be handed to a later kernel without waiting. When the cache would grow
//     past SD_XPU_POOL_CACHE_MB, the queue is drained and the largest cached blocks go back to the driver.
//   - trim() drains the queue and returns every cached block to the driver; live blocks are untouched.
//   - Resident allocations (weights) bypass the cache: alloc_resident / free_resident.
class Pool {
public:
    explicit Pool(sycl::queue& q)
        : q_(q) {
        cache_cap_ = static_cast<size_t>(env_i64("SD_XPU_POOL_CACHE_MB", 1024)) << 20;
        limit_     = static_cast<size_t>(env_i64("SD_XPU_POOL_LIMIT_MB", 0)) << 20;
        verbose_   = env_i64("SD_XPU_POOL_VERBOSE", 0) != 0;
    }
    ~Pool() = default;  // process exit: leave device memory to the driver

    static size_t round_size(size_t n) {
        if (n == 0) {
            n = 1;
        }
        const size_t g = n < (size_t(1) << 20) ? 512 : (size_t(2) << 20);
        return (n + g - 1) / g * g;
    }

    void* alloc(size_t n) {
        const size_t r = round_size(n);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = free_.lower_bound(r);
        if (it != free_.end() && it->first <= r + r / 8) {
            void* p          = it->second;
            const size_t got = it->first;
            free_.erase(it);
            cached_ -= got;
            live_ += got;
            peak_ = std::max(peak_, live_);
            return p;
        }
        if (limit_ != 0 && live_ + cached_ + r > limit_) {
            trim_locked();
            if (live_ + r > limit_) {
                throw std::runtime_error("xpu pool: allocation of " + std::to_string(r >> 20) + " MiB exceeds SD_XPU_POOL_LIMIT_MB (live " +
                                         std::to_string(live_ >> 20) + " MiB)");
            }
        }
        void* p = raw_malloc(r);
        if (p == nullptr) {
            trim_locked();
            p = raw_malloc(r);
        }
        if (p == nullptr) {
            throw std::runtime_error("xpu pool: out of device memory allocating " + std::to_string(r >> 20) + " MiB (live " +
                                     std::to_string(live_ >> 20) + " MiB, resident " + std::to_string(resident_ >> 20) + " MiB)");
        }
        sizes_[p] = r;
        live_ += r;
        peak_ = std::max(peak_, live_);
        return p;
    }

    void free(void* p) {
        if (p == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sizes_.find(p);
        if (it == sizes_.end()) {
            return;  // not ours (should not happen)
        }
        const size_t r = it->second;
        live_ -= r;
        free_.emplace(r, p);
        cached_ += r;
        if (cached_ > cache_cap_) {
            q_.wait();
            while (cached_ > cache_cap_ && !free_.empty()) {
                auto last = std::prev(free_.end());  // largest first
                raw_free(last->second);
                cached_ -= last->first;
                free_.erase(last);
            }
        }
    }

    void* alloc_resident(size_t n) {
        const size_t r = round_size(n);
        std::lock_guard<std::mutex> lock(mu_);
        void* p = raw_malloc(r);
        if (p == nullptr) {
            trim_locked();
            p = raw_malloc(r);
        }
        if (p == nullptr) {
            throw std::runtime_error("xpu: out of device memory for " + std::to_string(r >> 20) + " MiB of weights (resident " +
                                     std::to_string(resident_ >> 20) + " MiB)");
        }
        sizes_[p] = r;
        resident_ += r;
        return p;
    }

    void free_resident(void* p) {
        if (p == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sizes_.find(p);
        if (it == sizes_.end()) {
            return;
        }
        resident_ -= it->second;
        q_.wait();
        raw_free(p);
    }

    void trim() {
        std::lock_guard<std::mutex> lock(mu_);
        trim_locked();
    }

    void report(const char* tag) {
        std::lock_guard<std::mutex> lock(mu_);
        std::fprintf(stderr, "[%s] pool: live %.0f MiB, cached %.0f MiB, peak %.0f MiB, resident %.0f MiB\n", tag, live_ / 1048576.0,
                     cached_ / 1048576.0, peak_ / 1048576.0, resident_ / 1048576.0);
    }

    bool verbose() const { return verbose_; }
    size_t live() const { return live_; }
    size_t resident() const { return resident_; }
    void reset_peak() {
        std::lock_guard<std::mutex> lock(mu_);
        peak_ = live_;
    }

private:
    void* raw_malloc(size_t r) {
        try {
            return sycl::malloc_device(r, q_);
        } catch (const sycl::exception&) {
            return nullptr;
        }
    }
    void raw_free(void* p) {
        sizes_.erase(p);
        sycl::free(p, q_);
    }
    void trim_locked() {
        if (free_.empty()) {
            return;
        }
        q_.wait();
        for (auto& [r, p] : free_) {
            raw_free(p);
        }
        free_.clear();
        cached_ = 0;
    }

    sycl::queue& q_;
    std::mutex mu_;
    std::multimap<size_t, void*> free_;
    std::unordered_map<void*, size_t> sizes_;
    size_t live_ = 0, cached_ = 0, peak_ = 0, resident_ = 0;
    size_t cache_cap_ = 0, limit_ = 0;
    bool verbose_ = false;
};

inline Pool& pool() { return *rt().pool; }

// Error path of an exported entry point: hand the cached blocks back so a fallback has room. Not after a SYCL or
// oneDNN runtime error: the queue may hold a command that never completes, and trim's queue wait then spins forever
// (an H3 shot, 2026-10-17: UR_RESULT_ERROR_OUT_OF_RESOURCES, then 13 min at 100% CPU in Pool::trim).
inline void trim_after_error(const std::exception& e) {
    if (!g_rt_started || dynamic_cast<const sycl::exception*>(&e) != nullptr || dynamic_cast<const dnnl::error*>(&e) != nullptr) {
        return;
    }
    try {
        pool().trim();
    } catch (...) {
    }
}

// Owning handle of one device allocation (pooled scratch, or resident weights).
class Buf {
public:
    Buf() = default;
    Buf(size_t bytes, bool resident)
        : n_(bytes), resident_(resident) {
        p_ = resident ? pool().alloc_resident(bytes) : pool().alloc(bytes);
    }
    Buf(const Buf&)            = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept
        : p_(o.p_), n_(o.n_), resident_(o.resident_) {
        o.p_ = nullptr;
        o.n_ = 0;
    }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) {
            reset();
            p_        = o.p_;
            n_        = o.n_;
            resident_ = o.resident_;
            o.p_      = nullptr;
            o.n_      = 0;
        }
        return *this;
    }
    ~Buf() { reset(); }

    void reset() {
        if (p_ != nullptr) {
            if (resident_) {
                pool().free_resident(p_);
            } else {
                pool().free(p_);
            }
        }
        p_ = nullptr;
        n_ = 0;
    }
    template <class T = void>
    T* as() const {
        return static_cast<T*>(p_);
    }
    void* get() const { return p_; }
    size_t bytes() const { return n_; }
    bool empty() const { return p_ == nullptr; }
    bool resident() const { return resident_; }

private:
    void* p_       = nullptr;
    size_t n_      = 0;
    bool resident_ = false;
};

inline Buf scratch(size_t bytes) { return Buf(bytes, false); }
inline Buf resident(size_t bytes) { return Buf(bytes, true); }
template <class T>
inline Buf scratch_n(int64_t n) {
    return Buf(static_cast<size_t>(n) * sizeof(T), false);
}

// ---- oneDNN graph allocator (pool-backed) ----

inline void* graph_alloc(size_t size, size_t /*alignment*/, const void* /*dev*/, const void* /*ctx*/) { return pool().alloc(size); }
inline void graph_free(void* buf, const void* /*dev*/, const void* /*ctx*/, void* /*event*/) { pool().free(buf); }

inline sycl::device pick_device() {
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) {
            continue;
        }
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) {
            return d;
        }
    }
    return sycl::device(sycl::gpu_selector_v);
}

inline Runtime& rt() {
    static Runtime* r = [] {
        auto* s = new Runtime();  // never destroyed: device memory and queues go with the process
        // allow single device allocations above 4 GB (as torch-xpu does); must precede the first platform query
        setenv("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1", 0);
        s->q    = sycl::queue(pick_device(), sycl::property_list{sycl::property::queue::in_order{}});
        s->pool = std::make_unique<Pool>(s->q);
        s->eng  = dnnl::sycl_interop::make_engine(s->q.get_device(), s->q.get_context());
        s->strm = dnnl::sycl_interop::make_stream(s->eng, s->q);
        // the patched oneDNN fails graph compilation instead of falling back to kernels with an S x S intermediate
        setenv("ONEDNN_GRAPH_SDPA_PRIMITIVE_ONLY", "1", 0);
        static dnnl::graph::allocator alloc = dnnl::graph::sycl_interop::make_allocator(&graph_alloc, &graph_free);
        s->geng  = dnnl::graph::sycl_interop::make_engine_with_allocator(s->q.get_device(), s->q.get_context(), alloc);
        s->gstrm = dnnl::sycl_interop::make_stream(s->geng, s->q);
        g_rt_started = true;
        return s;
    }();
    return *r;
}

// ---- host <-> device copies through two bounded pinned staging buffers ----

class Staging {
public:
    static constexpr size_t kChunk = size_t(64) << 20;

    // copy host -> device; `fill(dst, offset, len)` writes the next host chunk into the pinned buffer (lets the
    // file reader stream straight into pinned memory)
    template <class Fill>
    void h2d_fill(void* dev, size_t bytes, Fill&& fill) {
        ensure();
        sycl::event ev[2];
        bool pending[2] = {false, false};
        size_t off      = 0;
        int i           = 0;
        while (off < bytes) {
            const size_t len = std::min(kChunk, bytes - off);
            if (pending[i]) {
                ev[i].wait();
            }
            fill(buf_[i], off, len);
            ev[i]      = Q().memcpy(static_cast<char*>(dev) + off, buf_[i], len);
            pending[i] = true;
            off += len;
            i ^= 1;
        }
        for (int k = 0; k < 2; ++k) {
            if (pending[k]) {
                ev[k].wait();
            }
        }
    }

    void h2d(void* dev, const void* host, size_t bytes) {
        h2d_fill(dev, bytes, [&](void* pinned, size_t off, size_t len) { std::memcpy(pinned, static_cast<const char*>(host) + off, len); });
    }

    void d2h(void* host, const void* dev, size_t bytes) {
        ensure();
        size_t off = 0;
        // device -> pinned (async) overlapped with pinned -> host of the previous chunk
        sycl::event ev[2];
        size_t len_of[2] = {0, 0}, off_of[2] = {0, 0};
        bool pending[2]  = {false, false};
        int i            = 0;
        auto drain       = [&](int k) {
            ev[k].wait();
            std::memcpy(static_cast<char*>(host) + off_of[k], buf_[k], len_of[k]);
            pending[k] = false;
        };
        while (off < bytes) {
            const size_t len = std::min(kChunk, bytes - off);
            if (pending[i]) {
                drain(i);
            }
            ev[i]      = Q().memcpy(buf_[i], static_cast<const char*>(dev) + off, len);
            len_of[i]  = len;
            off_of[i]  = off;
            pending[i] = true;
            off += len;
            i ^= 1;
        }
        for (int k = 0; k < 2; ++k) {
            if (pending[i]) {
                drain(i);
            }
            i ^= 1;
        }
    }

    void release() {
        for (auto& b : buf_) {
            if (b != nullptr) {
                sycl::free(b, Q());
                b = nullptr;
            }
        }
    }

private:
    void ensure() {
        for (auto& b : buf_) {
            if (b == nullptr) {
                b = sycl::malloc_host(kChunk, Q());
                if (b == nullptr) {
                    throw std::runtime_error("xpu: cannot allocate pinned staging memory");
                }
            }
        }
    }
    void* buf_[2] = {nullptr, nullptr};
};

inline Staging& staging() {
    static auto* s = new Staging();
    return *s;
}

inline void h2d(void* dev, const void* host, size_t bytes) { staging().h2d(dev, host, bytes); }
inline void d2h(void* host, const void* dev, size_t bytes) { staging().d2h(host, dev, bytes); }

// ---- safetensors ----

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end   = 0;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) {
            n *= d;
        }
        return n;
    }
};

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

    const TensorInfo& info(const std::string& name) const {
        auto it = index.find(name);
        if (it == index.end()) {
            throw std::runtime_error("missing tensor " + name + " in " + path);
        }
        return it->second;
    }

    void read_into(const TensorInfo& t, uint64_t off, void* dst, size_t len) {
        f.seekg(static_cast<std::streamoff>(base + t.begin + off));
        f.read(static_cast<char*>(dst), static_cast<std::streamsize>(len));
        if (!f) {
            throw std::runtime_error("short read in " + path);
        }
    }

    std::vector<char> bytes_of(const std::string& name) {
        const auto& t = info(name);
        std::vector<char> out(t.end - t.begin);
        read_into(t, 0, out.data(), out.size());
        return out;
    }

    std::string str(const std::string& name) {  // e.g. ".comfy_quant" blobs; "" when absent
        if (!has(name)) {
            return "";
        }
        auto b = bytes_of(name);
        return std::string(b.begin(), b.end());
    }

    // host copy converted to f32
    std::vector<float> host_f32(const std::string& name) {
        const auto& t = info(name);
        auto raw      = bytes_of(name);
        const int64_t n = t.numel();
        std::vector<float> out(static_cast<size_t>(n));
        const DType dt = dtype_of(t.dtype);
        for (int64_t i = 0; i < n; ++i) {
            switch (dt) {
                case DType::f32: std::memcpy(&out[i], raw.data() + 4 * i, 4); break;
                case DType::bf16: {
                    uint16_t b;
                    std::memcpy(&b, raw.data() + 2 * i, 2);
                    out[i] = bf16_bits_to_f32(b);
                } break;
                case DType::f16: {
                    uint16_t b;
                    std::memcpy(&b, raw.data() + 2 * i, 2);
                    out[i] = f16_bits_to_f32(b);
                } break;
                case DType::s8: out[i] = static_cast<float>(static_cast<int8_t>(raw[i])); break;
                case DType::u8: out[i] = static_cast<float>(static_cast<uint8_t>(raw[i])); break;
                default: throw std::runtime_error("host_f32: unsupported dtype for " + name);
            }
        }
        return out;
    }

    // upload the tensor's bytes as stored into dev (dev must hold t.end - t.begin bytes)
    void upload_raw(const TensorInfo& t, void* dev) {
        staging().h2d_fill(dev, t.end - t.begin, [&](void* pinned, size_t off, size_t len) { read_into(t, off, pinned, len); });
    }
};

// ---- elementwise kernels ----

template <class Ti, class To>
inline void convert(const Ti* in, To* out, int64_t n) {
    if (n <= 0) {
        return;
    }
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) { out[i] = static_cast<To>(static_cast<float>(in[i])); });
}

template <class To>
inline void convert_from(DType ti, const void* in, To* out, int64_t n) {
    switch (ti) {
        case DType::f32: convert(static_cast<const float*>(in), out, n); break;
        case DType::bf16: convert(static_cast<const bf16*>(in), out, n); break;
        case DType::f16: convert(static_cast<const f16*>(in), out, n); break;
        case DType::s8: convert(static_cast<const int8_t*>(in), out, n); break;
        case DType::u8: convert(static_cast<const uint8_t*>(in), out, n); break;
        default: throw std::runtime_error("convert: unsupported source dtype");
    }
}

inline void convert_any(DType ti, const void* in, DType to, void* out, int64_t n) {
    switch (to) {
        case DType::f32: convert_from(ti, in, static_cast<float*>(out), n); break;
        case DType::bf16: convert_from(ti, in, static_cast<bf16*>(out), n); break;
        case DType::f16: convert_from(ti, in, static_cast<f16*>(out), n); break;
        case DType::s8: convert_from(ti, in, static_cast<int8_t*>(out), n); break;
        default: throw std::runtime_error("convert: unsupported target dtype");
    }
}

// Load a safetensors tensor onto the device as `target` (resident or pooled). Same dtype: straight upload.
inline Buf load_tensor(SafeTensors& st, const std::string& name, DType target, bool is_resident = true,
                       std::vector<int64_t>* shape = nullptr) {
    const auto& t   = st.info(name);
    const DType src = dtype_of(t.dtype);
    const int64_t n = t.numel();
    if (static_cast<uint64_t>(n) * dsize(src) != t.end - t.begin) {
        throw std::runtime_error("size mismatch for " + name);
    }
    if (shape != nullptr) {
        *shape = t.shape;
    }
    Buf out(static_cast<size_t>(n) * dsize(target), is_resident);
    if (src == target) {
        st.upload_raw(t, out.get());
        return out;
    }
    Buf tmp = scratch(t.end - t.begin);
    st.upload_raw(t, tmp.get());
    convert_any(src, tmp.get(), target, out.get(), n);
    Q().wait();
    return out;
}

// x [rows, k]: each group of g columns -> x R with R the normalized g-point Hadamard (kron of h4 / 2,
// h4[a][b] = a + b == 3 ? -1 : 1; R symmetric, R^-1 = R). g must be a power of 4 (<= 4096). Computed in f32.
template <class Ti, class To>
inline void rotate_groups(const Ti* x, To* y, int64_t rows, int64_t k, int64_t g) {
    if (g < 4 || g > 4096 || (g & (g - 1)) != 0 || (__builtin_ctzll(static_cast<unsigned long long>(g)) & 1) != 0 || k % g != 0) {
        throw std::runtime_error("rotate: group size must be a power of 4 dividing the row width");
    }
    const int64_t ngroups = rows * (k / g);
    const int wg          = static_cast<int>(g / 4);
    int stages            = 0;
    for (int64_t s = 1; s < g; s *= 4) {
        ++stages;
    }
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(static_cast<size_t>(g)), h);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(ngroups * wg), static_cast<size_t>(wg)), [=](sycl::nd_item<1> it) {
            const int64_t grp = it.get_group(0);
            const int t       = static_cast<int>(it.get_local_id(0));
            const Ti* src     = x + grp * g;  // groups are consecutive in a contiguous [rows, k] matrix
            for (int m = 0; m < 4; ++m) {
                sh[t + m * wg] = static_cast<float>(src[t + m * wg]);
            }
            int64_t stride = 1;
            for (int s = 0; s < stages; ++s) {
                sycl::group_barrier(it.get_group());
                const int64_t b = (t / stride) * stride * 4 + (t % stride);
                float a0 = sh[b], a1 = sh[b + stride], a2 = sh[b + 2 * stride], a3 = sh[b + 3 * stride];
                const float sum = a0 + a1 + a2 + a3;
                // y_b = sum_a h4[a][b] x_a / 2 = (sum - 2 x_{3-b}) / 2
                sh[b]              = (sum - 2.f * a3) * 0.5f;
                sh[b + stride]     = (sum - 2.f * a2) * 0.5f;
                sh[b + 2 * stride] = (sum - 2.f * a1) * 0.5f;
                sh[b + 3 * stride] = (sum - 2.f * a0) * 0.5f;
                stride *= 4;
            }
            sycl::group_barrier(it.get_group());
            To* dst = y + grp * g;
            for (int m = 0; m < 4; ++m) {
                dst[t + m * wg] = static_cast<To>(sh[t + m * wg]);
            }
        });
    });
}

// y[n, k] = T(float(q[n, k]) * s[n])  (int8 weight expansion)
template <class T>
inline void dequant_rows(const int8_t* q, const float* s, T* y, int64_t n, int64_t k) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n * k)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        y[i]            = static_cast<T>(static_cast<float>(q[i]) * s[i / k]);
    });
}

// y = T(float(a) + float(b)), n elements (a += b with torch's per-op rounding)
template <class T>
inline void add_inplace(T* a, const T* b, int64_t n) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) { a[i] = static_cast<T>(static_cast<float>(a[i]) + static_cast<float>(b[i])); });
}

// gu [n, 2f] (gate | up) -> out [n, f] = T(float(T(silu(gate))) * float(up))  (silu(gate) * up with torch's per-op rounding;
// the product is commutative, so this also matches the Qwen order up * silu(gate))
template <class T>
inline void silu_mul(const T* gu, T* out, int64_t n, int64_t f) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n * f)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        const int64_t r = i / f, c = i % f;
        const float z   = static_cast<float>(gu[r * 2 * f + c]);
        const T sl      = static_cast<T>(z / (1.f + sycl::exp(-z)));
        out[i]          = static_cast<T>(static_cast<float>(sl) * static_cast<float>(gu[r * 2 * f + f + c]));
    });
}

// o [H, L, D] (dense, rows [t0, t0 + n) of L) -> y [n, H * D] token-major
template <class T>
inline void heads_to_tokens(const T* o, int64_t H, int64_t L, int64_t D, int64_t t0, int64_t n, T* y) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(n * H * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        const int64_t r = i / (H * D), f = i % (H * D);
        y[i]            = o[((f / D) * L + t0 + r) * D + f % D];
    });
}

// dst[r, c] = src[r * src_ld + c] for r < rows, c < cols (dst dense)
template <class T>
inline void copy2d(const T* src, int64_t src_ld, T* dst, int64_t rows, int64_t cols) {
    Q().parallel_for(sycl::range<1>(static_cast<size_t>(rows * cols)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        dst[i]          = src[(i / cols) * src_ld + i % cols];
    });
}

// xq[m, :] = round(x[m, :] / s[m]), s[m] = amax(|x[m, :]|) / 127 (1 for an all-zero row); one work-group per row.
// x bf16 [M, K], already rotated like the weights (activation quantization of the W8A8 linears).
inline void quantize_rows(const bf16* xp, int64_t M, int64_t K, int8_t* qp, float* sp) {
    constexpr int WG = 256;
    Q().parallel_for(sycl::nd_range<1>(static_cast<size_t>(M * WG), WG), [=](sycl::nd_item<1> it) {
        const int64_t row = it.get_group(0);
        const int lid     = static_cast<int>(it.get_local_id(0));
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

// ---- oneDNN matmul ----

inline dnnl::memory::data_type dnnl_dt(DType t) {
    using dt = dnnl::memory::data_type;
    switch (t) {
        case DType::f32: return dt::f32;
        case DType::bf16: return dt::bf16;
        case DType::f16: return dt::f16;
        case DType::s8: return dt::s8;
        case DType::u8: return dt::u8;
        case DType::s32: return dt::s32;
    }
    return dt::undef;
}

// y[m][n] (+)= sum_k x[m][k] * w(n, k) (+ bias[n]), accumulating in f32 (s32 for s8 x s8) like torch's oneDNN
// matmuls. x[m][k] at x + m * x_rs + k * x_cs; w [N][K] row-major (w_nk) or [K][N]; y[m][n] at y + m * y_ld + n.
// accumulate: y = x w^T + y (oneDNN sum post-op: one rounding, like at::addmm). int8: per-row x_scales [M] and
// per-output-column w_scales [N] applied in the epilogue.
struct MatmulArgs {
    int64_t M = 0, K = 0, N = 0;
    DType xt = DType::bf16;
    const void* x = nullptr;
    int64_t x_rs = -1, x_cs = 1;  // x_rs < 0: K (dense)
    DType wt = DType::bf16;
    const void* w = nullptr;
    bool w_nk = true;
    DType yt = DType::bf16;
    void* y = nullptr;
    int64_t y_ld = -1;  // < 0: N
    DType bt = DType::f32;
    const void* bias = nullptr;
    bool accumulate = false;
    const float* x_scales = nullptr;
    const float* w_scales = nullptr;
};

struct MatmulCache {
    struct Entry {
        dnnl::matmul prim;
        dnnl::memory::desc src, wei, dst, bias, scratch;
        size_t scratch_bytes = 0;
    };
    std::mutex mu;
    std::map<std::string, std::shared_ptr<Entry>> map;
};

inline MatmulCache& matmul_cache() {
    static auto* c = new MatmulCache();  // leaked: no oneDNN teardown after the SYCL runtime at process exit
    return *c;
}

inline void matmul(const MatmulArgs& a) {
    using namespace dnnl;
    using tag        = memory::format_tag;
    const int64_t xr = a.x_rs < 0 ? a.K : a.x_rs;
    const int64_t yl = a.y_ld < 0 ? a.N : a.y_ld;
    char key[512];
    std::snprintf(key, sizeof(key), "%lld,%lld,%lld|%d:%lld:%lld|%d:%d|%d:%lld|%d:%d|%d|%d%d", (long long)a.M, (long long)a.K,
                  (long long)a.N, (int)a.xt, (long long)xr, (long long)a.x_cs, (int)a.wt, (int)a.w_nk, (int)a.yt, (long long)yl,
                  a.bias != nullptr, (int)a.bt, (int)a.accumulate, a.x_scales != nullptr, a.w_scales != nullptr);
    auto& cache = matmul_cache();
    std::shared_ptr<MatmulCache::Entry> e;
    {
        std::lock_guard<std::mutex> lock(cache.mu);
        auto& slot = cache.map[key];
        if (!slot) {
            auto ent = std::make_shared<MatmulCache::Entry>();
            ent->src = memory::desc({a.M, a.K}, dnnl_dt(a.xt), memory::dims{xr, a.x_cs});
            ent->wei = memory::desc({a.K, a.N}, dnnl_dt(a.wt), a.w_nk ? tag::ba : tag::ab);
            ent->dst = memory::desc({a.M, a.N}, dnnl_dt(a.yt), memory::dims{yl, 1});
            primitive_attr attr;
            attr.set_scratchpad_mode(scratchpad_mode::user);
            if (a.x_scales != nullptr) {
                attr.set_scales_mask(DNNL_ARG_SRC, 1 << 0);  // per row of x
            }
            if (a.w_scales != nullptr) {
                attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);  // per output column
            }
            if (a.accumulate) {
                post_ops po;
                po.append_sum(1.f);
                attr.set_post_ops(po);
            }
            dnnl::matmul::primitive_desc pd;
            if (a.bias != nullptr) {
                ent->bias = memory::desc({1, a.N}, dnnl_dt(a.bt), tag::ab);
                pd        = dnnl::matmul::primitive_desc(rt().eng, ent->src, ent->wei, ent->bias, ent->dst, attr);
            } else {
                pd = dnnl::matmul::primitive_desc(rt().eng, ent->src, ent->wei, ent->dst, attr);
            }
            ent->scratch       = pd.scratchpad_desc();
            ent->scratch_bytes = ent->scratch.get_size();
            ent->prim          = dnnl::matmul(pd);
            slot               = ent;
        }
        e = slot;
    }
    auto mem = [&](const memory::desc& md, const void* p) {
        return sycl_interop::make_memory(md, rt().eng, sycl_interop::memory_kind::usm, const_cast<void*>(p));
    };
    std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, mem(e->src, a.x)}, {DNNL_ARG_WEIGHTS, mem(e->wei, a.w)}, {DNNL_ARG_DST, mem(e->dst, a.y)}};
    if (a.bias != nullptr) {
        args[DNNL_ARG_BIAS] = mem(e->bias, a.bias);
    }
    if (a.x_scales != nullptr) {
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = mem(memory::desc({a.M}, memory::data_type::f32, tag::a), a.x_scales);
    }
    if (a.w_scales != nullptr) {
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = mem(memory::desc({a.N}, memory::data_type::f32, tag::a), a.w_scales);
    }
    Buf scr;
    if (e->scratch_bytes > 0) {
        scr                       = scratch(e->scratch_bytes);
        args[DNNL_ARG_SCRATCHPAD] = mem(e->scratch, scr.get());
    }
    e->prim.execute(rt().strm, args);
    // scr returns to the pool here; later work on the in-order queue runs after this matmul
}

// ---- attention ----

// SYCL flash-attention fallback: softmax(q k^T * scale [+ causal mask]) v with f32 scores, online softmax and f32
// accumulation. q [H, Sq, D] at q + h * qh + i * qs + d (likewise k, v with Skv rows); out dense [H, Sq, D].
// causal_q0 >= 0: query i has absolute position causal_q0 + i and sees keys j <= causal_q0 + i.
// One sub-group of 16 lanes per query row, 8 rows per work-group, 16-key tiles staged in local memory.
template <class T, int DPL>  // DPL = D / 16 elements per lane
inline void flash_attention_sycl(const T* q, int64_t qh, int64_t qs, const T* k, int64_t kh, int64_t ks, const T* v, int64_t vh, int64_t vs,
                                 int64_t H, int64_t Sq, int64_t Skv, float scale, int64_t causal_q0, T* out) {
    constexpr int D  = DPL * 16;
    constexpr int NQ = 8;   // query rows (sub-groups) per work-group
    constexpr int KT = 16;  // keys per tile
    const int64_t qblocks = (Sq + NQ - 1) / NQ;
    Q().submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qsm(sycl::range<1>(NQ * D), h);
        sycl::local_accessor<float, 1> ksm(sycl::range<1>(KT * (D + 1)), h);  // +1 pad: lanes read different banks
        sycl::local_accessor<float, 1> vsm(sycl::range<1>(KT * D), h);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(H * qblocks * NQ * 16), NQ * 16),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                           const int64_t grp = it.get_group(0);
                           const int64_t hh  = grp / qblocks;
                           const int64_t i0  = (grp % qblocks) * NQ;
                           auto sg           = it.get_sub_group();
                           const int lane    = static_cast<int>(sg.get_local_linear_id());
                           const int sgi     = static_cast<int>(sg.get_group_linear_id());
                           const int lid     = static_cast<int>(it.get_local_id(0));
                           const int64_t i   = i0 + sgi;
                           const bool valid  = i < Sq;
                           // stage this group's query rows (f32)
                           for (int e = lid; e < NQ * D; e += NQ * 16) {
                               const int64_t r = i0 + e / D;
                               qsm[e]          = r < Sq ? static_cast<float>(q[hh * qh + r * qs + e % D]) : 0.f;
                           }
                           int64_t kend = Skv;
                           if (causal_q0 >= 0) {
                               kend = sycl::min(Skv, causal_q0 + sycl::min(Sq, i0 + NQ));  // last row's limit + 1
                           }
                           const int64_t my_lim = causal_q0 >= 0 ? causal_q0 + i : Skv - 1;  // last visible key
                           float m = -INFINITY, l = 0.f;
                           float acc[DPL];
                           for (int d = 0; d < DPL; ++d) {
                               acc[d] = 0.f;
                           }
                           for (int64_t j0 = 0; j0 < kend; j0 += KT) {
                               sycl::group_barrier(it.get_group());
                               for (int e = lid; e < KT * D; e += NQ * 16) {
                                   const int64_t j = j0 + e / D;
                                   const bool in   = j < Skv;
                                   ksm[(e / D) * (D + 1) + e % D] = in ? static_cast<float>(k[hh * kh + j * ks + e % D]) : 0.f;
                                   vsm[e]          = in ? static_cast<float>(v[hh * vh + j * vs + e % D]) : 0.f;
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
                                   continue;  // nothing visible yet (uniform across the sub-group)
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
                               T* o            = out + (hh * Sq + i) * D;
                               for (int d = 0; d < DPL; ++d) {
                                   o[d * 16 + lane] = static_cast<T>(acc[d] * inv);
                               }
                           }
                       });
    });
}

template <class T>
inline void attention_sycl(const T* q, int64_t qh, int64_t qs, const T* k, int64_t kh, int64_t ks, const T* v, int64_t vh, int64_t vs, int64_t H,
                           int64_t Sq, int64_t Skv, int64_t D, float scale, int64_t causal_q0, T* out) {
    switch (D) {
        case 64: flash_attention_sycl<T, 4>(q, qh, qs, k, kh, ks, v, vh, vs, H, Sq, Skv, scale, causal_q0, out); break;
        case 128: flash_attention_sycl<T, 8>(q, qh, qs, k, kh, ks, v, vh, vs, H, Sq, Skv, scale, causal_q0, out); break;
        case 256: flash_attention_sycl<T, 16>(q, qh, qs, k, kh, ks, v, vh, vs, H, Sq, Skv, scale, causal_q0, out); break;
        default: throw std::runtime_error("attention: head_dim must be 64, 128 or 256 for the SYCL kernel");
    }
}

namespace dg = dnnl::graph;

// oneDNN Graph SDPA partition: score = q k^T (f32) -> * scale (f32 scalar) -> softmax over keys (io-dtype
// probabilities) -> probs v (io dtype). q [1, H, Sq, D], k, v [1, H, Skv, D], any strides with unit stride on D;
// output dense [1, H, Sq, D]. Compiled once per shape/stride set.
class GraphSdpa {
public:
    // Returns false when oneDNN cannot run the shape as one fused kernel (the caller then uses the SYCL kernel).
    bool run(DType t, const void* q, int64_t qh, int64_t qs, const void* k, int64_t kh, int64_t ks, const void* v, int64_t vh, int64_t vs,
             int64_t H, int64_t Sq, int64_t Skv, int64_t D, float scale, void* out, int verbose, const char* tag) {
        if (t != DType::bf16 && t != DType::f16) {
            return false;
        }
        char key[256];
        std::snprintf(key, sizeof(key), "%d|%lld,%lld,%lld,%lld|%lld,%lld|%lld,%lld|%lld,%lld", (int)t, (long long)H, (long long)Sq, (long long)Skv,
                      (long long)D, (long long)qh, (long long)qs, (long long)kh, (long long)ks, (long long)vh, (long long)vs);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            it = cache_.emplace(key, compile(t, qh, qs, kh, ks, vh, vs, H, Sq, Skv, D, verbose, tag)).first;
        }
        Part& p = *it->second;
        if (!p.ok) {
            return false;
        }
        float* sc = scale_ptr(scale);
        std::vector<dg::tensor> ins  = {dg::tensor(p.q, rt().geng, const_cast<void*>(q)), dg::tensor(p.k, rt().geng, const_cast<void*>(k)),
                                        dg::tensor(p.scale, rt().geng, sc), dg::tensor(p.v, rt().geng, const_cast<void*>(v))};
        std::vector<dg::tensor> outs = {dg::tensor(p.out, rt().geng, out)};
        dg::sycl_interop::execute(p.cp, rt().gstrm, ins, outs);
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        cache_.clear();
        scales_.clear();
    }

private:
    struct Part {
        dg::compiled_partition cp;
        dg::logical_tensor q, k, v, scale, out;
        bool ok = false;
    };

    std::mutex mu_;
    std::map<std::string, std::unique_ptr<Part>> cache_;
    std::map<float, Buf> scales_;

    float* scale_ptr(float s) {
        auto& b = scales_[s];
        if (b.empty()) {
            b = resident(sizeof(float));
            Q().memcpy(b.get(), &s, sizeof(float)).wait();
        }
        return b.as<float>();
    }

    static std::unique_ptr<Part> compile(DType t, int64_t qh, int64_t qs, int64_t kh, int64_t ks, int64_t vh, int64_t vs, int64_t H, int64_t Sq,
                                         int64_t Skv, int64_t D, int verbose, const char* tag) {
        using lt   = dg::logical_tensor;
        using ldt  = lt::data_type;
        auto c     = std::make_unique<Part>();
        const auto t0 = std::chrono::steady_clock::now();
        try {
            const ldt io = t == DType::bf16 ? ldt::bf16 : ldt::f16;
            size_t id    = 0;
            c->q         = lt(id++, io, {1, H, Sq, D}, {H * qh, qh, qs, 1});
            c->k         = lt(id++, io, {1, H, Skv, D}, {H * kh, kh, ks, 1});
            auto score   = lt(id++, ldt::f32, {1, H, Sq, Skv}, lt::layout_type::strided);
            dg::op bmm1(id++, dg::op::kind::MatMul, "bmm1");
            bmm1.set_attr<bool>(dg::op::attr::transpose_b, true);
            bmm1.add_inputs({c->q, c->k});
            bmm1.add_outputs({score});
            c->scale    = lt(id++, ldt::f32, std::vector<int64_t>{1}, std::vector<int64_t>{1});
            auto scaled = lt(id++, ldt::f32, {1, H, Sq, Skv}, lt::layout_type::strided);
            dg::op mul(id++, dg::op::kind::Multiply, "scale");
            mul.add_inputs({score, c->scale});
            mul.add_outputs({scaled});
            auto probs = lt(id++, io, {1, H, Sq, Skv}, lt::layout_type::strided);
            dg::op softmax(id++, dg::op::kind::SoftMax, "softmax");
            softmax.set_attr<int64_t>(dg::op::attr::axis, -1);
            softmax.add_inputs({scaled});
            softmax.add_outputs({probs});
            c->v   = lt(id++, io, {1, H, Skv, D}, {H * vh, vh, vs, 1});
            c->out = lt(id++, io, {1, H, Sq, D}, {H * Sq * D, Sq * D, D, 1});
            dg::op bmm2(id++, dg::op::kind::MatMul, "bmm2");
            bmm2.add_inputs({probs, c->v});
            bmm2.add_outputs({c->out});
            dg::graph g(dnnl::engine::kind::gpu);
            g.add_op(bmm1);
            g.add_op(mul);
            g.add_op(softmax);
            g.add_op(bmm2);
            g.finalize();
            auto parts = g.get_partitions();
            if (parts.size() != 1 || !parts[0].is_supported()) {
                std::fprintf(stderr, "[%s] graph SDPA: pattern split into %zu partitions; using the SYCL kernel\n", tag, parts.size());
                return c;
            }
            c->cp                = parts[0].compile({c->q, c->k, c->scale, c->v}, {c->out}, rt().geng);
            const size_t scratch = c->cp.get_scratchpad_logical_tensor().get_mem_size();
            // the fused kernel needs (almost) no scratch; anything near the S x S score matrix means a decomposed path
            const size_t limit = static_cast<size_t>(H) * static_cast<size_t>(Sq) * 64;
            if (scratch > limit && scratch > (size_t(256) << 20)) {
                std::fprintf(stderr, "[%s] graph SDPA: scratchpad %.1f MB looks unfused; using the SYCL kernel\n", tag, scratch / 1048576.0);
                return c;
            }
            c->ok = true;
            if (verbose) {
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "[%s] graph SDPA compiled: H=%lld Sq=%lld Skv=%lld D=%lld, scratchpad %.1f MB, %.2f s\n", tag, (long long)H,
                             (long long)Sq, (long long)Skv, (long long)D, scratch / 1048576.0, s);
            }
        } catch (const dnnl::error& e) {
            std::fprintf(stderr, "[%s] graph SDPA unavailable (%s); using the SYCL kernel\n", tag, e.what());
            c->ok = false;
        }
        return c;
    }
};

inline GraphSdpa& graph_sdpa() {
    static auto* g = new GraphSdpa();  // leaked, see matmul_cache()
    return *g;
}

// Attention dispatcher. prefer_graph: try oneDNN Graph first (non-causal bf16/f16 only).
inline void attention(DType t, const void* q, int64_t qh, int64_t qs, const void* k, int64_t kh, int64_t ks, const void* v, int64_t vh, int64_t vs,
                      int64_t H, int64_t Sq, int64_t Skv, int64_t D, float scale, int64_t causal_q0, void* out, bool prefer_graph, int verbose,
                      const char* tag) {
    if (prefer_graph && causal_q0 < 0 && graph_sdpa().run(t, q, qh, qs, k, kh, ks, v, vh, vs, H, Sq, Skv, D, scale, out, verbose, tag)) {
        return;
    }
    switch (t) {
        case DType::bf16:
            attention_sycl(static_cast<const bf16*>(q), qh, qs, static_cast<const bf16*>(k), kh, ks, static_cast<const bf16*>(v), vh, vs, H, Sq, Skv,
                           D, scale, causal_q0, static_cast<bf16*>(out));
            break;
        case DType::f16:
            attention_sycl(static_cast<const f16*>(q), qh, qs, static_cast<const f16*>(k), kh, ks, static_cast<const f16*>(v), vh, vs, H, Sq, Skv, D,
                           scale, causal_q0, static_cast<f16*>(out));
            break;
        case DType::f32:
            attention_sycl(static_cast<const float*>(q), qh, qs, static_cast<const float*>(k), kh, ks, static_cast<const float*>(v), vh, vs, H, Sq,
                           Skv, D, scale, causal_q0, static_cast<float*>(out));
            break;
        default: throw std::runtime_error("attention: unsupported dtype");
    }
}

// Drop every cached oneDNN object and every cached pool block (between jobs / on release).
inline void release_all() {
    if (!g_rt_started) {
        return;  // never touched the device: nothing to release (and no runtime to start)
    }
    graph_sdpa().clear();
    {
        auto& c = matmul_cache();
        std::lock_guard<std::mutex> lock(c.mu);
        c.map.clear();
    }
    Q().wait();
    staging().release();
    pool().trim();
}

// ---- self-test helpers (comparison against reference outputs dumped from the torch libraries) ----

inline std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("selftest: cannot open " + path);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class T>
inline std::vector<T> read_vec(const std::string& path) {
    auto b = read_file(path);
    std::vector<T> v(b.size() / sizeof(T));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
    return v;
}

inline void write_file(const std::string& path, const void* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
}

inline std::map<std::string, double> read_meta(const std::string& dir) {
    std::ifstream f(dir + "/meta.txt");
    if (!f) {
        throw std::runtime_error("selftest: cannot open " + dir + "/meta.txt");
    }
    std::map<std::string, double> m;
    std::string k;
    double v;
    while (f >> k >> v) {
        m[k] = v;
    }
    return m;
}

// Prints RMS / max errors of `got` against `ref`; returns 0 when the relative RMS is within tol.
inline int compare_report(const char* tag, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] selftest: size mismatch %zu vs %zu\n", tag, got.size(), ref.size());
        return 1;
    }
    double se = 0, sr = 0, maxe = 0, maxr = 0;
    size_t nonfinite = 0, worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            ++nonfinite;
            continue;
        }
        const double e = static_cast<double>(got[i]) - ref[i];
        se += e * e;
        sr += static_cast<double>(ref[i]) * ref[i];
        if (std::fabs(e) > maxe) {
            maxe  = std::fabs(e);
            worst = i;
        }
        maxr = std::max(maxr, std::fabs(static_cast<double>(ref[i])));
    }
    const double n   = static_cast<double>(got.size());
    const double rms = std::sqrt(se / n), rref = std::sqrt(sr / n);
    const double rel = rref > 0 ? rms / rref : rms;
    const bool ok    = nonfinite == 0 && rel <= tol;
    std::fprintf(stderr,
                 "[%s] selftest: %zu values | RMS err %.3e | ref RMS %.3e | relative RMS %.3e | max abs err %.3e at %zu (ref %.4g, got %.4g) | "
                 "ref max %.3e | non-finite %zu | tol %.1e -> %s\n",
                 tag, got.size(), rms, rref, rel, maxe, worst, ref.empty() ? 0.0 : ref[worst], got.empty() ? 0.0 : got[worst], maxr, nonfinite,
                 tol, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace xn
