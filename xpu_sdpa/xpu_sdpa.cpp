// Scaled dot-product attention on an Intel XPU through libtorch (oneDNN / SYCL-TLA kernels).
// sd.cpp dlopens this library and calls it from a CPU-side ggml custom op, so the heavy attention
// runs on Intel's Level Zero stack (256-GRF kernels, ~95 TFLOPS on Arc Pro B70) instead of Vulkan.
//
// Layouts: q, k, v are contiguous f16, head-major [heads][len][dim]; out is f16 [len_q][heads][dim]
// (ggml's flash-attention output order).
//
// Host<->device copies go through pinned staging buffers that are reused across calls; the copies
// into and out of them are split across threads.

#include <ATen/ATen.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/core/Device.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

namespace {

void parallel_copy(void* dst, const void* src, size_t bytes) {
    const size_t n_threads = std::max<size_t>(1, std::min<size_t>(16, bytes / (64u << 20)));
    if (n_threads == 1) {
        std::memcpy(dst, src, bytes);
        return;
    }
    std::vector<std::thread> threads;
    const size_t chunk = (bytes + n_threads - 1) / n_threads;
    for (size_t t = 0; t < n_threads; ++t) {
        const size_t begin = t * chunk;
        if (begin >= bytes) {
            break;
        }
        const size_t len = std::min(chunk, bytes - begin);
        threads.emplace_back([=] {
            std::memcpy(static_cast<char*>(dst) + begin, static_cast<const char*>(src) + begin, len);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}

// Pinned host tensor that grows to the largest request and is reused.
at::Tensor pinned(at::Tensor& slot, int64_t numel) {
    if (!slot.defined() || slot.numel() < numel) {
        slot = at::empty({numel}, at::TensorOptions().dtype(at::kHalf).pinned_memory(true));
    }
    return slot.narrow(0, 0, numel);
}

at::Tensor g_q_host, g_k_host, g_v_host, g_o_host;

}  // namespace

extern "C" int xpu_sdpa_f16(const void* q,
                            const void* k,
                            const void* v,
                            void* out,
                            int64_t heads,
                            int64_t len_q,
                            int64_t len_k,
                            int64_t kv_heads,
                            int64_t dim,
                            double scale,
                            int verbose) {
    try {
        using clock    = std::chrono::steady_clock;
        const auto t0  = clock::now();
        const c10::Device xpu(c10::DeviceType::XPU, 0);
        const int64_t nq = heads * len_q * dim;
        const int64_t nk = kv_heads * len_k * dim;

        auto hq = pinned(g_q_host, nq);
        auto hk = pinned(g_k_host, nk);
        auto hv = pinned(g_v_host, nk);
        parallel_copy(hq.data_ptr(), q, nq * 2);
        parallel_copy(hk.data_ptr(), k, nk * 2);
        parallel_copy(hv.data_ptr(), v, nk * 2);
        auto tq = hq.to(xpu, /*non_blocking=*/true).view({1, heads, len_q, dim});
        auto tk = hk.to(xpu, true).view({1, kv_heads, len_k, dim});
        auto tv = hv.to(xpu, true).view({1, kv_heads, len_k, dim});
        const auto t1 = clock::now();

        auto o  = at::scaled_dot_product_attention(tq, tk, tv, {}, 0.0, false, scale, kv_heads != heads);
        auto oc = o.squeeze(0).transpose(0, 1).contiguous().view({nq});  // [Lq, H, D] f16
        auto ho = pinned(g_o_host, nq);
        ho.copy_(oc);  // blocking device -> pinned host
        const auto t2 = clock::now();
        parallel_copy(out, ho.data_ptr(), nq * 2);
        const auto t3 = clock::now();
        if (verbose) {
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            std::fprintf(stderr, "[xpu_sdpa] H=%lld Lq=%lld Lk=%lld D=%lld stage+upload %.1f ms, attn+download %.1f ms, unstage %.1f ms\n",
                         (long long)heads, (long long)len_q, (long long)len_k, (long long)dim,
                         ms(t0, t1), ms(t1, t2), ms(t2, t3));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[xpu_sdpa] error: %s\n", e.what());
        return 1;
    }
}
