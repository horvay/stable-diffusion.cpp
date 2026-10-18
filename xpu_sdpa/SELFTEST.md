# Native XPU libraries: build and validation

`libxpu_h3.so`, `libxpu_h3vae.so` and `libxpu_qwen21.so` now run on SYCL + oneDNN (primitives and the Graph API) over Level Zero, with no libtorch, ATen, c10 or IPEX. Their exported C functions and environment switches are unchanged, so `sd-cli` loads them through the same `SD_XPU_H3_LIB`, `SD_XPU_H3VAE_LIB` and `SD_XPU_QWEN21_LIB` variables.

## Build (compile only, no GPU)

```sh
xpu_sdpa/build_native.sh            # icpx (DPC++ 2025.3) + ~/opt/onednn-sycl-graph (libdnnl_h3)
ldd xpu_sdpa/libxpu_h3.so | grep -Ei 'torch|c10'   # must print nothing
```

The reference libraries for the self-test are built from the original libtorch sources in `torchref/`. They are identical to production except that they honour `SD_XPU_MAX_BLOCKS`:

```sh
TORCH_VENV=~/ai/weeb/.venv-xpu xpu_sdpa/torchref/build_torchref.sh
```

## Self-test (runs on the GPU; only start it when the B70 is idle)

Each library has a self-test mode. When `SD_XPU_H3_SELFTEST=<dir>`, `SD_XPU_H3VAE_SELFTEST=<dir>` or `SD_XPU_QWEN21_SELFTEST=<dir>` is set, the library's `*_init()` loads the weights, runs the inputs stored in `<dir>`, and compares the result with `<dir>/ref_out.f32`, which the libtorch build wrote. It prints the RMS error, reference RMS, relative RMS, the maximum absolute error and where it occurs, and any non-finite values. It returns non-zero when the relative RMS exceeds `SD_XPU_SELFTEST_TOL` (default 0.02). By default the comparison covers one block (`SD_XPU_MAX_BLOCKS=1`) and, for H3 and Qwen, the blocks' output hidden state.

`selftest.sh` runs the three steps, each in its own process:

1. `xpu_selftest gen` (CPU only) writes deterministic random inputs. The model dimensions come from the weights' safetensors header.
2. `xpu_selftest ref` runs the torchref library and writes `ref_out.f32`.
3. `xpu_selftest check` runs the native library with the self-test variable set.

```sh
cd ~/ai/stable-diffusion.cpp-b70-native/xpu_sdpa
SD_XPU_H3_WEIGHTS=~/ai/models/minimax-h3/DasiwaMinimaxH3_dasiwaHybridV4Turbo_3281858.safetensors ./selftest.sh h3
SD_XPU_H3_INT8=1 SD_XPU_H3_WEIGHTS=... ./selftest.sh h3                 # int8 W8A8 fused path
SD_XPU_H3VAE_WEIGHTS=<H3 video VAE .safetensors> ./selftest.sh h3vae
SD_XPU_QWEN21_WEIGHTS=~/ai/weeb/models/QWEN21/noctQAnime_v10_int8_convrot.safetensors ./selftest.sh qwen21
SD_XPU_MAX_BLOCKS=-1 SD_XPU_H3_WEIGHTS=... ./selftest.sh h3              # whole stack instead of one block
```

The Qwen inputs include two causal prefix segments, the second starting at a non-zero offset, so they test the masked attention path.

Expected results: about 1e-3 relative RMS for one bf16 block. Results that differ only in the last bf16 bit are normal, because the native kernels sum in a different order. The int8 paths can be somewhat higher, because a value that falls on a rounding edge can quantize to the neighbouring int8 level. For the full stack, expect the error to grow with depth while staying well under 1e-2 relative RMS. A NaN, a non-finite value, or an error above 2e-2 means a bug.

`SD_XPU_POOL_VERBOSE=1`, which `selftest.sh` sets by default, also prints the pool statistics: live, cached, peak and resident memory.

## Using the new libraries for a render

Only the library paths change:

```sh
N=~/ai/stable-diffusion.cpp-b70-native/xpu_sdpa
export SD_XPU_H3_LIB=$N/libxpu_h3.so SD_XPU_H3VAE_LIB=$N/libxpu_h3vae.so   # H3 video
export SD_XPU_QWEN21_LIB=$N/libxpu_qwen21.so                               # Qwen Image 2.1
```

All the existing variables keep their meaning (`SD_XPU_H3_WEIGHTS`, `SD_XPU_H3_INT8`, `SD_XPU_H3_LORA`, `SD_XPU_H3_CHUNK`, `SD_XPU_H3_TIMING`, `SD_XPU_H3VAE_WEIGHTS`, `SD_XPU_H3VAE_DTYPE`, `SD_XPU_QWEN21_WEIGHTS`, `SD_XPU_QWEN21_LORA`, `SD_XPU_QWEN21_INT8`, ...). These are new or changed:

| Variable | Meaning |
| --- | --- |
| `SD_XPU_H3_SDPA` | `graph` (the default) uses oneDNN Graph fused SDPA. `sycl` forces the SYCL flash-attention kernel. The old `torch` value now means graph. |
| `SD_XPU_H3_SDPA_CHECK` / `_TOL` | The first graph SDPA call at each sequence length is checked against the SYCL kernel on 256 query rows. |
| `SD_XPU_H3VAE_SDPA`, `SD_XPU_QWEN21_SDPA` | `sycl` forces the SYCL kernel. |
| `SD_XPU_POOL_CACHE_MB` | Default 1024. The most freed device memory the pool keeps for reuse. |
| `SD_XPU_POOL_LIMIT_MB` | Default 0, meaning off. A hard cap on live plus cached pool memory. Going over it fails with a clear error instead of a driver out-of-memory. |
| `SD_XPU_POOL_VERBOSE=1` | Prints pool statistics after each call and on release. |
| `SD_XPU_MAX_BLOCKS=N` | Debug and self-test only. Runs the first N blocks. |
| `SD_XPU_H3VAE_DEVICE=cpu` | No longer supported: there is no CPU path without torch. |

## Memory model

Weights are direct USM device allocations. The `*_release()` calls free them all: `xpu_h3_release()`, `xpu_h3vae_release(0)` and `xpu_qwen21_release()`. Every other device buffer comes from `xn::Pool` in `xpu_native.hpp`:

- **Cached blocks are bounded.** The pool caches freed blocks only up to `SD_XPU_POOL_CACHE_MB`.
- **Everything is returned after each call.** At the end of every `xpu_h3_blocks`, `xpu_h3vae_decode` and `xpu_qwen21_blocks` call, the pool returns all cached blocks to the driver.
- **oneDNN memory goes through the pool too.** oneDNN primitive scratchpads use user mode and come from the pool. oneDNN Graph allocations use a pool-backed allocator.
- **Release clears everything.** `*_release()` also drops the primitive and compiled-partition caches and the two 64 MB pinned staging buffers.

After `xpu_h3_release()`, the H3 library holds no device memory. This is unlike the torch caching allocator, which kept about 22 GB after a long shot.
