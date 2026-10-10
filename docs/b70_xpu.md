# MiniMax-H3 on Intel Arc Pro B70 (Xe2)

This branch makes MiniMax-H3 video generation practical on a single Arc Pro B70
(32 GB, Battlemage G31) under Linux. On a 1344x768, 107-frame Ref2VA job with
~50k tokens, one denoising step went from ~570 s to ~71 s.

| Change | Effect at H3 shapes |
| --- | --- |
| Intel prefill flash attention (llama.cpp PR #29357) + mask-less variant | Vulkan FA 3.5 -> ~18 TFLOPS; no more 5 s xe job-timeout "device lost" |
| Token-chunked MLP / QKV (`SD_H3_TOKEN_CHUNK`) | no tensor above the 4 GiB Vulkan buffer limit, so nothing falls back to CPU |
| Transformer blocks on the XPU through libtorch (`SD_XPU_H3_LIB`) | oneDNN bf16 matmuls ~100-140 TFLOPS, SDPA ~95 TFLOPS |
| Keyframe guides with Ref2VA (`--guide PATH@FRAME[@STRENGTH]`, `--guide-audio`) | ComfyUI `MiniMaxH3AddGuide` equivalent |

## Build

```sh
cmake -B build-vulkan -DSD_VULKAN=ON -DSD_WEBM=ON -DSD_WEBP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --target sd-cli -j
TORCH_VENV=/path/to/venv-with-torch-xpu xpu_sdpa/build.sh
```

Optional: Mesa ANV with MR !43384 (DPAS register coalescing) roughly doubles
the Vulkan FA kernels; build it ANV-only into a prefix and point
`VK_ICD_FILENAMES` at its `intel_icd.x86_64.json`.

## Run (transformer blocks on the XPU)

```sh
export SD_XPU_H3_LIB=$PWD/xpu_sdpa/libxpu_h3.so
export SD_XPU_H3_WEIGHTS=/models/DasiwaMinimaxH3_dasiwaHybrid4turboV1.safetensors  # ComfyUI int8 convrot checkpoint
sd-cli -M vid_gen --diffusion-model <same model, any format> --vae ... --audio-vae ... --llm ... \
  --backend diffusion=vulkan0,te=vulkan0,vae=vulkan0 --offload-to-cpu --fa --max-vram 0 \
  -r ref1.png -r ref2.png --ref-video motion_frames/ --guide opening.png@0 --guide end.png@93 \
  --cfg-scale 1 --sampling-method euler --scheduler simple --steps 4 --flow-shift 12 \
  -W 1344 -H 768 --video-frames 107 -o out.mp4
```

sd.cpp keeps the embeddings, token refiner, RoPE table, final layer, text
encoder and VAE on Vulkan. The XPU weights load lazily at the first step;
`release_idle_device_memory()` runs before sampling and `xpu_h3_release()`
after it, so both runtimes can share the 32 GB.

## Notes and limits

- Keep `--max-vram 0` with `SD_XPU_H3_LIB` so Vulkan budgets against live free memory.
- Ref2VA reference videos are VAE-encoded in one piece; keep them small (e.g. 448x256).
- VAE decode tiles above 256 px or overlap below 0.25 produce streaks with the H3 VAE.
- `SD_VAE_DECODE_TILE=WxH:overlap` overrides decode tiling (default off).
- `SD_LOG_UNSUPPORTED_OPS=1` lists ops the Vulkan backend rejects (they run on CPU).
- Without `SD_XPU_H3_LIB`, `SD_VK_XE_FA=1` + `GGML_VK_MAX_NODES_PER_SUBMIT=4` give the best
  pure-Vulkan path; `SD_XPU_SDPA_LIB` offloads attention only.

## H3 speed-ups (all opt-in)

| Switch | What it does |
| --- | --- |
| `SD_XPU_H3VAE_LIB=xpu_sdpa/libxpu_h3vae.so` + `SD_XPU_H3VAE_WEIGHTS=<video VAE .safetensors>` | Each 256 px decode tile (post_quant_conv, 36 decoder blocks, norm_out, proj_out, unpatchify) runs through libtorch on the XPU instead of the Vulkan graph. Tiling, temporal windows and blending are unchanged. Weights load on the first tile and stay resident across tiles and windows (no per-window restaging). |
| `SD_XPU_H3VAE_DTYPE=bf16` (default) / `fp16` / `f32` | dtype of the decoder linears and attention; residual stream, norms and proj_out stay f32. CPU reference check on a small tile: bf16 60 dB, fp16 78 dB PSNR against f32. |
| `sd-cli --multi-jobs FILE` | Several vid_gen jobs in one process: FILE has one JSON array of extra arguments per line (prompt, refs, guides, frames, seed, `-o`), appended to the command line. MiniMax-H3 prepares every job's reference latents and conditioning first, frees the text encoder (staged and host copies), then samples and decodes the jobs in order. Each job writes the same files a single run would. With `SD_XPU_H3VAE_LIB` the block weights also stay on the XPU between jobs (`SD_MULTI_KEEP_XPU_BLOCKS=0/1` overrides) and the VAE weights park in pinned host memory while sampling. |
| `SD_XPU_H3_INT8=1` (libxpu_h3.so from `build_icpx.sh`) | W8A8 transformer blocks like libxpu_qwen21: the checkpoint's int8 convrot weights as stored, oneDNN s8 GEMMs with fused scales, rotate+quantize fused into the RMSNorm/modulation, attention-output and SwiGLU kernels (`xpu_h3_kernels.hpp`). `SD_XPU_H3_LORA` stays exact for the base weights: A R is quantized per row and appended to the weight as extra output rows, and the bf16 (x A^T) B^T term is added after the GEMM. `SD_XPU_H3_INT8_UNFUSED=1` uses torch rotate + a quantize kernel instead of the fused kernels. Changes outputs slightly (activation quantization). |

Build: `TORCH_VENV=... xpu_sdpa/build.sh` (libxpu_h3vae, libxpu_sdpa) and `TORCH_VENV=... xpu_sdpa/build_icpx.sh`
(libxpu_qwen21, libxpu_h3; libxpu_h3 now needs icpx because of the int8 kernels).

# Qwen Image 2.1 on Intel Arc Pro B70 (XPU blocks)

`SD_XPU_QWEN21_LIB=xpu_sdpa/libxpu_qwen21.so` runs all 32 Qwen Image 2.1 transformer blocks on the XPU;
Vulkan keeps the text encoder, VAE, embeddings and final layer, and skips its copy of the block weights.
At 1024x1536, 25 steps, CFG 3: Vulkan 8.4 s/step (226 s/image) -> XPU 1.4 s/step (~43 s/image).

| Mode | Per model call | VRAM | Notes |
| --- | --- | --- | --- |
| `SD_XPU_QWEN21_INT8=0` | 1.18 s | 22.7 GiB | dense bf16 weights, oneDNN bf16 GEMM ~150 TFLOPS |
| int8, `SD_XPU_QWEN21_UNFUSED=1` | 0.99 s | 16.1 GiB | W8A8: oneDNN s8 GEMM with fused scales ~280 TOPS |
| int8 (default) | 0.67 s | 15.8 GiB | plus fused SYCL kernels (`xpu_qwen21_kernels.hpp`) |

- Weights: a ComfyUI int8 "convrot" checkpoint (`SD_XPU_QWEN21_WEIGHTS`), used as stored. Each linear's
  input is rotated by the 256-group Hadamard and quantized per row inside the kernel that produces it
  (gated residual + LayerNorm + modulation, attention output, SwiGLU), so activations are read once.
- LoRAs: `SD_XPU_QWEN21_LORA="a.safetensors@0.8,b.safetensors"` (PEFT, `.default`, or kohya keys; split
  `img_mlp.gate_layer`/`img_mlp.proj` map into the fused `gate_up`), merged in the rotated domain and
  requantized at load, so they cost nothing per step.
- Weights load on a background thread when the model is created (`xpu_qwen21_prefetch`).
- `SD_XPU_QWEN21_TIMING=1` prints per-block times; `SD_XPU_QWEN21_VERBOSE=1` per-call times.
- VAE: `--vae-tile-size 384 --vae-tile-overlap 0.125` decodes in 4.8 s vs 8.5 s at the 0.5 default with
  no visible seams; untiled decode is far slower (52 s).

Build (DPC++ must match torch's SYCL runtime: torch 2.11+xpu ships libsycl.so.8 = DPC++ 2025.3.2):

```sh
# DPC++ 2025.3.2 without root: unpack intel-oneapi-{dpcpp-cpp,compiler-*,openmp,tbb,tcm,umf}-2025.3 debs
# from https://apt.repos.intel.com/oneapi into ~/opt/oneapi-2025.3
# oneDNN with the SYCL GPU runtime (matmul/reorder, Xe2), Intel runtime libs linked statically:
CC=icx CXX=icpx cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt/onednn-sycl \
  -DDNNL_CPU_RUNTIME=NONE -DDNNL_GPU_RUNTIME=SYCL -DDNNL_GPU_VENDOR=INTEL -DONEDNN_BUILD_GRAPH=OFF \
  -DONEDNN_ENABLE_PRIMITIVE="MATMUL;REORDER" -DONEDNN_ENABLE_PRIMITIVE_GPU_ISA=XE2 \
  -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DCMAKE_SHARED_LINKER_FLAGS=-static-intel
TORCH_VENV=/path/to/venv-with-torch-xpu xpu_sdpa/build_icpx.sh
```

Tried and dropped: torch's `_int_mm` with separate scale passes (slower than bf16), `_weight_int8pack_mm`
(pathologically slow on XPU), and torch's SYCL-TLA flash attention, which is allowlisted to BMG G21;
unlocked for G31 it runs correctly but no faster than oneDNN's micro_sdpa (~98 TFLOPS).
