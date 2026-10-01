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
