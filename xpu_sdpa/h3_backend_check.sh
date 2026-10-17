#!/usr/bin/env bash
# One tiny MiniMax-H3 Ref2VA job (448x256, 25 frames, 2 steps, Gary shot v01-k03's refs and guides) with every runner
# output dumped (SD_DUMP_DIR), for comparing sd-cli on Vulkan against sd-cli on SYCL with compare_dumps.py.
# The transformer blocks and the video VAE decoder run on the native XPU libraries in both cases, as in production.
#   ./h3_backend_check.sh vulkan OUTDIR      (sd-cli from build-vulkan, Mesa ANV with the B70 patches)
#   ./h3_backend_check.sh sycl OUTDIR        (sd-cli from build-sycl, ggml's SYCL backend)
# Takes the B70 through b70-lease. Writes OUTDIR/{dump/,out.mp4.avi,run.log}.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODE=${1:?vulkan|sycl}; OUT=${2:?outdir}
SHOT=${SHOT:-$HOME/ai/clips/short-stories/gary-ai-takeover/04-video-generation/shots/v01-k03}
mkdir -p "$OUT/dump"; rm -f "$OUT"/dump/*.f32
NATIVE=$HOME/ai/stable-diffusion.cpp-b70-native/xpu_sdpa
export SD_XPU_H3_LIB=$NATIVE/libxpu_h3.so SD_XPU_H3VAE_LIB=$NATIVE/libxpu_h3vae.so SD_XPU_H3_INT8=1
export SD_XPU_H3_WEIGHTS=$HOME/ai/models/minimax-h3/DasiwaMinimaxH3_dasiwaHybridV4Turbo_3281858.safetensors
export SD_XPU_H3_LORA=$HOME/ai/models/minimax-h3/loras/better_motion_h3_lora_v1_500.safetensors@0.5
VAE=$HOME/ai/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors
export SD_XPU_H3VAE_WEIGHTS=$VAE
export SD_DUMP_DIR=$OUT/dump
case $MODE in
  vulkan)
    SD=${SD:-$HERE/build-vulkan/bin/sd-cli}; BACKEND=diffusion=vulkan0,te=vulkan0,vae=vulkan0
    export VK_ICD_FILENAMES=$HOME/ai/mesa-b70/install/share/vulkan/icd.d/intel_icd.x86_64.json
    export SD_VK_XE_FA=1 SD_VK_XE_FA_QCHUNK=8192 GGML_VK_MAX_NODES_PER_SUBMIT=4 ;;
  sycl)
    SD=${SD:-$HERE/build-sycl/bin/sd-cli}; BACKEND=diffusion=SYCL0,te=SYCL0,vae=SYCL0
    export ONEAPI_DEVICE_SELECTOR=level_zero:0 ;;
  *) echo "mode: vulkan|sycl" >&2; exit 2 ;;
esac
args=(-p "$(cat "$SHOT/prompt.txt")")
for f in "$SHOT"/inputs/ref-*.png; do args+=(-r "$f"); done
if [[ -f "$SHOT/inputs/guides.txt" ]]; then
  while read -r spec; do [[ -z "$spec" || "$spec" == \#* ]] || args+=(--guide "$SHOT/inputs/$spec"); done < "$SHOT/inputs/guides.txt"
fi
b70-lease run "h3-$MODE-check" -- "$SD" -M vid_gen \
  --diffusion-model "$HOME/ai/models/minimax-h3/dasiwa-hybridv4turbo-q8_0.gguf" \
  --vae "$VAE" --audio-vae "$HOME/ai/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors" \
  --llm "$HOME/ai/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf" \
  --backend "$BACKEND" --offload-to-cpu --fa --cfg-scale 1 --sampling-method euler --scheduler simple \
  --steps "${STEPS:-2}" --flow-shift 12 -W 448 -H 256 --fps 24 --rng cpu --max-vram 0 \
  --video-frames "${FRAMES:-25}" --seed 2026101701 -o "$OUT/out.mp4" -v "${args[@]}" 2>&1 | tee "$OUT/run.log"
grep -E "ggml_vulkan: Found|to Vulkan0" "$OUT/run.log" | head -3 || true
