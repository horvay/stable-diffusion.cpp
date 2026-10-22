#!/usr/bin/env bash
# One small Qwen Image 2.1 edit-mode still (768x512, 4 steps, one -r reference, so the Qwen3-VL vision tower, the VAE
# encoder and decoder all run) through ~/b70/tools/b70-image.sh, with every runner output dumped (SD_DUMP_DIR), for
# comparing sd-cli on Vulkan against sd-cli on SYCL with compare_dumps.py. The transformer blocks run on the native
# libxpu_qwen21.so in both cases.
#   ./qwen_backend_check.sh vulkan|sycl OUTDIR
# Takes the B70 through b70-lease. Writes OUTDIR/{dump/,out.png,out.log}.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODE=${1:?vulkan|sycl}; OUT=${2:?outdir}
REF=${REF:?REF=<a reference image> is needed}
mkdir -p "$OUT/dump"; rm -f "$OUT"/dump/*.f32
export SD_DUMP_DIR=$OUT/dump
export SD_XPU_QWEN21_LIB=$HOME/ai/stable-diffusion.cpp-b70-native/xpu_sdpa/libxpu_qwen21.so
case $MODE in
  vulkan) export SD_CLI=$HERE/build-vulkan/bin/sd-cli ;;
  sycl) export SD_CLI=$HERE/build-sycl/bin/sd-cli ONEAPI_DEVICE_SELECTOR=level_zero:0 ;;
  *) echo "mode: vulkan|sycl" >&2; exit 2 ;;
esac
b70-lease run "qwen-$MODE-check" -- ~/b70/tools/b70-image.sh -m qwen --xpu -W 768 -H 512 --steps "${STEPS:-4}" --seed 2026101701 \
  -r "$REF" -p "An anime illustration of the man in image 1 waving at the viewer in a sunny kitchen" -o "$OUT/out.png"
grep -E "ggml_vulkan: Found|to Vulkan0" "$OUT/out.log" | head -3 || true
