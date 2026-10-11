#!/usr/bin/env bash
# Native-vs-libtorch self-test for one library (see SELFTEST.md). Runs ON THE GPU: only start it when nothing else
# is using the B70. Three separate processes: gen (CPU only), ref (libtorch build), check (native build).
#
#   SD_XPU_H3_WEIGHTS=... ./selftest.sh h3        [dir]
#   SD_XPU_H3VAE_WEIGHTS=... ./selftest.sh h3vae  [dir]
#   SD_XPU_QWEN21_WEIGHTS=... ./selftest.sh qwen21 [dir]
#
# SD_XPU_MAX_BLOCKS (default 1) limits both builds to the first N blocks; unset it (SD_XPU_MAX_BLOCKS=-1) for the
# full stack. Mode switches (SD_XPU_H3_INT8, SD_XPU_QWEN21_INT8, SD_XPU_H3VAE_DTYPE, LoRA lists) apply to both runs;
# the default dir name includes them so a cached reference is never compared against another mode.
# REF_LIB overrides the reference library (default torchref/libxpu_<model>_torchref.so; with SD_XPU_MAX_BLOCKS=-1 the
# production libtorch .so works too). GEN_OPTS passes options to `xpu_selftest gen` (e.g. "rows=2048").
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=${1:?usage: selftest.sh h3|h3vae|qwen21 [dir]}
export SD_XPU_MAX_BLOCKS=${SD_XPU_MAX_BLOCKS:-1}
mode="b${SD_XPU_MAX_BLOCKS}_h3int8-${SD_XPU_H3_INT8:-0}_qint8-${SD_XPU_QWEN21_INT8:-1}_vae-${SD_XPU_H3VAE_DTYPE:-bf16}"
DIR=${2:-${TMPDIR:-/tmp}/xpu_selftest_${MODEL}_${mode}}
REF=${REF_LIB:-$HERE/torchref/libxpu_${MODEL}_torchref.so}
NATIVE=$HERE/libxpu_${MODEL}.so

[[ -x "$HERE/xpu_selftest" ]] || { echo "build first: $HERE/build_native.sh" >&2; exit 2; }
[[ -f "$REF" ]] || { echo "missing reference $REF (TORCH_VENV=... $HERE/torchref/build_torchref.sh)" >&2; exit 2; }
if [[ ! -f "$DIR/meta.txt" ]]; then
  # shellcheck disable=SC2086
  "$HERE/xpu_selftest" gen "$MODEL" "$DIR" ${GEN_OPTS:-}
fi
if [[ ! -f "$DIR/ref_out.f32" ]]; then
  echo "== reference: $REF"
  "$HERE/xpu_selftest" ref "$MODEL" "$REF" "$DIR"
fi
echo "== native: $NATIVE"
SD_XPU_POOL_VERBOSE=${SD_XPU_POOL_VERBOSE:-1} "$HERE/xpu_selftest" check "$MODEL" "$NATIVE" "$DIR"
