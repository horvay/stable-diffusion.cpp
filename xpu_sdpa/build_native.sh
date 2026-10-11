#!/usr/bin/env bash
# Build the native XPU libraries (SYCL + oneDNN on Level Zero, no libtorch) with Intel's DPC++ compiler:
#   libxpu_h3.so      MiniMax-H3 transformer blocks   (SD_XPU_H3_LIB)
#   libxpu_h3vae.so   MiniMax-H3 video VAE decoder    (SD_XPU_H3VAE_LIB)
#   libxpu_qwen21.so  Qwen Image 2.1 transformer blocks (SD_XPU_QWEN21_LIB)
# and xpu_selftest (plain g++ driver for the self-test, see SELFTEST.md).
#
#   ONEAPI=~/opt/oneapi-2025.3/opt/intel/oneapi   DPC++ 2025.3 (libsycl.so.8, Level Zero UR adapters, UMF)
#   ONEDNN=~/opt/onednn-sycl-graph                oneDNN 3.13 built with ONEDNN_GPU_RUNTIME=SYCL and the Graph API
#   DNNL_LIB=dnnl_h3                              its library name (the local SDPA-patched build)
# Usage: ./build_native.sh [src.cpp ...]   (default: all three libraries + xpu_selftest). One compile at a time.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
ONEDNN=${ONEDNN:-$HOME/opt/onednn-sycl-graph}
DNNL_LIB=${DNNL_LIB:-dnnl_h3}
CMPLR=$ONEAPI/compiler/2025.3
UMF=$ONEAPI/umf/1.0/lib
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib:$UMF${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

srcs=("$@")
[[ ${#srcs[@]} -gt 0 ]] || srcs=("$HERE/xpu_h3.cpp" "$HERE/xpu_h3vae.cpp" "$HERE/xpu_qwen21.cpp" "$HERE/xpu_selftest.cpp")
for src in "${srcs[@]}"; do
  name=$(basename "${src%.cpp}")
  if [[ "$name" == xpu_selftest ]]; then
    # host-only driver: dlopens the libraries, no SYCL / oneDNN / torch at link time
    g++ -O2 -std=c++17 "$src" -o "$HERE/xpu_selftest.new" -ldl
    mv "$HERE/xpu_selftest.new" "$HERE/xpu_selftest"
    echo "built $HERE/xpu_selftest"
    continue
  fi
  out="$HERE/lib$name.so"
  # spir64_gen AOT for Battlemage G31 (Arc Pro B70) plus spir64 for JIT on other devices.
  # -fp-model=precise: no fast-math (the kernels rely on IEEE infinities and on the torch path's op order).
  # libumf is linked directly so the Level Zero UR adapter (dlopened by the UR loader) finds it already loaded.
  icpx -O2 -std=c++17 -static-intel -fsycl -fsycl-targets=spir64_gen,spir64 -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    -shared -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -fp-model=precise -fno-sycl-id-queries-fit-in-int \
    -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING -I"$ONEDNN/include" -I"$HERE" \
    "$src" -o "$out.new" \
    -L"$ONEDNN/lib" -l"$DNNL_LIB" -L"$UMF" -Wl,--no-as-needed -lumf -Wl,--as-needed \
    -Wl,-rpath,"$CMPLR/lib:$UMF:$ONEDNN/lib"
  # rename so a running process keeps its mapped copy
  mv "$out.new" "$out"
  echo "built $out"
done
