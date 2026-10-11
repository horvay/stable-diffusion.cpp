#!/usr/bin/env bash
# DEPRECATED for xpu_h3 / xpu_qwen21: those sources are native SYCL + oneDNN now and build with build_native.sh.
# (Kept for building torch-based experiments; torchref/build_torchref.sh builds the libtorch self-test references.)
# Build SYCL/oneDNN helper libraries with Intel's DPC++ compiler (icpx); the version must match the venv's
# SYCL runtime (torch 2.11+xpu ships libsycl.so.8 = DPC++ 2025.3.2).
#   ONEAPI=~/opt/oneapi-2025.3/opt/intel/oneapi   DPC++ 2025.3.2 unpacked from Intel's APT packages
#   ONEDNN=~/opt/onednn-sycl                      oneDNN built with ONEDNN_GPU_RUNTIME=SYCL
# Usage: TORCH_VENV=/path/to/.venv-xpu ./build_icpx.sh [src.cpp ...]   (default: xpu_qwen21.cpp xpu_h3.cpp)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=${TORCH_VENV:?set TORCH_VENV to a venv with torch+xpu}
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
ONEDNN=${ONEDNN:-$HOME/opt/onednn-sycl}
CMPLR=$ONEAPI/compiler/2025.3
TORCH=$("$VENV/bin/python" -c 'import os, torch; print(os.path.dirname(torch.__file__))')
LIBDIR=$(cd "$(dirname "$TORCH")/../.." && pwd)
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib:$ONEAPI/umf/1.0/lib:$ONEAPI/tcm/1.4/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

srcs=("$@")
[[ ${#srcs[@]} -gt 0 ]] || { echo "build_icpx.sh: pass the torch-based sources to build (the block libraries use build_native.sh)" >&2; exit 2; }
for src in "${srcs[@]}"; do
  name=$(basename "${src%.cpp}")
  out="$(dirname "$src")/lib$name.so"
  [[ "$name" == *bench* || "$name" == *test* ]] && out="$(dirname "$src")/$name"
  shared=(-shared -fPIC)
  [[ "$out" == *.so ]] || shared=()
  # spir64_gen AOT for Battlemage G31 (Arc Pro B70) plus spir64 for JIT on other devices
  icpx -O2 -std=c++17 -static-intel -fsycl -fsycl-targets=spir64_gen,spir64 -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
    "${shared[@]}" -D_GLIBCXX_USE_CXX11_ABI=1 -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING \
    -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$ONEDNN/include" \
    "$src" -o "$out.new" \
    -L"$TORCH/lib" -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -L"$ONEDNN/lib" -ldnnl \
    -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$LIBDIR" -Wl,-rpath,"$ONEDNN/lib"
  mv "$out.new" "$out"
  echo "built $out"
done
