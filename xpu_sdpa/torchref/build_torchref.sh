#!/usr/bin/env bash
# Build the libtorch REFERENCE libraries for the native self-test (../SELFTEST.md). These are the original
# libtorch sources (production h3-speedups) plus SD_XPU_MAX_BLOCKS; they are validation artifacts only and are
# never used for rendering. Same toolchain as the production build (../build.sh, ../build_icpx.sh):
#   libxpu_h3_torchref.so, libxpu_qwen21_torchref.so   icpx + torch + oneDNN (ONEDNN=~/opt/onednn-sycl)
#   libxpu_h3vae_torchref.so                           g++ + torch
# Usage: TORCH_VENV=~/ai/weeb/.venv-xpu ./build_torchref.sh [h3|h3vae|qwen21 ...]   (default: all three)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=${TORCH_VENV:?set TORCH_VENV to a venv with torch+xpu}
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
ONEDNN=${ONEDNN:-$HOME/opt/onednn-sycl}
CMPLR=$ONEAPI/compiler/2025.3
TORCH=${TORCH_DIR:-$("$VENV/bin/python" -c 'import os, torch; print(os.path.dirname(torch.__file__))')}
LIBDIR=$(cd "$(dirname "$TORCH")/../.." && pwd)
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib:$ONEAPI/umf/1.0/lib:$ONEAPI/tcm/1.4/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

names=("$@")
[[ ${#names[@]} -gt 0 ]] || names=(h3 h3vae qwen21)
for name in "${names[@]}"; do
  src="$HERE/xpu_$name.cpp"
  out="$HERE/libxpu_${name}_torchref.so"
  if [[ "$name" == h3vae ]]; then
    g++ -O2 -std=c++17 -fPIC -shared -pthread -D_GLIBCXX_USE_CXX11_ABI=1 \
      -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
      "$src" -o "$out.new" \
      -L"$TORCH/lib" -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu \
      -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$LIBDIR"
  else
    icpx -O2 -std=c++17 -static-intel -fsycl -fsycl-targets=spir64_gen,spir64 -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
      -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING \
      -I"$ONEDNN/include" -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
      "$src" -o "$out.new" \
      -L"$TORCH/lib" -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu -L"$ONEDNN/lib" -ldnnl \
      -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$LIBDIR" -Wl,-rpath,"$ONEDNN/lib"
  fi
  mv "$out.new" "$out"
  echo "built $out"
done
