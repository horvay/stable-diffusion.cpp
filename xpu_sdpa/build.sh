#!/usr/bin/env bash
# Build the Intel XPU helper libraries that sd.cpp dlopens:
#   libxpu_h3vae.so - MiniMax-H3 video VAE decoder on the XPU (SD_XPU_H3VAE_LIB)
#   libxpu_sdpa.so  - attention-only offload (SD_XPU_SDPA_LIB)
#   (libxpu_h3.so, the MiniMax-H3 blocks, and libxpu_qwen21.so, the Qwen Image 2.1 blocks, have SYCL kernels
#    and build with build_icpx.sh instead)
#
# Needs a Python venv with PyTorch for XPU, e.g.:
#   python -m venv .venv-xpu && .venv-xpu/bin/pip install torch --index-url https://download.pytorch.org/whl/xpu
# Usage: TORCH_VENV=/path/to/.venv-xpu ./build.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=${TORCH_VENV:?set TORCH_VENV to a venv with torch+xpu}
TORCH=$("$VENV/bin/python" -c 'import os, torch; print(os.path.dirname(torch.__file__))')
SITE=$(dirname "$TORCH")
LIBDIR=$(cd "$SITE/../.." && pwd)  # venv lib/ (libsycl, MKL)

for name in ${XPU_LIBS:-xpu_h3vae xpu_sdpa}; do
  g++ -O2 -std=c++17 -fPIC -shared -pthread -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
    "$HERE/$name.cpp" -o "$HERE/lib$name.so.new" \
    -L"$TORCH/lib" -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu \
    -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$LIBDIR"
  # rename so a running sd-cli keeps its mapped copy
  mv "$HERE/lib$name.so.new" "$HERE/lib$name.so"
  echo "built $HERE/lib$name.so"
done
