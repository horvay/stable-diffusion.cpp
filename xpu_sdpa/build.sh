#!/usr/bin/env bash
# Build the Intel XPU helper libraries that sd.cpp dlopens:
#   libxpu_h3.so   - all MiniMax-H3 transformer blocks on the XPU (SD_XPU_H3_LIB)
#   libxpu_sdpa.so - attention-only offload (SD_XPU_SDPA_LIB)
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

for name in xpu_h3 xpu_sdpa; do
  g++ -O2 -std=c++17 -fPIC -shared -pthread -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
    "$HERE/$name.cpp" -o "$HERE/lib$name.so.new" \
    -L"$TORCH/lib" -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu \
    -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$LIBDIR"
  # rename so a running sd-cli keeps its mapped copy
  mv "$HERE/lib$name.so.new" "$HERE/lib$name.so"
  echo "built $HERE/lib$name.so"
done
