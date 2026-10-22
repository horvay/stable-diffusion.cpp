#!/usr/bin/env bash
# Build xpu_music3 (MiniMax Music 3 on the B70, SYCL + oneDNN, no torch / ggml) with Intel's DPC++ compiler.
# The prompt tokenizer is sd.cpp's HF tokenizer (src/tokenizers/hf_tokenizer.cpp + oniguruma + utf8proc), built
# here with a small core/util.h stand-in (music3_shim/) so nothing else from sd.cpp is linked. Needs the sd-cli build
# first (../build_sycl.sh), for oniguruma's generated config.h.
#   ./build_music3.sh                      build xpu_music3
#   ./build_music3.sh --selftest REF_DIR   (uses the B70: run it under b70-lease) then compare each stage with ref_music3.py's CPU dumps in REF_DIR/{dav,cond,dit,ar}
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SD=$(dirname "$HERE")
SD_BUILD=${SD_BUILD:-$SD/build-sycl}
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
ONEDNN=${ONEDNN:-$HOME/opt/onednn-sycl-graph}
DNNL_LIB=${DNNL_LIB:-dnnl_h3}
CMPLR=$ONEAPI/compiler/2025.3
UMF=$ONEAPI/umf/1.0/lib
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib:$UMF${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
B=$HERE/build-music3-tok
mkdir -p "$B"
if [[ ! -f $B/libmusic3_tok.a ]]; then
  T=$SD/thirdparty
  ONIG_CFG=$B/onig
  mkdir -p "$ONIG_CFG"
  cp "$SD_BUILD/thirdparty/oniguruma/config.h" "$ONIG_CFG/config.h"
  for c in ascii regcomp regenc regerror regexec regparse st unicode unicode_fold1_key unicode_fold2_key unicode_fold3_key unicode_unfold_key utf8; do
    gcc -O2 -fPIC -DONIG_STATIC -I"$T/oniguruma" -I"$ONIG_CFG" -c "$T/oniguruma/$c.c" -o "$B/onig_$c.o"
  done
  gcc -O2 -fPIC -DUTF8PROC_STATIC -c "$T/utf8proc/utf8proc.c" -o "$B/utf8proc.o"
  for f in tokenizers/hf_tokenizer tokenizers/tokenizer core/regex; do
    g++ -O2 -fPIC -std=c++17 -DONIG_STATIC -DUTF8PROC_STATIC -I"$HERE/music3_shim" -I"$SD/src" -I"$T" -I"$T/oniguruma" -I"$T/utf8proc" \
      -c "$SD/src/$f.cpp" -o "$B/$(basename $f).o"
  done
  ar rcs "$B/libmusic3_tok.a" "$B"/*.o
fi
nice -n 10 icpx -O2 -std=c++17 -static-intel -fsycl -fsycl-targets=spir64_gen,spir64 -Xsycl-target-backend=spir64_gen "-device bmg-g31" \
  -fp-model=precise -fno-sycl-id-queries-fit-in-int -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING \
  -I"$ONEDNN/include" -I"$HERE" -I"$HERE/music3_shim" -I"$SD/src" \
  "$HERE/xpu_music3.cpp" -o "$HERE/xpu_music3.new" "$B/libmusic3_tok.a" \
  -L"$ONEDNN/lib" -l"$DNNL_LIB" -L"$UMF" -Wl,--no-as-needed -lumf -Wl,--as-needed \
  -Wl,-rpath,"$CMPLR/lib:$UMF:$ONEDNN/lib"
mv "$HERE/xpu_music3.new" "$HERE/xpu_music3"
echo "built $HERE/xpu_music3"
if [[ ${1:-} == --selftest ]]; then
  for t in dav cond dit ar; do echo "== $t"; "$HERE/xpu_music3" --selftest $t "$2/$t" 2>&1 | grep -v '^\s*$' | tail -15; done
fi
