#!/usr/bin/env python3
"""Compare two SD_DUMP_DIR dumps (e.g. Vulkan vs SYCL sd-cli on the same job). CPU only, numpy.
Files pair up by name (<NNNN>_<runner desc>.f32). Prints shape, relative RMS (rms(a-b)/rms(a)), max abs error,
and non-finite counts; exits 1 if any pair is above --tol (default 0.02, as SELFTEST.md)."""
import argparse, os, sys
import numpy as np

def load(p):
    with open(p, "rb") as f:
        nd = int(np.frombuffer(f.read(8), np.int64)[0])
        shape = tuple(np.frombuffer(f.read(8 * nd), np.int64)[::-1])  # ggml order -> row-major
        return np.frombuffer(f.read(), np.float32).reshape(shape)

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("new"); ap.add_argument("--tol", type=float, default=0.02)
a = ap.parse_args()
names = sorted(set(os.listdir(a.ref)) & set(os.listdir(a.new)))
only = sorted(set(os.listdir(a.ref)) ^ set(os.listdir(a.new)))
bad = 0
for n in names:
    x, y = load(os.path.join(a.ref, n)), load(os.path.join(a.new, n))
    if x.shape != y.shape:
        print(f"{n}: shape {x.shape} vs {y.shape}"); bad += 1; continue
    nf = int((~np.isfinite(y)).sum())
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    rms = np.sqrt(np.mean(x64 ** 2)); err = np.sqrt(np.mean((x64 - y64) ** 2))
    rel = err / rms if rms > 0 else float(err > 0)
    flag = "  <-- FAIL" if (rel > a.tol or nf) else ""
    bad += bool(flag)
    print(f"{n:60s} {str(x.shape):24s} rel_rms {rel:.2e}  max_abs {np.max(np.abs(x64 - y64)):.3e}  nonfinite {nf}{flag}")
if only:
    print("unpaired:", " ".join(only))
sys.exit(1 if bad else 0)
