#!/usr/bin/env python3
"""Neuter torchaudio's over-strict CUDA minor-version assertion.

The pip torchaudio wheel is built for CUDA 13.0; the NGC 26.06 container torch is
CUDA 13.3. CUDA guarantees minor-version compatibility (a 13.0 build runs on a 13.3
runtime), and the only torchaudio functions braindecode uses (fftconvolve, filtfilt)
are pure-torch ops that don't touch the C-extension. Verified 2026-06-30: fftconvolve
runs on cuda:0. Patches ``_check_cuda_version`` in the installed torchaudio to a no-op.

Usage: python _patch_torchaudio_cudacheck.py [target_dir=/mnt/t9/eegfm_libs_2606]
"""
import pathlib
import re
import sys

target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/t9/eegfm_libs_2606")
p = target / "torchaudio" / "_extension" / "utils.py"
s = p.read_text()
s2 = re.sub(
    r"def _check_cuda_version\([^)]*\):.*?(?=\n(?:def |@|\Z))",
    "def _check_cuda_version(*a, **k):\n"
    "    return None  # patched: CUDA 13.0 build is minor-compatible with 13.3\n\n",
    s, count=1, flags=re.S)
p.write_text(s2)
print("patched" if "return None  # patched" in s2 else "NO-OP (pattern not matched)", p)
