"""Shared-library loading and NumPy buffer helpers."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(os.environ.get("MOJO_MERCANTILE_LIB", ROOT / "dist" / "libmojo-mercantile.so"))
I = ctypes.c_int64
I64 = ctypes.c_longlong
F = ctypes.c_double

_SIGNATURES = {
    "mm_xy": ([F, F, I, I], None),
    "mm_lnglat": ([F, F, I, I], None),
    "mm_tile": ([F, F, I, I, I], None),
    "mm_bounds": ([I64, I64, I64, I], None),
    "mm_xy_many": ([I, I, I, I], None),
    "mm_lnglat_many": ([I, I, I, I], None),
    "mm_tile_many": ([I, I, I, I, I], None),
    "mm_bounds_many": ([I, I, I], None),
}


def build() -> Path:
    """Build the shared library through the repository's supported command."""
    script = ROOT / "build" / "build.sh"
    source = ROOT / "src" / "mercantile.mojo"
    if LIB.exists() and LIB.stat().st_mtime >= source.stat().st_mtime:
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise RuntimeError("mojo not found; run this package through pixi")
    subprocess.run(["bash", str(script)], check=True, cwd=ROOT, timeout=1800)
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64_pairs(values) -> np.ndarray:
    """Return a C-contiguous float64 ``(n, 2)`` input without precision loss."""
    arr = np.asarray(values)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("expected an array shaped (n, 2)")
    if arr.dtype.kind not in "fiu":
        raise TypeError("coordinates must be real-valued")
    if arr.dtype.kind in "iu" and arr.size and np.max(np.abs(arr.astype(object))) > 2**53:
        raise ValueError("integer coordinates larger than 2**53 cannot be represented exactly as float64")
    return np.ascontiguousarray(arr, dtype=np.float64)


def i64_triples(values) -> np.ndarray:
    """Return a C-contiguous int64 ``(n, 3)`` input without coercion."""
    arr = np.asarray(values)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("expected an array shaped (n, 3)")
    if arr.dtype.kind not in "iu" or arr.dtype.kind == "b":
        raise TypeError("tiles must be an integer array; float values are not coerced")
    if arr.dtype.kind == "u" and arr.size and np.max(arr) > np.iinfo(np.int64).max:
        raise OverflowError("tile values must fit in int64")
    return np.ascontiguousarray(arr, dtype=np.int64)


def addr(arr: np.ndarray) -> int:
    return int(arr.ctypes.data)
