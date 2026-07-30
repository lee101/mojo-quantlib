"""ctypes bridge to the single mojo-quantlib shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(os.environ.get("MOJO_QUANTLIB_LIB", ROOT / "dist/libmojo-quantlib.so"))

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mql_dates_to_serial": ([I, I, I, I, I], None),
    "mql_serial_to_dates": ([I, I, I, I, I], None),
    "mql_year_fractions": ([I, I, I, I, I], None),
    "mql_discount_curve": ([I, I, I, I, I, I], None),
    "mql_zero_curve": ([I, I, I, I, I, I, I, I], None),
    "mql_black_formula": ([I, I, I, I, I, I, I, I], None),
    "mql_black_implied_stddev": ([I, I, I, I, I, I, I, I, I, F, I], I),
    "mql_black_implied_stddev_gpu": ([I, I, I, I, I, I, I, I, I, F, I], I),
    "mql_black_scholes": ([I] * 14, None),
    "mql_american_crr": ([I] * 11, None),
}

_library: ctypes.CDLL | None = None


def build(force: bool = False) -> str:
    sources = list((ROOT / "src").glob("*.mojo"))
    stale = (
        force
        or not LIB_PATH.exists()
        or (sources and LIB_PATH.stat().st_mtime < max(p.stat().st_mtime for p in sources))
    )
    if stale:
        proc = subprocess.run(
            ["bash", str(ROOT / "build/build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0 or not LIB_PATH.exists():
            raise RuntimeError((proc.stderr or proc.stdout).strip())
    return str(LIB_PATH)


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("expected real numeric values")
    if array.dtype.kind in "iu" and array.size and np.any(
        np.abs(array.astype(object)) > 2**53
    ):
        raise OverflowError("integer cannot be represented exactly as float64")
    result = np.asarray(array, dtype=np.float64).copy(order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError("values must be finite")
    return result


def i64(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biu":
        raise TypeError("expected integer values")
    if array.dtype.kind == "u" and array.size and np.any(array > np.iinfo(np.int64).max):
        raise OverflowError("integer is outside the int64 range")
    return np.asarray(array, dtype=np.int64).copy(order="C")


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null pointer for a non-empty buffer")
    return address
