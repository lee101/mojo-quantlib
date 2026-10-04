import ctypes, math, time
import numpy as np

lib = ctypes.CDLL('/tmp/mb.so')
for n in ('mb_scalar_exp','mb_simd_exp','mb_simd_log','mb_scalar_erfc','mb_simd_erfc','mb_simd_erfc_iface'):
    f = getattr(lib, n)
    f.argtypes = [ctypes.c_int64]*3
    f.restype = None

n = 1_000_000
rng = np.random.default_rng(1)
x = np.ascontiguousarray(np.abs(rng.normal(0, 1.5, n)))
o = np.empty(n)

def t(fn, reps=5):
    best = math.inf
    for _ in range(reps):
        s=time.perf_counter(); fn(); best=min(best,time.perf_counter()-s)
    return best*1000

for name in ('mb_scalar_exp','mb_simd_exp','mb_simd_log','mb_scalar_erfc','mb_simd_erfc','mb_simd_erfc_iface'):
    f = getattr(lib, name)
    f(x.ctypes.data, o.ctypes.data, n)
    ms = t(lambda: f(x.ctypes.data, o.ctypes.data, n))
    print("%-20s %8.2f ms  %6.2f ns/elem  checksum %.6f" % (name, ms, ms*1e6/n, o.sum()))
