import cProfile, pstats, io, time, math
import numpy as np
import mojo_quantlib as mql
from mojo_quantlib._lib import addr, lib
import mojo_quantlib.options as O

rng = np.random.default_rng(7)
n = 100_000
types = np.where(np.arange(n) % 2, mql.Option.Call, mql.Option.Put)
strikes = np.ascontiguousarray(rng.uniform(70,130,n))
forwards = np.ascontiguousarray(rng.uniform(70,130,n))
stddevs = np.ascontiguousarray(rng.uniform(0.1,0.8,n))
prices = mql.blackFormula(types, strikes, forwards, stddevs)
one = np.ones(1); zero1 = np.zeros(1); oi = np.empty(n, np.float64)

def bench(fn, reps=7):
    best = math.inf
    for _ in range(reps):
        s=time.perf_counter(); fn(); best=min(best,time.perf_counter()-s)
    return best*1000

print("kernel-only           %.2f" % bench(lambda: lib().mql_black_implied_stddev(addr(types),addr(strikes),addr(forwards),addr(prices),addr(one),addr(zero1),addr(oi),n,0,1e-6,100)))
print("f64 x3                %.2f" % bench(lambda: (O.f64(strikes),O.f64(forwards),O.f64(prices))))
print("bounds checks         %.2f" % bench(lambda: (np.any(strikes+zero1<=0) or np.any(forwards+zero1<=0) or np.any(one<=0) or np.any(prices<0))))
print("type check            %.2f" % bench(lambda: np.any((types != 1) & (types != -1))))
print("empty alloc           %.2f" % bench(lambda: np.empty(n, np.float64)))
print("wrapper               %.2f" % bench(lambda: mql.blackFormulaImpliedStdDev(types,strikes,forwards,prices)))

pr = cProfile.Profile()
for _ in range(5):
    mql.blackFormulaImpliedStdDev(types,strikes,forwards,prices)
pr.enable()
for _ in range(5):
    mql.blackFormulaImpliedStdDev(types,strikes,forwards,prices)
pr.disable()
s=io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(14); print(s.getvalue())
