import math, time
import numpy as np
import mojo_quantlib as mql
from mojo_quantlib._lib import addr, f64, i64, lib

def t(fn, reps=5):
    best = math.inf
    for _ in range(reps):
        s = time.perf_counter(); fn(); best = min(best, time.perf_counter()-s)
    return best*1000

rng = np.random.default_rng(7)
n = 1_000_000
types = np.where(np.arange(n) % 2, mql.Option.Call, mql.Option.Put)
strikes = np.ascontiguousarray(rng.uniform(70,130,n))
forwards = np.ascontiguousarray(rng.uniform(70,130,n))
stddevs = np.ascontiguousarray(rng.uniform(0.1,0.8,n))
prices = mql.blackFormula(types, strikes, forwards, stddevs)
ni = 100_000
ti=np.ascontiguousarray(types[:ni]); si=np.ascontiguousarray(strikes[:ni])
fi=np.ascontiguousarray(forwards[:ni]); pi=np.ascontiguousarray(prices[:ni])
one = np.ones(1); zero1 = np.zeros(1); oi = np.empty(ni, np.float64)
print("rc mask0:", lib().mql_black_implied_stddev(addr(ti),addr(si),addr(fi),addr(pi),addr(one),addr(zero1),addr(oi),ni,0,1e-6,100))
print("rc mask62:", lib().mql_black_implied_stddev(addr(ti),addr(si),addr(fi),addr(pi),addr(one),addr(zero1),addr(oi),ni,62,1e-6,100))
r = mql.blackFormulaImpliedStdDev(ti,si,fi,pi)
print("wrapper ok, first 5:", r[:5])
print("kernel-only  %.3f ms" % t(lambda: lib().mql_black_implied_stddev(addr(ti),addr(si),addr(fi),addr(pi),addr(one),addr(zero1),addr(oi),ni,62,1e-6,100)))
print("wrapper full %.3f ms" % t(lambda: mql.blackFormulaImpliedStdDev(ti,si,fi,pi)))
print("_broadcast_compact %.3f ms" % t(lambda: mql.options._broadcast_compact(ti,si,fi,pi)))
