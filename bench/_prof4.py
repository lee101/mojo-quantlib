import math, time
import numpy as np
import mojo_quantlib as mql
from mojo_quantlib._lib import addr, lib

def t(fn, reps=5):
    best = math.inf
    for _ in range(reps):
        s = time.perf_counter(); fn(); best = min(best, time.perf_counter()-s)
    return best*1000

rng = np.random.default_rng(7)
N = 1_000_000
types = np.where(np.arange(N) % 2, mql.Option.Call, mql.Option.Put).astype(np.int64)
strikes = np.ascontiguousarray(rng.uniform(70,130,N))
forwards = np.ascontiguousarray(rng.uniform(70,130,N))
stddevs = np.ascontiguousarray(rng.uniform(0.1,0.8,N))
prices = mql.blackFormula(types, strikes, forwards, stddevs)
one = np.ones(1); zero1 = np.zeros(1)

for n in (10_000, 100_000, 1_000_000):
    ti=np.ascontiguousarray(types[:n]); si=np.ascontiguousarray(strikes[:n])
    fi=np.ascontiguousarray(forwards[:n]); pi=np.ascontiguousarray(prices[:n])
    oi = np.empty(n, np.float64)
    disc = np.full(n, 1.0)
    rc = lib().mql_black_implied_stddev(addr(ti),addr(si),addr(fi),addr(pi),addr(one),addr(zero1),addr(oi),n,48,1e-6,100)
    ms = t(lambda: lib().mql_black_implied_stddev(addr(ti),addr(si),addr(fi),addr(pi),addr(one),addr(zero1),addr(oi),n,48,1e-6,100))
    ms2 = t(lambda: mql.blackFormulaImpliedStdDev(ti,si,fi,pi))
    print("n=%-8d rc=%d kernel %8.2f ms (%6.1f ns/opt)  wrapper %8.2f ms" % (n, rc, ms, ms*1e6/n, ms2))
    ref = mql.blackFormulaImpliedStdDev(ti,si,fi,pi)
    print("   sample", ref[:3], "match", np.allclose(ref, oi, rtol=1e-12, atol=0))

# how many iterations? vary accuracy
for acc in (1e-4, 1e-6, 1e-8):
    oi = np.empty(100_000, np.float64); dsc = np.full(100_000, 1.0)
    print("acc=%g  %.2f ms" % (acc, t(lambda: lib().mql_black_implied_stddev(addr(types[:100000]),addr(strikes[:100000]),addr(forwards[:100000]),addr(prices[:100000]),addr(one),addr(zero1),addr(oi),100000,48,acc,100))))
