import math, time
import numpy as np
import mojo_quantlib as mql
from mojo_quantlib._lib import addr, f64, i64, lib

def t(fn, reps=5):
    best = math.inf
    for _ in range(reps):
        s = time.perf_counter(); fn(); best = min(best, time.perf_counter()-s)
    return best*1000

nd = 500_000
print("=== curve ===")
nt = np.linspace(0, 30, nd)
dcur = mql.DiscountCurve([mql.Date(2,1,2024)+mql.Period(i,mql.Years) for i in range(31)],
                         np.exp(-np.linspace(0,30,31)*np.linspace(0.03,0.05,31)), mql.Actual365Fixed())
print("list(nt)                 %.2f" % t(lambda: list(nt)))
lst = list(nt)
print("[float(x) for x in list]  %.2f" % t(lambda: [float(x) for x in lst]))
fl = [float(x) for x in lst]
print("np.asarray(list)         %.2f" % t(lambda: np.asarray(fl, dtype=np.float64)))
print("_times_from_input        %.2f" % t(lambda: dcur._times_from_input(nt)))
tm,_ = dcur._times_from_input(nt)
print("_check_range             %.2f" % t(lambda: dcur._check_range(tm, False)))
print("maxTime                  %.2f" % t(lambda: dcur.maxTime()))
tn = f64(dcur.times()); ld = f64(np.log(dcur.discounts()))
cf = np.empty(nd)
print("curve kernel             %.2f" % t(lambda: lib().mql_discount_curve(addr(tn),addr(ld),31,addr(nt),addr(cf),nd)))
print("curve full               %.2f" % t(lambda: dcur.discount(nt)))

print("=== year fractions ===")
starts = [mql.Date(1,1,1901) + int(i % 90_000) for i in range(nd)]
ends = [d + 30 + i % 720 for i, d in enumerate(starts)]
print("listcomp serial x2       %.2f" % t(lambda: ([d.serialNumber() for d in starts], [d.serialNumber() for d in ends])))
print("i64(list)                %.2f" % t(lambda: i64([d.serialNumber() for d in starts])))
print("np.array(gen)            %.2f" % t(lambda: np.fromiter((d.serialNumber() for d in starts), np.int64, nd)))
sa = i64([d.serialNumber() for d in starts]); sb = i64([d.serialNumber() for d in ends])
fo = np.empty(nd)
print("yf kernel                %.2f" % t(lambda: lib().mql_year_fractions(addr(sa),addr(sb),addr(fo),nd,0)))
print("yf full                  %.2f" % t(lambda: mql.year_fractions(starts, ends, mql.Actual365Fixed())))

print("=== date serials ===")
years = np.ascontiguousarray(1901 + np.arange(nd) % 299, dtype=np.int64)
months = np.ascontiguousarray(1 + np.arange(nd) % 12, dtype=np.int64)
days = np.full(nd, 15, dtype=np.int64)
print("date_serials full        %.2f" % t(lambda: mql.date_serials(years,months,days)))
leap = (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
print("  leap calc              %.2f" % t(lambda: (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))))
print("  np.choose              %.2f" % t(lambda: np.choose(months-1, [31,28,31,30,31,30,31,31,30,31,30,31], mode='clip')))
print("  3 range checks         %.2f" % t(lambda: (np.any((years<1901)|(years>2199)) or np.any((months<1)|(months>12)) or np.any((days<1)|(days>31)))))
print("  i64 x3                 %.2f" % t(lambda: (i64(years),i64(months),i64(days))))
print("  kernel                 %.2f" % t(lambda: lib().mql_dates_to_serial(addr(years),addr(months),addr(days),addr(np.empty(nd,np.int64)),nd)))


print("=== serial extraction ===")
from operator import attrgetter
print("  method call listcomp   %.2f" % t(lambda: [d.serialNumber() for d in starts]))
print("  _serial listcomp       %.2f" % t(lambda: [d._serial for d in starts]))
g = attrgetter('_serial')
print("  fromiter attrgetter    %.2f" % t(lambda: np.fromiter(map(g, starts), np.int64, nd)))
print("  fromiter method        %.2f" % t(lambda: np.fromiter((d.serialNumber() for d in starts), np.int64, nd)))
print("  np.array(_serial)      %.2f" % t(lambda: np.array([d._serial for d in starts], dtype=np.int64)))
print("  np.fromiter(_serial)   %.2f" % t(lambda: np.fromiter((d._serial for d in starts), np.int64, nd)))

print("=== black formula py ===")
n = 1_000_000
rng = np.random.default_rng(7)
bt = np.where(np.arange(n) % 2, mql.Option.Call, mql.Option.Put)
bs = np.ascontiguousarray(rng.uniform(70,130,n)); bf = np.ascontiguousarray(rng.uniform(70,130,n))
bv = np.ascontiguousarray(rng.uniform(0.1,0.8,n)); bd = np.full(n,0.96)
print("black full               %.2f" % t(lambda: mql.blackFormula(bt,bs,bf,bv,bd)))
print("  _broadcast             %.2f" % t(lambda: mql.options._broadcast(bt,bs,bf,bv,bd,np.zeros(1))))
print("  2 bounds checks        %.2f" % t(lambda: (np.any(bs+0.0<=0) or np.any(bf+0.0<=0) or np.any(bv<0) or np.any(bd<=0))))
print("  kernel                 %.2f" % t(lambda: lib().mql_black_formula(addr(bt),addr(bs),addr(bf),addr(bv),addr(bd),addr(np.zeros(1)),addr(np.empty(n)),n)))
print("  f64 x5                 %.2f" % t(lambda: [f64(x) for x in (bs,bf,bv,bd)]))
