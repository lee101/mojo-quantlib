import math, time
import numpy as np
import mojo_quantlib as mql
from mojo_quantlib._lib import addr, f64, i64, lib

def t(fn, reps=5):
    best = math.inf
    for _ in range(reps):
        s = time.perf_counter(); fn(); best = min(best, time.perf_counter()-s)
    return best*1000

n = 1_000_000
rng = np.random.default_rng(7)
types = np.where(np.arange(n) % 2, mql.Option.Call, mql.Option.Put)
strikes = np.ascontiguousarray(rng.uniform(70,130,n))
forwards = np.ascontiguousarray(rng.uniform(70,130,n))
stddevs = np.ascontiguousarray(rng.uniform(0.1,0.8,n))
disc = np.full(n, 0.96)
prices = mql.blackFormula(types, strikes, forwards, stddevs, disc)
zero1 = np.zeros(1)
out = np.empty(n, np.float64)

print("--- iv implied stddev ---")
for ni in (16_000, 100_000, 1_000_000):
    ti=types[:ni]; si=strikes[:ni]; fi=forwards[:ni]; pi=prices[:ni]; di=disc[:ni]; oi=np.empty(ni,np.float64)
    r = lib().mql_black_implied_stddev(addr(ti),addr(si),addr(fi),addr(pi),addr(di),addr(zero1),addr(oi),ni,0,1e-6,100)
    print("iv kernel n=%-8d rc=%d  %.2f ms" % (ni, r, t(lambda: lib().mql_black_implied_stddev(addr(ti),addr(si),addr(fi),addr(pi),addr(di),addr(zero1),addr(oi),ni,0,1e-6,100))))
    o2 = np.empty(ni,np.float64)
    print("iv kernel n=%-8d all-scalar-mask=63  %.2f ms" % (ni, t(lambda: lib().mql_black_implied_stddev(addr(np.ones(ni,dtype=np.int64)),addr(np.full(ni,100.0)),addr(np.full(ni,100.0)),addr(pi),addr(di),addr(zero1),addr(o2),ni,63,1e-6,100))))
    print("iv full n=%-8d  %.2f ms" % (ni, t(lambda: mql.blackFormulaImpliedStdDev(ti,si,fi,pi,di))))

print("--- curve ---")
nd = 500_000
nt = np.linspace(0,30,nd)
dcur = mql.DiscountCurve([mql.Date(2,1,2024)+mql.Period(i,mql.Years) for i in range(31)], np.exp(-np.linspace(0,30,31)*np.linspace(0.03,0.05,31)), mql.Actual365Fixed())
tnode = dcur.times(); ldisc = np.log(dcur.discounts())
cfout = np.empty(nd,np.float64)
print("discount_curve kernel     %.2f ms" % t(lambda: lib().mql_discount_curve(addr(tnode),addr(ldisc),31,addr(nt),addr(cfout),nd)))
print("_times_from_input         %.2f ms" % t(lambda: dcur._times_from_input(nt)))
tm,_ = dcur._times_from_input(nt)
print("_check_range              %.2f ms" % t(lambda: dcur._check_range(tm, False)))
print("discount full             %.2f ms" % t(lambda: dcur.discount(nt)))
# random (unsorted) query times to measure branch cost
rnd = np.ascontiguousarray(rng.uniform(0,30,nd))
print("discount full random t    %.2f ms" % t(lambda: dcur.discount(rnd)))
print("discount kernel random t  %.2f ms" % t(lambda: lib().mql_discount_curve(addr(tnode),addr(ldisc),31,addr(rnd),addr(cfout),nd)))

print("--- dates ---")
years = np.ascontiguousarray(1901 + np.arange(nd) % 299, dtype=np.int64)
months = np.ascontiguousarray(1 + np.arange(nd) % 12, dtype=np.int64)
days = np.full(nd, 15, dtype=np.int64)
print("dates_to_serial kernel    %.2f ms" % t(lambda: lib().mql_dates_to_serial(addr(years),addr(months),addr(days),addr(np.empty(nd,np.int64)),nd)))
print("dates_to_serial full      %.2f ms" % t(lambda: mql.date_serials(years,months,days)))
print("serial_to_dates kernel    %.2f ms" % t(lambda: lib().mql_serial_to_dates(addr(years),addr(np.empty(nd,np.int64)),addr(np.empty(nd,np.int64)),addr(np.empty(nd,np.int64)),nd)))

print("--- year fractions ---")
starts = [mql.Date(1,1,1901) + int(i % 90_000) for i in range(200_000)]
ends = [d + 30 + i % 720 for i, d in enumerate(starts)]
a = i64([d.serialNumber() for d in starts]); b = i64([d.serialNumber() for d in ends])
fo = np.empty(len(a), np.float64)
dc = mql.Actual365Fixed()
print("year_frac kernel          %.2f ms" % t(lambda: lib().mql_year_fractions(addr(a),addr(b),addr(fo),len(a),0)))
print("listcomp serialNumber x2  %.2f ms" % t(lambda: ([d.serialNumber() for d in starts], [d.serialNumber() for d in ends])))
print("year_frac full (200k)     %.2f ms" % t(lambda: mql.year_fractions(starts, ends, dc)))

print("--- american ---")
spots = np.linspace(75,125,100)
print("american full             %.2f ms" % t(lambda: mql.american_option(mql.Option.Put, spots, 100.0,1.0,0.05,0.25,0.01,500)))
ar = [None,spots,np.full(100,100.0),np.full(100,1.0),np.full(100,0.05),np.full(100,0.25),np.full(100,0.01)]
print("american numpy prep       %.2f ms" % t(lambda: (
    np.any(ar[1]<=0) or np.any(ar[2]<=0) or np.any(ar[3]<0) or np.any(ar[5]<0))))
work = np.empty(501)
ao = np.empty(100)
print("american kernel only      %.2f ms" % t(lambda: lib().mql_american_crr(
    addr(np.ones(100,dtype=np.int64)), addr(spots), addr(ar[2]), addr(ar[3]), addr(ar[4]), addr(ar[5]), addr(ar[6]), addr(ao), 100, 500, addr(work))))

print("--- black formula py layers ---")
print("black_formula full        %.2f ms" % t(lambda: mql.blackFormula(types,strikes,forwards,stddevs,disc)))
print("black kernel only         %.2f ms" % t(lambda: lib().mql_black_formula(addr(types),addr(strikes),addr(forwards),addr(stddevs),addr(disc),addr(zero1),addr(out),n)))
print("_broadcast full           %.2f ms" % t(lambda: mql.options._broadcast(types,strikes,forwards,stddevs,disc,zero1)))
