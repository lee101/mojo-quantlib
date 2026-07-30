"""Benchmarks against QuantLib 1.43; run only through `pixi run bench`."""

from __future__ import annotations

import math
import platform
import time
from pathlib import Path

import numpy as np
import QuantLib as ql

import mojo_quantlib as mql
from mojo_quantlib.options import _gpu_memory_available


def timeit(fn, repeats=3):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def row(name, ours, upstream):
    ratio = upstream / ours
    result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
    return name, ours * 1000, upstream * 1000, result


def make_american_options(spots, steps):
    today = ql.Date(2, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    expiry = today + 365
    dc = ql.Actual365Fixed()
    options = []
    for spot in spots:
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(float(spot))),
            ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, dc)),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), 0.25, dc)
            ),
        )
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, 100.0),
            ql.AmericanExercise(today, expiry),
        )
        option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
        options.append(option)
    return options


def main():
    rows = []
    n_dates = 500_000
    years = np.ascontiguousarray(1901 + np.arange(n_dates) % 299, dtype=np.int64)
    months = np.ascontiguousarray(1 + np.arange(n_dates) % 12, dtype=np.int64)
    days = np.full(n_dates, 15, dtype=np.int64)
    ours_dates = lambda: mql.date_serials(years, months, days)
    upstream_dates = lambda: [
        ql.Date(int(d), int(m), int(y)).serialNumber()
        for y, m, d in zip(years, months, days)
    ]
    assert ours_dates()[12345] == upstream_dates()[12345]
    rows.append(row("Date serial conversion (500k)", timeit(ours_dates), timeit(upstream_dates)))

    dc_ours, dc_upstream = mql.Actual365Fixed(), ql.Actual365Fixed()
    starts_ours = [mql.Date(1, 1, 1901) + int(i % 90_000) for i in range(n_dates)]
    ends_ours = [date + 30 + i % 720 for i, date in enumerate(starts_ours)]
    starts_upstream = [ql.Date(d.serialNumber()) for d in starts_ours]
    ends_upstream = [ql.Date(d.serialNumber()) for d in ends_ours]
    ours_years = lambda: mql.year_fractions(starts_ours, ends_ours, dc_ours)
    upstream_years = lambda: [
        dc_upstream.yearFraction(a, b) for a, b in zip(starts_upstream, ends_upstream)
    ]
    assert ours_years()[9876] == upstream_years()[9876]
    rows.append(row("Actual/365 year fractions (500k)", timeit(ours_years), timeit(upstream_years)))

    dates_ours = [mql.Date(2, 1, 2024) + mql.Period(i, mql.Years) for i in range(31)]
    dates_upstream = [ql.Date(d.serialNumber()) for d in dates_ours]
    discounts = np.exp(-np.linspace(0, 30, 31) * np.linspace(0.03, 0.05, 31))
    curve_ours = mql.DiscountCurve(dates_ours, discounts, dc_ours)
    curve_upstream = ql.DiscountCurve(dates_upstream, discounts, dc_upstream)
    curve_times = np.linspace(0, curve_ours.maxTime(), 500_000)
    ours_curve = lambda: curve_ours.discount(curve_times)
    upstream_curve = lambda: [curve_upstream.discount(float(t)) for t in curve_times]
    assert ours_curve()[32123] == pytest_approx(upstream_curve()[32123], 1e-9)
    rows.append(row("Log-linear curve discounts (500k)", timeit(ours_curve), timeit(upstream_curve)))

    rng = np.random.default_rng(7)
    n_options = 1_000_000
    option_types = np.where(np.arange(n_options) % 2, mql.Option.Call, mql.Option.Put)
    strikes = np.ascontiguousarray(rng.uniform(70, 130, n_options))
    forwards = np.ascontiguousarray(rng.uniform(70, 130, n_options))
    stddevs = np.ascontiguousarray(rng.uniform(0.1, 0.8, n_options))
    discounts_opt = np.full(n_options, 0.96)
    ours_black = lambda: mql.blackFormula(
        option_types, strikes, forwards, stddevs, discounts_opt
    )
    upstream_black = lambda: [
        ql.blackFormula(int(o), float(k), float(f), float(s), 0.96)
        for o, k, f, s in zip(option_types, strikes, forwards, stddevs)
    ]
    assert ours_black()[4567] == pytest_approx(upstream_black()[4567], 1e-9)
    rows.append(row("Black-76 prices (1M)", timeit(ours_black), timeit(upstream_black)))

    n_iv = 100_000
    iv_strikes = strikes[:n_iv]
    iv_forwards = forwards[:n_iv]
    iv_stddevs = stddevs[:n_iv]
    iv_types = option_types[:n_iv]
    iv_prices = mql.blackFormula(iv_types, iv_strikes, iv_forwards, iv_stddevs)
    ours_iv = lambda: mql.blackFormulaImpliedStdDev(
        iv_types, iv_strikes, iv_forwards, iv_prices
    )
    upstream_iv = lambda: [
        ql.blackFormulaImpliedStdDev(int(o), float(k), float(f), float(p))
        for o, k, f, p in zip(iv_types, iv_strikes, iv_forwards, iv_prices)
    ]
    rows.append(row("Black implied stddev (100k)", timeit(ours_iv), timeit(upstream_iv)))

    gpu_skipped = not _gpu_memory_available()
    if not gpu_skipped:
        gpu_prices = mql.blackFormula(
            option_types, strikes, forwards, stddevs
        )
        ours_iv_cpu_large = lambda: mql.blackFormulaImpliedStdDev(
            option_types, strikes, forwards, gpu_prices
        )
        ours_iv_gpu = lambda: mql.blackFormulaImpliedStdDev(
            option_types, strikes, forwards, gpu_prices, device="gpu"
        )
        upstream_iv_large = lambda: [
            ql.blackFormulaImpliedStdDev(int(o), float(k), float(f), float(p))
            for o, k, f, p in zip(option_types, strikes, forwards, gpu_prices)
        ]
        assert ours_iv_gpu()[4567] == pytest_approx(
            ours_iv_cpu_large()[4567], 1e-6
        )
        upstream_iv_large_time = timeit(upstream_iv_large)
        rows.append(row(
            "Black implied stddev CPU (1M)",
            timeit(ours_iv_cpu_large),
            upstream_iv_large_time,
        ))
        rows.append(row(
            "Black implied stddev GPU (1M)",
            timeit(ours_iv_gpu),
            upstream_iv_large_time,
        ))

    american_spots = np.linspace(75, 125, 100)
    steps = 500
    ours_american = lambda: mql.american_option(
        mql.Option.Put, american_spots, 100.0, 1.0, 0.05, 0.25, 0.01, steps
    )
    upstream_american = lambda: [
        option.NPV() for option in make_american_options(american_spots, steps)
    ]
    assert ours_american()[50] == pytest_approx(upstream_american()[50], 5e-3)
    rows.append(row("American CRR, 500 steps (100)", timeit(ours_american), timeit(upstream_american)))

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.machine()}")
    print(f"Software: Mojo 1.0.0b3 nightly, Python {platform.python_version()}, QuantLib {ql.__version__}")
    print()
    print("| benchmark | mojo-quantlib | QuantLib Python | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours_ms, upstream_ms, result in rows:
        print(f"| {name} | {ours_ms:.2f} ms | {upstream_ms:.2f} ms | {result} |")
    if gpu_skipped:
        print("\nGPU benchmark skipped: less than 4000 MiB free or no compatible GPU.")


def pytest_approx(expected, tolerance):
    class Approx:
        def __eq__(self, actual):
            return abs(actual - expected) <= tolerance

    return Approx()


if __name__ == "__main__":
    main()
