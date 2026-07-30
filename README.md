# mojo-quantlib

`mojo-quantlib` is an independent Mojo implementation of a small,
compute-oriented subset of QuantLib: selected date math, yield curves, and
vanilla option pricing. Its Python API uses QuantLib names where behavior is
covered:

```python
import numpy as np
import mojo_quantlib as ql

today = ql.Date(2, ql.January, 2024)
curve = ql.DiscountCurve(
    [today, ql.Date(2, ql.January, 2025), ql.Date(2, ql.January, 2026)],
    [1.0, 0.957, 0.906],
    ql.Actual365Fixed(),
)

spots = np.array([90.0, 100.0, 110.0])
prices = ql.black_scholes(
    ql.Option.Call, spots, 100.0, 1.0, 0.05, 0.20
)
print(curve.discount([0.5, 1.0, 1.5]))
print(prices)
```

NumPy arrays are an extension: one Python call prices or interpolates a whole
portfolio in Mojo. This is not a drop-in replacement for QuantLib outside the
explicit subset below.

## Covered subset

| area | API |
| --- | --- |
| Dates and periods | `Date`, `Period`, day/week/month/year arithmetic, serial conversion, weekdays, month ends, vectorized `date_serials` and `dates_from_serial` |
| Calendars | `NullCalendar`, `WeekendsOnly`, and `TARGET`; Following, Modified Following, and Preceding adjustment; business-day advance/count, month ends, and holiday overrides |
| Day counters | Standard `Actual360`, standard and No-Leap `Actual365Fixed`, `ActualActual` ISDA, and `Thirty360` USA, Bond Basis/ISMA, European/Eurobond, Italian, German/ISDA, and NASD; vectorized standard actual year fractions |
| Curves | `DiscountCurve` with log-linear discounts, `ZeroCurve` with linear zero yields, `FlatForward`, `InterestRate`; date/time discount, zero-rate and forward-rate queries |
| European options | scalar and vectorized `blackFormula`, `blackFormulaImpliedStdDev`, `BlackCalculator`, Black-Scholes prices and analytic delta/gamma/vega/theta/rho |
| American options | vectorized Cox-Ross-Rubinstein pricing with early exercise for calls and puts |

The tests compare these claims directly with the `quantlib-python` bindings.
They include all nine exposed `Thirty360` conventions, TARGET holidays and
overrides, leap-date conversions, the three supported curve compounding
forms, Black-76, implied standard deviation, analytic Black-Scholes Greeks,
and American-option values. Unsupported Actual/Actual conventions and hybrid
compounding modes raise `NotImplementedError`.

This is deliberately not all of QuantLib. It does not include schedules,
cashflows, bonds, swaps, rate-helper bootstrapping, currencies, indexes,
fixings, credit models, stochastic-process families, Monte Carlo, finite
differences, or exotic payoffs. Curve interpolators other than the upstream
defaults above are not covered. `FlatForward` covers the reference-date/rate
constructor, not quote handles or settlement-day construction.

## Install

The supported install is a Linux x86-64 source checkout with
[Pixi](https://pixi.sh/) available. The environment pins the tested Mojo/MAX
nightly and installs the QuantLib bindings used by the parity tests:

```bash
git clone https://github.com/lee101/mojo-quantlib.git
cd mojo-quantlib
pixi install
pixi run build
```

Then run Python inside the environment:

```bash
pixi run python - <<'PY'
import mojo_quantlib as ql

price = ql.black_scholes(ql.Option.Call, 100.0, 100.0, 1.0, 0.05, 0.20)
print(f"{price:.6f}")
PY
```

This prints `10.450584`. `pixi run build` emits
`dist/libmojo-quantlib.so`. The Python loader rebuilds a missing or stale
library in a source checkout. `MOJO_QUANTLIB_LIB` can point to a compatible
prebuilt library.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14, QuantLib 1.43, and Mojo
1.0.0b3.dev2026072406. These are best-of-three wall times. The QuantLib column
uses its official Python bindings; the speedups primarily come from replacing
hundreds of thousands of SWIG calls with one array call.

| benchmark | mojo-quantlib | QuantLib Python | result |
| --- | ---: | ---: | ---: |
| Date serial conversion (500k) | 59.69 ms | 1410.90 ms | 23.64x faster |
| Actual/365 year fractions (500k) | 149.06 ms | 1714.36 ms | 11.50x faster |
| Log-linear curve discounts (500k) | 139.53 ms | 1761.89 ms | 12.63x faster |
| Black-76 prices (1M) | 175.29 ms | 1074.95 ms | 6.13x faster |
| Black implied stddev (100k) | 16.95 ms | 175.89 ms | 10.38x faster |
| Black implied stddev CPU (1M) | 98.92 ms | 2394.97 ms | 24.21x faster |
| Black implied stddev GPU (1M) | 78.29 ms | 2394.97 ms | 30.59x faster |
| American CRR, 500 steps (100) | 24.36 ms | 201.71 ms | 8.28x faster |

The benchmark script checks representative outputs before timing and prints
the full Markdown table, including slower results if a future environment
produces them. No benchmark uses synthetic or copied numbers.

CPU is the default. Black implied-standard-deviation inversion also accepts
`device="gpu"` because its repeated Black evaluations have enough arithmetic
intensity to amortize transfers at large portfolio sizes. The GPU benchmark
uses 1M options and at most 56 MB of device buffers. The call checks that GPU 0
has at least 4000 MiB free and uses the CPU when no suitable GPU is available.
An attempted GPU initialization, allocation, compilation, or launch failure is
reported as an error. Calls are capped below 2 GB of total device allocation.
No GPU path is used for the other
kernels: they were already more than 5x ahead and do not justify transfer and
launch overhead.

## How it works

All kernels and C exports live in one Mojo compilation unit. The build creates
one shared library, and Python calls it through `ctypes`. Array arguments are
broadcast with NumPy and made C-contiguous, then passed over the C ABI as
64-bit integer addresses. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` inside each non-parametric exported
function.

Python owns every input, output, and scratch buffer. Curves use parallel
contiguous `float64` arrays for node times and values; option portfolios use
one contiguous array per field. Mojo allocates no cross-language objects and
one FFI call processes the complete batch. Scalar calls use the same kernels
and return ordinary Python floats.

The implied-standard-deviation kernel uses native-width `float64` SIMD loads
and stores with a scalar tail. Portfolios of at least 16,384 options are split
into independent 256-option chunks across up to 16 workers; smaller calls stay
serial to avoid thread-launch overhead. Scalar broadcast inputs remain
one-element NumPy buffers across the FFI instead of being expanded into
portfolio-sized copies.

## License

MIT
