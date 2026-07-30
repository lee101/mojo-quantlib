import math

import numpy as np
import pytest
import QuantLib as ql

import mojo_quantlib as mql
from mojo_quantlib import options as mql_options


@pytest.mark.parametrize("option_type", [mql.Option.Call, mql.Option.Put])
@pytest.mark.parametrize(
    "strike,forward,stddev,discount,displacement",
    [
        (100.0, 105.0, 0.2, 0.97, 0.0),
        (120.0, 95.0, 0.8, 0.83, 0.0),
        (-0.01, 0.02, 0.4, 0.99, 0.05),
        (100.0, 100.0, 0.0, 1.0, 0.0),
    ],
)
def test_black_formula_matches_quantlib(
    option_type, strike, forward, stddev, discount, displacement
):
    assert mql.blackFormula(
        option_type, strike, forward, stddev, discount, displacement
    ) == pytest.approx(
        ql.blackFormula(
            option_type, strike, forward, stddev, discount, displacement
        ),
        rel=3e-14,
        abs=3e-14,
    )


def test_vector_black_formula_matches_quantlib():
    rng = np.random.default_rng(10)
    n = 20_000
    option_type = rng.choice([mql.Option.Call, mql.Option.Put], n)
    strike = rng.uniform(50, 150, n)
    forward = rng.uniform(50, 150, n)
    stddev = rng.uniform(0.01, 2.0, n)
    discount = rng.uniform(0.5, 1.0, n)
    got = mql.blackFormula(option_type, strike, forward, stddev, discount)
    expected = np.array(
        [
            ql.blackFormula(int(o), float(k), float(f), float(s), float(d))
            for o, k, f, s, d in zip(option_type, strike, forward, stddev, discount)
        ]
    )
    assert np.allclose(got, expected, rtol=2e-11, atol=1e-11)


def test_ffi_inputs_copy_strides_preserve_dtype_and_reject_narrowing():
    base = np.linspace(80, 120, 2002, dtype=np.float32)
    spots = base[::2]
    got = mql.black_scholes(mql.Option.Call, spots, 100, 1, 0.05, 0.2)
    assert got.dtype == np.float64
    assert got.shape == spots.shape
    assert np.all(np.isfinite(got))
    with pytest.raises(TypeError):
        mql.blackFormula(1.5, 100, 100, 0.2)
    with pytest.raises(OverflowError):
        mql.blackFormula(mql.Option.Call, 2**54, 2**54, 0.2)
    assert mql.blackFormula(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    ).shape == (0,)


@pytest.mark.parametrize("option_type", [mql.Option.Call, mql.Option.Put])
@pytest.mark.parametrize("stddev", [0.01, 0.15, 0.8, 2.0])
def test_implied_stddev_roundtrip_and_quantlib(option_type, stddev):
    price = ql.blackFormula(option_type, 100.0, 104.0, stddev, 0.96)
    got = mql.blackFormulaImpliedStdDev(
        option_type, 100.0, 104.0, price, 0.96, accuracy=1e-10
    )
    expected = ql.blackFormulaImpliedStdDev(
        option_type, 100.0, 104.0, price, 0.96, 0.0, stddev, 1e-10, 100
    )
    assert got == pytest.approx(stddev, abs=1e-9)
    assert got == pytest.approx(expected, abs=1e-9)


def test_vector_implied_stddev_roundtrip():
    strike = np.linspace(90, 110, 1000)
    forward = np.full(1000, 100.0)
    stddev = np.linspace(0.05, 1.5, 1000)
    types = np.where(np.arange(1000) % 2, mql.Option.Call, mql.Option.Put)
    prices = mql.blackFormula(types, strike, forward, stddev)
    got = mql.blackFormulaImpliedStdDev(
        types, strike, forward, prices, accuracy=1e-10
    )
    assert np.allclose(got, stddev, atol=1e-9)


@pytest.mark.parametrize("n", [1003, 16_383, 16_387])
def test_implied_stddev_simd_tail_and_parallel_threshold(n):
    strike = np.linspace(90, 110, n)
    forward = np.full(n, 100.0)
    stddev = np.linspace(0.05, 1.5, n)
    types = np.where(np.arange(n) % 2, mql.Option.Call, mql.Option.Put)
    prices = mql.blackFormula(types, strike, forward, stddev)
    got = mql.blackFormulaImpliedStdDev(
        types, strike, forward, prices, accuracy=1e-10
    )
    assert np.allclose(got, stddev, atol=1e-9)


def test_implied_stddev_gpu_path_or_silent_cpu_fallback():
    n = 1003
    strike = np.linspace(90, 110, n)
    forward = np.full(n, 100.0)
    stddev = np.linspace(0.05, 1.5, n)
    types = np.where(np.arange(n) % 2, mql.Option.Call, mql.Option.Put)
    prices = mql.blackFormula(types, strike, forward, stddev)
    got = mql.blackFormulaImpliedStdDev(
        types, strike, forward, prices, accuracy=1e-10, device="gpu"
    )
    assert np.allclose(got, stddev, atol=1e-9)


def test_implied_stddev_gpu_low_memory_falls_back(monkeypatch):
    monkeypatch.setattr(mql_options, "_gpu_memory_available", lambda: False)
    price = mql.blackFormula(mql.Option.Call, 100, 104, 0.3)
    got = mql.blackFormulaImpliedStdDev(
        mql.Option.Call, 100, 104, price, device="gpu"
    )
    assert got == pytest.approx(0.3, abs=1e-6)


def _quantlib_european(option_type, spot, strike, maturity, rate, vol, dividend):
    today = ql.Date(2, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    expiry = today + int(round(maturity * 365))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    r = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))
    q = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend, dc))
    sigma = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), vol, dc)
    )
    process = ql.BlackScholesMertonProcess(spot_handle, q, r, sigma)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(option_type, strike), ql.EuropeanExercise(expiry)
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


@pytest.mark.parametrize("option_type", [mql.Option.Call, mql.Option.Put])
def test_black_scholes_price_and_greeks_match_quantlib(option_type):
    args = (option_type, 103.0, 97.0, 2.0, 0.041, 0.27, 0.013)
    got = mql.black_scholes_greeks(*args)
    ref = _quantlib_european(*args)
    assert got.price == pytest.approx(ref.NPV(), rel=2e-13)
    assert got.delta == pytest.approx(ref.delta(), rel=2e-13)
    assert got.gamma == pytest.approx(ref.gamma(), rel=2e-13)
    assert got.vega == pytest.approx(ref.vega(), rel=2e-12)
    assert got.theta == pytest.approx(ref.theta(), rel=2e-12)
    assert got.rho == pytest.approx(ref.rho(), rel=2e-12)


def test_black_scholes_vector_broadcast_and_put_call_parity():
    spots = np.linspace(60, 140, 5000)
    call = mql.black_scholes(mql.Option.Call, spots, 100, 1.5, 0.03, 0.22, 0.01)
    put = mql.black_scholes(mql.Option.Put, spots, 100, 1.5, 0.03, 0.22, 0.01)
    parity = spots * np.exp(-0.01 * 1.5) - 100 * np.exp(-0.03 * 1.5)
    assert np.allclose(call - put, parity, rtol=2e-14, atol=2e-14)


def test_black_scholes_published_reference_value():
    # Standard textbook vector: S=K=100, r=5%, sigma=20%, T=1.
    price = mql.black_scholes(mql.Option.Call, 100, 100, 1, 0.05, 0.2)
    assert isinstance(price, float)
    assert price == pytest.approx(10.450583572185565, abs=2e-13)


def test_black_calculator_core_methods_match_quantlib():
    ours = mql.BlackCalculator(mql.PlainVanillaPayoff(mql.Option.Call, 100), 105, 0.25, 0.96)
    theirs = ql.BlackCalculator(ql.PlainVanillaPayoff(ql.Option.Call, 100), 105, 0.25, 0.96)
    for method, args in [
        ("value", ()),
        ("deltaForward", ()),
        ("delta", (98.0,)),
        ("gammaForward", ()),
        ("gamma", (98.0,)),
        ("vega", (1.5,)),
        ("itmCashProbability", ()),
        ("itmAssetProbability", ()),
    ]:
        assert getattr(ours, method)(*args) == pytest.approx(
            getattr(theirs, method)(*args), rel=3e-13, abs=3e-13
        )


def _quantlib_american(option_type, spot, strike, maturity, rate, vol, dividend, steps):
    today = ql.Date(2, 1, 2024)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    expiry = today + int(round(maturity * 365))
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, dividend, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), vol, dc)
        ),
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(option_type, strike), ql.AmericanExercise(today, expiry)
    )
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
    return option.NPV()


@pytest.mark.parametrize(
    "option_type,spot,strike,rate,vol,dividend",
    [
        (mql.Option.Put, 100, 100, 0.05, 0.20, 0.00),
        (mql.Option.Put, 80, 100, 0.03, 0.35, 0.01),
        (mql.Option.Call, 100, 90, 0.04, 0.25, 0.08),
    ],
)
def test_american_crr_converges_to_quantlib(
    option_type, spot, strike, rate, vol, dividend
):
    steps = 1001
    got = mql.american_option(
        option_type, spot, strike, 1.0, rate, vol, dividend, steps
    )
    expected = _quantlib_american(
        option_type, spot, strike, 1.0, rate, vol, dividend, steps
    )
    assert got == pytest.approx(expected, rel=3e-3, abs=3e-3)


def test_american_put_dominates_european_put():
    american = mql.american_option(mql.Option.Put, 100, 100, 1, 0.08, 0.2, steps=800)
    european = mql.black_scholes(mql.Option.Put, 100, 100, 1, 0.08, 0.2)
    assert american >= european


def test_invalid_black_inputs_raise():
    with pytest.raises(ValueError):
        mql.blackFormula(mql.Option.Call, 0, 100, 0.2)
    with pytest.raises(RuntimeError):
        mql.blackFormulaImpliedStdDev(mql.Option.Call, 100, 100, 200)
    with pytest.raises(ValueError):
        mql.blackFormulaImpliedStdDev(
            mql.Option.Call, 100, 100, 10, device="accelerator"
        )
    with pytest.raises(ValueError):
        mql.blackFormula(mql.Option.Call, 100, 100, np.nan)
    with pytest.raises(ValueError):
        mql.blackFormulaImpliedStdDev(
            mql.Option.Call, 100, 100, 10, maxIterations=1.5
        )
    with pytest.raises(ValueError):
        mql.american_option(
            mql.Option.Put, 100, 100, 1, 0.05, 0.2, steps=10.5
        )
