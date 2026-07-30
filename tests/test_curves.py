import numpy as np
import pytest
import QuantLib as ql

import mojo_quantlib as mql


def curve_dates(module):
    return [
        module.Date(2, 1, 2024),
        module.Date(2, 7, 2024),
        module.Date(2, 1, 2025),
        module.Date(2, 1, 2026),
        module.Date(2, 1, 2029),
    ]


def test_discount_curve_nodes_and_scalar_parity():
    discounts = [1.0, 0.982, 0.957, 0.906, 0.771]
    ours = mql.DiscountCurve(curve_dates(mql), discounts, mql.Actual365Fixed())
    theirs = ql.DiscountCurve(curve_dates(ql), discounts, ql.Actual365Fixed())
    assert ours.referenceDate().serialNumber() == theirs.referenceDate().serialNumber()
    assert ours.maxDate().serialNumber() == theirs.maxDate().serialNumber()
    assert ours.discounts() == pytest.approx(theirs.discounts())
    for t in np.linspace(0, theirs.maxTime(), 101):
        assert ours.discount(float(t)) == pytest.approx(
            theirs.discount(float(t)), rel=5e-11, abs=2e-13
        )


def test_discount_curve_vector_parity_and_extrapolation():
    discounts = [1.0, 0.982, 0.957, 0.906, 0.771]
    ours = mql.DiscountCurve(curve_dates(mql), discounts, mql.Actual365Fixed())
    theirs = ql.DiscountCurve(curve_dates(ql), discounts, ql.Actual365Fixed())
    times = np.linspace(0, ours.maxTime(), 20_000)
    got = ours.discount(times)
    expected = np.array([theirs.discount(float(t)) for t in times])
    assert np.allclose(got, expected, rtol=5e-11, atol=2e-13)
    with pytest.raises(RuntimeError):
        ours.discount(ours.maxTime() + 1)
    assert ours.discount(ours.maxTime() + 1, True) == pytest.approx(
        theirs.discount(theirs.maxTime() + 1, True), rel=5e-11
    )


@pytest.mark.parametrize(
    "compounding,frequency",
    [
        (mql.Continuous, mql.Annual),
        (mql.Compounded, mql.Semiannual),
        (mql.Simple, mql.Annual),
    ],
)
def test_zero_curve_parity(compounding, frequency):
    rates = [0.031, 0.033, 0.036, 0.039, 0.043]
    ours = mql.ZeroCurve(
        curve_dates(mql), rates, mql.Actual365Fixed(),
        compounding=compounding, frequency=frequency,
    )
    theirs = ql.ZeroCurve(
        curve_dates(ql), rates, ql.Actual365Fixed(), ql.NullCalendar(),
        ql.Linear(), compounding, frequency,
    )
    for t in np.linspace(0, ours.maxTime(), 101):
        assert ours.discount(float(t)) == pytest.approx(
            theirs.discount(float(t)), rel=2e-11, abs=2e-13
        )


@pytest.mark.parametrize(
    "compounding,frequency",
    [
        (mql.Continuous, mql.Annual),
        (mql.Compounded, mql.Quarterly),
        (mql.Simple, mql.Annual),
    ],
)
def test_flat_forward_parity(compounding, frequency):
    date_ours = mql.Date(2, 1, 2024)
    date_theirs = ql.Date(2, 1, 2024)
    ours = mql.FlatForward(
        date_ours, 0.0375, mql.Actual365Fixed(), compounding, frequency
    )
    theirs = ql.FlatForward(
        date_theirs, 0.0375, ql.Actual365Fixed(), compounding, frequency
    )
    times = np.linspace(0, 30, 301)
    assert np.allclose(
        ours.discount(times),
        [theirs.discount(float(t)) for t in times],
        rtol=2e-14,
        atol=2e-14,
    )


def test_zero_and_forward_rates_match_quantlib():
    discounts = [1.0, 0.982, 0.957, 0.906, 0.771]
    ours = mql.DiscountCurve(curve_dates(mql), discounts, mql.Actual365Fixed())
    theirs = ql.DiscountCurve(curve_dates(ql), discounts, ql.Actual365Fixed())
    for compounding, frequency in [
        (mql.Continuous, mql.Annual),
        (mql.Compounded, mql.Semiannual),
        (mql.Simple, mql.Annual),
    ]:
        assert ours.zeroRate(2.0, compounding, frequency).rate() == pytest.approx(
            theirs.zeroRate(2.0, compounding, frequency).rate(), rel=1e-9
        )
        assert ours.forwardRate(1.0, 3.0, compounding, frequency).rate() == pytest.approx(
            theirs.forwardRate(1.0, 3.0, compounding, frequency).rate(), rel=1e-9
        )


def test_curve_date_overload_matches_time_overload():
    dates = curve_dates(mql)
    curve = mql.DiscountCurve(
        dates, [1.0, 0.982, 0.957, 0.906, 0.771], mql.Actual365Fixed()
    )
    query = mql.Date(17, 9, 2025)
    assert curve.discount(query) == curve.discount(curve.timeFromReference(query))


def test_curve_rejects_bad_nodes():
    with pytest.raises(ValueError):
        mql.DiscountCurve(
            [mql.Date(1, 1, 2024), mql.Date(1, 1, 2024)],
            [1.0, 0.9],
            mql.Actual365Fixed(),
        )
    with pytest.raises(ValueError):
        mql.DiscountCurve(
            [mql.Date(1, 1, 2024), mql.Date(1, 1, 2025)],
            [1.0, -0.9],
            mql.Actual365Fixed(),
        )


def test_unsupported_compounding_fails_instead_of_using_continuous():
    with pytest.raises(NotImplementedError):
        mql.InterestRate(
            0.03,
            mql.Actual365Fixed(),
            mql.SimpleThenCompounded,
            mql.Annual,
        )
