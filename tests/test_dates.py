import datetime

import numpy as np
import pytest
import QuantLib as ql

import mojo_quantlib as mql


@pytest.mark.parametrize(
    "day,month,year",
    [
        (1, 1, 1901),
        (28, 2, 1904),
        (29, 2, 1904),
        (1, 1, 1970),
        (29, 2, 2000),
        (31, 12, 2199),
    ],
)
def test_date_fields_and_serial_match_quantlib(day, month, year):
    ours = mql.Date(day, month, year)
    theirs = ql.Date(day, month, year)
    assert ours.serialNumber() == theirs.serialNumber()
    assert (ours.year(), ours.month(), ours.dayOfMonth()) == (
        theirs.year(),
        int(theirs.month()),
        theirs.dayOfMonth(),
    )
    assert ours.weekday() == int(theirs.weekday())
    assert ours.dayOfYear() == theirs.dayOfYear()
    assert mql.Date.ISO(ours) == ql.Date.ISO(theirs)


def test_date_roundtrip_datetime_and_serial():
    pydate = datetime.date(2026, 7, 30)
    ours = mql.Date.from_date(pydate)
    assert ours.to_date() == pydate
    assert mql.Date(ours.serialNumber()) == ours


@pytest.mark.parametrize(
    "start,length,unit",
    [
        ((31, 1, 2024), 1, mql.Months),
        ((29, 2, 2024), 1, mql.Years),
        ((31, 12, 2023), 8, mql.Weeks),
        ((1, 3, 2024), -1, mql.Days),
    ],
)
def test_date_period_arithmetic_matches_quantlib(start, length, unit):
    day, month, year = start
    ours = mql.Date(day, month, year) + mql.Period(length, unit)
    theirs = ql.Date(day, month, year) + ql.Period(length, unit)
    assert ours.serialNumber() == theirs.serialNumber()


def test_period_string_and_normalization():
    assert mql.Period("12M").normalized() == mql.Period(1, mql.Years)
    assert str(mql.Period("6M")) == "6M"
    assert mql.Period("3M").frequency() == ql.Period("3M").frequency()


def test_vector_date_conversion_matches_quantlib():
    rng = np.random.default_rng(3)
    years = rng.integers(1901, 2200, 10_000)
    months = rng.integers(1, 13, 10_000)
    max_days = np.array(
        [datetime.date(int(y), int(m % 12 + 1), 1).toordinal()
         - datetime.date(int(y), int(m), 1).toordinal()
         if m < 12 else 31 for y, m in zip(years, months)]
    )
    days = rng.integers(1, max_days + 1)
    serials = mql.date_serials(years, months, days)
    expected = np.array(
        [ql.Date(int(d), int(m), int(y)).serialNumber()
         for y, m, d in zip(years, months, days)]
    )
    assert np.array_equal(serials, expected)
    yy, mm, dd = mql.dates_from_serial(serials)
    assert np.array_equal(yy, years)
    assert np.array_equal(mm, months)
    assert np.array_equal(dd, days)


def test_vector_date_conversion_rejects_narrowing_and_handles_empty_strides():
    with pytest.raises(TypeError):
        mql.date_serials([2024.5], [1], [1])
    with pytest.raises(OverflowError):
        mql.dates_from_serial(np.array([2**63], dtype=np.uint64))
    empty = np.arange(0, dtype=np.int64)[::2]
    assert mql.date_serials(empty, empty, empty).shape == (0,)


@pytest.mark.parametrize(
    "counter_ours,counter_theirs",
    [
        (mql.Actual365Fixed(), ql.Actual365Fixed()),
        (mql.Actual360(), ql.Actual360()),
        (
            mql.Thirty360(mql.Thirty360.BondBasis),
            ql.Thirty360(ql.Thirty360.BondBasis),
        ),
        (
            mql.Thirty360(mql.Thirty360.European),
            ql.Thirty360(ql.Thirty360.European),
        ),
        (
            mql.Thirty360(mql.Thirty360.USA),
            ql.Thirty360(ql.Thirty360.USA),
        ),
    ],
)
def test_day_counters_match_quantlib(counter_ours, counter_theirs):
    pairs = [
        ((31, 1, 2023), (28, 2, 2023)),
        ((29, 2, 2024), (31, 3, 2024)),
        ((15, 1, 2024), (15, 7, 2024)),
        ((30, 12, 2024), (3, 2, 2031)),
    ]
    for a, b in pairs:
        oa, ob = mql.Date(*a), mql.Date(*b)
        qa, qb = ql.Date(*a), ql.Date(*b)
        assert counter_ours.dayCount(oa, ob) == counter_theirs.dayCount(qa, qb)
        assert counter_ours.yearFraction(oa, ob) == pytest.approx(
            counter_theirs.yearFraction(qa, qb), abs=1e-14
        )


def test_actual_actual_isda_matches_quantlib():
    ours = mql.ActualActual(mql.ActualActual.ISDA)
    theirs = ql.ActualActual(ql.ActualActual.ISDA)
    for a, b in [
        ((1, 7, 2023), (1, 7, 2025)),
        ((28, 2, 2024), (1, 3, 2024)),
        ((15, 12, 1999), (15, 3, 2001)),
    ]:
        assert ours.yearFraction(mql.Date(*a), mql.Date(*b)) == pytest.approx(
            theirs.yearFraction(ql.Date(*a), ql.Date(*b)), abs=1e-14
        )


def test_actual365_no_leap_matches_quantlib():
    ours = mql.Actual365Fixed(mql.Actual365Fixed.NoLeap)
    theirs = ql.Actual365Fixed(ql.Actual365Fixed.NoLeap)
    for a, b in [
        ((1, 1, 2023), (1, 1, 2025)),
        ((28, 2, 2024), (1, 3, 2024)),
    ]:
        assert ours.dayCount(mql.Date(*a), mql.Date(*b)) == theirs.dayCount(
            ql.Date(*a), ql.Date(*b)
        )


def test_unsupported_actual_actual_conventions_fail_explicitly():
    with pytest.raises(NotImplementedError):
        mql.ActualActual(mql.ActualActual.AFB)


@pytest.mark.parametrize(
    "name",
    [
        "USA",
        "BondBasis",
        "European",
        "EurobondBasis",
        "Italian",
        "German",
        "ISMA",
        "ISDA",
        "NASD",
    ],
)
def test_all_advertised_thirty360_conventions_match_quantlib(name):
    ours = mql.Thirty360(getattr(mql.Thirty360, name))
    theirs = ql.Thirty360(getattr(ql.Thirty360, name))
    for a, b in [
        ((31, 1, 2023), (28, 2, 2023)),
        ((29, 2, 2024), (31, 3, 2024)),
        ((28, 2, 2023), (31, 8, 2024)),
        ((31, 12, 2024), (3, 2, 2031)),
    ]:
        assert ours.dayCount(mql.Date(*a), mql.Date(*b)) == theirs.dayCount(
            ql.Date(*a), ql.Date(*b)
        )


def test_vector_year_fractions_match_quantlib():
    starts = [mql.Date(1, 1, 2000) + i * 17 for i in range(1000)]
    ends = [d + 365 + i % 31 for i, d in enumerate(starts)]
    got = mql.year_fractions(starts, ends, mql.Actual365Fixed())
    expected = np.array(
        [
            ql.Actual365Fixed().yearFraction(ql.Date(a.serialNumber()), ql.Date(b.serialNumber()))
            for a, b in zip(starts, ends)
        ]
    )
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("calendar_name", ["WeekendsOnly", "TARGET"])
def test_calendar_adjust_and_advance_match_quantlib(calendar_name):
    ours = getattr(mql, calendar_name)()
    theirs = getattr(ql, calendar_name)()
    dates = [
        (1, 1, 2024),
        (29, 3, 2024),
        (31, 3, 2024),
        (1, 5, 2024),
        (25, 12, 2024),
        (31, 8, 2024),
    ]
    for parts in dates:
        od, qd = mql.Date(*parts), ql.Date(*parts)
        assert ours.isBusinessDay(od) == theirs.isBusinessDay(qd)
        for convention in (mql.Following, mql.ModifiedFollowing, mql.Preceding):
            assert ours.adjust(od, convention).serialNumber() == theirs.adjust(
                qd, convention
            ).serialNumber()
        assert ours.advance(od, 10, mql.Days).serialNumber() == theirs.advance(
            qd, 10, ql.Days
        ).serialNumber()


def test_calendar_business_days_between_matches_quantlib():
    ours, theirs = mql.TARGET(), ql.TARGET()
    a, b = mql.Date(1, 1, 2024), mql.Date(31, 12, 2024)
    assert ours.businessDaysBetween(a, b) == theirs.businessDaysBetween(
        ql.Date(a.serialNumber()), ql.Date(b.serialNumber())
    )


def test_calendar_overrides_null_calendar_and_month_boundaries():
    date = mql.Date(1, 1, 2024)
    null = mql.NullCalendar()
    assert null.isBusinessDay(date)
    calendar = mql.TARGET()
    assert calendar.isHoliday(date)
    calendar.removeHoliday(date)
    assert calendar.isBusinessDay(date)
    calendar.addHoliday(date)
    assert calendar.isHoliday(date)
    calendar.resetAddedAndRemovedHolidays()
    assert calendar.isHoliday(date)
    for parts in [(31, 3, 2024), (31, 8, 2024), (30, 11, 2024)]:
        ours, theirs = mql.Date(*parts), ql.Date(*parts)
        assert calendar.endOfMonth(ours).serialNumber() == ql.TARGET().endOfMonth(
            theirs
        ).serialNumber()
