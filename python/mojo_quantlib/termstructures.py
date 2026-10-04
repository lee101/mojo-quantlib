"""Log-linear discount, linear zero, and flat-forward curves."""

from __future__ import annotations

import math

import numpy as np

from ._lib import addr, f64, lib
from .dates import Actual365Fixed, Calendar, Date, DayCounter

Simple, Compounded, Continuous, SimpleThenCompounded, CompoundedThenSimple = range(5)
NoFrequency, Once, Annual, Semiannual, EveryFourthMonth = -1, 0, 1, 2, 3
Quarterly, Bimonthly, Monthly, EveryFourthWeek = 4, 6, 12, 13
Biweekly, Weekly, Daily = 26, 52, 365


def _validate_compounding(compounding, frequency):
    if compounding not in (Simple, Compounded, Continuous):
        raise NotImplementedError(
            "only Simple, Compounded, and Continuous compounding are covered"
        )
    if compounding == Compounded and (
        isinstance(frequency, bool)
        or not isinstance(frequency, (int, np.integer))
        or frequency <= 0
    ):
        raise ValueError("compounded rates require a positive integer frequency")


class InterestRate:
    def __init__(self, rate, dayCounter=None, compounding=Continuous, frequency=Annual):
        self._rate = float(rate)
        self._day_counter = dayCounter
        self._compounding = compounding
        self._frequency = frequency
        _validate_compounding(self._compounding, self._frequency)

    def rate(self) -> float:
        return self._rate

    def dayCounter(self):
        return self._day_counter

    def compounding(self) -> int:
        return self._compounding

    def frequency(self) -> int:
        return self._frequency

    def discountFactor(self, t: float) -> float:
        return _discount_from_rate(self._rate, float(t), self._compounding, self._frequency)

    def __repr__(self):
        return f"InterestRate({self._rate})"


def _discount_from_rate(rate, t, compounding, frequency):
    rate, t = np.asarray(rate), np.asarray(t)
    if compounding == Simple:
        return 1.0 / (1.0 + rate * t)
    if compounding == Compounded:
        return np.power(1.0 + rate / frequency, -frequency * t)
    return np.exp(-rate * t)


def _rate_from_discount(discount, t, compounding, frequency):
    if t <= 0:
        raise ValueError("positive time required")
    if compounding == Simple:
        return (1.0 / discount - 1.0) / t
    if compounding == Compounded:
        return frequency * (discount ** (-1.0 / (frequency * t)) - 1.0)
    return -math.log(discount) / t


class YieldTermStructure:
    def __init__(self, reference_date: Date, day_counter: DayCounter, calendar=None):
        self._reference_date = reference_date
        self._day_counter = day_counter
        self._calendar = calendar or Calendar()
        self._extrapolate = False

    def referenceDate(self) -> Date:
        return self._reference_date

    def dayCounter(self) -> DayCounter:
        return self._day_counter

    def calendar(self):
        return self._calendar

    def timeFromReference(self, date: Date) -> float:
        return self._day_counter.yearFraction(self._reference_date, date)

    def enableExtrapolation(self):
        self._extrapolate = True

    def disableExtrapolation(self):
        self._extrapolate = False

    def allowsExtrapolation(self) -> bool:
        return self._extrapolate

    def maxTime(self) -> float:
        return self.timeFromReference(self.maxDate())

    def _times_from_input(self, value):
        if type(value) is np.ndarray and value.ndim == 1 and value.dtype.kind in "biuf":
            return np.ascontiguousarray(value, dtype=np.float64), False
        scalar = isinstance(value, (Date, int, float, np.number))
        values = [value] if scalar else list(value)
        times = [
            self.timeFromReference(x) if isinstance(x, Date) else float(x)
            for x in values
        ]
        return np.asarray(times, dtype=np.float64), scalar

    def _check_range(self, times, extrapolate):
        if np.any(times < 0):
            raise RuntimeError("negative time given")
        if np.any(times > self.maxTime()) and not (extrapolate or self._extrapolate):
            raise RuntimeError("time is past max curve time")

    def zeroRate(self, *args):
        if not args:
            raise TypeError("zeroRate requires a date or time")
        point = args[0]
        if isinstance(point, Date):
            day_counter = args[1]
            compounding = args[2]
            frequency = args[3] if len(args) > 3 else Annual
            extrapolate = args[4] if len(args) > 4 else False
            t = day_counter.yearFraction(self.referenceDate(), point)
        else:
            compounding = args[1]
            frequency = args[2] if len(args) > 2 else Annual
            extrapolate = args[3] if len(args) > 3 else False
            t = float(point)
            day_counter = self.dayCounter()
        effective_t = t if t > 1e-12 else 1e-8
        discount = self.discount(effective_t, extrapolate)
        return InterestRate(
            _rate_from_discount(discount, effective_t, compounding, frequency),
            day_counter,
            compounding,
            frequency,
        )

    def forwardRate(self, *args):
        if len(args) < 3:
            raise TypeError("forwardRate requires two dates/times and compounding")
        first, last = args[0], args[1]
        if isinstance(first, Date):
            day_counter, compounding = args[2], args[3]
            frequency = args[4] if len(args) > 4 else Annual
            extrapolate = args[5] if len(args) > 5 else False
            t1 = self.timeFromReference(first)
            span = day_counter.yearFraction(first, last)
            t2 = self.timeFromReference(last)
        else:
            compounding = args[2]
            frequency = args[3] if len(args) > 3 else Annual
            extrapolate = args[4] if len(args) > 4 else False
            t1, t2 = float(first), float(last)
            span = t2 - t1
            day_counter = self.dayCounter()
        if span <= 0:
            raise ValueError("second date/time must be later")
        compound = self.discount(t1, extrapolate) / self.discount(t2, extrapolate)
        rate = _rate_from_discount(1.0 / compound, span, compounding, frequency)
        return InterestRate(rate, day_counter, compounding, frequency)


class DiscountCurve(YieldTermStructure):
    def __init__(self, dates, discounts, dayCounter, calendar=None, i=None):
        dates = list(dates)
        discounts = f64(discounts)
        if len(dates) != discounts.size or len(dates) < 2:
            raise ValueError("dates and discounts require at least two matching nodes")
        if any(b <= a for a, b in zip(dates, dates[1:])):
            raise ValueError("dates must be strictly increasing")
        if np.any(discounts <= 0):
            raise ValueError("discounts must be positive")
        super().__init__(dates[0], dayCounter, calendar)
        self._dates = dates
        self._discounts = discounts
        self._log_discounts = f64(np.log(discounts))
        self._times = f64([self.timeFromReference(d) for d in dates])

    def dates(self):
        return list(self._dates)

    def discounts(self):
        return list(self._discounts)

    def data(self):
        return list(self._discounts)

    def times(self):
        return list(self._times)

    def nodes(self):
        return list(zip(self._dates, self._discounts.tolist()))

    def maxDate(self):
        return self._dates[-1]

    def discount(self, value, extrapolate=False):
        times, scalar = self._times_from_input(value)
        self._check_range(times, extrapolate)
        result = np.empty(times.size, dtype=np.float64)
        lib().mql_discount_curve(
            addr(self._times), addr(self._log_discounts), self._times.size,
            addr(times), addr(result), times.size,
        )
        return float(result[0]) if scalar else result


class ZeroCurve(YieldTermStructure):
    def __init__(
        self, dates, yields, dayCounter, calendar=None, i=None,
        compounding=Continuous, frequency=Annual,
    ):
        dates = list(dates)
        yields = f64(yields)
        if len(dates) != yields.size or len(dates) < 2:
            raise ValueError("dates and yields require at least two matching nodes")
        if any(b <= a for a, b in zip(dates, dates[1:])):
            raise ValueError("dates must be strictly increasing")
        super().__init__(dates[0], dayCounter, calendar)
        self._dates = dates
        self._times = f64([self.timeFromReference(d) for d in dates])
        self._compounding = int(compounding)
        self._frequency = int(frequency)
        _validate_compounding(self._compounding, self._frequency)
        if self._compounding == Simple:
            effective_times = np.where(self._times == 0.0, 1.0 / 365.0, self._times)
            self._yields = np.log1p(yields * effective_times) / effective_times
        elif self._compounding == Compounded:
            self._yields = self._frequency * np.log1p(yields / self._frequency)
        else:
            self._yields = yields
        self._yields = f64(self._yields)

    def dates(self):
        return list(self._dates)

    def data(self):
        return list(self._yields)

    def times(self):
        return list(self._times)

    def nodes(self):
        return list(zip(self._dates, self._yields.tolist()))

    def maxDate(self):
        return self._dates[-1]

    def discount(self, value, extrapolate=False):
        times, scalar = self._times_from_input(value)
        self._check_range(times, extrapolate)
        result = np.empty(times.size, dtype=np.float64)
        lib().mql_zero_curve(
            addr(self._times), addr(self._yields), self._times.size,
            addr(times), addr(result), times.size, Continuous, Annual,
        )
        return float(result[0]) if scalar else result


class FlatForward(YieldTermStructure):
    def __init__(
        self, referenceDate, forward, dayCounter, compounding=Continuous,
        frequency=Annual,
    ):
        if not isinstance(referenceDate, Date):
            raise TypeError("settlement-day constructor is not covered")
        super().__init__(referenceDate, dayCounter)
        self._forward = float(forward)
        self._compounding = int(compounding)
        self._frequency = int(frequency)
        _validate_compounding(self._compounding, self._frequency)

    def maxDate(self):
        return Date.maxDate()

    def discount(self, value, extrapolate=False):
        times, scalar = self._times_from_input(value)
        self._check_range(times, extrapolate)
        result = _discount_from_rate(
            self._forward, times, self._compounding, self._frequency
        )
        return float(result[0]) if scalar else result
