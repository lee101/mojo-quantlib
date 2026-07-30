"""QuantLib-compatible dates, periods, calendars, and day counters."""

from __future__ import annotations

import calendar as _calendar
import datetime as _datetime
import re
from typing import Iterable

import numpy as np

from ._lib import addr, i64, lib

Days, Weeks, Months, Years = 0, 1, 2, 3
Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday = range(1, 8)
Following, ModifiedFollowing, Preceding, ModifiedPreceding = 0, 1, 2, 3
Unadjusted, HalfMonthModifiedFollowing, Nearest = 4, 5, 6

January, February, March, April, May, June = range(1, 7)
July, August, September, October, November, December = range(7, 13)

_SERIAL_OFFSET = 693_594
_MONTH_CODES = {"D": Days, "W": Weeks, "M": Months, "Y": Years}


class Period:
    def __init__(self, length, units=None):
        if units is None:
            match = re.fullmatch(r"\s*([+-]?\d+)\s*([DdWwMmYy])\s*", str(length))
            if not match:
                raise ValueError(f"invalid period: {length!r}")
            length, units = int(match.group(1)), _MONTH_CODES[match.group(2).upper()]
        self._length = int(length)
        self._units = int(units)
        if self._units not in (Days, Weeks, Months, Years):
            raise ValueError("unknown time unit")

    def length(self) -> int:
        return self._length

    def units(self) -> int:
        return self._units

    def normalized(self) -> "Period":
        if self._units == Months and self._length % 12 == 0:
            return Period(self._length // 12, Years)
        if self._units == Days and self._length % 7 == 0:
            return Period(self._length // 7, Weeks)
        return Period(self._length, self._units)

    def frequency(self) -> int:
        if self._length == 0:
            return 0
        if self._units == Years:
            return 1 // self._length if self._length == 1 else -1
        if self._units == Months and 12 % self._length == 0:
            return 12 // self._length
        if self._units == Weeks and 52 % self._length == 0:
            return 52 // self._length
        if self._units == Days and 365 % self._length == 0:
            return 365 // self._length
        return -1

    def __neg__(self):
        return Period(-self._length, self._units)

    def __repr__(self) -> str:
        return f"Period({self._length}, {self._units})"

    def __str__(self) -> str:
        return f"{self._length}{'DWMY'[self._units]}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Period):
            return NotImplemented
        a, b = self.normalized(), other.normalized()
        return (a._length, a._units) == (b._length, b._units)


class Date:
    min_year = 1901
    max_year = 2199

    def __init__(self, *args):
        if not args:
            self._serial = 0
        elif len(args) == 1:
            self._serial = int(args[0])
            if self._serial and not 367 <= self._serial <= 109_574:
                raise RuntimeError("date serial number out of range")
        elif len(args) == 3:
            day, month, year = map(int, args)
            self._validate(day, month, year)
            self._serial = _datetime.date(year, month, day).toordinal() - _SERIAL_OFFSET
        else:
            raise TypeError("Date expects (), (serial), or (day, month, year)")

    @staticmethod
    def _validate(day: int, month: int, year: int) -> None:
        if not Date.min_year <= year <= Date.max_year:
            raise RuntimeError(f"year {year} out of bound")
        _datetime.date(year, month, day)

    @classmethod
    def from_date(cls, value: _datetime.date) -> "Date":
        return cls(value.day, value.month, value.year)

    def to_date(self) -> _datetime.date:
        if self._serial == 0:
            raise RuntimeError("null date")
        return _datetime.date.fromordinal(self._serial + _SERIAL_OFFSET)

    @staticmethod
    def todaysDate() -> "Date":
        return Date.from_date(_datetime.date.today())

    @staticmethod
    def minDate() -> "Date":
        return Date(1, 1, Date.min_year)

    @staticmethod
    def maxDate() -> "Date":
        return Date(31, 12, Date.max_year)

    @staticmethod
    def isLeap(year: int) -> bool:
        return _calendar.isleap(year)

    @staticmethod
    def endOfMonth(date: "Date") -> "Date":
        return Date(_calendar.monthrange(date.year(), date.month())[1], date.month(), date.year())

    @staticmethod
    def startOfMonth(date: "Date") -> "Date":
        return Date(1, date.month(), date.year())

    @staticmethod
    def nextWeekday(date: "Date", weekday: int) -> "Date":
        return date + ((int(weekday) - date.weekday()) % 7)

    @staticmethod
    def nthWeekday(nth: int, weekday: int, month: int, year: int) -> "Date":
        first = Date(1, month, year)
        return first + ((weekday - first.weekday()) % 7) + 7 * (nth - 1)

    @staticmethod
    def ISO(date: "Date") -> str:
        return date.to_date().isoformat()

    def serialNumber(self) -> int:
        return self._serial

    def year(self) -> int:
        return self.to_date().year

    def month(self) -> int:
        return self.to_date().month

    def dayOfMonth(self) -> int:
        return self.to_date().day

    def dayOfYear(self) -> int:
        return self.to_date().timetuple().tm_yday

    def weekday(self) -> int:
        return self.to_date().isoweekday() % 7 + 1

    def isEndOfMonth(self) -> bool:
        return self.dayOfMonth() == _calendar.monthrange(self.year(), self.month())[1]

    def isStartOfMonth(self) -> bool:
        return self.dayOfMonth() == 1

    def __add__(self, value):
        if isinstance(value, Period):
            if value.units() == Days:
                return Date(self._serial + value.length())
            if value.units() == Weeks:
                return Date(self._serial + 7 * value.length())
            months = value.length() * (12 if value.units() == Years else 1)
            total = self.year() * 12 + self.month() - 1 + months
            year, month0 = divmod(total, 12)
            day = min(self.dayOfMonth(), _calendar.monthrange(year, month0 + 1)[1])
            return Date(day, month0 + 1, year)
        if isinstance(value, (int, np.integer)):
            return Date(self._serial + int(value))
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, value):
        if isinstance(value, Date):
            return self._serial - value._serial
        if isinstance(value, Period):
            return self + (-value)
        if isinstance(value, (int, np.integer)):
            return Date(self._serial - int(value))
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._serial)

    def __eq__(self, other) -> bool:
        return isinstance(other, Date) and self._serial == other._serial

    def __lt__(self, other) -> bool:
        return self._serial < _as_date(other)._serial

    def __le__(self, other) -> bool:
        return self._serial <= _as_date(other)._serial

    def __gt__(self, other) -> bool:
        return self._serial > _as_date(other)._serial

    def __ge__(self, other) -> bool:
        return self._serial >= _as_date(other)._serial

    def __repr__(self) -> str:
        if not self._serial:
            return "Date()"
        return f"Date({self.dayOfMonth()}, {self.month()}, {self.year()})"

    def __str__(self) -> str:
        return self.to_date().strftime("%B %d, %Y")


def _as_date(value) -> Date:
    return value if isinstance(value, Date) else Date(int(value))


def date_serials(years, months, days):
    years, months, days = np.broadcast_arrays(i64(years), i64(months), i64(days))
    leap = (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
    lengths = np.choose(
        months - 1,
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        mode="clip",
    ) + ((months == 2) & leap)
    if (
        np.any((years < Date.min_year) | (years > Date.max_year))
        or np.any((months < 1) | (months > 12))
        or np.any((days < 1) | (days > lengths))
    ):
        raise ValueError("invalid date component")
    shape = years.shape
    years, months, days = (np.ascontiguousarray(x.ravel()) for x in (years, months, days))
    result = np.empty(years.size, dtype=np.int64)
    lib().mql_dates_to_serial(
        addr(years), addr(months), addr(days), addr(result), result.size
    )
    return result.reshape(shape)


def dates_from_serial(serials):
    serials = i64(serials)
    if np.any((serials < 367) | (serials > 109_574)):
        raise ValueError("serial number out of range")
    shape = serials.shape
    flat = np.ascontiguousarray(serials.ravel())
    y = np.empty(flat.size, dtype=np.int64)
    m = np.empty_like(y)
    d = np.empty_like(y)
    lib().mql_serial_to_dates(addr(flat), addr(y), addr(m), addr(d), flat.size)
    return y.reshape(shape), m.reshape(shape), d.reshape(shape)


class Calendar:
    def __init__(self):
        self._added: set[int] = set()
        self._removed: set[int] = set()

    def name(self) -> str:
        return "Calendar"

    def isWeekend(self, weekday: int) -> bool:
        return weekday in (Saturday, Sunday)

    def _base_business_day(self, date: Date) -> bool:
        return not self.isWeekend(date.weekday())

    def isBusinessDay(self, date: Date) -> bool:
        serial = date.serialNumber()
        if serial in self._added:
            return False
        if serial in self._removed:
            return True
        return self._base_business_day(date)

    def isHoliday(self, date: Date) -> bool:
        return not self.isBusinessDay(date)

    def addHoliday(self, date: Date) -> None:
        self._removed.discard(date.serialNumber())
        self._added.add(date.serialNumber())

    def removeHoliday(self, date: Date) -> None:
        self._added.discard(date.serialNumber())
        self._removed.add(date.serialNumber())

    def resetAddedAndRemovedHolidays(self) -> None:
        self._added.clear()
        self._removed.clear()

    def addedHolidays(self):
        return [Date(x) for x in sorted(self._added)]

    def removedHolidays(self):
        return [Date(x) for x in sorted(self._removed)]

    def adjust(self, date: Date, convention: int = Following) -> Date:
        if convention == Unadjusted or self.isBusinessDay(date):
            return Date(date.serialNumber())
        if convention in (Following, ModifiedFollowing, HalfMonthModifiedFollowing):
            adjusted = Date(date.serialNumber())
            while self.isHoliday(adjusted):
                adjusted += 1
            modified = (
                convention == ModifiedFollowing and adjusted.month() != date.month()
            )
            if convention == HalfMonthModifiedFollowing:
                modified |= date.dayOfMonth() <= 15 < adjusted.dayOfMonth()
            return self.adjust(date, Preceding) if modified else adjusted
        if convention in (Preceding, ModifiedPreceding):
            adjusted = Date(date.serialNumber())
            while self.isHoliday(adjusted):
                adjusted -= 1
            if convention == ModifiedPreceding and adjusted.month() != date.month():
                return self.adjust(date, Following)
            return adjusted
        if convention == Nearest:
            before, after = date - 1, date + 1
            while True:
                if self.isBusinessDay(before):
                    return before
                if self.isBusinessDay(after):
                    return after
                before -= 1
                after += 1
        raise ValueError("unknown business-day convention")

    def advance(self, date: Date, n, unit=None, convention=Following, endOfMonth=False) -> Date:
        period = n if isinstance(n, Period) else Period(n, unit)
        if period.length() == 0:
            return self.adjust(date, convention)
        if period.units() == Days:
            step = 1 if period.length() > 0 else -1
            result = Date(date.serialNumber())
            for _ in range(abs(period.length())):
                result += step
                while self.isHoliday(result):
                    result += step
            return result
        if period.units() == Weeks:
            return self.adjust(date + period, convention)
        result = date + period
        if endOfMonth and self.isEndOfMonth(date):
            return self.endOfMonth(result)
        return self.adjust(result, convention)

    def businessDaysBetween(
        self, first: Date, last: Date, includeFirst=True, includeLast=False
    ) -> int:
        if first == last:
            return int(includeFirst and includeLast and self.isBusinessDay(first))
        sign = 1 if first < last else -1
        lo, hi = (first, last) if sign > 0 else (last, first)
        count = sum(
            self.isBusinessDay(Date(serial))
            for serial in range(lo.serialNumber(), hi.serialNumber() + 1)
        )
        if not includeFirst and self.isBusinessDay(first):
            count -= 1
        if not includeLast and self.isBusinessDay(last):
            count -= 1
        return sign * count

    def endOfMonth(self, date: Date) -> Date:
        result = Date.endOfMonth(date)
        while self.isHoliday(result):
            result -= 1
        return result

    def startOfMonth(self, date: Date) -> Date:
        result = Date.startOfMonth(date)
        while self.isHoliday(result):
            result += 1
        return result

    def isEndOfMonth(self, date: Date) -> bool:
        return date == self.endOfMonth(date)

    def isStartOfMonth(self, date: Date) -> bool:
        return date == self.startOfMonth(date)


class WeekendsOnly(Calendar):
    def name(self) -> str:
        return "weekends only"


class NullCalendar(Calendar):
    def name(self) -> str:
        return "Null"

    def isWeekend(self, weekday: int) -> bool:
        return False


def _easter_sunday(year: int) -> _datetime.date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day0 = divmod(h + l - 7 * m + 114, 31)
    return _datetime.date(year, month, day0 + 1)


class TARGET(Calendar):
    def name(self) -> str:
        return "TARGET"

    def _base_business_day(self, date: Date) -> bool:
        if super()._base_business_day(date) is False:
            return False
        day, month, year = date.dayOfMonth(), date.month(), date.year()
        easter = _easter_sunday(year)
        pydate = date.to_date()
        return not (
            (day == 1 and month == January)
            or pydate in (easter - _datetime.timedelta(days=2), easter + _datetime.timedelta(days=1))
            or (year >= 2000 and day == 1 and month == May)
            or (day == 25 and month == December)
            or (year >= 2000 and day == 26 and month == December)
            or (day == 31 and month == December and year in (1998, 1999, 2001))
        )


class DayCounter:
    def dayCount(self, first: Date, last: Date) -> int:
        return last - first

    def yearFraction(self, first: Date, last: Date, refStart=None, refEnd=None) -> float:
        raise NotImplementedError

    def empty(self) -> bool:
        return False


class Actual365Fixed(DayCounter):
    Standard, Canadian, NoLeap = 0, 1, 2

    def __init__(self, convention=Standard):
        self.convention = int(convention)

    def name(self) -> str:
        names = {self.Standard: "Actual/365 (Fixed)", self.Canadian: "Actual/365 (Fixed) Canadian Bond", self.NoLeap: "Actual/365 (No Leap)"}
        return names[self.convention]

    def dayCount(self, first: Date, last: Date) -> int:
        days = last - first
        if self.convention != self.NoLeap or days == 0:
            return days
        sign = 1 if days > 0 else -1
        start, end = (first, last) if sign > 0 else (last, first)
        leap_days = sum(
            start <= Date(29, February, year) < end
            for year in range(start.year(), end.year() + 1)
            if _calendar.isleap(year)
        )
        return sign * (abs(days) - leap_days)

    def yearFraction(self, first, last, refStart=None, refEnd=None) -> float:
        if self.convention == self.Canadian:
            if refStart is None or refEnd is None:
                raise ValueError("Canadian convention requires reference dates")
            dcs = refEnd - refStart
            months = round(12 * dcs / 365)
            frequency = 12 / months
            dcc = last - first
            return dcc / (365.0 if dcc < 365 / frequency else frequency * dcs)
        return self.dayCount(first, last) / 365.0


class Actual360(DayCounter):
    def __init__(self, includeLastDay=False):
        self.includeLastDay = bool(includeLastDay)

    def name(self) -> str:
        return "Actual/360 (inc)" if self.includeLastDay else "Actual/360"

    def dayCount(self, first, last) -> int:
        return last - first + int(self.includeLastDay)

    def yearFraction(self, first, last, refStart=None, refEnd=None) -> float:
        return self.dayCount(first, last) / 360.0


class Thirty360(DayCounter):
    USA, BondBasis, European, EurobondBasis, Italian, German, ISMA, ISDA, NASD = range(9)

    def __init__(self, convention=BondBasis, terminationDate=None):
        self.convention = int(convention)
        self.terminationDate = terminationDate

    def name(self) -> str:
        return "30/360"

    def dayCount(self, first, last) -> int:
        y1, m1, d1 = first.year(), first.month(), first.dayOfMonth()
        y2, m2, d2 = last.year(), last.month(), last.dayOfMonth()
        if self.convention in (self.European, self.EurobondBasis):
            d1, d2 = min(d1, 30), min(d2, 30)
        elif self.convention in (self.BondBasis, self.ISMA):
            d1 = min(d1, 30)
            if d1 == 30:
                d2 = min(d2, 30)
        elif self.convention == self.Italian:
            d1 = 30 if m1 == 2 and d1 > 27 else min(d1, 30)
            d2 = 30 if m2 == 2 and d2 > 27 else min(d2, 30)
        else:
            first_feb_end = m1 == 2 and first.isEndOfMonth()
            last_feb_end = m2 == 2 and last.isEndOfMonth()
            if self.convention == self.USA:
                if first_feb_end:
                    d1 = 30
                if last_feb_end and first_feb_end:
                    d2 = 30
                if d2 == 31 and d1 >= 30:
                    d2 = 30
                if d1 == 31:
                    d1 = 30
            elif self.convention in (self.ISDA, self.German):
                if first.isEndOfMonth():
                    d1 = 30
                if last.isEndOfMonth() and not (
                    self.terminationDate == last and m2 == February
                ):
                    d2 = 30
            elif self.convention == self.NASD:
                if d1 == 31:
                    d1 = 30
                if d2 == 31:
                    if d1 < 30:
                        d2 = 1
                        m2 += 1
                        if m2 == 13:
                            m2, y2 = 1, y2 + 1
                    else:
                        d2 = 30
        return 360 * (y2 - y1) + 30 * (m2 - m1) + d2 - d1

    def yearFraction(self, first, last, refStart=None, refEnd=None) -> float:
        return self.dayCount(first, last) / 360.0


class ActualActual(DayCounter):
    ISMA, Bond, ISDA, Historical, Actual365, AFB, Euro = range(7)

    def __init__(self, convention=ISDA, schedule=None):
        self.convention = int(convention)
        if self.convention not in (self.ISDA, self.Historical, self.Actual365):
            raise NotImplementedError("only the Actual/Actual ISDA convention is covered")
        self.schedule = schedule

    def name(self) -> str:
        return "Actual/Actual (ISDA)"

    def yearFraction(self, first, last, refStart=None, refEnd=None) -> float:
        if first == last:
            return 0.0
        if first > last:
            return -self.yearFraction(last, first, refStart, refEnd)
        total = 0.0
        cursor = first
        while cursor.year() < last.year():
            boundary = Date(1, January, cursor.year() + 1)
            total += (boundary - cursor) / (366.0 if Date.isLeap(cursor.year()) else 365.0)
            cursor = boundary
        return total + (last - cursor) / (366.0 if Date.isLeap(last.year()) else 365.0)


def year_fractions(starts: Iterable[Date], ends: Iterable[Date], day_counter: DayCounter):
    starts = list(starts)
    ends = list(ends)
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length")
    if isinstance(day_counter, (Actual365Fixed, Actual360)) and not (
        isinstance(day_counter, Actual365Fixed) and day_counter.convention != Actual365Fixed.Standard
    ):
        a = i64([d.serialNumber() for d in starts])
        b = i64([d.serialNumber() for d in ends])
        result = np.empty(len(a), dtype=np.float64)
        basis = 1 if isinstance(day_counter, Actual360) else 0
        lib().mql_year_fractions(addr(a), addr(b), addr(result), len(a), basis)
        if isinstance(day_counter, Actual360) and day_counter.includeLastDay:
            result += 1 / 360
        return result
    return np.array([day_counter.yearFraction(a, b) for a, b in zip(starts, ends)])
