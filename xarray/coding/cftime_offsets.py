"""Time offset classes for use with cftime.datetime objects"""
from __future__ import annotations
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Literal
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from xarray.coding.times import _is_standard_calendar, _should_cftime_be_used, convert_time_or_go_back, format_cftime_datetime
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
from xarray.core.pdcompat import NoDefault, count_not_none, nanosecond_precision_timestamp, no_default
from xarray.core.utils import emit_user_level_warning
try:
    import cftime
except ImportError:
    cftime = None
if TYPE_CHECKING:
    from xarray.core.types import InclusiveOptions, Self, SideOptions, TypeAlias
DayOption: TypeAlias = Literal['start', 'end']

def get_date_type(calendar, use_cftime=True):
    """Return the cftime date type for a given calendar name."""
    pass

class BaseCFTimeOffset:
    _freq: ClassVar[str | None] = None
    _day_option: ClassVar[DayOption | None] = None
    n: int

    def __init__(self, n: int=1) -> None:
        if not isinstance(n, int):
            raise TypeError(f"The provided multiple 'n' must be an integer. Instead a value of type {type(n)!r} was provided.")
        self.n = n

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseCFTimeOffset):
            return NotImplemented
        return self.n == other.n and self.rule_code() == other.rule_code()

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __add__(self, other):
        return self.__apply__(other)

    def __sub__(self, other):
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract a cftime.datetime from a time offset.')
        elif type(other) == type(self):
            return type(self)(self.n - other.n)
        else:
            return NotImplemented

    def __mul__(self, other: int) -> Self:
        if not isinstance(other, int):
            return NotImplemented
        return type(self)(n=other * self.n)

    def __neg__(self) -> Self:
        return self * -1

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, BaseCFTimeOffset) and type(self) != type(other):
            raise TypeError('Cannot subtract cftime offsets of differing types')
        return -self + other

    def __apply__(self, other):
        return NotImplemented

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        pass

    def __str__(self):
        return f'<{type(self).__name__}: n={self.n}>'

    def __repr__(self):
        return str(self)

class Tick(BaseCFTimeOffset):

    def __mul__(self, other: int | float) -> Tick:
        if not isinstance(other, (int, float)):
            return NotImplemented
        if isinstance(other, float):
            n = other * self.n
            if np.isclose(n % 1, 0):
                return type(self)(int(n))
            new_self = self._next_higher_resolution()
            return new_self * other
        return type(self)(n=other * self.n)

    def as_timedelta(self) -> timedelta:
        """All Tick subclasses must implement an as_timedelta method."""
        pass

def _get_day_of_month(other, day_option: DayOption) -> int:
    """Find the day in `other`'s month that satisfies a BaseCFTimeOffset's
    onOffset policy, as described by the `day_option` argument.

    Parameters
    ----------
    other : cftime.datetime
    day_option : 'start', 'end'
        'start': returns 1
        'end': returns last day of the month

    Returns
    -------
    day_of_month : int

    """
    pass

def _days_in_month(date):
    """The number of days in the month of the given date"""
    pass

def _adjust_n_months(other_day, n, reference_day):
    """Adjust the number of times a monthly offset is applied based
    on the day of a given date, and the reference day provided.
    """
    pass

def _adjust_n_years(other, n, month, reference_day):
    """Adjust the number of times an annual offset is applied based on
    another date, and the reference day provided"""
    pass

def _shift_month(date, months, day_option: DayOption='start'):
    """Shift the date to a month start or end a given number of months away."""
    pass

def roll_qtrday(other, n: int, month: int, day_option: DayOption, modby: int=3) -> int:
    """Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : cftime.datetime
    n : number of periods to increment, before adjusting for rolling
    month : int reference month giving the first month of the year
    day_option : 'start', 'end'
        The convention to use in finding the day in a given month against
        which to compare for rollforward/rollbackward decisions.
    modby : int 3 for quarters, 12 for years

    Returns
    -------
    n : int number of periods to increment

    See Also
    --------
    _get_day_of_month : Find the day in a month provided an offset.
    """
    pass

class MonthBegin(BaseCFTimeOffset):
    _freq = 'MS'

    def __apply__(self, other):
        n = _adjust_n_months(other.day, self.n, 1)
        return _shift_month(other, n, 'start')

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        pass

class MonthEnd(BaseCFTimeOffset):
    _freq = 'ME'

    def __apply__(self, other):
        n = _adjust_n_months(other.day, self.n, _days_in_month(other))
        return _shift_month(other, n, 'end')

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        pass
_MONTH_ABBREVIATIONS = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}

class QuarterOffset(BaseCFTimeOffset):
    """Quarter representation copied off of pandas/tseries/offsets.py"""
    _default_month: ClassVar[int]
    month: int

    def __init__(self, n: int=1, month: int | None=None) -> None:
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        months_since = other.month % 3 - self.month % 3
        qtrs = roll_qtrday(other, self.n, self.month, day_option=self._day_option, modby=3)
        months = qtrs * 3 - months_since
        return _shift_month(other, months, self._day_option)

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        pass

    def __sub__(self, other: Self) -> Self:
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract cftime.datetime from offset.')
        if type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)

    def __str__(self):
        return f'<{type(self).__name__}: n={self.n}, month={self.month}>'

class QuarterBegin(QuarterOffset):
    _default_month = 3
    _freq = 'QS'
    _day_option = 'start'

    def rollforward(self, date):
        """Roll date forward to nearest start of quarter"""
        pass

    def rollback(self, date):
        """Roll date backward to nearest start of quarter"""
        pass

class QuarterEnd(QuarterOffset):
    _default_month = 3
    _freq = 'QE'
    _day_option = 'end'

    def rollforward(self, date):
        """Roll date forward to nearest end of quarter"""
        pass

    def rollback(self, date):
        """Roll date backward to nearest end of quarter"""
        pass

class YearOffset(BaseCFTimeOffset):
    _default_month: ClassVar[int]
    month: int

    def __init__(self, n: int=1, month: int | None=None) -> None:
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        reference_day = _get_day_of_month(other, self._day_option)
        years = _adjust_n_years(other, self.n, self.month, reference_day)
        months = years * 12 + (self.month - other.month)
        return _shift_month(other, months, self._day_option)

    def __sub__(self, other):
        if cftime is None:
            raise ModuleNotFoundError("No module named 'cftime'")
        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract cftime.datetime from offset.')
        elif type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)

    def __str__(self) -> str:
        return f'<{type(self).__name__}: n={self.n}, month={self.month}>'

class YearBegin(YearOffset):
    _freq = 'YS'
    _day_option = 'start'
    _default_month = 1

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        pass

    def rollforward(self, date):
        """Roll date forward to nearest start of year"""
        pass

    def rollback(self, date):
        """Roll date backward to nearest start of year"""
        pass

class YearEnd(YearOffset):
    _freq = 'YE'
    _day_option = 'end'
    _default_month = 12

    def onOffset(self, date) -> bool:
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        pass

    def rollforward(self, date):
        """Roll date forward to nearest end of year"""
        pass

    def rollback(self, date):
        """Roll date backward to nearest end of year"""
        pass

class Day(Tick):
    _freq = 'D'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Hour(Tick):
    _freq = 'h'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Minute(Tick):
    _freq = 'min'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Second(Tick):
    _freq = 's'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Millisecond(Tick):
    _freq = 'ms'

    def __apply__(self, other):
        return other + self.as_timedelta()

class Microsecond(Tick):
    _freq = 'us'

    def __apply__(self, other):
        return other + self.as_timedelta()
_FREQUENCIES: Mapping[str, type[BaseCFTimeOffset]] = {'A': YearEnd, 'AS': YearBegin, 'Y': YearEnd, 'YE': YearEnd, 'YS': YearBegin, 'Q': partial(QuarterEnd, month=12), 'QE': partial(QuarterEnd, month=12), 'QS': partial(QuarterBegin, month=1), 'M': MonthEnd, 'ME': MonthEnd, 'MS': MonthBegin, 'D': Day, 'H': Hour, 'h': Hour, 'T': Minute, 'min': Minute, 'S': Second, 's': Second, 'L': Millisecond, 'ms': Millisecond, 'U': Microsecond, 'us': Microsecond, **_generate_anchored_offsets('AS', YearBegin), **_generate_anchored_offsets('A', YearEnd), **_generate_anchored_offsets('YS', YearBegin), **_generate_anchored_offsets('Y', YearEnd), **_generate_anchored_offsets('YE', YearEnd), **_generate_anchored_offsets('QS', QuarterBegin), **_generate_anchored_offsets('Q', QuarterEnd), **_generate_anchored_offsets('QE', QuarterEnd)}
_FREQUENCY_CONDITION = '|'.join(_FREQUENCIES.keys())
_PATTERN = f'^((?P<multiple>[+-]?\\d+)|())(?P<freq>({_FREQUENCY_CONDITION}))$'
CFTIME_TICKS = (Day, Hour, Minute, Second)
_DEPRECATED_FREQUENICES: dict[str, str] = {'A': 'YE', 'Y': 'YE', 'AS': 'YS', 'Q': 'QE', 'M': 'ME', 'H': 'h', 'T': 'min', 'S': 's', 'L': 'ms', 'U': 'us', **_generate_anchored_deprecated_frequencies('A', 'YE'), **_generate_anchored_deprecated_frequencies('Y', 'YE'), **_generate_anchored_deprecated_frequencies('AS', 'YS'), **_generate_anchored_deprecated_frequencies('Q', 'QE')}
_DEPRECATION_MESSAGE = '{deprecated_freq!r} is deprecated and will be removed in a future version. Please use {recommended_freq!r} instead of {deprecated_freq!r}.'

def to_offset(freq: BaseCFTimeOffset | str, warn: bool=True) -> BaseCFTimeOffset:
    """Convert a frequency string to the appropriate subclass of
    BaseCFTimeOffset."""
    pass

def normalize_date(date):
    """Round datetime down to midnight."""
    pass

def _maybe_normalize_date(date, normalize):
    """Round datetime down to midnight if normalize is True."""
    pass

def _generate_linear_range(start, end, periods):
    """Generate an equally-spaced sequence of cftime.datetime objects between
    and including two dates (whose length equals the number of periods)."""
    pass

def _generate_range(start, end, periods, offset):
    """Generate a regular range of cftime.datetime objects with a
    given time offset.

    Adapted from pandas.tseries.offsets.generate_range (now at
    pandas.core.arrays.datetimes._generate_range).

    Parameters
    ----------
    start : cftime.datetime, or None
        Start of range
    end : cftime.datetime, or None
        End of range
    periods : int, or None
        Number of elements in the sequence
    offset : BaseCFTimeOffset
        An offset class designed for working with cftime.datetime objects

    Returns
    -------
    A generator object
    """
    pass

def _translate_closed_to_inclusive(closed):
    """Follows code added in pandas #43504."""
    pass

def _infer_inclusive(closed: NoDefault | SideOptions, inclusive: InclusiveOptions | None) -> InclusiveOptions:
    """Follows code added in pandas #43504."""
    pass

def cftime_range(start=None, end=None, periods=None, freq=None, normalize=False, name=None, closed: NoDefault | SideOptions=no_default, inclusive: None | InclusiveOptions=None, calendar='standard') -> CFTimeIndex:
    """Return a fixed frequency CFTimeIndex.

    Parameters
    ----------
    start : str or cftime.datetime, optional
        Left bound for generating dates.
    end : str or cftime.datetime, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str or None, default: "D"
        Frequency strings can have multiples, e.g. "5h" and negative values, e.g. "-1D".
    normalize : bool, default: False
        Normalize start/end dates to midnight before generating date range.
    name : str, default: None
        Name of the resulting index
    closed : {None, "left", "right"}, default: "NO_DEFAULT"
        Make the interval closed with respect to the given frequency to the
        "left", "right", or both sides (None).

        .. deprecated:: 2023.02.0
            Following pandas, the ``closed`` parameter is deprecated in favor
            of the ``inclusive`` parameter, and will be removed in a future
            version of xarray.

    inclusive : {None, "both", "neither", "left", "right"}, default None
        Include boundaries; whether to set each bound as closed or open.

        .. versionadded:: 2023.02.0

    calendar : str, default: "standard"
        Calendar type for the datetimes.

    Returns
    -------
    CFTimeIndex

    Notes
    -----
    This function is an analog of ``pandas.date_range`` for use in generating
    sequences of ``cftime.datetime`` objects.  It supports most of the
    features of ``pandas.date_range`` (e.g. specifying how the index is
    ``closed`` on either side, or whether or not to ``normalize`` the start and
    end bounds); however, there are some notable exceptions:

    - You cannot specify a ``tz`` (time zone) argument.
    - Start or end dates specified as partial-datetime strings must use the
      `ISO-8601 format <https://en.wikipedia.org/wiki/ISO_8601>`_.
    - It supports many, but not all, frequencies supported by
      ``pandas.date_range``.  For example it does not currently support any of
      the business-related or semi-monthly frequencies.
    - Compound sub-monthly frequencies are not supported, e.g. '1H1min', as
      these can easily be written in terms of the finest common resolution,
      e.g. '61min'.

    Valid simple frequency strings for use with ``cftime``-calendars include
    any multiples of the following.

    +--------+--------------------------+
    | Alias  | Description              |
    +========+==========================+
    | YE     | Year-end frequency       |
    +--------+--------------------------+
    | YS     | Year-start frequency     |
    +--------+--------------------------+
    | QE     | Quarter-end frequency    |
    +--------+--------------------------+
    | QS     | Quarter-start frequency  |
    +--------+--------------------------+
    | ME     | Month-end frequency      |
    +--------+--------------------------+
    | MS     | Month-start frequency    |
    +--------+--------------------------+
    | D      | Day frequency            |
    +--------+--------------------------+
    | h      | Hour frequency           |
    +--------+--------------------------+
    | min    | Minute frequency         |
    +--------+--------------------------+
    | s      | Second frequency         |
    +--------+--------------------------+
    | ms     | Millisecond frequency    |
    +--------+--------------------------+
    | us     | Microsecond frequency    |
    +--------+--------------------------+

    Any multiples of the following anchored offsets are also supported.

    +------------+--------------------------------------------------------------------+
    | Alias      | Description                                                        |
    +============+====================================================================+
    | Y(E,S)-JAN | Annual frequency, anchored at the (end, beginning) of January      |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-FEB | Annual frequency, anchored at the (end, beginning) of February     |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-MAR | Annual frequency, anchored at the (end, beginning) of March        |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-APR | Annual frequency, anchored at the (end, beginning) of April        |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-MAY | Annual frequency, anchored at the (end, beginning) of May          |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-JUN | Annual frequency, anchored at the (end, beginning) of June         |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-JUL | Annual frequency, anchored at the (end, beginning) of July         |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-AUG | Annual frequency, anchored at the (end, beginning) of August       |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-SEP | Annual frequency, anchored at the (end, beginning) of September    |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-OCT | Annual frequency, anchored at the (end, beginning) of October      |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-NOV | Annual frequency, anchored at the (end, beginning) of November     |
    +------------+--------------------------------------------------------------------+
    | Y(E,S)-DEC | Annual frequency, anchored at the (end, beginning) of December     |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-JAN | Quarter frequency, anchored at the (end, beginning) of January     |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-FEB | Quarter frequency, anchored at the (end, beginning) of February    |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-MAR | Quarter frequency, anchored at the (end, beginning) of March       |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-APR | Quarter frequency, anchored at the (end, beginning) of April       |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-MAY | Quarter frequency, anchored at the (end, beginning) of May         |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-JUN | Quarter frequency, anchored at the (end, beginning) of June        |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-JUL | Quarter frequency, anchored at the (end, beginning) of July        |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-AUG | Quarter frequency, anchored at the (end, beginning) of August      |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-SEP | Quarter frequency, anchored at the (end, beginning) of September   |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-OCT | Quarter frequency, anchored at the (end, beginning) of October     |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-NOV | Quarter frequency, anchored at the (end, beginning) of November    |
    +------------+--------------------------------------------------------------------+
    | Q(E,S)-DEC | Quarter frequency, anchored at the (end, beginning) of December    |
    +------------+--------------------------------------------------------------------+

    Finally, the following calendar aliases are supported.

    +--------------------------------+---------------------------------------+
    | Alias                          | Date type                             |
    +================================+=======================================+
    | standard, gregorian            | ``cftime.DatetimeGregorian``          |
    +--------------------------------+---------------------------------------+
    | proleptic_gregorian            | ``cftime.DatetimeProlepticGregorian`` |
    +--------------------------------+---------------------------------------+
    | noleap, 365_day                | ``cftime.DatetimeNoLeap``             |
    +--------------------------------+---------------------------------------+
    | all_leap, 366_day              | ``cftime.DatetimeAllLeap``            |
    +--------------------------------+---------------------------------------+
    | 360_day                        | ``cftime.Datetime360Day``             |
    +--------------------------------+---------------------------------------+
    | julian                         | ``cftime.DatetimeJulian``             |
    +--------------------------------+---------------------------------------+

    Examples
    --------
    This function returns a ``CFTimeIndex``, populated with ``cftime.datetime``
    objects associated with the specified calendar type, e.g.

    >>> xr.cftime_range(start="2000", periods=6, freq="2MS", calendar="noleap")
    CFTimeIndex([2000-01-01 00:00:00, 2000-03-01 00:00:00, 2000-05-01 00:00:00,
                 2000-07-01 00:00:00, 2000-09-01 00:00:00, 2000-11-01 00:00:00],
                dtype='object', length=6, calendar='noleap', freq='2MS')

    As in the standard pandas function, three of the ``start``, ``end``,
    ``periods``, or ``freq`` arguments must be specified at a given time, with
    the other set to ``None``.  See the `pandas documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html>`_
    for more examples of the behavior of ``date_range`` with each of the
    parameters.

    See Also
    --------
    pandas.date_range
    """
    pass

def date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed: NoDefault | SideOptions=no_default, inclusive: None | InclusiveOptions=None, calendar='standard', use_cftime=None):
    """Return a fixed frequency datetime index.

    The type (:py:class:`xarray.CFTimeIndex` or :py:class:`pandas.DatetimeIndex`)
    of the returned index depends on the requested calendar and on `use_cftime`.

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str or None, default: "D"
        Frequency strings can have multiples, e.g. "5h" and negative values, e.g. "-1D".
    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex, for example
        'Asia/Hong_Kong'. By default, the resulting DatetimeIndex is
        timezone-naive. Only valid with pandas DatetimeIndex.
    normalize : bool, default: False
        Normalize start/end dates to midnight before generating date range.
    name : str, default: None
        Name of the resulting index
    closed : {None, "left", "right"}, default: "NO_DEFAULT"
        Make the interval closed with respect to the given frequency to the
        "left", "right", or both sides (None).

        .. deprecated:: 2023.02.0
            Following pandas, the `closed` parameter is deprecated in favor
            of the `inclusive` parameter, and will be removed in a future
            version of xarray.

    inclusive : {None, "both", "neither", "left", "right"}, default: None
        Include boundaries; whether to set each bound as closed or open.

        .. versionadded:: 2023.02.0

    calendar : str, default: "standard"
        Calendar type for the datetimes.
    use_cftime : boolean, optional
        If True, always return a CFTimeIndex.
        If False, return a pd.DatetimeIndex if possible or raise a ValueError.
        If None (default), return a pd.DatetimeIndex if possible,
        otherwise return a CFTimeIndex. Defaults to False if `tz` is not None.

    Returns
    -------
    CFTimeIndex or pd.DatetimeIndex

    See also
    --------
    pandas.date_range
    cftime_range
    date_range_like
    """
    pass

def date_range_like(source, calendar, use_cftime=None):
    """Generate a datetime array with the same frequency, start and end as
    another one, but in a different calendar.

    Parameters
    ----------
    source : DataArray, CFTimeIndex, or pd.DatetimeIndex
        1D datetime array
    calendar : str
        New calendar name.
    use_cftime : bool, optional
        If True, the output uses :py:class:`cftime.datetime` objects.
        If None (default), :py:class:`numpy.datetime64` values are used if possible.
        If False, :py:class:`numpy.datetime64` values are used or an error is raised.

    Returns
    -------
    DataArray
        1D datetime coordinate with the same start, end and frequency as the
        source, but in the new calendar. The start date is assumed to exist in
        the target calendar. If the end date doesn't exist, the code tries 1
        and 2 calendar days before. There is a special case when the source time
        series is daily or coarser and the end of the input range is on the
        last day of the month. Then the output range will also end on the last
        day of the month in the new calendar.
    """
    pass