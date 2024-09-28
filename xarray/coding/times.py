from __future__ import annotations
import re
import warnings
from collections.abc import Hashable
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, Literal, Union, cast
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
from xarray.coding.variables import SerializationWarning, VariableCoder, lazy_elemwise_func, pop_to, safe_setitem, unpack_for_decoding, unpack_for_encoding
from xarray.core import indexing
from xarray.core.common import contains_cftime_datetimes, is_np_datetime_like
from xarray.core.duck_array_ops import asarray, ravel, reshape
from xarray.core.formatting import first_n_items, format_timestamp, last_item
from xarray.core.pdcompat import nanosecond_precision_timestamp
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import T_ChunkedArray, get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.namedarray.utils import is_duck_dask_array
try:
    import cftime
except ImportError:
    cftime = None
from xarray.core.types import CFCalendar, NPDatetimeUnitOptions, T_DuckArray
T_Name = Union[Hashable, None]
_STANDARD_CALENDARS = {'standard', 'gregorian', 'proleptic_gregorian'}
_NS_PER_TIME_DELTA = {'ns': 1, 'us': int(1000.0), 'ms': int(1000000.0), 's': int(1000000000.0), 'm': int(1000000000.0) * 60, 'h': int(1000000000.0) * 60 * 60, 'D': int(1000000000.0) * 60 * 60 * 24}
_US_PER_TIME_DELTA = {'microseconds': 1, 'milliseconds': 1000, 'seconds': 1000000, 'minutes': 60 * 1000000, 'hours': 60 * 60 * 1000000, 'days': 24 * 60 * 60 * 1000000}
_NETCDF_TIME_UNITS_CFTIME = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds']
_NETCDF_TIME_UNITS_NUMPY = _NETCDF_TIME_UNITS_CFTIME + ['nanoseconds']
TIME_UNITS = frozenset(['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds'])

def decode_cf_datetime(num_dates, units: str, calendar: str | None=None, use_cftime: bool | None=None) -> np.ndarray:
    """Given an array of numeric dates in netCDF format, convert it into a
    numpy array of date time objects.

    For standard (Gregorian) calendars, this function uses vectorized
    operations, which makes it much faster than cftime.num2date. In such a
    case, the returned array will be of type np.datetime64.

    Note that time unit in `units` must not be smaller than microseconds and
    not larger than days.

    See Also
    --------
    cftime.num2date
    """
    pass

def decode_cf_timedelta(num_timedeltas, units: str) -> np.ndarray:
    """Given an array of numeric timedeltas in netCDF format, convert it into a
    numpy timedelta64[ns] array.
    """
    pass

def infer_calendar_name(dates) -> CFCalendar:
    """Given an array of datetimes, infer the CF calendar name"""
    pass

def infer_datetime_units(dates) -> str:
    """Given an array of datetimes, returns a CF compatible time-unit string of
    the form "{time_unit} since {date[0]}", where `time_unit` is 'days',
    'hours', 'minutes' or 'seconds' (the first one that can evenly divide all
    unique time deltas in `dates`)
    """
    pass

def format_cftime_datetime(date) -> str:
    """Converts a cftime.datetime object to a string with the format:
    YYYY-MM-DD HH:MM:SS.UUUUUU
    """
    pass

def infer_timedelta_units(deltas) -> str:
    """Given an array of timedeltas, returns a CF compatible time-unit from
    {'days', 'hours', 'minutes' 'seconds'} (the first one that can evenly
    divide all unique time deltas in `deltas`)
    """
    pass

def cftime_to_nptime(times, raise_on_invalid: bool=True) -> np.ndarray:
    """Given an array of cftime.datetime objects, return an array of
    numpy.datetime64 objects of the same size

    If raise_on_invalid is True (default), invalid dates trigger a ValueError.
    Otherwise, the invalid element is replaced by np.NaT."""
    pass

def convert_times(times, date_type, raise_on_invalid: bool=True) -> np.ndarray:
    """Given an array of datetimes, return the same dates in another cftime or numpy date type.

    Useful to convert between calendars in numpy and cftime or between cftime calendars.

    If raise_on_valid is True (default), invalid dates trigger a ValueError.
    Otherwise, the invalid element is replaced by np.nan for cftime types and np.NaT for np.datetime64.
    """
    pass

def convert_time_or_go_back(date, date_type):
    """Convert a single date to a new date_type (cftime.datetime or pd.Timestamp).

    If the new date is invalid, it goes back a day and tries again. If it is still
    invalid, goes back a second day.

    This is meant to convert end-of-month dates into a new calendar.
    """
    pass

def _should_cftime_be_used(source, target_calendar: str, use_cftime: bool | None) -> bool:
    """Return whether conversion of the source to the target calendar should
    result in a cftime-backed array.

    Source is a 1D datetime array, target_cal a string (calendar name) and
    use_cftime is a boolean or None. If use_cftime is None, this returns True
    if the source's range and target calendar are convertible to np.datetime64 objects.
    """
    pass

def _encode_datetime_with_cftime(dates, units: str, calendar: str) -> np.ndarray:
    """Fallback method for encoding dates using cftime.

    This method is more flexible than xarray's parsing using datetime64[ns]
    arrays but also slower because it loops over each element.
    """
    pass

def encode_cf_datetime(dates: T_DuckArray, units: str | None=None, calendar: str | None=None, dtype: np.dtype | None=None) -> tuple[T_DuckArray, str, str]:
    """Given an array of datetime objects, returns the tuple `(num, units,
    calendar)` suitable for a CF compliant time variable.

    Unlike `date2num`, this function can handle datetime64 arrays.

    See Also
    --------
    cftime.date2num
    """
    pass

class CFDatetimeCoder(VariableCoder):

    def __init__(self, use_cftime: bool | None=None) -> None:
        self.use_cftime = use_cftime

class CFTimedeltaCoder(VariableCoder):
    pass