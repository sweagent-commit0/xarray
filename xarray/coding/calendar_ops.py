from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import date_range_like, get_date_type
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.coding.times import _should_cftime_be_used, convert_times
from xarray.core.common import _contains_datetime_like_objects, is_np_datetime_like
try:
    import cftime
except ImportError:
    cftime = None
_CALENDARS_WITHOUT_YEAR_ZERO = ['gregorian', 'proleptic_gregorian', 'julian', 'standard']

def _days_in_year(year, calendar, use_cftime=True):
    """Return the number of days in the input year according to the input calendar."""
    pass

def convert_calendar(obj, calendar, dim='time', align_on=None, missing=None, use_cftime=None):
    """Transform a time-indexed Dataset or DataArray to one that uses another calendar.

    This function only converts the individual timestamps; it does not modify any
    data except in dropping invalid/surplus dates, or inserting values for missing dates.

    If the source and target calendars are both from a standard type, only the
    type of the time array is modified. When converting to a calendar with a
    leap year from to a calendar without a leap year, the 29th of February will
    be removed from the array. In the other direction the 29th of February will
    be missing in the output, unless `missing` is specified, in which case that
    value is inserted. For conversions involving the `360_day` calendar, see Notes.

    This method is safe to use with sub-daily data as it doesn't touch the time
    part of the timestamps.

    Parameters
    ----------
    obj : DataArray or Dataset
      Input DataArray or Dataset with a time coordinate of a valid dtype
      (:py:class:`numpy.datetime64`  or :py:class:`cftime.datetime`).
    calendar : str
      The target calendar name.
    dim : str
      Name of the time coordinate in the input DataArray or Dataset.
    align_on : {None, 'date', 'year', 'random'}
      Must be specified when either the source or target is a `"360_day"`
      calendar; ignored otherwise. See Notes.
    missing : any, optional
      By default, i.e. if the value is None, this method will simply attempt
      to convert the dates in the source calendar to the same dates in the
      target calendar, and drop any of those that are not possible to
      represent.  If a value is provided, a new time coordinate will be
      created in the target calendar with the same frequency as the original
      time coordinate; for any dates that are not present in the source, the
      data will be filled with this value.  Note that using this mode requires
      that the source data have an inferable frequency; for more information
      see :py:func:`xarray.infer_freq`.  For certain frequency, source, and
      target calendar combinations, this could result in many missing values, see notes.
    use_cftime : bool, optional
      Whether to use cftime objects in the output, only used if `calendar` is
      one of {"proleptic_gregorian", "gregorian" or "standard"}.
      If True, the new time axis uses cftime objects.
      If None (default), it uses :py:class:`numpy.datetime64` values if the date
          range permits it, and :py:class:`cftime.datetime` objects if not.
      If False, it uses :py:class:`numpy.datetime64`  or fails.

    Returns
    -------
      Copy of source with the time coordinate converted to the target calendar.
      If `missing` was None (default), invalid dates in the new calendar are
      dropped, but missing dates are not inserted.
      If `missing` was given, the new data is reindexed to have a time axis
      with the same frequency as the source, but in the new calendar; any
      missing datapoints are filled with `missing`.

    Notes
    -----
    Passing a value to `missing` is only usable if the source's time coordinate as an
    inferable frequencies (see :py:func:`~xarray.infer_freq`) and is only appropriate
    if the target coordinate, generated from this frequency, has dates equivalent to the
    source. It is usually **not** appropriate to use this mode with:

    - Period-end frequencies: 'A', 'Y', 'Q' or 'M', in opposition to 'AS' 'YS', 'QS' and 'MS'
    - Sub-monthly frequencies that do not divide a day evenly: 'W', 'nD' where `n != 1`
      or 'mH' where 24 % m != 0).

    If one of the source or target calendars is `"360_day"`, `align_on` must
    be specified and two options are offered.

    "year"
      The dates are translated according to their relative position in the year,
      ignoring their original month and day information, meaning that the
      missing/surplus days are added/removed at regular intervals.

      From a `360_day` to a standard calendar, the output will be missing the
      following dates (day of year in parentheses):
        To a leap year:
          January 31st (31), March 31st (91), June 1st (153), July 31st (213),
          September 31st (275) and November 30th (335).
        To a non-leap year:
          February 6th (36), April 19th (109), July 2nd (183),
          September 12th (255), November 25th (329).

      From a standard calendar to a `"360_day"`, the following dates in the
      source array will be dropped:
        From a leap year:
          January 31st (31), April 1st (92), June 1st (153), August 1st (214),
          September 31st (275), December 1st (336)
        From a non-leap year:
          February 6th (37), April 20th (110), July 2nd (183),
          September 13th (256), November 25th (329)

      This option is best used on daily and subdaily data.

    "date"
      The month/day information is conserved and invalid dates are dropped
      from the output. This means that when converting from a `"360_day"` to a
      standard calendar, all 31sts (Jan, March, May, July, August, October and
      December) will be missing as there is no equivalent dates in the
      `"360_day"` calendar and the 29th (on non-leap years) and 30th of February
      will be dropped as there are no equivalent dates in a standard calendar.

      This option is best used with data on a frequency coarser than daily.

    "random"
      Similar to "year", each day of year of the source is mapped to another day of year
      of the target. However, instead of having always the same missing days according
      the source and target years, here 5 days are chosen randomly, one for each fifth
      of the year. However, February 29th is always missing when converting to a leap year,
      or its value is dropped when converting from a leap year. This is similar to the method
      used in the LOCA dataset (see Pierce, Cayan, and Thrasher (2014). doi:10.1175/JHM-D-14-0082.1).

      This option is best used on daily data.
    """
    pass

def _interpolate_day_of_year(time, target_calendar, use_cftime):
    """Returns the nearest day in the target calendar of the corresponding
    "decimal year" in the source calendar.
    """
    pass

def _random_day_of_year(time, target_calendar, use_cftime):
    """Return a day of year in the new calendar.

    Removes Feb 29th and five other days chosen randomly within five sections of 72 days.
    """
    pass

def _convert_to_new_calendar_with_new_day_of_year(date, day_of_year, calendar, use_cftime):
    """Convert a datetime object to another calendar with a new day of year.

    Redefines the day of year (and thus ignores the month and day information
    from the source datetime).
    Nanosecond information is lost as cftime.datetime doesn't support it.
    """
    pass

def _datetime_to_decimal_year(times, dim='time', calendar=None):
    """Convert a datetime DataArray to decimal years according to its calendar or the given one.

    The decimal year of a timestamp is its year plus its sub-year component
    converted to the fraction of its year.
    Ex: '2000-03-01 12:00' is 2000.1653 in a standard calendar,
      2000.16301 in a "noleap" or 2000.16806 in a "360_day".
    """
    pass

def interp_calendar(source, target, dim='time'):
    """Interpolates a DataArray or Dataset indexed by a time coordinate to
    another calendar based on decimal year measure.

    Each timestamp in `source` and `target` are first converted to their decimal
    year equivalent then `source` is interpolated on the target coordinate.
    The decimal year of a timestamp is its year plus its sub-year component
    converted to the fraction of its year. For example "2000-03-01 12:00" is
    2000.1653 in a standard calendar or 2000.16301 in a `"noleap"` calendar.

    This method should only be used when the time (HH:MM:SS) information of
    time coordinate is not important.

    Parameters
    ----------
    source: DataArray or Dataset
      The source data to interpolate; must have a time coordinate of a valid
      dtype (:py:class:`numpy.datetime64` or :py:class:`cftime.datetime` objects)
    target: DataArray, DatetimeIndex, or CFTimeIndex
      The target time coordinate of a valid dtype (np.datetime64 or cftime objects)
    dim : str
      The time coordinate name.

    Return
    ------
    DataArray or Dataset
      The source interpolated on the decimal years of target,
    """
    pass