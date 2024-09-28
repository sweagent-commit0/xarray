"""FrequencyInferer analog for cftime.datetime objects"""
from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
_ONE_MICRO = 1
_ONE_MILLI = _ONE_MICRO * 1000
_ONE_SECOND = _ONE_MILLI * 1000
_ONE_MINUTE = 60 * _ONE_SECOND
_ONE_HOUR = 60 * _ONE_MINUTE
_ONE_DAY = 24 * _ONE_HOUR

def infer_freq(index):
    """
    Infer the most likely frequency given the input index.

    Parameters
    ----------
    index : CFTimeIndex, DataArray, DatetimeIndex, TimedeltaIndex, Series
        If not passed a CFTimeIndex, this simply calls `pandas.infer_freq`.
        If passed a Series or a DataArray will use the values of the series (NOT THE INDEX).

    Returns
    -------
    str or None
        None if no discernible frequency.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If there are fewer than three values or the index is not 1D.
    """
    pass

class _CFTimeFrequencyInferer:

    def __init__(self, index):
        self.index = index
        self.values = index.asi8
        if len(index) < 3:
            raise ValueError('Need at least 3 dates to infer frequency')
        self.is_monotonic = self.index.is_monotonic_decreasing or self.index.is_monotonic_increasing
        self._deltas = None
        self._year_deltas = None
        self._month_deltas = None

    def get_freq(self):
        """Find the appropriate frequency string to describe the inferred frequency of self.index

        Adapted from `pandas.tsseries.frequencies._FrequencyInferer.get_freq` for CFTimeIndexes.

        Returns
        -------
        str or None
        """
        pass

    @property
    def deltas(self):
        """Sorted unique timedeltas as microseconds."""
        pass

    @property
    def year_deltas(self):
        """Sorted unique year deltas."""
        pass

    @property
    def month_deltas(self):
        """Sorted unique month deltas."""
        pass

def _unique_deltas(arr):
    """Sorted unique deltas of numpy array"""
    pass

def _is_multiple(us, mult: int):
    """Whether us is a multiple of mult"""
    pass

def _maybe_add_count(base: str, count: float):
    """If count is greater than 1, add it to the base offset string"""
    pass

def month_anchor_check(dates):
    """Return the monthly offset string.

    Return "cs" if all dates are the first days of the month,
    "ce" if all dates are the last day of the month,
    None otherwise.

    Replicated pandas._libs.tslibs.resolution.month_position_check
    but without business offset handling.
    """
    pass