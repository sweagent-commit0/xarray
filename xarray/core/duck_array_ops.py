"""Compatibility module defining operations on duck numpy-arrays.

Currently, this means Dask or NumPy arrays. None of these functions should
accept or return xarray objects.
"""
from __future__ import annotations
import contextlib
import datetime
import inspect
import warnings
from functools import partial
from importlib import import_module
import numpy as np
import pandas as pd
from numpy import all as array_all
from numpy import any as array_any
from numpy import around, full_like, gradient, isclose, isin, isnat, take, tensordot, transpose, unravel_index
from numpy import concatenate as _concatenate
from numpy.lib.stride_tricks import sliding_window_view
from packaging.version import Version
from pandas.api.types import is_extension_array_dtype
from xarray.core import dask_array_ops, dtypes, nputils
from xarray.core.options import OPTIONS
from xarray.core.utils import is_duck_array, is_duck_dask_array, module_available
from xarray.namedarray import pycompat
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, is_chunked_array
if module_available('numpy', minversion='2.0.0.dev0'):
    from numpy.lib.array_utils import normalize_axis_index
else:
    from numpy.core.multiarray import normalize_axis_index
dask_available = module_available('dask')

def _dask_or_eager_func(name, eager_module=np, dask_module='dask.array'):
    """Create a function that dispatches to dask for dask array inputs."""
    pass
pandas_isnull = _dask_or_eager_func('isnull', eager_module=pd, dask_module='dask.array')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.,  2.])', 'array([0., 2.])')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.,  2.])', 'array([0., 2.])')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.4,  1.6])', 'array([0.4, 1.6])')
around.__doc__ = str.replace(around.__doc__ or '', 'array([0.,  2.,  2.,  4.,  4.])', 'array([0., 2., 2., 4., 4.])')
around.__doc__ = str.replace(around.__doc__ or '', '    .. [2] "How Futile are Mindless Assessments of\n           Roundoff in Floating-Point Computation?", William Kahan,\n           https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf\n', '')
masked_invalid = _dask_or_eager_func('masked_invalid', eager_module=np.ma, dask_module='dask.array.ma')

def as_shared_dtype(scalars_or_arrays, xp=None):
    """Cast a arrays to a shared dtype using xarray's type promotion rules."""
    pass

def lazy_array_equiv(arr1, arr2):
    """Like array_equal, but doesn't actually compare values.
    Returns True when arr1, arr2 identical or their dask tokens are equal.
    Returns False when shapes are not equal.
    Returns None when equality cannot determined: one or both of arr1, arr2 are numpy arrays;
    or their dask tokens are not equal
    """
    pass

def allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08):
    """Like np.allclose, but also allows values to be NaN in both arrays"""
    pass

def array_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in both arrays"""
    pass

def array_notnull_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in either or both
    arrays
    """
    pass

def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes"""
    pass

def where(condition, x, y):
    """Three argument where() with better dtype promotion rules."""
    pass

def concatenate(arrays, axis=0):
    """concatenate() with better dtype promotion rules."""
    pass

def stack(arrays, axis=0):
    """stack() with better dtype promotion rules."""
    pass
argmax = _create_nan_agg_method('argmax', coerce_strings=True)
argmin = _create_nan_agg_method('argmin', coerce_strings=True)
max = _create_nan_agg_method('max', coerce_strings=True, invariant_0d=True)
min = _create_nan_agg_method('min', coerce_strings=True, invariant_0d=True)
sum = _create_nan_agg_method('sum', invariant_0d=True)
sum.numeric_only = True
sum.available_min_count = True
std = _create_nan_agg_method('std')
std.numeric_only = True
var = _create_nan_agg_method('var')
var.numeric_only = True
median = _create_nan_agg_method('median', invariant_0d=True)
median.numeric_only = True
prod = _create_nan_agg_method('prod', invariant_0d=True)
prod.numeric_only = True
prod.available_min_count = True
cumprod_1d = _create_nan_agg_method('cumprod', invariant_0d=True)
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method('cumsum', invariant_0d=True)
cumsum_1d.numeric_only = True
_mean = _create_nan_agg_method('mean', invariant_0d=True)

def _datetime_nanmin(array):
    """nanmin() function for datetime64.

    Caveats that this function deals with:

    - In numpy < 1.18, min() on datetime64 incorrectly ignores NaT
    - numpy nanmin() don't work on datetime64 (all versions at the moment of writing)
    - dask min() does not work on datetime64 (all versions at the moment of writing)
    """
    pass

def datetime_to_numeric(array, offset=None, datetime_unit=None, dtype=float):
    """Convert an array containing datetime-like data to numerical values.
    Convert the datetime array to a timedelta relative to an offset.
    Parameters
    ----------
    array : array-like
        Input data
    offset : None, datetime or cftime.datetime
        Datetime offset. If None, this is set by default to the array's minimum
        value to reduce round off errors.
    datetime_unit : {None, Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
        If not None, convert output to a given datetime unit. Note that some
        conversions are not allowed due to non-linear relationships between units.
    dtype : dtype
        Output dtype.
    Returns
    -------
    array
        Numerical representation of datetime object relative to an offset.
    Notes
    -----
    Some datetime unit conversions won't work, for example from days to years, even
    though some calendars would allow for them (e.g. no_leap). This is because there
    is no `cftime.timedelta` object.
    """
    pass

def timedelta_to_numeric(value, datetime_unit='ns', dtype=float):
    """Convert a timedelta-like object to numerical values.

    Parameters
    ----------
    value : datetime.timedelta, numpy.timedelta64, pandas.Timedelta, str
        Time delta representation.
    datetime_unit : {Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
        The time units of the output values. Note that some conversions are not allowed due to
        non-linear relationships between units.
    dtype : type
        The output data type.

    """
    pass

def np_timedelta64_to_float(array, datetime_unit):
    """Convert numpy.timedelta64 to float.

    Notes
    -----
    The array is first converted to microseconds, which is less likely to
    cause overflow errors.
    """
    pass

def pd_timedelta_to_float(value, datetime_unit):
    """Convert pandas.Timedelta to float.

    Notes
    -----
    Built on the assumption that pandas timedelta values are in nanoseconds,
    which is also the numpy default resolution.
    """
    pass

def py_timedelta_to_float(array, datetime_unit):
    """Convert a timedelta object to a float, possibly at a loss of resolution."""
    pass

def mean(array, axis=None, skipna=None, **kwargs):
    """inhouse mean that can handle np.datetime64 or cftime.datetime
    dtypes"""
    pass
mean.numeric_only = True

def cumprod(array, axis=None, **kwargs):
    """N-dimensional version of cumprod."""
    pass

def cumsum(array, axis=None, **kwargs):
    """N-dimensional version of cumsum."""
    pass

def first(values, axis, skipna=None):
    """Return the first non-NA elements in this array along the given axis"""
    pass

def last(values, axis, skipna=None):
    """Return the last non-NA elements in this array along the given axis"""
    pass

def least_squares(lhs, rhs, rcond=None, skipna=False):
    """Return the coefficients and residuals of a least-squares fit."""
    pass

def _push(array, n: int | None=None, axis: int=-1):
    """
    Use either bottleneck or numbagg depending on options & what's available
    """
    pass