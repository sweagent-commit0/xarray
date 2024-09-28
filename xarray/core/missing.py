from __future__ import annotations
import datetime as dt
import warnings
from collections.abc import Hashable, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, get_args
import numpy as np
import pandas as pd
from xarray.core import utils
from xarray.core.common import _contains_datetime_like_objects, ones_like
from xarray.core.computation import apply_ufunc
from xarray.core.duck_array_ops import datetime_to_numeric, push, reshape, timedelta_to_numeric
from xarray.core.options import _get_keep_attrs
from xarray.core.types import Interp1dOptions, InterpOptions
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import Variable, broadcast_variables
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

def _get_nan_block_lengths(obj: Dataset | DataArray | Variable, dim: Hashable, index: Variable):
    """
    Return an object where each NaN element in 'obj' is replaced by the
    length of the gap the element is in.
    """
    pass

class BaseInterpolator:
    """Generic interpolator class for normalizing interpolation methods"""
    cons_kwargs: dict[str, Any]
    call_kwargs: dict[str, Any]
    f: Callable
    method: str

    def __call__(self, x):
        return self.f(x, **self.call_kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}: method={self.method}'

class NumpyInterpolator(BaseInterpolator):
    """One-dimensional linear interpolation.

    See Also
    --------
    numpy.interp
    """

    def __init__(self, xi, yi, method='linear', fill_value=None, period=None):
        if method != 'linear':
            raise ValueError('only method `linear` is valid for the NumpyInterpolator')
        self.method = method
        self.f = np.interp
        self.cons_kwargs = {}
        self.call_kwargs = {'period': period}
        self._xi = xi
        self._yi = yi
        nan = np.nan if yi.dtype.kind != 'c' else np.nan + np.nan * 1j
        if fill_value is None:
            self._left = nan
            self._right = nan
        elif isinstance(fill_value, Sequence) and len(fill_value) == 2:
            self._left = fill_value[0]
            self._right = fill_value[1]
        elif is_scalar(fill_value):
            self._left = fill_value
            self._right = fill_value
        else:
            raise ValueError(f'{fill_value} is not a valid fill_value')

    def __call__(self, x):
        return self.f(x, self._xi, self._yi, left=self._left, right=self._right, **self.call_kwargs)

class ScipyInterpolator(BaseInterpolator):
    """Interpolate a 1-D function using Scipy interp1d

    See Also
    --------
    scipy.interpolate.interp1d
    """

    def __init__(self, xi, yi, method=None, fill_value=None, assume_sorted=True, copy=False, bounds_error=False, order=None, **kwargs):
        from scipy.interpolate import interp1d
        if method is None:
            raise ValueError('method is a required argument, please supply a valid scipy.inter1d method (kind)')
        if method == 'polynomial':
            if order is None:
                raise ValueError('order is required when method=polynomial')
            method = order
        self.method = method
        self.cons_kwargs = kwargs
        self.call_kwargs = {}
        nan = np.nan if yi.dtype.kind != 'c' else np.nan + np.nan * 1j
        if fill_value is None and method == 'linear':
            fill_value = (nan, nan)
        elif fill_value is None:
            fill_value = nan
        self.f = interp1d(xi, yi, kind=self.method, fill_value=fill_value, bounds_error=bounds_error, assume_sorted=assume_sorted, copy=copy, **self.cons_kwargs)

class SplineInterpolator(BaseInterpolator):
    """One-dimensional smoothing spline fit to a given set of data points.

    See Also
    --------
    scipy.interpolate.UnivariateSpline
    """

    def __init__(self, xi, yi, method='spline', fill_value=None, order=3, nu=0, ext=None, **kwargs):
        from scipy.interpolate import UnivariateSpline
        if method != 'spline':
            raise ValueError('only method `spline` is valid for the SplineInterpolator')
        self.method = method
        self.cons_kwargs = kwargs
        self.call_kwargs = {'nu': nu, 'ext': ext}
        if fill_value is not None:
            raise ValueError('SplineInterpolator does not support fill_value')
        self.f = UnivariateSpline(xi, yi, k=order, **self.cons_kwargs)

def _apply_over_vars_with_dim(func, self, dim=None, **kwargs):
    """Wrapper for datasets"""
    pass

def get_clean_interp_index(arr, dim: Hashable, use_coordinate: str | bool=True, strict: bool=True):
    """Return index to use for x values in interpolation or curve fitting.

    Parameters
    ----------
    arr : DataArray
        Array to interpolate or fit to a curve.
    dim : str
        Name of dimension along which to fit.
    use_coordinate : str or bool
        If use_coordinate is True, the coordinate that shares the name of the
        dimension along which interpolation is being performed will be used as the
        x values. If False, the x values are set as an equally spaced sequence.
    strict : bool
        Whether to raise errors if the index is either non-unique or non-monotonic (default).

    Returns
    -------
    Variable
        Numerical values for the x-coordinates.

    Notes
    -----
    If indexing is along the time dimension, datetime coordinates are converted
    to time deltas with respect to 1970-01-01.
    """
    pass

def interp_na(self, dim: Hashable | None=None, use_coordinate: bool | str=True, method: InterpOptions='linear', limit: int | None=None, max_gap: int | float | str | pd.Timedelta | np.timedelta64 | dt.timedelta | None=None, keep_attrs: bool | None=None, **kwargs):
    """Interpolate values according to different methods."""
    pass

def func_interpolate_na(interpolator, y, x, **kwargs):
    """helper function to apply interpolation along 1 dimension"""
    pass

def _bfill(arr, n=None, axis=-1):
    """inverse of ffill"""
    pass

def ffill(arr, dim=None, limit=None):
    """forward fill missing values"""
    pass

def bfill(arr, dim=None, limit=None):
    """backfill missing values"""
    pass

def _import_interpolant(interpolant, method):
    """Import interpolant from scipy.interpolate."""
    pass

def _get_interpolator(method: InterpOptions, vectorizeable_only: bool=False, **kwargs):
    """helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    """
    pass

def _get_interpolator_nd(method, **kwargs):
    """helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    """
    pass

def _get_valid_fill_mask(arr, dim, limit):
    """helper function to determine values that can be filled when limit is not
    None"""
    pass

def _localize(var, indexes_coords):
    """Speed up for linear and nearest neighbor method.
    Only consider a subspace that is needed for the interpolation
    """
    pass

def _floatize_x(x, new_x):
    """Make x and new_x float.
    This is particularly useful for datetime dtype.
    x, new_x: tuple of np.ndarray
    """
    pass

def interp(var, indexes_coords, method: InterpOptions, **kwargs):
    """Make an interpolation of Variable

    Parameters
    ----------
    var : Variable
    indexes_coords
        Mapping from dimension name to a pair of original and new coordinates.
        Original coordinates should be sorted in strictly ascending order.
        Note that all the coordinates should be Variable objects.
    method : string
        One of {'linear', 'nearest', 'zero', 'slinear', 'quadratic',
        'cubic'}. For multidimensional interpolation, only
        {'linear', 'nearest'} can be used.
    **kwargs
        keyword arguments to be passed to scipy.interpolate

    Returns
    -------
    Interpolated Variable

    See Also
    --------
    DataArray.interp
    Dataset.interp
    """
    pass

def interp_func(var, x, new_x, method: InterpOptions, kwargs):
    """
    multi-dimensional interpolation for array-like. Interpolated axes should be
    located in the last position.

    Parameters
    ----------
    var : np.ndarray or dask.array.Array
        Array to be interpolated. The final dimension is interpolated.
    x : a list of 1d array.
        Original coordinates. Should not contain NaN.
    new_x : a list of 1d array
        New coordinates. Should not contain NaN.
    method : string
        {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'} for
        1-dimensional interpolation.
        {'linear', 'nearest'} for multidimensional interpolation
    **kwargs
        Optional keyword arguments to be passed to scipy.interpolator

    Returns
    -------
    interpolated: array
        Interpolated array

    Notes
    -----
    This requires scipy installed.

    See Also
    --------
    scipy.interpolate.interp1d
    """
    pass

def _chunked_aware_interpnd(var, *coords, interp_func, interp_kwargs, localize=True):
    """Wrapper for `_interpnd` through `blockwise` for chunked arrays.

    The first half arrays in `coords` are original coordinates,
    the other half are destination coordinates
    """
    pass

def decompose_interp(indexes_coords):
    """Decompose the interpolation into a succession of independent interpolation keeping the order"""
    pass