from __future__ import annotations
from xarray.core import dtypes, nputils

def dask_rolling_wrapper(moving_func, a, window, min_count=None, axis=-1):
    """Wrapper to apply bottleneck moving window funcs on dask arrays"""
    pass

def push(array, n, axis):
    """
    Dask-aware bottleneck.push
    """
    pass