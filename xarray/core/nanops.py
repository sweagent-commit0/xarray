from __future__ import annotations
import warnings
import numpy as np
from xarray.core import dtypes, duck_array_ops, nputils, utils
from xarray.core.duck_array_ops import astype, count, fillna, isnull, sum_where, where, where_method

def _maybe_null_out(result, axis, mask, min_count=1):
    """
    xarray version of pandas.core.nanops._maybe_null_out
    """
    pass

def _nan_argminmax_object(func, fill_value, value, axis=None, **kwargs):
    """In house nanargmin, nanargmax for object arrays. Always return integer
    type
    """
    pass

def _nan_minmax_object(func, fill_value, value, axis=None, **kwargs):
    """In house nanmin and nanmax for object array"""
    pass

def _nanmean_ddof_object(ddof, value, axis=None, dtype=None, **kwargs):
    """In house nanmean. ddof argument will be used in _nanvar method"""
    pass