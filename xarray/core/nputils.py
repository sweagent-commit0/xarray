from __future__ import annotations
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_duck_array, module_available
from xarray.namedarray import pycompat
if module_available('numpy', minversion='2.0.0.dev0'):
    from numpy.lib.array_utils import normalize_axis_index
else:
    from numpy.core.multiarray import normalize_axis_index
try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning
from xarray.core.options import OPTIONS
try:
    import bottleneck as bn
    _BOTTLENECK_AVAILABLE = True
except ImportError:
    bn = np
    _BOTTLENECK_AVAILABLE = False

def inverse_permutation(indices: np.ndarray, N: int | None=None) -> np.ndarray:
    """Return indices for an inverse permutation.

    Parameters
    ----------
    indices : 1D np.ndarray with dtype=int
        Integer positions to assign elements to.
    N : int, optional
        Size of the array

    Returns
    -------
    inverse_permutation : 1D np.ndarray with dtype=int
        Integer indices to take from the original array to create the
        permutation.
    """
    pass

def _is_contiguous(positions):
    """Given a non-empty list, does it consist of contiguous integers?"""
    pass

def _advanced_indexer_subspaces(key):
    """Indices of the advanced indexes subspaces for mixed indexing and vindex."""
    pass

class NumpyVIndexAdapter:
    """Object that implements indexing like vindex on a np.ndarray.

    This is a pure Python implementation of (some of) the logic in this NumPy
    proposal: https://github.com/numpy/numpy/pull/6256
    """

    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        return np.moveaxis(self._array[key], mixed_positions, vindex_positions)

    def __setitem__(self, key, value):
        """Value must have dimensionality matching the key."""
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        self._array[key] = np.moveaxis(value, vindex_positions, mixed_positions)
nanmin = _create_method('nanmin')
nanmax = _create_method('nanmax')
nanmean = _create_method('nanmean')
nanmedian = _create_method('nanmedian')
nanvar = _create_method('nanvar')
nanstd = _create_method('nanstd')
nanprod = _create_method('nanprod')
nancumsum = _create_method('nancumsum')
nancumprod = _create_method('nancumprod')
nanargmin = _create_method('nanargmin')
nanargmax = _create_method('nanargmax')
nanquantile = _create_method('nanquantile')