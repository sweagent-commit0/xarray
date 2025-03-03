"""Coders for strings."""
from __future__ import annotations
from functools import partial
import numpy as np
from xarray.coding.variables import VariableCoder, lazy_elemwise_func, pop_to, safe_setitem, unpack_for_decoding, unpack_for_encoding
from xarray.core import indexing
from xarray.core.utils import module_available
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
HAS_NUMPY_2_0 = module_available('numpy', minversion='2.0.0.dev0')

class EncodedStringCoder(VariableCoder):
    """Transforms between unicode strings and fixed-width UTF-8 bytes."""

    def __init__(self, allows_unicode=True):
        self.allows_unicode = allows_unicode

def ensure_fixed_length_bytes(var: Variable) -> Variable:
    """Ensure that a variable with vlen bytes is converted to fixed width."""
    pass

class CharacterArrayCoder(VariableCoder):
    """Transforms between arrays containing bytes and character arrays."""

def bytes_to_char(arr):
    """Convert numpy/dask arrays from fixed width bytes to characters."""
    pass

def _numpy_bytes_to_char(arr):
    """Like netCDF4.stringtochar, but faster and more flexible."""
    pass

def char_to_bytes(arr):
    """Convert numpy/dask arrays from characters to fixed width bytes."""
    pass

def _numpy_char_to_bytes(arr):
    """Like netCDF4.chartostring, but faster and more flexible."""
    pass

class StackedBytesArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessed, are automatically stacked along the last dimension.

    >>> indexer = indexing.BasicIndexer((slice(None),))
    >>> StackedBytesArray(np.array(["a", "b", "c"], dtype="S1"))[indexer]
    array(b'abc', dtype='|S3')
    """

    def __init__(self, array):
        """
        Parameters
        ----------
        array : array-like
            Original array of values to wrap.
        """
        if array.dtype != 'S1':
            raise ValueError("can only use StackedBytesArray if argument has dtype='S1'")
        self.array = indexing.as_indexable(array)

    def __repr__(self):
        return f'{type(self).__name__}({self.array!r})'

    def __getitem__(self, key):
        key = type(key)(indexing.expanded_indexer(key.tuple, self.array.ndim))
        if key.tuple[-1] != slice(None):
            raise IndexError('too many indices')
        return _numpy_char_to_bytes(self.array[key])