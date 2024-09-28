from __future__ import annotations
import functools
from typing import Any
import numpy as np
from pandas.api.types import is_extension_array_dtype
from xarray.core import array_api_compat, npcompat, utils
NA = utils.ReprObject('<NA>')

@functools.total_ordering
class AlwaysGreaterThan:

    def __gt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))

@functools.total_ordering
class AlwaysLessThan:

    def __lt__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))
INF = AlwaysGreaterThan()
NINF = AlwaysLessThan()
PROMOTE_TO_OBJECT: tuple[tuple[type[np.generic], type[np.generic]], ...] = ((np.number, np.character), (np.bool_, np.character), (np.bytes_, np.str_))

def maybe_promote(dtype: np.dtype) -> tuple[np.dtype, Any]:
    """Simpler equivalent of pandas.core.common._maybe_promote

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    dtype : Promoted dtype that can hold missing values.
    fill_value : Valid missing value for the promoted dtype.
    """
    pass
NAT_TYPES = {np.datetime64('NaT').dtype, np.timedelta64('NaT').dtype}

def get_fill_value(dtype):
    """Return an appropriate fill value for this dtype.

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    fill_value : Missing value corresponding to this dtype.
    """
    pass

def get_pos_infinity(dtype, max_for_int=False):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype
    max_for_int : bool
        Return np.iinfo(dtype).max instead of np.inf

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    pass

def get_neg_infinity(dtype, min_for_int=False):
    """Return an appropriate positive infinity for this dtype.

    Parameters
    ----------
    dtype : np.dtype
    min_for_int : bool
        Return np.iinfo(dtype).min instead of -np.inf

    Returns
    -------
    fill_value : positive infinity value corresponding to this dtype.
    """
    pass

def is_datetime_like(dtype) -> bool:
    """Check if a dtype is a subclass of the numpy datetime types"""
    pass

def is_object(dtype) -> bool:
    """Check if a dtype is object"""
    pass

def is_string(dtype) -> bool:
    """Check if a dtype is a string dtype"""
    pass

def isdtype(dtype, kind: str | tuple[str, ...], xp=None) -> bool:
    """Compatibility wrapper for isdtype() from the array API standard.

    Unlike xp.isdtype(), kind must be a string.
    """
    pass

def result_type(*arrays_and_dtypes: np.typing.ArrayLike | np.typing.DTypeLike, xp=None) -> np.dtype:
    """Like np.result_type, but with type promotion rules matching pandas.

    Examples of changed behavior:
    number + string -> object (not string)
    bytes + unicode -> object (not unicode)

    Parameters
    ----------
    *arrays_and_dtypes : list of arrays and dtypes
        The dtype is extracted from both numpy and dask arrays.

    Returns
    -------
    numpy.dtype for the result.
    """
    pass