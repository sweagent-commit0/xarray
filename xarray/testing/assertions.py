"""Testing functions exposed to the user API"""
import functools
import warnings
from collections.abc import Hashable
from typing import Union, overload
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops, formatting, utils
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.formatting import diff_datatree_repr
from xarray.core.indexes import Index, PandasIndex, PandasMultiIndex, default_indexes
from xarray.core.variable import IndexVariable, Variable

@ensure_warnings
def assert_isomorphic(a: DataTree, b: DataTree, from_root: bool=False):
    """
    Two DataTrees are considered isomorphic if every node has the same number of children.

    Nothing about the data or attrs in each node is checked.

    Isomorphism is a necessary condition for two trees to be used in a nodewise binary operation,
    such as tree1 + tree2.

    By default this function does not check any part of the tree above the given node.
    Therefore this function can be used as default to check that two subtrees are isomorphic.

    Parameters
    ----------
    a : DataTree
        The first object to compare.
    b : DataTree
        The second object to compare.
    from_root : bool, optional, default is False
        Whether or not to first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.

    See Also
    --------
    DataTree.isomorphic
    assert_equal
    assert_identical
    """
    pass

def maybe_transpose_dims(a, b, check_dim_order: bool):
    """Helper for assert_equal/allclose/identical"""
    pass

@ensure_warnings
def assert_equal(a, b, from_root=True, check_dim_order: bool=True):
    """Like :py:func:`numpy.testing.assert_array_equal`, but for xarray
    objects.

    Raises an AssertionError if two objects are not equal. This will match
    data values, dimensions and coordinates, but not names or attributes
    (except for Dataset objects for which the variable names must match).
    Arrays with NaN in the same location are considered equal.

    For DataTree objects, assert_equal is mapped over all Datasets on each node,
    with the DataTrees being equal if both are isomorphic and the corresponding
    Datasets at each node are themselves equal.

    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray, xarray.Variable, xarray.Coordinates
        or xarray.core.datatree.DataTree. The first object to compare.
    b : xarray.Dataset, xarray.DataArray, xarray.Variable, xarray.Coordinates
        or xarray.core.datatree.DataTree. The second object to compare.
    from_root : bool, optional, default is True
        Only used when comparing DataTree objects. Indicates whether or not to
        first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.
    check_dim_order : bool, optional, default is True
        Whether dimensions must be in the same order.

    See Also
    --------
    assert_identical, assert_allclose, Dataset.equals, DataArray.equals
    numpy.testing.assert_array_equal
    """
    pass

@ensure_warnings
def assert_identical(a, b, from_root=True):
    """Like :py:func:`xarray.testing.assert_equal`, but also matches the
    objects' names and attributes.

    Raises an AssertionError if two objects are not identical.

    For DataTree objects, assert_identical is mapped over all Datasets on each
    node, with the DataTrees being identical if both are isomorphic and the
    corresponding Datasets at each node are themselves identical.

    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray, xarray.Variable or xarray.Coordinates
        The first object to compare.
    b : xarray.Dataset, xarray.DataArray, xarray.Variable or xarray.Coordinates
        The second object to compare.
    from_root : bool, optional, default is True
        Only used when comparing DataTree objects. Indicates whether or not to
        first traverse to the root of the trees before checking for isomorphism.
        If a & b have no parents then this has no effect.
    check_dim_order : bool, optional, default is True
        Whether dimensions must be in the same order.

    See Also
    --------
    assert_equal, assert_allclose, Dataset.equals, DataArray.equals
    """
    pass

@ensure_warnings
def assert_allclose(a, b, rtol=1e-05, atol=1e-08, decode_bytes=True, check_dim_order: bool=True):
    """Like :py:func:`numpy.testing.assert_allclose`, but for xarray objects.

    Raises an AssertionError if two objects are not equal up to desired
    tolerance.

    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray or xarray.Variable
        The first object to compare.
    b : xarray.Dataset, xarray.DataArray or xarray.Variable
        The second object to compare.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    decode_bytes : bool, optional
        Whether byte dtypes should be decoded to strings as UTF-8 or not.
        This is useful for testing serialization methods on Python 3 that
        return saved strings as bytes.
    check_dim_order : bool, optional, default is True
        Whether dimensions must be in the same order.

    See Also
    --------
    assert_identical, assert_equal, numpy.testing.assert_allclose
    """
    pass

@ensure_warnings
def assert_duckarray_allclose(actual, desired, rtol=1e-07, atol=0, err_msg='', verbose=True):
    """Like `np.testing.assert_allclose`, but for duckarrays."""
    pass

@ensure_warnings
def assert_duckarray_equal(x, y, err_msg='', verbose=True):
    """Like `np.testing.assert_array_equal`, but for duckarrays"""
    pass

def assert_chunks_equal(a, b):
    """
    Assert that chunksizes along chunked dimensions are equal.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        The first object to compare.
    b : xarray.Dataset or xarray.DataArray
        The second object to compare.
    """
    pass

def _assert_internal_invariants(xarray_obj: Union[DataArray, Dataset, Variable], check_default_indexes: bool):
    """Validate that an xarray object satisfies its own internal invariants.

    This exists for the benefit of xarray's own test suite, but may be useful
    in external projects if they (ill-advisedly) create objects using xarray's
    private APIs.
    """
    pass