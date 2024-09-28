from __future__ import annotations
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, utils
from xarray.core.alignment import align, reindex_variables
from xarray.core.coordinates import Coordinates
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import Index, PandasIndex
from xarray.core.merge import _VALID_COMPAT, collect_variables_and_indexes, merge_attrs, merge_collected
from xarray.core.types import T_DataArray, T_Dataset, T_Variable
from xarray.core.variable import Variable
from xarray.core.variable import concat as concat_vars
if TYPE_CHECKING:
    from xarray.core.types import CombineAttrsOptions, CompatOptions, ConcatOptions, JoinOptions
    T_DataVars = Union[ConcatOptions, Iterable[Hashable]]

def concat(objs, dim, data_vars: T_DataVars='all', coords='different', compat: CompatOptions='equals', positions=None, fill_value=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='override', create_index_for_new_dim: bool=True):
    """Concatenate xarray objects along a new or existing dimension.

    Parameters
    ----------
    objs : sequence of Dataset and DataArray
        xarray objects to concatenate together. Each object is expected to
        consist of variables and coordinates with matching shapes except for
        along the concatenated dimension.
    dim : Hashable or Variable or DataArray or pandas.Index
        Name of the dimension to concatenate along. This can either be a new
        dimension name, in which case it is added along axis=0, or an existing
        dimension name, in which case the location of the dimension is
        unchanged. If dimension is provided as a Variable, DataArray or Index, its name
        is used as the dimension to concatenate along and the values are added
        as a coordinate.
    data_vars : {"minimal", "different", "all"} or list of Hashable, optional
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of dims: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.

        If objects are DataArrays, data_vars must be "all".
    coords : {"minimal", "different", "all"} or list of Hashable, optional
        These coordinate variables will be concatenated together:
          * "minimal": Only coordinates in which the dimension already appears
            are included.
          * "different": Coordinates which are not equal (ignoring attributes)
            across all datasets are also concatenated (as well as all for which
            dimension already appears). Beware: this option may load the data
            payload of coordinate variables into memory if they are not already
            loaded.
          * "all": All coordinate variables will be concatenated, except
            those corresponding to other dimensions.
          * list of Hashable: The listed coordinate variables will be concatenated,
            in addition to the "minimal" coordinates.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare non-concatenated variables of the same name for
        potential conflicts. This is passed down to merge.

        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset
    positions : None or list of integer arrays, optional
        List of integer arrays which specifies the integer positions to which
        to assign each dataset along the concatenated dimension. If not
        supplied, objects are concatenated in the provided order.
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.
    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes
        (excluding dim) in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                      "override"} or callable, default: "override"
        A callable or a string indicating how to combine attrs of the objects being
        merged:

        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "drop_conflicts": attrs from all objects are combined, any that have
          the same name but different values are dropped.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.

        If a callable, it must expect a sequence of ``attrs`` dicts and a context object
        as its only parameters.
    create_index_for_new_dim : bool, default: True
        Whether to create a new ``PandasIndex`` object when the objects being concatenated contain scalar variables named ``dim``.

    Returns
    -------
    concatenated : type of objs

    See also
    --------
    merge

    Examples
    --------
    >>> da = xr.DataArray(
    ...     np.arange(6).reshape(2, 3), [("x", ["a", "b"]), ("y", [10, 20, 30])]
    ... )
    >>> da
    <xarray.DataArray (x: 2, y: 3)> Size: 48B
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
      * x        (x) <U1 8B 'a' 'b'
      * y        (y) int64 24B 10 20 30

    >>> xr.concat([da.isel(y=slice(0, 1)), da.isel(y=slice(1, None))], dim="y")
    <xarray.DataArray (x: 2, y: 3)> Size: 48B
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
      * x        (x) <U1 8B 'a' 'b'
      * y        (y) int64 24B 10 20 30

    >>> xr.concat([da.isel(x=0), da.isel(x=1)], "x")
    <xarray.DataArray (x: 2, y: 3)> Size: 48B
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
      * x        (x) <U1 8B 'a' 'b'
      * y        (y) int64 24B 10 20 30

    >>> xr.concat([da.isel(x=0), da.isel(x=1)], "new_dim")
    <xarray.DataArray (new_dim: 2, y: 3)> Size: 48B
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
        x        (new_dim) <U1 8B 'a' 'b'
      * y        (y) int64 24B 10 20 30
    Dimensions without coordinates: new_dim

    >>> xr.concat([da.isel(x=0), da.isel(x=1)], pd.Index([-90, -100], name="new_dim"))
    <xarray.DataArray (new_dim: 2, y: 3)> Size: 48B
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
        x        (new_dim) <U1 8B 'a' 'b'
      * y        (y) int64 24B 10 20 30
      * new_dim  (new_dim) int64 16B -90 -100

    # Concatenate a scalar variable along a new dimension of the same name with and without creating a new index

    >>> ds = xr.Dataset(coords={"x": 0})
    >>> xr.concat([ds, ds], dim="x")
    <xarray.Dataset> Size: 16B
    Dimensions:  (x: 2)
    Coordinates:
      * x        (x) int64 16B 0 0
    Data variables:
        *empty*

    >>> xr.concat([ds, ds], dim="x").indexes
    Indexes:
        x        Index([0, 0], dtype='int64', name='x')

    >>> xr.concat([ds, ds], dim="x", create_index_for_new_dim=False).indexes
    Indexes:
        *empty*
    """
    pass

def _calc_concat_dim_index(dim_or_data: Hashable | Any) -> tuple[Hashable, PandasIndex | None]:
    """Infer the dimension name and 1d index / coordinate variable (if appropriate)
    for concatenating along the new dimension.

    """
    pass

def _calc_concat_over(datasets, dim, dim_names, data_vars: T_DataVars, coords, compat):
    """
    Determine which dataset variables need to be concatenated in the result,
    """
    pass

def _dataset_concat(datasets: Iterable[T_Dataset], dim: str | T_Variable | T_DataArray | pd.Index, data_vars: T_DataVars, coords: str | list[str], compat: CompatOptions, positions: Iterable[Iterable[int]] | None, fill_value: Any=dtypes.NA, join: JoinOptions='outer', combine_attrs: CombineAttrsOptions='override', create_index_for_new_dim: bool=True) -> T_Dataset:
    """
    Concatenate a sequence of datasets along a new or existing dimension
    """
    pass