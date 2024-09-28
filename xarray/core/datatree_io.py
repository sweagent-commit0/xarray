from __future__ import annotations
from collections.abc import Mapping, MutableMapping
from os import PathLike
from typing import Any, Literal, get_args
from xarray.core.datatree import DataTree
from xarray.core.types import NetcdfWriteModes, ZarrWriteModes
T_DataTreeNetcdfEngine = Literal['netcdf4', 'h5netcdf']
T_DataTreeNetcdfTypes = Literal['NETCDF4']

def _datatree_to_netcdf(dt: DataTree, filepath: str | PathLike, mode: NetcdfWriteModes='w', encoding: Mapping[str, Any] | None=None, unlimited_dims: Mapping | None=None, format: T_DataTreeNetcdfTypes | None=None, engine: T_DataTreeNetcdfEngine | None=None, group: str | None=None, compute: bool=True, **kwargs):
    """This function creates an appropriate datastore for writing a datatree to
    disk as a netCDF file.

    See `DataTree.to_netcdf` for full API docs.
    """
    pass

def _datatree_to_zarr(dt: DataTree, store: MutableMapping | str | PathLike[str], mode: ZarrWriteModes='w-', encoding: Mapping[str, Any] | None=None, consolidated: bool=True, group: str | None=None, compute: Literal[True]=True, **kwargs):
    """This function creates an appropriate datastore for writing a datatree
    to a zarr store.

    See `DataTree.to_zarr` for full API docs.
    """
    pass