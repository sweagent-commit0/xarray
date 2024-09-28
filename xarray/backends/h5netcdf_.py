from __future__ import annotations
import functools
import io
import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any
from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendEntrypoint, WritableCFDataStore, _normalize_path, find_root_and_group
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import HDF5_LOCK, combine_locks, ensure_lock, get_write_lock
from xarray.backends.netCDF4_ import BaseNetCDF4Array, _encode_nc4_variable, _ensure_no_forward_slash_in_name, _extract_nc4_variable_encoding, _get_datatype, _nc4_require_group
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict, emit_user_level_warning, is_remote_uri, read_magic_number_from_file, try_read_magic_number_from_file_or_path
from xarray.core.variable import Variable
if TYPE_CHECKING:
    from io import BufferedIOBase
    from xarray.backends.common import AbstractDataStore
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree

class H5NetCDFArrayWrapper(BaseNetCDF4Array):

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._getitem)
_extract_h5nc_encoding = functools.partial(_extract_nc4_variable_encoding, lsd_okay=False, h5py_okay=True, backend='h5netcdf', unlimited_dims=None)

class H5NetCDFStore(WritableCFDataStore):
    """Store for reading and writing data via h5netcdf"""
    __slots__ = ('autoclose', 'format', 'is_remote', 'lock', '_filename', '_group', '_manager', '_mode')

    def __init__(self, manager, group=None, mode=None, lock=HDF5_LOCK, autoclose=False):
        import h5netcdf
        if isinstance(manager, (h5netcdf.File, h5netcdf.Group)):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if type(manager) is not h5netcdf.File:
                    raise ValueError('must supply a h5netcdf.File if the group argument is provided')
                root = manager
            manager = DummyFileManager(root)
        self._manager = manager
        self._group = group
        self._mode = mode
        self.format = None
        self._filename = find_root_and_group(self.ds)[0].filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self.autoclose = autoclose

class H5netcdfBackendEntrypoint(BackendEntrypoint):
    """
    Backend for netCDF files based on the h5netcdf package.

    It can open ".nc", ".nc4", ".cdf" files but will only be
    selected as the default if the "netcdf4" engine is not available.

    Additionally it can open valid HDF5 files, see
    https://h5netcdf.org/#invalid-netcdf-files for more info.
    It will not be detected as valid backend for such files, so make
    sure to specify ``engine="h5netcdf"`` in ``open_dataset``.

    For more information about the underlying library, visit:
    https://h5netcdf.org

    See Also
    --------
    backends.H5NetCDFStore
    backends.NetCDF4BackendEntrypoint
    backends.ScipyBackendEntrypoint
    """
    description = 'Open netCDF (.nc, .nc4 and .cdf) and most HDF5 files using h5netcdf in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.H5netcdfBackendEntrypoint.html'
BACKEND_ENTRYPOINTS['h5netcdf'] = ('h5netcdf', H5netcdfBackendEntrypoint)