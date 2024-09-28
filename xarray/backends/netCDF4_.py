from __future__ import annotations
import functools
import operator
import os
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding
from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendArray, BackendEntrypoint, WritableCFDataStore, _normalize_path, find_root_and_group, robust_getitem
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import HDF5_LOCK, NETCDFC_LOCK, combine_locks, ensure_lock, get_write_lock
from xarray.backends.netcdf3 import encode_nc3_attr_value, encode_nc3_variable
from xarray.backends.store import StoreBackendEntrypoint
from xarray.coding.variables import pop_to
from xarray.core import indexing
from xarray.core.utils import FrozenDict, close_on_error, is_remote_uri, try_read_magic_number_from_path
from xarray.core.variable import Variable
if TYPE_CHECKING:
    from io import BufferedIOBase
    from xarray.backends.common import AbstractDataStore
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree
_endian_lookup = {'=': 'native', '>': 'big', '<': 'little', '|': 'native'}
NETCDF4_PYTHON_LOCK = combine_locks([NETCDFC_LOCK, HDF5_LOCK])

class BaseNetCDF4Array(BackendArray):
    __slots__ = ('datastore', 'dtype', 'shape', 'variable_name')

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_array()
        self.shape = array.shape
        dtype = array.dtype
        if dtype is str:
            dtype = coding.strings.create_vlen_dtype(str)
        self.dtype = dtype

    def __setitem__(self, key, value):
        with self.datastore.lock:
            data = self.get_array(needs_lock=False)
            data[key] = value
            if self.datastore.autoclose:
                self.datastore.close(needs_lock=False)

class NetCDF4ArrayWrapper(BaseNetCDF4Array):
    __slots__ = ()

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER, self._getitem)

class NetCDF4DataStore(WritableCFDataStore):
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """
    __slots__ = ('autoclose', 'format', 'is_remote', 'lock', '_filename', '_group', '_manager', '_mode')

    def __init__(self, manager, group=None, mode=None, lock=NETCDF4_PYTHON_LOCK, autoclose=False):
        import netCDF4
        if isinstance(manager, netCDF4.Dataset):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if type(manager) is not netCDF4.Dataset:
                    raise ValueError('must supply a root netCDF4.Dataset if the group argument is provided')
                root = manager
            manager = DummyFileManager(root)
        self._manager = manager
        self._group = group
        self._mode = mode
        self.format = self.ds.data_model
        self._filename = self.ds.filepath()
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self.autoclose = autoclose

    def _build_and_get_enum(self, var_name: str, dtype: np.dtype, enum_name: str, enum_dict: dict[str, int]) -> Any:
        """
        Add or get the netCDF4 Enum based on the dtype in encoding.
        The return type should be ``netCDF4.EnumType``,
        but we avoid importing netCDF4 globally for performances.
        """
        pass

class NetCDF4BackendEntrypoint(BackendEntrypoint):
    """
    Backend for netCDF files based on the netCDF4 package.

    It can open ".nc", ".nc4", ".cdf" files and will be chosen
    as default for these files.

    Additionally it can open valid HDF5 files, see
    https://h5netcdf.org/#invalid-netcdf-files for more info.
    It will not be detected as valid backend for such files, so make
    sure to specify ``engine="netcdf4"`` in ``open_dataset``.

    For more information about the underlying library, visit:
    https://unidata.github.io/netcdf4-python

    See Also
    --------
    backends.NetCDF4DataStore
    backends.H5netcdfBackendEntrypoint
    backends.ScipyBackendEntrypoint
    """
    description = 'Open netCDF (.nc, .nc4 and .cdf) and most HDF5 files using netCDF4 in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.NetCDF4BackendEntrypoint.html'
BACKEND_ENTRYPOINTS['netcdf4'] = ('netCDF4', NetCDF4BackendEntrypoint)