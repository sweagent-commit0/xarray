from __future__ import annotations
import gzip
import io
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendArray, BackendEntrypoint, WritableCFDataStore, _normalize_path
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import ensure_lock, get_write_lock
from xarray.backends.netcdf3 import encode_nc3_attr_value, encode_nc3_variable, is_valid_nc3_name
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import Frozen, FrozenDict, close_on_error, module_available, try_read_magic_number_from_file_or_path
from xarray.core.variable import Variable
if TYPE_CHECKING:
    from io import BufferedIOBase
    from xarray.backends.common import AbstractDataStore
    from xarray.core.dataset import Dataset
HAS_NUMPY_2_0 = module_available('numpy', minversion='2.0.0.dev0')

class ScipyArrayWrapper(BackendArray):

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_variable().data
        self.shape = array.shape
        self.dtype = np.dtype(array.dtype.kind + str(array.dtype.itemsize))

    def __getitem__(self, key):
        data = indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._getitem)
        copy = self.datastore.ds.use_mmap
        copy = None if HAS_NUMPY_2_0 and copy is False else copy
        return np.array(data, dtype=self.dtype, copy=copy)

    def __setitem__(self, key, value):
        with self.datastore.lock:
            data = self.get_variable(needs_lock=False)
            try:
                data[key] = value
            except TypeError:
                if key is Ellipsis:
                    data[:] = value
                else:
                    raise

class ScipyDataStore(WritableCFDataStore):
    """Store for reading and writing data via scipy.io.netcdf.

    This store has the advantage of being able to be initialized with a
    StringIO object, allow for serialization without writing to disk.

    It only supports the NetCDF3 file-format.
    """

    def __init__(self, filename_or_obj, mode='r', format=None, group=None, mmap=None, lock=None):
        if group is not None:
            raise ValueError('cannot save to a group with the scipy.io.netcdf backend')
        if format is None or format == 'NETCDF3_64BIT':
            version = 2
        elif format == 'NETCDF3_CLASSIC':
            version = 1
        else:
            raise ValueError(f'invalid format for scipy.io.netcdf backend: {format!r}')
        if lock is None and mode != 'r' and isinstance(filename_or_obj, str):
            lock = get_write_lock(filename_or_obj)
        self.lock = ensure_lock(lock)
        if isinstance(filename_or_obj, str):
            manager = CachingFileManager(_open_scipy_netcdf, filename_or_obj, mode=mode, lock=lock, kwargs=dict(mmap=mmap, version=version))
        else:
            scipy_dataset = _open_scipy_netcdf(filename_or_obj, mode=mode, mmap=mmap, version=version)
            manager = DummyFileManager(scipy_dataset)
        self._manager = manager

class ScipyBackendEntrypoint(BackendEntrypoint):
    """
    Backend for netCDF files based on the scipy package.

    It can open ".nc", ".nc4", ".cdf" and ".gz" files but will only be
    selected as the default if the "netcdf4" and "h5netcdf" engines are
    not available. It has the advantage that is is a lightweight engine
    that has no system requirements (unlike netcdf4 and h5netcdf).

    Additionally it can open gizp compressed (".gz") files.

    For more information about the underlying library, visit:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.html

    See Also
    --------
    backends.ScipyDataStore
    backends.NetCDF4BackendEntrypoint
    backends.H5netcdfBackendEntrypoint
    """
    description = 'Open netCDF files (.nc, .nc4, .cdf and .gz) using scipy in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.ScipyBackendEntrypoint.html'
BACKEND_ENTRYPOINTS['scipy'] = ('scipy', ScipyBackendEntrypoint)