from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import BACKEND_ENTRYPOINTS, AbstractDataStore, BackendArray, BackendEntrypoint, robust_getitem
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import Frozen, FrozenDict, close_on_error, is_dict_like, is_remote_uri
from xarray.core.variable import Variable
from xarray.namedarray.pycompat import integer_types
if TYPE_CHECKING:
    import os
    from io import BufferedIOBase
    from xarray.core.dataset import Dataset

class PydapArrayWrapper(BackendArray):

    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.BASIC, self._getitem)

class PydapDataStore(AbstractDataStore):
    """Store for accessing OpenDAP datasets with pydap.

    This store provides an alternative way to access OpenDAP datasets that may
    be useful if the netCDF4 library is not available.
    """

    def __init__(self, ds):
        """
        Parameters
        ----------
        ds : pydap DatasetType
        """
        self.ds = ds

class PydapBackendEntrypoint(BackendEntrypoint):
    """
    Backend for steaming datasets over the internet using
    the Data Access Protocol, also known as DODS or OPeNDAP
    based on the pydap package.

    This backend is selected by default for urls.

    For more information about the underlying library, visit:
    https://www.pydap.org

    See Also
    --------
    backends.PydapDataStore
    """
    description = 'Open remote datasets via OPeNDAP using pydap in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.PydapBackendEntrypoint.html'
BACKEND_ENTRYPOINTS['pydap'] = ('pydap', PydapBackendEntrypoint)