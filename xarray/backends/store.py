from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from xarray import conventions
from xarray.backends.common import BACKEND_ENTRYPOINTS, AbstractDataStore, BackendEntrypoint
from xarray.core.dataset import Dataset
if TYPE_CHECKING:
    import os
    from io import BufferedIOBase

class StoreBackendEntrypoint(BackendEntrypoint):
    description = 'Open AbstractDataStore instances in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.StoreBackendEntrypoint.html'
BACKEND_ENTRYPOINTS['store'] = (None, StoreBackendEntrypoint)