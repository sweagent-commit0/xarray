from __future__ import annotations
import functools
import inspect
import itertools
import sys
import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable
from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendEntrypoint
from xarray.core.utils import module_available
if TYPE_CHECKING:
    import os
    from importlib.metadata import EntryPoint
    if sys.version_info >= (3, 10):
        from importlib.metadata import EntryPoints
    else:
        EntryPoints = list[EntryPoint]
    from io import BufferedIOBase
    from xarray.backends.common import AbstractDataStore
STANDARD_BACKENDS_ORDER = ['netcdf4', 'h5netcdf', 'scipy']

@functools.lru_cache(maxsize=1)
def list_engines() -> dict[str, BackendEntrypoint]:
    """
    Return a dictionary of available engines and their BackendEntrypoint objects.

    Returns
    -------
    dictionary

    Notes
    -----
    This function lives in the backends namespace (``engs=xr.backends.list_engines()``).
    If available, more information is available about each backend via ``engs["eng_name"]``.

    # New selection mechanism introduced with Python 3.10. See GH6514.
    """
    pass

def refresh_engines() -> None:
    """Refreshes the backend engines based on installed packages."""
    pass

def get_backend(engine: str | type[BackendEntrypoint]) -> BackendEntrypoint:
    """Select open_dataset method based on current engine."""
    pass