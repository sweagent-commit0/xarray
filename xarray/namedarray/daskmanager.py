from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
from packaging.version import Version
from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray
from xarray.namedarray.utils import is_duck_dask_array, module_available
if TYPE_CHECKING:
    from xarray.namedarray._typing import T_Chunks, _DType_co, _NormalizedChunks, duckarray
    try:
        from dask.array import Array as DaskArray
    except ImportError:
        DaskArray = np.ndarray[Any, Any]
dask_available = module_available('dask')

class DaskManager(ChunkManagerEntrypoint['DaskArray']):
    array_cls: type[DaskArray]
    available: bool = dask_available

    def __init__(self) -> None:
        from dask.array import Array
        self.array_cls = Array

    def normalize_chunks(self, chunks: T_Chunks | _NormalizedChunks, shape: tuple[int, ...] | None=None, limit: int | None=None, dtype: _DType_co | None=None, previous_chunks: _NormalizedChunks | None=None) -> Any:
        """Called by open_dataset"""
        pass