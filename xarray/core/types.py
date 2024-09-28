from __future__ import annotations
import datetime
import sys
from collections.abc import Collection, Hashable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, SupportsIndex, TypeVar, Union
import numpy as np
import pandas as pd
try:
    if sys.version_info >= (3, 11):
        from typing import Self, TypeAlias
    else:
        from typing_extensions import Self, TypeAlias
except ImportError:
    if TYPE_CHECKING:
        raise
    else:
        Self: Any = None
from numpy._typing import _SupportsDType
from numpy.typing import ArrayLike
if TYPE_CHECKING:
    from xarray.backends.common import BackendEntrypoint
    from xarray.core.alignment import Aligner
    from xarray.core.common import AbstractArray, DataWithCoords
    from xarray.core.coordinates import Coordinates
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.indexes import Index, Indexes
    from xarray.core.utils import Frozen
    from xarray.core.variable import Variable
    from xarray.groupers import TimeResampler
    try:
        from dask.array import Array as DaskArray
    except ImportError:
        DaskArray = np.ndarray
    try:
        from cubed import Array as CubedArray
    except ImportError:
        CubedArray = np.ndarray
    try:
        from zarr.core import Array as ZarrArray
    except ImportError:
        ZarrArray = np.ndarray
    _ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
    _DTypeLikeNested = Any
    DTypeLikeSave = Union[np.dtype[Any], None, type[Any], str, tuple[_DTypeLikeNested, int], tuple[_DTypeLikeNested, _ShapeLike], tuple[_DTypeLikeNested, _DTypeLikeNested], list[Any], _SupportsDType[np.dtype[Any]]]
else:
    DTypeLikeSave: Any = None
try:
    from cftime import datetime as CFTimeDatetime
except ImportError:
    CFTimeDatetime = np.datetime64
DatetimeLike: TypeAlias = Union[pd.Timestamp, datetime.datetime, np.datetime64, CFTimeDatetime]

class Alignable(Protocol):
    """Represents any Xarray type that supports alignment.

    It may be ``Dataset``, ``DataArray`` or ``Coordinates``. This protocol class
    is needed since those types do not all have a common base class.

    """

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[Hashable]:
        ...
T_Alignable = TypeVar('T_Alignable', bound='Alignable')
T_Backend = TypeVar('T_Backend', bound='BackendEntrypoint')
T_Dataset = TypeVar('T_Dataset', bound='Dataset')
T_DataArray = TypeVar('T_DataArray', bound='DataArray')
T_Variable = TypeVar('T_Variable', bound='Variable')
T_Coordinates = TypeVar('T_Coordinates', bound='Coordinates')
T_Array = TypeVar('T_Array', bound='AbstractArray')
T_Index = TypeVar('T_Index', bound='Index')
T_Xarray = TypeVar('T_Xarray', 'DataArray', 'Dataset')
T_DataArrayOrSet = TypeVar('T_DataArrayOrSet', bound=Union['Dataset', 'DataArray'])
T_DataWithCoords = TypeVar('T_DataWithCoords', bound='DataWithCoords')
T_DuckArray = TypeVar('T_DuckArray', bound=Any, covariant=True)
T_ExtensionArray = TypeVar('T_ExtensionArray', bound=pd.api.extensions.ExtensionArray)
ScalarOrArray = Union['ArrayLike', np.generic]
VarCompatible = Union['Variable', 'ScalarOrArray']
DaCompatible = Union['DataArray', 'VarCompatible']
DsCompatible = Union['Dataset', 'DaCompatible']
GroupByCompatible = Union['Dataset', 'DataArray']
Dims = Union[str, Collection[Hashable], 'ellipsis', None]
T_ChunkDim: TypeAlias = Union[str, int, Literal['auto'], None, tuple[int, ...]]
T_ChunkDimFreq: TypeAlias = Union['TimeResampler', T_ChunkDim]
T_ChunksFreq: TypeAlias = Union[T_ChunkDim, Mapping[Any, T_ChunkDimFreq]]
T_Chunks: TypeAlias = Union[T_ChunkDim, Mapping[Any, T_ChunkDim]]
T_NormalizedChunks = tuple[tuple[int, ...], ...]
DataVars = Mapping[Any, Any]
ErrorOptions = Literal['raise', 'ignore']
ErrorOptionsWithWarn = Literal['raise', 'warn', 'ignore']
CompatOptions = Literal['identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override', 'minimal']
ConcatOptions = Literal['all', 'minimal', 'different']
CombineAttrsOptions = Union[Literal['drop', 'identical', 'no_conflicts', 'drop_conflicts', 'override'], Callable[..., Any]]
JoinOptions = Literal['outer', 'inner', 'left', 'right', 'exact', 'override']
Interp1dOptions = Literal['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial']
InterpolantOptions = Literal['barycentric', 'krogh', 'pchip', 'spline', 'akima']
InterpOptions = Union[Interp1dOptions, InterpolantOptions]
DatetimeUnitOptions = Literal['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'μs', 'ns', 'ps', 'fs', 'as', None]
NPDatetimeUnitOptions = Literal['D', 'h', 'm', 's', 'ms', 'us', 'ns']
QueryEngineOptions = Literal['python', 'numexpr', None]
QueryParserOptions = Literal['pandas', 'python']
ReindexMethodOptions = Literal['nearest', 'pad', 'ffill', 'backfill', 'bfill', None]
PadModeOptions = Literal['constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap']
PadReflectOptions = Literal['even', 'odd', None]
CFCalendar = Literal['standard', 'gregorian', 'proleptic_gregorian', 'noleap', '365_day', '360_day', 'julian', 'all_leap', '366_day']
CoarsenBoundaryOptions = Literal['exact', 'trim', 'pad']
SideOptions = Literal['left', 'right']
InclusiveOptions = Literal['both', 'neither', 'left', 'right']
ScaleOptions = Literal['linear', 'symlog', 'log', 'logit', None]
HueStyleOptions = Literal['continuous', 'discrete', None]
AspectOptions = Union[Literal['auto', 'equal'], float, None]
ExtendOptions = Literal['neither', 'both', 'min', 'max', None]
_T = TypeVar('_T')
NestedSequence = Union[_T, Sequence[_T], Sequence[Sequence[_T]], Sequence[Sequence[Sequence[_T]]], Sequence[Sequence[Sequence[Sequence[_T]]]]]
QuantileMethods = Literal['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased', 'normal_unbiased', 'lower', 'higher', 'midpoint', 'nearest']
NetcdfWriteModes = Literal['w', 'a']
ZarrWriteModes = Literal['w', 'w-', 'a', 'a-', 'r+', 'r']
GroupKey = Any
GroupIndex = Union[int, slice, list[int]]
GroupIndices = tuple[GroupIndex, ...]
Bins = Union[int, Sequence[int], Sequence[float], Sequence[pd.Timestamp], np.ndarray, pd.Index]