from __future__ import annotations
import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Final, Literal, Protocol, SupportsIndex, TypeVar, Union, overload, runtime_checkable
import numpy as np
try:
    if sys.version_info >= (3, 11):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
except ImportError:
    if TYPE_CHECKING:
        raise
    else:
        Self: Any = None

class Default(Enum):
    token: Final = 0
_default = Default.token
_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_dtype = np.dtype
_DType = TypeVar('_DType', bound=np.dtype[Any])
_DType_co = TypeVar('_DType_co', covariant=True, bound=np.dtype[Any])
_ScalarType = TypeVar('_ScalarType', bound=np.generic)
_ScalarType_co = TypeVar('_ScalarType_co', bound=np.generic, covariant=True)

@runtime_checkable
class _SupportsDType(Protocol[_DType_co]):
    pass
_DTypeLike = Union[np.dtype[_ScalarType], type[_ScalarType], _SupportsDType[np.dtype[_ScalarType]]]
_IntOrUnknown = int
_Shape = tuple[_IntOrUnknown, ...]
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
_ShapeType = TypeVar('_ShapeType', bound=Any)
_ShapeType_co = TypeVar('_ShapeType_co', bound=Any, covariant=True)
_Axis = int
_Axes = tuple[_Axis, ...]
_AxisLike = Union[_Axis, _Axes]
_Chunks = tuple[_Shape, ...]
_NormalizedChunks = tuple[tuple[int, ...], ...]
T_ChunkDim: TypeAlias = Union[int, Literal['auto'], None, tuple[int, ...]]
T_Chunks: TypeAlias = Union[T_ChunkDim, Mapping[Any, T_ChunkDim]]
_Dim = Hashable
_Dims = tuple[_Dim, ...]
_DimsLike = Union[str, Iterable[_Dim]]
_IndexKey = Union[int, slice, 'ellipsis']
_IndexKeys = tuple[Union[_IndexKey], ...]
_IndexKeyLike = Union[_IndexKey, _IndexKeys]
_AttrsLike = Union[Mapping[Any, Any], None]

class _SupportsReal(Protocol[_T_co]):
    pass

class _SupportsImag(Protocol[_T_co]):
    pass

@runtime_checkable
class _array(Protocol[_ShapeType_co, _DType_co]):
    """
    Minimal duck array named array uses.

    Corresponds to np.ndarray.
    """

@runtime_checkable
class _arrayfunction(_array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    @overload
    def __getitem__(self, key: _arrayfunction[Any, Any] | tuple[_arrayfunction[Any, Any], ...], /) -> _arrayfunction[Any, _DType_co]:
        ...

    @overload
    def __getitem__(self, key: _IndexKeyLike, /) -> Any:
        ...

    def __getitem__(self, key: _IndexKeyLike | _arrayfunction[Any, Any] | tuple[_arrayfunction[Any, Any], ...], /) -> _arrayfunction[Any, _DType_co] | Any:
        ...

    @overload
    def __array__(self, dtype: None=..., /, *, copy: None | bool=...) -> np.ndarray[Any, _DType_co]:
        ...

    @overload
    def __array__(self, dtype: _DType, /, *, copy: None | bool=...) -> np.ndarray[Any, _DType]:
        ...

    def __array__(self, dtype: _DType | None=..., /, *, copy: None | bool=...) -> np.ndarray[Any, _DType] | np.ndarray[Any, _DType_co]:
        ...

    def __array_ufunc__(self, ufunc: Any, method: Any, *inputs: Any, **kwargs: Any) -> Any:
        ...

    def __array_function__(self, func: Callable[..., Any], types: Iterable[type], args: Iterable[Any], kwargs: Mapping[str, Any]) -> Any:
        ...

@runtime_checkable
class _arrayapi(_array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    def __getitem__(self, key: _IndexKeyLike | Any, /) -> _arrayapi[Any, Any]:
        ...

    def __array_namespace__(self) -> ModuleType:
        ...
_arrayfunction_or_api = (_arrayfunction, _arrayapi)
duckarray = Union[_arrayfunction[_ShapeType_co, _DType_co], _arrayapi[_ShapeType_co, _DType_co]]
DuckArray = _arrayfunction[Any, np.dtype[_ScalarType_co]]

@runtime_checkable
class _chunkedarray(_array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Minimal chunked duck array.

    Corresponds to np.ndarray.
    """

@runtime_checkable
class _chunkedarrayfunction(_arrayfunction[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Chunked duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

@runtime_checkable
class _chunkedarrayapi(_arrayapi[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Chunked duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """
_chunkedarrayfunction_or_api = (_chunkedarrayfunction, _chunkedarrayapi)
chunkedduckarray = Union[_chunkedarrayfunction[_ShapeType_co, _DType_co], _chunkedarrayapi[_ShapeType_co, _DType_co]]

@runtime_checkable
class _sparsearray(_array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Minimal sparse duck array.

    Corresponds to np.ndarray.
    """

@runtime_checkable
class _sparsearrayfunction(_arrayfunction[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Sparse duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

@runtime_checkable
class _sparsearrayapi(_arrayapi[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Sparse duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """
_sparsearrayfunction_or_api = (_sparsearrayfunction, _sparsearrayapi)
sparseduckarray = Union[_sparsearrayfunction[_ShapeType_co, _DType_co], _sparsearrayapi[_ShapeType_co, _DType_co]]
ErrorOptions = Literal['raise', 'ignore']
ErrorOptionsWithWarn = Literal['raise', 'warn', 'ignore']