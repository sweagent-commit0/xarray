from __future__ import annotations
import copy
import datetime
import inspect
import itertools
import math
import sys
import warnings
from collections import defaultdict
from collections.abc import Collection, Hashable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from functools import partial
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import IO, TYPE_CHECKING, Any, Callable, Generic, Literal, cast, overload
import numpy as np
from pandas.api.types import is_extension_array_dtype
try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from xarray.core import alignment, duck_array_ops, formatting, formatting_html, ops, utils
from xarray.core import dtypes as xrdtypes
from xarray.core._aggregations import DatasetAggregations
from xarray.core.alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from xarray.core.arithmetic import DatasetArithmetic
from xarray.core.common import DataWithCoords, _contains_datetime_like_objects, get_chunksizes
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import Coordinates, DatasetCoordinates, assert_coordinate_consistent, create_coords_with_default_indexes
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.indexes import Index, Indexes, PandasIndex, PandasMultiIndex, assert_no_index_corrupted, create_default_index_implicit, filter_indexes_from_coords, isel_indexes, remove_unused_levels_categories, roll_indexes
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import dataset_merge_method, dataset_update_method, merge_coordinates_without_align, merge_core
from xarray.core.missing import get_clean_interp_index
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import Bins, NetcdfWriteModes, QuantileMethods, Self, T_ChunkDim, T_ChunksFreq, T_DataArray, T_DataArrayOrSet, T_Dataset, ZarrWriteModes
from xarray.core.utils import Default, Frozen, FrozenMappingWarningOnValuesAccess, HybridMappingProxy, OrderedSet, _default, decode_numpy_dict_values, drop_dims_from_indexers, either_dict_or_kwargs, emit_user_level_warning, infix_dims, is_dict_like, is_duck_array, is_duck_dask_array, is_scalar, maybe_wrap_array
from xarray.core.variable import IndexVariable, Variable, as_variable, broadcast_variables, calculate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import array_type, is_chunked_array
from xarray.plot.accessor import DatasetPlotAccessor
from xarray.util.deprecation_helpers import _deprecate_positional_args, deprecate_dims
if TYPE_CHECKING:
    from dask.dataframe import DataFrame as DaskDataFrame
    from dask.delayed import Delayed
    from numpy.typing import ArrayLike
    from xarray.backends import AbstractDataStore, ZarrStore
    from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes
    from xarray.core.dataarray import DataArray
    from xarray.core.groupby import DatasetGroupBy
    from xarray.core.merge import CoercibleMapping, CoercibleValue, _MergeResult
    from xarray.core.resample import DatasetResample
    from xarray.core.rolling import DatasetCoarsen, DatasetRolling
    from xarray.core.types import CFCalendar, CoarsenBoundaryOptions, CombineAttrsOptions, CompatOptions, DataVars, DatetimeLike, DatetimeUnitOptions, Dims, DsCompatible, ErrorOptions, ErrorOptionsWithWarn, InterpOptions, JoinOptions, PadModeOptions, PadReflectOptions, QueryEngineOptions, QueryParserOptions, ReindexMethodOptions, SideOptions, T_ChunkDimFreq, T_Xarray
    from xarray.core.weighted import DatasetWeighted
    from xarray.groupers import Grouper, Resampler
    from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint
_DATETIMEINDEX_COMPONENTS = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'nanosecond', 'date', 'time', 'dayofyear', 'weekofyear', 'dayofweek', 'quarter']

def _get_virtual_variable(variables, key: Hashable, dim_sizes: Mapping | None=None) -> tuple[Hashable, Hashable, Variable]:
    """Get a virtual variable (e.g., 'time.year') from a dict of xarray.Variable
    objects (if possible)

    """
    pass

def _get_chunk(var: Variable, chunks, chunkmanager: ChunkManagerEntrypoint):
    """
    Return map from each dim to chunk sizes, accounting for backend's preferred chunks.
    """
    pass

def as_dataset(obj: Any) -> Dataset:
    """Cast the given object to a Dataset.

    Handles Datasets, DataArrays and dictionaries of variables. A new Dataset
    object is only created if the provided object is not already one.
    """
    pass

def _get_func_args(func, param_names):
    """Use `inspect.signature` to try accessing `func` args. Otherwise, ensure
    they are provided by user.
    """
    pass

def _initialize_curvefit_params(params, p0, bounds, func_args):
    """Set initial guess and bounds for curvefit.
    Priority: 1) passed args 2) func signature 3) scipy defaults
    """
    pass

def merge_data_and_coords(data_vars: DataVars, coords) -> _MergeResult:
    """Used in Dataset.__init__."""
    pass

class DataVariables(Mapping[Any, 'DataArray']):
    __slots__ = ('_dataset',)

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __iter__(self) -> Iterator[Hashable]:
        return (key for key in self._dataset._variables if key not in self._dataset._coord_names)

    def __len__(self) -> int:
        length = len(self._dataset._variables) - len(self._dataset._coord_names)
        assert length >= 0, 'something is wrong with Dataset._coord_names'
        return length

    def __contains__(self, key: Hashable) -> bool:
        return key in self._dataset._variables and key not in self._dataset._coord_names

    def __getitem__(self, key: Hashable) -> DataArray:
        if key not in self._dataset._coord_names:
            return self._dataset[key]
        raise KeyError(key)

    def __repr__(self) -> str:
        return formatting.data_vars_repr(self)

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        """Mapping from data variable names to dtypes.

        Cannot be modified directly, but is updated when adding new variables.

        See Also
        --------
        Dataset.dtype
        """
        pass

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython."""
        pass

class _LocIndexer(Generic[T_Dataset]):
    __slots__ = ('dataset',)

    def __init__(self, dataset: T_Dataset):
        self.dataset = dataset

    def __getitem__(self, key: Mapping[Any, Any]) -> T_Dataset:
        if not utils.is_dict_like(key):
            raise TypeError('can only lookup dictionaries from Dataset.loc')
        return self.dataset.sel(key)

    def __setitem__(self, key, value) -> None:
        if not utils.is_dict_like(key):
            raise TypeError(f'can only set locations defined by dictionaries from Dataset.loc. Got: {key}')
        dim_indexers = map_index_queries(self.dataset, key).dim_indexers
        self.dataset[dim_indexers] = value

class Dataset(DataWithCoords, DatasetAggregations, DatasetArithmetic, Mapping[Hashable, 'DataArray']):
    """A multi-dimensional, in memory, array database.

    A dataset resembles an in-memory representation of a NetCDF file,
    and consists of variables, coordinates and attributes which
    together form a self describing dataset.

    Dataset implements the mapping interface with keys given by variable
    names and values given by DataArray objects for each variable name.

    By default, pandas indexes are created for one dimensional variables with
    name equal to their dimension (i.e., :term:`Dimension coordinate`) so those
    variables can be readily used as coordinates for label based indexing. When a
    :py:class:`~xarray.Coordinates` object is passed to ``coords``, any existing
    index(es) built from those coordinates will be added to the Dataset.

    To load data from a file or file-like object, use the `open_dataset`
    function.

    Parameters
    ----------
    data_vars : dict-like, optional
        A mapping from variable names to :py:class:`~xarray.DataArray`
        objects, :py:class:`~xarray.Variable` objects or to tuples of
        the form ``(dims, data[, attrs])`` which can be used as
        arguments to create a new ``Variable``. Each dimension must
        have the same length in all variables in which it appears.

        The following notations are accepted:

        - mapping {var name: DataArray}
        - mapping {var name: Variable}
        - mapping {var name: (dimension name, array-like)}
        - mapping {var name: (tuple of dimension names, array-like)}
        - mapping {dimension name: array-like}
          (if array-like is not a scalar it will be automatically moved to coords,
          see below)

        Each dimension must have the same length in all variables in
        which it appears.
    coords : :py:class:`~xarray.Coordinates` or dict-like, optional
        A :py:class:`~xarray.Coordinates` object or another mapping in
        similar form as the `data_vars` argument, except that each item
        is saved on the dataset as a "coordinate".
        These variables have an associated meaning: they describe
        constant/fixed/independent quantities, unlike the
        varying/measured/dependent quantities that belong in
        `variables`.

        The following notations are accepted for arbitrary mappings:

        - mapping {coord name: DataArray}
        - mapping {coord name: Variable}
        - mapping {coord name: (dimension name, array-like)}
        - mapping {coord name: (tuple of dimension names, array-like)}
        - mapping {dimension name: array-like}
          (the dimension name is implicitly set to be the same as the
          coord name)

        The last notation implies either that the coordinate value is a scalar
        or that it is a 1-dimensional array and the coord name is the same as
        the dimension name (i.e., a :term:`Dimension coordinate`). In the latter
        case, the 1-dimensional array will be assumed to give index values
        along the dimension with the same name.

        Alternatively, a :py:class:`~xarray.Coordinates` object may be used in
        order to explicitly pass indexes (e.g., a multi-index or any custom
        Xarray index) or to bypass the creation of a default index for any
        :term:`Dimension coordinate` included in that object.

    attrs : dict-like, optional
        Global attributes to save on this dataset.

    Examples
    --------
    In this example dataset, we will represent measurements of the temperature
    and pressure that were made under various conditions:

    * the measurements were made on four different days;
    * they were made at two separate locations, which we will represent using
      their latitude and longitude; and
    * they were made using three instrument developed by three different
      manufacturers, which we will refer to using the strings `'manufac1'`,
      `'manufac2'`, and `'manufac3'`.

    >>> np.random.seed(0)
    >>> temperature = 15 + 8 * np.random.randn(2, 3, 4)
    >>> precipitation = 10 * np.random.rand(2, 3, 4)
    >>> lon = [-99.83, -99.32]
    >>> lat = [42.25, 42.21]
    >>> instruments = ["manufac1", "manufac2", "manufac3"]
    >>> time = pd.date_range("2014-09-06", periods=4)
    >>> reference_time = pd.Timestamp("2014-09-05")

    Here, we initialize the dataset with multiple dimensions. We use the string
    `"loc"` to represent the location dimension of the data, the string
    `"instrument"` to represent the instrument manufacturer dimension, and the
    string `"time"` for the time dimension.

    >>> ds = xr.Dataset(
    ...     data_vars=dict(
    ...         temperature=(["loc", "instrument", "time"], temperature),
    ...         precipitation=(["loc", "instrument", "time"], precipitation),
    ...     ),
    ...     coords=dict(
    ...         lon=("loc", lon),
    ...         lat=("loc", lat),
    ...         instrument=instruments,
    ...         time=time,
    ...         reference_time=reference_time,
    ...     ),
    ...     attrs=dict(description="Weather related data."),
    ... )
    >>> ds
    <xarray.Dataset> Size: 552B
    Dimensions:         (loc: 2, instrument: 3, time: 4)
    Coordinates:
        lon             (loc) float64 16B -99.83 -99.32
        lat             (loc) float64 16B 42.25 42.21
      * instrument      (instrument) <U8 96B 'manufac1' 'manufac2' 'manufac3'
      * time            (time) datetime64[ns] 32B 2014-09-06 ... 2014-09-09
        reference_time  datetime64[ns] 8B 2014-09-05
    Dimensions without coordinates: loc
    Data variables:
        temperature     (loc, instrument, time) float64 192B 29.11 18.2 ... 9.063
        precipitation   (loc, instrument, time) float64 192B 4.562 5.684 ... 1.613
    Attributes:
        description:  Weather related data.

    Find out where the coldest temperature was and what values the
    other variables had:

    >>> ds.isel(ds.temperature.argmin(...))
    <xarray.Dataset> Size: 80B
    Dimensions:         ()
    Coordinates:
        lon             float64 8B -99.32
        lat             float64 8B 42.21
        instrument      <U8 32B 'manufac3'
        time            datetime64[ns] 8B 2014-09-06
        reference_time  datetime64[ns] 8B 2014-09-05
    Data variables:
        temperature     float64 8B -5.424
        precipitation   float64 8B 9.884
    Attributes:
        description:  Weather related data.

    """
    _attrs: dict[Hashable, Any] | None
    _cache: dict[str, Any]
    _coord_names: set[Hashable]
    _dims: dict[Hashable, int]
    _encoding: dict[Hashable, Any] | None
    _close: Callable[[], None] | None
    _indexes: dict[Hashable, Index]
    _variables: dict[Hashable, Variable]
    __slots__ = ('_attrs', '_cache', '_coord_names', '_dims', '_encoding', '_close', '_indexes', '_variables', '__weakref__')

    def __init__(self, data_vars: DataVars | None=None, coords: Mapping[Any, Any] | None=None, attrs: Mapping[Any, Any] | None=None) -> None:
        if data_vars is None:
            data_vars = {}
        if coords is None:
            coords = {}
        both_data_and_coords = set(data_vars) & set(coords)
        if both_data_and_coords:
            raise ValueError(f'variables {both_data_and_coords!r} are found in both data_vars and coords')
        if isinstance(coords, Dataset):
            coords = coords._variables
        variables, coord_names, dims, indexes, _ = merge_data_and_coords(data_vars, coords)
        self._attrs = dict(attrs) if attrs else None
        self._close = None
        self._encoding = None
        self._variables = variables
        self._coord_names = coord_names
        self._dims = dims
        self._indexes = indexes

    def __eq__(self, other: DsCompatible) -> Self:
        return super().__eq__(other)

    @classmethod
    def load_store(cls, store, decoder=None) -> Self:
        """Create a new dataset from the contents of a backends.*DataStore
        object
        """
        pass

    @property
    def variables(self) -> Frozen[Hashable, Variable]:
        """Low level interface to Dataset contents as dict of Variable objects.

        This ordered dictionary is frozen to prevent mutation that could
        violate Dataset invariants. It contains all variable objects
        constituting the Dataset, including both data variables and
        coordinates.
        """
        pass

    @property
    def attrs(self) -> dict[Any, Any]:
        """Dictionary of global attributes on this dataset"""
        pass

    @property
    def encoding(self) -> dict[Any, Any]:
        """Dictionary of global encoding attributes on this dataset"""
        pass

    def drop_encoding(self) -> Self:
        """Return a new Dataset without encoding on the dataset or any of its
        variables/coords."""
        pass

    @property
    def dims(self) -> Frozen[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `Dataset.sizes` and `DataArray.sizes` for consistently named
        properties. This property will be changed to return a type more consistent with
        `DataArray.dims` in the future, i.e. a set of dimension names.

        See Also
        --------
        Dataset.sizes
        DataArray.dims
        """
        pass

    @property
    def sizes(self) -> Frozen[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `Dataset.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See Also
        --------
        DataArray.sizes
        """
        pass

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        """Mapping from data variable names to dtypes.

        Cannot be modified directly, but is updated when adding new variables.

        See Also
        --------
        DataArray.dtype
        """
        pass

    def load(self, **kwargs) -> Self:
        """Manually trigger loading and/or computation of this dataset's data
        from disk or a remote source into memory and return this dataset.
        Unlike compute, the original dataset is modified and returned.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.compute``.

        See Also
        --------
        dask.compute
        """
        pass

    def __dask_tokenize__(self) -> object:
        from dask.base import normalize_token
        return normalize_token((type(self), self._variables, self._coord_names, self._attrs or None))

    def __dask_graph__(self):
        graphs = {k: v.__dask_graph__() for k, v in self.variables.items()}
        graphs = {k: v for k, v in graphs.items() if v is not None}
        if not graphs:
            return None
        else:
            try:
                from dask.highlevelgraph import HighLevelGraph
                return HighLevelGraph.merge(*graphs.values())
            except ImportError:
                from dask import sharedict
                return sharedict.merge(*graphs.values())

    def __dask_keys__(self):
        import dask
        return [v.__dask_keys__() for v in self.variables.values() if dask.is_dask_collection(v)]

    def __dask_layers__(self):
        import dask
        return sum((v.__dask_layers__() for v in self.variables.values() if dask.is_dask_collection(v)), ())

    @property
    def __dask_optimize__(self):
        import dask.array as da
        return da.Array.__dask_optimize__

    @property
    def __dask_scheduler__(self):
        import dask.array as da
        return da.Array.__dask_scheduler__

    def __dask_postcompute__(self):
        return (self._dask_postcompute, ())

    def __dask_postpersist__(self):
        return (self._dask_postpersist, ())

    def compute(self, **kwargs) -> Self:
        """Manually trigger loading and/or computation of this dataset's data
        from disk or a remote source into memory and return a new dataset.
        Unlike load, the original dataset is left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.compute``.

        Returns
        -------
        object : Dataset
            New object with lazy data variables and coordinates as in-memory arrays.

        See Also
        --------
        dask.compute
        """
        pass

    def _persist_inplace(self, **kwargs) -> Self:
        """Persist all Dask arrays in memory"""
        pass

    def persist(self, **kwargs) -> Self:
        """Trigger computation, keeping data as dask arrays

        This operation can be used to trigger computation on underlying dask
        arrays, similar to ``.compute()`` or ``.load()``.  However this
        operation keeps the data as dask arrays. This is particularly useful
        when using the dask.distributed scheduler and you want to load a large
        amount of data into distributed memory.
        Like compute (but unlike load), the original dataset is left unaltered.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.persist``.

        Returns
        -------
        object : Dataset
            New object with all dask-backed coordinates and data variables as persisted dask arrays.

        See Also
        --------
        dask.persist
        """
        pass

    @classmethod
    def _construct_direct(cls, variables: dict[Any, Variable], coord_names: set[Hashable], dims: dict[Any, int] | None=None, attrs: dict | None=None, indexes: dict[Any, Index] | None=None, encoding: dict | None=None, close: Callable[[], None] | None=None) -> Self:
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        pass

    def _replace(self, variables: dict[Hashable, Variable] | None=None, coord_names: set[Hashable] | None=None, dims: dict[Any, int] | None=None, attrs: dict[Hashable, Any] | None | Default=_default, indexes: dict[Hashable, Index] | None=None, encoding: dict | None | Default=_default, inplace: bool=False) -> Self:
        """Fastpath constructor for internal use.

        Returns an object with optionally with replaced attributes.

        Explicitly passed arguments are *not* copied when placed on the new
        dataset. It is up to the caller to ensure that they have the right type
        and are not used elsewhere.
        """
        pass

    def _replace_with_new_dims(self, variables: dict[Hashable, Variable], coord_names: set | None=None, attrs: dict[Hashable, Any] | None | Default=_default, indexes: dict[Hashable, Index] | None=None, inplace: bool=False) -> Self:
        """Replace variables with recalculated dimensions."""
        pass

    def _replace_vars_and_dims(self, variables: dict[Hashable, Variable], coord_names: set | None=None, dims: dict[Hashable, int] | None=None, attrs: dict[Hashable, Any] | None | Default=_default, inplace: bool=False) -> Self:
        """Deprecated version of _replace_with_new_dims().

        Unlike _replace_with_new_dims(), this method always recalculates
        indexes from variables.
        """
        pass

    def _overwrite_indexes(self, indexes: Mapping[Hashable, Index], variables: Mapping[Hashable, Variable] | None=None, drop_variables: list[Hashable] | None=None, drop_indexes: list[Hashable] | None=None, rename_dims: Mapping[Hashable, Hashable] | None=None) -> Self:
        """Maybe replace indexes.

        This function may do a lot more depending on index query
        results.

        """
        pass

    def copy(self, deep: bool=False, data: DataVars | None=None) -> Self:
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy of each of the component variable is made, so
        that the underlying memory region of the new dataset is the same as in
        the original dataset.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, default: False
            Whether each component variable is loaded into memory and copied onto
            the new object. Default is False.
        data : dict-like or None, optional
            Data to use in the new object. Each item in `data` must have same
            shape as corresponding data variable in original. When `data` is
            used, `deep` is ignored for the data variables and only used for
            coords.

        Returns
        -------
        object : Dataset
            New object with dimensions, attributes, coordinates, name, encoding,
            and optionally data copied from original.

        Examples
        --------
        Shallow copy versus deep copy

        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset(
        ...     {"foo": da, "bar": ("x", [-1, 2])},
        ...     coords={"x": ["one", "two"]},
        ... )
        >>> ds.copy()
        <xarray.Dataset> Size: 88B
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 24B 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 48B 1.764 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 16B -1 2

        >>> ds_0 = ds.copy(deep=False)
        >>> ds_0["foo"][0, 0] = 7
        >>> ds_0
        <xarray.Dataset> Size: 88B
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 24B 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 48B 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 16B -1 2

        >>> ds
        <xarray.Dataset> Size: 88B
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 24B 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 48B 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 16B -1 2

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> ds.copy(data={"foo": np.arange(6).reshape(2, 3), "bar": ["a", "b"]})
        <xarray.Dataset> Size: 80B
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 24B 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) int64 48B 0 1 2 3 4 5
            bar      (x) <U1 8B 'a' 'b'

        >>> ds
        <xarray.Dataset> Size: 88B
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 24B 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 48B 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 16B -1 2

        See Also
        --------
        pandas.DataFrame.copy
        """
        pass

    def __copy__(self) -> Self:
        return self._copy(deep=False)

    def __deepcopy__(self, memo: dict[int, Any] | None=None) -> Self:
        return self._copy(deep=True, memo=memo)

    def as_numpy(self) -> Self:
        """
        Coerces wrapped data and coordinates into numpy arrays, returning a Dataset.

        See also
        --------
        DataArray.as_numpy
        DataArray.to_numpy : Returns only the data as a numpy.ndarray object.
        """
        pass

    def _copy_listed(self, names: Iterable[Hashable]) -> Self:
        """Create a new Dataset with the listed variables from this dataset and
        the all relevant coordinates. Skips all validation.
        """
        pass

    def _construct_dataarray(self, name: Hashable) -> DataArray:
        """Construct a DataArray by indexing this dataset"""
        pass

    @property
    def _attr_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for attribute-style access"""
        pass

    @property
    def _item_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for key-completion"""
        pass

    def __contains__(self, key: object) -> bool:
        """The 'in' operator will return true or false depending on whether
        'key' is an array in the dataset or not.
        """
        return key in self._variables

    def __len__(self) -> int:
        return len(self.data_vars)

    def __bool__(self) -> bool:
        return bool(self.data_vars)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.data_vars)
    if TYPE_CHECKING:
        __array__ = None
    else:

        def __array__(self, dtype=None, copy=None):
            raise TypeError('cannot directly convert an xarray.Dataset into a numpy array. Instead, create an xarray.DataArray first, either with indexing on the Dataset or by invoking the `to_dataarray()` method.')

    @property
    def nbytes(self) -> int:
        """
        Total bytes consumed by the data arrays of all variables in this dataset.

        If the backend array for any variable does not include ``nbytes``, estimates
        the total bytes for that array based on the ``size`` and ``dtype``.
        """
        pass

    @property
    def loc(self) -> _LocIndexer[Self]:
        """Attribute for location based indexing. Only supports __getitem__,
        and only when the key is a dict of the form {dim: labels}.
        """
        pass

    @overload
    def __getitem__(self, key: Hashable) -> DataArray:
        ...

    @overload
    def __getitem__(self, key: Iterable[Hashable]) -> Self:
        ...

    def __getitem__(self, key: Mapping[Any, Any] | Hashable | Iterable[Hashable]) -> Self | DataArray:
        """Access variables or coordinates of this dataset as a
        :py:class:`~xarray.DataArray` or a subset of variables or a indexed dataset.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        from xarray.core.formatting import shorten_list_repr
        if utils.is_dict_like(key):
            return self.isel(**key)
        if utils.hashable(key):
            try:
                return self._construct_dataarray(key)
            except KeyError as e:
                raise KeyError(f'No variable named {key!r}. Variables on the dataset include {shorten_list_repr(list(self.variables.keys()), max_items=10)}') from e
        if utils.iterable_of_hashable(key):
            return self._copy_listed(key)
        raise ValueError(f'Unsupported key-type {type(key)}')

    def __setitem__(self, key: Hashable | Iterable[Hashable] | Mapping, value: Any) -> None:
        """Add an array to this dataset.
        Multiple arrays can be added at the same time, in which case each of
        the following operations is applied to the respective value.

        If key is dict-like, update all variables in the dataset
        one by one with the given value at the given location.
        If the given value is also a dataset, select corresponding variables
        in the given value and in the dataset to be changed.

        If value is a `
        from .dataarray import DataArray`, call its `select_vars()` method, rename it
        to `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is a `Variable` object (or tuple of form
        ``(dims, data[, attrs])``), add it to this dataset as a new
        variable.
        """
        from xarray.core.dataarray import DataArray
        if utils.is_dict_like(key):
            value = self._setitem_check(key, value)
            processed = []
            for name, var in self.items():
                try:
                    var[key] = value[name]
                    processed.append(name)
                except Exception as e:
                    if processed:
                        raise RuntimeError(f"An error occurred while setting values of the variable '{name}'. The following variables have been successfully updated:\n{processed}") from e
                    else:
                        raise e
        elif utils.hashable(key):
            if isinstance(value, Dataset):
                raise TypeError('Cannot assign a Dataset to a single key - only a DataArray or Variable object can be stored under a single key.')
            self.update({key: value})
        elif utils.iterable_of_hashable(key):
            keylist = list(key)
            if len(keylist) == 0:
                raise ValueError('Empty list of variables to be set')
            if len(keylist) == 1:
                self.update({keylist[0]: value})
            else:
                if len(keylist) != len(value):
                    raise ValueError(f'Different lengths of variables to be set ({len(keylist)}) and data used as input for setting ({len(value)})')
                if isinstance(value, Dataset):
                    self.update(dict(zip(keylist, value.data_vars.values())))
                elif isinstance(value, DataArray):
                    raise ValueError('Cannot assign single DataArray to multiple keys')
                else:
                    self.update(dict(zip(keylist, value)))
        else:
            raise ValueError(f'Unsupported key-type {type(key)}')

    def _setitem_check(self, key, value):
        """Consistency check for __setitem__

        When assigning values to a subset of a Dataset, do consistency check beforehand
        to avoid leaving the dataset in a partially updated state when an error occurs.
        """
        pass

    def __delitem__(self, key: Hashable) -> None:
        """Remove a variable from this dataset."""
        assert_no_index_corrupted(self.xindexes, {key})
        if key in self._indexes:
            del self._indexes[key]
        del self._variables[key]
        self._coord_names.discard(key)
        self._dims = calculate_dimensions(self._variables)
    __hash__ = None

    def _all_compat(self, other: Self, compat_str: str) -> bool:
        """Helper function for equals and identical"""
        pass

    def broadcast_equals(self, other: Self) -> bool:
        """Two Datasets are broadcast equal if they are equal after
        broadcasting all variables against each other.

        For example, variables that are scalar in one dataset but non-scalar in
        the other dataset can still be broadcast equal if the the non-scalar
        variable is a constant.

        Examples
        --------

        # 2D array with shape (1, 3)

        >>> data = np.array([[1, 2, 3]])
        >>> a = xr.Dataset(
        ...     {"variable_name": (("space", "time"), data)},
        ...     coords={"space": [0], "time": [0, 1, 2]},
        ... )
        >>> a
        <xarray.Dataset> Size: 56B
        Dimensions:        (space: 1, time: 3)
        Coordinates:
          * space          (space) int64 8B 0
          * time           (time) int64 24B 0 1 2
        Data variables:
            variable_name  (space, time) int64 24B 1 2 3

        # 2D array with shape (3, 1)

        >>> data = np.array([[1], [2], [3]])
        >>> b = xr.Dataset(
        ...     {"variable_name": (("time", "space"), data)},
        ...     coords={"time": [0, 1, 2], "space": [0]},
        ... )
        >>> b
        <xarray.Dataset> Size: 56B
        Dimensions:        (time: 3, space: 1)
        Coordinates:
          * time           (time) int64 24B 0 1 2
          * space          (space) int64 8B 0
        Data variables:
            variable_name  (time, space) int64 24B 1 2 3

        .equals returns True if two Datasets have the same values, dimensions, and coordinates. .broadcast_equals returns True if the
        results of broadcasting two Datasets against each other have the same values, dimensions, and coordinates.

        >>> a.equals(b)
        False

        >>> a.broadcast_equals(b)
        True

        >>> a2, b2 = xr.broadcast(a, b)
        >>> a2.equals(b2)
        True

        See Also
        --------
        Dataset.equals
        Dataset.identical
        Dataset.broadcast
        """
        pass

    def equals(self, other: Self) -> bool:
        """Two Datasets are equal if they have matching variables and
        coordinates, all of which are equal.

        Datasets can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for ``Dataset``
        does element-wise comparisons (like numpy.ndarrays).

        Examples
        --------

        # 2D array with shape (1, 3)

        >>> data = np.array([[1, 2, 3]])
        >>> dataset1 = xr.Dataset(
        ...     {"variable_name": (("space", "time"), data)},
        ...     coords={"space": [0], "time": [0, 1, 2]},
        ... )
        >>> dataset1
        <xarray.Dataset> Size: 56B
        Dimensions:        (space: 1, time: 3)
        Coordinates:
          * space          (space) int64 8B 0
          * time           (time) int64 24B 0 1 2
        Data variables:
            variable_name  (space, time) int64 24B 1 2 3

        # 2D array with shape (3, 1)

        >>> data = np.array([[1], [2], [3]])
        >>> dataset2 = xr.Dataset(
        ...     {"variable_name": (("time", "space"), data)},
        ...     coords={"time": [0, 1, 2], "space": [0]},
        ... )
        >>> dataset2
        <xarray.Dataset> Size: 56B
        Dimensions:        (time: 3, space: 1)
        Coordinates:
          * time           (time) int64 24B 0 1 2
          * space          (space) int64 8B 0
        Data variables:
            variable_name  (time, space) int64 24B 1 2 3
        >>> dataset1.equals(dataset2)
        False

        >>> dataset1.broadcast_equals(dataset2)
        True

        .equals returns True if two Datasets have the same values, dimensions, and coordinates. .broadcast_equals returns True if the
        results of broadcasting two Datasets against each other have the same values, dimensions, and coordinates.

        Similar for missing values too:

        >>> ds1 = xr.Dataset(
        ...     {
        ...         "temperature": (["x", "y"], [[1, np.nan], [3, 4]]),
        ...     },
        ...     coords={"x": [0, 1], "y": [0, 1]},
        ... )

        >>> ds2 = xr.Dataset(
        ...     {
        ...         "temperature": (["x", "y"], [[1, np.nan], [3, 4]]),
        ...     },
        ...     coords={"x": [0, 1], "y": [0, 1]},
        ... )
        >>> ds1.equals(ds2)
        True

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.identical
        """
        pass

    def identical(self, other: Self) -> bool:
        """Like equals, but also checks all dataset attributes and the
        attributes on all variables and coordinates.

        Example
        -------

        >>> a = xr.Dataset(
        ...     {"Width": ("X", [1, 2, 3])},
        ...     coords={"X": [1, 2, 3]},
        ...     attrs={"units": "m"},
        ... )
        >>> b = xr.Dataset(
        ...     {"Width": ("X", [1, 2, 3])},
        ...     coords={"X": [1, 2, 3]},
        ...     attrs={"units": "m"},
        ... )
        >>> c = xr.Dataset(
        ...     {"Width": ("X", [1, 2, 3])},
        ...     coords={"X": [1, 2, 3]},
        ...     attrs={"units": "ft"},
        ... )
        >>> a
        <xarray.Dataset> Size: 48B
        Dimensions:  (X: 3)
        Coordinates:
          * X        (X) int64 24B 1 2 3
        Data variables:
            Width    (X) int64 24B 1 2 3
        Attributes:
            units:    m

        >>> b
        <xarray.Dataset> Size: 48B
        Dimensions:  (X: 3)
        Coordinates:
          * X        (X) int64 24B 1 2 3
        Data variables:
            Width    (X) int64 24B 1 2 3
        Attributes:
            units:    m

        >>> c
        <xarray.Dataset> Size: 48B
        Dimensions:  (X: 3)
        Coordinates:
          * X        (X) int64 24B 1 2 3
        Data variables:
            Width    (X) int64 24B 1 2 3
        Attributes:
            units:    ft

        >>> a.equals(b)
        True

        >>> a.identical(b)
        True

        >>> a.equals(c)
        True

        >>> a.identical(c)
        False

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.equals
        """
        pass

    @property
    def indexes(self) -> Indexes[pd.Index]:
        """Mapping of pandas.Index objects used for label based indexing.

        Raises an error if this Dataset has indexes that cannot be coerced
        to pandas.Index objects.

        See Also
        --------
        Dataset.xindexes

        """
        pass

    @property
    def xindexes(self) -> Indexes[Index]:
        """Mapping of :py:class:`~xarray.indexes.Index` objects
        used for label based indexing.
        """
        pass

    @property
    def coords(self) -> DatasetCoordinates:
        """Mapping of :py:class:`~xarray.DataArray` objects corresponding to
        coordinate variables.

        See Also
        --------
        Coordinates
        """
        pass

    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables"""
        pass

    def set_coords(self, names: Hashable | Iterable[Hashable]) -> Self:
        """Given names of one or more variables, set them as coordinates

        Parameters
        ----------
        names : hashable or iterable of hashable
            Name(s) of variables in this dataset to convert into coordinates.

        Examples
        --------
        >>> dataset = xr.Dataset(
        ...     {
        ...         "pressure": ("time", [1.013, 1.2, 3.5]),
        ...         "time": pd.date_range("2023-01-01", periods=3),
        ...     }
        ... )
        >>> dataset
        <xarray.Dataset> Size: 48B
        Dimensions:   (time: 3)
        Coordinates:
          * time      (time) datetime64[ns] 24B 2023-01-01 2023-01-02 2023-01-03
        Data variables:
            pressure  (time) float64 24B 1.013 1.2 3.5

        >>> dataset.set_coords("pressure")
        <xarray.Dataset> Size: 48B
        Dimensions:   (time: 3)
        Coordinates:
            pressure  (time) float64 24B 1.013 1.2 3.5
          * time      (time) datetime64[ns] 24B 2023-01-01 2023-01-02 2023-01-03
        Data variables:
            *empty*

        On calling ``set_coords`` , these data variables are converted to coordinates, as shown in the final dataset.

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.swap_dims
        Dataset.assign_coords
        """
        pass

    def reset_coords(self, names: Dims=None, drop: bool=False) -> Self:
        """Given names of coordinates, reset them to become variables

        Parameters
        ----------
        names : str, Iterable of Hashable or None, optional
            Name(s) of non-index coordinates in this dataset to reset into
            variables. By default, all non-index coordinates are reset.
        drop : bool, default: False
            If True, remove coordinates instead of converting them into
            variables.

        Examples
        --------
        >>> dataset = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             ["time", "lat", "lon"],
        ...             [[[25, 26], [27, 28]], [[29, 30], [31, 32]]],
        ...         ),
        ...         "precipitation": (
        ...             ["time", "lat", "lon"],
        ...             [[[0.5, 0.8], [0.2, 0.4]], [[0.3, 0.6], [0.7, 0.9]]],
        ...         ),
        ...     },
        ...     coords={
        ...         "time": pd.date_range(start="2023-01-01", periods=2),
        ...         "lat": [40, 41],
        ...         "lon": [-80, -79],
        ...         "altitude": 1000,
        ...     },
        ... )

        # Dataset before resetting coordinates

        >>> dataset
        <xarray.Dataset> Size: 184B
        Dimensions:        (time: 2, lat: 2, lon: 2)
        Coordinates:
          * time           (time) datetime64[ns] 16B 2023-01-01 2023-01-02
          * lat            (lat) int64 16B 40 41
          * lon            (lon) int64 16B -80 -79
            altitude       int64 8B 1000
        Data variables:
            temperature    (time, lat, lon) int64 64B 25 26 27 28 29 30 31 32
            precipitation  (time, lat, lon) float64 64B 0.5 0.8 0.2 0.4 0.3 0.6 0.7 0.9

        # Reset the 'altitude' coordinate

        >>> dataset_reset = dataset.reset_coords("altitude")

        # Dataset after resetting coordinates

        >>> dataset_reset
        <xarray.Dataset> Size: 184B
        Dimensions:        (time: 2, lat: 2, lon: 2)
        Coordinates:
          * time           (time) datetime64[ns] 16B 2023-01-01 2023-01-02
          * lat            (lat) int64 16B 40 41
          * lon            (lon) int64 16B -80 -79
        Data variables:
            temperature    (time, lat, lon) int64 64B 25 26 27 28 29 30 31 32
            precipitation  (time, lat, lon) float64 64B 0.5 0.8 0.2 0.4 0.3 0.6 0.7 0.9
            altitude       int64 8B 1000

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.set_coords
        """
        pass

    def dump_to_store(self, store: AbstractDataStore, **kwargs) -> None:
        """Store dataset contents to a backends.*DataStore object."""
        pass

    def to_netcdf(self, path: str | PathLike | None=None, mode: NetcdfWriteModes='w', format: T_NetcdfTypes | None=None, group: str | None=None, engine: T_NetcdfEngine | None=None, encoding: Mapping[Any, Mapping[str, Any]] | None=None, unlimited_dims: Iterable[Hashable] | None=None, compute: bool=True, invalid_netcdf: bool=False) -> bytes | Delayed | None:
        """Write dataset contents to a netCDF file.

        Parameters
        ----------
        path : str, path-like or file-like, optional
            Path to which to save this dataset. File-like objects are only
            supported by the scipy engine. If no path is provided, this
            function returns the resulting netCDF file as bytes; in this case,
            we need to use scipy, which does not support netCDF version 4 (the
            default format becomes NETCDF3_64BIT).
        mode : {"w", "a"}, default: "w"
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten.
        format : {"NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT",                   "NETCDF3_CLASSIC"}, optional
            File format for the resulting netCDF file:

            * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
              features.
            * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
              netCDF 3 compatible API features.
            * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
              which fully supports 2+ GB files, but is only compatible with
              clients linked against netCDF version 3.6.0 or later.
            * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
              handle 2+ GB files very well.

            All formats are supported by the netCDF4-python library.
            scipy.io.netcdf only supports the last two formats.

            The default format is NETCDF4 if you are saving a file to disk and
            have the netCDF4-python library available. Otherwise, xarray falls
            back to using scipy to write netCDF files and defaults to the
            NETCDF3_64BIT format (scipy does not support netCDF4).
        group : str, optional
            Path to the netCDF4 group in the given file to open (only works for
            format='NETCDF4'). The group(s) will be created if necessary.
        engine : {"netcdf4", "scipy", "h5netcdf"}, optional
            Engine to use when writing netCDF files. If not provided, the
            default engine is chosen based on available dependencies, with a
            preference for 'netcdf4' if writing to a file on disk.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"my_variable": {"dtype": "int16", "scale_factor": 0.1,
            "zlib": True}, ...}``.
            If ``encoding`` is specified the original encoding of the variables of
            the dataset is ignored.

            The `h5netcdf` engine supports both the NetCDF4-style compression
            encoding parameters ``{"zlib": True, "complevel": 9}`` and the h5py
            ones ``{"compression": "gzip", "compression_opts": 9}``.
            This allows using any compression plugin installed in the HDF5
            library, e.g. LZF.

        unlimited_dims : iterable of hashable, optional
            Dimension(s) that should be serialized as unlimited dimensions.
            By default, no dimensions are treated as unlimited dimensions.
            Note that unlimited_dims may also be set via
            ``dataset.encoding["unlimited_dims"]``.
        compute: bool, default: True
            If true compute immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed later.
        invalid_netcdf: bool, default: False
            Only valid along with ``engine="h5netcdf"``. If True, allow writing
            hdf5 files which are invalid netcdf as described in
            https://github.com/h5netcdf/h5netcdf.

        Returns
        -------
            * ``bytes`` if path is None
            * ``dask.delayed.Delayed`` if compute is False
            * None otherwise

        See Also
        --------
        DataArray.to_netcdf
        """
        pass

    def to_zarr(self, store: MutableMapping | str | PathLike[str] | None=None, chunk_store: MutableMapping | str | PathLike | None=None, mode: ZarrWriteModes | None=None, synchronizer=None, group: str | None=None, encoding: Mapping | None=None, *, compute: bool=True, consolidated: bool | None=None, append_dim: Hashable | None=None, region: Mapping[str, slice | Literal['auto']] | Literal['auto'] | None=None, safe_chunks: bool=True, storage_options: dict[str, str] | None=None, zarr_version: int | None=None, write_empty_chunks: bool | None=None, chunkmanager_store_kwargs: dict[str, Any] | None=None) -> ZarrStore | Delayed:
        """Write dataset contents to a zarr group.

        Zarr chunks are determined in the following way:

        - From the ``chunks`` attribute in each variable's ``encoding``
          (can be set via `Dataset.chunk`).
        - If the variable is a Dask array, from the dask chunks
        - If neither Dask chunks nor encoding chunks are present, chunks will
          be determined automatically by Zarr
        - If both Dask chunks and encoding chunks are present, encoding chunks
          will be used, provided that there is a many-to-one relationship between
          encoding chunks and dask chunks (i.e. Dask chunks are bigger than and
          evenly divide encoding chunks); otherwise raise a ``ValueError``.
          This restriction ensures that no synchronization / locks are required
          when writing. To disable this restriction, use ``safe_chunks=False``.

        Parameters
        ----------
        store : MutableMapping, str or path-like, optional
            Store or path to directory in local or remote file system.
        chunk_store : MutableMapping, str or path-like, optional
            Store or path to directory in local or remote file system only for Zarr
            array chunks. Requires zarr-python v2.4.0 or later.
        mode : {"w", "w-", "a", "a-", r+", None}, optional
            Persistence mode: "w" means create (overwrite if exists);
            "w-" means create (fail if exists);
            "a" means override all existing variables including dimension coordinates (create if does not exist);
            "a-" means only append those variables that have ``append_dim``.
            "r+" means modify existing array *values* only (raise an error if
            any metadata or shapes would change).
            The default mode is "a" if ``append_dim`` is set. Otherwise, it is
            "r+" if ``region`` is set and ``w-`` otherwise.
        synchronizer : object, optional
            Zarr array synchronizer.
        group : str, optional
            Group path. (a.k.a. `path` in zarr terminology.)
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"my_variable": {"dtype": "int16", "scale_factor": 0.1,}, ...}``
        compute : bool, default: True
            If True write array data immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed to write
            array data later. Metadata is always updated eagerly.
        consolidated : bool, optional
            If True, apply :func:`zarr.convenience.consolidate_metadata`
            after writing metadata and read existing stores with consolidated
            metadata; if False, do not. The default (`consolidated=None`) means
            write consolidated metadata and attempt to read consolidated
            metadata for existing stores (falling back to non-consolidated).

            When the experimental ``zarr_version=3``, ``consolidated`` must be
            either be ``None`` or ``False``.
        append_dim : hashable, optional
            If set, the dimension along which the data will be appended. All
            other dimensions on overridden variables must remain the same size.
        region : dict or "auto", optional
            Optional mapping from dimension names to either a) ``"auto"``, or b) integer
            slices, indicating the region of existing zarr array(s) in which to write
            this dataset's data.

            If ``"auto"`` is provided the existing store will be opened and the region
            inferred by matching indexes. ``"auto"`` can be used as a single string,
            which will automatically infer the region for all dimensions, or as
            dictionary values for specific dimensions mixed together with explicit
            slices for other dimensions.

            Alternatively integer slices can be provided; for example, ``{'x': slice(0,
            1000), 'y': slice(10000, 11000)}`` would indicate that values should be
            written to the region ``0:1000`` along ``x`` and ``10000:11000`` along
            ``y``.

            Two restrictions apply to the use of ``region``:

            - If ``region`` is set, _all_ variables in a dataset must have at
              least one dimension in common with the region. Other variables
              should be written in a separate single call to ``to_zarr()``.
            - Dimensions cannot be included in both ``region`` and
              ``append_dim`` at the same time. To create empty arrays to fill
              in with ``region``, use a separate call to ``to_zarr()`` with
              ``compute=False``. See "Appending to existing Zarr stores" in
              the reference documentation for full details.

            Users are expected to ensure that the specified region aligns with
            Zarr chunk boundaries, and that dask chunks are also aligned.
            Xarray makes limited checks that these multiple chunk boundaries line up.
            It is possible to write incomplete chunks and corrupt the data with this
            option if you are not careful.
        safe_chunks : bool, default: True
            If True, only allow writes to when there is a many-to-one relationship
            between Zarr chunks (specified in encoding) and Dask chunks.
            Set False to override this restriction; however, data may become corrupted
            if Zarr arrays are written in parallel. This option may be useful in combination
            with ``compute=False`` to initialize a Zarr from an existing
            Dataset with arbitrary chunk structure.
        storage_options : dict, optional
            Any additional parameters for the storage backend (ignored for local
            paths).
        zarr_version : int or None, optional
            The desired zarr spec version to target (currently 2 or 3). The
            default of None will attempt to determine the zarr version from
            ``store`` when possible, otherwise defaulting to 2.
        write_empty_chunks : bool or None, optional
            If True, all chunks will be stored regardless of their
            contents. If False, each chunk is compared to the array's fill value
            prior to storing. If a chunk is uniformly equal to the fill value, then
            that chunk is not be stored, and the store entry for that chunk's key
            is deleted. This setting enables sparser storage, as only chunks with
            non-fill-value data are stored, at the expense of overhead associated
            with checking the data of each chunk. If None (default) fall back to
            specification(s) in ``encoding`` or Zarr defaults. A ``ValueError``
            will be raised if the value of this (if not None) differs with
            ``encoding``.
        chunkmanager_store_kwargs : dict, optional
            Additional keyword arguments passed on to the `ChunkManager.store` method used to store
            chunked arrays. For example for a dask array additional kwargs will be passed eventually to
            :py:func:`dask.array.store()`. Experimental API that should not be relied upon.

        Returns
        -------
            * ``dask.delayed.Delayed`` if compute is False
            * ZarrStore otherwise

        References
        ----------
        https://zarr.readthedocs.io/

        Notes
        -----
        Zarr chunking behavior:
            If chunks are found in the encoding argument or attribute
            corresponding to any DataArray, those chunks are used.
            If a DataArray is a dask array, it is written with those chunks.
            If not other chunks are found, Zarr uses its own heuristics to
            choose automatic chunk sizes.

        encoding:
            The encoding attribute (if exists) of the DataArray(s) will be
            used. Override any existing encodings by providing the ``encoding`` kwarg.

        See Also
        --------
        :ref:`io.zarr`
            The I/O user guide, with more details and examples.
        """
        pass

    def __repr__(self) -> str:
        return formatting.dataset_repr(self)

    def info(self, buf: IO | None=None) -> None:
        """
        Concise summary of a Dataset variables and attributes.

        Parameters
        ----------
        buf : file-like, default: sys.stdout
            writable buffer

        See Also
        --------
        pandas.DataFrame.assign
        ncdump : netCDF's ncdump
        """
        pass

    @property
    def chunks(self) -> Mapping[Hashable, tuple[int, ...]]:
        """
        Mapping from dimension names to block lengths for this dataset's data, or None if
        the underlying data is not a dask array.
        Cannot be modified directly, but can be modified by calling .chunk().

        Same as Dataset.chunksizes, but maintained for backwards compatibility.

        See Also
        --------
        Dataset.chunk
        Dataset.chunksizes
        xarray.unify_chunks
        """
        pass

    @property
    def chunksizes(self) -> Mapping[Hashable, tuple[int, ...]]:
        """
        Mapping from dimension names to block lengths for this dataset's data, or None if
        the underlying data is not a dask array.
        Cannot be modified directly, but can be modified by calling .chunk().

        Same as Dataset.chunks.

        See Also
        --------
        Dataset.chunk
        Dataset.chunks
        xarray.unify_chunks
        """
        pass

    def chunk(self, chunks: T_ChunksFreq={}, name_prefix: str='xarray-', token: str | None=None, lock: bool=False, inline_array: bool=False, chunked_array_type: str | ChunkManagerEntrypoint | None=None, from_array_kwargs=None, **chunks_kwargs: T_ChunkDimFreq) -> Self:
        """Coerce all arrays in this dataset into dask arrays with the given
        chunks.

        Non-dask arrays in this dataset will be converted to dask arrays. Dask
        arrays will be rechunked to the given chunk sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Along datetime-like dimensions, a :py:class:`groupers.TimeResampler` object is also accepted.

        Parameters
        ----------
        chunks : int, tuple of int, "auto" or mapping of hashable to int or a TimeResampler, optional
            Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, or
            ``{"x": 5, "y": 5}`` or ``{"x": 5, "time": TimeResampler(freq="YE")}``.
        name_prefix : str, default: "xarray-"
            Prefix for the name of any new dask arrays.
        token : str, optional
            Token uniquely identifying this dataset.
        lock : bool, default: False
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.
        inline_array: bool, default: False
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.
        chunked_array_type: str, optional
            Which chunked array type to coerce this datasets' arrays to.
            Defaults to 'dask' if installed, else whatever is registered via the `ChunkManagerEntryPoint` system.
            Experimental API that should not be relied upon.
        from_array_kwargs: dict, optional
            Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create
            chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.
            For example, with dask as the default chunked array type, this method would pass additional kwargs
            to :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.
        **chunks_kwargs : {dim: chunks, ...}, optional
            The keyword arguments form of ``chunks``.
            One of chunks or chunks_kwargs must be provided

        Returns
        -------
        chunked : xarray.Dataset

        See Also
        --------
        Dataset.chunks
        Dataset.chunksizes
        xarray.unify_chunks
        dask.array.from_array
        """
        pass

    def _validate_indexers(self, indexers: Mapping[Any, Any], missing_dims: ErrorOptionsWithWarn='raise') -> Iterator[tuple[Hashable, int | slice | np.ndarray | Variable]]:
        """Here we make sure
        + indexer has a valid keys
        + indexer is in a valid data type
        + string indexers are cast to the appropriate date type if the
          associated index is a DatetimeIndex or CFTimeIndex
        """
        pass

    def _validate_interp_indexers(self, indexers: Mapping[Any, Any]) -> Iterator[tuple[Hashable, Variable]]:
        """Variant of _validate_indexers to be used for interpolation"""
        pass

    def _get_indexers_coords_and_indexes(self, indexers):
        """Extract coordinates and indexes from indexers.

        Only coordinate with a name different from any of self.variables will
        be attached.
        """
        pass

    def isel(self, indexers: Mapping[Any, Any] | None=None, drop: bool=False, missing_dims: ErrorOptionsWithWarn='raise', **indexers_kwargs: Any) -> Self:
        """Returns a new dataset with each array indexed along the specified
        dimension(s).

        This method selects values from each array using its `__getitem__`
        method, except this method does not require knowing the order of
        each array's dimensions.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, default: False
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this dataset, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        Examples
        --------

        >>> dataset = xr.Dataset(
        ...     {
        ...         "math_scores": (
        ...             ["student", "test"],
        ...             [[90, 85, 92], [78, 80, 85], [95, 92, 98]],
        ...         ),
        ...         "english_scores": (
        ...             ["student", "test"],
        ...             [[88, 90, 92], [75, 82, 79], [93, 96, 91]],
        ...         ),
        ...     },
        ...     coords={
        ...         "student": ["Alice", "Bob", "Charlie"],
        ...         "test": ["Test 1", "Test 2", "Test 3"],
        ...     },
        ... )

        # A specific element from the dataset is selected

        >>> dataset.isel(student=1, test=0)
        <xarray.Dataset> Size: 68B
        Dimensions:         ()
        Coordinates:
            student         <U7 28B 'Bob'
            test            <U6 24B 'Test 1'
        Data variables:
            math_scores     int64 8B 78
            english_scores  int64 8B 75

        # Indexing with a slice using isel

        >>> slice_of_data = dataset.isel(student=slice(0, 2), test=slice(0, 2))
        >>> slice_of_data
        <xarray.Dataset> Size: 168B
        Dimensions:         (student: 2, test: 2)
        Coordinates:
          * student         (student) <U7 56B 'Alice' 'Bob'
          * test            (test) <U6 48B 'Test 1' 'Test 2'
        Data variables:
            math_scores     (student, test) int64 32B 90 85 78 80
            english_scores  (student, test) int64 32B 88 90 75 82

        >>> index_array = xr.DataArray([0, 2], dims="student")
        >>> indexed_data = dataset.isel(student=index_array)
        >>> indexed_data
        <xarray.Dataset> Size: 224B
        Dimensions:         (student: 2, test: 3)
        Coordinates:
          * student         (student) <U7 56B 'Alice' 'Charlie'
          * test            (test) <U6 72B 'Test 1' 'Test 2' 'Test 3'
        Data variables:
            math_scores     (student, test) int64 48B 90 85 92 95 92 98
            english_scores  (student, test) int64 48B 88 90 92 93 96 91

        See Also
        --------
        Dataset.sel
        DataArray.isel

        :doc:`xarray-tutorial:intermediate/indexing/indexing`
            Tutorial material on indexing with Xarray objects

        :doc:`xarray-tutorial:fundamentals/02.1_indexing_Basic`
            Tutorial material on basics of indexing

        """
        pass

    def sel(self, indexers: Mapping[Any, Any] | None=None, method: str | None=None, tolerance: int | float | Iterable[int | float] | None=None, drop: bool=False, **indexers_kwargs: Any) -> Self:
        """Returns a new dataset with each array indexed by tick labels
        along the specified dimension(s).

        In contrast to `Dataset.isel`, indexers for this method should use
        labels instead of integers.

        Under the hood, this method is powered by using pandas's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for inexact matches:

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            variable and dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this dataset, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.isel
        DataArray.sel

        :doc:`xarray-tutorial:intermediate/indexing/indexing`
            Tutorial material on indexing with Xarray objects

        :doc:`xarray-tutorial:fundamentals/02.1_indexing_Basic`
            Tutorial material on basics of indexing

        """
        pass

    def head(self, indexers: Mapping[Any, int] | int | None=None, **indexers_kwargs: Any) -> Self:
        """Returns a new dataset with the first `n` values of each array
        for the specified dimension(s).

        Parameters
        ----------
        indexers : dict or int, default: 5
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Examples
        --------
        >>> dates = pd.date_range(start="2023-01-01", periods=5)
        >>> pageviews = [1200, 1500, 900, 1800, 2000]
        >>> visitors = [800, 1000, 600, 1200, 1500]
        >>> dataset = xr.Dataset(
        ...     {
        ...         "pageviews": (("date"), pageviews),
        ...         "visitors": (("date"), visitors),
        ...     },
        ...     coords={"date": dates},
        ... )
        >>> busiest_days = dataset.sortby("pageviews", ascending=False)
        >>> busiest_days.head()
        <xarray.Dataset> Size: 120B
        Dimensions:    (date: 5)
        Coordinates:
          * date       (date) datetime64[ns] 40B 2023-01-05 2023-01-04 ... 2023-01-03
        Data variables:
            pageviews  (date) int64 40B 2000 1800 1500 1200 900
            visitors   (date) int64 40B 1500 1200 1000 800 600

        # Retrieve the 3 most busiest days in terms of pageviews

        >>> busiest_days.head(3)
        <xarray.Dataset> Size: 72B
        Dimensions:    (date: 3)
        Coordinates:
          * date       (date) datetime64[ns] 24B 2023-01-05 2023-01-04 2023-01-02
        Data variables:
            pageviews  (date) int64 24B 2000 1800 1500
            visitors   (date) int64 24B 1500 1200 1000

        # Using a dictionary to specify the number of elements for specific dimensions

        >>> busiest_days.head({"date": 3})
        <xarray.Dataset> Size: 72B
        Dimensions:    (date: 3)
        Coordinates:
          * date       (date) datetime64[ns] 24B 2023-01-05 2023-01-04 2023-01-02
        Data variables:
            pageviews  (date) int64 24B 2000 1800 1500
            visitors   (date) int64 24B 1500 1200 1000

        See Also
        --------
        Dataset.tail
        Dataset.thin
        DataArray.head
        """
        pass

    def tail(self, indexers: Mapping[Any, int] | int | None=None, **indexers_kwargs: Any) -> Self:
        """Returns a new dataset with the last `n` values of each array
        for the specified dimension(s).

        Parameters
        ----------
        indexers : dict or int, default: 5
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Examples
        --------
        >>> activity_names = ["Walking", "Running", "Cycling", "Swimming", "Yoga"]
        >>> durations = [30, 45, 60, 45, 60]  # in minutes
        >>> energies = [150, 300, 250, 400, 100]  # in calories
        >>> dataset = xr.Dataset(
        ...     {
        ...         "duration": (["activity"], durations),
        ...         "energy_expenditure": (["activity"], energies),
        ...     },
        ...     coords={"activity": activity_names},
        ... )
        >>> sorted_dataset = dataset.sortby("energy_expenditure", ascending=False)
        >>> sorted_dataset
        <xarray.Dataset> Size: 240B
        Dimensions:             (activity: 5)
        Coordinates:
          * activity            (activity) <U8 160B 'Swimming' 'Running' ... 'Yoga'
        Data variables:
            duration            (activity) int64 40B 45 45 60 30 60
            energy_expenditure  (activity) int64 40B 400 300 250 150 100

        # Activities with the least energy expenditures using tail()

        >>> sorted_dataset.tail(3)
        <xarray.Dataset> Size: 144B
        Dimensions:             (activity: 3)
        Coordinates:
          * activity            (activity) <U8 96B 'Cycling' 'Walking' 'Yoga'
        Data variables:
            duration            (activity) int64 24B 60 30 60
            energy_expenditure  (activity) int64 24B 250 150 100

        >>> sorted_dataset.tail({"activity": 3})
        <xarray.Dataset> Size: 144B
        Dimensions:             (activity: 3)
        Coordinates:
          * activity            (activity) <U8 96B 'Cycling' 'Walking' 'Yoga'
        Data variables:
            duration            (activity) int64 24B 60 30 60
            energy_expenditure  (activity) int64 24B 250 150 100

        See Also
        --------
        Dataset.head
        Dataset.thin
        DataArray.tail
        """
        pass

    def thin(self, indexers: Mapping[Any, int] | int | None=None, **indexers_kwargs: Any) -> Self:
        """Returns a new dataset with each array indexed along every `n`-th
        value for the specified dimension(s)

        Parameters
        ----------
        indexers : dict or int
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Examples
        --------
        >>> x_arr = np.arange(0, 26)
        >>> x_arr
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
               17, 18, 19, 20, 21, 22, 23, 24, 25])
        >>> x = xr.DataArray(
        ...     np.reshape(x_arr, (2, 13)),
        ...     dims=("x", "y"),
        ...     coords={"x": [0, 1], "y": np.arange(0, 13)},
        ... )
        >>> x_ds = xr.Dataset({"foo": x})
        >>> x_ds
        <xarray.Dataset> Size: 328B
        Dimensions:  (x: 2, y: 13)
        Coordinates:
          * x        (x) int64 16B 0 1
          * y        (y) int64 104B 0 1 2 3 4 5 6 7 8 9 10 11 12
        Data variables:
            foo      (x, y) int64 208B 0 1 2 3 4 5 6 7 8 ... 17 18 19 20 21 22 23 24 25

        >>> x_ds.thin(3)
        <xarray.Dataset> Size: 88B
        Dimensions:  (x: 1, y: 5)
        Coordinates:
          * x        (x) int64 8B 0
          * y        (y) int64 40B 0 3 6 9 12
        Data variables:
            foo      (x, y) int64 40B 0 3 6 9 12
        >>> x.thin({"x": 2, "y": 5})
        <xarray.DataArray (x: 1, y: 3)> Size: 24B
        array([[ 0,  5, 10]])
        Coordinates:
          * x        (x) int64 8B 0
          * y        (y) int64 24B 0 5 10

        See Also
        --------
        Dataset.head
        Dataset.tail
        DataArray.thin
        """
        pass

    def broadcast_like(self, other: T_DataArrayOrSet, exclude: Iterable[Hashable] | None=None) -> Self:
        """Broadcast this DataArray against another Dataset or DataArray.
        This is equivalent to xr.broadcast(other, self)[1]

        Parameters
        ----------
        other : Dataset or DataArray
            Object against which to broadcast this array.
        exclude : iterable of hashable, optional
            Dimensions that must not be broadcasted

        """
        pass

    def _reindex_callback(self, aligner: alignment.Aligner, dim_pos_indexers: dict[Hashable, Any], variables: dict[Hashable, Variable], indexes: dict[Hashable, Index], fill_value: Any, exclude_dims: frozenset[Hashable], exclude_vars: frozenset[Hashable]) -> Self:
        """Callback called from ``Aligner`` to create a new reindexed Dataset."""
        pass

    def reindex_like(self, other: T_Xarray, method: ReindexMethodOptions=None, tolerance: float | Iterable[float] | str | None=None, copy: bool=True, fill_value: Any=xrdtypes.NA) -> Self:
        """
        Conform this object onto the indexes of another object, for indexes which the
        objects share. Missing values are filled with ``fill_value``. The default fill
        value is NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to pandas.Index objects, which provides coordinates upon
            which to index the variables in this dataset. The indexes on this
            other object need not be the same as the indexes on this
            dataset. Any mis-matched index values will be filled in with
            NaN, and any mis-matched dimension names will simply be ignored.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill", None}, optional
            Method to use for filling index values from other not found in this
            dataset:

            - None (default): don't fill gaps
            - "pad" / "ffill": propagate last valid index value forward
            - "backfill" / "bfill": propagate next valid index value backward
            - "nearest": use nearest valid index value

        tolerance : float | Iterable[float] | str | None, default: None
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like must be the same size as the index and its dtype
            must exactly match the index’s type.
        copy : bool, default: True
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like maps
            variable names to fill values.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but coordinates from the
            other object.

        See Also
        --------
        Dataset.reindex
        DataArray.reindex_like
        align

        """
        pass

    def reindex(self, indexers: Mapping[Any, Any] | None=None, method: ReindexMethodOptions=None, tolerance: float | Iterable[float] | str | None=None, copy: bool=True, fill_value: Any=xrdtypes.NA, **indexers_kwargs: Any) -> Self:
        """Conform this object onto a new set of indexes, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        indexers : dict, optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill", None}, optional
            Method to use for filling index values in ``indexers`` not found in
            this dataset:

            - None (default): don't fill gaps
            - "pad" / "ffill": propagate last valid index value forward
            - "backfill" / "bfill": propagate next valid index value backward
            - "nearest": use nearest valid index value

        tolerance : float | Iterable[float] | str | None, default: None
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like must be the same size as the index and its dtype
            must exactly match the index’s type.
        copy : bool, default: True
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like,
            maps variable names (including coordinates) to fill values.
        sparse : bool, default: False
            use sparse-array.
        **indexers_kwargs : {dim: indexer, ...}, optional
            Keyword arguments in the same form as ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.reindex_like
        align
        pandas.Index.get_indexer

        Examples
        --------
        Create a dataset with some fictional data.

        >>> x = xr.Dataset(
        ...     {
        ...         "temperature": ("station", 20 * np.random.rand(4)),
        ...         "pressure": ("station", 500 * np.random.rand(4)),
        ...     },
        ...     coords={"station": ["boston", "nyc", "seattle", "denver"]},
        ... )
        >>> x
        <xarray.Dataset> Size: 176B
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 112B 'boston' 'nyc' 'seattle' 'denver'
        Data variables:
            temperature  (station) float64 32B 10.98 14.3 12.06 10.9
            pressure     (station) float64 32B 211.8 322.9 218.8 445.9
        >>> x.indexes
        Indexes:
            station  Index(['boston', 'nyc', 'seattle', 'denver'], dtype='object', name='station')

        Create a new index and reindex the dataset. By default values in the new index that
        do not have corresponding records in the dataset are assigned `NaN`.

        >>> new_index = ["boston", "austin", "seattle", "lincoln"]
        >>> x.reindex({"station": new_index})
        <xarray.Dataset> Size: 176B
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 112B 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 32B 10.98 nan 12.06 nan
            pressure     (station) float64 32B 211.8 nan 218.8 nan

        We can fill in the missing values by passing a value to the keyword `fill_value`.

        >>> x.reindex({"station": new_index}, fill_value=0)
        <xarray.Dataset> Size: 176B
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 112B 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 32B 10.98 0.0 12.06 0.0
            pressure     (station) float64 32B 211.8 0.0 218.8 0.0

        We can also use different fill values for each variable.

        >>> x.reindex(
        ...     {"station": new_index}, fill_value={"temperature": 0, "pressure": 100}
        ... )
        <xarray.Dataset> Size: 176B
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 112B 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 32B 10.98 0.0 12.06 0.0
            pressure     (station) float64 32B 211.8 100.0 218.8 100.0

        Because the index is not monotonically increasing or decreasing, we cannot use arguments
        to the keyword method to fill the `NaN` values.

        >>> x.reindex({"station": new_index}, method="nearest")
        Traceback (most recent call last):
        ...
            raise ValueError('index must be monotonic increasing or decreasing')
        ValueError: index must be monotonic increasing or decreasing

        To further illustrate the filling functionality in reindex, we will create a
        dataset with a monotonically increasing index (for example, a sequence of dates).

        >>> x2 = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             "time",
        ...             [15.57, 12.77, np.nan, 0.3081, 16.59, 15.12],
        ...         ),
        ...         "pressure": ("time", 500 * np.random.rand(6)),
        ...     },
        ...     coords={"time": pd.date_range("01/01/2019", periods=6, freq="D")},
        ... )
        >>> x2
        <xarray.Dataset> Size: 144B
        Dimensions:      (time: 6)
        Coordinates:
          * time         (time) datetime64[ns] 48B 2019-01-01 2019-01-02 ... 2019-01-06
        Data variables:
            temperature  (time) float64 48B 15.57 12.77 nan 0.3081 16.59 15.12
            pressure     (time) float64 48B 481.8 191.7 395.9 264.4 284.0 462.8

        Suppose we decide to expand the dataset to cover a wider date range.

        >>> time_index2 = pd.date_range("12/29/2018", periods=10, freq="D")
        >>> x2.reindex({"time": time_index2})
        <xarray.Dataset> Size: 240B
        Dimensions:      (time: 10)
        Coordinates:
          * time         (time) datetime64[ns] 80B 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 80B nan nan nan 15.57 ... 0.3081 16.59 15.12 nan
            pressure     (time) float64 80B nan nan nan 481.8 ... 264.4 284.0 462.8 nan

        The index entries that did not have a value in the original data frame (for example, `2018-12-29`)
        are by default filled with NaN. If desired, we can fill in the missing values using one of several options.

        For example, to back-propagate the last valid value to fill the `NaN` values,
        pass `bfill` as an argument to the `method` keyword.

        >>> x3 = x2.reindex({"time": time_index2}, method="bfill")
        >>> x3
        <xarray.Dataset> Size: 240B
        Dimensions:      (time: 10)
        Coordinates:
          * time         (time) datetime64[ns] 80B 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 80B 15.57 15.57 15.57 15.57 ... 16.59 15.12 nan
            pressure     (time) float64 80B 481.8 481.8 481.8 481.8 ... 284.0 462.8 nan

        Please note that the `NaN` value present in the original dataset (at index value `2019-01-03`)
        will not be filled by any of the value propagation schemes.

        >>> x2.where(x2.temperature.isnull(), drop=True)
        <xarray.Dataset> Size: 24B
        Dimensions:      (time: 1)
        Coordinates:
          * time         (time) datetime64[ns] 8B 2019-01-03
        Data variables:
            temperature  (time) float64 8B nan
            pressure     (time) float64 8B 395.9
        >>> x3.where(x3.temperature.isnull(), drop=True)
        <xarray.Dataset> Size: 48B
        Dimensions:      (time: 2)
        Coordinates:
          * time         (time) datetime64[ns] 16B 2019-01-03 2019-01-07
        Data variables:
            temperature  (time) float64 16B nan nan
            pressure     (time) float64 16B 395.9 nan

        This is because filling while reindexing does not look at dataset values, but only compares
        the original and desired indexes. If you do want to fill in the `NaN` values present in the
        original dataset, use the :py:meth:`~Dataset.fillna()` method.

        """
        pass

    def _reindex(self, indexers: Mapping[Any, Any] | None=None, method: str | None=None, tolerance: int | float | Iterable[int | float] | None=None, copy: bool=True, fill_value: Any=xrdtypes.NA, sparse: bool=False, **indexers_kwargs: Any) -> Self:
        """
        Same as reindex but supports sparse option.
        """
        pass

    def interp(self, coords: Mapping[Any, Any] | None=None, method: InterpOptions='linear', assume_sorted: bool=False, kwargs: Mapping[str, Any] | None=None, method_non_numeric: str='nearest', **coords_kwargs: Any) -> Self:
        """Interpolate a Dataset onto new coordinates

        Performs univariate or multivariate interpolation of a Dataset onto
        new coordinates using scipy's interpolation routines. If interpolating
        along an existing dimension, :py:class:`scipy.interpolate.interp1d` is
        called.  When interpolating along multiple existing dimensions, an
        attempt is made to decompose the interpolation into multiple
        1-dimensional interpolations. If this is possible,
        :py:class:`scipy.interpolate.interp1d` is called. Otherwise,
        :py:func:`scipy.interpolate.interpn` is called.

        Parameters
        ----------
        coords : dict, optional
            Mapping from dimension names to the new coordinates.
            New coordinate can be a scalar, array-like or DataArray.
            If DataArrays are passed as new coordinates, their dimensions are
            used for the broadcasting. Missing values are skipped.
        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial",             "barycentric", "krogh", "pchip", "spline", "akima"}, default: "linear"
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation. Additional keyword
              arguments are passed to :py:func:`numpy.interp`
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial':
              are passed to :py:func:`scipy.interpolate.interp1d`. If
              ``method='polynomial'``, the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krogh', 'pchip', 'spline', 'akima': use their
              respective :py:class:`scipy.interpolate` classes.

        assume_sorted : bool, default: False
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs : dict, optional
            Additional keyword arguments passed to scipy's interpolator. Valid
            options and their behavior depend whether ``interp1d`` or
            ``interpn`` is used.
        method_non_numeric : {"nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method for non-numeric types. Passed on to :py:meth:`Dataset.reindex`.
            ``"nearest"`` is used by default.
        **coords_kwargs : {dim: coordinate, ...}, optional
            The keyword arguments form of ``coords``.
            One of coords or coords_kwargs must be provided.

        Returns
        -------
        interpolated : Dataset
            New dataset on the new coordinates.

        Notes
        -----
        scipy is required.

        See Also
        --------
        scipy.interpolate.interp1d
        scipy.interpolate.interpn

        :doc:`xarray-tutorial:fundamentals/02.2_manipulating_dimensions`
            Tutorial material on manipulating data resolution using :py:func:`~xarray.Dataset.interp`

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={
        ...         "a": ("x", [5, 7, 4]),
        ...         "b": (
        ...             ("x", "y"),
        ...             [[1, 4, 2, 9], [2, 7, 6, np.nan], [6, np.nan, 5, 8]],
        ...         ),
        ...     },
        ...     coords={"x": [0, 1, 2], "y": [10, 12, 14, 16]},
        ... )
        >>> ds
        <xarray.Dataset> Size: 176B
        Dimensions:  (x: 3, y: 4)
        Coordinates:
          * x        (x) int64 24B 0 1 2
          * y        (y) int64 32B 10 12 14 16
        Data variables:
            a        (x) int64 24B 5 7 4
            b        (x, y) float64 96B 1.0 4.0 2.0 9.0 2.0 7.0 6.0 nan 6.0 nan 5.0 8.0

        1D interpolation with the default method (linear):

        >>> ds.interp(x=[0, 0.75, 1.25, 1.75])
        <xarray.Dataset> Size: 224B
        Dimensions:  (x: 4, y: 4)
        Coordinates:
          * y        (y) int64 32B 10 12 14 16
          * x        (x) float64 32B 0.0 0.75 1.25 1.75
        Data variables:
            a        (x) float64 32B 5.0 6.5 6.25 4.75
            b        (x, y) float64 128B 1.0 4.0 2.0 nan 1.75 ... nan 5.0 nan 5.25 nan

        1D interpolation with a different method:

        >>> ds.interp(x=[0, 0.75, 1.25, 1.75], method="nearest")
        <xarray.Dataset> Size: 224B
        Dimensions:  (x: 4, y: 4)
        Coordinates:
          * y        (y) int64 32B 10 12 14 16
          * x        (x) float64 32B 0.0 0.75 1.25 1.75
        Data variables:
            a        (x) float64 32B 5.0 7.0 7.0 4.0
            b        (x, y) float64 128B 1.0 4.0 2.0 9.0 2.0 7.0 ... nan 6.0 nan 5.0 8.0

        1D extrapolation:

        >>> ds.interp(
        ...     x=[1, 1.5, 2.5, 3.5],
        ...     method="linear",
        ...     kwargs={"fill_value": "extrapolate"},
        ... )
        <xarray.Dataset> Size: 224B
        Dimensions:  (x: 4, y: 4)
        Coordinates:
          * y        (y) int64 32B 10 12 14 16
          * x        (x) float64 32B 1.0 1.5 2.5 3.5
        Data variables:
            a        (x) float64 32B 7.0 5.5 2.5 -0.5
            b        (x, y) float64 128B 2.0 7.0 6.0 nan 4.0 ... nan 12.0 nan 3.5 nan

        2D interpolation:

        >>> ds.interp(x=[0, 0.75, 1.25, 1.75], y=[11, 13, 15], method="linear")
        <xarray.Dataset> Size: 184B
        Dimensions:  (x: 4, y: 3)
        Coordinates:
          * x        (x) float64 32B 0.0 0.75 1.25 1.75
          * y        (y) int64 24B 11 13 15
        Data variables:
            a        (x) float64 32B 5.0 6.5 6.25 4.75
            b        (x, y) float64 96B 2.5 3.0 nan 4.0 5.625 ... nan nan nan nan nan
        """
        pass

    def interp_like(self, other: T_Xarray, method: InterpOptions='linear', assume_sorted: bool=False, kwargs: Mapping[str, Any] | None=None, method_non_numeric: str='nearest') -> Self:
        """Interpolate this object onto the coordinates of another object,
        filling the out of range values with NaN.

        If interpolating along a single existing dimension,
        :py:class:`scipy.interpolate.interp1d` is called. When interpolating
        along multiple existing dimensions, an attempt is made to decompose the
        interpolation into multiple 1-dimensional interpolations. If this is
        possible, :py:class:`scipy.interpolate.interp1d` is called. Otherwise,
        :py:func:`scipy.interpolate.interpn` is called.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to an 1d array-like, which provides coordinates upon
            which to index the variables in this dataset. Missing values are skipped.
        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial",             "barycentric", "krogh", "pchip", "spline", "akima"}, default: "linear"
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation. Additional keyword
              arguments are passed to :py:func:`numpy.interp`
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial':
              are passed to :py:func:`scipy.interpolate.interp1d`. If
              ``method='polynomial'``, the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krogh', 'pchip', 'spline', 'akima': use their
              respective :py:class:`scipy.interpolate` classes.

        assume_sorted : bool, default: False
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs : dict, optional
            Additional keyword passed to scipy's interpolator.
        method_non_numeric : {"nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method for non-numeric types. Passed on to :py:meth:`Dataset.reindex`.
            ``"nearest"`` is used by default.

        Returns
        -------
        interpolated : Dataset
            Another dataset by interpolating this dataset's data along the
            coordinates of the other object.

        Notes
        -----
        scipy is required.
        If the dataset has object-type coordinates, reindex is used for these
        coordinates instead of the interpolation.

        See Also
        --------
        Dataset.interp
        Dataset.reindex_like
        """
        pass

    def _rename(self, name_dict: Mapping[Any, Hashable] | None=None, **names: Hashable) -> Self:
        """Also used internally by DataArray so that the warning (if any)
        is raised at the right stack level.
        """
        pass

    def rename(self, name_dict: Mapping[Any, Hashable] | None=None, **names: Hashable) -> Self:
        """Returns a new object with renamed variables, coordinates and dimensions.

        Parameters
        ----------
        name_dict : dict-like, optional
            Dictionary whose keys are current variable, coordinate or dimension names and
            whose values are the desired names.
        **names : optional
            Keyword form of ``name_dict``.
            One of name_dict or names must be provided.

        Returns
        -------
        renamed : Dataset
            Dataset with renamed variables, coordinates and dimensions.

        See Also
        --------
        Dataset.swap_dims
        Dataset.rename_vars
        Dataset.rename_dims
        DataArray.rename
        """
        pass

    def rename_dims(self, dims_dict: Mapping[Any, Hashable] | None=None, **dims: Hashable) -> Self:
        """Returns a new object with renamed dimensions only.

        Parameters
        ----------
        dims_dict : dict-like, optional
            Dictionary whose keys are current dimension names and
            whose values are the desired names. The desired names must
            not be the name of an existing dimension or Variable in the Dataset.
        **dims : optional
            Keyword form of ``dims_dict``.
            One of dims_dict or dims must be provided.

        Returns
        -------
        renamed : Dataset
            Dataset with renamed dimensions.

        See Also
        --------
        Dataset.swap_dims
        Dataset.rename
        Dataset.rename_vars
        DataArray.rename
        """
        pass

    def rename_vars(self, name_dict: Mapping[Any, Hashable] | None=None, **names: Hashable) -> Self:
        """Returns a new object with renamed variables including coordinates

        Parameters
        ----------
        name_dict : dict-like, optional
            Dictionary whose keys are current variable or coordinate names and
            whose values are the desired names.
        **names : optional
            Keyword form of ``name_dict``.
            One of name_dict or names must be provided.

        Returns
        -------
        renamed : Dataset
            Dataset with renamed variables including coordinates

        See Also
        --------
        Dataset.swap_dims
        Dataset.rename
        Dataset.rename_dims
        DataArray.rename
        """
        pass

    def swap_dims(self, dims_dict: Mapping[Any, Hashable] | None=None, **dims_kwargs) -> Self:
        """Returns a new object with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names.
        **dims_kwargs : {existing_dim: new_dim, ...}, optional
            The keyword arguments form of ``dims_dict``.
            One of dims_dict or dims_kwargs must be provided.

        Returns
        -------
        swapped : Dataset
            Dataset with swapped dimensions.

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 7]), "b": ("x", [0.1, 2.4])},
        ...     coords={"x": ["a", "b"], "y": ("x", [0, 1])},
        ... )
        >>> ds
        <xarray.Dataset> Size: 56B
        Dimensions:  (x: 2)
        Coordinates:
          * x        (x) <U1 8B 'a' 'b'
            y        (x) int64 16B 0 1
        Data variables:
            a        (x) int64 16B 5 7
            b        (x) float64 16B 0.1 2.4

        >>> ds.swap_dims({"x": "y"})
        <xarray.Dataset> Size: 56B
        Dimensions:  (y: 2)
        Coordinates:
            x        (y) <U1 8B 'a' 'b'
          * y        (y) int64 16B 0 1
        Data variables:
            a        (y) int64 16B 5 7
            b        (y) float64 16B 0.1 2.4

        >>> ds.swap_dims({"x": "z"})
        <xarray.Dataset> Size: 56B
        Dimensions:  (z: 2)
        Coordinates:
            x        (z) <U1 8B 'a' 'b'
            y        (z) int64 16B 0 1
        Dimensions without coordinates: z
        Data variables:
            a        (z) int64 16B 5 7
            b        (z) float64 16B 0.1 2.4

        See Also
        --------
        Dataset.rename
        DataArray.swap_dims
        """
        pass

    def expand_dims(self, dim: None | Hashable | Sequence[Hashable] | Mapping[Any, Any]=None, axis: None | int | Sequence[int]=None, create_index_for_new_dim: bool=True, **dim_kwargs: Any) -> Self:
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape.  The new object is a
        view into the underlying array, not a copy.

        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        The automatic creation of indexes to back new 1D coordinate variables
        controlled by the create_index_for_new_dim kwarg.

        Parameters
        ----------
        dim : hashable, sequence of hashable, mapping, or None
            Dimensions to include on the new variable. If provided as hashable
            or sequence of hashable, then dimensions are inserted with length
            1. If provided as a mapping, then the keys are the new dimensions
            and the values are either integers (giving the length of the new
            dimensions) or array-like (giving the coordinates of the new
            dimensions).
        axis : int, sequence of int, or None, default: None
            Axis position(s) where new axis is to be inserted (position(s) on
            the result array). If a sequence of integers is passed,
            multiple axes are inserted. In this case, dim arguments should be
            same length list. If axis=None is passed, all the axes will be
            inserted to the start of the result array.
        create_index_for_new_dim : bool, default: True
            Whether to create new ``PandasIndex`` objects when the object being expanded contains scalar variables with names in ``dim``.
        **dim_kwargs : int or sequence or ndarray
            The keywords are arbitrary dimensions being inserted and the values
            are either the lengths of the new dims (if int is given), or their
            coordinates. Note, this is an alternative to passing a dict to the
            dim kwarg and will only be used if dim is None.

        Returns
        -------
        expanded : Dataset
            This object, but with additional dimension(s).

        Examples
        --------
        >>> dataset = xr.Dataset({"temperature": ([], 25.0)})
        >>> dataset
        <xarray.Dataset> Size: 8B
        Dimensions:      ()
        Data variables:
            temperature  float64 8B 25.0

        # Expand the dataset with a new dimension called "time"

        >>> dataset.expand_dims(dim="time")
        <xarray.Dataset> Size: 8B
        Dimensions:      (time: 1)
        Dimensions without coordinates: time
        Data variables:
            temperature  (time) float64 8B 25.0

        # 1D data

        >>> temperature_1d = xr.DataArray([25.0, 26.5, 24.8], dims="x")
        >>> dataset_1d = xr.Dataset({"temperature": temperature_1d})
        >>> dataset_1d
        <xarray.Dataset> Size: 24B
        Dimensions:      (x: 3)
        Dimensions without coordinates: x
        Data variables:
            temperature  (x) float64 24B 25.0 26.5 24.8

        # Expand the dataset with a new dimension called "time" using axis argument

        >>> dataset_1d.expand_dims(dim="time", axis=0)
        <xarray.Dataset> Size: 24B
        Dimensions:      (time: 1, x: 3)
        Dimensions without coordinates: time, x
        Data variables:
            temperature  (time, x) float64 24B 25.0 26.5 24.8

        # 2D data

        >>> temperature_2d = xr.DataArray(np.random.rand(3, 4), dims=("y", "x"))
        >>> dataset_2d = xr.Dataset({"temperature": temperature_2d})
        >>> dataset_2d
        <xarray.Dataset> Size: 96B
        Dimensions:      (y: 3, x: 4)
        Dimensions without coordinates: y, x
        Data variables:
            temperature  (y, x) float64 96B 0.5488 0.7152 0.6028 ... 0.7917 0.5289

        # Expand the dataset with a new dimension called "time" using axis argument

        >>> dataset_2d.expand_dims(dim="time", axis=2)
        <xarray.Dataset> Size: 96B
        Dimensions:      (y: 3, x: 4, time: 1)
        Dimensions without coordinates: y, x, time
        Data variables:
            temperature  (y, x, time) float64 96B 0.5488 0.7152 0.6028 ... 0.7917 0.5289

        # Expand a scalar variable along a new dimension of the same name with and without creating a new index

        >>> ds = xr.Dataset(coords={"x": 0})
        >>> ds
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Coordinates:
            x        int64 8B 0
        Data variables:
            *empty*

        >>> ds.expand_dims("x")
        <xarray.Dataset> Size: 8B
        Dimensions:  (x: 1)
        Coordinates:
          * x        (x) int64 8B 0
        Data variables:
            *empty*

        >>> ds.expand_dims("x").indexes
        Indexes:
            x        Index([0], dtype='int64', name='x')

        >>> ds.expand_dims("x", create_index_for_new_dim=False).indexes
        Indexes:
            *empty*

        See Also
        --------
        DataArray.expand_dims
        """
        pass

    def set_index(self, indexes: Mapping[Any, Hashable | Sequence[Hashable]] | None=None, append: bool=False, **indexes_kwargs: Hashable | Sequence[Hashable]) -> Self:
        """Set Dataset (multi-)indexes using one or more existing coordinates
        or variables.

        This legacy method is limited to pandas (multi-)indexes and
        1-dimensional "dimension" coordinates. See
        :py:meth:`~Dataset.set_xindex` for setting a pandas or a custom
        Xarray-compatible index from one or more arbitrary coordinates.

        Parameters
        ----------
        indexes : {dim: index, ...}
            Mapping from names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.
        append : bool, default: False
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es) (default).
        **indexes_kwargs : optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     data=np.ones((2, 3)),
        ...     dims=["x", "y"],
        ...     coords={"x": range(2), "y": range(3), "a": ("x", [3, 4])},
        ... )
        >>> ds = xr.Dataset({"v": arr})
        >>> ds
        <xarray.Dataset> Size: 104B
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 16B 0 1
          * y        (y) int64 24B 0 1 2
            a        (x) int64 16B 3 4
        Data variables:
            v        (x, y) float64 48B 1.0 1.0 1.0 1.0 1.0 1.0
        >>> ds.set_index(x="a")
        <xarray.Dataset> Size: 88B
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 16B 3 4
          * y        (y) int64 24B 0 1 2
        Data variables:
            v        (x, y) float64 48B 1.0 1.0 1.0 1.0 1.0 1.0

        See Also
        --------
        Dataset.reset_index
        Dataset.set_xindex
        Dataset.swap_dims
        """
        pass

    @_deprecate_positional_args('v2023.10.0')
    def reset_index(self, dims_or_levels: Hashable | Sequence[Hashable], *, drop: bool=False) -> Self:
        """Reset the specified index(es) or multi-index level(s).

        This legacy method is specific to pandas (multi-)indexes and
        1-dimensional "dimension" coordinates. See the more generic
        :py:meth:`~Dataset.drop_indexes` and :py:meth:`~Dataset.set_xindex`
        method to respectively drop and set pandas or custom indexes for
        arbitrary coordinates.

        Parameters
        ----------
        dims_or_levels : Hashable or Sequence of Hashable
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, default: False
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.set_index
        Dataset.set_xindex
        Dataset.drop_indexes
        """
        pass

    def set_xindex(self, coord_names: str | Sequence[Hashable], index_cls: type[Index] | None=None, **options) -> Self:
        """Set a new, Xarray-compatible index from one or more existing
        coordinate(s).

        Parameters
        ----------
        coord_names : str or list
            Name(s) of the coordinate(s) used to build the index.
            If several names are given, their order matters.
        index_cls : subclass of :class:`~xarray.indexes.Index`, optional
            The type of index to create. By default, try setting
            a ``PandasIndex`` if ``len(coord_names) == 1``,
            otherwise a ``PandasMultiIndex``.
        **options
            Options passed to the index constructor.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data and with a new index.

        """
        pass

    def reorder_levels(self, dim_order: Mapping[Any, Sequence[int | Hashable]] | None=None, **dim_order_kwargs: Sequence[int | Hashable]) -> Self:
        """Rearrange index levels using input order.

        Parameters
        ----------
        dim_order : dict-like of Hashable to Sequence of int or Hashable, optional
            Mapping from names matching dimensions and values given
            by lists representing new level orders. Every given dimension
            must have a multi-index.
        **dim_order_kwargs : Sequence of int or Hashable, optional
            The keyword arguments form of ``dim_order``.
            One of dim_order or dim_order_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced
            coordinates.
        """
        pass

    def _get_stack_index(self, dim, multi=False, create_index=False) -> tuple[Index | None, dict[Hashable, Variable]]:
        """Used by stack and unstack to get one pandas (multi-)index among
        the indexed coordinates along dimension `dim`.

        If exactly one index is found, return it with its corresponding
        coordinate variables(s), otherwise return None and an empty dict.

        If `create_index=True`, create a new index if none is found or raise
        an error if multiple indexes are found.

        """
        pass

    @partial(deprecate_dims, old_name='dimensions')
    def stack(self, dim: Mapping[Any, Sequence[Hashable | ellipsis]] | None=None, create_index: bool | None=True, index_cls: type[Index]=PandasMultiIndex, **dim_kwargs: Sequence[Hashable | ellipsis]) -> Self:
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and by default the corresponding
        coordinate variables will be combined into a MultiIndex.

        Parameters
        ----------
        dim : mapping of hashable to sequence of hashable
            Mapping of the form `new_name=(dim1, dim2, ...)`. Names of new
            dimensions, and the existing dimensions that they replace. An
            ellipsis (`...`) will be replaced by all unlisted dimensions.
            Passing a list containing an ellipsis (`stacked_dim=[...]`) will stack over
            all dimensions.
        create_index : bool or None, default: True

            - True: create a multi-index for each of the stacked dimensions.
            - False: don't create any index.
            - None. create a multi-index only if exactly one single (1-d) coordinate
              index is found for every dimension to stack.

        index_cls: Index-class, default: PandasMultiIndex
            Can be used to pass a custom multi-index type (must be an Xarray index that
            implements `.stack()`). By default, a pandas multi-index wrapper is used.
        **dim_kwargs
            The keyword arguments form of ``dim``.
            One of dim or dim_kwargs must be provided.

        Returns
        -------
        stacked : Dataset
            Dataset with stacked data.

        See Also
        --------
        Dataset.unstack
        """
        pass

    def to_stacked_array(self, new_dim: Hashable, sample_dims: Collection[Hashable], variable_dim: Hashable='variable', name: Hashable | None=None) -> DataArray:
        """Combine variables of differing dimensionality into a DataArray
        without broadcasting.

        This method is similar to Dataset.to_dataarray but does not broadcast the
        variables.

        Parameters
        ----------
        new_dim : hashable
            Name of the new stacked coordinate
        sample_dims : Collection of hashables
            List of dimensions that **will not** be stacked. Each array in the
            dataset must share these dimensions. For machine learning
            applications, these define the dimensions over which samples are
            drawn.
        variable_dim : hashable, default: "variable"
            Name of the level in the stacked coordinate which corresponds to
            the variables.
        name : hashable, optional
            Name of the new data array.

        Returns
        -------
        stacked : DataArray
            DataArray with the specified dimensions and data variables
            stacked together. The stacked coordinate is named ``new_dim``
            and represented by a MultiIndex object with a level containing the
            data variable names. The name of this level is controlled using
            the ``variable_dim`` argument.

        See Also
        --------
        Dataset.to_dataarray
        Dataset.stack
        DataArray.to_unstacked_dataset

        Examples
        --------
        >>> data = xr.Dataset(
        ...     data_vars={
        ...         "a": (("x", "y"), [[0, 1, 2], [3, 4, 5]]),
        ...         "b": ("x", [6, 7]),
        ...     },
        ...     coords={"y": ["u", "v", "w"]},
        ... )

        >>> data
        <xarray.Dataset> Size: 76B
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * y        (y) <U1 12B 'u' 'v' 'w'
        Dimensions without coordinates: x
        Data variables:
            a        (x, y) int64 48B 0 1 2 3 4 5
            b        (x) int64 16B 6 7

        >>> data.to_stacked_array("z", sample_dims=["x"])
        <xarray.DataArray 'a' (x: 2, z: 4)> Size: 64B
        array([[0, 1, 2, 6],
               [3, 4, 5, 7]])
        Coordinates:
          * z         (z) object 32B MultiIndex
          * variable  (z) <U1 16B 'a' 'a' 'a' 'b'
          * y         (z) object 32B 'u' 'v' 'w' nan
        Dimensions without coordinates: x

        """
        pass

    @_deprecate_positional_args('v2023.10.0')
    def unstack(self, dim: Dims=None, *, fill_value: Any=xrdtypes.NA, sparse: bool=False) -> Self:
        """
        Unstack existing dimensions corresponding to MultiIndexes into
        multiple new dimensions.

        New dimensions will be added at the end.

        Parameters
        ----------
        dim : str, Iterable of Hashable or None, optional
            Dimension(s) over which to unstack. By default unstacks all
            MultiIndexes.
        fill_value : scalar or dict-like, default: nan
            value to be filled. If a dict-like, maps variable names to
            fill values. If not provided or if the dict-like does not
            contain all variables, the dtype's NA value will be used.
        sparse : bool, default: False
            use sparse-array if True

        Returns
        -------
        unstacked : Dataset
            Dataset with unstacked data.

        See Also
        --------
        Dataset.stack
        """
        pass

    def update(self, other: CoercibleMapping) -> Self:
        """Update this dataset's variables with those from another dataset.

        Just like :py:meth:`dict.update` this is a in-place operation.
        For a non-inplace version, see :py:meth:`Dataset.merge`.

        Parameters
        ----------
        other : Dataset or mapping
            Variables with which to update this dataset. One of:

            - Dataset
            - mapping {var name: DataArray}
            - mapping {var name: Variable}
            - mapping {var name: (dimension name, array-like)}
            - mapping {var name: (tuple of dimension names, array-like)}

        Returns
        -------
        updated : Dataset
            Updated dataset. Note that since the update is in-place this is the input
            dataset.

            It is deprecated since version 0.17 and scheduled to be removed in 0.21.

        Raises
        ------
        ValueError
            If any dimensions would have inconsistent sizes in the updated
            dataset.

        See Also
        --------
        Dataset.assign
        Dataset.merge
        """
        pass

    def merge(self, other: CoercibleMapping | DataArray, overwrite_vars: Hashable | Iterable[Hashable]=frozenset(), compat: CompatOptions='no_conflicts', join: JoinOptions='outer', fill_value: Any=xrdtypes.NA, combine_attrs: CombineAttrsOptions='override') -> Self:
        """Merge the arrays of two datasets into a single dataset.

        This method generally does not allow for overriding data, with the
        exception of attributes, which are ignored on the second dataset.
        Variables with the same name are checked for conflicts via the equals
        or identical methods.

        Parameters
        ----------
        other : Dataset or mapping
            Dataset or variables to merge with this dataset.
        overwrite_vars : hashable or iterable of hashable, optional
            If provided, update variables of these name(s) without checking for
            conflicts in this dataset.
        compat : {"identical", "equals", "broadcast_equals",                   "no_conflicts", "override", "minimal"}, default: "no_conflicts"
            String indicating how to compare variables of the same name for
            potential conflicts:

            - 'identical': all values, dimensions and attributes must be the
              same.
            - 'equals': all values and dimensions must be the same.
            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'no_conflicts': only values which are not null in both datasets
              must be equal. The returned dataset then contains the combination
              of all non-null values.
            - 'override': skip comparing and pick variable from first dataset
            - 'minimal': drop conflicting coordinates

        join : {"outer", "inner", "left", "right", "exact", "override"},                default: "outer"
            Method for joining ``self`` and ``other`` along shared dimensions:

            - 'outer': use the union of the indexes
            - 'inner': use the intersection of the indexes
            - 'left': use indexes from ``self``
            - 'right': use indexes from ``other``
            - 'exact': error instead of aligning non-equal indexes
            - 'override': use indexes from ``self`` that are the same size
              as those of ``other`` in that dimension

        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like, maps
            variable names (including coordinates) to fill values.
        combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",                          "override"} or callable, default: "override"
            A callable or a string indicating how to combine attrs of the objects being
            merged:

            - "drop": empty attrs on returned Dataset.
            - "identical": all attrs must be the same on every object.
            - "no_conflicts": attrs from all objects are combined, any that have
              the same name must also have the same value.
            - "drop_conflicts": attrs from all objects are combined, any that have
              the same name but different values are dropped.
            - "override": skip comparing and copy attrs from the first dataset to
              the result.

            If a callable, it must expect a sequence of ``attrs`` dicts and a context object
            as its only parameters.

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        MergeError
            If any variables conflict (see ``compat``).

        See Also
        --------
        Dataset.update
        """
        pass

    def drop_vars(self, names: str | Iterable[Hashable] | Callable[[Self], str | Iterable[Hashable]], *, errors: ErrorOptions='raise') -> Self:
        """Drop variables from this dataset.

        Parameters
        ----------
        names : Hashable or iterable of Hashable or Callable
            Name(s) of variables to drop. If a Callable, this object is passed as its
            only argument and its result is used.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a ValueError error if any of the variable
            passed are not in the dataset. If 'ignore', any given names that are in the
            dataset are dropped and no error is raised.

        Examples
        --------

        >>> dataset = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             ["time", "latitude", "longitude"],
        ...             [[[25.5, 26.3], [27.1, 28.0]]],
        ...         ),
        ...         "humidity": (
        ...             ["time", "latitude", "longitude"],
        ...             [[[65.0, 63.8], [58.2, 59.6]]],
        ...         ),
        ...         "wind_speed": (
        ...             ["time", "latitude", "longitude"],
        ...             [[[10.2, 8.5], [12.1, 9.8]]],
        ...         ),
        ...     },
        ...     coords={
        ...         "time": pd.date_range("2023-07-01", periods=1),
        ...         "latitude": [40.0, 40.2],
        ...         "longitude": [-75.0, -74.8],
        ...     },
        ... )
        >>> dataset
        <xarray.Dataset> Size: 136B
        Dimensions:      (time: 1, latitude: 2, longitude: 2)
        Coordinates:
          * time         (time) datetime64[ns] 8B 2023-07-01
          * latitude     (latitude) float64 16B 40.0 40.2
          * longitude    (longitude) float64 16B -75.0 -74.8
        Data variables:
            temperature  (time, latitude, longitude) float64 32B 25.5 26.3 27.1 28.0
            humidity     (time, latitude, longitude) float64 32B 65.0 63.8 58.2 59.6
            wind_speed   (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Drop the 'humidity' variable

        >>> dataset.drop_vars(["humidity"])
        <xarray.Dataset> Size: 104B
        Dimensions:      (time: 1, latitude: 2, longitude: 2)
        Coordinates:
          * time         (time) datetime64[ns] 8B 2023-07-01
          * latitude     (latitude) float64 16B 40.0 40.2
          * longitude    (longitude) float64 16B -75.0 -74.8
        Data variables:
            temperature  (time, latitude, longitude) float64 32B 25.5 26.3 27.1 28.0
            wind_speed   (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Drop the 'humidity', 'temperature' variables

        >>> dataset.drop_vars(["humidity", "temperature"])
        <xarray.Dataset> Size: 72B
        Dimensions:     (time: 1, latitude: 2, longitude: 2)
        Coordinates:
          * time        (time) datetime64[ns] 8B 2023-07-01
          * latitude    (latitude) float64 16B 40.0 40.2
          * longitude   (longitude) float64 16B -75.0 -74.8
        Data variables:
            wind_speed  (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Drop all indexes

        >>> dataset.drop_vars(lambda x: x.indexes)
        <xarray.Dataset> Size: 96B
        Dimensions:      (time: 1, latitude: 2, longitude: 2)
        Dimensions without coordinates: time, latitude, longitude
        Data variables:
            temperature  (time, latitude, longitude) float64 32B 25.5 26.3 27.1 28.0
            humidity     (time, latitude, longitude) float64 32B 65.0 63.8 58.2 59.6
            wind_speed   (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Attempt to drop non-existent variable with errors="ignore"

        >>> dataset.drop_vars(["pressure"], errors="ignore")
        <xarray.Dataset> Size: 136B
        Dimensions:      (time: 1, latitude: 2, longitude: 2)
        Coordinates:
          * time         (time) datetime64[ns] 8B 2023-07-01
          * latitude     (latitude) float64 16B 40.0 40.2
          * longitude    (longitude) float64 16B -75.0 -74.8
        Data variables:
            temperature  (time, latitude, longitude) float64 32B 25.5 26.3 27.1 28.0
            humidity     (time, latitude, longitude) float64 32B 65.0 63.8 58.2 59.6
            wind_speed   (time, latitude, longitude) float64 32B 10.2 8.5 12.1 9.8

        Attempt to drop non-existent variable with errors="raise"

        >>> dataset.drop_vars(["pressure"], errors="raise")
        Traceback (most recent call last):
        ValueError: These variables cannot be found in this dataset: ['pressure']

        Raises
        ------
        ValueError
             Raised if you attempt to drop a variable which is not present, and the kwarg ``errors='raise'``.

        Returns
        -------
        dropped : Dataset

        See Also
        --------
        DataArray.drop_vars

        """
        pass

    def drop_indexes(self, coord_names: Hashable | Iterable[Hashable], *, errors: ErrorOptions='raise') -> Self:
        """Drop the indexes assigned to the given coordinates.

        Parameters
        ----------
        coord_names : hashable or iterable of hashable
            Name(s) of the coordinate(s) for which to drop the index.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a ValueError error if any of the coordinates
            passed have no index or are not in the dataset.
            If 'ignore', no error is raised.

        Returns
        -------
        dropped : Dataset
            A new dataset with dropped indexes.

        """
        pass

    def drop(self, labels=None, dim=None, *, errors: ErrorOptions='raise', **labels_kwargs) -> Self:
        """Backward compatible method based on `drop_vars` and `drop_sel`

        Using either `drop_vars` or `drop_sel` is encouraged

        See Also
        --------
        Dataset.drop_vars
        Dataset.drop_sel
        """
        pass

    def drop_sel(self, labels=None, *, errors: ErrorOptions='raise', **labels_kwargs) -> Self:
        """Drop index labels from this dataset.

        Parameters
        ----------
        labels : mapping of hashable to Any
            Index labels to drop
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a ValueError error if
            any of the index labels passed are not
            in the dataset. If 'ignore', any given labels that are in the
            dataset are dropped and no error is raised.
        **labels_kwargs : {dim: label, ...}, optional
            The keyword arguments form of ``dim`` and ``labels``

        Returns
        -------
        dropped : Dataset

        Examples
        --------
        >>> data = np.arange(6).reshape(2, 3)
        >>> labels = ["a", "b", "c"]
        >>> ds = xr.Dataset({"A": (["x", "y"], data), "y": labels})
        >>> ds
        <xarray.Dataset> Size: 60B
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * y        (y) <U1 12B 'a' 'b' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 48B 0 1 2 3 4 5
        >>> ds.drop_sel(y=["a", "c"])
        <xarray.Dataset> Size: 20B
        Dimensions:  (x: 2, y: 1)
        Coordinates:
          * y        (y) <U1 4B 'b'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 16B 1 4
        >>> ds.drop_sel(y="b")
        <xarray.Dataset> Size: 40B
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * y        (y) <U1 8B 'a' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 32B 0 2 3 5
        """
        pass

    def drop_isel(self, indexers=None, **indexers_kwargs) -> Self:
        """Drop index positions from this Dataset.

        Parameters
        ----------
        indexers : mapping of hashable to Any
            Index locations to drop
        **indexers_kwargs : {dim: position, ...}, optional
            The keyword arguments form of ``dim`` and ``positions``

        Returns
        -------
        dropped : Dataset

        Raises
        ------
        IndexError

        Examples
        --------
        >>> data = np.arange(6).reshape(2, 3)
        >>> labels = ["a", "b", "c"]
        >>> ds = xr.Dataset({"A": (["x", "y"], data), "y": labels})
        >>> ds
        <xarray.Dataset> Size: 60B
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * y        (y) <U1 12B 'a' 'b' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 48B 0 1 2 3 4 5
        >>> ds.drop_isel(y=[0, 2])
        <xarray.Dataset> Size: 20B
        Dimensions:  (x: 2, y: 1)
        Coordinates:
          * y        (y) <U1 4B 'b'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 16B 1 4
        >>> ds.drop_isel(y=1)
        <xarray.Dataset> Size: 40B
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * y        (y) <U1 8B 'a' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 32B 0 2 3 5
        """
        pass

    def drop_dims(self, drop_dims: str | Iterable[Hashable], *, errors: ErrorOptions='raise') -> Self:
        """Drop dimensions and associated variables from this dataset.

        Parameters
        ----------
        drop_dims : str or Iterable of Hashable
            Dimension or dimensions to drop.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a ValueError error if any of the
            dimensions passed are not in the dataset. If 'ignore', any given
            dimensions that are in the dataset are dropped and no error is raised.

        Returns
        -------
        obj : Dataset
            The dataset without the given dimensions (or any variables
            containing those dimensions).
        """
        pass

    @deprecate_dims
    def transpose(self, *dim: Hashable, missing_dims: ErrorOptionsWithWarn='raise') -> Self:
        """Return a new Dataset object with all array dimensions transposed.

        Although the order of dimensions on each array will change, the dataset
        dimensions themselves will remain in fixed (sorted) order.

        Parameters
        ----------
        *dim : hashable, optional
            By default, reverse the dimensions on each array. Otherwise,
            reorder the dimensions to this order.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        Returns
        -------
        transposed : Dataset
            Each array in the dataset (including) coordinates will be
            transposed to the given order.

        Notes
        -----
        This operation returns a view of each array's data. It is
        lazy for dask-backed DataArrays but not for numpy-backed DataArrays
        -- the data will be fully loaded into memory.

        See Also
        --------
        numpy.transpose
        DataArray.transpose
        """
        pass

    @_deprecate_positional_args('v2023.10.0')
    def dropna(self, dim: Hashable, *, how: Literal['any', 'all']='any', thresh: int | None=None, subset: Iterable[Hashable] | None=None) -> Self:
        """Returns a new dataset with dropped labels for missing values along
        the provided dimension.

        Parameters
        ----------
        dim : hashable
            Dimension along which to drop missing values. Dropping along
            multiple dimensions simultaneously is not yet supported.
        how : {"any", "all"}, default: "any"
            - any : if any NA values are present, drop that label
            - all : if all values are NA, drop that label

        thresh : int or None, optional
            If supplied, require this many non-NA values (summed over all the subset variables).
        subset : iterable of hashable or None, optional
            Which variables to check for missing values. By default, all
            variables in the dataset are checked.

        Examples
        --------
        >>> dataset = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             ["time", "location"],
        ...             [[23.4, 24.1], [np.nan, 22.1], [21.8, 24.2], [20.5, 25.3]],
        ...         )
        ...     },
        ...     coords={"time": [1, 2, 3, 4], "location": ["A", "B"]},
        ... )
        >>> dataset
        <xarray.Dataset> Size: 104B
        Dimensions:      (time: 4, location: 2)
        Coordinates:
          * time         (time) int64 32B 1 2 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 64B 23.4 24.1 nan ... 24.2 20.5 25.3

        Drop NaN values from the dataset

        >>> dataset.dropna(dim="time")
        <xarray.Dataset> Size: 80B
        Dimensions:      (time: 3, location: 2)
        Coordinates:
          * time         (time) int64 24B 1 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 48B 23.4 24.1 21.8 24.2 20.5 25.3

        Drop labels with any NaN values

        >>> dataset.dropna(dim="time", how="any")
        <xarray.Dataset> Size: 80B
        Dimensions:      (time: 3, location: 2)
        Coordinates:
          * time         (time) int64 24B 1 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 48B 23.4 24.1 21.8 24.2 20.5 25.3

        Drop labels with all NAN values

        >>> dataset.dropna(dim="time", how="all")
        <xarray.Dataset> Size: 104B
        Dimensions:      (time: 4, location: 2)
        Coordinates:
          * time         (time) int64 32B 1 2 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 64B 23.4 24.1 nan ... 24.2 20.5 25.3

        Drop labels with less than 2 non-NA values

        >>> dataset.dropna(dim="time", thresh=2)
        <xarray.Dataset> Size: 80B
        Dimensions:      (time: 3, location: 2)
        Coordinates:
          * time         (time) int64 24B 1 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 48B 23.4 24.1 21.8 24.2 20.5 25.3

        Returns
        -------
        Dataset

        See Also
        --------
        DataArray.dropna
        pandas.DataFrame.dropna
        """
        pass

    def fillna(self, value: Any) -> Self:
        """Fill missing values in this object.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value : scalar, ndarray, DataArray, dict or Dataset
            Used to fill all matching missing values in this dataset's data
            variables. Scalars, ndarrays or DataArrays arguments are used to
            fill all data with aligned coordinates (for DataArrays).
            Dictionaries or datasets match data variables and then align
            coordinates if necessary.

        Returns
        -------
        Dataset

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {
        ...         "A": ("x", [np.nan, 2, np.nan, 0]),
        ...         "B": ("x", [3, 4, np.nan, 1]),
        ...         "C": ("x", [np.nan, np.nan, np.nan, 5]),
        ...         "D": ("x", [np.nan, 3, np.nan, 4]),
        ...     },
        ...     coords={"x": [0, 1, 2, 3]},
        ... )
        >>> ds
        <xarray.Dataset> Size: 160B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
        Data variables:
            A        (x) float64 32B nan 2.0 nan 0.0
            B        (x) float64 32B 3.0 4.0 nan 1.0
            C        (x) float64 32B nan nan nan 5.0
            D        (x) float64 32B nan 3.0 nan 4.0

        Replace all `NaN` values with 0s.

        >>> ds.fillna(0)
        <xarray.Dataset> Size: 160B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
        Data variables:
            A        (x) float64 32B 0.0 2.0 0.0 0.0
            B        (x) float64 32B 3.0 4.0 0.0 1.0
            C        (x) float64 32B 0.0 0.0 0.0 5.0
            D        (x) float64 32B 0.0 3.0 0.0 4.0

        Replace all `NaN` elements in column ‘A’, ‘B’, ‘C’, and ‘D’, with 0, 1, 2, and 3 respectively.

        >>> values = {"A": 0, "B": 1, "C": 2, "D": 3}
        >>> ds.fillna(value=values)
        <xarray.Dataset> Size: 160B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
        Data variables:
            A        (x) float64 32B 0.0 2.0 0.0 0.0
            B        (x) float64 32B 3.0 4.0 1.0 1.0
            C        (x) float64 32B 2.0 2.0 2.0 5.0
            D        (x) float64 32B 3.0 3.0 3.0 4.0
        """
        pass

    def interpolate_na(self, dim: Hashable | None=None, method: InterpOptions='linear', limit: int | None=None, use_coordinate: bool | Hashable=True, max_gap: int | float | str | pd.Timedelta | np.timedelta64 | datetime.timedelta | None=None, **kwargs: Any) -> Self:
        """Fill in NaNs by interpolating according to different methods.

        Parameters
        ----------
        dim : Hashable or None, optional
            Specifies the dimension along which to interpolate.
        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial",             "barycentric", "krogh", "pchip", "spline", "akima"}, default: "linear"
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation. Additional keyword
              arguments are passed to :py:func:`numpy.interp`
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial':
              are passed to :py:func:`scipy.interpolate.interp1d`. If
              ``method='polynomial'``, the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krogh', 'pchip', 'spline', 'akima': use their
              respective :py:class:`scipy.interpolate` classes.

        use_coordinate : bool or Hashable, default: True
            Specifies which index to use as the x values in the interpolation
            formulated as `y = f(x)`. If False, values are treated as if
            equally-spaced along ``dim``. If True, the IndexVariable `dim` is
            used. If ``use_coordinate`` is a string, it specifies the name of a
            coordinate variable to use as the index.
        limit : int, default: None
            Maximum number of consecutive NaNs to fill. Must be greater than 0
            or None for no limit. This filling is done regardless of the size of
            the gap in the data. To only interpolate over gaps less than a given length,
            see ``max_gap``.
        max_gap : int, float, str, pandas.Timedelta, numpy.timedelta64, datetime.timedelta             or None, default: None
            Maximum size of gap, a continuous sequence of NaNs, that will be filled.
            Use None for no limit. When interpolating along a datetime64 dimension
            and ``use_coordinate=True``, ``max_gap`` can be one of the following:

            - a string that is valid input for pandas.to_timedelta
            - a :py:class:`numpy.timedelta64` object
            - a :py:class:`pandas.Timedelta` object
            - a :py:class:`datetime.timedelta` object

            Otherwise, ``max_gap`` must be an int or a float. Use of ``max_gap`` with unlabeled
            dimensions has not been implemented yet. Gap length is defined as the difference
            between coordinate values at the first data point after a gap and the last value
            before a gap. For gaps at the beginning (end), gap length is defined as the difference
            between coordinate values at the first (last) valid data point and the first (last) NaN.
            For example, consider::

                <xarray.DataArray (x: 9)>
                array([nan, nan, nan,  1., nan, nan,  4., nan, nan])
                Coordinates:
                  * x        (x) int64 0 1 2 3 4 5 6 7 8

            The gap lengths are 3-0 = 3; 6-3 = 3; and 8-6 = 2 respectively
        **kwargs : dict, optional
            parameters passed verbatim to the underlying interpolation function

        Returns
        -------
        interpolated: Dataset
            Filled in Dataset.

        Warning
        --------
        When passing fill_value as a keyword argument with method="linear", it does not use
        ``numpy.interp`` but it uses ``scipy.interpolate.interp1d``, which provides the fill_value parameter.

        See Also
        --------
        numpy.interp
        scipy.interpolate

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {
        ...         "A": ("x", [np.nan, 2, 3, np.nan, 0]),
        ...         "B": ("x", [3, 4, np.nan, 1, 7]),
        ...         "C": ("x", [np.nan, np.nan, np.nan, 5, 0]),
        ...         "D": ("x", [np.nan, 3, np.nan, -1, 4]),
        ...     },
        ...     coords={"x": [0, 1, 2, 3, 4]},
        ... )
        >>> ds
        <xarray.Dataset> Size: 200B
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 40B 0 1 2 3 4
        Data variables:
            A        (x) float64 40B nan 2.0 3.0 nan 0.0
            B        (x) float64 40B 3.0 4.0 nan 1.0 7.0
            C        (x) float64 40B nan nan nan 5.0 0.0
            D        (x) float64 40B nan 3.0 nan -1.0 4.0

        >>> ds.interpolate_na(dim="x", method="linear")
        <xarray.Dataset> Size: 200B
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 40B 0 1 2 3 4
        Data variables:
            A        (x) float64 40B nan 2.0 3.0 1.5 0.0
            B        (x) float64 40B 3.0 4.0 2.5 1.0 7.0
            C        (x) float64 40B nan nan nan 5.0 0.0
            D        (x) float64 40B nan 3.0 1.0 -1.0 4.0

        >>> ds.interpolate_na(dim="x", method="linear", fill_value="extrapolate")
        <xarray.Dataset> Size: 200B
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 40B 0 1 2 3 4
        Data variables:
            A        (x) float64 40B 1.0 2.0 3.0 1.5 0.0
            B        (x) float64 40B 3.0 4.0 2.5 1.0 7.0
            C        (x) float64 40B 20.0 15.0 10.0 5.0 0.0
            D        (x) float64 40B 5.0 3.0 1.0 -1.0 4.0
        """
        pass

    def ffill(self, dim: Hashable, limit: int | None=None) -> Self:
        """Fill NaN values by propagating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : Hashable
            Specifies the dimension along which to propagate values when filling.
        limit : int or None, optional
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit. Must be None or greater than or equal
            to axis length if filling along chunked axes (dimensions).

        Examples
        --------
        >>> time = pd.date_range("2023-01-01", periods=10, freq="D")
        >>> data = np.array(
        ...     [1, np.nan, np.nan, np.nan, 5, np.nan, np.nan, 8, np.nan, 10]
        ... )
        >>> dataset = xr.Dataset({"data": (("time",), data)}, coords={"time": time})
        >>> dataset
        <xarray.Dataset> Size: 160B
        Dimensions:  (time: 10)
        Coordinates:
          * time     (time) datetime64[ns] 80B 2023-01-01 2023-01-02 ... 2023-01-10
        Data variables:
            data     (time) float64 80B 1.0 nan nan nan 5.0 nan nan 8.0 nan 10.0

        # Perform forward fill (ffill) on the dataset

        >>> dataset.ffill(dim="time")
        <xarray.Dataset> Size: 160B
        Dimensions:  (time: 10)
        Coordinates:
          * time     (time) datetime64[ns] 80B 2023-01-01 2023-01-02 ... 2023-01-10
        Data variables:
            data     (time) float64 80B 1.0 1.0 1.0 1.0 5.0 5.0 5.0 8.0 8.0 10.0

        # Limit the forward filling to a maximum of 2 consecutive NaN values

        >>> dataset.ffill(dim="time", limit=2)
        <xarray.Dataset> Size: 160B
        Dimensions:  (time: 10)
        Coordinates:
          * time     (time) datetime64[ns] 80B 2023-01-01 2023-01-02 ... 2023-01-10
        Data variables:
            data     (time) float64 80B 1.0 1.0 1.0 nan 5.0 5.0 5.0 8.0 8.0 10.0

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.bfill
        """
        pass

    def bfill(self, dim: Hashable, limit: int | None=None) -> Self:
        """Fill NaN values by propagating values backward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : Hashable
            Specifies the dimension along which to propagate values when
            filling.
        limit : int or None, optional
            The maximum number of consecutive NaN values to backward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit. Must be None or greater than or equal
            to axis length if filling along chunked axes (dimensions).

        Examples
        --------
        >>> time = pd.date_range("2023-01-01", periods=10, freq="D")
        >>> data = np.array(
        ...     [1, np.nan, np.nan, np.nan, 5, np.nan, np.nan, 8, np.nan, 10]
        ... )
        >>> dataset = xr.Dataset({"data": (("time",), data)}, coords={"time": time})
        >>> dataset
        <xarray.Dataset> Size: 160B
        Dimensions:  (time: 10)
        Coordinates:
          * time     (time) datetime64[ns] 80B 2023-01-01 2023-01-02 ... 2023-01-10
        Data variables:
            data     (time) float64 80B 1.0 nan nan nan 5.0 nan nan 8.0 nan 10.0

        # filled dataset, fills NaN values by propagating values backward

        >>> dataset.bfill(dim="time")
        <xarray.Dataset> Size: 160B
        Dimensions:  (time: 10)
        Coordinates:
          * time     (time) datetime64[ns] 80B 2023-01-01 2023-01-02 ... 2023-01-10
        Data variables:
            data     (time) float64 80B 1.0 5.0 5.0 5.0 5.0 8.0 8.0 8.0 10.0 10.0

        # Limit the backward filling to a maximum of 2 consecutive NaN values

        >>> dataset.bfill(dim="time", limit=2)
        <xarray.Dataset> Size: 160B
        Dimensions:  (time: 10)
        Coordinates:
          * time     (time) datetime64[ns] 80B 2023-01-01 2023-01-02 ... 2023-01-10
        Data variables:
            data     (time) float64 80B 1.0 nan 5.0 5.0 5.0 8.0 8.0 8.0 10.0 10.0

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.ffill
        """
        pass

    def combine_first(self, other: Self) -> Self:
        """Combine two Datasets, default to data_vars of self.

        The new coordinates follow the normal broadcasting and alignment rules
        of ``join='outer'``.  Vacant cells in the expanded coordinates are
        filled with np.nan.

        Parameters
        ----------
        other : Dataset
            Used to fill all matching missing values in this array.

        Returns
        -------
        Dataset
        """
        pass

    def reduce(self, func: Callable, dim: Dims=None, *, keep_attrs: bool | None=None, keepdims: bool=False, numeric_only: bool=False, **kwargs: Any) -> Self:
        """Reduce this dataset by applying `func` along some dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. By default `func` is
            applied over all dimensions.
        keep_attrs : bool or None, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        keepdims : bool, default: False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one. Coordinates that use these dimensions
            are removed.
        numeric_only : bool, default: False
            If True, only apply ``func`` to variables with a numeric dtype.
        **kwargs : Any
            Additional keyword arguments passed on to ``func``.

        Returns
        -------
        reduced : Dataset
            Dataset with this object's DataArrays replaced with new DataArrays
            of summarized data and the indicated dimension(s) removed.

        Examples
        --------

        >>> dataset = xr.Dataset(
        ...     {
        ...         "math_scores": (
        ...             ["student", "test"],
        ...             [[90, 85, 92], [78, 80, 85], [95, 92, 98]],
        ...         ),
        ...         "english_scores": (
        ...             ["student", "test"],
        ...             [[88, 90, 92], [75, 82, 79], [93, 96, 91]],
        ...         ),
        ...     },
        ...     coords={
        ...         "student": ["Alice", "Bob", "Charlie"],
        ...         "test": ["Test 1", "Test 2", "Test 3"],
        ...     },
        ... )

        # Calculate the 75th percentile of math scores for each student using np.percentile

        >>> percentile_scores = dataset.reduce(np.percentile, q=75, dim="test")
        >>> percentile_scores
        <xarray.Dataset> Size: 132B
        Dimensions:         (student: 3)
        Coordinates:
          * student         (student) <U7 84B 'Alice' 'Bob' 'Charlie'
        Data variables:
            math_scores     (student) float64 24B 91.0 82.5 96.5
            english_scores  (student) float64 24B 91.0 80.5 94.5
        """
        pass

    def map(self, func: Callable, keep_attrs: bool | None=None, args: Iterable[Any]=(), **kwargs: Any) -> Self:
        """Apply a function to each data variable in this dataset

        Parameters
        ----------
        func : callable
            Function which can be called in the form `func(x, *args, **kwargs)`
            to transform each DataArray `x` in this dataset into another
            DataArray.
        keep_attrs : bool or None, optional
            If True, both the dataset's and variables' attributes (`attrs`) will be
            copied from the original objects to the new ones. If False, the new dataset
            and variables will be returned without copying the attributes.
        args : iterable, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.

        Returns
        -------
        applied : Dataset
            Resulting dataset from applying ``func`` to each data variable.

        Examples
        --------
        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset({"foo": da, "bar": ("x", [-1, 2])})
        >>> ds
        <xarray.Dataset> Size: 64B
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Dimensions without coordinates: dim_0, dim_1, x
        Data variables:
            foo      (dim_0, dim_1) float64 48B 1.764 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 16B -1 2
        >>> ds.map(np.fabs)
        <xarray.Dataset> Size: 64B
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Dimensions without coordinates: dim_0, dim_1, x
        Data variables:
            foo      (dim_0, dim_1) float64 48B 1.764 0.4002 0.9787 2.241 1.868 0.9773
            bar      (x) float64 16B 1.0 2.0
        """
        pass

    def apply(self, func: Callable, keep_attrs: bool | None=None, args: Iterable[Any]=(), **kwargs: Any) -> Self:
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        Dataset.map
        """
        pass

    def assign(self, variables: Mapping[Any, Any] | None=None, **variables_kwargs: Any) -> Self:
        """Assign new data variables to a Dataset, returning a new object
        with all the original variables in addition to the new ones.

        Parameters
        ----------
        variables : mapping of hashable to Any
            Mapping from variables names to the new values. If the new values
            are callable, they are computed on the Dataset and assigned to new
            data variables. If the values are not callable, (e.g. a DataArray,
            scalar, or array), they are simply assigned.
        **variables_kwargs
            The keyword arguments form of ``variables``.
            One of variables or variables_kwargs must be provided.

        Returns
        -------
        ds : Dataset
            A new Dataset with the new variables in addition to all the
            existing variables.

        Notes
        -----
        Since ``kwargs`` is a dictionary, the order of your arguments may not
        be preserved, and so the order of the new variables is not well
        defined. Assigning multiple variables within the same ``assign`` is
        possible, but you cannot reference other variables created within the
        same ``assign`` call.

        The new assigned variables that replace existing coordinates in the
        original dataset are still listed as coordinates in the returned
        Dataset.

        See Also
        --------
        pandas.DataFrame.assign

        Examples
        --------
        >>> x = xr.Dataset(
        ...     {
        ...         "temperature_c": (
        ...             ("lat", "lon"),
        ...             20 * np.random.rand(4).reshape(2, 2),
        ...         ),
        ...         "precipitation": (("lat", "lon"), np.random.rand(4).reshape(2, 2)),
        ...     },
        ...     coords={"lat": [10, 20], "lon": [150, 160]},
        ... )
        >>> x
        <xarray.Dataset> Size: 96B
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 16B 10 20
          * lon            (lon) int64 16B 150 160
        Data variables:
            temperature_c  (lat, lon) float64 32B 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 32B 0.4237 0.6459 0.4376 0.8918

        Where the value is a callable, evaluated on dataset:

        >>> x.assign(temperature_f=lambda x: x.temperature_c * 9 / 5 + 32)
        <xarray.Dataset> Size: 128B
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 16B 10 20
          * lon            (lon) int64 16B 150 160
        Data variables:
            temperature_c  (lat, lon) float64 32B 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 32B 0.4237 0.6459 0.4376 0.8918
            temperature_f  (lat, lon) float64 32B 51.76 57.75 53.7 51.62

        Alternatively, the same behavior can be achieved by directly referencing an existing dataarray:

        >>> x.assign(temperature_f=x["temperature_c"] * 9 / 5 + 32)
        <xarray.Dataset> Size: 128B
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 16B 10 20
          * lon            (lon) int64 16B 150 160
        Data variables:
            temperature_c  (lat, lon) float64 32B 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 32B 0.4237 0.6459 0.4376 0.8918
            temperature_f  (lat, lon) float64 32B 51.76 57.75 53.7 51.62

        """
        pass

    def to_dataarray(self, dim: Hashable='variable', name: Hashable | None=None) -> DataArray:
        """Convert this dataset into an xarray.DataArray

        The data variables of this dataset will be broadcast against each other
        and stacked along the first axis of the new array. All coordinates of
        this dataset will remain coordinates.

        Parameters
        ----------
        dim : Hashable, default: "variable"
            Name of the new dimension.
        name : Hashable or None, optional
            Name of the new data array.

        Returns
        -------
        array : xarray.DataArray
        """
        pass

    def to_array(self, dim: Hashable='variable', name: Hashable | None=None) -> DataArray:
        """Deprecated version of to_dataarray"""
        pass

    def _normalize_dim_order(self, dim_order: Sequence[Hashable] | None=None) -> dict[Hashable, int]:
        """
        Check the validity of the provided dimensions if any and return the mapping
        between dimension name and their size.

        Parameters
        ----------
        dim_order: Sequence of Hashable or None, optional
            Dimension order to validate (default to the alphabetical order if None).

        Returns
        -------
        result : dict[Hashable, int]
            Validated dimensions mapping.

        """
        pass

    def to_pandas(self) -> pd.Series | pd.DataFrame:
        """Convert this dataset into a pandas object without changing the number of dimensions.

        The type of the returned object depends on the number of Dataset
        dimensions:

        * 0D -> `pandas.Series`
        * 1D -> `pandas.DataFrame`

        Only works for Datasets with 1 or fewer dimensions.
        """
        pass

    def to_dataframe(self, dim_order: Sequence[Hashable] | None=None) -> pd.DataFrame:
        """Convert this dataset into a pandas.DataFrame.

        Non-index variables in this dataset form the columns of the
        DataFrame. The DataFrame is indexed by the Cartesian product of
        this dataset's indices.

        Parameters
        ----------
        dim_order: Sequence of Hashable or None, optional
            Hierarchical dimension order for the resulting dataframe. All
            arrays are transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting
            dataframe.

            If provided, must include all dimensions of this dataset. By
            default, dimensions are sorted alphabetically.

        Returns
        -------
        result : DataFrame
            Dataset as a pandas DataFrame.

        """
        pass

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, sparse: bool=False) -> Self:
        """Convert a pandas.DataFrame into an xarray.Dataset

        Each column will be converted into an independent variable in the
        Dataset. If the dataframe's index is a MultiIndex, it will be expanded
        into a tensor product of one-dimensional indices (filling in missing
        values with NaN). If you rather preserve the MultiIndex use
        `xr.Dataset(df)`. This method will produce a Dataset very similar to
        that on which the 'to_dataframe' method was called, except with
        possibly redundant dimensions (since all dataset variables will have
        the same dimensionality).

        Parameters
        ----------
        dataframe : DataFrame
            DataFrame from which to copy data and indices.
        sparse : bool, default: False
            If true, create a sparse arrays instead of dense numpy arrays. This
            can potentially save a large amount of memory if the DataFrame has
            a MultiIndex. Requires the sparse package (sparse.pydata.org).

        Returns
        -------
        New Dataset.

        See Also
        --------
        xarray.DataArray.from_series
        pandas.DataFrame.to_xarray
        """
        pass

    def to_dask_dataframe(self, dim_order: Sequence[Hashable] | None=None, set_index: bool=False) -> DaskDataFrame:
        """
        Convert this dataset into a dask.dataframe.DataFrame.

        The dimensions, coordinates and data variables in this dataset form
        the columns of the DataFrame.

        Parameters
        ----------
        dim_order : list, optional
            Hierarchical dimension order for the resulting dataframe. All
            arrays are transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting dask
            dataframe.

            If provided, must include all dimensions of this dataset. By
            default, dimensions are sorted alphabetically.
        set_index : bool, default: False
            If set_index=True, the dask DataFrame is indexed by this dataset's
            coordinate. Since dask DataFrames do not support multi-indexes,
            set_index only works if the dataset only contains one dimension.

        Returns
        -------
        dask.dataframe.DataFrame
        """
        pass

    def to_dict(self, data: bool | Literal['list', 'array']='list', encoding: bool=False) -> dict[str, Any]:
        """
        Convert this dataset to a dictionary following xarray naming
        conventions.

        Converts all variables and attributes to native Python objects
        Useful for converting to json. To avoid datetime incompatibility
        use decode_times=False kwarg in xarrray.open_dataset.

        Parameters
        ----------
        data : bool or {"list", "array"}, default: "list"
            Whether to include the actual data in the dictionary. When set to
            False, returns just the schema. If set to "array", returns data as
            underlying array type. If set to "list" (or True for backwards
            compatibility), returns data in lists of Python data types. Note
            that for obtaining the "list" output efficiently, use
            `ds.compute().to_dict(data="list")`.

        encoding : bool, default: False
            Whether to include the Dataset's encoding in the dictionary.

        Returns
        -------
        d : dict
            Dict with keys: "coords", "attrs", "dims", "data_vars" and optionally
            "encoding".

        See Also
        --------
        Dataset.from_dict
        DataArray.to_dict
        """
        pass

    @classmethod
    def from_dict(cls, d: Mapping[Any, Any]) -> Self:
        """Convert a dictionary into an xarray.Dataset.

        Parameters
        ----------
        d : dict-like
            Mapping with a minimum structure of
                ``{"var_0": {"dims": [..], "data": [..]},                             ...}``

        Returns
        -------
        obj : Dataset

        See also
        --------
        Dataset.to_dict
        DataArray.from_dict

        Examples
        --------
        >>> d = {
        ...     "t": {"dims": ("t"), "data": [0, 1, 2]},
        ...     "a": {"dims": ("t"), "data": ["a", "b", "c"]},
        ...     "b": {"dims": ("t"), "data": [10, 20, 30]},
        ... }
        >>> ds = xr.Dataset.from_dict(d)
        >>> ds
        <xarray.Dataset> Size: 60B
        Dimensions:  (t: 3)
        Coordinates:
          * t        (t) int64 24B 0 1 2
        Data variables:
            a        (t) <U1 12B 'a' 'b' 'c'
            b        (t) int64 24B 10 20 30

        >>> d = {
        ...     "coords": {
        ...         "t": {"dims": "t", "data": [0, 1, 2], "attrs": {"units": "s"}}
        ...     },
        ...     "attrs": {"title": "air temperature"},
        ...     "dims": "t",
        ...     "data_vars": {
        ...         "a": {"dims": "t", "data": [10, 20, 30]},
        ...         "b": {"dims": "t", "data": ["a", "b", "c"]},
        ...     },
        ... }
        >>> ds = xr.Dataset.from_dict(d)
        >>> ds
        <xarray.Dataset> Size: 60B
        Dimensions:  (t: 3)
        Coordinates:
          * t        (t) int64 24B 0 1 2
        Data variables:
            a        (t) int64 24B 10 20 30
            b        (t) <U1 12B 'a' 'b' 'c'
        Attributes:
            title:    air temperature

        """
        pass

    @_deprecate_positional_args('v2023.10.0')
    def diff(self, dim: Hashable, n: int=1, *, label: Literal['upper', 'lower']='upper') -> Self:
        """Calculate the n-th order discrete difference along given axis.

        Parameters
        ----------
        dim : Hashable
            Dimension over which to calculate the finite difference.
        n : int, default: 1
            The number of times values are differenced.
        label : {"upper", "lower"}, default: "upper"
            The new coordinate in dimension ``dim`` will have the
            values of either the minuend's or subtrahend's coordinate
            for values 'upper' and 'lower', respectively.

        Returns
        -------
        difference : Dataset
            The n-th order finite difference of this object.

        Notes
        -----
        `n` matches numpy's behavior and is different from pandas' first argument named
        `periods`.

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", [5, 5, 6, 6])})
        >>> ds.diff("x")
        <xarray.Dataset> Size: 24B
        Dimensions:  (x: 3)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) int64 24B 0 1 0
        >>> ds.diff("x", 2)
        <xarray.Dataset> Size: 16B
        Dimensions:  (x: 2)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) int64 16B 1 -1

        See Also
        --------
        Dataset.differentiate
        """
        pass

    def shift(self, shifts: Mapping[Any, int] | None=None, fill_value: Any=xrdtypes.NA, **shifts_kwargs: int) -> Self:
        """Shift this dataset by an offset along one or more dimensions.

        Only data variables are moved; coordinates stay in place. This is
        consistent with the behavior of ``shift`` in pandas.

        Values shifted from beyond array bounds will appear at one end of
        each dimension, which are filled according to `fill_value`. For periodic
        offsets instead see `roll`.

        Parameters
        ----------
        shifts : mapping of hashable to int
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like, maps
            variable names (including coordinates) to fill values.
        **shifts_kwargs
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        shifted : Dataset
            Dataset with the same coordinates and attributes but shifted data
            variables.

        See Also
        --------
        roll

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", list("abcde"))})
        >>> ds.shift(x=2)
        <xarray.Dataset> Size: 40B
        Dimensions:  (x: 5)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) object 40B nan nan 'a' 'b' 'c'
        """
        pass

    def roll(self, shifts: Mapping[Any, int] | None=None, roll_coords: bool=False, **shifts_kwargs: int) -> Self:
        """Roll this dataset by an offset along one or more dimensions.

        Unlike shift, roll treats the given dimensions as periodic, so will not
        create any missing values to be filled.

        Also unlike shift, roll may rotate all variables, including coordinates
        if specified. The direction of rotation is consistent with
        :py:func:`numpy.roll`.

        Parameters
        ----------
        shifts : mapping of hashable to int, optional
            A dict with keys matching dimensions and values given
            by integers to rotate each of the given dimensions. Positive
            offsets roll to the right; negative offsets roll to the left.
        roll_coords : bool, default: False
            Indicates whether to roll the coordinates by the offset too.
        **shifts_kwargs : {dim: offset, ...}, optional
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        rolled : Dataset
            Dataset with the same attributes but rolled data and coordinates.

        See Also
        --------
        shift

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", list("abcde"))}, coords={"x": np.arange(5)})
        >>> ds.roll(x=2)
        <xarray.Dataset> Size: 60B
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 40B 0 1 2 3 4
        Data variables:
            foo      (x) <U1 20B 'd' 'e' 'a' 'b' 'c'

        >>> ds.roll(x=2, roll_coords=True)
        <xarray.Dataset> Size: 60B
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 40B 3 4 0 1 2
        Data variables:
            foo      (x) <U1 20B 'd' 'e' 'a' 'b' 'c'

        """
        pass

    def sortby(self, variables: Hashable | DataArray | Sequence[Hashable | DataArray] | Callable[[Self], Hashable | DataArray | list[Hashable | DataArray]], ascending: bool=True) -> Self:
        """
        Sort object by labels or values (along an axis).

        Sorts the dataset, either along specified dimensions,
        or according to values of 1-D dataarrays that share dimension
        with calling object.

        If the input variables are dataarrays, then the dataarrays are aligned
        (via left-join) to the calling object prior to sorting by cell values.
        NaNs are sorted to the end, following Numpy convention.

        If multiple sorts along the same dimension is
        given, numpy's lexsort is performed along that dimension:
        https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html
        and the FIRST key in the sequence is used as the primary sort key,
        followed by the 2nd key, etc.

        Parameters
        ----------
        variables : Hashable, DataArray, sequence of Hashable or DataArray, or Callable
            1D DataArray objects or name(s) of 1D variable(s) in coords whose values are
            used to sort this array. If a callable, the callable is passed this object,
            and the result is used as the value for cond.
        ascending : bool, default: True
            Whether to sort by ascending or descending order.

        Returns
        -------
        sorted : Dataset
            A new dataset where all the specified dims are sorted by dim
            labels.

        See Also
        --------
        DataArray.sortby
        numpy.sort
        pandas.sort_values
        pandas.sort_index

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {
        ...         "A": (("x", "y"), [[1, 2], [3, 4]]),
        ...         "B": (("x", "y"), [[5, 6], [7, 8]]),
        ...     },
        ...     coords={"x": ["b", "a"], "y": [1, 0]},
        ... )
        >>> ds.sortby("x")
        <xarray.Dataset> Size: 88B
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * x        (x) <U1 8B 'a' 'b'
          * y        (y) int64 16B 1 0
        Data variables:
            A        (x, y) int64 32B 3 4 1 2
            B        (x, y) int64 32B 7 8 5 6
        >>> ds.sortby(lambda x: -x["y"])
        <xarray.Dataset> Size: 88B
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * x        (x) <U1 8B 'b' 'a'
          * y        (y) int64 16B 1 0
        Data variables:
            A        (x, y) int64 32B 1 2 3 4
            B        (x, y) int64 32B 5 6 7 8
        """
        pass

    @_deprecate_positional_args('v2023.10.0')
    def quantile(self, q: ArrayLike, dim: Dims=None, *, method: QuantileMethods='linear', numeric_only: bool=False, keep_attrs: bool | None=None, skipna: bool | None=None, interpolation: QuantileMethods | None=None) -> Self:
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements for each variable
        in the Dataset.

        Parameters
        ----------
        q : float or array-like of float
            Quantile to compute, which must be between 0 and 1 inclusive.
        dim : str or Iterable of Hashable, optional
            Dimension(s) over which to apply quantile.
        method : str, default: "linear"
            This optional parameter specifies the interpolation method to use when the
            desired quantile lies between two data points. The options sorted by their R
            type as summarized in the H&F paper [1]_ are:

                1. "inverted_cdf"
                2. "averaged_inverted_cdf"
                3. "closest_observation"
                4. "interpolated_inverted_cdf"
                5. "hazen"
                6. "weibull"
                7. "linear"  (default)
                8. "median_unbiased"
                9. "normal_unbiased"

            The first three methods are discontiuous.  The following discontinuous
            variations of the default "linear" (7.) option are also available:

                * "lower"
                * "higher"
                * "midpoint"
                * "nearest"

            See :py:func:`numpy.quantile` or [1]_ for details. The "method" argument
            was previously called "interpolation", renamed in accordance with numpy
            version 1.22.0.

        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        numeric_only : bool, optional
            If True, only apply ``func`` to variables with a numeric dtype.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        quantiles : Dataset
            If `q` is a single quantile, then the result is a scalar for each
            variable in data_vars. If multiple percentiles are given, first
            axis of the result corresponds to the quantile and a quantile
            dimension is added to the return Dataset. The other dimensions are
            the dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, DataArray.quantile

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {"a": (("x", "y"), [[0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]])},
        ...     coords={"x": [7, 9], "y": [1, 1.5, 2, 2.5]},
        ... )
        >>> ds.quantile(0)  # or ds.quantile(0, dim=...)
        <xarray.Dataset> Size: 16B
        Dimensions:   ()
        Coordinates:
            quantile  float64 8B 0.0
        Data variables:
            a         float64 8B 0.7
        >>> ds.quantile(0, dim="x")
        <xarray.Dataset> Size: 72B
        Dimensions:   (y: 4)
        Coordinates:
          * y         (y) float64 32B 1.0 1.5 2.0 2.5
            quantile  float64 8B 0.0
        Data variables:
            a         (y) float64 32B 0.7 4.2 2.6 1.5
        >>> ds.quantile([0, 0.5, 1])
        <xarray.Dataset> Size: 48B
        Dimensions:   (quantile: 3)
        Coordinates:
          * quantile  (quantile) float64 24B 0.0 0.5 1.0
        Data variables:
            a         (quantile) float64 24B 0.7 3.4 9.4
        >>> ds.quantile([0, 0.5, 1], dim="x")
        <xarray.Dataset> Size: 152B
        Dimensions:   (quantile: 3, y: 4)
        Coordinates:
          * y         (y) float64 32B 1.0 1.5 2.0 2.5
          * quantile  (quantile) float64 24B 0.0 0.5 1.0
        Data variables:
            a         (quantile, y) float64 96B 0.7 4.2 2.6 1.5 3.6 ... 6.5 7.3 9.4 1.9

        References
        ----------
        .. [1] R. J. Hyndman and Y. Fan,
           "Sample quantiles in statistical packages,"
           The American Statistician, 50(4), pp. 361-365, 1996
        """
        pass

    @_deprecate_positional_args('v2023.10.0')
    def rank(self, dim: Hashable, *, pct: bool=False, keep_attrs: bool | None=None) -> Self:
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within
        that set.
        Ranks begin at 1, not 0. If pct is True, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : Hashable
            Dimension over which to compute rank.
        pct : bool, default: False
            If True, compute percentage ranks, otherwise compute integer ranks.
        keep_attrs : bool or None, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False, the new
            object will be returned without attributes.

        Returns
        -------
        ranked : Dataset
            Variables that do not depend on `dim` are dropped.
        """
        pass

    def differentiate(self, coord: Hashable, edge_order: Literal[1, 2]=1, datetime_unit: DatetimeUnitOptions | None=None) -> Self:
        """Differentiate with the second order accurate central
        differences.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord : Hashable
            The coordinate to be used to compute the gradient.
        edge_order : {1, 2}, default: 1
            N-th order accurate differences at the boundaries.
        datetime_unit : None or {"Y", "M", "W", "D", "h", "m", "s", "ms",             "us", "ns", "ps", "fs", "as", None}, default: None
            Unit to compute gradient. Only valid for datetime coordinate.

        Returns
        -------
        differentiated: Dataset

        See also
        --------
        numpy.gradient: corresponding numpy function
        """
        pass

    def integrate(self, coord: Hashable | Sequence[Hashable], datetime_unit: DatetimeUnitOptions=None) -> Self:
        """Integrate along the given coordinate using the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord : hashable, or sequence of hashable
            Coordinate(s) used for the integration.
        datetime_unit : {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns',                         'ps', 'fs', 'as', None}, optional
            Specify the unit if datetime coordinate is used.

        Returns
        -------
        integrated : Dataset

        See also
        --------
        DataArray.integrate
        numpy.trapz : corresponding numpy function

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 5, 6, 6]), "b": ("x", [1, 2, 1, 0])},
        ...     coords={"x": [0, 1, 2, 3], "y": ("x", [1, 7, 3, 5])},
        ... )
        >>> ds
        <xarray.Dataset> Size: 128B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
            y        (x) int64 32B 1 7 3 5
        Data variables:
            a        (x) int64 32B 5 5 6 6
            b        (x) int64 32B 1 2 1 0
        >>> ds.integrate("x")
        <xarray.Dataset> Size: 16B
        Dimensions:  ()
        Data variables:
            a        float64 8B 16.5
            b        float64 8B 3.5
        >>> ds.integrate("y")
        <xarray.Dataset> Size: 16B
        Dimensions:  ()
        Data variables:
            a        float64 8B 20.0
            b        float64 8B 4.0
        """
        pass

    def cumulative_integrate(self, coord: Hashable | Sequence[Hashable], datetime_unit: DatetimeUnitOptions=None) -> Self:
        """Integrate along the given coordinate using the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

            The first entry of the cumulative integral of each variable is always 0, in
            order to keep the length of the dimension unchanged between input and
            output.

        Parameters
        ----------
        coord : hashable, or sequence of hashable
            Coordinate(s) used for the integration.
        datetime_unit : {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns',                         'ps', 'fs', 'as', None}, optional
            Specify the unit if datetime coordinate is used.

        Returns
        -------
        integrated : Dataset

        See also
        --------
        DataArray.cumulative_integrate
        scipy.integrate.cumulative_trapezoid : corresponding scipy function

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 5, 6, 6]), "b": ("x", [1, 2, 1, 0])},
        ...     coords={"x": [0, 1, 2, 3], "y": ("x", [1, 7, 3, 5])},
        ... )
        >>> ds
        <xarray.Dataset> Size: 128B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
            y        (x) int64 32B 1 7 3 5
        Data variables:
            a        (x) int64 32B 5 5 6 6
            b        (x) int64 32B 1 2 1 0
        >>> ds.cumulative_integrate("x")
        <xarray.Dataset> Size: 128B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
            y        (x) int64 32B 1 7 3 5
        Data variables:
            a        (x) float64 32B 0.0 5.0 10.5 16.5
            b        (x) float64 32B 0.0 1.5 3.0 3.5
        >>> ds.cumulative_integrate("y")
        <xarray.Dataset> Size: 128B
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 32B 0 1 2 3
            y        (x) int64 32B 1 7 3 5
        Data variables:
            a        (x) float64 32B 0.0 30.0 8.0 20.0
            b        (x) float64 32B 0.0 9.0 3.0 4.0
        """
        pass

    @property
    def real(self) -> Self:
        """
        The real part of each data variable.

        See Also
        --------
        numpy.ndarray.real
        """
        pass

    @property
    def imag(self) -> Self:
        """
        The imaginary part of each data variable.

        See Also
        --------
        numpy.ndarray.imag
        """
        pass
    plot = utils.UncachedAccessor(DatasetPlotAccessor)

    def filter_by_attrs(self, **kwargs) -> Self:
        """Returns a ``Dataset`` with variables that match specific conditions.

        Can pass in ``key=value`` or ``key=callable``.  A Dataset is returned
        containing only the variables for which all the filter tests pass.
        These tests are either ``key=value`` for which the attribute ``key``
        has the exact value ``value`` or the callable passed into
        ``key=callable`` returns True. The callable will be passed a single
        value, either the value of the attribute ``key`` or ``None`` if the
        DataArray does not have an attribute with the name ``key``.

        Parameters
        ----------
        **kwargs
            key : str
                Attribute name.
            value : callable or obj
                If value is a callable, it should return a boolean in the form
                of bool = func(attr) where attr is da.attrs[key].
                Otherwise, value will be compared to the each
                DataArray's attrs[key].

        Returns
        -------
        new : Dataset
            New dataset with variables filtered by attribute.

        Examples
        --------
        >>> temp = 15 + 8 * np.random.randn(2, 2, 3)
        >>> precip = 10 * np.random.rand(2, 2, 3)
        >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
        >>> lat = [[42.25, 42.21], [42.63, 42.59]]
        >>> dims = ["x", "y", "time"]
        >>> temp_attr = dict(standard_name="air_potential_temperature")
        >>> precip_attr = dict(standard_name="convective_precipitation_flux")

        >>> ds = xr.Dataset(
        ...     dict(
        ...         temperature=(dims, temp, temp_attr),
        ...         precipitation=(dims, precip, precip_attr),
        ...     ),
        ...     coords=dict(
        ...         lon=(["x", "y"], lon),
        ...         lat=(["x", "y"], lat),
        ...         time=pd.date_range("2014-09-06", periods=3),
        ...         reference_time=pd.Timestamp("2014-09-05"),
        ...     ),
        ... )

        Get variables matching a specific standard_name:

        >>> ds.filter_by_attrs(standard_name="convective_precipitation_flux")
        <xarray.Dataset> Size: 192B
        Dimensions:         (x: 2, y: 2, time: 3)
        Coordinates:
            lon             (x, y) float64 32B -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 32B 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 24B 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 8B 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            precipitation   (x, y, time) float64 96B 5.68 9.256 0.7104 ... 4.615 7.805

        Get all variables that have a standard_name attribute:

        >>> standard_name = lambda v: v is not None
        >>> ds.filter_by_attrs(standard_name=standard_name)
        <xarray.Dataset> Size: 288B
        Dimensions:         (x: 2, y: 2, time: 3)
        Coordinates:
            lon             (x, y) float64 32B -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 32B 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 24B 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 8B 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            temperature     (x, y, time) float64 96B 29.11 18.2 22.83 ... 16.15 26.63
            precipitation   (x, y, time) float64 96B 5.68 9.256 0.7104 ... 4.615 7.805

        """
        pass

    def unify_chunks(self) -> Self:
        """Unify chunk size along all chunked dimensions of this Dataset.

        Returns
        -------
        Dataset with consistent chunk sizes for all dask-array variables

        See Also
        --------
        dask.array.core.unify_chunks
        """
        pass

    def map_blocks(self, func: Callable[..., T_Xarray], args: Sequence[Any]=(), kwargs: Mapping[str, Any] | None=None, template: DataArray | Dataset | None=None) -> T_Xarray:
        """
        Apply a function to each block of this Dataset.

        .. warning::
            This method is experimental and its signature may change.

        Parameters
        ----------
        func : callable
            User-provided function that accepts a Dataset as its first
            parameter. The function will receive a subset or 'block' of this Dataset (see below),
            corresponding to one chunk along each chunked dimension. ``func`` will be
            executed as ``func(subset_dataset, *subset_args, **kwargs)``.

            This function must return either a single DataArray or a single Dataset.

            This function cannot add a new chunked dimension.
        args : sequence
            Passed to func after unpacking and subsetting any xarray objects by blocks.
            xarray objects in args must be aligned with obj, otherwise an error is raised.
        kwargs : Mapping or None
            Passed verbatim to func after unpacking. xarray objects, if any, will not be
            subset to blocks. Passing dask collections in kwargs is not allowed.
        template : DataArray, Dataset or None, optional
            xarray object representing the final result after compute is called. If not provided,
            the function will be first run on mocked-up data, that looks like this object but
            has sizes 0, to determine properties of the returned object such as dtype,
            variable names, attributes, new dimensions and new indexes (if any).
            ``template`` must be provided if the function changes the size of existing dimensions.
            When provided, ``attrs`` on variables in `template` are copied over to the result. Any
            ``attrs`` set by ``func`` will be ignored.

        Returns
        -------
        A single DataArray or Dataset with dask backend, reassembled from the outputs of the
        function.

        Notes
        -----
        This function is designed for when ``func`` needs to manipulate a whole xarray object
        subset to each block. Each block is loaded into memory. In the more common case where
        ``func`` can work on numpy arrays, it is recommended to use ``apply_ufunc``.

        If none of the variables in this object is backed by dask arrays, calling this function is
        equivalent to calling ``func(obj, *args, **kwargs)``.

        See Also
        --------
        dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks
        xarray.DataArray.map_blocks

        :doc:`xarray-tutorial:advanced/map_blocks/map_blocks`
            Advanced Tutorial on map_blocks with dask


        Examples
        --------
        Calculate an anomaly from climatology using ``.groupby()``. Using
        ``xr.map_blocks()`` allows for parallel operations with knowledge of ``xarray``,
        its indices, and its methods like ``.groupby()``.

        >>> def calculate_anomaly(da, groupby_type="time.month"):
        ...     gb = da.groupby(groupby_type)
        ...     clim = gb.mean(dim="time")
        ...     return gb - clim
        ...
        >>> time = xr.cftime_range("1990-01", "1992-01", freq="ME")
        >>> month = xr.DataArray(time.month, coords={"time": time}, dims=["time"])
        >>> np.random.seed(123)
        >>> array = xr.DataArray(
        ...     np.random.rand(len(time)),
        ...     dims=["time"],
        ...     coords={"time": time, "month": month},
        ... ).chunk()
        >>> ds = xr.Dataset({"a": array})
        >>> ds.map_blocks(calculate_anomaly, template=ds).compute()
        <xarray.Dataset> Size: 576B
        Dimensions:  (time: 24)
        Coordinates:
          * time     (time) object 192B 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
            month    (time) int64 192B 1 2 3 4 5 6 7 8 9 10 ... 3 4 5 6 7 8 9 10 11 12
        Data variables:
            a        (time) float64 192B 0.1289 0.1132 -0.0856 ... 0.1906 -0.05901

        Note that one must explicitly use ``args=[]`` and ``kwargs={}`` to pass arguments
        to the function being applied in ``xr.map_blocks()``:

        >>> ds.map_blocks(
        ...     calculate_anomaly,
        ...     kwargs={"groupby_type": "time.year"},
        ...     template=ds,
        ... )
        <xarray.Dataset> Size: 576B
        Dimensions:  (time: 24)
        Coordinates:
          * time     (time) object 192B 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
            month    (time) int64 192B dask.array<chunksize=(24,), meta=np.ndarray>
        Data variables:
            a        (time) float64 192B dask.array<chunksize=(24,), meta=np.ndarray>
        """
        pass

    def polyfit(self, dim: Hashable, deg: int, skipna: bool | None=None, rcond: float | None=None, w: Hashable | Any=None, full: bool=False, cov: bool | Literal['unscaled']=False) -> Self:
        """
        Least squares polynomial fit.

        This replicates the behaviour of `numpy.polyfit` but differs by skipping
        invalid values when `skipna = True`.

        Parameters
        ----------
        dim : hashable
            Coordinate along which to fit the polynomials.
        deg : int
            Degree of the fitting polynomial.
        skipna : bool or None, optional
            If True, removes all invalid values before fitting each 1D slices of the array.
            Default is True if data is stored in a dask.array or if there is any
            invalid values, False otherwise.
        rcond : float or None, optional
            Relative condition number to the fit.
        w : hashable or Any, optional
            Weights to apply to the y-coordinate of the sample points.
            Can be an array-like object or the name of a coordinate in the dataset.
        full : bool, default: False
            Whether to return the residuals, matrix rank and singular values in addition
            to the coefficients.
        cov : bool or "unscaled", default: False
            Whether to return to the covariance matrix in addition to the coefficients.
            The matrix is not scaled if `cov='unscaled'`.

        Returns
        -------
        polyfit_results : Dataset
            A single dataset which contains (for each "var" in the input dataset):

            [var]_polyfit_coefficients
                The coefficients of the best fit for each variable in this dataset.
            [var]_polyfit_residuals
                The residuals of the least-square computation for each variable (only included if `full=True`)
                When the matrix rank is deficient, np.nan is returned.
            [dim]_matrix_rank
                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)
                The rank is computed ignoring the NaN values that might be skipped.
            [dim]_singular_values
                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)
            [var]_polyfit_covariance
                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)

        Warns
        -----
        RankWarning
            The rank of the coefficient matrix in the least-squares fit is deficient.
            The warning is not raised with in-memory (not dask) data and `full=True`.

        See Also
        --------
        numpy.polyfit
        numpy.polyval
        xarray.polyval
        """
        pass

    def pad(self, pad_width: Mapping[Any, int | tuple[int, int]] | None=None, mode: PadModeOptions='constant', stat_length: int | tuple[int, int] | Mapping[Any, tuple[int, int]] | None=None, constant_values: float | tuple[float, float] | Mapping[Any, tuple[float, float]] | None=None, end_values: int | tuple[int, int] | Mapping[Any, tuple[int, int]] | None=None, reflect_type: PadReflectOptions=None, keep_attrs: bool | None=None, **pad_width_kwargs: Any) -> Self:
        """Pad this dataset along one or more dimensions.

        .. warning::
            This function is experimental and its behaviour is likely to change
            especially regarding padding of dimension coordinates (or IndexVariables).

        When using one of the modes ("edge", "reflect", "symmetric", "wrap"),
        coordinates will be padded with the same mode, otherwise coordinates
        are padded using the "constant" mode with fill_value dtypes.NA.

        Parameters
        ----------
        pad_width : mapping of hashable to tuple of int
            Mapping with the form of {dim: (pad_before, pad_after)}
            describing the number of values padded along each dimension.
            {dim: pad} is a shortcut for pad_before = pad_after = pad
        mode : {"constant", "edge", "linear_ramp", "maximum", "mean", "median",             "minimum", "reflect", "symmetric", "wrap"}, default: "constant"
            How to pad the DataArray (taken from numpy docs):

            - "constant": Pads with a constant value.
            - "edge": Pads with the edge values of array.
            - "linear_ramp": Pads with the linear ramp between end_value and the
              array edge value.
            - "maximum": Pads with the maximum value of all or part of the
              vector along each axis.
            - "mean": Pads with the mean value of all or part of the
              vector along each axis.
            - "median": Pads with the median value of all or part of the
              vector along each axis.
            - "minimum": Pads with the minimum value of all or part of the
              vector along each axis.
            - "reflect": Pads with the reflection of the vector mirrored on
              the first and last values of the vector along each axis.
            - "symmetric": Pads with the reflection of the vector mirrored
              along the edge of the array.
            - "wrap": Pads with the wrap of the vector along the axis.
              The first values are used to pad the end and the
              end values are used to pad the beginning.

        stat_length : int, tuple or mapping of hashable to tuple, default: None
            Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
            values at edge of each axis used to calculate the statistic value.
            {dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)} unique
            statistic lengths along each dimension.
            ((before, after),) yields same before and after statistic lengths
            for each dimension.
            (stat_length,) or int is a shortcut for before = after = statistic
            length for all axes.
            Default is ``None``, to use the entire axis.
        constant_values : scalar, tuple or mapping of hashable to tuple, default: 0
            Used in 'constant'.  The values to set the padded values for each
            axis.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            pad constants along each dimension.
            ``((before, after),)`` yields same before and after constants for each
            dimension.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all dimensions.
            Default is 0.
        end_values : scalar, tuple or mapping of hashable to tuple, default: 0
            Used in 'linear_ramp'.  The values used for the ending value of the
            linear_ramp and that will form the edge of the padded array.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            end values along each dimension.
            ``((before, after),)`` yields same before and after end values for each
            axis.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all axes.
            Default is 0.
        reflect_type : {"even", "odd", None}, optional
            Used in "reflect", and "symmetric".  The "even" style is the
            default with an unaltered reflection around the edge value.  For
            the "odd" style, the extended part of the array is created by
            subtracting the reflected values from two times the edge value.
        keep_attrs : bool or None, optional
            If True, the attributes (``attrs``) will be copied from the
            original object to the new one. If False, the new object
            will be returned without attributes.
        **pad_width_kwargs
            The keyword arguments form of ``pad_width``.
            One of ``pad_width`` or ``pad_width_kwargs`` must be provided.

        Returns
        -------
        padded : Dataset
            Dataset with the padded coordinates and data.

        See Also
        --------
        Dataset.shift, Dataset.roll, Dataset.bfill, Dataset.ffill, numpy.pad, dask.array.pad

        Notes
        -----
        By default when ``mode="constant"`` and ``constant_values=None``, integer types will be
        promoted to ``float`` and padded with ``np.nan``. To avoid type promotion
        specify ``constant_values=np.nan``

        Padding coordinates will drop their corresponding index (if any) and will reset default
        indexes for dimension coordinates.

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", range(5))})
        >>> ds.pad(x=(1, 2))
        <xarray.Dataset> Size: 64B
        Dimensions:  (x: 8)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) float64 64B nan 0.0 1.0 2.0 3.0 4.0 nan nan
        """
        pass

    @_deprecate_positional_args('v2023.10.0')
    def idxmin(self, dim: Hashable | None=None, *, skipna: bool | None=None, fill_value: Any=xrdtypes.NA, keep_attrs: bool | None=None) -> Self:
        """Return the coordinate label of the minimum value along a dimension.

        Returns a new `Dataset` named after the dimension with the values of
        the coordinate labels along that dimension corresponding to minimum
        values along that dimension.

        In comparison to :py:meth:`~Dataset.argmin`, this returns the
        coordinate label while :py:meth:`~Dataset.argmin` returns the index.

        Parameters
        ----------
        dim : Hashable, optional
            Dimension over which to apply `idxmin`.  This is optional for 1D
            variables, but required for variables with 2 or more dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for ``float``, ``complex``, and ``object``
            dtypes; other dtypes either do not have a sentinel missing value
            (``int``) or ``skipna=True`` has not been implemented
            (``datetime64`` or ``timedelta64``).
        fill_value : Any, default: NaN
            Value to be filled in case all of the values along a dimension are
            null.  By default this is NaN.  The fill value and result are
            automatically converted to a compatible dtype if possible.
            Ignored if ``skipna`` is False.
        keep_attrs : bool or None, optional
            If True, the attributes (``attrs``) will be copied from the
            original object to the new one. If False, the new object
            will be returned without attributes.

        Returns
        -------
        reduced : Dataset
            New `Dataset` object with `idxmin` applied to its data and the
            indicated dimension removed.

        See Also
        --------
        DataArray.idxmin, Dataset.idxmax, Dataset.min, Dataset.argmin

        Examples
        --------
        >>> array1 = xr.DataArray(
        ...     [0, 2, 1, 0, -2], dims="x", coords={"x": ["a", "b", "c", "d", "e"]}
        ... )
        >>> array2 = xr.DataArray(
        ...     [
        ...         [2.0, 1.0, 2.0, 0.0, -2.0],
        ...         [-4.0, np.nan, 2.0, np.nan, -2.0],
        ...         [np.nan, np.nan, 1.0, np.nan, np.nan],
        ...     ],
        ...     dims=["y", "x"],
        ...     coords={"y": [-1, 0, 1], "x": ["a", "b", "c", "d", "e"]},
        ... )
        >>> ds = xr.Dataset({"int": array1, "float": array2})
        >>> ds.min(dim="x")
        <xarray.Dataset> Size: 56B
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 24B -1 0 1
        Data variables:
            int      int64 8B -2
            float    (y) float64 24B -2.0 -4.0 1.0
        >>> ds.argmin(dim="x")
        <xarray.Dataset> Size: 56B
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 24B -1 0 1
        Data variables:
            int      int64 8B 4
            float    (y) int64 24B 4 0 2
        >>> ds.idxmin(dim="x")
        <xarray.Dataset> Size: 52B
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 24B -1 0 1
        Data variables:
            int      <U1 4B 'e'
            float    (y) object 24B 'e' 'a' 'c'
        """
        pass

    @_deprecate_positional_args('v2023.10.0')
    def idxmax(self, dim: Hashable | None=None, *, skipna: bool | None=None, fill_value: Any=xrdtypes.NA, keep_attrs: bool | None=None) -> Self:
        """Return the coordinate label of the maximum value along a dimension.

        Returns a new `Dataset` named after the dimension with the values of
        the coordinate labels along that dimension corresponding to maximum
        values along that dimension.

        In comparison to :py:meth:`~Dataset.argmax`, this returns the
        coordinate label while :py:meth:`~Dataset.argmax` returns the index.

        Parameters
        ----------
        dim : str, optional
            Dimension over which to apply `idxmax`.  This is optional for 1D
            variables, but required for variables with 2 or more dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for ``float``, ``complex``, and ``object``
            dtypes; other dtypes either do not have a sentinel missing value
            (``int``) or ``skipna=True`` has not been implemented
            (``datetime64`` or ``timedelta64``).
        fill_value : Any, default: NaN
            Value to be filled in case all of the values along a dimension are
            null.  By default this is NaN.  The fill value and result are
            automatically converted to a compatible dtype if possible.
            Ignored if ``skipna`` is False.
        keep_attrs : bool or None, optional
            If True, the attributes (``attrs``) will be copied from the
            original object to the new one. If False, the new object
            will be returned without attributes.

        Returns
        -------
        reduced : Dataset
            New `Dataset` object with `idxmax` applied to its data and the
            indicated dimension removed.

        See Also
        --------
        DataArray.idxmax, Dataset.idxmin, Dataset.max, Dataset.argmax

        Examples
        --------
        >>> array1 = xr.DataArray(
        ...     [0, 2, 1, 0, -2], dims="x", coords={"x": ["a", "b", "c", "d", "e"]}
        ... )
        >>> array2 = xr.DataArray(
        ...     [
        ...         [2.0, 1.0, 2.0, 0.0, -2.0],
        ...         [-4.0, np.nan, 2.0, np.nan, -2.0],
        ...         [np.nan, np.nan, 1.0, np.nan, np.nan],
        ...     ],
        ...     dims=["y", "x"],
        ...     coords={"y": [-1, 0, 1], "x": ["a", "b", "c", "d", "e"]},
        ... )
        >>> ds = xr.Dataset({"int": array1, "float": array2})
        >>> ds.max(dim="x")
        <xarray.Dataset> Size: 56B
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 24B -1 0 1
        Data variables:
            int      int64 8B 2
            float    (y) float64 24B 2.0 2.0 1.0
        >>> ds.argmax(dim="x")
        <xarray.Dataset> Size: 56B
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 24B -1 0 1
        Data variables:
            int      int64 8B 1
            float    (y) int64 24B 0 2 2
        >>> ds.idxmax(dim="x")
        <xarray.Dataset> Size: 52B
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 24B -1 0 1
        Data variables:
            int      <U1 4B 'b'
            float    (y) object 24B 'a' 'c' 'c'
        """
        pass

    def argmin(self, dim: Hashable | None=None, **kwargs) -> Self:
        """Indices of the minima of the member variables.

        If there are multiple minima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : Hashable, optional
            The dimension over which to find the minimum. By default, finds minimum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will be an error, since DataArray.argmin will
            return a dict with indices for all dimensions, which does not make sense for
            a Dataset.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : Dataset

        Examples
        --------
        >>> dataset = xr.Dataset(
        ...     {
        ...         "math_scores": (
        ...             ["student", "test"],
        ...             [[90, 85, 79], [78, 80, 85], [95, 92, 98]],
        ...         ),
        ...         "english_scores": (
        ...             ["student", "test"],
        ...             [[88, 90, 92], [75, 82, 79], [39, 96, 78]],
        ...         ),
        ...     },
        ...     coords={
        ...         "student": ["Alice", "Bob", "Charlie"],
        ...         "test": ["Test 1", "Test 2", "Test 3"],
        ...     },
        ... )

        # Indices of the minimum values along the 'student' dimension are calculated

        >>> argmin_indices = dataset.argmin(dim="student")

        >>> min_score_in_math = dataset["student"].isel(
        ...     student=argmin_indices["math_scores"]
        ... )
        >>> min_score_in_math
        <xarray.DataArray 'student' (test: 3)> Size: 84B
        array(['Bob', 'Bob', 'Alice'], dtype='<U7')
        Coordinates:
            student  (test) <U7 84B 'Bob' 'Bob' 'Alice'
          * test     (test) <U6 72B 'Test 1' 'Test 2' 'Test 3'

        >>> min_score_in_english = dataset["student"].isel(
        ...     student=argmin_indices["english_scores"]
        ... )
        >>> min_score_in_english
        <xarray.DataArray 'student' (test: 3)> Size: 84B
        array(['Charlie', 'Bob', 'Charlie'], dtype='<U7')
        Coordinates:
            student  (test) <U7 84B 'Charlie' 'Bob' 'Charlie'
          * test     (test) <U6 72B 'Test 1' 'Test 2' 'Test 3'

        See Also
        --------
        Dataset.idxmin
        DataArray.argmin
        """
        pass

    def argmax(self, dim: Hashable | None=None, **kwargs) -> Self:
        """Indices of the maxima of the member variables.

        If there are multiple maxima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : str, optional
            The dimension over which to find the maximum. By default, finds maximum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will be an error, since DataArray.argmax will
            return a dict with indices for all dimensions, which does not make sense for
            a Dataset.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : Dataset

        Examples
        --------

        >>> dataset = xr.Dataset(
        ...     {
        ...         "math_scores": (
        ...             ["student", "test"],
        ...             [[90, 85, 92], [78, 80, 85], [95, 92, 98]],
        ...         ),
        ...         "english_scores": (
        ...             ["student", "test"],
        ...             [[88, 90, 92], [75, 82, 79], [93, 96, 91]],
        ...         ),
        ...     },
        ...     coords={
        ...         "student": ["Alice", "Bob", "Charlie"],
        ...         "test": ["Test 1", "Test 2", "Test 3"],
        ...     },
        ... )

        # Indices of the maximum values along the 'student' dimension are calculated

        >>> argmax_indices = dataset.argmax(dim="test")

        >>> argmax_indices
        <xarray.Dataset> Size: 132B
        Dimensions:         (student: 3)
        Coordinates:
          * student         (student) <U7 84B 'Alice' 'Bob' 'Charlie'
        Data variables:
            math_scores     (student) int64 24B 2 2 2
            english_scores  (student) int64 24B 2 1 1

        See Also
        --------
        DataArray.argmax

        """
        pass

    def eval(self, statement: str, *, parser: QueryParserOptions='pandas') -> Self | T_DataArray:
        """
        Calculate an expression supplied as a string in the context of the dataset.

        This is currently experimental; the API may change particularly around
        assignments, which currently returnn a ``Dataset`` with the additional variable.
        Currently only the ``python`` engine is supported, which has the same
        performance as executing in python.

        Parameters
        ----------
        statement : str
            String containing the Python-like expression to evaluate.

        Returns
        -------
        result : Dataset or DataArray, depending on whether ``statement`` contains an
        assignment.

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {"a": ("x", np.arange(0, 5, 1)), "b": ("x", np.linspace(0, 1, 5))}
        ... )
        >>> ds
        <xarray.Dataset> Size: 80B
        Dimensions:  (x: 5)
        Dimensions without coordinates: x
        Data variables:
            a        (x) int64 40B 0 1 2 3 4
            b        (x) float64 40B 0.0 0.25 0.5 0.75 1.0

        >>> ds.eval("a + b")
        <xarray.DataArray (x: 5)> Size: 40B
        array([0.  , 1.25, 2.5 , 3.75, 5.  ])
        Dimensions without coordinates: x

        >>> ds.eval("c = a + b")
        <xarray.Dataset> Size: 120B
        Dimensions:  (x: 5)
        Dimensions without coordinates: x
        Data variables:
            a        (x) int64 40B 0 1 2 3 4
            b        (x) float64 40B 0.0 0.25 0.5 0.75 1.0
            c        (x) float64 40B 0.0 1.25 2.5 3.75 5.0
        """
        pass

    def query(self, queries: Mapping[Any, Any] | None=None, parser: QueryParserOptions='pandas', engine: QueryEngineOptions=None, missing_dims: ErrorOptionsWithWarn='raise', **queries_kwargs: Any) -> Self:
        """Return a new dataset with each array indexed along the specified
        dimension(s), where the indexers are given as strings containing
        Python expressions to be evaluated against the data variables in the
        dataset.

        Parameters
        ----------
        queries : dict-like, optional
            A dict-like with keys matching dimensions and values given by strings
            containing Python expressions to be evaluated against the data variables
            in the dataset. The expressions will be evaluated using the pandas
            eval() function, and can contain any valid Python expressions but cannot
            contain any Python statements.
        parser : {"pandas", "python"}, default: "pandas"
            The parser to use to construct the syntax tree from the expression.
            The default of 'pandas' parses code slightly different than standard
            Python. Alternatively, you can parse an expression using the 'python'
            parser to retain strict Python semantics.
        engine : {"python", "numexpr", None}, default: None
            The engine used to evaluate the expression. Supported engines are:

            - None: tries to use numexpr, falls back to python
            - "numexpr": evaluates expressions using numexpr
            - "python": performs operations as if you had eval’d in top level python

        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:

            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        **queries_kwargs : {dim: query, ...}, optional
            The keyword arguments form of ``queries``.
            One of queries or queries_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the results of the appropriate
            queries.

        See Also
        --------
        Dataset.isel
        pandas.eval

        Examples
        --------
        >>> a = np.arange(0, 5, 1)
        >>> b = np.linspace(0, 1, 5)
        >>> ds = xr.Dataset({"a": ("x", a), "b": ("x", b)})
        >>> ds
        <xarray.Dataset> Size: 80B
        Dimensions:  (x: 5)
        Dimensions without coordinates: x
        Data variables:
            a        (x) int64 40B 0 1 2 3 4
            b        (x) float64 40B 0.0 0.25 0.5 0.75 1.0
        >>> ds.query(x="a > 2")
        <xarray.Dataset> Size: 32B
        Dimensions:  (x: 2)
        Dimensions without coordinates: x
        Data variables:
            a        (x) int64 16B 3 4
            b        (x) float64 16B 0.75 1.0
        """
        pass

    def curvefit(self, coords: str | DataArray | Iterable[str | DataArray], func: Callable[..., Any], reduce_dims: Dims=None, skipna: bool=True, p0: Mapping[str, float | DataArray] | None=None, bounds: Mapping[str, tuple[float | DataArray, float | DataArray]] | None=None, param_names: Sequence[str] | None=None, errors: ErrorOptions='raise', kwargs: dict[str, Any] | None=None) -> Self:
        """
        Curve fitting optimization for arbitrary functions.

        Wraps `scipy.optimize.curve_fit` with `apply_ufunc`.

        Parameters
        ----------
        coords : hashable, DataArray, or sequence of hashable or DataArray
            Independent coordinate(s) over which to perform the curve fitting. Must share
            at least one dimension with the calling object. When fitting multi-dimensional
            functions, supply `coords` as a sequence in the same order as arguments in
            `func`. To fit along existing dimensions of the calling object, `coords` can
            also be specified as a str or sequence of strs.
        func : callable
            User specified function in the form `f(x, *params)` which returns a numpy
            array of length `len(x)`. `params` are the fittable parameters which are optimized
            by scipy curve_fit. `x` can also be specified as a sequence containing multiple
            coordinates, e.g. `f((x0, x1), *params)`.
        reduce_dims : str, Iterable of Hashable or None, optional
            Additional dimension(s) over which to aggregate while fitting. For example,
            calling `ds.curvefit(coords='time', reduce_dims=['lat', 'lon'], ...)` will
            aggregate all lat and lon points and fit the specified function along the
            time dimension.
        skipna : bool, default: True
            Whether to skip missing values when fitting. Default is True.
        p0 : dict-like, optional
            Optional dictionary of parameter names to initial guesses passed to the
            `curve_fit` `p0` arg. If the values are DataArrays, they will be appropriately
            broadcast to the coordinates of the array. If none or only some parameters are
            passed, the rest will be assigned initial values following the default scipy
            behavior.
        bounds : dict-like, optional
            Optional dictionary of parameter names to tuples of bounding values passed to the
            `curve_fit` `bounds` arg. If any of the bounds are DataArrays, they will be
            appropriately broadcast to the coordinates of the array. If none or only some
            parameters are passed, the rest will be unbounded following the default scipy
            behavior.
        param_names : sequence of hashable, optional
            Sequence of names for the fittable parameters of `func`. If not supplied,
            this will be automatically determined by arguments of `func`. `param_names`
            should be manually supplied when fitting a function that takes a variable
            number of parameters.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', any errors from the `scipy.optimize_curve_fit` optimization will
            raise an exception. If 'ignore', the coefficients and covariances for the
            coordinates where the fitting failed will be NaN.
        **kwargs : optional
            Additional keyword arguments to passed to scipy curve_fit.

        Returns
        -------
        curvefit_results : Dataset
            A single dataset which contains:

            [var]_curvefit_coefficients
                The coefficients of the best fit.
            [var]_curvefit_covariance
                The covariance matrix of the coefficient estimates.

        See Also
        --------
        Dataset.polyfit
        scipy.optimize.curve_fit
        """
        pass

    @_deprecate_positional_args('v2023.10.0')
    def drop_duplicates(self, dim: Hashable | Iterable[Hashable], *, keep: Literal['first', 'last', False]='first') -> Self:
        """Returns a new Dataset with duplicate dimension values removed.

        Parameters
        ----------
        dim : dimension label or labels
            Pass `...` to drop duplicates along all dimensions.
        keep : {"first", "last", False}, default: "first"
            Determines which duplicates (if any) to keep.
            - ``"first"`` : Drop duplicates except for the first occurrence.
            - ``"last"`` : Drop duplicates except for the last occurrence.
            - False : Drop all duplicates.

        Returns
        -------
        Dataset

        See Also
        --------
        DataArray.drop_duplicates
        """
        pass

    def convert_calendar(self, calendar: CFCalendar, dim: Hashable='time', align_on: Literal['date', 'year', None]=None, missing: Any | None=None, use_cftime: bool | None=None) -> Self:
        """Convert the Dataset to another calendar.

        Only converts the individual timestamps, does not modify any data except
        in dropping invalid/surplus dates or inserting missing dates.

        If the source and target calendars are either no_leap, all_leap or a
        standard type, only the type of the time array is modified.
        When converting to a leap year from a non-leap year, the 29th of February
        is removed from the array. In the other direction the 29th of February
        will be missing in the output, unless `missing` is specified,
        in which case that value is inserted.

        For conversions involving `360_day` calendars, see Notes.

        This method is safe to use with sub-daily data as it doesn't touch the
        time part of the timestamps.

        Parameters
        ---------
        calendar : str
            The target calendar name.
        dim : Hashable, default: "time"
            Name of the time coordinate.
        align_on : {None, 'date', 'year'}, optional
            Must be specified when either source or target is a `360_day` calendar,
            ignored otherwise. See Notes.
        missing : Any or None, optional
            By default, i.e. if the value is None, this method will simply attempt
            to convert the dates in the source calendar to the same dates in the
            target calendar, and drop any of those that are not possible to
            represent.  If a value is provided, a new time coordinate will be
            created in the target calendar with the same frequency as the original
            time coordinate; for any dates that are not present in the source, the
            data will be filled with this value.  Note that using this mode requires
            that the source data have an inferable frequency; for more information
            see :py:func:`xarray.infer_freq`.  For certain frequency, source, and
            target calendar combinations, this could result in many missing values, see notes.
        use_cftime : bool or None, optional
            Whether to use cftime objects in the output, only used if `calendar`
            is one of {"proleptic_gregorian", "gregorian" or "standard"}.
            If True, the new time axis uses cftime objects.
            If None (default), it uses :py:class:`numpy.datetime64` values if the
            date range permits it, and :py:class:`cftime.datetime` objects if not.
            If False, it uses :py:class:`numpy.datetime64`  or fails.

        Returns
        -------
        Dataset
            Copy of the dataarray with the time coordinate converted to the
            target calendar. If 'missing' was None (default), invalid dates in
            the new calendar are dropped, but missing dates are not inserted.
            If `missing` was given, the new data is reindexed to have a time axis
            with the same frequency as the source, but in the new calendar; any
            missing datapoints are filled with `missing`.

        Notes
        -----
        Passing a value to `missing` is only usable if the source's time coordinate as an
        inferable frequencies (see :py:func:`~xarray.infer_freq`) and is only appropriate
        if the target coordinate, generated from this frequency, has dates equivalent to the
        source. It is usually **not** appropriate to use this mode with:

        - Period-end frequencies : 'A', 'Y', 'Q' or 'M', in opposition to 'AS' 'YS', 'QS' and 'MS'
        - Sub-monthly frequencies that do not divide a day evenly : 'W', 'nD' where `N != 1`
            or 'mH' where 24 % m != 0).

        If one of the source or target calendars is `"360_day"`, `align_on` must
        be specified and two options are offered.

        - "year"
            The dates are translated according to their relative position in the year,
            ignoring their original month and day information, meaning that the
            missing/surplus days are added/removed at regular intervals.

            From a `360_day` to a standard calendar, the output will be missing the
            following dates (day of year in parentheses):

            To a leap year:
                January 31st (31), March 31st (91), June 1st (153), July 31st (213),
                September 31st (275) and November 30th (335).
            To a non-leap year:
                February 6th (36), April 19th (109), July 2nd (183),
                September 12th (255), November 25th (329).

            From a standard calendar to a `"360_day"`, the following dates in the
            source array will be dropped:

            From a leap year:
                January 31st (31), April 1st (92), June 1st (153), August 1st (214),
                September 31st (275), December 1st (336)
            From a non-leap year:
                February 6th (37), April 20th (110), July 2nd (183),
                September 13th (256), November 25th (329)

            This option is best used on daily and subdaily data.

        - "date"
            The month/day information is conserved and invalid dates are dropped
            from the output. This means that when converting from a `"360_day"` to a
            standard calendar, all 31st (Jan, March, May, July, August, October and
            December) will be missing as there is no equivalent dates in the
            `"360_day"` calendar and the 29th (on non-leap years) and 30th of February
            will be dropped as there are no equivalent dates in a standard calendar.

            This option is best used with data on a frequency coarser than daily.
        """
        pass

    def interp_calendar(self, target: pd.DatetimeIndex | CFTimeIndex | DataArray, dim: Hashable='time') -> Self:
        """Interpolates the Dataset to another calendar based on decimal year measure.

        Each timestamp in `source` and `target` are first converted to their decimal
        year equivalent then `source` is interpolated on the target coordinate.
        The decimal year of a timestamp is its year plus its sub-year component
        converted to the fraction of its year. For example "2000-03-01 12:00" is
        2000.1653 in a standard calendar or 2000.16301 in a `"noleap"` calendar.

        This method should only be used when the time (HH:MM:SS) information of
        time coordinate is not important.

        Parameters
        ----------
        target: DataArray or DatetimeIndex or CFTimeIndex
            The target time coordinate of a valid dtype
            (np.datetime64 or cftime objects)
        dim : Hashable, default: "time"
            The time coordinate name.

        Return
        ------
        DataArray
            The source interpolated on the decimal years of target,
        """
        pass

    @_deprecate_positional_args('v2024.07.0')
    def groupby(self, group: Hashable | DataArray | IndexVariable | Mapping[Any, Grouper] | None=None, *, squeeze: Literal[False]=False, restore_coord_dims: bool=False, **groupers: Grouper) -> DatasetGroupBy:
        """Returns a DatasetGroupBy object for performing grouped operations.

        Parameters
        ----------
        group : Hashable or DataArray or IndexVariable or mapping of Hashable to Grouper
            Array whose unique values should be used to group this array. If a
            Hashable, must be the name of a coordinate contained in this dataarray. If a dictionary,
            must map an existing variable name to a :py:class:`Grouper` instance.
        squeeze : bool, default: True
            If "group" is a dimension of any arrays in this dataset, `squeeze`
            controls whether the subarrays have a dimension of length 1 along
            that dimension or if the dimension is squeezed out.
        restore_coord_dims : bool, default: False
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        **groupers : Mapping of str to Grouper or Resampler
            Mapping of variable name to group by to :py:class:`Grouper` or :py:class:`Resampler` object.
            One of ``group`` or ``groupers`` must be provided.
            Only a single ``grouper`` is allowed at present.

        Returns
        -------
        grouped : DatasetGroupBy
            A `DatasetGroupBy` object patterned after `pandas.GroupBy` that can be
            iterated over in the form of `(unique_value, grouped_array)` pairs.

        See Also
        --------
        :ref:`groupby`
            Users guide explanation of how to group and bin data.

        :doc:`xarray-tutorial:intermediate/01-high-level-computation-patterns`
            Tutorial on :py:func:`~xarray.Dataset.Groupby` for windowed computation.

        :doc:`xarray-tutorial:fundamentals/03.2_groupby_with_xarray`
            Tutorial on :py:func:`~xarray.Dataset.Groupby` demonstrating reductions, transformation and comparison with :py:func:`~xarray.Dataset.resample`.

        Dataset.groupby_bins
        DataArray.groupby
        core.groupby.DatasetGroupBy
        pandas.DataFrame.groupby
        Dataset.coarsen
        Dataset.resample
        DataArray.resample
        """
        pass

    @_deprecate_positional_args('v2024.07.0')
    def groupby_bins(self, group: Hashable | DataArray | IndexVariable, bins: Bins, right: bool=True, labels: ArrayLike | None=None, precision: int=3, include_lowest: bool=False, squeeze: Literal[False]=False, restore_coord_dims: bool=False, duplicates: Literal['raise', 'drop']='raise') -> DatasetGroupBy:
        """Returns a DatasetGroupBy object for performing grouped operations.

        Rather than using all unique values of `group`, the values are discretized
        first by applying `pandas.cut` [1]_ to `group`.

        Parameters
        ----------
        group : Hashable, DataArray or IndexVariable
            Array whose binned values should be used to group this array. If a
            string, must be the name of a variable contained in this dataset.
        bins : int or array-like
            If bins is an int, it defines the number of equal-width bins in the
            range of x. However, in this case, the range of x is extended by .1%
            on each side to include the min or max values of x. If bins is a
            sequence it defines the bin edges allowing for non-uniform bin
            width. No extension of the range of x is done in this case.
        right : bool, default: True
            Indicates whether the bins include the rightmost edge or not. If
            right == True (the default), then the bins [1,2,3,4] indicate
            (1,2], (2,3], (3,4].
        labels : array-like or bool, default: None
            Used as labels for the resulting bins. Must be of the same length as
            the resulting bins. If False, string bin labels are assigned by
            `pandas.cut`.
        precision : int, default: 3
            The precision at which to store and display the bins labels.
        include_lowest : bool, default: False
            Whether the first interval should be left-inclusive or not.
        squeeze : False
            This argument is deprecated.
        restore_coord_dims : bool, default: False
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        duplicates : {"raise", "drop"}, default: "raise"
            If bin edges are not unique, raise ValueError or drop non-uniques.

        Returns
        -------
        grouped : DatasetGroupBy
            A `DatasetGroupBy` object patterned after `pandas.GroupBy` that can be
            iterated over in the form of `(unique_value, grouped_array)` pairs.
            The name of the group has the added suffix `_bins` in order to
            distinguish it from the original variable.

        See Also
        --------
        :ref:`groupby`
            Users guide explanation of how to group and bin data.
        Dataset.groupby
        DataArray.groupby_bins
        core.groupby.DatasetGroupBy
        pandas.DataFrame.groupby

        References
        ----------
        .. [1] http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
        """
        pass

    def weighted(self, weights: DataArray) -> DatasetWeighted:
        """
        Weighted Dataset operations.

        Parameters
        ----------
        weights : DataArray
            An array of weights associated with the values in this Dataset.
            Each value in the data contributes to the reduction operation
            according to its associated weight.

        Notes
        -----
        ``weights`` must be a DataArray and cannot contain missing values.
        Missing values can be replaced by ``weights.fillna(0)``.

        Returns
        -------
        core.weighted.DatasetWeighted

        See Also
        --------
        DataArray.weighted

        :ref:`comput.weighted`
            User guide on weighted array reduction using :py:func:`~xarray.Dataset.weighted`

        :doc:`xarray-tutorial:fundamentals/03.4_weighted`
            Tutorial on Weighted Reduction using :py:func:`~xarray.Dataset.weighted`

        """
        pass

    def rolling(self, dim: Mapping[Any, int] | None=None, min_periods: int | None=None, center: bool | Mapping[Any, bool]=False, **window_kwargs: int) -> DatasetRolling:
        """
        Rolling window object for Datasets.

        Parameters
        ----------
        dim : dict, optional
            Mapping from the dimension name to create the rolling iterator
            along (e.g. `time`) to its moving window size.
        min_periods : int or None, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool or Mapping to int, default: False
            Set the labels at the center of the window.
        **window_kwargs : optional
            The keyword arguments form of ``dim``.
            One of dim or window_kwargs must be provided.

        Returns
        -------
        core.rolling.DatasetRolling

        See Also
        --------
        Dataset.cumulative
        DataArray.rolling
        core.rolling.DatasetRolling
        """
        pass

    def cumulative(self, dim: str | Iterable[Hashable], min_periods: int=1) -> DatasetRolling:
        """
        Accumulating object for Datasets

        Parameters
        ----------
        dims : iterable of hashable
            The name(s) of the dimensions to create the cumulative window along
        min_periods : int, default: 1
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default is 1 (note this is different
            from ``Rolling``, whose default is the size of the window).

        Returns
        -------
        core.rolling.DatasetRolling

        See Also
        --------
        Dataset.rolling
        DataArray.cumulative
        core.rolling.DatasetRolling
        """
        pass

    def coarsen(self, dim: Mapping[Any, int] | None=None, boundary: CoarsenBoundaryOptions='exact', side: SideOptions | Mapping[Any, SideOptions]='left', coord_func: str | Callable | Mapping[Any, str | Callable]='mean', **window_kwargs: int) -> DatasetCoarsen:
        """
        Coarsen object for Datasets.

        Parameters
        ----------
        dim : mapping of hashable to int, optional
            Mapping from the dimension name to the window size.
        boundary : {"exact", "trim", "pad"}, default: "exact"
            If 'exact', a ValueError will be raised if dimension size is not a
            multiple of the window size. If 'trim', the excess entries are
            dropped. If 'pad', NA will be padded.
        side : {"left", "right"} or mapping of str to {"left", "right"}, default: "left"
        coord_func : str or mapping of hashable to str, default: "mean"
            function (name) that is applied to the coordinates,
            or a mapping from coordinate name to function (name).

        Returns
        -------
        core.rolling.DatasetCoarsen

        See Also
        --------
        core.rolling.DatasetCoarsen
        DataArray.coarsen

        :ref:`reshape.coarsen`
            User guide describing :py:func:`~xarray.Dataset.coarsen`

        :ref:`compute.coarsen`
            User guide on block arrgragation :py:func:`~xarray.Dataset.coarsen`

        :doc:`xarray-tutorial:fundamentals/03.3_windowed`
            Tutorial on windowed computation using :py:func:`~xarray.Dataset.coarsen`

        """
        pass

    @_deprecate_positional_args('v2024.07.0')
    def resample(self, indexer: Mapping[Any, str | Resampler] | None=None, *, skipna: bool | None=None, closed: SideOptions | None=None, label: SideOptions | None=None, offset: pd.Timedelta | datetime.timedelta | str | None=None, origin: str | DatetimeLike='start_day', restore_coord_dims: bool | None=None, **indexer_kwargs: str | Resampler) -> DatasetResample:
        """Returns a Resample object for performing resampling operations.

        Handles both downsampling and upsampling. The resampled
        dimension must be a datetime-like coordinate. If any intervals
        contain no values from the original object, they will be given
        the value ``NaN``.

        Parameters
        ----------
        indexer : Mapping of Hashable to str, optional
            Mapping from the dimension name to resample frequency [1]_. The
            dimension must be datetime-like.
        skipna : bool, optional
            Whether to skip missing values when aggregating in downsampling.
        closed : {"left", "right"}, optional
            Side of each interval to treat as closed.
        label : {"left", "right"}, optional
            Side of each interval to use for labeling.
        origin : {'epoch', 'start', 'start_day', 'end', 'end_day'}, pd.Timestamp, datetime.datetime, np.datetime64, or cftime.datetime, default 'start_day'
            The datetime on which to adjust the grouping. The timezone of origin
            must match the timezone of the index.

            If a datetime is not used, these values are also supported:
            - 'epoch': `origin` is 1970-01-01
            - 'start': `origin` is the first value of the timeseries
            - 'start_day': `origin` is the first day at midnight of the timeseries
            - 'end': `origin` is the last value of the timeseries
            - 'end_day': `origin` is the ceiling midnight of the last day
        offset : pd.Timedelta, datetime.timedelta, or str, default is None
            An offset timedelta added to the origin.
        restore_coord_dims : bool, optional
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        **indexer_kwargs : str
            The keyword arguments form of ``indexer``.
            One of indexer or indexer_kwargs must be provided.

        Returns
        -------
        resampled : core.resample.DataArrayResample
            This object resampled.

        See Also
        --------
        DataArray.resample
        pandas.Series.resample
        pandas.DataFrame.resample
        Dataset.groupby
        DataArray.groupby

        References
        ----------
        .. [1] http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        """
        pass

    def drop_attrs(self, *, deep: bool=True) -> Self:
        """
        Removes all attributes from the Dataset and its variables.

        Parameters
        ----------
        deep : bool, default True
            Removes attributes from all variables.

        Returns
        -------
        Dataset
        """
        pass