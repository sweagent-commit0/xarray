from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union
import numpy as np
import pandas as pd
from xarray.coding import strings, times, variables
from xarray.coding.variables import SerializationWarning, pop_to
from xarray.core import indexing
from xarray.core.common import _contains_datetime_like_objects, contains_cftime_datetimes
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import IndexVariable, Variable
from xarray.namedarray.utils import is_duck_dask_array
CF_RELATED_DATA = ('bounds', 'grid_mapping', 'climatology', 'geometry', 'node_coordinates', 'node_count', 'part_node_count', 'interior_ring', 'cell_measures', 'formula_terms')
CF_RELATED_DATA_NEEDS_PARSING = ('cell_measures', 'formula_terms')
if TYPE_CHECKING:
    from xarray.backends.common import AbstractDataStore
    from xarray.core.dataset import Dataset
    T_VarTuple = tuple[tuple[Hashable, ...], Any, dict, dict]
    T_Name = Union[Hashable, None]
    T_Variables = Mapping[Any, Variable]
    T_Attrs = MutableMapping[Any, Any]
    T_DropVariables = Union[str, Iterable[Hashable], None]
    T_DatasetOrAbstractstore = Union[Dataset, AbstractDataStore]

def _infer_dtype(array, name=None):
    """Given an object array with no missing values, infer its dtype from all elements."""
    pass

def _copy_with_dtype(data, dtype: np.typing.DTypeLike):
    """Create a copy of an array with the given dtype.

    We use this instead of np.array() to ensure that custom object dtypes end
    up on the resulting array.
    """
    pass

def encode_cf_variable(var: Variable, needs_copy: bool=True, name: T_Name=None) -> Variable:
    """
    Converts a Variable into a Variable which follows some
    of the CF conventions:

        - Nans are masked using _FillValue (or the deprecated missing_value)
        - Rescaling via: scale_factor and add_offset
        - datetimes are converted to the CF 'units since time' format
        - dtype encodings are enforced.

    Parameters
    ----------
    var : Variable
        A variable holding un-encoded data.

    Returns
    -------
    out : Variable
        A variable which has been encoded as described above.
    """
    pass

def decode_cf_variable(name: Hashable, var: Variable, concat_characters: bool=True, mask_and_scale: bool=True, decode_times: bool=True, decode_endianness: bool=True, stack_char_dim: bool=True, use_cftime: bool | None=None, decode_timedelta: bool | None=None) -> Variable:
    """
    Decodes a variable which may hold CF encoded information.

    This includes variables that have been masked and scaled, which
    hold CF style time variables (this is almost always the case if
    the dataset has been serialized) and which have strings encoded
    as character arrays.

    Parameters
    ----------
    name : str
        Name of the variable. Used for better error messages.
    var : Variable
        A variable holding potentially CF encoded information.
    concat_characters : bool
        Should character arrays be concatenated to strings, for
        example: ["h", "e", "l", "l", "o"] -> "hello"
    mask_and_scale : bool
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue). If the _Unsigned attribute is present
        treat integer arrays as unsigned.
    decode_times : bool
        Decode cf times ("hours since 2000-01-01") to np.datetime64.
    decode_endianness : bool
        Decode arrays from non-native to native endianness.
    stack_char_dim : bool
        Whether to stack characters into bytes along the last dimension of this
        array. Passed as an argument because we need to look at the full
        dataset to figure out if this is appropriate.
    use_cftime : bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.

    Returns
    -------
    out : Variable
        A variable holding the decoded equivalent of var.
    """
    pass

def _update_bounds_attributes(variables: T_Variables) -> None:
    """Adds time attributes to time bounds variables.

    Variables handling time bounds ("Cell boundaries" in the CF
    conventions) do not necessarily carry the necessary attributes to be
    decoded. This copies the attributes from the time variable to the
    associated boundaries.

    See Also:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/
         cf-conventions.html#cell-boundaries

    https://github.com/pydata/xarray/issues/2565
    """
    pass

def _update_bounds_encoding(variables: T_Variables) -> None:
    """Adds time encoding to time bounds variables.

    Variables handling time bounds ("Cell boundaries" in the CF
    conventions) do not necessarily carry the necessary attributes to be
    decoded. This copies the encoding from the time variable to the
    associated bounds variable so that we write CF-compliant files.

    See Also:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/
         cf-conventions.html#cell-boundaries

    https://github.com/pydata/xarray/issues/2565
    """
    pass
T = TypeVar('T')

def _item_or_default(obj: Mapping[Any, T] | T, key: Hashable, default: T) -> T:
    """
    Return item by key if obj is mapping and key is present, else return default value.
    """
    pass

def decode_cf_variables(variables: T_Variables, attributes: T_Attrs, concat_characters: bool | Mapping[str, bool]=True, mask_and_scale: bool | Mapping[str, bool]=True, decode_times: bool | Mapping[str, bool]=True, decode_coords: bool | Literal['coordinates', 'all']=True, drop_variables: T_DropVariables=None, use_cftime: bool | Mapping[str, bool] | None=None, decode_timedelta: bool | Mapping[str, bool] | None=None) -> tuple[T_Variables, T_Attrs, set[Hashable]]:
    """
    Decode several CF encoded variables.

    See: decode_cf_variable
    """
    pass

def decode_cf(obj: T_DatasetOrAbstractstore, concat_characters: bool=True, mask_and_scale: bool=True, decode_times: bool=True, decode_coords: bool | Literal['coordinates', 'all']=True, drop_variables: T_DropVariables=None, use_cftime: bool | None=None, decode_timedelta: bool | None=None) -> Dataset:
    """Decode the given Dataset or Datastore according to CF conventions into
    a new Dataset.

    Parameters
    ----------
    obj : Dataset or DataStore
        Object to decode.
    concat_characters : bool, optional
        Should character arrays be concatenated to strings, for
        example: ["h", "e", "l", "l", "o"] -> "hello"
    mask_and_scale : bool, optional
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue).
    decode_times : bool, optional
        Decode cf times (e.g., integers since "hours since 2000-01-01") to
        np.datetime64.
    decode_coords : bool or {"coordinates", "all"}, optional
        Controls which variables are set as coordinate variables:

        - "coordinates" or True: Set variables referred to in the
          ``'coordinates'`` attribute of the datasets or individual variables
          as coordinate variables.
        - "all": Set variables referred to in  ``'grid_mapping'``, ``'bounds'`` and
          other attributes as coordinate variables.
    drop_variables : str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    use_cftime : bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.

    Returns
    -------
    decoded : Dataset
    """
    pass

def cf_decoder(variables: T_Variables, attributes: T_Attrs, concat_characters: bool=True, mask_and_scale: bool=True, decode_times: bool=True) -> tuple[T_Variables, T_Attrs]:
    """
    Decode a set of CF encoded variables and attributes.

    Parameters
    ----------
    variables : dict
        A dictionary mapping from variable name to xarray.Variable
    attributes : dict
        A dictionary mapping from attribute name to value
    concat_characters : bool
        Should character arrays be concatenated to strings, for
        example: ["h", "e", "l", "l", "o"] -> "hello"
    mask_and_scale : bool
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue).
    decode_times : bool
        Decode cf times ("hours since 2000-01-01") to np.datetime64.

    Returns
    -------
    decoded_variables : dict
        A dictionary mapping from variable name to xarray.Variable objects.
    decoded_attributes : dict
        A dictionary mapping from attribute name to values.

    See Also
    --------
    decode_cf_variable
    """
    pass

def encode_dataset_coordinates(dataset: Dataset):
    """Encode coordinates on the given dataset object into variable specific
    and global attributes.

    When possible, this is done according to CF conventions.

    Parameters
    ----------
    dataset : Dataset
        Object to encode.

    Returns
    -------
    variables : dict
    attrs : dict
    """
    pass

def cf_encoder(variables: T_Variables, attributes: T_Attrs):
    """
    Encode a set of CF encoded variables and attributes.
    Takes a dicts of variables and attributes and encodes them
    to conform to CF conventions as much as possible.
    This includes masking, scaling, character array handling,
    and CF-time encoding.

    Parameters
    ----------
    variables : dict
        A dictionary mapping from variable name to xarray.Variable
    attributes : dict
        A dictionary mapping from attribute name to value

    Returns
    -------
    encoded_variables : dict
        A dictionary mapping from variable name to xarray.Variable,
    encoded_attributes : dict
        A dictionary mapping from attribute name to value

    See Also
    --------
    decode_cf_variable, encode_cf_variable
    """
    pass