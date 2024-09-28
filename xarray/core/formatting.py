"""String formatting routines for __repr__.
"""
from __future__ import annotations
import contextlib
import functools
import math
from collections import defaultdict
from collections.abc import Collection, Hashable, Sequence
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from reprlib import recursive_repr
from textwrap import dedent
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from xarray.core.datatree_render import RenderDataTree
from xarray.core.duck_array_ops import array_equiv, astype
from xarray.core.indexing import MemoryCachedArray
from xarray.core.iterators import LevelOrderIter
from xarray.core.options import OPTIONS, _get_boolean_with_default
from xarray.core.utils import is_duck_array
from xarray.namedarray.pycompat import array_type, to_duck_array, to_numpy
if TYPE_CHECKING:
    from xarray.core.coordinates import AbstractCoordinates
    from xarray.core.datatree import DataTree
UNITS = ('B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')

def pretty_print(x, numchars: int):
    """Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    pass

def first_n_items(array, n_desired):
    """Returns the first n_desired items of an array"""
    pass

def last_n_items(array, n_desired):
    """Returns the last n_desired items of an array"""
    pass

def last_item(array):
    """Returns the last item of an array in a list or an empty list."""
    pass

def calc_max_rows_first(max_rows: int) -> int:
    """Calculate the first rows to maintain the max number of rows."""
    pass

def calc_max_rows_last(max_rows: int) -> int:
    """Calculate the last rows to maintain the max number of rows."""
    pass

def format_timestamp(t):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    pass

def format_timedelta(t, timedelta_format=None):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    pass

def format_item(x, timedelta_format=None, quote_strings=True):
    """Returns a succinct summary of an object as a string"""
    pass

def format_items(x):
    """Returns a succinct summaries of all items in a sequence as strings"""
    pass

def format_array_flat(array, max_width: int):
    """Return a formatted string for as many items in the flattened version of
    array that will fit within max_width characters.
    """
    pass
_KNOWN_TYPE_REPRS = {('numpy', 'ndarray'): 'np.ndarray', ('sparse._coo.core', 'COO'): 'sparse.COO'}

def inline_dask_repr(array):
    """Similar to dask.array.DataArray.__repr__, but without
    redundant information that's already printed by the repr
    function of the xarray wrapper.
    """
    pass

def inline_sparse_repr(array):
    """Similar to sparse.COO.__repr__, but without the redundant shape/dtype."""
    pass

def inline_variable_array_repr(var, max_width):
    """Build a one-line summary of a variable's data."""
    pass

def summarize_variable(name: Hashable, var, col_width: int, max_width: int | None=None, is_index: bool=False):
    """Summarize a variable in one line, e.g., for the Dataset.__repr__."""
    pass

def summarize_attr(key, value, col_width=None):
    """Summary for __repr__ - use ``X.attrs[key]`` for full value."""
    pass
EMPTY_REPR = '    *empty*'
data_vars_repr = functools.partial(_mapping_repr, title='Data variables', summarizer=summarize_variable, expand_option_name='display_expand_data_vars')
attrs_repr = functools.partial(_mapping_repr, title='Attributes', summarizer=summarize_attr, expand_option_name='display_expand_attrs')

def _element_formatter(elements: Collection[Hashable], col_width: int, max_rows: int | None=None, delimiter: str=', ') -> str:
    """
    Formats elements for better readability.

    Once it becomes wider than the display width it will create a newline and
    continue indented to col_width.
    Once there are more rows than the maximum displayed rows it will start
    removing rows.

    Parameters
    ----------
    elements : Collection of hashable
        Elements to join together.
    col_width : int
        The width to indent to if a newline has been made.
    max_rows : int, optional
        The maximum number of allowed rows. The default is None.
    delimiter : str, optional
        Delimiter to use between each element. The default is ", ".
    """
    pass

def limit_lines(string: str, *, limit: int):
    """
    If the string is more lines than the limit,
    this returns the middle lines replaced by an ellipsis
    """
    pass

def short_data_repr(array):
    """Format "data" for DataArray and Variable."""
    pass

def dims_and_coords_repr(ds) -> str:
    """Partial Dataset repr for use inside DataTree inheritance errors."""
    pass
diff_data_vars_repr = functools.partial(_diff_mapping_repr, title='Data variables', summarizer=summarize_variable)
diff_attrs_repr = functools.partial(_diff_mapping_repr, title='Attributes', summarizer=summarize_attr)

def diff_treestructure(a: DataTree, b: DataTree, require_names_equal: bool) -> str:
    """
    Return a summary of why two trees are not isomorphic.
    If they are isomorphic return an empty string.
    """
    pass

def diff_nodewise_summary(a: DataTree, b: DataTree, compat):
    """Iterates over all corresponding nodes, recording differences between data at each location."""
    pass

def _single_node_repr(node: DataTree) -> str:
    """Information about this node, not including its relationships to other nodes."""
    pass

def datatree_repr(dt: DataTree):
    """A printable representation of the structure of this entire tree."""
    pass

def render_human_readable_nbytes(nbytes: int, /, *, attempt_constant_width: bool=False) -> str:
    """Renders simple human-readable byte count representation

    This is only a quick representation that should not be relied upon for precise needs.

    To get the exact byte count, please use the ``nbytes`` attribute directly.

    Parameters
    ----------
    nbytes
        Byte count
    attempt_constant_width
        For reasonable nbytes sizes, tries to render a fixed-width representation.

    Returns
    -------
        Human-readable representation of the byte count
    """
    pass