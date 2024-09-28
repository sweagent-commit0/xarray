"""Functions for converting to and from xarray objects
"""
from collections import Counter
import numpy as np
from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder
from xarray.conventions import decode_cf
from xarray.core import duck_array_ops
from xarray.core.dataarray import DataArray
from xarray.core.dtypes import get_fill_value
from xarray.namedarray.pycompat import array_type
iris_forbidden_keys = {'standard_name', 'long_name', 'units', 'bounds', 'axis', 'calendar', 'leap_month', 'leap_year', 'month_lengths', 'coordinates', 'grid_mapping', 'climatology', 'cell_methods', 'formula_terms', 'compress', 'missing_value', 'add_offset', 'scale_factor', 'valid_max', 'valid_min', 'valid_range', '_FillValue'}
cell_methods_strings = {'point', 'sum', 'maximum', 'median', 'mid_range', 'minimum', 'mean', 'mode', 'standard_deviation', 'variance'}

def _filter_attrs(attrs, ignored_attrs):
    """Return attrs that are not in ignored_attrs"""
    pass

def _pick_attrs(attrs, keys):
    """Return attrs with keys in keys list"""
    pass

def _get_iris_args(attrs):
    """Converts the xarray attrs into args that can be passed into Iris"""
    pass

def to_iris(dataarray):
    """Convert a DataArray into a Iris Cube"""
    pass

def _iris_obj_to_attrs(obj):
    """Return a dictionary of attrs when given a Iris object"""
    pass

def _iris_cell_methods_to_str(cell_methods_obj):
    """Converts a Iris cell methods into a string"""
    pass

def _name(iris_obj, default='unknown'):
    """Mimics `iris_obj.name()` but with different name resolution order.

    Similar to iris_obj.name() method, but using iris_obj.var_name first to
    enable roundtripping.
    """
    pass

def from_iris(cube):
    """Convert a Iris cube into an DataArray"""
    pass