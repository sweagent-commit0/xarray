from __future__ import annotations
import functools
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Literal, NoReturn, overload
import numpy as np
from xarray.plot import dataarray_plot, dataset_plot
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection, PathCollection, QuadMesh
    from matplotlib.colors import Normalize
    from matplotlib.container import BarContainer
    from matplotlib.contour import QuadContourSet
    from matplotlib.image import AxesImage
    from matplotlib.patches import Polygon
    from matplotlib.quiver import Quiver
    from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
    from numpy.typing import ArrayLike
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.types import AspectOptions, HueStyleOptions, ScaleOptions
    from xarray.plot.facetgrid import FacetGrid

class DataArrayPlotAccessor:
    """
    Enables use of xarray.plot functions as attributes on a DataArray.
    For example, DataArray.plot.imshow
    """
    _da: DataArray
    __slots__ = ('_da',)
    __doc__ = dataarray_plot.plot.__doc__

    def __init__(self, darray: DataArray) -> None:
        self._da = darray

    @functools.wraps(dataarray_plot.plot, assigned=('__doc__', '__annotations__'))
    def __call__(self, **kwargs) -> Any:
        return dataarray_plot.plot(self._da, **kwargs)

class DatasetPlotAccessor:
    """
    Enables use of xarray.plot functions as attributes on a Dataset.
    For example, Dataset.plot.scatter
    """
    _ds: Dataset
    __slots__ = ('_ds',)

    def __init__(self, dataset: Dataset) -> None:
        self._ds = dataset

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError('Dataset.plot cannot be called directly. Use an explicit plot method, e.g. ds.plot.scatter(...)')