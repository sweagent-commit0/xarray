from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import _LINEWIDTH_RANGE, _MARKERSIZE_RANGE, _add_legend, _determine_guide, _get_nice_quiver_magnitude, _guess_coords_to_plot, _infer_xy_labels, _Normalize, _parse_size, _process_cmap_cbar_kwargs, label_from_attrs
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.quiver import QuiverKey
    from matplotlib.text import Annotation
    from xarray.core.dataarray import DataArray
_FONTSIZE = 'small'
_NTICKS = 5

def _nicetitle(coord, value, maxchar, template):
    """
    Put coord, value in template and truncate at maxchar
    """
    pass
T_FacetGrid = TypeVar('T_FacetGrid', bound='FacetGrid')

class FacetGrid(Generic[T_DataArrayOrSet]):
    """
    Initialize the Matplotlib figure and FacetGrid object.

    The :class:`FacetGrid` is an object that links a xarray DataArray to
    a Matplotlib figure with a particular structure.

    In particular, :class:`FacetGrid` is used to draw plots with multiple
    axes, where each axes shows the same relationship conditioned on
    different levels of some dimension. It's possible to condition on up to
    two variables by assigning variables to the rows and columns of the
    grid.

    The general approach to plotting here is called "small multiples",
    where the same kind of plot is repeated multiple times, and the
    specific use of small multiples to display the same relationship
    conditioned on one or more other variables is often called a "trellis
    plot".

    The basic workflow is to initialize the :class:`FacetGrid` object with
    the DataArray and the variable names that are used to structure the grid.
    Then plotting functions can be applied to each subset by calling
    :meth:`FacetGrid.map_dataarray` or :meth:`FacetGrid.map`.

    Attributes
    ----------
    axs : ndarray of matplotlib.axes.Axes
        Array containing axes in corresponding position, as returned from
        :py:func:`matplotlib.pyplot.subplots`.
    col_labels : list of matplotlib.text.Annotation
        Column titles.
    row_labels : list of matplotlib.text.Annotation
        Row titles.
    fig : matplotlib.figure.Figure
        The figure containing all the axes.
    name_dicts : ndarray of dict
        Array containing dictionaries mapping coordinate names to values. ``None`` is
        used as a sentinel value for axes that should remain empty, i.e.,
        sometimes the rightmost grid positions in the bottom row.
    """
    data: T_DataArrayOrSet
    name_dicts: np.ndarray
    fig: Figure
    axs: np.ndarray
    row_names: list[np.ndarray]
    col_names: list[np.ndarray]
    figlegend: Legend | None
    quiverkey: QuiverKey | None
    cbar: Colorbar | None
    _single_group: bool | Hashable
    _nrow: int
    _row_var: Hashable | None
    _ncol: int
    _col_var: Hashable | None
    _col_wrap: int | None
    row_labels: list[Annotation | None]
    col_labels: list[Annotation | None]
    _x_var: None
    _y_var: None
    _cmap_extend: Any | None
    _mappables: list[ScalarMappable]
    _finalized: bool

    def __init__(self, data: T_DataArrayOrSet, col: Hashable | None=None, row: Hashable | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, figsize: Iterable[float] | None=None, aspect: float=1, size: float=3, subplot_kws: dict[str, Any] | None=None) -> None:
        """
        Parameters
        ----------
        data : DataArray or Dataset
            DataArray or Dataset to be plotted.
        row, col : str
            Dimension names that define subsets of the data, which will be drawn
            on separate facets in the grid.
        col_wrap : int, optional
            "Wrap" the grid the for the column variable after this number of columns,
            adding rows if ``col_wrap`` is less than the number of facets.
        sharex : bool, optional
            If true, the facets will share *x* axes.
        sharey : bool, optional
            If true, the facets will share *y* axes.
        figsize : Iterable of float or None, optional
            A tuple (width, height) of the figure in inches.
            If set, overrides ``size`` and ``aspect``.
        aspect : scalar, default: 1
            Aspect ratio of each facet, so that ``aspect * size`` gives the
            width of each facet in inches.
        size : scalar, default: 3
            Height (in inches) of each facet. See also: ``aspect``.
        subplot_kws : dict, optional
            Dictionary of keyword arguments for Matplotlib subplots
            (:py:func:`matplotlib.pyplot.subplots`).

        """
        import matplotlib.pyplot as plt
        rep_col = col is not None and (not data[col].to_index().is_unique)
        rep_row = row is not None and (not data[row].to_index().is_unique)
        if rep_col or rep_row:
            raise ValueError('Coordinates used for faceting cannot contain repeated (nonunique) values.')
        single_group: bool | Hashable
        if col and row:
            single_group = False
            nrow = len(data[row])
            ncol = len(data[col])
            nfacet = nrow * ncol
            if col_wrap is not None:
                warnings.warn('Ignoring col_wrap since both col and row were passed')
        elif row and (not col):
            single_group = row
        elif not row and col:
            single_group = col
        else:
            raise ValueError('Pass a coordinate name as an argument for row or col')
        if single_group:
            nfacet = len(data[single_group])
            if col:
                ncol = nfacet
            if row:
                ncol = 1
            if col_wrap is not None:
                ncol = col_wrap
            nrow = int(np.ceil(nfacet / ncol))
        subplot_kws = {} if subplot_kws is None else subplot_kws
        if figsize is None:
            cbar_space = 1
            figsize = (ncol * size * aspect + cbar_space, nrow * size)
        fig, axs = plt.subplots(nrow, ncol, sharex=sharex, sharey=sharey, squeeze=False, figsize=figsize, subplot_kw=subplot_kws)
        col_names = list(data[col].to_numpy()) if col else []
        row_names = list(data[row].to_numpy()) if row else []
        if single_group:
            full: list[dict[Hashable, Any] | None] = [{single_group: x} for x in data[single_group].to_numpy()]
            empty: list[dict[Hashable, Any] | None] = [None for x in range(nrow * ncol - len(full))]
            name_dict_list = full + empty
        else:
            rowcols = itertools.product(row_names, col_names)
            name_dict_list = [{row: r, col: c} for r, c in rowcols]
        name_dicts = np.array(name_dict_list).reshape(nrow, ncol)
        self.data = data
        self.name_dicts = name_dicts
        self.fig = fig
        self.axs = axs
        self.row_names = row_names
        self.col_names = col_names
        self.figlegend = None
        self.quiverkey = None
        self.cbar = None
        self._single_group = single_group
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col
        self._col_wrap = col_wrap
        self.row_labels = [None] * nrow
        self.col_labels = [None] * ncol
        self._x_var = None
        self._y_var = None
        self._cmap_extend = None
        self._mappables = []
        self._finalized = False

    def map_dataarray(self: T_FacetGrid, func: Callable, x: Hashable | None, y: Hashable | None, **kwargs: Any) -> T_FacetGrid:
        """
        Apply a plotting function to a 2d facet's subset of the data.

        This is more convenient and less general than ``FacetGrid.map``

        Parameters
        ----------
        func : callable
            A plotting function with the same signature as a 2d xarray
            plotting method such as `xarray.plot.imshow`
        x, y : string
            Names of the coordinates to plot on x, y axes
        **kwargs
            additional keyword arguments to func

        Returns
        -------
        self : FacetGrid object

        """
        pass

    def map_plot1d(self: T_FacetGrid, func: Callable, x: Hashable | None, y: Hashable | None, *, z: Hashable | None=None, hue: Hashable | None=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, **kwargs: Any) -> T_FacetGrid:
        """
        Apply a plotting function to a 1d facet's subset of the data.

        This is more convenient and less general than ``FacetGrid.map``

        Parameters
        ----------
        func :
            A plotting function with the same signature as a 1d xarray
            plotting method such as `xarray.plot.scatter`
        x, y :
            Names of the coordinates to plot on x, y axes
        **kwargs
            additional keyword arguments to func

        Returns
        -------
        self : FacetGrid object

        """
        pass

    def _finalize_grid(self, *axlabels: Hashable) -> None:
        """Finalize the annotations and layout."""
        pass

    def add_colorbar(self, **kwargs: Any) -> None:
        """Draw a colorbar."""
        pass

    def _get_largest_lims(self) -> dict[str, tuple[float, float]]:
        """
        Get largest limits in the facetgrid.

        Returns
        -------
        lims_largest : dict[str, tuple[float, float]]
            Dictionary with the largest limits along each axis.

        Examples
        --------
        >>> ds = xr.tutorial.scatter_example_dataset(seed=42)
        >>> fg = ds.plot.scatter(x="A", y="B", hue="y", row="x", col="w")
        >>> round(fg._get_largest_lims()["x"][0], 3)
        np.float64(-0.334)
        """
        pass

    def _set_lims(self, x: tuple[float, float] | None=None, y: tuple[float, float] | None=None, z: tuple[float, float] | None=None) -> None:
        """
        Set the same limits for all the subplots in the facetgrid.

        Parameters
        ----------
        x : tuple[float, float] or None, optional
            x axis limits.
        y : tuple[float, float] or None, optional
            y axis limits.
        z : tuple[float, float] or None, optional
            z axis limits.

        Examples
        --------
        >>> ds = xr.tutorial.scatter_example_dataset(seed=42)
        >>> fg = ds.plot.scatter(x="A", y="B", hue="y", row="x", col="w")
        >>> fg._set_lims(x=(-0.3, 0.3), y=(0, 2), z=(0, 4))
        >>> fg.axs[0, 0].get_xlim(), fg.axs[0, 0].get_ylim()
        ((np.float64(-0.3), np.float64(0.3)), (np.float64(0.0), np.float64(2.0)))
        """
        pass

    def set_axis_labels(self, *axlabels: Hashable) -> None:
        """Set axis labels on the left column and bottom row of the grid."""
        pass

    def set_xlabels(self, label: None | str=None, **kwargs: Any) -> None:
        """Label the x axis on the bottom row of the grid."""
        pass

    def set_ylabels(self, label: None | str=None, **kwargs: Any) -> None:
        """Label the y axis on the left column of the grid."""
        pass

    def set_zlabels(self, label: None | str=None, **kwargs: Any) -> None:
        """Label the z axis."""
        pass

    def set_titles(self, template: str='{coord} = {value}', maxchar: int=30, size=None, **kwargs) -> None:
        """
        Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : str, default: "{coord} = {value}"
            Template for plot titles containing {coord} and {value}
        maxchar : int, default: 30
            Truncate titles at maxchar
        **kwargs : keyword args
            additional arguments to matplotlib.text

        Returns
        -------
        self: FacetGrid object

        """
        pass

    def set_ticks(self, max_xticks: int=_NTICKS, max_yticks: int=_NTICKS, fontsize: str | int=_FONTSIZE) -> None:
        """
        Set and control tick behavior.

        Parameters
        ----------
        max_xticks, max_yticks : int, optional
            Maximum number of labeled ticks to plot on x, y axes
        fontsize : string or int
            Font size as used by matplotlib text

        Returns
        -------
        self : FacetGrid object

        """
        pass

    def map(self: T_FacetGrid, func: Callable, *args: Hashable, **kwargs: Any) -> T_FacetGrid:
        """
        Apply a plotting function to each facet's subset of the data.

        Parameters
        ----------
        func : callable
            A plotting function that takes data and keyword arguments. It
            must plot to the currently active matplotlib Axes and take a
            `color` keyword argument. If faceting on the `hue` dimension,
            it must also take a `label` keyword argument.
        *args : Hashable
            Column names in self.data that identify variables with data to
            plot. The data for each variable is passed to `func` in the
            order the variables are specified in the call.
        **kwargs : keyword arguments
            All keyword arguments are passed to the plotting function.

        Returns
        -------
        self : FacetGrid object

        """
        pass

def _easy_facetgrid(data: T_DataArrayOrSet, plotfunc: Callable, kind: Literal['line', 'dataarray', 'dataset', 'plot1d'], x: Hashable | None=None, y: Hashable | None=None, row: Hashable | None=None, col: Hashable | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, aspect: float | None=None, size: float | None=None, subplot_kws: dict[str, Any] | None=None, ax: Axes | None=None, figsize: Iterable[float] | None=None, **kwargs: Any) -> FacetGrid[T_DataArrayOrSet]:
    """
    Convenience method to call xarray.plot.FacetGrid from 2d plotting methods

    kwargs are the arguments to 2d plotting method
    """
    pass