"""
Useful for:

* users learning xarray
* building tutorials in the documentation.

"""
from __future__ import annotations
import os
import pathlib
from typing import TYPE_CHECKING
import numpy as np
from xarray.backends.api import open_dataset as _open_dataset
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
if TYPE_CHECKING:
    from xarray.backends.api import T_Engine
_default_cache_dir_name = 'xarray_tutorial_data'
base_url = 'https://github.com/pydata/xarray-data'
version = 'master'
external_urls = {}
file_formats = {'air_temperature': 3, 'air_temperature_gradient': 4, 'ASE_ice_velocity': 4, 'basin_mask': 4, 'ersstv5': 4, 'rasm': 3, 'ROMS_example': 4, 'tiny': 3, 'eraint_uvz': 3}

def open_dataset(name: str, cache: bool=True, cache_dir: None | str | os.PathLike=None, *, engine: T_Engine=None, **kws) -> Dataset:
    """
    Open a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Available datasets:

    * ``"air_temperature"``: NCEP reanalysis subset
    * ``"air_temperature_gradient"``: NCEP reanalysis subset with approximate x,y gradients
    * ``"basin_mask"``: Dataset with ocean basins marked using integers
    * ``"ASE_ice_velocity"``: MEaSUREs InSAR-Based Ice Velocity of the Amundsen Sea Embayment, Antarctica, Version 1
    * ``"rasm"``: Output of the Regional Arctic System Model (RASM)
    * ``"ROMS_example"``: Regional Ocean Model System (ROMS) output
    * ``"tiny"``: small synthetic dataset with a 1D data variable
    * ``"era5-2mt-2019-03-uk.grib"``: ERA5 temperature data over the UK
    * ``"eraint_uvz"``: data from ERA-Interim reanalysis, monthly averages of upper level data
    * ``"ersstv5"``: NOAA's Extended Reconstructed Sea Surface Temperature monthly averages

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
        e.g. 'air_temperature'
    cache_dir : path-like, optional
        The directory in which to search for and write cached data.
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    **kws : dict, optional
        Passed to xarray.open_dataset

    See Also
    --------
    tutorial.load_dataset
    open_dataset
    load_dataset
    """
    pass

def load_dataset(*args, **kwargs) -> Dataset:
    """
    Open, load into memory, and close a dataset from the online repository
    (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Available datasets:

    * ``"air_temperature"``: NCEP reanalysis subset
    * ``"air_temperature_gradient"``: NCEP reanalysis subset with approximate x,y gradients
    * ``"basin_mask"``: Dataset with ocean basins marked using integers
    * ``"rasm"``: Output of the Regional Arctic System Model (RASM)
    * ``"ROMS_example"``: Regional Ocean Model System (ROMS) output
    * ``"tiny"``: small synthetic dataset with a 1D data variable
    * ``"era5-2mt-2019-03-uk.grib"``: ERA5 temperature data over the UK
    * ``"eraint_uvz"``: data from ERA-Interim reanalysis, monthly averages of upper level data
    * ``"ersstv5"``: NOAA's Extended Reconstructed Sea Surface Temperature monthly averages

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
        e.g. 'air_temperature'
    cache_dir : path-like, optional
        The directory in which to search for and write cached data.
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    **kws : dict, optional
        Passed to xarray.open_dataset

    See Also
    --------
    tutorial.open_dataset
    open_dataset
    load_dataset
    """
    pass

def scatter_example_dataset(*, seed: None | int=None) -> Dataset:
    """
    Create an example dataset.

    Parameters
    ----------
    seed : int, optional
        Seed for the random number generation.
    """
    pass