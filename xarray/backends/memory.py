from __future__ import annotations
import copy
import numpy as np
from xarray.backends.common import AbstractWritableDataStore
from xarray.core.variable import Variable

class InMemoryDataStore(AbstractWritableDataStore):
    """
    Stores dimensions, variables and attributes in ordered dictionaries, making
    this store fast compared to stores which save to disk.

    This store exists purely for internal testing purposes.
    """

    def __init__(self, variables=None, attributes=None):
        self._variables = {} if variables is None else variables
        self._attributes = {} if attributes is None else attributes