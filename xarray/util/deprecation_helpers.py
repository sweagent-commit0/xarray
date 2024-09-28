import inspect
import warnings
from functools import wraps
from typing import Callable, TypeVar
from xarray.core.utils import emit_user_level_warning
T = TypeVar('T', bound=Callable)
POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
EMPTY = inspect.Parameter.empty

def _deprecate_positional_args(version) -> Callable[[T], T]:
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    ``*`` will issue a warning when passed as a positional argument.

    Parameters
    ----------
    version : str
        version of the library when the positional arguments were deprecated

    Examples
    --------
    Deprecate passing `b` as positional argument:

    def func(a, b=1):
        pass

    @_deprecate_positional_args("v0.1.0")
    def func(a, *, b=2):
        pass

    func(1, 2)

    Notes
    -----
    This function is adapted from scikit-learn under the terms of its license. See
    licences/SCIKIT_LEARN_LICENSE
    """
    pass

def deprecate_dims(func: T, old_name='dims') -> T:
    """
    For functions that previously took `dims` as a kwarg, and have now transitioned to
    `dim`. This decorator will issue a warning if `dims` is passed while forwarding it
    to `dim`.
    """
    pass