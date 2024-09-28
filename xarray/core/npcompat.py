from __future__ import annotations
from typing import Any
try:
    from numpy import isdtype
except ImportError:
    import numpy as np
    from numpy.typing import DTypeLike
    kind_mapping = {'bool': np.bool_, 'signed integer': np.signedinteger, 'unsigned integer': np.unsignedinteger, 'integral': np.integer, 'real floating': np.floating, 'complex floating': np.complexfloating, 'numeric': np.number}