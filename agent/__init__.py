# DEPRECATED: Use src.agent instead.
import warnings

warnings.warn(
    "Importing from 'agent' is deprecated. Use 'src.agent' instead.",
    DeprecationWarning,
    stacklevel=2
)
from src.agent import *