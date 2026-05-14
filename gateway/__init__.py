# DEPRECATED: Use src.serving instead.
import warnings

warnings.warn(
    "Importing from 'gateway' is deprecated. Use 'src.serving' instead.",
    DeprecationWarning,
    stacklevel=2
)
from src.serving import *