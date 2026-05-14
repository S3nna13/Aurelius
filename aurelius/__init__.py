# DEPRECATED: Use src directly.
import warnings
import os

warnings.warn(
    "Importing from 'aurelius' is deprecated. Use 'src' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Expose src as a namespace package by extending __path__ to include the
# top-level ``src`` directory. This allows ``import aurelius.xxx`` to
# resolve to ``src/xxx`` without eagerly importing all subpackages.
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src_path not in __path__:
    __path__.append(_src_path)
