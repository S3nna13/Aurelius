"""Python-version compatibility shims."""
import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum
    class StrEnum(str, Enum):
        """Backport of Python 3.11 StrEnum."""
        def __str__(self) -> str:
            return self.value
