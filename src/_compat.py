"""Python-version compatibility shims."""

import sys

if sys.version_info >= (3, 11):  # noqa: UP036 - keep Python 3.9 compatibility
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042 - backport for Python < 3.11
        """Backport of Python 3.11 StrEnum."""

        def __str__(self) -> str:
            return self.value
