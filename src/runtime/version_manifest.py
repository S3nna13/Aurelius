"""Runtime version manifest for system health reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VersionManifest:
    """Record runtime and dependency versions for diagnostics."""

    python_version: str = ""
    torch_version: str = ""
    numpy_version: str = ""
    aurelius_version: str = "0.0.0"
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python_version,
            "torch": self.torch_version,
            "numpy": self.numpy_version,
            "aurelius": self.aurelius_version,
            **self.extra,
        }

    @classmethod
    def detect(cls) -> VersionManifest:
        import sys

        v = cls(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        try:
            import torch

            v.torch_version = torch.__version__
        except ImportError:
            pass
        try:
            import numpy as np

            v.numpy_version = np.__version__
        except ImportError:
            pass
        return v


VERSION_MANIFEST = VersionManifest.detect()
