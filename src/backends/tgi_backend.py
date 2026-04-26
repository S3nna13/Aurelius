from __future__ import annotations

from typing import Any


class TgiBackend:
    name = "tgi"

    def __init__(self) -> None:
        self._endpoint: str | None = None

    def health(self) -> bool:
        return False

    def capabilities(self) -> list[str]:
        return ["text_generation", "streaming"]

    def configure(self, **kwargs: Any) -> None:
        if "endpoint" in kwargs:
            self._endpoint = kwargs["endpoint"]


TGI_BACKEND = TgiBackend()
