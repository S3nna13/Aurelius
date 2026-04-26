from __future__ import annotations

from typing import Any


class OnnxBackend:
    name = "onnx"

    def health(self) -> bool:
        return False

    def capabilities(self) -> list[str]:
        return ["text_classification", "token_classification"]

    def configure(self, **kwargs: Any) -> None:
        pass


ONNX_BACKEND = OnnxBackend()
