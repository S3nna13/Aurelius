from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OllamaConfig",
    "OllamaAdapter",
    "OLLAMA_REGISTRY",
]


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3"
    timeout_s: float = 60.0
    stream: bool = False


class OllamaAdapter:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self._config = config if config is not None else OllamaConfig()

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        cfg = self._config
        body = json.dumps({
            "model": cfg.model,
            "prompt": prompt,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{cfg.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                data = json.loads(resp.read())
            return data["response"]
        except Exception as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}") from exc

    def list_models(self) -> list[str]:
        cfg = self._config
        req = urllib.request.Request(f"{cfg.host}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}") from exc

    def is_available(self) -> bool:
        cfg = self._config
        req = urllib.request.Request(f"{cfg.host}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                return resp.status == 200
        except Exception:
            return False


OLLAMA_REGISTRY: dict[str, type[OllamaAdapter]] = {"ollama": OllamaAdapter}
