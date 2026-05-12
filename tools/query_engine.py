"""JSONPath-like query engine for nested dict/list traversal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QueryEngine:
    def get(self, data: dict[str, Any] | list, path: str) -> Any | None:
        parts = path.strip(".").split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if 0 <= idx < len(current) else None
            else:
                return None
        return current

    def set(self, data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
        parts = path.strip(".").split(".")
        current = data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
        return data

    def exists(self, data: dict[str, Any], path: str) -> bool:
        return self.get(data, path) is not None


QUERY_ENGINE = QueryEngine()
