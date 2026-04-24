from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class JSONToolConfig:
    indent: int = 2
    sort_keys: bool = False


class JSONTool:
    def __init__(self, config: JSONToolConfig | None = None) -> None:
        self.config = config or JSONToolConfig()

    def validate(self, text: str) -> tuple[bool, str]:
        try:
            json.loads(text)
            return True, ""
        except json.JSONDecodeError as e:
            return False, str(e)

    def format(self, text: str) -> str:
        data = json.loads(text)
        return json.dumps(data, indent=self.config.indent, sort_keys=self.config.sort_keys)

    def extract_path(self, data: dict | list, path: str) -> object:
        current = data
        for part in path.split("."):
            if isinstance(current, list):
                current = current[int(part)]
            else:
                current = current[part]
        return current

    def merge(self, base: dict, override: dict, deep: bool = True) -> dict:
        if not deep:
            return {**base, **override}
        result = dict(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = self.merge(result[key], val, deep=True)
            else:
                result[key] = val
        return result

    def diff_keys(self, a: dict, b: dict) -> dict:
        a_keys = set(a)
        b_keys = set(b)
        added = sorted(b_keys - a_keys)
        removed = sorted(a_keys - b_keys)
        changed = sorted(k for k in a_keys & b_keys if a[k] != b[k])
        return {"added": added, "removed": removed, "changed": changed}


JSON_TOOL_REGISTRY: dict[str, type[JSONTool]] = {"default": JSONTool}
