"""Tool chain validator with dependency and cycle detection."""
from __future__ import annotations

import re
from dataclasses import dataclass


class ToolChainError(Exception):
    """Raised when a tool chain validation fails in strict mode."""


@dataclass
class ToolChainValidator:
    """Validate chains of tool calls for structural correctness and acyclicity."""

    _name_re: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_-]+$")
    _ref_pattern: re.Pattern[str] = re.compile(r"\$\{?([a-zA-Z0-9_-]+)\}?")

    def validate_chain(
        self,
        tools: list[dict],
        *,
        allow_duplicates: bool = False,
    ) -> list[str]:
        """Validate a chain of tool calls.

        Returns a list of error strings; an empty list means the chain is valid.
        """
        errors: list[str] = []

        if len(tools) > 32:
            errors.append(f"chain exceeds maximum length of 32 (got {len(tools)})")

        for idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                errors.append(f"tool at index {idx} is not a dict")
                continue
            if "name" not in tool:
                errors.append(f"tool at index {idx} missing required key 'name'")
            if "args" not in tool:
                errors.append(f"tool at index {idx} missing required key 'args'")

        if any("missing required key" in e or "not a dict" in e for e in errors):
            return errors

        names_seen: set[str] = set()
        for idx, tool in enumerate(tools):
            name = tool["name"]
            if not isinstance(name, str):
                errors.append(
                    f"tool at index {idx} name must be a string, got {type(name).__name__}"
                )
                continue
            if not name:
                errors.append(f"tool at index {idx} name must be non-empty")
                continue
            if len(name) > 64:
                errors.append(f"tool at index {idx} name exceeds 64 characters")
                continue
            if not self._name_re.match(name):
                errors.append(f"tool at index {idx} name contains invalid characters: {name!r}")
                continue
            if not allow_duplicates and name in names_seen:
                errors.append(f"duplicate tool name in chain: {name!r}")
            names_seen.add(name)

        if any("name" in e for e in errors):
            return errors

        if not self.check_acyclic(tools):
            errors.append("circular dependency detected in tool chain")

        return errors

    def check_acyclic(self, chain: list[dict]) -> bool:
        """Return True if the chain contains no circular dependencies."""
        graph: dict[int, set[int]] = {}
        name_to_idx: dict[str, int] = {}

        for idx, tool in enumerate(chain):
            name = tool.get("name")
            if isinstance(name, str):
                name_to_idx[name] = idx
            graph[idx] = set()

        for idx, tool in enumerate(chain):
            args = tool.get("args", {})
            if not isinstance(args, dict):
                continue

            refs: set[int] = set()
            for val in args.values():
                if isinstance(val, str):
                    for match in self._ref_pattern.finditer(val):
                        ref_name = match.group(1)
                        if ref_name in name_to_idx:
                            refs.add(name_to_idx[ref_name])

            graph[idx] = refs

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {idx: WHITE for idx in graph}

        def dfs(node: int) -> bool:
            color[node] = GRAY
            for neighbor in graph.get(node, set()):
                if color.get(neighbor, BLACK) == GRAY:
                    return False
                if color.get(neighbor, BLACK) == WHITE:
                    if not dfs(neighbor):
                        return False
            color[node] = BLACK
            return True

        for idx in graph:
            if color[idx] == WHITE:
                if not dfs(idx):
                    return False

        return True


TOOL_CHAIN_VALIDATOR_REGISTRY = ToolChainValidator()
