#!/usr/bin/env python3
"""Detect import cycles across the Aurelius codebase.

This script scans the repository Python modules, resolves internal imports,
and fails if any strongly connected dependency cycle is found.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_GLOBS = (
    "agent_core.py",
    "agent_loop.py",
    "agent_registry.py",
    "agent_train.py",
    "api_server.py",
    "api_registry.py",
    "brain_layer.py",
    "aurelius/__init__.py",
    "aurelius_model.py",
    "aurelius_model_1b.py",
    "aurelius_model_3b.py",
    "aurelius_model_7b.py",
    "aurelius_model_14b.py",
    "aurelius_model_32b.py",
    "distributed.py",
    "fused_kernels.py",
    "inference.py",
    "mobile_inference.py",
    "memory_core.py",
    "registry_snapshot.py",
    "rlhf_lora.py",
    "rust_bridge.py",
    "skills.py",
    "skills_registry.py",
    "train.py",
    "train_3b.py",
    "train_7b.py",
    "train_optimized.py",
    "unified_manager.py",
)


def should_include(path: Path) -> bool:
    return True


def module_name_for(path: Path) -> str | None:
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    if not parts:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = path.stem
    if not parts:
        return None
    return ".".join(parts)


def build_module_index(paths: Iterable[Path]) -> dict[str, Path]:
    module_index: dict[str, Path] = {}
    for path in paths:
        module = module_name_for(path)
        if module is None:
            continue
        module_index[module] = path
        if module.startswith("src."):
            module_index.setdefault(module.replace("src.", "aurelius.", 1), path)
        elif module not in {"aurelius"}:
            module_index.setdefault(f"aurelius.{module}", path)
    return module_index


def resolve_relative(
    module: str,
    current_module: str,
    level: int,
    *,
    is_package: bool,
) -> str | None:
    package_parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    if level > len(package_parts) + 1:
        return None
    base_parts = package_parts[: len(package_parts) - (level - 1)]
    if module:
        base_parts = base_parts + module.split(".")
    if not base_parts:
        return None
    return ".".join(base_parts)


def collect_dependencies(path: Path, module_index: dict[str, Path]) -> set[str]:
    module = module_name_for(path)
    if module is None:
        return set()
    is_package = path.name == "__init__.py"

    tree = ast.parse(path.read_text())
    deps: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidate = alias.name
                if candidate in module_index:
                    deps.add(candidate)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                candidate_base = resolve_relative(
                    node.module or "",
                    module,
                    node.level,
                    is_package=is_package,
                )
            else:
                candidate_base = node.module
            if not candidate_base:
                continue

            if candidate_base in module_index:
                deps.add(candidate_base)

            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{candidate_base}.{alias.name}"
                if candidate in module_index:
                    deps.add(candidate)

    return deps


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    visited: set[str] = set()
    stack: list[str] = []
    stack_index: dict[str, int] = {}
    cycles: list[list[str]] = []
    seen_cycle_keys: set[tuple[str, ...]] = set()

    def record_cycle(start_idx: int, node: str) -> None:
        cycle = stack[start_idx:] + [node]
        if len(cycle) < 2:
            return
        key = tuple(cycle)
        if key not in seen_cycle_keys:
            seen_cycle_keys.add(key)
            cycles.append(cycle)

    def dfs(node: str) -> None:
        visited.add(node)
        stack_index[node] = len(stack)
        stack.append(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in stack_index:
                record_cycle(stack_index[neighbor], neighbor)
        stack.pop()
        stack_index.pop(node, None)

    for node in sorted(graph):
        if node not in visited:
            dfs(node)

    return cycles


def main() -> int:
    paths = [
        ROOT / rel
        for rel in TARGET_GLOBS
        if (ROOT / rel).exists() and should_include(ROOT / rel)
    ]
    module_index = build_module_index(paths)
    graph = {
        module: collect_dependencies(path, module_index)
        for module, path in module_index.items()
    }
    cycles = find_cycles(graph)
    if cycles:
        print("Import cycles detected:")
        for cycle in cycles:
            print(" -> ".join(cycle))
        return 1

    print(f"No import cycles detected across {len(graph)} modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
