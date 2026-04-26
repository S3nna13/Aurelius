"""Dependency resolver for agent plugins.

Provides topological ordering of plugin dependencies with cycle detection.
"""

from __future__ import annotations

from dataclasses import dataclass


class DependencyCycleError(Exception):
    """Raised when a circular dependency is detected."""


@dataclass
class DependencyResolver:
    """Resolves plugin dependencies into topological load order."""

    def __init__(self) -> None:
        self._dependencies: dict[str, list[str]] = {}
        self._resolved: dict[str, list[str]] = {}

    def register(self, plugin_id: str, dependencies: list[str]) -> None:
        """Register a plugin and its direct dependencies.

        Args:
            plugin_id: Unique identifier for the plugin.
            dependencies: List of plugin IDs this plugin depends on.

        Raises:
            DependencyCycleError: If the plugin depends on itself.
            ValueError: If plugin_id or any dependency is not a non-empty string.
        """
        if not isinstance(plugin_id, str) or not plugin_id:
            raise ValueError("plugin_id must be a non-empty string")
        for dep in dependencies:
            if not isinstance(dep, str) or not dep:
                raise ValueError("all dependencies must be non-empty strings")
        if plugin_id in dependencies:
            raise DependencyCycleError(f"Plugin '{plugin_id}' cannot depend on itself")
        self._dependencies[plugin_id] = list(dependencies)
        # Invalidate cache for any resolved plugin that might be affected.
        keys_to_remove = [
            key
            for key, resolved in self._resolved.items()
            if plugin_id in resolved or key == plugin_id
        ]
        for key in keys_to_remove:
            del self._resolved[key]

    def resolve(self, plugin_id: str) -> list[str]:
        """Return topological load order for *plugin_id* (deps first, then self).

        Args:
            plugin_id: The plugin to resolve.

        Returns:
            Ordered list of plugin IDs with dependencies before dependents.

        Raises:
            DependencyCycleError: If a circular dependency is detected.
        """
        if plugin_id in self._resolved:
            return self._resolved[plugin_id]

        result: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def _visit(node: str, path: list[str]) -> None:
            if node in visiting:
                cycle_start = path.index(node)
                cycle = " -> ".join(path[cycle_start:] + [node])
                raise DependencyCycleError(f"Circular dependency detected: {cycle}")
            if node in visited:
                return
            visiting.add(node)
            path.append(node)
            for dep in self._dependencies.get(node, []):
                _visit(dep, path)
            path.pop()
            visiting.remove(node)
            visited.add(node)
            result.append(node)

        _visit(plugin_id, [])
        self._resolved[plugin_id] = result
        return result

    def resolve_batch(self, plugin_ids: list[str]) -> list[str]:
        """Resolve multiple plugins into a single topological order.

        Args:
            plugin_ids: List of plugin IDs to resolve.

        Returns:
            Ordered list of plugin IDs with dependencies before dependents.

        Raises:
            DependencyCycleError: If a circular dependency is detected.
        """
        visiting: set[str] = set()
        visited: set[str] = set()
        result: list[str] = []

        def _visit(node: str, path: list[str]) -> None:
            if node in visiting:
                cycle_start = path.index(node)
                cycle = " -> ".join(path[cycle_start:] + [node])
                raise DependencyCycleError(f"Circular dependency detected: {cycle}")
            if node in visited:
                return
            visiting.add(node)
            path.append(node)
            for dep in self._dependencies.get(node, []):
                _visit(dep, path)
            path.pop()
            visiting.remove(node)
            visited.add(node)
            result.append(node)

        for pid in plugin_ids:
            _visit(pid, [])
        return result

    def get_direct_dependencies(self, plugin_id: str) -> list[str]:
        """Return the direct dependencies registered for *plugin_id*."""
        return list(self._dependencies.get(plugin_id, []))

    def clear(self) -> None:
        """Clear all registered dependencies and cached resolutions."""
        self._dependencies.clear()
        self._resolved.clear()

    def has_dependency(self, plugin_id: str, dependency_id: str) -> bool:
        """Check if *plugin_id* directly depends on *dependency_id*."""
        return dependency_id in self._dependencies.get(plugin_id, [])


DEFAULT_DEPENDENCY_RESOLVER = DependencyResolver()
DEPENDENCY_RESOLVER_REGISTRY: dict[str, DependencyResolver] = {
    "default": DEFAULT_DEPENDENCY_RESOLVER,
}

__all__ = [
    "DEFAULT_DEPENDENCY_RESOLVER",
    "DEPENDENCY_RESOLVER_REGISTRY",
    "DependencyCycleError",
    "DependencyResolver",
]
