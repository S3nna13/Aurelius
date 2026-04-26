"""Semantic memory: concept graph, relation store, lookup."""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum


class RelationType(StrEnum):
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSES = "causes"


@dataclass
class Concept:
    """A named concept node in the semantic graph."""

    name: str
    description: str = ""
    attributes: dict = field(default_factory=dict)


@dataclass
class Relation:
    """A directed, typed edge between two concepts."""

    source: str
    relation_type: RelationType
    target: str
    weight: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class SemanticMemory:
    """Concept graph with relation store and path-finding."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._relations: list[Relation] = []

    # ------------------------------------------------------------------
    # Concepts
    # ------------------------------------------------------------------

    def add_concept(self, name: str, description: str = "", **attributes) -> Concept:
        """Register a concept (idempotent: returns existing if already present)."""
        if name not in self._concepts:
            self._concepts[name] = Concept(
                name=name, description=description, attributes=dict(attributes)
            )
        return self._concepts[name]

    def get_concept(self, name: str) -> Concept | None:
        """Return Concept by name, or None if not found."""
        return self._concepts.get(name)

    def concept_count(self) -> int:
        return len(self._concepts)

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def add_relation(
        self,
        source: str,
        relation_type: RelationType,
        target: str,
        weight: float = 1.0,
    ) -> Relation:
        """Add a directed relation. Source and target must be registered concepts."""
        if source not in self._concepts:
            raise ValueError(f"Unknown source concept: {source!r}")
        if target not in self._concepts:
            raise ValueError(f"Unknown target concept: {target!r}")
        rel = Relation(source=source, relation_type=relation_type, target=target, weight=weight)
        self._relations.append(rel)
        return rel

    def get_relations(self, concept_name: str) -> list[Relation]:
        """Return all relations where concept_name is source OR target."""
        return [r for r in self._relations if r.source == concept_name or r.target == concept_name]

    def relation_count(self) -> int:
        return len(self._relations)

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def neighbors(self, concept_name: str) -> set[str]:
        """Return concept names reachable from concept_name via any relation (both directions)."""
        result: set[str] = set()
        for r in self._relations:
            if r.source == concept_name:
                result.add(r.target)
            elif r.target == concept_name:
                result.add(r.source)
        return result

    def find_path(self, start: str, end: str, max_depth: int = 5) -> list[str] | None:
        """BFS shortest path from start to end. Returns [start, ..., end] or None."""
        if start == end:
            return [start]
        visited: set[str] = {start}
        queue: deque[list[str]] = deque([[start]])
        while queue:
            path = queue.popleft()
            if len(path) > max_depth:
                continue
            current = path[-1]
            for neighbor in self.neighbors(current):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None
