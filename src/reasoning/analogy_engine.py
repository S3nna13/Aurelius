from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Concept:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Analogy:
    source: Concept
    target: Concept
    score: float = 0.0


class AnalogyEngine:
    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}

    def add_concept(self, concept: Concept) -> None:
        self._concepts[concept.name] = concept

    def find_analogies(self, query: Concept, top_k: int = 5) -> list[Analogy]:
        scored: list[Analogy] = []
        for name, concept in self._concepts.items():
            if name == query.name:
                continue
            if not query.attributes or not concept.attributes:
                continue
            common = set(query.attributes.keys()) & set(concept.attributes.keys())
            if not common:
                continue
            total = 0.0
            for attr in common:
                qv = query.attributes[attr]
                cv = concept.attributes[attr]
                if isinstance(qv, bool) and isinstance(cv, bool):
                    total += 1.0 if qv == cv else 0.0
                else:
                    total += 1.0 if qv == cv else 0.0
            score = total / max(len(query.attributes), 1)
            scored.append(Analogy(source=query, target=concept, score=score))
        scored.sort(key=lambda a: a.score, reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self._concepts.clear()


ANALOGY_ENGINE = AnalogyEngine()
