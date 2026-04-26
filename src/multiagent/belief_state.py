"""Agent belief state tracking via a probabilistic world model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Proposition:
    name: str
    probability: float
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class BeliefUpdate:
    proposition_name: str
    old_probability: float
    new_probability: float
    reason: str


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class BeliefState:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._props: dict[str, Proposition] = {}

    def believe(
        self,
        name: str,
        probability: float,
        confidence: float = 1.0,
        evidence: list[str] | None = None,
    ) -> Proposition:
        prop = Proposition(
            name=name,
            probability=_clamp01(probability),
            confidence=_clamp01(confidence),
            evidence=list(evidence) if evidence else [],
        )
        self._props[name] = prop
        return prop

    def update(self, name: str, new_probability: float, reason: str = "") -> BeliefUpdate | None:
        existing = self._props.get(name)
        if existing is None:
            return None
        new_p = _clamp01(new_probability)
        updated = Proposition(
            name=existing.name,
            probability=new_p,
            confidence=existing.confidence,
            evidence=list(existing.evidence),
        )
        self._props[name] = updated
        return BeliefUpdate(
            proposition_name=name,
            old_probability=existing.probability,
            new_probability=new_p,
            reason=reason,
        )

    def get(self, name: str) -> Proposition | None:
        return self._props.get(name)

    def most_certain(self, k: int = 5) -> list[Proposition]:
        props = list(self._props.values())
        props.sort(key=lambda p: p.confidence * p.probability, reverse=True)
        return props[:k]

    def contradictions(self) -> list[tuple[str, str]]:
        names = list(self._props.keys())
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for n1 in names:
            for prefix in ("not_", "no_"):
                if n1.startswith(prefix):
                    base = n1[len(prefix) :]
                    if base in self._props:
                        p1 = self._props[n1]
                        p2 = self._props[base]
                        if abs(p2.probability - (1.0 - p1.probability)) < 0.1:
                            key = tuple(sorted((n1, base)))
                            if key not in seen:
                                seen.add(key)
                                pairs.append((base, n1))
        return pairs

    def bayesian_update(
        self,
        name: str,
        likelihood_given_true: float,
        prior_evidence_prob: float,
    ) -> BeliefUpdate | None:
        existing = self._props.get(name)
        if existing is None:
            return None
        if prior_evidence_prob <= 0.0:
            return None
        new_p = _clamp01(likelihood_given_true * existing.probability / prior_evidence_prob)
        updated = Proposition(
            name=existing.name,
            probability=new_p,
            confidence=existing.confidence,
            evidence=list(existing.evidence),
        )
        self._props[name] = updated
        return BeliefUpdate(
            proposition_name=name,
            old_probability=existing.probability,
            new_probability=new_p,
            reason="bayesian_update",
        )

    def all_propositions(self) -> list[Proposition]:
        return list(self._props.values())


BELIEF_STATE_REGISTRY: dict[str, type[BeliefState]] = {"default": BeliefState}
