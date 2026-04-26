from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Observation:
    statement: str


@dataclass
class Hypothesis:
    explanation: str
    plausibility: float = 0.5


def abduce(obs: Observation, rules: dict[str, str]) -> list[Hypothesis]:
    hyps: list[Hypothesis] = []
    for explanation, effect in rules.items():
        if effect in obs.statement or obs.statement in effect:
            hyps.append(Hypothesis(explanation=explanation))
    return hyps


class AbductiveReasoner:
    def __init__(self) -> None:
        self._rules: dict[str, tuple[str, float]] = {}

    def add_rule(self, explanation: str, effect: str, plausibility: float = 0.5) -> None:
        self._rules[explanation] = (effect, plausibility)

    def explain(self, obs: Observation) -> list[Hypothesis]:
        hyps: list[Hypothesis] = []
        for explanation, (effect, plaus) in self._rules.items():
            if effect in obs.statement or obs.statement in effect:
                hyps.append(Hypothesis(explanation=explanation, plausibility=plaus))
        hyps.sort(key=lambda h: h.plausibility, reverse=True)
        return hyps

    def most_plausible(self, obs: Observation) -> Hypothesis | None:
        hyps = self.explain(obs)
        return hyps[0] if hyps else None

    def rule_count(self) -> int:
        return len(self._rules)


ABDUCTIVE_REASONER = AbductiveReasoner()
