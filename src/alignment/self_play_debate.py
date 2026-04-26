"""Self-play debate coordination helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DebateMove:
    agent: str
    claim: str
    score: float


def alternate_agents(agent_a: str, agent_b: str, n_rounds: int) -> list[str]:
    """Return an alternating speaker order."""
    if n_rounds < 0:
        raise ValueError("n_rounds must be non-negative")
    order = []
    for round_idx in range(n_rounds):
        order.append(agent_a if round_idx % 2 == 0 else agent_b)
    return order


def self_play_margin(moves: list[DebateMove], agent_a: str) -> float:
    """Aggregate score margin for one agent."""
    score_a = sum(move.score for move in moves if move.agent == agent_a)
    score_b = sum(move.score for move in moves if move.agent != agent_a)
    return score_a - score_b


def winning_agent(moves: list[DebateMove], agent_a: str, agent_b: str) -> str:
    """Return the winning agent or tie."""
    margin = self_play_margin(moves, agent_a)
    if margin > 0:
        return agent_a
    if margin < 0:
        return agent_b
    return "tie"


def summarize_claims(moves: list[DebateMove], top_k: int = 2) -> list[str]:
    """Return the top-k claims by score."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    ranked = sorted(moves, key=lambda move: move.score, reverse=True)
    return [move.claim for move in ranked[:top_k]]
