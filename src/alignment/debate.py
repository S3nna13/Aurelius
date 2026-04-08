"""Debate-style alignment helpers for aggregating arguments and verdicts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DebateTurn:
    speaker: str
    stance: str
    text: str
    score: float


def stance_margin(turns: list[DebateTurn], pro_stance: str = "pro") -> float:
    """Compute the score margin between pro and non-pro turns."""
    pro_total = sum(turn.score for turn in turns if turn.stance == pro_stance)
    con_total = sum(turn.score for turn in turns if turn.stance != pro_stance)
    return pro_total - con_total


def majority_verdict(turns: list[DebateTurn], pro_stance: str = "pro") -> str:
    """Return the winning stance from aggregate turn scores."""
    margin = stance_margin(turns, pro_stance=pro_stance)
    if margin > 0:
        return pro_stance
    if margin < 0:
        return "con"
    return "tie"


def self_consistency_score(verdicts: list[str], positive_label: str = "pro") -> float:
    """Fraction of verdicts supporting the positive label."""
    if not verdicts:
        return 0.0
    return sum(verdict == positive_label for verdict in verdicts) / len(verdicts)


def debate_preference_loss(pro_scores: torch.Tensor, con_scores: torch.Tensor) -> torch.Tensor:
    """Pairwise loss encouraging pro arguments to outrank con arguments."""
    if pro_scores.shape != con_scores.shape:
        raise ValueError("pro_scores and con_scores must match")
    return torch.nn.functional.softplus(-(pro_scores - con_scores)).mean()


def rank_turns(turns: list[DebateTurn]) -> list[DebateTurn]:
    """Sort debate turns by descending score."""
    return sorted(turns, key=lambda turn: turn.score, reverse=True)
