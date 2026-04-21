"""Generative Reward Model — Kimi K2.5 §3.5 + GLM-5 §4.2.

Replaces binary verifiers for open-ended tasks with multi-dimension scoring.
For verifiable tasks (math/code), hybrid mode "rule" uses rule_reward directly.

Four scoring dimensions:
  - helpfulness: does the response serve the user's intent?
  - adherence:   were all instructions followed?
  - relevance:   on-topic vs hallucinated detail?
  - detail:      appropriate depth for the task?

References
----------
Kimi K2.5: arXiv:2602.02276, §3.5
GLM-5:     arXiv:2602.15763, §4.2
"""
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

DIMENSIONS = ["helpfulness", "adherence", "relevance", "detail"]


@dataclass
class GRMConfig:
    """Configuration for GenerativeRewardModel.

    Parameters
    ----------
    weights:
        Per-dimension importance weights.  Need not sum to 1; they are
        normalised internally.  Default: equal weight across all four
        canonical dimensions.
    mode:
        ``"grm"``  — always use weighted multi-dimension scoring.
        ``"rule"`` — use ``rule_reward`` directly when it is provided
                     (verifiable tasks: math / code).  Falls back to GRM
                     scoring when ``rule_reward`` is ``None``.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {d: 0.25 for d in DIMENSIONS}
    )
    mode: Literal["grm", "rule"] = "grm"


class GenerativeRewardModel:
    """Hybrid generative / rule-based reward model.

    Parameters
    ----------
    config:
        ``GRMConfig`` instance.  Defaults to ``GRMConfig()`` (equal weights,
        GRM mode).

    Examples
    --------
    >>> grm = GenerativeRewardModel()
    >>> scores = {"helpfulness": 0.9, "adherence": 0.8,
    ...           "relevance": 0.7, "detail": 0.6}
    >>> grm.score(scores)   # returns tensor in [0, 1]
    tensor(0.7500)
    """

    def __init__(self, config: Optional[GRMConfig] = None) -> None:
        self.config = config or GRMConfig()
        w = self.config.weights
        total = sum(w.values()) or 1.0
        # Normalised weight dict — guaranteed to sum to 1.0
        self._w: Dict[str, float] = {k: v / total for k, v in w.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        dim_scores: Dict[str, float | torch.Tensor],
        rule_reward: Optional[float | torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute a scalar reward in [0, 1].

        Parameters
        ----------
        dim_scores:
            Mapping from dimension name to score value (scalar or tensor).
            Dimensions not present in ``_w`` contribute 0.  Dimensions
            present in ``_w`` but missing from ``dim_scores`` also
            contribute 0 — they are not an error.
        rule_reward:
            Pre-computed reward from a rule-based verifier.  Honoured only
            when ``config.mode == "rule"`` **and** the value is not ``None``.

        Returns
        -------
        torch.Tensor
            Scalar tensor clamped to [0, 1].
        """
        if self.config.mode == "rule" and rule_reward is not None:
            if not isinstance(rule_reward, torch.Tensor):
                rule_reward = torch.tensor(rule_reward, dtype=torch.float32)
            return rule_reward

        # Weighted sum over available dimensions
        out: float | torch.Tensor = 0.0
        for k, v in dim_scores.items():
            w = self._w.get(k, 0.0)
            if w == 0.0:
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            out = out + w * v

        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out, dtype=torch.float32)

        return torch.clamp(out, 0.0, 1.0)
