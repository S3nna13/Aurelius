"""
Constitutional Constraint Enforcement for Aurelius LLM.

Provides rule-based output filtering, logit-level constraint enforcement,
and constrained decoding for constitutional AI principles.

Pure PyTorch only - no external dependencies beyond stdlib and torch.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ConstitutionalRule
# ---------------------------------------------------------------------------

class ConstitutionalRule:
    """A single constitutional constraint rule.

    Parameters
    ----------
    name:
        Human-readable identifier for the rule.
    constraint_type:
        One of ``"forbidden_tokens"``, ``"required_prefix"``,
        ``"max_repetition"``, ``"sentiment_bound"``, ``"length_bound"``.
    parameters:
        Type-specific configuration dict.  Keys vary by type:

        forbidden_tokens
            ``token_ids`` - list[int] of disallowed token ids.
        required_prefix
            ``prefix`` - list[int] token ids that must start the sequence.
        max_repetition
            ``max_reps`` - int, maximum allowed repetitions of any single token.
        sentiment_bound
            ``min_positive_ratio`` - float in [0, 1]; fraction of tokens that
            must *not* be in ``negative_token_ids``.
            ``negative_token_ids`` - list[int].
        length_bound
            ``min_length`` - int (default 0).
            ``max_length`` - int (default sys.maxsize).
    """

    VALID_TYPES = frozenset({
        "forbidden_tokens",
        "required_prefix",
        "max_repetition",
        "sentiment_bound",
        "length_bound",
    })

    def __init__(self, name: str, constraint_type: str, parameters: dict) -> None:
        if constraint_type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown constraint_type '{constraint_type}'. "
                f"Valid: {sorted(self.VALID_TYPES)}"
            )
        self.name = name
        self.constraint_type = constraint_type
        self.parameters = dict(parameters)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, token_ids: List[int]) -> bool:
        """Return True when the constraint is satisfied."""
        return self.violation_score(token_ids) == 0.0

    def violation_score(self, token_ids: List[int]) -> float:
        """Return a non-negative score; 0 means fully satisfied."""
        ct = self.constraint_type
        if ct == "forbidden_tokens":
            return self._forbidden_violation(token_ids)
        if ct == "required_prefix":
            return self._required_prefix_violation(token_ids)
        if ct == "max_repetition":
            return self._max_repetition_violation(token_ids)
        if ct == "sentiment_bound":
            return self._sentiment_violation(token_ids)
        if ct == "length_bound":
            return self._length_violation(token_ids)
        return 0.0  # unreachable after __init__ guard

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forbidden_violation(self, token_ids: List[int]) -> float:
        forbidden = set(self.parameters.get("token_ids", []))
        count = sum(1 for t in token_ids if t in forbidden)
        return float(count)

    def _required_prefix_violation(self, token_ids: List[int]) -> float:
        prefix: List[int] = self.parameters.get("prefix", [])
        if not prefix:
            return 0.0
        if len(token_ids) < len(prefix):
            mismatches = sum(
                1 for i, p in enumerate(prefix[: len(token_ids)])
                if token_ids[i] != p
            )
            remaining = len(prefix) - len(token_ids)
            return float(mismatches + remaining)
        mismatches = sum(
            1 for i, p in enumerate(prefix) if token_ids[i] != p
        )
        return float(mismatches)

    def _max_repetition_violation(self, token_ids: List[int]) -> float:
        max_reps: int = self.parameters.get("max_reps", 3)
        if not token_ids:
            return 0.0
        counts = Counter(token_ids)
        excess = sum(max(0, c - max_reps) for c in counts.values())
        return float(excess)

    def _sentiment_violation(self, token_ids: List[int]) -> float:
        neg_ids = set(self.parameters.get("negative_token_ids", []))
        min_positive_ratio: float = self.parameters.get("min_positive_ratio", 0.0)
        if not token_ids:
            return 0.0
        neg_count = sum(1 for t in token_ids if t in neg_ids)
        positive_ratio = 1.0 - neg_count / len(token_ids)
        shortfall = max(0.0, min_positive_ratio - positive_ratio)
        return shortfall

    def _length_violation(self, token_ids: List[int]) -> float:
        min_len: int = self.parameters.get("min_length", 0)
        max_len: int = self.parameters.get("max_length", 2 ** 31 - 1)
        length = len(token_ids)
        violation = 0.0
        if length < min_len:
            violation += float(min_len - length)
        if length > max_len:
            violation += float(length - max_len)
        return violation


# ---------------------------------------------------------------------------
# TokenConstraintSet
# ---------------------------------------------------------------------------

class TokenConstraintSet:
    """Aggregates multiple :class:`ConstitutionalRule` objects."""

    def __init__(self, rules: List[ConstitutionalRule]) -> None:
        self.rules = list(rules)

    def check_all(self, token_ids: List[int]) -> Dict[str, bool]:
        """Return a mapping of rule name to satisfied (bool)."""
        return {r.name: r.check(token_ids) for r in self.rules}

    def total_violation(self, token_ids: List[int]) -> float:
        """Sum of all individual violation scores (always >= 0)."""
        return sum(r.violation_score(token_ids) for r in self.rules)

    def satisfied(self, token_ids: List[int]) -> bool:
        """True only when every rule is satisfied."""
        return all(r.check(token_ids) for r in self.rules)


# ---------------------------------------------------------------------------
# LogitConstraintEnforcer
# ---------------------------------------------------------------------------

class LogitConstraintEnforcer(nn.Module):
    """Applies constitutional constraints directly to logits at decode time.

    Parameters
    ----------
    vocab_size:
        Size of the token vocabulary.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        # Persistent buffers so they move with the module's device
        self.register_buffer(
            "forbidden_token_mask", torch.zeros(vocab_size, dtype=torch.bool)
        )
        self.required_tokens: List[int] = []
        self._rep_penalty: float = 1.0
        self._max_reps: int = 3
        self._required_boost: float = 5.0

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_forbidden(self, token_ids: List[int]) -> None:
        """Mark tokens as forbidden (will be set to -inf in forward)."""
        self.forbidden_token_mask.zero_()
        if token_ids:
            idx = torch.tensor(token_ids, dtype=torch.long)
            self.forbidden_token_mask[idx] = True

    def set_required_tokens(self, token_ids: List[int]) -> None:
        """Tokens that must appear eventually; will be boosted until generated."""
        self.required_tokens = list(token_ids)

    def set_repetition_penalty(self, penalty: float, max_reps: int) -> None:
        """Configure repetition penalty applied after ``max_reps`` occurrences."""
        self._rep_penalty = float(penalty)
        self._max_reps = int(max_reps)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,            # [B, vocab_size]
        generated_so_far: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:                   # [B, vocab_size]
        """Apply all constraints to logits and return modified logits."""
        B, V = logits.shape
        logits = logits.clone()

        # 1. Forbidden tokens -> -inf
        if self.forbidden_token_mask.any():
            logits[:, self.forbidden_token_mask] = float("-inf")

        # 2. Required tokens not yet generated -> boost
        if self.required_tokens:
            for b in range(B):
                generated_set = set(generated_so_far[b].tolist())
                for tok in self.required_tokens:
                    if tok not in generated_set:
                        # Only boost if not forbidden
                        if not self.forbidden_token_mask[tok]:
                            logits[b, tok] = logits[b, tok] + self._required_boost

        # 3. Repetition penalty: penalise tokens appearing >= max_reps times
        if self._rep_penalty != 1.0:
            for b in range(B):
                counts = Counter(generated_so_far[b].tolist())
                for tok_id, cnt in counts.items():
                    if cnt >= self._max_reps and tok_id < V:
                        val = logits[b, tok_id]
                        if val != float("-inf"):
                            if val > 0:
                                logits[b, tok_id] = val / self._rep_penalty
                            else:
                                logits[b, tok_id] = val * self._rep_penalty

        return logits


# ---------------------------------------------------------------------------
# ConstrainedDecoder
# ---------------------------------------------------------------------------

class ConstrainedDecoder:
    """Autoregressive decoder that applies :class:`LogitConstraintEnforcer`.

    Parameters
    ----------
    model:
        An ``nn.Module`` whose ``forward(input_ids)`` returns logits of shape
        ``[B, T, vocab_size]``.
    enforcer:
        A configured :class:`LogitConstraintEnforcer` instance.
    """

    def __init__(self, model: nn.Module, enforcer: LogitConstraintEnforcer) -> None:
        self.model = model
        self.enforcer = enforcer

    # ------------------------------------------------------------------
    # Greedy decode with constraint enforcement
    # ------------------------------------------------------------------

    def decode(
        self,
        input_ids: torch.Tensor,  # [B, T]
        max_new: int,
    ) -> torch.Tensor:            # [B, T + max_new]
        """Greedy autoregressive decode with enforcer applied at each step."""
        self.model.eval()
        ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new):
                logits_all = self.model(ids)          # [B, T_cur, V]
                logits_last = logits_all[:, -1, :]    # [B, V]
                logits_last = self.enforcer(logits_last, ids)
                next_token = logits_last.argmax(dim=-1, keepdim=True)  # [B, 1]
                ids = torch.cat([ids, next_token], dim=1)

        return ids

    # ------------------------------------------------------------------
    # Beam search with constraint enforcement
    # ------------------------------------------------------------------

    def beam_decode_constrained(
        self,
        input_ids: torch.Tensor,  # [B, T]
        beam_size: int,
        max_new: int,
    ) -> torch.Tensor:             # [B, T + max_new]
        """Beam search with per-step constraint enforcement and constraint-aware scoring."""
        self.model.eval()
        B, T = input_ids.shape
        V = self.enforcer.vocab_size

        # beams[b] = list of (log_prob_sum, token_id_tensor [T_cur])
        beams: List[List[Tuple[float, torch.Tensor]]] = [
            [(0.0, input_ids[b].clone())] for b in range(B)
        ]

        with torch.no_grad():
            for step in range(max_new):
                new_beams: List[List[Tuple[float, torch.Tensor]]] = [[] for _ in range(B)]

                for b in range(B):
                    candidates: List[Tuple[float, torch.Tensor]] = []
                    for score, seq in beams[b]:
                        seq_batch = seq.unsqueeze(0)            # [1, T_cur]
                        logits_all = self.model(seq_batch)      # [1, T_cur, V]
                        logits_last = logits_all[:, -1, :]      # [1, V]
                        logits_last = self.enforcer(logits_last, seq_batch)
                        log_probs = F.log_softmax(logits_last, dim=-1)  # [1, V]

                        topk_lp, topk_ids = log_probs[0].topk(
                            min(beam_size, V), dim=-1
                        )
                        for lp, tok in zip(topk_lp.tolist(), topk_ids.tolist()):
                            new_seq = torch.cat(
                                [seq, torch.tensor([tok], dtype=seq.dtype)], dim=0
                            )
                            rule_penalty = self._violation_penalty(new_seq.tolist())
                            candidates.append((score + lp - rule_penalty, new_seq))

                    candidates.sort(key=lambda x: x[0], reverse=True)
                    new_beams[b] = candidates[:beam_size]

                beams = new_beams

        # Extract best beam per batch item
        result_list = []
        for b in range(B):
            best_score, best_seq = max(beams[b], key=lambda x: x[0])
            result_list.append(best_seq)

        # Pad / stack to [B, T + max_new]
        max_len = max(s.shape[0] for s in result_list)
        out = torch.zeros(B, max_len, dtype=input_ids.dtype)
        for b, seq in enumerate(result_list):
            out[b, : seq.shape[0]] = seq
        return out

    def _violation_penalty(self, token_ids: List[int]) -> float:
        """Scalar penalty from the enforcer's forbidden mask."""
        forbidden_count = sum(
            1 for t in token_ids
            if t < self.enforcer.vocab_size
            and self.enforcer.forbidden_token_mask[t].item()
        )
        return float(forbidden_count) * 10.0


# ---------------------------------------------------------------------------
# ConstraintRepairModel
# ---------------------------------------------------------------------------

class ConstraintRepairModel(nn.Module):
    """Sequence-to-sequence model trained to repair constitutional violations.

    Given a token sequence that violates one or more rules, it predicts
    replacement (repaired) token logits at every position.

    Architecture: embedding -> transformer encoder layers -> linear head.

    Parameters
    ----------
    d_model:
        Model dimension.
    vocab_size:
        Vocabulary size.
    n_layers:
        Number of transformer encoder layers.
    """

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, d_model // 16),
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : [B, T]

        Returns
        -------
        repair_logits : [B, T, vocab_size]
        """
        x = self.embed(input_ids)   # [B, T, d_model]
        x = self.encoder(x)         # [B, T, d_model]
        logits = self.head(x)       # [B, T, vocab_size]
        return logits

    def repair_loss(
        self,
        input_ids: torch.Tensor,   # [B, T]
        target_ids: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        """Cross-entropy loss between repair predictions and target tokens."""
        logits = self.forward(input_ids)    # [B, T, V]
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            target_ids.reshape(B * T),
        )
        return loss


# ---------------------------------------------------------------------------
# ConstitutionalSampler
# ---------------------------------------------------------------------------

class ConstitutionalSampler:
    """Samples multiple candidate sequences and selects the least violating one.

    Parameters
    ----------
    model:
        An ``nn.Module`` with ``forward(input_ids) -> [B, T, vocab_size]``.
    rules:
        List of :class:`ConstitutionalRule` instances used for scoring.
    n_candidates:
        Number of candidates to sample in :meth:`best_of_n`.
    """

    def __init__(
        self,
        model: nn.Module,
        rules: List[ConstitutionalRule],
        n_candidates: int = 4,
    ) -> None:
        self.model = model
        self.constraint_set = TokenConstraintSet(rules)
        self.n_candidates = n_candidates

    # ------------------------------------------------------------------
    # best_of_n
    # ------------------------------------------------------------------

    def best_of_n(
        self,
        input_ids: torch.Tensor,  # [B, T]
        max_new: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n_candidates completions; return least-violating.

        Returns
        -------
        best_ids : [B, T + max_new]
        scores   : [B]  (violation scores, lower = better)
        """
        self.model.eval()
        B = input_ids.shape[0]

        all_seqs: List[torch.Tensor] = []
        all_violations: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.n_candidates):
                sample = self._sample(input_ids, max_new)  # [B, T+max_new]
                all_seqs.append(sample)

                viol = torch.tensor(
                    [
                        self.constraint_set.total_violation(sample[b].tolist())
                        for b in range(B)
                    ],
                    dtype=torch.float,
                )
                all_violations.append(viol)

        # Stack -> [n_candidates, B]
        violations_stack = torch.stack(all_violations, dim=0)

        best_ids_list = []
        best_scores_list = []
        for b in range(B):
            best_k = int(violations_stack[:, b].argmin().item())
            best_ids_list.append(all_seqs[best_k][b])
            best_scores_list.append(violations_stack[best_k, b])

        best_ids = torch.stack(best_ids_list, dim=0)
        best_scores = torch.stack(best_scores_list, dim=0)
        return best_ids, best_scores

    # ------------------------------------------------------------------
    # rejection_sample
    # ------------------------------------------------------------------

    def rejection_sample(
        self,
        input_ids: torch.Tensor,  # [B, T]
        max_new: int = 8,
        max_attempts: int = 10,
    ) -> torch.Tensor:            # [B, T + max_new]
        """Keep sampling until all constraints satisfied or max_attempts reached."""
        self.model.eval()
        B = input_ids.shape[0]

        best_sample: Optional[torch.Tensor] = None
        best_violation: float = float("inf")

        with torch.no_grad():
            for _ in range(max_attempts):
                sample = self._sample(input_ids, max_new)

                all_satisfied = all(
                    self.constraint_set.satisfied(sample[b].tolist())
                    for b in range(B)
                )
                total_viol = sum(
                    self.constraint_set.total_violation(sample[b].tolist())
                    for b in range(B)
                )
                if best_sample is None or total_viol < best_violation:
                    best_sample = sample
                    best_violation = total_viol

                if all_satisfied:
                    return sample

        return best_sample  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample(
        self,
        input_ids: torch.Tensor,  # [B, T]
        max_new: int,
    ) -> torch.Tensor:            # [B, T + max_new]
        """Single-pass temperature-1 multinomial sampling."""
        ids = input_ids.clone()
        for _ in range(max_new):
            logits_all = self.model(ids)         # [B, T_cur, V]
            logits_last = logits_all[:, -1, :]   # [B, V]
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            ids = torch.cat([ids, next_token], dim=1)
        return ids


# ---------------------------------------------------------------------------
# ConstitutionalConfig
# ---------------------------------------------------------------------------

@dataclass
class ConstitutionalConfig:
    """Default hyperparameters for the constitutional constraint system."""

    vocab_size: int = 64
    d_model: int = 32
    n_layers: int = 2

    # Forbidden tokens (list of int token ids)
    forbidden_tokens: List[int] = field(default_factory=list)

    # Repetition penalty settings
    max_rep_penalty: float = 1.5
    max_reps: int = 3

    # Sampling settings
    n_candidates: int = 4
    max_attempts: int = 10
