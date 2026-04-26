"""Beam search v2 — callable model_fn-based API.

Implements beam search with:
- Configurable beam width and length penalty
- Optional EOS early stopping
- No-repeat n-gram blocking
- Top-p (nucleus) filtering utility

Pure PyTorch — no external dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BeamSearchConfig:
    """Configuration for beam search decoding."""

    beam_width: int = 4
    max_new_tokens: int = 50
    length_penalty: float = 1.0
    eos_token_id: int | None = None
    no_repeat_ngram_size: int = 0
    temperature: float = 1.0


@dataclass
class BeamHypothesis:
    """A single beam hypothesis."""

    token_ids: Tensor
    score: float
    is_done: bool = False


def normalize_score(score: float, length: int, length_penalty: float) -> float:
    if length == 0:
        return score
    return score / (length**length_penalty)


def get_ngram_blocked_tokens(token_ids: Tensor, ngram_size: int) -> set[int]:
    if ngram_size <= 0 or token_ids.numel() < ngram_size - 1:
        return set()
    ids = token_ids.tolist()
    n = ngram_size
    prefix = tuple(ids[-(n - 1) :]) if n > 1 else ()
    blocked: set[int] = set()
    for i in range(len(ids) - n + 1):
        if tuple(ids[i : i + n - 1]) == prefix:
            blocked.add(ids[i + n - 1])
    return blocked


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_remove = (cumprobs - F.softmax(sorted_logits, dim=-1)) > top_p
    sorted_logits = sorted_logits.clone()
    sorted_logits[sorted_remove] = float("-inf")
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


class BeamSearch:
    """Beam search decoder using a callable model_fn.

    Args:
        model_fn: Callable taking (B, T) → (B, T, V) logits.
        config: BeamSearchConfig.
    """

    def __init__(
        self, model_fn: Callable[[Tensor], Tensor], config: BeamSearchConfig | None = None
    ) -> None:
        self.model_fn = model_fn
        self.config = config or BeamSearchConfig()

    def initialize_beams(self, prompt_ids: Tensor) -> list[BeamHypothesis]:
        if prompt_ids.dim() == 2:
            prompt_ids = prompt_ids[0]
        return [
            BeamHypothesis(token_ids=prompt_ids.clone(), score=0.0)
            for _ in range(self.config.beam_width)
        ]

    def expand_beams(self, beams: list[BeamHypothesis]) -> tuple[list[BeamHypothesis], bool]:
        cfg = self.config
        candidates: list[tuple[float, Tensor, bool]] = []

        for beam in beams:
            if beam.is_done:
                candidates.append((beam.score, beam.token_ids, True))
                continue

            ids = beam.token_ids.unsqueeze(0)
            with torch.no_grad():
                logits = self.model_fn(ids)
            next_logits = logits[0, -1, :]
            if cfg.temperature != 1.0:
                next_logits = next_logits / cfg.temperature
            log_probs = F.log_softmax(next_logits, dim=-1)

            blocked = get_ngram_blocked_tokens(beam.token_ids, cfg.no_repeat_ngram_size)
            if blocked:
                for tok in blocked:
                    log_probs[tok] = float("-inf")

            k = min(cfg.beam_width, log_probs.shape[0])
            top_lp, top_idx = torch.topk(log_probs, k)

            for lp, tok_id in zip(top_lp.tolist(), top_idx.tolist()):
                new_score = beam.score + lp
                new_ids = torch.cat(
                    [beam.token_ids, torch.tensor([tok_id], dtype=beam.token_ids.dtype)]
                )
                done = cfg.eos_token_id is not None and tok_id == cfg.eos_token_id
                candidates.append((new_score, new_ids, done))

        def _norm(c: tuple[float, Tensor, bool]) -> float:
            return normalize_score(c[0], c[1].shape[0], cfg.length_penalty)

        candidates.sort(key=_norm, reverse=True)
        selected = candidates[: cfg.beam_width]
        new_beams = [
            BeamHypothesis(token_ids=ids, score=score, is_done=done)
            for score, ids, done in selected
        ]
        return new_beams, all(b.is_done for b in new_beams)

    def search(self, prompt_ids: Tensor) -> Tensor:
        beams = self.initialize_beams(prompt_ids)
        for _ in range(self.config.max_new_tokens):
            beams, all_done = self.expand_beams(beams)
            if all_done:
                break

        def _norm_beam(b: BeamHypothesis) -> float:
            return normalize_score(b.score, b.token_ids.shape[0], self.config.length_penalty)

        return max(beams, key=_norm_beam).token_ids

    def search_with_scores(self, prompt_ids: Tensor) -> list[BeamHypothesis]:
        beams = self.initialize_beams(prompt_ids)
        for _ in range(self.config.max_new_tokens):
            beams, all_done = self.expand_beams(beams)
            if all_done:
                break

        def _norm_beam(b: BeamHypothesis) -> float:
            return normalize_score(b.score, b.token_ids.shape[0], self.config.length_penalty)

        return sorted(beams, key=_norm_beam, reverse=True)
