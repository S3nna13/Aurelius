"""Beam search v3 — practical improvements for LLM decoding.

Implements beam search with:
- Google NMT length penalty
- Repetition penalty
- No-repeat n-gram blocking
- Min-length EOS suppression
- Temperature scaling

Pure PyTorch — no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BeamConfig:
    """Configuration for beam search decoding."""

    beam_width: int = 4
    max_new_tokens: int = 128
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0   # 0 = disabled
    min_length: int = 0
    eos_token_id: int = 2
    temperature: float = 1.0
    repetition_penalty: float = 1.0


@dataclass
class BeamHypothesis:
    """A single beam hypothesis."""

    token_ids: List[int]
    score: float
    length: int


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def apply_length_penalty(scores: Tensor, lengths: Tensor, alpha: float) -> Tensor:
    """Normalize beam scores using the Google NMT length penalty.

    Formula: score / ((5 + length)^alpha / (5 + 1)^alpha)

    Args:
        scores:  (B,) float tensor of raw beam scores.
        lengths: (B,) int tensor of sequence lengths.
        alpha:   Length penalty exponent.

    Returns:
        (B,) normalized scores.
    """
    if alpha == 0.0:
        return scores.clone()
    penalty = ((5.0 + lengths.float()) / 6.0) ** alpha
    return scores / penalty


def apply_repetition_penalty(logits: Tensor, input_ids: Tensor, penalty: float) -> Tensor:
    """Apply repetition penalty to logits.

    For each token already present in *input_ids*:
      - If the logit is positive, divide by *penalty*.
      - If the logit is negative (or zero), multiply by *penalty*.

    This discourages the model from repeating tokens it has already produced.

    Args:
        logits:    (vocab,) float tensor.
        input_ids: (T,) int tensor of already-generated token ids.
        penalty:   Repetition penalty factor (>= 1.0 to penalise, < 1.0 to promote).

    Returns:
        (vocab,) modified logits.
    """
    if penalty == 1.0 or input_ids.numel() == 0:
        return logits.clone()

    logits = logits.clone()
    unique_ids = input_ids.unique()
    score = logits[unique_ids]
    # positive logits → divide; non-positive → multiply
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits[unique_ids] = score
    return logits


def get_ngram_blocked_tokens(sequence: List[int], ngram_size: int) -> Set[int]:
    """Return the set of token ids that would form a repeated n-gram.

    Looks at the last (ngram_size - 1) tokens of *sequence* as a prefix.
    Any token that followed that same prefix earlier in *sequence* is blocked.

    Args:
        sequence:   List of already-generated token ids.
        ngram_size: Size of n-gram to block (0 = disabled, returns empty set).

    Returns:
        Set of blocked token ids.
    """
    if ngram_size <= 0 or len(sequence) < ngram_size - 1:
        return set()

    n = ngram_size
    prefix_len = n - 1

    if prefix_len == 0:
        # unigram blocking: block every token already in sequence
        return set(sequence)

    prefix = tuple(sequence[-prefix_len:])
    blocked: Set[int] = set()

    # Slide a window of length (n-1) over all positions except the tail
    for i in range(len(sequence) - prefix_len):
        if tuple(sequence[i : i + prefix_len]) == prefix:
            # the next token after this prefix was sequence[i + prefix_len]
            if i + prefix_len < len(sequence):
                blocked.add(sequence[i + prefix_len])

    return blocked


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class BeamSearchDecoder:
    """Beam search decoder backed by an arbitrary callable model function.

    Args:
        model_fn: Callable that accepts a (1, T) int64 tensor of token ids and
                  returns a (1, T, vocab_size) float tensor of logits.
        config:   BeamConfig instance.
    """

    def __init__(self, model_fn: Callable[[Tensor], Tensor], config: BeamConfig) -> None:
        self.model_fn = model_fn
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialize_beams(self, prompt_ids: Tensor) -> List[BeamHypothesis]:
        """Create beam_width copies of the prompt as initial hypotheses (score 0.0).

        Args:
            prompt_ids: 1-D or 2-D (1, T) int tensor.

        Returns:
            List of *beam_width* BeamHypothesis objects.
        """
        if prompt_ids.dim() == 2:
            prompt_ids = prompt_ids[0]
        ids = prompt_ids.tolist()
        return [
            BeamHypothesis(token_ids=list(ids), score=0.0, length=len(ids))
            for _ in range(self.config.beam_width)
        ]

    def step(
        self,
        hypotheses: List[BeamHypothesis],
    ) -> Tuple[List[BeamHypothesis], List[BeamHypothesis]]:
        """Perform one beam-search expansion step.

        For each active hypothesis:
          1. Run model_fn to get next-token logits.
          2. Apply temperature scaling.
          3. Apply repetition penalty.
          4. Apply no-repeat n-gram blocking.
          5. Suppress EOS when below min_length.
          6. Convert to log-probabilities and take top-beam_width continuations.

        All candidate beams are then ranked and the top *beam_width* are kept.
        Any beam that ended with the EOS token is moved to *completed_beams*.

        Args:
            hypotheses: Currently active BeamHypothesis objects.

        Returns:
            (active_beams, completed_beams) tuple.
        """
        cfg = self.config
        candidates: List[BeamHypothesis] = []
        completed: List[BeamHypothesis] = []

        for hyp in hypotheses:
            ids_tensor = torch.tensor(hyp.token_ids, dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                logits = self.model_fn(ids_tensor)  # (1, T, V)

            next_logits = logits[0, -1, :].float()  # (V,)

            # 1. Temperature
            if cfg.temperature != 1.0:
                next_logits = next_logits / cfg.temperature

            # 2. Repetition penalty
            if cfg.repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(
                    next_logits,
                    torch.tensor(hyp.token_ids, dtype=torch.long),
                    cfg.repetition_penalty,
                )

            # 3. N-gram blocking
            if cfg.no_repeat_ngram_size > 0:
                blocked = get_ngram_blocked_tokens(hyp.token_ids, cfg.no_repeat_ngram_size)
                for tok in blocked:
                    next_logits[tok] = float("-inf")

            # 4. Suppress EOS below min_length
            generated_len = len(hyp.token_ids)  # includes prompt
            if generated_len < cfg.min_length:
                next_logits[cfg.eos_token_id] = float("-inf")

            log_probs = F.log_softmax(next_logits, dim=-1)

            # Expand top-beam_width continuations
            k = min(cfg.beam_width, log_probs.shape[0])
            top_lp, top_idx = torch.topk(log_probs, k)

            for lp, tok_id in zip(top_lp.tolist(), top_idx.tolist()):
                new_ids = hyp.token_ids + [tok_id]
                new_score = hyp.score + lp
                new_hyp = BeamHypothesis(
                    token_ids=new_ids,
                    score=new_score,
                    length=len(new_ids),
                )
                candidates.append(new_hyp)

        # Sort all candidates by length-penalised score and keep top-k
        all_hyps = sorted(candidates, key=self._normed_score, reverse=True)
        active: List[BeamHypothesis] = []

        for hyp in all_hyps:
            if hyp.token_ids[-1] == cfg.eos_token_id:
                completed.append(hyp)
            else:
                if len(active) < cfg.beam_width:
                    active.append(hyp)

            if len(active) == cfg.beam_width:
                break

        # If we ran out of active beams, pad from completed
        return active, completed

    def decode(self, prompt_ids: Tensor) -> Tensor:
        """Run full beam search and return the best sequence as a 1-D tensor.

        Args:
            prompt_ids: 1-D or 2-D (1, T) int tensor (the prompt).

        Returns:
            1-D int64 tensor containing prompt + generated tokens.
        """
        active = self.initialize_beams(prompt_ids)
        all_completed: List[BeamHypothesis] = []

        for _ in range(self.config.max_new_tokens):
            if not active:
                break
            active, newly_completed = self.step(active)
            all_completed.extend(newly_completed)
            if not active:
                break

        # Treat any still-active beams as completed (hit max_new_tokens)
        all_completed.extend(active)

        if not all_completed:
            # Fallback: return the prompt unchanged
            if prompt_ids.dim() == 2:
                prompt_ids = prompt_ids[0]
            return prompt_ids.long()

        best = max(all_completed, key=self._normed_score)
        return torch.tensor(best.token_ids, dtype=torch.long)

    def decode_with_scores(
        self, prompt_ids: Tensor
    ) -> List[Tuple[Tensor, float]]:
        """Run full beam search and return all completed beams sorted best-first.

        Args:
            prompt_ids: 1-D or 2-D (1, T) int tensor (the prompt).

        Returns:
            List of (sequence_tensor, normalized_score) pairs, sorted by
            normalised score descending.
        """
        active = self.initialize_beams(prompt_ids)
        all_completed: List[BeamHypothesis] = []

        for _ in range(self.config.max_new_tokens):
            if not active:
                break
            active, newly_completed = self.step(active)
            all_completed.extend(newly_completed)
            if not active:
                break

        all_completed.extend(active)

        if not all_completed:
            if prompt_ids.dim() == 2:
                prompt_ids = prompt_ids[0]
            return [(prompt_ids.long(), 0.0)]

        sorted_hyps = sorted(all_completed, key=self._normed_score, reverse=True)
        return [
            (torch.tensor(h.token_ids, dtype=torch.long), self._normed_score(h))
            for h in sorted_hyps
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normed_score(self, hyp: BeamHypothesis) -> float:
        alpha = self.config.length_penalty
        if alpha == 0.0:
            return hyp.score
        penalty = ((5.0 + hyp.length) / 6.0) ** alpha
        return hyp.score / penalty
