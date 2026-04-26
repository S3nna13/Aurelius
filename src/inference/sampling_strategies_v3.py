"""Advanced sampling strategies v3 for AureliusTransformer.

Implements composable token-sampling filters:
  - min-p sampling
  - η (eta) / typical sampling
  - DRY (Don't Repeat Yourself) repetition penalty
  - Temperature, top-k, top-p, standard repetition penalty
  - A LogitsProcessor pipeline wiring them together
  - A Sampler with multinomial, greedy, and beam-search step
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    """Configuration for all token-sampling filters and DRY penalty."""

    temperature: float = 1.0
    top_k: int = 0  # 0 = disabled
    top_p: float = 1.0  # 1.0 = disabled
    min_p: float = 0.0  # 0.0 = disabled
    typical_p: float = 1.0  # 1.0 = disabled
    repetition_penalty: float = 1.0  # 1.0 = disabled

    # DRY (Don't Repeat Yourself) repetition penalty
    dry_multiplier: float = 0.0  # 0.0 = disabled
    dry_base: float = 1.75
    dry_allowed_length: int = 2


# ---------------------------------------------------------------------------
# Min-P Sampling
# ---------------------------------------------------------------------------


class MinPSampling:
    """Filter logits using the min-p algorithm.

    Tokens whose softmax probability falls below `min_p * p_max` are
    removed by setting their logit to -inf.
    """

    def filter(self, logits: Tensor, min_p: float) -> Tensor:
        """Return logits with low-probability tokens zeroed out (set to -inf).

        Args:
            logits: Shape (V,) raw logits for one position.
            min_p:  Minimum probability threshold as a fraction of p_max.

        Returns:
            Filtered logits of shape (V,).
        """
        if min_p <= 0.0:
            return logits

        probs = torch.softmax(logits, dim=-1)
        p_max = probs.max()
        threshold = min_p * p_max
        filtered = logits.clone()
        filtered[probs < threshold] = float("-inf")
        return filtered


# ---------------------------------------------------------------------------
# Typical (η) Sampling
# ---------------------------------------------------------------------------


class TypicalSampling:
    """Locally-typical sampling filter.

    Retains the smallest set of tokens whose cumulative probability covers
    `typical_p`, selected by deviation from the distribution's entropy
    (i.e. tokens whose log-probability is closest to the entropy).
    """

    def filter(self, logits: Tensor, typical_p: float) -> Tensor:
        """Return logits with atypical tokens set to -inf.

        Args:
            logits:    Shape (V,) raw logits.
            typical_p: Mass to retain; 1.0 keeps everything.

        Returns:
            Filtered logits of shape (V,).
        """
        if typical_p >= 1.0:
            return logits

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-38))

        # Shannon entropy of the distribution
        entropy = -(probs * log_probs).sum()

        # Deviation of each token's log-prob from the entropy
        deviation = (log_probs + entropy).abs()

        # Sort tokens by deviation (smallest = most typical first)
        sorted_indices = torch.argsort(deviation)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        # Keep tokens until we exceed typical_p
        # Shift by 1 so the token that pushes us over the threshold is kept
        keep_mask = cumulative_probs - sorted_probs < typical_p

        # Map back to original vocabulary indices
        remove_mask = torch.zeros_like(logits, dtype=torch.bool)
        remove_mask[sorted_indices[~keep_mask]] = True

        filtered = logits.clone()
        filtered[remove_mask] = float("-inf")
        return filtered


# ---------------------------------------------------------------------------
# DRY Repetition Penalty
# ---------------------------------------------------------------------------


class DRYRepetitionPenalty:
    """Don't-Repeat-Yourself (DRY) repetition penalty.

    For every candidate token v, the penalty is proportional to
    ``base ^ (match_length - allowed_length)`` where `match_length` is
    the length of the longest suffix of ``context_ids + [v]`` that already
    appears somewhere earlier in `context_ids`.  Only matches longer than
    `allowed_length` are penalised.
    """

    def __init__(self, multiplier: float, base: float, allowed_length: int) -> None:
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length

    def _longest_suffix_match(self, context: list[int], token: int) -> int:
        """Return the length of the longest suffix of context+[token] that
        appears as a sub-sequence starting somewhere earlier in context."""
        seq = context + [token]
        seq_len = len(seq)
        max_match = 0

        # We look for matches of increasing length.
        # A match of length L means the L-token suffix of seq appears
        # starting at some index i in seq (with i+L < seq_len, so it is
        # strictly earlier than the very last position).
        for start in range(seq_len - 1):
            # Length of the potential match from this start
            seq_len - 1  # last added token index in seq

            # Walk forward from `start` and backward from tail
            # to find how many tokens match the current suffix
            match_len = 0
            j = start
            # suffix we are trying to match ends at seq_len-1
            # start matching from the end
            suffix_start = seq_len - 1
            while j < seq_len and seq[j] == seq[suffix_start]:
                if j == suffix_start:
                    # overlap with itself — only accept if strictly earlier
                    break
                j += 1
                suffix_start += 1
                match_len += 1
                if suffix_start >= seq_len:
                    break

            if match_len > max_match:
                max_match = match_len

        return max_match

    def _longest_suffix_match_fast(self, context: list[int], token: int) -> int:
        """Efficient suffix-match: find longest suffix of context+[token]
        that appears as a contiguous sub-sequence starting earlier in context."""
        seq = context + [token]
        n = len(seq)
        if n < 2:
            return 0

        best = 0
        # For each starting position i in context (not the last token position)
        for i in range(n - 1):
            length = 0
            # Compare seq[i:] with suffix of seq ending at n-1
            # The suffix of length L is seq[n-L:n]
            # We want the longest L such that seq[i:i+L] == seq[n-L:n]
            # Equivalently, match seq[i+k] == seq[n-L+k] for k in 0..L-1
            # Grow the match greedily
            # The suffix tail pointer starts at the end and works backward
            # Grow from length=1 upward
            for length in range(1, n - i + 1):
                # suffix of length `length` starts at n-length
                suffix_idx = n - length
                # check if seq[i:i+length] == seq[suffix_idx:suffix_idx+length]
                # i.e. seq[i+length-1] == seq[suffix_idx+length-1] == seq[n-1]
                # We only need to check the new last element each iteration
                new_j = i + length - 1
                new_s = suffix_idx + length - 1
                if new_j >= new_s:
                    # overlap — stop
                    break
                if seq[new_j] != seq[new_s]:
                    break
                best = max(best, length)
        return best

    def compute_penalty(self, logits: Tensor, context_ids: list[int]) -> Tensor:
        """Apply DRY penalty and return modified logits.

        Args:
            logits:      Shape (V,) raw logits.
            context_ids: List of previously generated token ids.

        Returns:
            Modified logits of shape (V,).
        """
        if self.multiplier == 0.0 or not context_ids:
            return logits

        modified = logits.clone()
        vocab_size = logits.shape[0]

        for v in range(vocab_size):
            match_len = self._longest_suffix_match_fast(context_ids, v)
            if match_len > self.allowed_length:
                penalty = self.multiplier * (self.base ** (match_len - self.allowed_length))
                modified[v] = modified[v] - penalty

        return modified


# ---------------------------------------------------------------------------
# Logits Processor
# ---------------------------------------------------------------------------


class LogitsProcessor:
    """Composable pipeline of logit filters driven by a SamplingConfig."""

    def __init__(self, config: SamplingConfig) -> None:
        self.config = config
        self._min_p = MinPSampling()
        self._typical = TypicalSampling()
        self._dry: DRYRepetitionPenalty | None = None
        if config.dry_multiplier > 0.0:
            self._dry = DRYRepetitionPenalty(
                multiplier=config.dry_multiplier,
                base=config.dry_base,
                allowed_length=config.dry_allowed_length,
            )

    # ------------------------------------------------------------------
    # Individual filter methods
    # ------------------------------------------------------------------

    def apply_temperature(self, logits: Tensor) -> Tensor:
        """Divide logits by temperature (sharpens when < 1, flattens when > 1)."""
        t = self.config.temperature
        if t == 1.0:
            return logits
        return logits / t

    def apply_repetition_penalty(self, logits: Tensor, context_ids: list[int]) -> Tensor:
        """Standard repetition penalty: divide positive logits, multiply negative ones."""
        penalty = self.config.repetition_penalty
        if penalty == 1.0 or not context_ids:
            return logits

        modified = logits.clone()
        unique_ids = list(set(context_ids))
        for t in unique_ids:
            if 0 <= t < logits.shape[0]:
                if modified[t] > 0:
                    modified[t] = modified[t] / penalty
                else:
                    modified[t] = modified[t] * penalty
        return modified

    def apply_top_k(self, logits: Tensor, k: int) -> Tensor:
        """Keep only the top-k logits; set the rest to -inf."""
        if k <= 0:
            return logits
        vocab_size = logits.shape[0]
        k = min(k, vocab_size)
        top_k_values, _ = torch.topk(logits, k)
        threshold = top_k_values[..., -1]
        filtered = logits.clone()
        filtered[filtered < threshold] = float("-inf")
        return filtered

    def apply_top_p(self, logits: Tensor, p: float) -> Tensor:
        """Nucleus (top-p) filtering: keep the smallest set covering mass p."""
        if p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        # Remove tokens once cumulative prob exceeds p
        sorted_remove = cumulative_probs - sorted_probs > p
        sorted_logits_filtered = sorted_logits.clone()
        sorted_logits_filtered[sorted_remove] = float("-inf")

        # Scatter back to original ordering
        filtered = torch.zeros_like(logits).fill_(float("-inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits_filtered)
        return filtered

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        logits: Tensor,
        context_ids: list[int] | None = None,
    ) -> Tensor:
        """Apply all enabled filters in order.

        Order: temperature → repetition_penalty → top_k → top_p → min_p →
               typical_p → DRY

        Args:
            logits:      Shape (V,) raw logits.
            context_ids: Previously generated token ids (may be None).

        Returns:
            Processed logits of shape (V,).
        """
        ctx = context_ids or []

        logits = self.apply_temperature(logits)

        if self.config.repetition_penalty != 1.0:
            logits = self.apply_repetition_penalty(logits, ctx)

        if self.config.top_k > 0:
            logits = self.apply_top_k(logits, self.config.top_k)

        if self.config.top_p < 1.0:
            logits = self.apply_top_p(logits, self.config.top_p)

        if self.config.min_p > 0.0:
            logits = self._min_p.filter(logits, self.config.min_p)

        if self.config.typical_p < 1.0:
            logits = self._typical.filter(logits, self.config.typical_p)

        if self._dry is not None:
            logits = self._dry.compute_penalty(logits, ctx)

        return logits


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class Sampler:
    """Token sampling utilities: multinomial, greedy, and beam-search step."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def sample(self, logits: Tensor) -> int:
        """Multinomial sample from softmax distribution.

        Args:
            logits: Shape (V,) raw logits.

        Returns:
            Sampled token id as a Python int.
        """
        probs = torch.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)
        return int(token_id.item())

    def greedy(self, logits: Tensor) -> int:
        """Greedy (argmax) decoding.

        Args:
            logits: Shape (V,) raw logits.

        Returns:
            Token id with highest logit as a Python int.
        """
        return int(logits.argmax().item())

    def beam_step(
        self,
        logits: Tensor,
        beam_scores: Tensor,
        beam_size: int,
    ) -> tuple[Tensor, Tensor]:
        """One beam-search expansion step.

        For each of the `beam` active hypotheses, add the log-softmax of
        `logits` to the beam score, then select the overall top-`beam_size`
        (score, token) pairs.

        Args:
            logits:      Shape (beam, V) or (V,) logits.
                         If 1-D it is broadcast across all beams.
            beam_scores: Shape (beam,) current cumulative log-probs.
            beam_size:   Number of beams to keep.

        Returns:
            (new_beam_scores, new_token_ids), both shape (beam_size,).
        """
        if logits.dim() == 1:
            # Broadcast single logit vector across all beams
            logits = logits.unsqueeze(0).expand(beam_scores.shape[0], -1)

        beam_scores.shape[0]
        vocab = logits.shape[-1]

        log_probs = torch.log_softmax(logits, dim=-1)  # (beam, V)

        # Add current beam score to every candidate token score
        candidate_scores = beam_scores.unsqueeze(1) + log_probs  # (beam, V)

        # Flatten and pick top beam_size
        flat_scores = candidate_scores.view(-1)  # (beam * V,)
        top_scores, top_indices = torch.topk(flat_scores, k=beam_size)

        new_beam_scores = top_scores  # (beam_size,)
        new_token_ids = top_indices % vocab  # (beam_size,)

        return new_beam_scores, new_token_ids
