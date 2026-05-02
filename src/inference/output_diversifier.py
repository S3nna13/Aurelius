"""Output diversification for AureliusTransformer inference.

Output diversification ensures that when sampling multiple completions, they
are meaningfully different rather than near-duplicates. This is critical for
best-of-N sampling, debate, self-consistency, and similar multi-sample methods.

Techniques implemented:
  - N-gram blocking across samples (NgramBlocker)
  - Diversity bonus to logits based on Hamming or cosine distance (DiversityPenalty)
  - Diverse batch generation combining the above (OutputDiversifier)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiversifierConfig:
    """Configuration for output diversification."""

    strategy: str = "nucleus"
    """Sampling strategy: 'nucleus', 'diverse_beam', or 'stochastic_beam'."""

    diversity_penalty: float = 0.5
    """Strength of the diversity bonus applied to logits."""

    ngram_size: int = 3
    """N-gram size used for cross-sample blocking."""

    min_diversity: float = 0.3
    """Minimum pairwise diversity required to retain a sequence in filter_diverse."""

    temperature: float = 1.0
    """Sampling temperature (>1 flatter, <1 sharper)."""


# ---------------------------------------------------------------------------
# NgramBlocker
# ---------------------------------------------------------------------------


class NgramBlocker:
    """Block tokens that would extend an n-gram already seen in previous samples.

    At each decoding step the blocker examines the last (ngram_size-1) tokens
    of the current context.  Any token whose addition would create an n-gram
    that already appears in the blocked set has its logit set to -inf.

    Args:
        ngram_size: Size of the n-grams to track (default 3).
    """

    def __init__(self, ngram_size: int = 3) -> None:
        self.ngram_size = ngram_size
        self._blocked: set[tuple[int, ...]] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, sequence_ids: torch.Tensor) -> None:
        """Add all n-grams from *sequence_ids* to the blocked set.

        Args:
            sequence_ids: 1-D LongTensor of token ids for a completed (or
                in-progress) sequence.
        """
        ids = sequence_ids.tolist()
        n = self.ngram_size
        for i in range(len(ids) - n + 1):
            ngram = tuple(ids[i : i + n])
            self._blocked.add(ngram)

    def apply_blocking_mask(self, logits: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        """Return logits with blocked tokens set to -inf.

        A token ``t`` is blocked if the tuple
        ``(*context_ids[-(ngram_size-1):], t)`` is in the blocked n-gram set.

        Args:
            logits: 1-D float tensor of shape (vocab_size,).
            context_ids: 1-D LongTensor of already-generated token ids (the
                current decoding context).

        Returns:
            Modified (vocab_size,) logits tensor.
        """
        if not self._blocked or self.ngram_size <= 0:
            return logits

        prefix_len = self.ngram_size - 1
        ctx = context_ids.tolist()
        prefix = tuple(ctx[-prefix_len:]) if prefix_len > 0 else ()

        logits = logits.clone()
        vocab_size = logits.shape[0]

        for tok in range(vocab_size):
            candidate = prefix + (tok,)
            if len(candidate) == self.ngram_size and candidate in self._blocked:
                logits[tok] = float("-inf")

        return logits

    def reset(self) -> None:
        """Clear all blocked n-grams."""
        self._blocked.clear()


# ---------------------------------------------------------------------------
# DiversityPenalty
# ---------------------------------------------------------------------------


class DiversityPenalty:
    """Compute a per-token diversity bonus based on previous sequences.

    Args:
        penalty_type: 'hamming' (token overlap) or 'cosine' (embedding
            similarity, falls back to hamming when no embeddings available).
        alpha: Scaling factor for the bonus (higher → stronger push toward
            novel tokens).
    """

    def __init__(self, penalty_type: str = "hamming", alpha: float = 0.5) -> None:
        if penalty_type not in ("hamming", "cosine"):
            raise ValueError(f"penalty_type must be 'hamming' or 'cosine', got '{penalty_type}'")
        self.penalty_type = penalty_type
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_diversity_bonus(
        self,
        candidate_ids: torch.Tensor,
        previous_ids: list[torch.Tensor],
    ) -> torch.Tensor:
        """Return a (vocab_size,) bonus tensor.

        Tokens that do NOT appear in any previous sequence receive a bonus of
        ``alpha``; tokens that do appear receive 0.  This steers the current
        sample away from tokens already used in earlier samples.

        For the 'cosine' penalty type the same token-overlap heuristic is used
        (a full embedding lookup would require access to model weights that are
        not available here).

        Args:
            candidate_ids: 1-D LongTensor — vocabulary indices to evaluate.
                Typically ``torch.arange(vocab_size)``.
            previous_ids: List of 1-D LongTensors — completed (or partial)
                token sequences from earlier samples.

        Returns:
            Float tensor of shape (vocab_size,) with values in {0, alpha}.
        """
        vocab_size = candidate_ids.shape[0]
        bonus = torch.full((vocab_size,), self.alpha, dtype=torch.float)

        if not previous_ids:
            # No previous sequences — every token is equally "novel".
            return bonus

        # Collect the set of all token ids seen in previous sequences.
        seen: set[int] = set()
        for seq in previous_ids:
            seen.update(seq.tolist())

        # Tokens present in previous sequences get zero bonus.
        for tok in seen:
            if 0 <= tok < vocab_size:
                bonus[tok] = 0.0

        return bonus


# ---------------------------------------------------------------------------
# OutputDiversifier
# ---------------------------------------------------------------------------


class OutputDiversifier:
    """Generate multiple diverse completions from an AureliusTransformer model.

    Combines n-gram blocking and diversity bonuses to ensure that sampled
    completions are meaningfully different from each other.

    Args:
        model: A ``nn.Module`` whose forward method accepts
            ``(input_ids: LongTensor[1, seq])`` and returns a float tensor of
            shape ``(1, seq, vocab_size)`` (logits).
        strategy: Sampling strategy — 'nucleus', 'diverse_beam', or
            'stochastic_beam'.  Currently nucleus and stochastic_beam both
            use top-p nucleus sampling; 'diverse_beam' uses greedy argmax with
            diversity bonuses.
        diversity_penalty: Scaling factor for the per-token diversity bonus.
        ngram_size: N-gram size for cross-sample blocking.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "nucleus",
        diversity_penalty: float = 0.5,
        ngram_size: int = 3,
    ) -> None:
        if strategy not in ("nucleus", "diverse_beam", "stochastic_beam"):
            raise ValueError(
                f"strategy must be 'nucleus', 'diverse_beam', or 'stochastic_beam', "
                f"got '{strategy}'"
            )
        self.model = model
        self.strategy = strategy
        self.diversity_penalty = diversity_penalty
        self.ngram_size = ngram_size

        self._blocker = NgramBlocker(ngram_size=ngram_size)
        self._div_penalty = DiversityPenalty(penalty_type="hamming", alpha=diversity_penalty)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return last-position logits (vocab_size,).

        Args:
            input_ids: 1-D LongTensor of shape (seq_len,).

        Returns:
            1-D float tensor of shape (vocab_size,).
        """
        with torch.no_grad():
            out = self.model(input_ids.unsqueeze(0))  # (1, seq, vocab)
            # Support models that return a tuple (logits, ...) or just logits.
            if isinstance(out, tuple):
                out = out[0]
            logits = out[0, -1, :]  # last position, shape (vocab_size,)
        return logits

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> int:
        """Sample a single token from logits using nucleus (top-p) sampling.

        Args:
            logits: 1-D float tensor (vocab_size,).
            temperature: Softmax temperature.
            top_p: Nucleus probability threshold.

        Returns:
            Sampled token id (int).
        """
        scaled = logits / max(temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)

        # Nucleus filtering.
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        # Shift by one so the token that tips cumsum over top_p is included.
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-12)

        sampled = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[sampled].item())

    def _greedy_token(self, logits: torch.Tensor) -> int:
        """Return argmax token (used for diverse_beam strategy)."""
        return int(logits.argmax().item())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_diverse_batch(
        self,
        input_ids: torch.Tensor,
        n_samples: int,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate *n_samples* diverse completions.

        For each sample the n-gram blocker is updated with all previously
        generated sequences, and the diversity bonus is applied to logits
        before sampling.

        Args:
            input_ids: 1-D LongTensor prompt of shape (prompt_len,).
            n_samples: Number of completions to generate.
            max_new_tokens: Maximum number of new tokens per completion.
            temperature: Sampling temperature.

        Returns:
            LongTensor of shape (n_samples, max_new_tokens).
        """
        self._blocker.reset()
        completed: list[torch.Tensor] = []
        all_generated = torch.zeros(n_samples, max_new_tokens, dtype=torch.long)

        for sample_idx in range(n_samples):
            context = input_ids.clone()
            generated_tokens: list[int] = []

            for step in range(max_new_tokens):
                logits = self._get_logits(context)

                # Apply n-gram blocking.
                logits = self._blocker.apply_blocking_mask(logits, context)

                # Apply diversity bonus from previous sequences.
                vocab_size = logits.shape[0]
                bonus = self._div_penalty.compute_diversity_bonus(
                    torch.arange(vocab_size), completed
                )
                logits = logits + bonus

                # Sample next token according to strategy.
                if self.strategy == "diverse_beam":
                    token = self._greedy_token(logits)
                else:  # nucleus / stochastic_beam
                    token = self._sample_token(logits, temperature=temperature)

                generated_tokens.append(token)
                context = torch.cat([context, torch.tensor([token], dtype=torch.long)])

            seq_tensor = torch.tensor(generated_tokens, dtype=torch.long)
            all_generated[sample_idx] = seq_tensor

            # Update blocker with the full context (prompt + generated).
            self._blocker.update(context)
            completed.append(seq_tensor)

        return all_generated

    def compute_pairwise_diversity(self, sequences: torch.Tensor) -> float:
        """Compute mean pairwise token-level diversity across samples.

        Diversity for a pair (a, b) is ``1 - overlap_ratio`` where
        ``overlap_ratio = |tokens_in_common| / seq_len``.

        Args:
            sequences: LongTensor of shape (n_samples, seq_len).

        Returns:
            Mean pairwise diversity in [0, 1].  Returns 0.0 for fewer than 2
            sequences.
        """
        n = sequences.shape[0]
        if n < 2:
            return 0.0

        seq_len = sequences.shape[1]
        if seq_len == 0:
            return 0.0

        total_diversity = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                matches = (sequences[i] == sequences[j]).sum().item()
                overlap = matches / seq_len
                total_diversity += 1.0 - overlap
                count += 1

        return total_diversity / count if count > 0 else 0.0

    def filter_diverse(self, sequences: torch.Tensor, min_diversity: float = 0.3) -> torch.Tensor:
        """Remove near-duplicate sequences, keeping only the diverse subset.

        Iteratively adds sequences to a kept set; a new sequence is only kept
        if its minimum pairwise diversity to all already-kept sequences is
        >= *min_diversity*.  The first sequence is always kept.

        Args:
            sequences: LongTensor of shape (n_samples, seq_len).
            min_diversity: Minimum diversity threshold in [0, 1].

        Returns:
            LongTensor of shape (n_kept, seq_len) containing the diverse
            subset.  At least one sequence is always returned.
        """
        n = sequences.shape[0]
        if n == 0:
            return sequences

        kept_indices: list[int] = [0]

        for i in range(1, n):
            is_diverse = True
            for j in kept_indices:
                seq_len = sequences.shape[1]
                if seq_len == 0:
                    break
                matches = (sequences[i] == sequences[j]).sum().item()
                overlap = matches / seq_len
                diversity = 1.0 - overlap
                if diversity < min_diversity:
                    is_diverse = False
                    break
            if is_diverse:
                kept_indices.append(i)

        return sequences[kept_indices]
