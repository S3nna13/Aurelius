"""Diverse beam search decoding for AureliusTransformer.

Diverse beam search (DBS) partitions beams into groups and applies an
inter-group diversity penalty so that each group explores a different region
of the output space.  The algorithm is described in:

    Vijayakumar et al., 2016 — "Diverse Beam Search: Decoding Diverse
    Solutions from Neural Sequence Models"
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DiverseBeamConfig:
    """Configuration for diverse beam search decoding."""

    beam_width: int = 4
    """Total number of beams (shared across all groups)."""

    n_groups: int = 2
    """Number of diversity groups.  beam_width must be divisible by n_groups."""

    diversity_penalty: float = 0.5
    """Penalty subtracted from logits for tokens already chosen by earlier groups."""

    max_new_tokens: int = 50
    """Maximum number of tokens to generate beyond the prompt."""

    length_penalty: float = 1.0
    """Exponent for length normalisation: score = log_prob / length^length_penalty."""

    temperature: float = 1.0
    """Softmax temperature applied to logits before scoring (>1 → flatter)."""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_diversity_penalty(
    prev_group_tokens: list[int],
    vocab_size: int,
    penalty: float,
) -> torch.Tensor:
    """Build a (vocab_size,) diversity-penalty tensor.

    Tokens that appear in *prev_group_tokens* receive a penalty of
    ``-penalty``; all other positions are 0.

    Args:
        prev_group_tokens: Token ids chosen by earlier groups at the current
            decoding step.
        vocab_size: Size of the vocabulary.
        penalty: Non-negative diversity penalty value.

    Returns:
        Float tensor of shape (vocab_size,).
    """
    pen = torch.zeros(vocab_size, dtype=torch.float)
    if prev_group_tokens and penalty > 0.0:
        # Clamp indices to valid range to avoid index errors.
        valid = [t for t in prev_group_tokens if 0 <= t < vocab_size]
        if valid:
            idx = torch.tensor(valid, dtype=torch.long)
            pen[idx] = -penalty
    return pen


def group_beam_search_step(
    logits: torch.Tensor,
    beams: list[tuple[list[int], float]],
    diversity_penalty: torch.Tensor,
    beam_width: int,
    temperature: float = 1.0,
) -> list[tuple[list[int], float]]:
    """Perform one beam-search expansion step for a single group.

    Args:
        logits: 1-D tensor of shape (vocab_size,) — raw next-token logits
            produced by the model for the *last* position of the best beam in
            this group.
        beams: Current beams for this group, each a (token_sequence, score)
            tuple.  Scores are length-normalised log-probabilities.
        diversity_penalty: (vocab_size,) additive penalty tensor (typically
            non-positive) from :func:`compute_diversity_penalty`.
        beam_width: Number of candidates to keep.
        temperature: Softmax temperature.

    Returns:
        New list of up to *beam_width* (token_sequence, score) tuples sorted
        by score descending.
    """
    vocab_size = logits.shape[0]
    candidates: list[tuple[list[int], float]] = []

    for seq, score in beams:
        # Apply temperature.
        scaled = logits.float() / max(temperature, 1e-8)
        # Apply diversity penalty (additive in logit space).
        scaled = scaled + diversity_penalty

        log_probs = F.log_softmax(scaled, dim=-1)

        k = min(beam_width, vocab_size)
        topk_lp, topk_idx = torch.topk(log_probs, k)

        # Recover the *un-normalised* accumulated log-prob from the current
        # score.  score = accumulated_lp / length^lp, so we cannot reverse
        # without knowing length_penalty; instead we track the raw cumulative
        # log-prob separately by convention: we store raw accumulated log-prob
        # as score when length_penalty=1 and length=1 for simplicity.  Full
        # normalisation happens at the end of search().
        # To keep the step self-contained we use the raw accumulated log-prob
        # (score IS the raw log-prob for this function; normalisation applied
        # externally).
        for i in range(k):
            token_id = int(topk_idx[i].item())
            new_score = score + float(topk_lp[i].item())
            candidates.append((seq + [token_id], new_score))

    # Sort descending by raw accumulated log-prob and keep top beam_width.
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:beam_width]


# ---------------------------------------------------------------------------
# Edit-distance helper (pure Python / PyTorch, no scipy)
# ---------------------------------------------------------------------------


def _token_edit_distance(a: list[int], b: list[int]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    m, n = len(a), len(b)
    # Allocate DP table (two rows suffice).
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def compute_sequence_diversity(sequences: list[torch.Tensor]) -> float:
    """Compute mean pairwise token-level edit distance between sequences.

    Args:
        sequences: List of 1-D LongTensors.

    Returns:
        Mean pairwise edit distance (>= 0).  Returns 0.0 for fewer than 2
        sequences.
    """
    n = len(sequences)
    if n < 2:
        return 0.0

    token_lists = [s.tolist() for s in sequences]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _token_edit_distance(token_lists[i], token_lists[j])
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DiverseBeamSearch:
    """Diverse beam search decoder.

    The model callable receives a 1-D LongTensor (token ids) and returns a
    1-D float tensor of shape (vocab_size,) representing raw logits for the
    next token.  This matches the simple mock pattern used in tests; adapters
    for real AureliusTransformer models can be created externally.

    Args:
        model_fn: Callable that maps a 1-D token-id tensor to a 1-D logit
            tensor of shape (vocab_size,).
        config: :class:`DiverseBeamConfig` instance.
        eos_token_id: Token id that signals end of sequence.
    """

    def __init__(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        config: DiverseBeamConfig,
        eos_token_id: int = 2,
    ) -> None:
        self.model_fn = model_fn
        self.config = config
        self.eos_token_id = eos_token_id

        if config.beam_width % config.n_groups != 0:
            raise ValueError(
                f"beam_width ({config.beam_width}) must be divisible by "
                f"n_groups ({config.n_groups})."
            )
        self._beams_per_group = config.beam_width // config.n_groups

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_groups(self, prompt: torch.Tensor) -> list[list[tuple[list[int], float]]]:
        """Create n_groups groups, each with beam_width // n_groups beams.

        All beams start from the prompt token ids with a raw log-prob of 0.

        Args:
            prompt: 1-D LongTensor of prompt token ids.

        Returns:
            List of length n_groups.  Each element is a list of
            (token_sequence, score) tuples of length beam_width // n_groups.
        """
        prompt_tokens = prompt.tolist()
        initial_beam: tuple[list[int], float] = (list(prompt_tokens), 0.0)
        groups: list[list[tuple[list[int], float]]] = []
        for _ in range(self.config.n_groups):
            groups.append([initial_beam] * self._beams_per_group)
        return groups

    # ------------------------------------------------------------------
    # Internal step
    # ------------------------------------------------------------------

    def _step_all_groups(
        self,
        groups: list[list[tuple[list[int], float]]],
    ) -> list[list[tuple[list[int], float]]]:
        """Run one decoding step across all groups with diversity penalty."""
        cfg = self.config
        new_groups: list[list[tuple[list[int], float]]] = []
        prev_group_tokens: list[int] = []

        for g, beams in enumerate(groups):
            # Use the best (first) beam's sequence to query the model.
            best_seq = beams[0][0]
            input_ids = torch.tensor(best_seq, dtype=torch.long)

            with torch.no_grad():
                logits = self.model_fn(input_ids)  # (vocab_size,)

            vocab_size = logits.shape[0]

            # Build diversity penalty from tokens chosen by earlier groups.
            div_pen = compute_diversity_penalty(
                prev_group_tokens, vocab_size, cfg.diversity_penalty
            )

            new_beams = group_beam_search_step(
                logits=logits,
                beams=beams,
                diversity_penalty=div_pen,
                beam_width=self._beams_per_group,
                temperature=cfg.temperature,
            )
            new_groups.append(new_beams)

            # Collect token(s) chosen by this group for the next group's penalty.
            # Use the top-beam's last token.
            if new_beams:
                prev_group_tokens.append(new_beams[0][0][-1])

        return new_groups

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    def search(self, prompt: torch.Tensor) -> list[torch.Tensor]:
        """Run diverse beam search and return the best sequence per group.

        Args:
            prompt: 1-D LongTensor of prompt token ids.

        Returns:
            List of n_groups 1-D LongTensors.  Each tensor is the best
            sequence found for that group (prompt prefix included).
        """
        cfg = self.config
        groups = self.initialize_groups(prompt)

        for _step in range(cfg.max_new_tokens):
            groups = self._step_all_groups(groups)

            # Early stopping: check if best beam in every group ends with EOS.
            all_done = all(
                beams and beams[0][0] and beams[0][0][-1] == self.eos_token_id for beams in groups
            )
            if all_done:
                break

        # Extract best beam per group.
        results: list[torch.Tensor] = []
        for beams in groups:
            best_seq, _ = beams[0]
            results.append(torch.tensor(best_seq, dtype=torch.long))
        return results

    def search_with_scores(self, prompt: torch.Tensor) -> list[tuple[torch.Tensor, float]]:
        """Run diverse beam search and return (sequence, normalised_score) pairs.

        Scores are length-normalised: raw_log_prob / len^length_penalty.
        Results are sorted by score descending.

        Args:
            prompt: 1-D LongTensor of prompt token ids.

        Returns:
            List of (sequence_tensor, normalised_score) tuples, sorted by
            score descending.
        """
        cfg = self.config
        groups = self.initialize_groups(prompt)

        for _step in range(cfg.max_new_tokens):
            groups = self._step_all_groups(groups)

            all_done = all(
                beams and beams[0][0] and beams[0][0][-1] == self.eos_token_id for beams in groups
            )
            if all_done:
                break

        output: list[tuple[torch.Tensor, float]] = []
        for beams in groups:
            best_seq, raw_score = beams[0]
            length = len(best_seq)
            if length > 0:
                norm_score = raw_score / (length**cfg.length_penalty)
            else:
                norm_score = raw_score
            output.append((torch.tensor(best_seq, dtype=torch.long), norm_score))

        output.sort(key=lambda x: x[1], reverse=True)
        return output
