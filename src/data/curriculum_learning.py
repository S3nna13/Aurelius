"""Curriculum Learning and Data Selection for LLM training.

Implements difficulty scoring, pacing functions, curriculum-aware data loaders,
and data quality filters. Supports easy->hard (curriculum), hard->easy
(anti-curriculum), and mixed strategies.

References:
    - Bengio et al. (2009) "Curriculum Learning"
    - Platanios et al. (2019) "Competence-based Curriculum Learning" (Baby Steps)
    - Swayamdipta et al. (2020) "Dataset Cartography"
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DifficultyScorer
# ---------------------------------------------------------------------------


class DifficultyScorer:
    """Compute per-example difficulty scores from various signals.

    All methods operate on batched input_ids tensors and return per-example
    scalar scores as 1-D Tensors of shape [B].
    """

    @torch.no_grad()
    def perplexity_score(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:  # [B]
        """Per-example perplexity: exp(mean cross-entropy over tokens).

        Performs a causal LM forward pass and computes per-example CE loss,
        then exponentiates.  Returns values >= 1.0.

        Args:
            model: nn.Module that returns (loss, logits, *) or just logits.
            input_ids: LongTensor [B, T].

        Returns:
            Perplexity tensor [B], values >= 1.0.
        """
        model.train(False)
        B, T = input_ids.shape

        # Forward — accept models that return (loss, logits, ...) or just logits.
        out = model(input_ids)
        if isinstance(out, tuple):
            logits = out[1] if len(out) > 1 else out[0]
        else:
            logits = out  # [B, T, V]

        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
        shift_labels = input_ids[:, 1:].contiguous()  # [B, T-1]

        # Compute per-token log-probs
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
        # Gather the log-prob of the true token
        token_lp = log_probs.gather(
            dim=2,
            index=shift_labels.unsqueeze(2),
        ).squeeze(2)  # [B, T-1]

        # Mean NLL per example, then exp
        mean_nll = -token_lp.mean(dim=1)  # [B]
        # Cap to avoid float overflow (perplexity > e^20 is meaningless)
        mean_nll = mean_nll.clamp(max=20.0)
        perplexity = mean_nll.exp()  # [B]
        return perplexity

    def length_score(
        self,
        input_ids: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:  # [B]
        """Normalized sequence length score in [0, 1].

        Longer sequences are scored higher (harder).  Token id 0 is treated
        as padding and excluded from the count.  If all sequences have equal
        length, returns a zero tensor.

        Args:
            input_ids: LongTensor [B, T].

        Returns:
            Float tensor [B] in [0, 1].
        """
        # Count non-padding tokens (treat 0 as pad)
        lengths = (input_ids != 0).sum(dim=1).float()  # [B]
        max_len = lengths.max()
        if max_len.item() == 0:
            return torch.zeros(input_ids.size(0), dtype=torch.float32)
        return lengths / max_len

    def vocabulary_richness(
        self,
        input_ids: torch.Tensor,  # [B, T]
        vocab_size: int,
    ) -> torch.Tensor:  # [B]
        """Unique-token ratio per example: |unique tokens| / sequence_length.

        Values in [0, 1].  A sequence with all identical tokens has richness
        near 0; a sequence with all unique tokens has richness 1.

        Args:
            input_ids: LongTensor [B, T].
            vocab_size: total vocabulary size (kept for API consistency).

        Returns:
            Float tensor [B] in [0, 1].
        """
        B, T = input_ids.shape
        scores = torch.zeros(B, dtype=torch.float32)
        for i in range(B):
            seq = input_ids[i]
            n_unique = seq.unique().numel()
            scores[i] = n_unique / T if T > 0 else 0.0
        return scores

    def combined_score(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,  # [B, T]
        weights: dict[str, float],
    ) -> torch.Tensor:  # [B]
        """Weighted combination of perplexity, length, and vocabulary richness.

        Each component is normalised to [0, 1] before weighting so that the
        magnitude of perplexity doesn't dominate.

        Args:
            model: language model for perplexity computation.
            input_ids: LongTensor [B, T].
            weights: dict with keys "perplexity", "length", "vocab".
                     Missing keys default to 0.0.

        Returns:
            Float tensor [B] combined score.
        """
        w_ppl = weights.get("perplexity", 0.0)
        w_len = weights.get("length", 0.0)
        w_voc = weights.get("vocab", 0.0)

        B = input_ids.size(0)
        combined = torch.zeros(B, dtype=torch.float32)

        if w_ppl != 0.0:
            ppl = self.perplexity_score(model, input_ids).float()
            mn, mx = ppl.min(), ppl.max()
            if mx > mn:
                ppl_norm = (ppl - mn) / (mx - mn)
            else:
                ppl_norm = torch.zeros_like(ppl)
            combined = combined + w_ppl * ppl_norm

        if w_len != 0.0:
            combined = combined + w_len * self.length_score(input_ids).float()

        if w_voc != 0.0:
            vocab_size = int(input_ids.max().item()) + 1
            combined = combined + w_voc * self.vocabulary_richness(input_ids, vocab_size).float()

        return combined


# ---------------------------------------------------------------------------
# PacingFunction
# ---------------------------------------------------------------------------


class PacingFunction:
    """Static-method collection for curriculum pacing schedules.

    Each method returns the *fraction* of the dataset (sorted by difficulty)
    to expose at the given training step.  Values are clamped to [0, 1].
    """

    @staticmethod
    def linear_pacing(
        step: int,
        total_steps: int,
        start_frac: float = 0.1,
    ) -> float:
        """Linearly ramp data fraction from start_frac to 1.0.

        At step=0 returns start_frac; at step=total_steps returns 1.0.

        Args:
            step:        current training step (0-indexed).
            total_steps: total number of training steps.
            start_frac:  starting fraction of data to use.

        Returns:
            Float in [start_frac, 1.0].
        """
        if total_steps <= 0:
            return 1.0
        progress = min(step / total_steps, 1.0)
        frac = start_frac + (1.0 - start_frac) * progress
        return float(min(max(frac, 0.0), 1.0))

    @staticmethod
    def competence_pacing(
        step: int,
        total_steps: int,
        c0: float = 0.01,
    ) -> float:
        """Baby Steps competence-based pacing (Platanios et al., 2019).

        c(t) = sqrt(t/T * (1 - c0^2) + c0^2)

        Monotonically increases from c0 at t=0 to 1.0 at t=T.

        Args:
            step:        current training step.
            total_steps: total number of training steps.
            c0:          initial competence (default 0.01).

        Returns:
            Float in [c0, 1.0].
        """
        if total_steps <= 0:
            return 1.0
        t_norm = min(step / total_steps, 1.0)
        val = math.sqrt(t_norm * (1.0 - c0**2) + c0**2)
        return float(min(max(val, 0.0), 1.0))

    @staticmethod
    def exponential_pacing(
        step: int,
        total_steps: int,
        k: float = 5.0,
    ) -> float:
        """Exponential pacing: 1 - exp(-k * step / total_steps).

        Starts near 0, approaches 1 asymptotically.

        Args:
            step:        current training step.
            total_steps: total number of training steps.
            k:           rate parameter (larger = faster approach to 1).

        Returns:
            Float in [0, 1).
        """
        if total_steps <= 0:
            return 1.0
        t_norm = min(step / total_steps, 1.0)
        val = 1.0 - math.exp(-k * t_norm)
        return float(min(max(val, 0.0), 1.0))

    @staticmethod
    def step_pacing(
        step: int,
        milestones: list[int],
        fracs: list[float],
    ) -> float:
        """Step-function pacing: fixed fractions at milestone steps.

        Returns fracs[i] for the largest milestone[i] <= step.
        If step < milestones[0], returns fracs[0].

        Args:
            step:       current training step.
            milestones: ascending list of step thresholds.
            fracs:      fractions to use at each milestone.
                        len(fracs) must equal len(milestones).

        Returns:
            Float fraction.
        """
        if not milestones or not fracs:
            return 1.0
        assert len(milestones) == len(fracs), "milestones and fracs must have the same length"  # noqa: S101
        current_frac = fracs[0]
        for milestone, frac in zip(milestones, fracs):
            if step >= milestone:
                current_frac = frac
            else:
                break
        return float(current_frac)


# ---------------------------------------------------------------------------
# CurriculumDataset
# ---------------------------------------------------------------------------


class CurriculumDataset:
    """Curriculum-aware dataset wrapper.

    Stores pre-scored examples and exposes batches ordered from easiest to
    hardest.  The fraction of data available at each step is governed by a
    caller-supplied pacing function.

    Args:
        input_ids: LongTensor [N, T] of all training sequences.
        scores:    Float tensor [N] of per-example difficulty scores.
                   Lower score = easier example.
        pacing_fn: callable(step: int) -> float in [0, 1].
    """

    def __init__(
        self,
        input_ids: torch.Tensor,  # [N, T]
        scores: torch.Tensor,  # [N]
        pacing_fn: Callable[[int], float],
    ) -> None:
        assert input_ids.size(0) == scores.size(0), (  # noqa: S101
            "input_ids and scores must have the same first dimension"
        )
        self.input_ids = input_ids
        self.scores = scores
        self.pacing_fn = pacing_fn
        self.step: int = 0

        # Pre-sort indices from easiest (lowest score) to hardest (highest)
        self._sorted_indices: list[int] = torch.argsort(scores).tolist()

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def advance(self, n_steps: int = 1) -> None:
        """Advance the internal step counter by n_steps."""
        self.step += n_steps

    # ------------------------------------------------------------------
    # Batch sampling
    # ------------------------------------------------------------------

    def _available_pool(self) -> list[int]:
        """Indices of examples available at the current pacing fraction."""
        frac = self.pacing_fn(self.step)
        frac = min(max(frac, 0.0), 1.0)
        n_avail = max(1, int(math.ceil(frac * len(self._sorted_indices))))
        return self._sorted_indices[:n_avail]

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """Sample a batch from the currently available (easiest) examples.

        At pacing fraction p, only the easiest ceil(p * N) examples are
        eligible for sampling.  Samples uniformly without replacement when
        batch_size <= pool size, with replacement otherwise.

        Args:
            batch_size: number of examples per batch.

        Returns:
            LongTensor [B, T].
        """
        pool = self._available_pool()
        if batch_size <= len(pool):
            chosen = random.sample(pool, batch_size)
        else:
            chosen = random.choices(pool, k=batch_size)
        return self.input_ids[torch.tensor(chosen)]

    def get_anti_curriculum_batch(self, batch_size: int) -> torch.Tensor:
        """Sample a batch from the hardest available examples (reverse curriculum).

        Uses the same pacing-fraction pool but samples from the hard end.

        Args:
            batch_size: number of examples per batch.

        Returns:
            LongTensor [B, T].
        """
        pool = self._available_pool()
        # Hard end: last elements of the sorted (ascending difficulty) pool
        hard_pool = pool[-batch_size:] if len(pool) >= batch_size else pool
        if batch_size <= len(hard_pool):
            chosen = random.sample(hard_pool, batch_size)
        else:
            chosen = random.choices(hard_pool, k=batch_size)
        return self.input_ids[torch.tensor(chosen)]


# ---------------------------------------------------------------------------
# DataSelectionFilter
# ---------------------------------------------------------------------------


class DataSelectionFilter:
    """Filters for dataset quality and deduplication.

    Args:
        threshold: difficulty score threshold for filter_by_score.
                   Examples with score <= threshold are kept.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def filter_by_score(
        self,
        input_ids: torch.Tensor,  # [N, T]
        scores: torch.Tensor,  # [N]
    ) -> torch.Tensor:  # [M, T]
        """Keep only examples whose difficulty score is at or below threshold.

        Args:
            input_ids: LongTensor [N, T].
            scores:    Float tensor [N].

        Returns:
            LongTensor [M, T] with M <= N.
        """
        mask = scores <= self.threshold
        return input_ids[mask]

    def deduplication_filter(
        self,
        input_ids: torch.Tensor,  # [N, T]
        sim_threshold: float = 0.9,
    ) -> torch.Tensor:  # [M, T]
        """Remove near-duplicate sequences using Jaccard similarity on token sets.

        Two sequences are near-duplicates if their token-set Jaccard
        similarity >= sim_threshold.  Keeps the first occurrence.

        Jaccard(A, B) = |A intersect B| / |A union B|

        Args:
            input_ids:     LongTensor [N, T].
            sim_threshold: Jaccard threshold (default 0.9).

        Returns:
            LongTensor [M, T] with duplicates removed.
        """
        N = input_ids.size(0)
        token_sets: list[set] = []
        for i in range(N):
            token_sets.append(set(input_ids[i].tolist()))

        keep: list[int] = []
        for i in range(N):
            is_dup = False
            set_i = token_sets[i]
            for j in keep:
                set_j = token_sets[j]
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                if union == 0:
                    jaccard = 1.0
                else:
                    jaccard = intersection / union
                if jaccard >= sim_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(i)

        if not keep:
            return input_ids[:0]
        return input_ids[torch.tensor(keep)]

    def quality_filter(
        self,
        input_ids: torch.Tensor,  # [N, T]
        min_len: int = 4,
        max_rep_ratio: float = 0.5,
    ) -> torch.Tensor:  # [M, T]
        """Remove sequences that are too short or highly repetitive.

        A sequence is removed if:
        - Its non-padding (non-zero) length < min_len, OR
        - The most-frequent token occupies > max_rep_ratio of all tokens.

        Args:
            input_ids:     LongTensor [N, T].
            min_len:       minimum non-padding sequence length.
            max_rep_ratio: maximum fraction for the most common token.

        Returns:
            LongTensor [M, T] with low-quality sequences removed.
        """
        N = input_ids.size(0)
        keep: list[int] = []
        for i in range(N):
            seq = input_ids[i]
            # Non-padding tokens (token 0 treated as pad)
            non_pad = seq[seq != 0]
            length = non_pad.numel()
            if length < min_len:
                continue
            counts = torch.bincount(non_pad.long())
            max_count = int(counts.max().item())
            rep_ratio = max_count / length
            if rep_ratio > max_rep_ratio:
                continue
            keep.append(i)

        if not keep:
            return input_ids[:0]
        return input_ids[torch.tensor(keep)]


# ---------------------------------------------------------------------------
# CurriculumTrainer
# ---------------------------------------------------------------------------


class CurriculumTrainer:
    """Minimal curriculum training loop around a CurriculumDataset.

    Performs single-step training with Adam and tracks curriculum statistics.

    Args:
        model:   nn.Module language model.
        dataset: CurriculumDataset providing curriculum-aware batches.
        lr:      learning rate for Adam optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: CurriculumDataset,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._n_seen_easy: int = 0
        self._n_seen_hard: int = 0

    def train_step(self) -> tuple:
        """Perform one curriculum training step.

        Samples a batch from the current pacing fraction, computes causal LM
        cross-entropy loss, and updates model weights.

        Returns:
            (loss: float, step: int) after advancing the dataset step.
        """
        self.model.train(True)
        batch_size = getattr(self.dataset, "batch_size", 4)
        input_ids = self.dataset.get_batch(batch_size)

        out = self.model(input_ids)
        if isinstance(out, tuple):
            logits = out[1] if len(out) > 1 else out[0]
        else:
            logits = out  # [B, T, V]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track stats
        pool = self.dataset._available_pool()
        mid = max(len(pool) // 2, 1)
        easy_set = set(pool[:mid])
        hard_set = set(pool[mid:])
        for idx in pool[:batch_size]:
            if idx in easy_set:
                self._n_seen_easy += 1
            elif idx in hard_set:
                self._n_seen_hard += 1

        self.dataset.advance(1)
        return float(loss.item()), self.dataset.step

    def get_training_stats(self) -> dict:
        """Return a snapshot of current training statistics.

        Returns:
            dict with keys: current_frac, step, n_seen_easy, n_seen_hard.
        """
        current_frac = self.dataset.pacing_fn(self.dataset.step)
        return {
            "current_frac": float(current_frac),
            "step": self.dataset.step,
            "n_seen_easy": self._n_seen_easy,
            "n_seen_hard": self._n_seen_hard,
        }


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning experiments.

    Attributes:
        pacing:      pacing schedule name: "competence", "linear",
                     "exponential", or "step".
        start_frac:  initial data fraction (for linear/competence pacing).
        c0:          initial competence (for competence pacing).
        total_steps: total training steps (for continuous pacing schedules).
        batch_size:  training batch size.
    """

    pacing: str = "competence"
    start_frac: float = 0.1
    c0: float = 0.01
    total_steps: int = 1000
    batch_size: int = 4

    def build_pacing_fn(self) -> Callable[[int], float]:
        """Construct the pacing callable from config fields.

        Returns:
            Callable mapping step -> fraction float.
        """
        if self.pacing == "competence":
            c0 = self.c0
            total = self.total_steps
            return lambda step: PacingFunction.competence_pacing(step, total, c0)
        elif self.pacing == "linear":
            sf = self.start_frac
            total = self.total_steps
            return lambda step: PacingFunction.linear_pacing(step, total, sf)
        elif self.pacing == "exponential":
            total = self.total_steps
            return lambda step: PacingFunction.exponential_pacing(step, total)
        elif self.pacing == "step":
            T = self.total_steps
            milestones = [0, T // 4, T // 2, 3 * T // 4]
            fracs = [0.25, 0.5, 0.75, 1.0]
            return lambda step: PacingFunction.step_pacing(step, milestones, fracs)
        else:
            raise ValueError(f"Unknown pacing schedule: {self.pacing!r}")
