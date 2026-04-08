"""Dataset difficulty scoring via model perplexity.

Scores each training example by the current model's perplexity.
High perplexity = difficult example (model doesn't know it well).
Low perplexity = easy example (model already knows this).

Used by curriculum learning to schedule easy->hard,
and by importance sampling to upweight hard examples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class DifficultyScore:
    """Difficulty score for a single training example."""
    index: int           # position in dataset
    perplexity: float    # model perplexity on this example
    loss: float          # mean cross-entropy loss
    n_tokens: int        # number of tokens in example


@dataclass
class DifficultyScoreConfig:
    batch_size: int = 8
    device: str = "cpu"
    percentile_easy: float = 33.0    # examples below this ppl percentile = "easy"
    percentile_hard: float = 66.0    # examples above this ppl percentile = "hard"


class DifficultyScorer:
    """Scores training examples by current model perplexity.

    Usage:
        scorer = DifficultyScorer(model, cfg)
        scores = scorer.score_dataset(dataloader)
        easy, medium, hard = scorer.partition(scores)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DifficultyScoreConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or DifficultyScoreConfig()

    @torch.no_grad()
    def score_batch(
        self,
        input_ids: torch.Tensor,   # (B, S)
        labels: torch.Tensor,      # (B, S) -- shifted labels
    ) -> list[DifficultyScore]:
        """Score a batch of examples. Returns one DifficultyScore per example."""
        self.model.eval()
        device = self.cfg.device
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass -- model returns (loss, logits, pkv)
        _, logits, _ = self.model(input_ids)

        B, S, V = logits.shape
        scores: list[DifficultyScore] = []

        for i in range(B):
            # logits[:, :-1] predicts labels[:, 1:] (standard causal LM shift)
            logits_i = logits[i, :-1, :]          # (S-1, V)
            targets_i = labels[i, 1:]              # (S-1,)

            # Mask out padding tokens if labels contain -100
            valid_mask = targets_i != -100
            n_tokens = int(valid_mask.sum().item())

            if n_tokens == 0:
                scores.append(DifficultyScore(index=0, perplexity=float("inf"), loss=float("inf"), n_tokens=0))
                continue

            per_token_loss = F.cross_entropy(
                logits_i[valid_mask],
                targets_i[valid_mask],
                reduction="mean",
            )
            mean_loss = per_token_loss.item()
            ppl = math.exp(min(mean_loss, 20.0))  # cap to avoid overflow

            scores.append(DifficultyScore(
                index=0,           # placeholder; caller sets index
                perplexity=ppl,
                loss=mean_loss,
                n_tokens=n_tokens,
            ))

        return scores

    def score_dataset(
        self,
        dataloader: DataLoader,
    ) -> list[DifficultyScore]:
        """Score all examples in a DataLoader.

        DataLoader should yield {"input_ids": Tensor, "labels": Tensor} dicts.
        Returns list of DifficultyScore sorted by index.
        """
        all_scores: list[DifficultyScore] = []
        global_idx = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
            else:
                input_ids, labels = batch[0], batch[1]

            batch_scores = self.score_batch(input_ids, labels)
            for score in batch_scores:
                score.index = global_idx
                global_idx += 1
                all_scores.append(score)

        all_scores.sort(key=lambda s: s.index)
        return all_scores

    def partition(
        self,
        scores: list[DifficultyScore],
    ) -> tuple[list[DifficultyScore], list[DifficultyScore], list[DifficultyScore]]:
        """Split scores into (easy, medium, hard) based on config percentiles.

        Returns:
            (easy, medium, hard) -- three lists of DifficultyScore
        """
        if not scores:
            return [], [], []

        ppls = np.array([s.perplexity for s in scores], dtype=np.float64)
        thresh_easy = float(np.percentile(ppls, self.cfg.percentile_easy))
        thresh_hard = float(np.percentile(ppls, self.cfg.percentile_hard))

        easy: list[DifficultyScore] = []
        medium: list[DifficultyScore] = []
        hard: list[DifficultyScore] = []

        for score in scores:
            if score.perplexity <= thresh_easy:
                easy.append(score)
            elif score.perplexity > thresh_hard:
                hard.append(score)
            else:
                medium.append(score)

        return easy, medium, hard

    def get_sampling_weights(
        self,
        scores: list[DifficultyScore],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute importance sampling weights proportional to perplexity.

        weight[i] = ppl[i]^(1/temperature) / sum(ppl[j]^(1/temperature))
        Higher temperature -> more uniform (less emphasis on hard examples).

        Returns normalized weights tensor of shape (N,).
        """
        ppls = torch.tensor([s.perplexity for s in scores], dtype=torch.float32)
        ppls = ppls.clamp(min=1e-8)
        exponent = 1.0 / max(temperature, 1e-8)
        raw = ppls ** exponent
        weights = raw / raw.sum()
        return weights

    def save_scores(self, scores: list[DifficultyScore], path: str | Path) -> None:
        """Save scores to .npy file as structured array."""
        path = Path(path)
        dtype = np.dtype([
            ("index", np.int64),
            ("perplexity", np.float64),
            ("loss", np.float64),
            ("n_tokens", np.int64),
        ])
        arr = np.empty(len(scores), dtype=dtype)
        for i, s in enumerate(scores):
            arr[i] = (s.index, s.perplexity, s.loss, s.n_tokens)
        np.save(path, arr)

    def load_scores(self, path: str | Path) -> list[DifficultyScore]:
        """Load scores from .npy file."""
        path = Path(path)
        arr = np.load(path, allow_pickle=False)
        return [
            DifficultyScore(
                index=int(row["index"]),
                perplexity=float(row["perplexity"]),
                loss=float(row["loss"]),
                n_tokens=int(row["n_tokens"]),
            )
            for row in arr
        ]
