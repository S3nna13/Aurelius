"""Adversarial robustness evaluator for transformer language models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RobustnessReport:
    """Aggregated robustness metrics for a batch of evaluated samples."""

    clean_accuracy: float
    attack_success_rate: float
    certified_radius: float
    semantic_preservation: float
    n_samples: int


class RobustnessEvaluator:
    """Evaluates adversarial robustness of an AureliusTransformer model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_last_token(self, input_ids: torch.Tensor) -> int:
        """Run a forward pass and return the top-1 token at the last position.

        Args:
            input_ids: (1, S) token id tensor.

        Returns:
            Predicted token id as an int.
        """
        _, logits, _ = self.model(input_ids)
        last_logits = logits[:, -1, :]  # (1, vocab_size)
        return int(last_logits.argmax(dim=-1).item())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_accuracy(
        self,
        input_ids_list: list[torch.Tensor],
        label_ids: list[int],
    ) -> float:
        """Fraction of samples where the model's last-token top-1 matches label_id.

        Args:
            input_ids_list: List of (1, S) input tensors.
            label_ids: Ground-truth token ids, one per sample.

        Returns:
            Accuracy in [0, 1].
        """
        if not input_ids_list:
            return 0.0
        correct = sum(
            1
            for ids, label in zip(input_ids_list, label_ids)
            if self._predict_last_token(ids) == label
        )
        return correct / len(input_ids_list)

    def attack_success_rate(
        self,
        input_ids_list: list[torch.Tensor],
        perturbed_ids_list: list[torch.Tensor],
        label_ids: list[int],
    ) -> float:
        """Fraction of samples that were correct on clean but wrong after perturbation.

        Args:
            input_ids_list: List of clean (1, S) input tensors.
            perturbed_ids_list: List of perturbed (1, S) input tensors.
            label_ids: Ground-truth token ids, one per sample.

        Returns:
            Attack success rate in [0, 1].
        """
        if not input_ids_list:
            return 0.0
        successes = 0
        for ids, p_ids, label in zip(input_ids_list, perturbed_ids_list, label_ids):
            clean_pred = self._predict_last_token(ids)
            if clean_pred == label:
                perturbed_pred = self._predict_last_token(p_ids)
                if perturbed_pred != label:
                    successes += 1
        return successes / len(input_ids_list)

    def semantic_preservation(
        self,
        input_ids_list: list[torch.Tensor],
        perturbed_ids_list: list[torch.Tensor],
    ) -> float:
        """Mean token overlap rate between original and perturbed sequences.

        Args:
            input_ids_list: List of (1, S) clean input tensors.
            perturbed_ids_list: List of (1, S) perturbed input tensors.

        Returns:
            Mean fraction of identical tokens in [0, 1].
        """
        if not input_ids_list:
            return 0.0
        overlap_rates: list[float] = []
        for ids, p_ids in zip(input_ids_list, perturbed_ids_list):
            orig = ids.view(-1)
            pert = p_ids.view(-1)
            n = orig.shape[0]
            if n == 0:
                overlap_rates.append(1.0)
            else:
                matches = int((orig == pert).sum().item())
                overlap_rates.append(matches / n)
        return sum(overlap_rates) / len(overlap_rates)

    def certified_lower_bound(
        self,
        input_ids: torch.Tensor,
        n_samples: int = 100,
        sigma: float = 0.1,
    ) -> float:
        """Estimate a certified radius lower bound via randomized smoothing.

        Adds Gaussian noise to the output of the last transformer layer via a
        forward hook and runs n_samples noisy forward passes.  The top-1 class
        vote fraction is used to derive a certified radius.

        Args:
            input_ids: (1, S) input tensor.
            n_samples: Number of noisy forward passes.
            sigma: Noise standard deviation.

        Returns:
            Certified radius lower bound (non-negative float).
        """
        last_layer = self.model.layers[-1]
        votes: list[int] = []
        normal = torch.distributions.Normal(0.0, 1.0)

        for _ in range(n_samples):
            # Closure captures a fresh noise tensor each iteration.
            def _hook(
                module: nn.Module,
                input: tuple,
                output: tuple,
                _sigma: float = sigma,
            ) -> tuple:
                hidden, kv = output
                noise = torch.randn_like(hidden) * _sigma
                return hidden + noise, kv

            handle = last_layer.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    _, logits, _ = self.model(input_ids)
                last_logits = logits[:, -1, :]
                pred = int(last_logits.argmax(dim=-1).item())
            finally:
                handle.remove()
            votes.append(pred)

        if not votes:
            return 0.0

        # Plurality class and vote fraction
        vote_tensor = torch.tensor(votes, dtype=torch.long)
        num_classes = logits.shape[-1]
        counts = torch.bincount(vote_tensor, minlength=num_classes)
        top_count = int(counts.max().item())
        vote_fraction = top_count / len(votes)

        # Clamp away from 0.5 boundary to keep icdf well-defined
        p_hat = max(vote_fraction, 0.5 + 1e-6)
        radius = float(sigma * normal.icdf(torch.tensor(p_hat)).item())
        return max(0.0, radius)

    def evaluate(
        self,
        input_ids_list: list[torch.Tensor],
        perturbed_ids_list: list[torch.Tensor],
        label_ids: list[int],
    ) -> RobustnessReport:
        """Compute all robustness metrics and return a RobustnessReport.

        Args:
            input_ids_list: List of clean (1, S) input tensors.
            perturbed_ids_list: List of perturbed (1, S) input tensors.
            label_ids: Ground-truth token ids, one per sample.

        Returns:
            RobustnessReport with aggregated metrics.
        """
        c_acc = self.clean_accuracy(input_ids_list, label_ids)
        asr = self.attack_success_rate(input_ids_list, perturbed_ids_list, label_ids)
        sem_pres = self.semantic_preservation(input_ids_list, perturbed_ids_list)

        # Use first sample for certified bound estimate
        if input_ids_list:
            cert_rad = self.certified_lower_bound(input_ids_list[0])
        else:
            cert_rad = 0.0

        return RobustnessReport(
            clean_accuracy=c_acc,
            attack_success_rate=asr,
            certified_radius=cert_rad,
            semantic_preservation=sem_pres,
            n_samples=len(input_ids_list),
        )
