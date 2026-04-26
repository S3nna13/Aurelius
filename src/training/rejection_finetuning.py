"""
Rejection Fine-Tuning (RFT / Self-Improvement)

Iterative self-improvement where a model generates multiple solutions,
filters to correct ones via a verifier, and trains on filtered correct
trajectories. Based on Yuan et al. 2023 and related work.

Pure PyTorch only — no transformers, einops, trl, xformers, flash_attn,
bitsandbytes, peft, diffusers, datasets, accelerate, or deepspeed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Solution Verifiers
# ---------------------------------------------------------------------------


class SolutionVerifier(ABC):
    """Abstract base class for verifying whether a generated solution is correct."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def verify(self, prompt_ids: Tensor, solution_ids: Tensor, ground_truth: Any) -> bool:
        """Return True if solution is considered correct for the given prompt."""


class ExactMatchVerifier(SolutionVerifier):
    """Verify by checking whether the tail of solution_ids matches ground_truth token IDs."""

    def __init__(self) -> None:
        super().__init__()

    def verify(self, prompt_ids: Tensor, solution_ids: Tensor, ground_truth: Tensor) -> bool:
        gt = ground_truth.to(solution_ids.device)
        n = gt.numel()
        if solution_ids.numel() < n:
            return False
        tail = solution_ids[-n:]
        return bool(torch.all(tail == gt).item())


class RewardVerifier(SolutionVerifier):
    """Verify by checking whether a reward function exceeds a threshold."""

    def __init__(self, reward_fn: Callable[[Tensor], float], threshold: float = 0.5) -> None:
        super().__init__()
        self.reward_fn = reward_fn
        self.threshold = threshold

    def verify(self, prompt_ids: Tensor, solution_ids: Tensor, ground_truth: Any) -> bool:
        reward = self.reward_fn(solution_ids)
        return float(reward) > self.threshold


class EnsembleVerifier(SolutionVerifier):
    """Verify by combining multiple verifiers with AND or OR logic."""

    def __init__(self, verifiers: list[SolutionVerifier], require_all: bool = True) -> None:
        super().__init__()
        if len(verifiers) == 0:
            raise ValueError("verifiers list must not be empty")
        self.verifiers = verifiers
        self.require_all = require_all

    def verify(self, prompt_ids: Tensor, solution_ids: Tensor, ground_truth: Any) -> bool:
        results = [v.verify(prompt_ids, solution_ids, ground_truth) for v in self.verifiers]
        if self.require_all:
            return all(results)
        return any(results)


# ---------------------------------------------------------------------------
# Solution Sampler
# ---------------------------------------------------------------------------


class SolutionSampler:
    """Sample multiple candidate solutions autoregressively from a model."""

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 8,
        temperature: float = 0.8,
        max_new_tokens: int = 32,
    ) -> None:
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def _generate_one(self, prompt_ids: Tensor, temperature: float) -> tuple[Tensor, float]:
        """Generate a single sequence and return (tokens, log_prob_sum)."""
        ids = prompt_ids.clone()  # (seq,)
        log_prob_accum = 0.0

        for _ in range(self.max_new_tokens):
            # model may return logits directly (shape [seq, vocab]) or a tuple
            out = self.model(ids.unsqueeze(0))  # (1, seq, vocab) or tuple
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            # logits: (1, seq, vocab) or (1, vocab)
            if logits.dim() == 3:
                next_logits = logits[0, -1, :]  # (vocab,)
            elif logits.dim() == 2:
                next_logits = logits[0, :]
            else:
                next_logits = logits

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                lp = 0.0
            else:
                scaled = next_logits / temperature
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                lp = float(F.log_softmax(scaled, dim=-1)[next_token.item()].item())

            log_prob_accum += lp
            ids = torch.cat([ids, next_token.squeeze(0).unsqueeze(0)], dim=0)

        # Return only the generated part
        generated = ids[prompt_ids.numel() :]
        return generated, log_prob_accum

    def sample(self, prompt_ids: Tensor) -> tuple[list[Tensor], list[float]]:
        """Sample n_samples sequences. Returns (solutions, log_probs)."""
        solutions: list[Tensor] = []
        log_probs: list[float] = []
        for _ in range(self.n_samples):
            sol, lp = self._generate_one(prompt_ids, self.temperature)
            solutions.append(sol)
            log_probs.append(lp)
        return solutions, log_probs

    def diverse_sample(self, prompt_ids: Tensor) -> tuple[list[Tensor], list[float]]:
        """Sample n_samples sequences cycling through diverse temperatures."""
        temperatures = [0.5, 0.8, 1.0, 1.2]
        solutions: list[Tensor] = []
        log_probs: list[float] = []
        for i in range(self.n_samples):
            temp = temperatures[i % len(temperatures)]
            sol, lp = self._generate_one(prompt_ids, temp)
            solutions.append(sol)
            log_probs.append(lp)
        return solutions, log_probs


# ---------------------------------------------------------------------------
# Rejection Filter
# ---------------------------------------------------------------------------


class RejectionFilter:
    """Filter candidate solutions by correctness using a SolutionVerifier."""

    def __init__(self, verifier: SolutionVerifier) -> None:
        self.verifier = verifier

    def filter(
        self,
        prompt_ids: Tensor,
        solutions: list[Tensor],
        ground_truth: Any,
    ) -> tuple[list[Tensor], int, float]:
        """Return (kept, n_kept, acceptance_rate)."""
        kept: list[Tensor] = []
        for sol in solutions:
            if self.verifier.verify(prompt_ids, sol, ground_truth):
                kept.append(sol)
        n_kept = len(kept)
        acceptance_rate = n_kept / len(solutions) if len(solutions) > 0 else 0.0
        return kept, n_kept, acceptance_rate

    def filter_batch(
        self,
        prompts: list[Tensor],
        solutions_batch: list[list[Tensor]],
        ground_truths: list[Any],
    ) -> list[list[Tensor]]:
        """Filter solutions for each prompt in a batch."""
        results: list[list[Tensor]] = []
        for prompt, solutions, gt in zip(prompts, solutions_batch, ground_truths):
            kept, _, _ = self.filter(prompt, solutions, gt)
            results.append(kept)
        return results

    def best_of_k(
        self,
        prompt_ids: Tensor,
        solutions: list[Tensor],
        log_probs: list[float],
    ) -> Tensor:
        """Return the solution with the highest log probability (no verifier)."""
        if len(solutions) == 0:
            raise ValueError("solutions list is empty")
        best_idx = int(max(range(len(log_probs)), key=lambda i: log_probs[i]))
        return solutions[best_idx]


# ---------------------------------------------------------------------------
# RFT Dataset
# ---------------------------------------------------------------------------


class RFTDataset:
    """Dataset of filtered correct (prompt, solution) pairs."""

    def __init__(self) -> None:
        self._prompts: list[Tensor] = []
        self._solutions: list[Tensor] = []

    def add(self, prompt_ids: Tensor, solution_ids: Tensor) -> None:
        """Add a verified correct (prompt, solution) pair."""
        self._prompts.append(prompt_ids.detach().cpu())
        self._solutions.append(solution_ids.detach().cpu())

    def __len__(self) -> int:
        return len(self._prompts)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._prompts[idx], self._solutions[idx]

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Pad sequences to the same length within the batch."""
        prompts, solutions = zip(*batch)

        max_p = max(p.numel() for p in prompts)
        max_s = max(s.numel() for s in solutions)

        padded_prompts = torch.zeros(len(prompts), max_p, dtype=torch.long)
        padded_solutions = torch.zeros(len(solutions), max_s, dtype=torch.long)

        for i, (p, s) in enumerate(zip(prompts, solutions)):
            padded_prompts[i, : p.numel()] = p
            padded_solutions[i, : s.numel()] = s

        return padded_prompts, padded_solutions

    def stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        n_examples = len(self._solutions)
        mean_solution_length = (
            float(sum(s.numel() for s in self._solutions)) / n_examples if n_examples > 0 else 0.0
        )
        unique_prompts = len({tuple(p.tolist()) for p in self._prompts})
        return {
            "n_examples": n_examples,
            "mean_solution_length": mean_solution_length,
            "unique_prompts": unique_prompts,
        }


# ---------------------------------------------------------------------------
# RFT Trainer
# ---------------------------------------------------------------------------


class RFTTrainer:
    """Iterative self-improvement trainer via rejection fine-tuning."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        sampler: SolutionSampler,
        verifier: SolutionVerifier,
        n_iterations: int = 3,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.sampler = sampler
        self.verifier = verifier
        self.n_iterations = n_iterations
        self._filter = RejectionFilter(verifier)

    def iteration(
        self,
        prompts: list[Tensor],
        ground_truths: list[Any],
    ) -> dict[str, Any]:
        """Run one iteration of sample -> filter -> train.

        Returns dict with keys: n_correct, acceptance_rate, loss.
        """
        dataset = RFTDataset()

        total_solutions = 0
        total_correct = 0

        # (1) Sample solutions for each prompt
        all_solutions: list[list[Tensor]] = []
        all_log_probs: list[list[float]] = []
        for prompt in prompts:
            sols, lps = self.sampler.sample(prompt)
            all_solutions.append(sols)
            all_log_probs.append(lps)
            total_solutions += len(sols)

        # (2) Filter via verifier
        for prompt, solutions, gt in zip(prompts, all_solutions, ground_truths):
            kept, n_kept, _ = self._filter.filter(prompt, solutions, gt)
            total_correct += n_kept
            for sol in kept:
                dataset.add(prompt, sol)

        acceptance_rate = total_correct / total_solutions if total_solutions > 0 else 0.0

        # (3) Train on correct solutions with CE loss (SFT on filtered data)
        loss_val = float("nan")
        if len(dataset) > 0:
            self.model.train()
            self.optimizer.zero_grad()

            total_loss = torch.tensor(0.0)
            n_tokens = 0

            for i in range(len(dataset)):
                prompt_ids, solution_ids = dataset[i]
                # Construct input: prompt + solution (minus last token)
                # Target: solution tokens
                input_ids = torch.cat([prompt_ids, solution_ids[:-1]], dim=0).unsqueeze(0)
                solution_ids.unsqueeze(0)

                out = self.model(input_ids)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                # logits: (1, seq_len, vocab)
                # We only need the logits corresponding to the solution part
                sol_len = (
                    solution_ids.numel() - 1 if solution_ids.numel() > 1 else solution_ids.numel()
                )
                # logits shape: (1, prompt_len + sol_len - 1, vocab) or (1, seq, vocab)
                if logits.dim() == 3:
                    # Take last sol_len positions (corresponding to predicting each solution token)
                    sol_logits = logits[0, -sol_len:, :]  # (sol_len, vocab)
                    tgt = solution_ids[-sol_len:]  # (sol_len,)
                else:
                    sol_logits = logits[0]
                    tgt = solution_ids[-1:]

                ce = F.cross_entropy(sol_logits, tgt)
                total_loss = total_loss + ce
                n_tokens += tgt.numel()

            avg_loss = total_loss / len(dataset)
            avg_loss.backward()
            self.optimizer.step()
            loss_val = float(avg_loss.item())

        return {
            "n_correct": total_correct,
            "acceptance_rate": acceptance_rate,
            "loss": loss_val,
        }

    def self_improvement_loop(
        self,
        prompts: list[Tensor],
        ground_truths: list[Any],
    ) -> list[dict[str, Any]]:
        """Run n_iterations of self-improvement. Returns per-iteration stats."""
        all_stats: list[dict[str, Any]] = []
        for _ in range(self.n_iterations):
            stats = self.iteration(prompts, ground_truths)
            all_stats.append(stats)
        return all_stats
