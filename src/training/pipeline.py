"""Autonomous training pipeline orchestrator.

Coordinates the full self-improvement loop:
  1. Data generation (Magpie or existing shard data)
  2. SFT on generated data
  3. Perplexity evaluation
  4. GRPO/DPO alignment (optional)
  5. Checkpoint best model
  6. Repeat

This is the "autonomous" core of the Aurelius project — the model can
continuously improve itself without human intervention beyond setting up
the initial data and reward function.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the autonomous training pipeline."""

    # Output
    output_dir: str = "runs/pipeline"

    # Phases to run
    run_sft: bool = True
    run_grpo: bool = False
    run_eval: bool = True

    # SFT settings
    sft_epochs: int = 1
    sft_lr: float = 2e-5
    sft_batch_size: int = 4
    sft_max_seq_len: int = 512

    # GRPO settings
    grpo_steps: int = 50
    grpo_num_rollouts: int = 4
    grpo_lr: float = 1e-6

    # Evaluation
    eval_n_sequences: int = 50
    eval_seq_len: int = 128

    # Checkpointing
    keep_last_n: int = 3

    # Iteration
    n_iterations: int = 3


@dataclass
class PipelineResult:
    """Result of one pipeline iteration."""
    iteration: int
    sft_loss: float | None = None
    eval_ppl: float | None = None
    grpo_mean_reward: float | None = None
    checkpoint_path: str | None = None


class TrainingPipeline:
    """Orchestrates the autonomous Aurelius training loop.

    Args:
        model: The model to train and improve.
        tokenizer: Tokenizer for encoding/decoding text.
        cfg: Pipeline configuration.
        reward_fn: Optional reward function for GRPO (prompt, response) -> float.
                   Required if cfg.run_grpo=True.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        cfg: PipelineConfig | None = None,
        reward_fn=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or PipelineConfig()
        self.reward_fn = reward_fn
        self._history: list[PipelineResult] = []

    def run_sft_phase(
        self,
        train_data: list[dict],
        iteration: int,
    ) -> float | None:
        """Run one SFT fine-tuning phase.

        Args:
            train_data: List of {"input_ids": Tensor, "labels": Tensor} dicts.
            iteration: Current pipeline iteration number.

        Returns:
            Final training loss, or None if no data.
        """
        if not train_data or not self.cfg.run_sft:
            return None

        from torch.utils.data import DataLoader
        from src.alignment.sft import _resolve_target_modules, _cosine_warmup
        from src.alignment.dora import apply_dora_to_model, DoRALinear

        # Apply DoRA to model for efficient fine-tuning, but only to nn.Linear layers
        # (skip any that have already been replaced with DoRALinear)
        target_modules = _resolve_target_modules(
            self.model, ("q_proj", "v_proj", "k_proj", "o_proj")
        )
        linear_targets = []
        for path in target_modules:
            parts = path.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            leaf = getattr(parent, parts[-1])
            if isinstance(leaf, nn.Linear) and not isinstance(leaf, DoRALinear):
                linear_targets.append(path)

        if linear_targets:
            apply_dora_to_model(self.model, linear_targets, rank=16, alpha=16.0)

        # Build DataLoader
        def collate(batch):
            ids = torch.stack([b["input_ids"] for b in batch])
            lbs = torch.stack([b["labels"] for b in batch])
            return {"input_ids": ids, "labels": lbs}

        loader = DataLoader(
            train_data,
            batch_size=self.cfg.sft_batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.sft_lr,
        )

        self.model.train()
        total_steps = len(loader) * self.cfg.sft_epochs
        step = 0
        last_loss = float("inf")

        for epoch in range(self.cfg.sft_epochs):
            for batch in loader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                loss, _, _ = self.model(input_ids=input_ids, labels=labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                _cosine_warmup(optimizer, step, total_steps, warmup_steps=max(1, total_steps // 10))
                optimizer.zero_grad()

                last_loss = loss.item()
                step += 1

        logger.info("SFT phase done. Final loss: %.4f", last_loss)
        return last_loss

    def run_eval_phase(self, eval_data: list[list[int]] | None = None) -> float | None:
        """Run perplexity evaluation.

        Args:
            eval_data: List of token sequences. If None, generates random data.

        Returns:
            Perplexity score, or None if eval is disabled.
        """
        if not self.cfg.run_eval:
            return None

        from src.eval.perplexity import compute_perplexity

        if eval_data is None:
            # Use random token sequences as placeholder eval data
            torch.manual_seed(42)
            eval_data = [
                torch.randint(1, self.model.config.vocab_size, (self.cfg.eval_seq_len,)).tolist()
                for _ in range(self.cfg.eval_n_sequences)
            ]

        result = compute_perplexity(self.model, eval_data)
        logger.info("Eval PPL: %.2f", result.perplexity)
        return result.perplexity

    def run_grpo_phase(self, prompts: list[torch.Tensor] | None = None) -> float | None:
        """Run GRPO alignment phase.

        Args:
            prompts: List of (1, seq_len) prompt tensors. Uses short random prompts if None.

        Returns:
            Mean reward from the last GRPO step.
        """
        if not self.cfg.run_grpo or self.reward_fn is None:
            return None

        from src.alignment.grpo import GRPOTrainer, GRPOConfig

        grpo_cfg = GRPOConfig(
            num_rollouts=self.cfg.grpo_num_rollouts,
            num_steps=self.cfg.grpo_steps,
            learning_rate=self.cfg.grpo_lr,
            max_new_tokens=64,
        )
        trainer = GRPOTrainer(self.model, self.reward_fn, grpo_cfg)

        if prompts is None:
            prompts = [torch.randint(1, self.model.config.vocab_size, (1, 8))]

        last_metrics = None
        for i, prompt in enumerate(prompts[:self.cfg.grpo_steps]):
            last_metrics = trainer.step(prompt, "", self.tokenizer)

        mean_reward = last_metrics["mean_reward"] if last_metrics else None
        logger.info("GRPO phase done. Mean reward: %s", mean_reward)
        return mean_reward

    def run(
        self,
        train_data: list[dict] | None = None,
        eval_data: list[list[int]] | None = None,
        grpo_prompts: list[torch.Tensor] | None = None,
    ) -> list[PipelineResult]:
        """Run the full pipeline for n_iterations.

        Args:
            train_data: SFT training data (list of tokenized dicts).
            eval_data: Evaluation sequences (list of token id lists).
            grpo_prompts: GRPO prompt tensors.

        Returns:
            List of PipelineResult per iteration.
        """
        from src.training.checkpoint import save_checkpoint

        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(1, self.cfg.n_iterations + 1):
            logger.info("=== Pipeline iteration %d / %d ===", iteration, self.cfg.n_iterations)
            result = PipelineResult(iteration=iteration)

            # SFT phase
            result.sft_loss = self.run_sft_phase(train_data or [], iteration)

            # Eval phase
            result.eval_ppl = self.run_eval_phase(eval_data)

            # GRPO phase
            result.grpo_mean_reward = self.run_grpo_phase(grpo_prompts)

            # Checkpoint
            ckpt_dir = save_checkpoint(
                self.model,
                optimizer=None,
                step=iteration,
                epoch=iteration,
                train_loss=result.sft_loss or 0.0,
                output_dir=output_dir,
                val_loss=result.eval_ppl,
                keep_last_n=self.cfg.keep_last_n,
            )
            result.checkpoint_path = str(ckpt_dir)

            self._history.append(result)
            logger.info(
                "Iteration %d: sft_loss=%s  ppl=%s  reward=%s",
                iteration,
                f"{result.sft_loss:.4f}" if result.sft_loss else "N/A",
                f"{result.eval_ppl:.2f}" if result.eval_ppl else "N/A",
                f"{result.grpo_mean_reward:.3f}" if result.grpo_mean_reward else "N/A",
            )

        return self._history

    @property
    def history(self) -> list[PipelineResult]:
        """Training history across all iterations."""
        return list(self._history)
