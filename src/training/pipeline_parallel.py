"""Pipeline parallelism: split model layers across stages for very large model training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism."""

    n_stages: int = 2
    n_micro_batches: int = 4
    interleaved: bool = False


class PipelineStage(nn.Module):
    """Wraps a sequence of transformer layers assigned to one pipeline stage.

    Because AureliusTransformer layers require (x, freqs_cis, mask, past_kv)
    this stage stores freqs_cis as a buffer so it can run layers sequentially.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        n_stages: int,
        freqs_cis: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.n_stages = n_stages
        self.is_first: bool = stage_id == 0
        self.is_last: bool = stage_id == n_stages - 1

        if freqs_cis is not None:
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        else:
            self.freqs_cis = None  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        """Run all layers in this stage sequentially.

        Handles both plain nn.Module layers (just x) and AureliusTransformerBlock
        layers that expect (x, freqs_cis, mask, past_kv).
        """
        for layer in self.layers:
            # Check if the layer is a TransformerBlock (has freqs_cis argument)
            import inspect
            sig = inspect.signature(layer.forward)
            params = list(sig.parameters.keys())
            if "freqs_cis" in params:
                # AureliusTransformer block — needs freqs_cis and mask
                T = x.shape[1]
                freqs = self.freqs_cis[:T] if self.freqs_cis is not None else None
                x, _ = layer(x, freqs, None, None)
            else:
                x = layer(x)
        return x


def partition_layers(layers: nn.ModuleList, n_stages: int) -> list[nn.ModuleList]:
    """Evenly split layers across n_stages. Last stage gets any remainder.

    Args:
        layers: Full list of model layers.
        n_stages: Number of pipeline stages.

    Returns:
        List of n_stages nn.ModuleLists.
    """
    n = len(layers)
    base = n // n_stages
    partitions: list[nn.ModuleList] = []
    start = 0
    for i in range(n_stages):
        if i < n_stages - 1:
            end = start + base
        else:
            end = n  # last stage takes remainder
        partitions.append(nn.ModuleList(list(layers)[start:end]))
        start = end
    return partitions


class PipelinedModel(nn.Module):
    """Assembles pipeline stages into a sequential forward pass (single-device simulation).

    Layout: embed -> stage_0 -> stage_1 -> ... -> stage_(n-1) -> head
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        embed: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.stages = nn.ModuleList(stages)
        self.head = head

    def forward(self, input_ids: Tensor) -> Tensor:
        """embed -> stage_0 -> ... -> stage_N-1 -> head.

        Returns:
            logits: (B, T, V)
        """
        x = self.embed(input_ids)
        for stage in self.stages:
            x = stage(x)
        logits = self.head(x)
        return logits

    def stage_params(self, stage_id: int) -> list[nn.Parameter]:
        """Return parameters for a single pipeline stage.

        Args:
            stage_id: Index into self.stages.

        Returns:
            List of nn.Parameter objects belonging to that stage.
        """
        return list(self.stages[stage_id].parameters())


class MicroBatchScheduler:
    """Simulates a 1F1B (one-forward-one-backward) pipeline schedule.

    Schedule:
      - Warmup: n_stages forward passes
      - Steady state: (n_micro_batches - n_stages) 1F1B pairs
      - Drain: n_stages backward passes
    """

    def __init__(self, n_stages: int, n_micro_batches: int) -> None:
        self.n_stages = n_stages
        self.n_micro_batches = n_micro_batches

    def generate_schedule(self) -> list[tuple[str, int, int]]:
        """Generate 1F1B pipeline schedule.

        Returns:
            List of (action, stage_id, micro_batch_id) tuples where
            action is "forward" or "backward".
        """
        schedule: list[tuple[str, int, int]] = []

        n_warmup = min(self.n_stages, self.n_micro_batches)
        n_steady = self.n_micro_batches - n_warmup

        # --- Warmup: fill pipeline with forward passes ---
        for mb in range(n_warmup):
            for stage in range(self.n_stages):
                schedule.append(("forward", stage, mb))

        # --- Steady state: 1F1B ---
        for mb in range(n_steady):
            fwd_mb = n_warmup + mb
            bwd_mb = mb
            # One forward
            for stage in range(self.n_stages):
                schedule.append(("forward", stage, fwd_mb))
            # One backward (reverse stage order)
            for stage in range(self.n_stages - 1, -1, -1):
                schedule.append(("backward", stage, bwd_mb))

        # --- Drain: flush backward passes ---
        for mb in range(n_warmup):
            bwd_mb = n_steady + mb
            for stage in range(self.n_stages - 1, -1, -1):
                schedule.append(("backward", stage, bwd_mb))

        return schedule

    def pipeline_bubble_fraction(self) -> float:
        """Fraction of pipeline time spent in bubble.

        Returns:
            (n_stages - 1) / n_micro_batches
        """
        return (self.n_stages - 1) / self.n_micro_batches


def split_model_into_stages(model: nn.Module, n_stages: int) -> PipelinedModel:
    """Auto-split an AureliusTransformer into n_stages pipeline stages.

    Accesses model.layers for TransformerBlocks, model.embed (or
    model.token_embedding) for the embedding, and model.lm_head (or
    model.head) for the output projection.

    Args:
        model: AureliusTransformer instance.
        n_stages: Number of pipeline stages.

    Returns:
        PipelinedModel wrapping the split stages.
    """
    # Resolve layer list
    layers: nn.ModuleList = model.layers  # type: ignore[attr-defined]

    # Resolve embedding
    if hasattr(model, "embed"):
        embed = model.embed  # type: ignore[attr-defined]
    elif hasattr(model, "token_embedding"):
        embed = model.token_embedding  # type: ignore[attr-defined]
    else:
        raise AttributeError("Model has no 'embed' or 'token_embedding' attribute.")

    # Resolve output head
    if hasattr(model, "lm_head"):
        head = model.lm_head  # type: ignore[attr-defined]
    elif hasattr(model, "head"):
        head = model.head  # type: ignore[attr-defined]
    else:
        raise AttributeError("Model has no 'lm_head' or 'head' attribute.")

    # Retrieve freqs_cis buffer if available (AureliusTransformer registers it)
    freqs_cis: Tensor | None = getattr(model, "freqs_cis", None)

    # Partition layers across stages
    partitions = partition_layers(layers, n_stages)

    stages = [
        PipelineStage(
            layers=partition,
            stage_id=i,
            n_stages=n_stages,
            freqs_cis=freqs_cis,
        )
        for i, partition in enumerate(partitions)
    ]

    return PipelinedModel(stages=stages, embed=embed, head=head)


class GradientAccumulationPipeline:
    """Accumulates gradients across micro-batches before an optimizer step.

    Each train_step forward-passes and loss-computes all micro-batches,
    accumulates gradients, normalises by n_micro_batches, then steps.
    """

    def __init__(
        self,
        model: PipelinedModel,
        optimizer: torch.optim.Optimizer,
        n_micro_batches: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.n_micro_batches = n_micro_batches

    def train_step(
        self,
        micro_batches: list[tuple[Tensor, Tensor]],
    ) -> dict[str, float]:
        """Forward + loss for each micro-batch, accumulate grads, then step.

        Args:
            micro_batches: List of (input_ids, labels) tensors, one per micro-batch.

        Returns:
            {"loss": average_loss, "n_micro_batches": int}
        """
        self.optimizer.zero_grad()
        total_loss = 0.0

        for input_ids, labels in micro_batches:
            logits = self.model(input_ids)  # (B, T, V)

            # Shift for causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Normalise by n_micro_batches before accumulating
            (loss / self.n_micro_batches).backward()
            total_loss += loss.item()

        self.optimizer.step()
        avg_loss = total_loss / len(micro_batches)
        return {"loss": avg_loss, "n_micro_batches": len(micro_batches)}
