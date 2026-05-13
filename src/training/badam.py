"""BAdam — Block Coordinate Adam for memory-efficient full-parameter fine-tuning.

Updates one transformer layer block per optimizer step. Requires optimizer
state for only one block at a time → LoRA-level memory with full-param quality.
Paper: arXiv:2404.02827 (NeurIPS 2024)
"""

from __future__ import annotations

from torch import nn


class BAdamScheduler:
    """Cycles through transformer layer blocks, enabling one per step."""

    def __init__(self, model: nn.Module, cycle_length: int = 1):
        """
        Args:
            model: the transformer to train
            cycle_length: steps per block before advancing to next block
        """
        self.blocks = self._get_blocks(model)
        self.current = 0
        self.cycle_length = cycle_length
        self.step_count = 0
        self._set_active(self.current)

    def _get_blocks(self, model: nn.Module) -> list[list[nn.Parameter]]:
        """Group parameters by transformer layer."""
        # Try to detect standard layer naming
        blocks = []
        for name, module in model.named_children():
            if hasattr(module, "__len__"):  # ModuleList of layers
                for layer in module:
                    blocks.append(list(layer.parameters()))
            elif any(x in name for x in ["layer", "block", "transformer"]):
                blocks.append(list(module.parameters()))
        if not blocks:  # fallback: all params as one block
            blocks = [list(model.parameters())]
        return blocks

    def _set_active(self, idx: int):
        """Enable gradient only for the active block."""
        for i, block in enumerate(self.blocks):
            for p in block:
                p.requires_grad_(i == idx)

    def step(self):
        """Advance to next block after cycle_length steps."""
        self.step_count += 1
        if self.step_count % self.cycle_length == 0:
            self.current = (self.current + 1) % len(self.blocks)
            self._set_active(self.current)


__all__ = ["BAdamScheduler"]
