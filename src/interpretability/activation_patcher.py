from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class PatchSpec:
    layer_name: str
    token_idx: int
    source_value: torch.Tensor | None = None


@dataclass
class PatchResult:
    original_logits: torch.Tensor
    patched_logits: torch.Tensor
    patch_spec: PatchSpec
    effect_score: float


class ActivationPatcher:
    """Causal intervention via forward hook activation patching."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def _make_hook(self, patch: PatchSpec) -> Callable:
        fill = patch.source_value

        def hook(
            module: nn.Module,
            input: tuple,
            output: torch.Tensor,
        ) -> torch.Tensor:
            if not isinstance(output, torch.Tensor):
                return output
            patched = output.clone()
            value = torch.zeros_like(output[:, patch.token_idx, :]) if fill is None else fill.to(output.device)
            patched[:, patch.token_idx, :] = value
            return patched

        return hook

    def _get_named_module(self, layer_name: str) -> nn.Module:
        for name, module in self._model.named_modules():
            if name == layer_name:
                return module
        raise KeyError(f"Layer '{layer_name}' not found in model")

    def patch(
        self,
        input_ids: torch.Tensor,
        patch: PatchSpec,
        target_token_idx: int = -1,
    ) -> PatchResult:
        with torch.no_grad():
            original_out = self._model(input_ids)

        module = self._get_named_module(patch.layer_name)
        hook_fn = self._make_hook(patch)
        handle = module.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                patched_out = self._model(input_ids)
        finally:
            handle.remove()

        def _extract(out: torch.Tensor) -> torch.Tensor:
            if out.dim() == 3:
                return out[0, target_token_idx, :]
            if out.dim() == 2:
                return out[0, :]
            return out.squeeze()

        orig_logits = _extract(original_out)
        ptch_logits = _extract(patched_out)
        effect = float(torch.linalg.norm(ptch_logits - orig_logits).item())

        return PatchResult(
            original_logits=orig_logits,
            patched_logits=ptch_logits,
            patch_spec=patch,
            effect_score=effect,
        )

    def patch_batch(
        self,
        input_ids: torch.Tensor,
        patches: list[PatchSpec],
    ) -> list[PatchResult]:
        return [self.patch(input_ids, p) for p in patches]

    def zero_ablate(
        self,
        input_ids: torch.Tensor,
        layer_name: str,
        token_idx: int,
    ) -> PatchResult:
        spec = PatchSpec(layer_name=layer_name, token_idx=token_idx, source_value=None)
        return self.patch(input_ids, spec)
