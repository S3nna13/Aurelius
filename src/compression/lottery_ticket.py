from __future__ import annotations

import torch
import torch.nn as nn


class LotteryTicketPruner:
    def __init__(self, prune_fraction: float = 0.2) -> None:
        if not 0.0 <= prune_fraction <= 1.0:
            raise ValueError(f"prune_fraction must be in [0,1], got {prune_fraction}")
        self.prune_fraction = prune_fraction
        self.mask: torch.Tensor | None = None
        self._initial_weights: dict[str, torch.Tensor] = {}

    def prune(self, module: nn.Module) -> None:
        if self.mask is None:
            self._create_initial_mask(module)
        if self.mask is None:
            return
        with torch.no_grad():
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    param.data.mul_(self.mask.to(param.device))

    def _create_initial_mask(self, module: nn.Module) -> None:
        if self.prune_fraction <= 0.0:
            with torch.no_grad():
                for name, param in module.named_parameters():
                    if "weight" in name and param.dim() >= 2:
                        self._initial_weights[name.replace(".", "_")] = param.data.clone()
            self.mask = None
            return
        if self.prune_fraction >= 1.0:
            self.mask = None
            with torch.no_grad():
                for name, param in module.named_parameters():
                    if "weight" in name and param.dim() >= 2:
                        param.data.zero_()
            return

        for name, param in module.named_parameters():
            if "weight" in name and param.dim() >= 2:
                self._initial_weights[name.replace(".", "_")] = param.data.clone()
        masks = []
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() >= 2:
                flat = param.data.abs().flatten()
                keep = max(1, int(flat.numel() * (1.0 - self.prune_fraction)))
                threshold = flat.kthvalue(keep).values if keep < flat.numel() else flat.min()
                m = param.data.abs() >= threshold
                masks.append(m)
        if not masks:
            self.mask = torch.tensor([])
            return
        self.mask = masks[0]

    def rewind(self, module: nn.Module) -> None:
        with torch.no_grad():
            for name, param in module.named_parameters():
                key = name.replace(".", "_")
                if key in self._initial_weights:
                    param.data.copy_(self._initial_weights[key])
