"""Knowledge transfer utilities for copying or blending weights between models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TransferConfig:
    layers_to_transfer: list[int] | None = None
    transfer_mode: str = "copy"
    interpolation_alpha: float = 0.5


@dataclass
class TransferResult:
    n_params_transferred: int
    n_layers_transferred: int
    mode: str


class KnowledgeTransfer:
    """Transfer weights between models of compatible architecture."""

    def __init__(self, config: TransferConfig | None = None) -> None:
        self.config = config or TransferConfig()

    def compatible_keys(self, source: torch.nn.Module, target: torch.nn.Module) -> list[str]:
        src_sd = source.state_dict()
        tgt_sd = target.state_dict()
        return [k for k in src_sd if k in tgt_sd and src_sd[k].shape == tgt_sd[k].shape]

    def _layer_index(self, key: str) -> int | None:
        """Extract the first integer segment from a parameter key, or None."""
        for part in key.split("."):
            if part.isdigit():
                return int(part)
        return None

    def transfer(self, source: torch.nn.Module, target: torch.nn.Module) -> TransferResult:
        mode = self.config.transfer_mode
        alpha = self.config.interpolation_alpha
        layers_filter = self.config.layers_to_transfer

        src_sd = source.state_dict()
        tgt_sd = target.state_dict()

        n_params = 0
        layers_seen: set[int] = set()

        for k in list(src_sd.keys()):
            if k not in tgt_sd:
                continue

            layer_idx = self._layer_index(k)
            if layers_filter is not None and layer_idx not in layers_filter:
                continue

            src_t = src_sd[k]
            tgt_t = tgt_sd[k]

            if mode == "copy":
                if src_t.shape == tgt_t.shape:
                    tgt_sd[k] = src_t.clone()
                    n_params += src_t.numel()
                    if layer_idx is not None:
                        layers_seen.add(layer_idx)

            elif mode == "interpolate":
                if src_t.shape == tgt_t.shape:
                    tgt_sd[k] = alpha * src_t.to(tgt_t.dtype) + (1.0 - alpha) * tgt_t
                    n_params += src_t.numel()
                    if layer_idx is not None:
                        layers_seen.add(layer_idx)

            elif mode == "expand":
                if src_t.shape == tgt_t.shape:
                    tgt_sd[k] = src_t.clone()
                    n_params += src_t.numel()
                    if layer_idx is not None:
                        layers_seen.add(layer_idx)
                elif src_t.numel() <= tgt_t.numel() and src_t.ndim == tgt_t.ndim:
                    slices = tuple(slice(0, s) for s in src_t.shape)
                    tgt_sd[k][slices] = src_t.to(tgt_t.dtype)
                    n_params += src_t.numel()
                    if layer_idx is not None:
                        layers_seen.add(layer_idx)

        target.load_state_dict(tgt_sd)

        n_layers = len(layers_seen) if layers_seen else (1 if n_params > 0 else 0)
        return TransferResult(
            n_params_transferred=n_params,
            n_layers_transferred=n_layers,
            mode=mode,
        )

    def partial_transfer(
        self,
        source: torch.nn.Module,
        target: torch.nn.Module,
        key_pattern: str,
    ) -> TransferResult:
        mode = self.config.transfer_mode
        alpha = self.config.interpolation_alpha

        src_sd = source.state_dict()
        tgt_sd = target.state_dict()

        n_params = 0
        layers_seen: set[int] = set()

        for k in list(src_sd.keys()):
            if key_pattern not in k:
                continue
            if k not in tgt_sd:
                continue

            src_t = src_sd[k]
            tgt_t = tgt_sd[k]

            if src_t.shape != tgt_t.shape:
                continue

            if mode == "interpolate":
                tgt_sd[k] = alpha * src_t.to(tgt_t.dtype) + (1.0 - alpha) * tgt_t
            else:
                tgt_sd[k] = src_t.clone()

            n_params += src_t.numel()
            layer_idx = self._layer_index(k)
            if layer_idx is not None:
                layers_seen.add(layer_idx)

        target.load_state_dict(tgt_sd)

        n_layers = len(layers_seen) if layers_seen else (1 if n_params > 0 else 0)
        return TransferResult(
            n_params_transferred=n_params,
            n_layers_transferred=n_layers,
            mode=mode,
        )
