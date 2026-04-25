"""Maximal Update Parametrization (muP) scaling rules (Yang et al. 2203.03466)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MuPConfig:
    base_width: int = 256
    base_lr: float = 1e-3
    base_init_std: float = 0.02


class MuPScaler:
    """Apply muP scaling rules to learning rates and initialisation std-devs."""

    def __init__(self, config: MuPConfig | None = None) -> None:
        self.config = config if config is not None else MuPConfig()

    # ------------------------------------------------------------------
    # Scaling rules
    # ------------------------------------------------------------------

    def scale_lr(self, base_lr: float, width: float, base_width: float) -> float:
        """muP learning-rate rule: lr * (base_width / width)."""
        return base_lr * (base_width / width)

    def scale_init_std(
        self,
        base_std: float,
        width: float,
        base_width: float,
        layer_type: str = "attention",
    ) -> float:
        """Return scaled initialisation std for *layer_type*.

        attention / hidden / default : std * sqrt(base_width / width)
        output                       : std * (base_width / width)
        embedding                    : std  (no scaling)
        """
        lt = layer_type.lower()
        if lt == "embedding":
            return base_std
        elif lt == "output":
            return base_std * (base_width / width)
        else:
            # attention and everything else
            return base_std * math.sqrt(base_width / width)

    def get_param_groups(
        self,
        named_params: list[tuple[str, object]],
        width: float,
    ) -> list[dict]:
        """Partition parameters into muP groups and assign scaled learning rates.

        Classification rules (checked in order):
          "embed"                  → embedding
          "attn" or "attention"    → attention
          "out_proj" or "output"   → output
          else                     → hidden
        """
        groups: dict[str, list] = {
            "embedding": [],
            "attention": [],
            "output": [],
            "hidden": [],
        }

        for name, param in named_params:
            n = name.lower()
            if "embed" in n:
                groups["embedding"].append(param)
            elif "attn" in n or "attention" in n:
                groups["attention"].append(param)
            elif "out_proj" in n or "output" in n:
                groups["output"].append(param)
            else:
                groups["hidden"].append(param)

        result = []
        base_lr = self.config.base_lr
        base_width = float(self.config.base_width)

        for group_name, params in groups.items():
            if not params:
                continue
            if group_name == "embedding":
                # No lr scaling for embeddings
                lr = base_lr
            else:
                lr = self.scale_lr(base_lr, width, base_width)
            result.append({"params": params, "lr": lr, "group_name": group_name})

        return result

    def width_multiplier(self, width: float) -> float:
        """Return width / config.base_width."""
        return width / self.config.base_width
