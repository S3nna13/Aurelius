from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.alignment.praxis.config import PRAXISConfig


class SteeringRewardCorrespondence:
    """Measures alignment between reward gradient and steering direction.

    Captures hidden states at steer_layers with and without additive steering,
    computes cosine distance between the two, and returns a negative scalar
    reward proportional to that distance (high correspondence = near zero).
    """

    def __init__(self, model: nn.Module, config: PRAXISConfig) -> None:
        self.model = model
        self.config = config

    def _capture_hidden(self, input_ids: Tensor, apply_steer: bool = False, **kwargs) -> dict[int, Tensor]:
        captures: dict[int, Tensor] = {}
        hooks = []

        for idx in self.config.steer_layers:
            if idx >= len(self.model.layers):
                continue

            def make_hook(layer_idx: int, steer: bool):
                def hook(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if steer:
                        direction = torch.randn_like(h)
                        direction = F.normalize(direction, dim=-1)
                        h = h + self.config.steer_alpha * direction
                    captures[layer_idx] = h.detach()
                return hook

            handle = self.model.layers[idx].register_forward_hook(make_hook(idx, apply_steer))
            hooks.append(handle)

        with torch.no_grad():
            self.model(input_ids)

        for handle in hooks:
            handle.remove()

        return captures

    def compute(self, input_ids: Tensor, **kwargs) -> Tensor:
        """Compute SRC reward signal.

        Args:
            input_ids: (B, T) long tensor of token IDs — the model's embed layer
                       is applied internally, and hooks capture layer outputs.

        Returns a non-positive scalar: 0 when steering has no effect (ideal),
        negative when hidden states diverge significantly from steering.
        """
        unsteered = self._capture_hidden(input_ids, apply_steer=False, **kwargs)
        steered   = self._capture_hidden(input_ids, apply_steer=True, **kwargs)

        distances: list[Tensor] = []
        for idx in self.config.steer_layers:
            if idx not in unsteered or idx not in steered:
                continue
            u = unsteered[idx]
            s = steered[idx]
            cos_sim  = F.cosine_similarity(u.reshape(-1, u.shape[-1]),
                                           s.reshape(-1, s.shape[-1]), dim=-1)
            distances.append(1.0 - cos_sim.mean())

        if not distances:
            return torch.tensor(0.0)

        mean_dist = torch.stack(distances).mean()
        return -self.config.lambda_src * mean_dist