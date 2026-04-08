"""Model ensemble: combine predictions from multiple model instances."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleMode(Enum):
    PROB_MEAN = "prob_mean"  # average probabilities (arithmetic mean)
    LOGIT_MEAN = "logit_mean"  # average logits (then softmax)
    LOG_PROB_MEAN = "log_prob_mean"  # average log-probs (geometric mean of probs)


@dataclass
class EnsembleConfig:
    mode: EnsembleMode = EnsembleMode.PROB_MEAN
    weights: list[float] | None = None  # per-model weights (normalized internally)
    temperature: float = 1.0


class ModelEnsemble(nn.Module):
    """Combines predictions from multiple models.

    All models must have the same architecture and vocabulary.

    Usage:
        ensemble = ModelEnsemble([model_a, model_b, model_c], cfg)
        _, logits, _ = ensemble(input_ids)  # combined logits
    """

    def __init__(self, models: list[nn.Module], cfg: EnsembleConfig | None = None) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.cfg = cfg or EnsembleConfig()

        # Normalize weights
        n = len(models)
        if self.cfg.weights is not None:
            w = torch.tensor(self.cfg.weights, dtype=torch.float)
            self._weights = (w / w.sum()).tolist()
        else:
            self._weights = [1.0 / n] * n

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, None]:
        """Run all models and combine their predictions.

        Returns:
            (loss, combined_logits, None)
            - loss: mean CE loss across models if labels given, else None
            - combined_logits: (B, S, V) combined logits
        """
        all_logits: list[tuple[torch.Tensor, float]] = []
        all_losses: list[torch.Tensor] = []

        for model, weight in zip(self.models, self._weights):
            loss_val, logits, _ = model(input_ids, labels=labels)
            if labels is not None and loss_val is not None:
                all_losses.append(loss_val)
            all_logits.append((logits, weight))

        # Combine logits based on mode
        cfg = self.cfg
        if cfg.mode == EnsembleMode.LOGIT_MEAN:
            combined = sum(w * lg for lg, w in all_logits)
        elif cfg.mode == EnsembleMode.PROB_MEAN:
            probs_sum = sum(
                w * F.softmax(lg / cfg.temperature, dim=-1) for lg, w in all_logits
            )
            combined = probs_sum.log()  # convert back to logit-like for consistency
        elif cfg.mode == EnsembleMode.LOG_PROB_MEAN:
            log_probs_sum = sum(
                w * F.log_softmax(lg, dim=-1) for lg, w in all_logits
            )
            combined = log_probs_sum
        else:
            raise ValueError(f"Unknown ensemble mode: {cfg.mode}")

        loss = torch.stack(all_losses).mean() if all_losses else None
        return loss, combined, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Greedy/sampling generate using ensemble distribution."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            _, logits, _ = self.forward(generated)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)

            # Top-p sampling
            sorted_probs, sorted_idx = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            cutoff = (cumsum - sorted_probs) > top_p
            sorted_probs[cutoff] = 0.0
            sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

            next_token = torch.multinomial(sorted_probs, 1)
            next_token = sorted_idx.gather(-1, next_token)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


def ensemble_from_checkpoints(
    checkpoint_paths: list[str],
    model_cls: type,
    model_cfg: object,
    cfg: EnsembleConfig | None = None,
    device: str = "cpu",
) -> ModelEnsemble:
    """Load multiple checkpoints and create an ensemble.

    Args:
        checkpoint_paths: list of paths to checkpoint directories (with model.pt)
        model_cls: class to instantiate (e.g. AureliusTransformer)
        model_cfg: config to pass to model_cls
        cfg: EnsembleConfig
        device: target device

    Returns:
        ModelEnsemble with loaded models.
    """
    models: list[nn.Module] = []
    for path in checkpoint_paths:
        model = model_cls(model_cfg)
        ckpt = torch.load(
            os.path.join(path, "model.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        models.append(model)
    return ModelEnsemble(models, cfg)
