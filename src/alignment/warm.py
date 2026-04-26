"""WARM: Weight Averaged Reward Models (Ramé et al., arXiv:2401.12187).

Training multiple reward models from the same pre-trained base with different
fine-tuning runs, then averaging their weights leads to more robust reward
models that are less susceptible to reward hacking.

Key observations from the paper:
1. Weight averaging in weight space (not prediction space) benefits from loss
   barrier flatness — models fine-tuned from the same base are connected by
   low-loss paths in weight space.
2. Averaging k reward models reduces variance and improves rank correlation
   with human preferences.
3. The averaged model often outperforms any individual model.

Pure PyTorch implementation — no HuggingFace, einops, flash_attn, etc.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

# ---------------------------------------------------------------------------
# WARMRewardModel
# ---------------------------------------------------------------------------


class WARMRewardModel(nn.Module):
    """Thin reward model wrapper.

    Wraps an arbitrary backbone that produces hidden states and attaches a
    scalar reward head.  Supports both 2-D (B, d_model) and 3-D
    (B, T, d_model) inputs; for 3-D inputs the last token's representation
    is used (GPT-style).

    Args:
        backbone: nn.Module whose ``forward`` output is used as hidden states.
                  For 2-D input (B, d_model) the backbone is called as-is.
                  For 3-D input (B, T, d_model) the backbone is still called
                  with the full tensor and the last-token slice is taken.
        d_model:  Hidden dimension fed into the reward head.
    """

    def __init__(self, backbone: nn.Module, d_model: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.reward_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Compute scalar reward for each example in the batch.

        Args:
            x: (B, d_model) feature tensor  or  (B, T, d_model) sequence
               tensor.  When 3-D, the last token along the T dimension is
               used as the input to the reward head.

        Returns:
            (B,) scalar reward tensor.
        """
        h = self.backbone(x)  # may be (B, d_model) or (B, T, d_model)

        # Handle both 2-D and 3-D outputs from the backbone.
        if h.dim() == 3:
            h = h[:, -1, :]  # last token → (B, d_model)
        elif h.dim() == 2:
            pass  # already (B, d_model)
        else:
            raise ValueError(f"backbone output must be 2-D or 3-D, got {h.dim()}-D")

        reward = self.reward_head(h)  # (B, 1)
        return reward.squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# WARMEnsemble
# ---------------------------------------------------------------------------


class WARMEnsemble:
    """Stores multiple reward model state dicts and averages their weights.

    Implements weight-space averaging (the core of WARM) as well as
    prediction-space ensembling for comparison.

    Args:
        models: List of nn.Modules, all sharing the same architecture.
                State dicts are cloned at construction time so later
                mutations do not affect stored weights.
    """

    def __init__(self, models: list[nn.Module]) -> None:
        if not models:
            raise ValueError("models must be a non-empty list")
        self._state_dicts: list[dict[str, Tensor]] = [
            {k: v.clone() for k, v in m.state_dict().items()} for m in models
        ]
        self._models: list[nn.Module] = models

    # ------------------------------------------------------------------
    # Weight-space averaging
    # ------------------------------------------------------------------

    def average_weights(self) -> dict[str, Tensor]:
        """Return a state dict that is the uniform average of all models.

        Returns:
            Dict[str, Tensor]: Averaged state dict with the same keys as
            the individual models.
        """
        k = len(self._state_dicts)
        averaged: dict[str, Tensor] = {}
        for key in self._state_dicts[0]:
            acc = torch.zeros_like(self._state_dicts[0][key], dtype=torch.float32)
            for sd in self._state_dicts:
                acc += sd[key].float()
            averaged[key] = (acc / k).to(self._state_dicts[0][key].dtype)
        return averaged

    def get_averaged_model(self, base_model: nn.Module) -> nn.Module:
        """Load averaged weights into a deep copy of base_model and return it.

        Args:
            base_model: An nn.Module with the same architecture as the stored
                        models.  A deep copy is made — the original is not
                        mutated.

        Returns:
            nn.Module with weights set to the WARM average.
        """
        averaged_sd = self.average_weights()
        model_copy = copy.deepcopy(base_model)
        model_copy.load_state_dict(averaged_sd)
        return model_copy

    # ------------------------------------------------------------------
    # Prediction-space ensembling (for comparison)
    # ------------------------------------------------------------------

    def predict(self, x: Tensor) -> Tensor:
        """Average the predictions of all stored models (prediction-space ensemble).

        Each model is run in ``torch.no_grad()`` mode and its output is
        squeezed to (B,) before averaging.

        Args:
            x: Input tensor passed to each model.

        Returns:
            (B,) averaged prediction tensor.
        """
        all_preds: list[Tensor] = []
        for model in self._models:
            with torch.no_grad():
                out = model(x)
            out = out.squeeze(-1)  # ensure (B,)
            all_preds.append(out)
        stacked = torch.stack(all_preds, dim=0)  # (n_models, B)
        return stacked.mean(dim=0)  # (B,)


# ---------------------------------------------------------------------------
# WARMInterpolation
# ---------------------------------------------------------------------------


class WARMInterpolation:
    """Interpolates between two reward models in weight space.

    Supports linear interpolation (lerp) and sweep across a range of alpha
    values.  This is useful for studying the loss landscape between two
    independently fine-tuned reward models.

    Args:
        model_a: First reward model (alpha = 0.0 endpoint).
        model_b: Second reward model (alpha = 1.0 endpoint).
    """

    def __init__(self, model_a: nn.Module, model_b: nn.Module) -> None:
        self._sd_a: dict[str, Tensor] = {k: v.clone() for k, v in model_a.state_dict().items()}
        self._sd_b: dict[str, Tensor] = {k: v.clone() for k, v in model_b.state_dict().items()}

    def interpolate(self, alpha: float) -> dict[str, Tensor]:
        """Return (1 - alpha) * W_a + alpha * W_b state dict.

        Args:
            alpha: Interpolation coefficient in [0, 1].
                   ``alpha=0.0`` returns model_a weights exactly.
                   ``alpha=1.0`` returns model_b weights exactly.

        Returns:
            Interpolated state dict.
        """
        result: dict[str, Tensor] = {}
        for key in self._sd_a:
            wa = self._sd_a[key].float()
            wb = self._sd_b[key].float()
            blended = (1.0 - alpha) * wa + alpha * wb
            result[key] = blended.to(self._sd_a[key].dtype)
        return result

    def sweep(self, alphas: list[float], base_model: nn.Module) -> list[nn.Module]:
        """Return a list of interpolated models, one per alpha value.

        Args:
            alphas:     List of alpha values to evaluate.
            base_model: nn.Module used as the architecture template.  A
                        fresh deep copy is made for each alpha — the original
                        is never mutated.

        Returns:
            List of nn.Modules, one per alpha, with weights set to the
            corresponding interpolated state dict.
        """
        models: list[nn.Module] = []
        for alpha in alphas:
            sd = self.interpolate(alpha)
            m = copy.deepcopy(base_model)
            m.load_state_dict(sd)
            models.append(m)
        return models


# ---------------------------------------------------------------------------
# WARMTrainer
# ---------------------------------------------------------------------------


class WARMTrainer:
    """Coordinates training multiple reward models for eventual WARM averaging.

    Each model is trained independently (different random seeds / orderings)
    from the same pre-trained base, after which their weights are averaged via
    :class:`WARMEnsemble`.

    The Bradley–Terry loss is used::

        loss = -log σ(r_w - r_l)

    where r_w and r_l are the scalar rewards for the winning and losing
    responses respectively.

    Args:
        models:     List of :class:`WARMRewardModel` instances.
        optimizers: One optimizer per model (must have the same length).
    """

    def __init__(
        self,
        models: list[WARMRewardModel],
        optimizers: list[Optimizer],
    ) -> None:
        if len(models) != len(optimizers):
            raise ValueError(
                f"len(models)={len(models)} must equal len(optimizers)={len(optimizers)}"
            )
        if not models:
            raise ValueError("models must be a non-empty list")
        self.models = models
        self.optimizers = optimizers

    def train_step(
        self,
        x_w: Tensor,
        x_l: Tensor,
    ) -> list[Tensor]:
        """Perform one gradient step on all models using the BT preference loss.

        Args:
            x_w: (B, d_model) feature tensor for winning (preferred) responses.
            x_l: (B, d_model) feature tensor for losing (rejected) responses.

        Returns:
            List of per-model loss tensors (scalar, detached).
        """
        losses: list[Tensor] = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

            r_w = model(x_w)  # (B,)
            r_l = model(x_l)  # (B,)

            # Bradley–Terry loss: -E[log σ(r_w - r_l)]
            loss = -F.logsigmoid(r_w - r_l).mean()

            loss.backward()
            optimizer.step()

            losses.append(loss.detach())

        return losses

    def get_ensemble(self) -> WARMEnsemble:
        """Return a :class:`WARMEnsemble` over the current model weights.

        Returns:
            WARMEnsemble constructed from the current state of all models.
        """
        return WARMEnsemble(self.models)
