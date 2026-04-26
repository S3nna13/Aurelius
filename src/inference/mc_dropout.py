"""MC Dropout inference-time uncertainty for AureliusTransformer.

Provides configurable MC Dropout with multiple aggregation strategies,
DropoutWrapper patching, and structured inference with epistemic/aleatoric
uncertainty decomposition distinct from the simpler uncertainty.py module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MCDropoutConfig:
    """Configuration for MC Dropout inference."""

    n_forward_passes: int = 10
    dropout_rate: float = 0.1
    temperature: float = 1.0
    aggregate: str = "mean"  # "mean" | "majority_vote" | "entropy_weighted"


class DropoutWrapper(nn.Module):
    """Wraps a model and ensures dropout is active during inference.

    If the model already contains nn.Dropout layers, their p is set to
    dropout_rate and they remain active (training=True) during forward.
    If no dropout layers exist, forward hooks inject dropout after each
    top-level child module.
    """

    def __init__(self, model: nn.Module, dropout_rate: float) -> None:
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self._hooks: list = []

        # Patch existing nn.Dropout layers
        self._dropout_modules: list[nn.Module] = []
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.p = dropout_rate
                self._dropout_modules.append(module)

        # If no dropout layers, attach forward hooks to inject dropout
        if not self._dropout_modules:
            self._attach_output_hooks()

    def _attach_output_hooks(self) -> None:
        """Attach forward hooks to inject dropout after each top-level child."""
        p = self.dropout_rate

        def make_hook(rate: float):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    dropped = F.dropout(output[0], p=rate, training=True)
                    return (dropped,) + output[1:]
                elif isinstance(output, torch.Tensor):
                    return F.dropout(output, p=rate, training=True)
                return output

            return hook

        for child in self.model.children():
            h = child.register_forward_hook(make_hook(p))
            self._hooks.append(h)

    def enable_mc_dropout(self) -> None:
        """Set model to train mode but keep BatchNorm / LayerNorm in eval mode."""
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                module.eval()

    def forward(self, input_ids: torch.Tensor) -> tuple:
        """Forward in train mode to keep dropout stochastic."""
        self.enable_mc_dropout()
        return self.model(input_ids)

    def __del__(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:  # noqa: S110
                pass


def run_mc_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_passes: int,
    dropout_rate: float,
) -> torch.Tensor:
    """Run model n_passes times with dropout active.

    Uses model.train() for stochasticity, wrapped in torch.no_grad() for
    memory efficiency.

    Args:
        model: AureliusTransformer or compatible
        input_ids: (B, T) token ids
        n_passes: number of stochastic forward passes
        dropout_rate: dropout probability applied via DropoutWrapper

    Returns:
        logits_stack: (n_passes, B, T, V)
    """
    wrapper = DropoutWrapper(model, dropout_rate)
    all_logits: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_passes):
            wrapper.enable_mc_dropout()
            _, logits, _ = wrapper.model(input_ids)  # (B, T, V)
            all_logits.append(logits)

    return torch.stack(all_logits, dim=0)  # (n_passes, B, T, V)


def compute_predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Entropy of the mean predictive distribution.

    Args:
        probs: (n_passes, B, T, V)

    Returns:
        (B, T) predictive entropy >= 0
    """
    mean_p = probs.mean(dim=0)  # (B, T, V)
    entropy = -(mean_p * torch.log(mean_p + 1e-9)).sum(dim=-1)  # (B, T)
    return entropy


def compute_mutual_information(probs: torch.Tensor) -> torch.Tensor:
    """Mutual information (epistemic uncertainty).

    MI = H(mean_p) - mean(H(p_i))

    Args:
        probs: (n_passes, B, T, V)

    Returns:
        (B, T) mutual information >= 0
    """
    mean_p = probs.mean(dim=0)  # (B, T, V)
    H_mean = -(mean_p * torch.log(mean_p + 1e-9)).sum(dim=-1)  # (B, T)

    per_pass_H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # (n_passes, B, T)
    mean_H = per_pass_H.mean(dim=0)  # (B, T)

    return (H_mean - mean_H).clamp(min=0.0)


def aggregate_predictions(
    logits_stack: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """Aggregate n forward-pass logits into a single (B, T, V) output.

    Args:
        logits_stack: (n_passes, B, T, V)
        method: "mean" | "majority_vote" | "entropy_weighted"

    Returns:
        (B, T, V) aggregated tensor
    """
    n_passes, B, T, V = logits_stack.shape
    probs = F.softmax(logits_stack, dim=-1)  # (n_passes, B, T, V)

    if method == "mean":
        mean_probs = probs.mean(dim=0)  # (B, T, V)
        return torch.log(mean_probs + 1e-9)

    elif method == "majority_vote":
        argmaxes = logits_stack.argmax(dim=-1)  # (n_passes, B, T)
        result = torch.zeros(B, T, V, device=logits_stack.device)
        for b in range(B):
            for t in range(T):
                counts = torch.bincount(argmaxes[:, b, t], minlength=V).float()
                winner = counts.argmax()
                result[b, t, winner] = 1.0
        return result

    elif method == "entropy_weighted":
        # Weight each pass inversely by its mean entropy
        per_pass_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # (n_passes, B, T)
        weights = 1.0 / (per_pass_entropy + 1e-9)  # (n_passes, B, T)
        weights = weights / weights.sum(dim=0, keepdim=True)  # normalize
        weighted_probs = (probs * weights.unsqueeze(-1)).sum(dim=0)  # (B, T, V)
        return torch.log(weighted_probs + 1e-9)

    else:
        raise ValueError(f"Unknown aggregate method: {method!r}")


class MCDropoutInference:
    """Structured MC Dropout inference with uncertainty decomposition."""

    def __init__(self, model: nn.Module, config: MCDropoutConfig) -> None:
        self.model = model
        self.config = config

    def predict(self, input_ids: torch.Tensor) -> dict:
        """Run n_forward_passes and return predictions + uncertainty metrics.

        Args:
            input_ids: (B, T)

        Returns:
            dict with keys:
                logits               (B, T, V)
                predictive_entropy   (B, T)
                mutual_information   (B, T)
                epistemic_uncertainty float
                aleatoric_uncertainty float
        """
        logits_stack = run_mc_forward(
            self.model,
            input_ids,
            n_passes=self.config.n_forward_passes,
            dropout_rate=self.config.dropout_rate,
        )  # (n_passes, B, T, V)

        if self.config.temperature != 1.0:
            logits_stack = logits_stack / self.config.temperature

        probs = F.softmax(logits_stack, dim=-1)  # (n_passes, B, T, V)

        pred_entropy = compute_predictive_entropy(probs)  # (B, T)
        mi = compute_mutual_information(probs)  # (B, T)

        epistemic = mi.mean().item()
        aleatoric = (pred_entropy - mi).clamp(min=0.0).mean().item()

        aggregated_logits = aggregate_predictions(logits_stack, method=self.config.aggregate)

        return {
            "logits": aggregated_logits,
            "predictive_entropy": pred_entropy,
            "mutual_information": mi,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
        }

    def get_uncertain_positions(
        self,
        input_ids: torch.Tensor,
        threshold: float = 0.5,
    ) -> list[tuple[int, int]]:
        """Return (batch, position) pairs where predictive entropy > threshold.

        Args:
            input_ids: (B, T)
            threshold: entropy threshold

        Returns:
            list of (batch_idx, token_position) tuples
        """
        result = self.predict(input_ids)
        entropy = result["predictive_entropy"]  # (B, T)
        positions = (entropy > threshold).nonzero(as_tuple=False)
        return [(int(r[0]), int(r[1])) for r in positions]
