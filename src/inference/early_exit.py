"""Early exit inference: adaptive computation that stops at intermediate layers when confident."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EarlyExitConfig:
    """Configuration for early exit inference."""

    # Per-exit-point confidence thresholds
    exit_thresholds: list[float] = field(default_factory=lambda: [0.9, 0.85, 0.8])
    # Layer indices at which to check for exit
    exit_layers: list[int] = field(default_factory=lambda: [4, 8, 12])
    # Never exit before this layer
    min_exit_layer: int = 2
    # Confidence metric: "max_prob" | "entropy" | "margin"
    confidence_metric: str = "max_prob"
    # Whether to use learned highway classifiers at exit points
    use_highway: bool = True


def compute_confidence(logits: Tensor, metric: str) -> Tensor:
    """Compute per-sample confidence scores from last-token logits.

    Args:
        logits: (B, V) — last token logits.
        metric: "max_prob" | "entropy" | "margin".

    Returns:
        (B,) confidence scores in [0, 1].
    """
    if metric == "max_prob":
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values

    elif metric == "entropy":
        probs = F.softmax(logits, dim=-1)
        # Clamp to avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1)  # (B,)
        V = logits.shape[-1]
        max_entropy = math.log(V)
        # normalized_entropy in [0,1], where 0 = confident, 1 = uniform
        normalized_entropy = entropy / max_entropy
        # Invert so that 1 = confident, 0 = uniform
        return 1.0 - normalized_entropy

    elif metric == "margin":
        probs = F.softmax(logits, dim=-1)
        # Get top-2 probabilities
        top2, _ = probs.topk(2, dim=-1)  # (B, 2)
        margin = top2[:, 0] - top2[:, 1]  # (B,)
        return margin

    else:
        raise ValueError(
            f"Unknown confidence metric: {metric!r}. Choose from 'max_prob', 'entropy', 'margin'."
        )


class ExitClassifier(nn.Module):
    """Highway classifier at an intermediate layer.

    Projects from hidden state dimension to vocabulary size,
    producing logits that can be used for early exit decisions.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Args:
            hidden: (B, d_model) — last token hidden state.

        Returns:
            (B, vocab_size) logits.
        """
        return self.proj(hidden)


class EarlyExitWrapper(nn.Module):
    """Wraps a transformer model with early exit capabilities.

    Uses forward hooks on intermediate layers to capture hidden states,
    then applies exit classifiers to decide whether to exit early.
    """

    def __init__(
        self,
        model: nn.Module,
        config: EarlyExitConfig,
        d_model: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.d_model = d_model
        self.vocab_size = vocab_size

        # One exit classifier per exit point
        self.exit_classifiers = nn.ModuleList(
            [ExitClassifier(d_model, vocab_size) for _ in config.exit_layers]
        )

    def _run_layer_by_layer(self, input_ids: Tensor) -> tuple[dict[int, Tensor], Tensor]:
        """Run the backbone layer by layer, capturing hidden states at exit layers.

        Returns:
            hidden_at_exit: dict mapping layer_idx -> (B, d_model) hidden state (last token).
            final_logits: (B, seq_len, vocab_size) logits from the full model.
        """
        # Collect hidden states at exit layers via hooks
        exit_layer_set = set(self.config.exit_layers)
        captured: dict[int, Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # output is (hidden, kv) from TransformerBlock
                if isinstance(output, (tuple, list)):
                    hidden = output[0]
                else:
                    hidden = output
                # Capture last token: (B, d_model)
                captured[layer_idx] = hidden[:, -1, :].detach()

            return hook_fn

        for i, layer in enumerate(self.model.layers):
            if i in exit_layer_set:
                hooks.append(layer.register_forward_hook(make_hook(i)))

        try:
            with torch.no_grad():
                _loss, final_logits, _pkv = self.model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        return captured, final_logits

    def forward_with_exits(self, input_ids: Tensor) -> tuple[Tensor, int]:
        """Run backbone layer by layer and possibly exit early.

        At each configured exit layer, computes confidence from the highway classifier.
        If confidence exceeds the threshold (and the layer meets min_exit_layer),
        returns early with that exit's logits.

        Args:
            input_ids: (B, S) token ids.

        Returns:
            (logits, layer_idx_exited):
                logits: (B, vocab_size) last-token logits.
                layer_idx_exited: layer index where exit occurred, or n_layers-1 if no early exit.
        """
        n_layers = len(self.model.layers)
        exit_layers = self.config.exit_layers
        exit_thresholds = self.config.exit_thresholds
        metric = self.config.confidence_metric

        # Run model with hooks to capture hidden states
        captured, final_logits = self._run_layer_by_layer(input_ids)

        # Check each exit point in order
        for exit_idx, layer_idx in enumerate(exit_layers):
            # Skip if below minimum exit layer
            if layer_idx < self.config.min_exit_layer:
                continue

            # Skip if hidden state wasn't captured (shouldn't happen)
            if layer_idx not in captured:
                continue

            hidden = captured[layer_idx]  # (B, d_model)

            if self.config.use_highway:
                logits_at_exit = self.exit_classifiers[exit_idx](hidden)  # (B, vocab_size)
            else:
                # No highway: use the final lm_head directly on the captured hidden
                logits_at_exit = self.exit_classifiers[exit_idx](hidden)

            confidence = compute_confidence(logits_at_exit, metric)  # (B,)

            threshold = (
                exit_thresholds[exit_idx]
                if exit_idx < len(exit_thresholds)
                else exit_thresholds[-1]
            )

            # Exit if ALL samples in batch are confident enough
            if confidence.min().item() >= threshold:
                return logits_at_exit, layer_idx

        # No early exit: return final layer logits (last token)
        final_last_token = final_logits[:, -1, :]  # (B, vocab_size)
        return final_last_token, n_layers - 1

    def forward(self, input_ids: Tensor) -> Tensor:
        """Run forward with early exit, returning just the logits.

        Args:
            input_ids: (B, S) token ids.

        Returns:
            (B, vocab_size) last-token logits.
        """
        logits, _exit_layer = self.forward_with_exits(input_ids)
        return logits

    def compute_exit_stats(self, input_ids_batch: list[Tensor]) -> dict[str, float]:
        """Run each input and collect exit layer statistics.

        Args:
            input_ids_batch: list of (B, S) tensors (or (1, S) for single samples).

        Returns:
            dict with keys:
                "mean_exit_layer": average layer at which exit occurred.
                "early_exit_rate": fraction of inputs that exited early.
                "speedup_estimate": n_layers / mean_exit_layer (naive estimate).
        """
        n_layers = len(self.model.layers)
        exit_layers_collected = []

        for input_ids in input_ids_batch:
            _logits, exit_layer = self.forward_with_exits(input_ids)
            exit_layers_collected.append(exit_layer)

        mean_exit_layer = float(sum(exit_layers_collected)) / max(len(exit_layers_collected), 1)

        # "Early exit" means the model stopped before the last layer
        early_exit_count = sum(1 for line in exit_layers_collected if line < n_layers - 1)
        early_exit_rate = early_exit_count / max(len(exit_layers_collected), 1)

        # Naive speedup: ratio of full depth to average exit depth (+1 to avoid div-by-zero)
        speedup_estimate = n_layers / max(mean_exit_layer + 1, 1)

        return {
            "mean_exit_layer": mean_exit_layer,
            "early_exit_rate": early_exit_rate,
            "speedup_estimate": speedup_estimate,
        }


def train_exit_classifiers(
    model: nn.Module,
    wrapper: EarlyExitWrapper,
    data: Tensor,
    n_steps: int,
) -> dict[str, float]:
    """Train exit classifiers to predict the same token distribution as the full model.

    Uses KL divergence between exit classifier logits and final layer logits as supervision.

    Args:
        model: The backbone transformer model.
        wrapper: EarlyExitWrapper containing the exit classifiers to train.
        data: (N, S) input token ids dataset.
        n_steps: Number of gradient steps.

    Returns:
        {"mean_kl_loss": float}
    """
    optimizer = torch.optim.Adam(wrapper.exit_classifiers.parameters(), lr=1e-3)
    exit_layer_set = set(wrapper.config.exit_layers)
    total_kl = 0.0
    count = 0

    for step in range(n_steps):
        # Pick a random batch item
        idx = step % data.shape[0]
        input_ids = data[idx : idx + 1]  # (1, S)

        # Capture hidden states and final logits
        captured: dict[int, Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                if isinstance(output, (tuple, list)):
                    hidden = output[0]
                else:
                    hidden = output
                captured[layer_idx] = hidden[:, -1, :]  # (1, d_model), keep grad

            return hook_fn

        for i, layer in enumerate(model.layers):
            if i in exit_layer_set:
                hooks.append(layer.register_forward_hook(make_hook(i)))

        try:
            with torch.no_grad():
                _loss, final_logits, _pkv = model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        # Target distribution from the full model (last token)
        target_logits = final_logits[:, -1, :].detach()  # (1, vocab_size)
        target_log_probs = F.log_softmax(target_logits, dim=-1)

        # Compute KL loss for each exit classifier
        optimizer.zero_grad()
        step_kl = torch.tensor(0.0)
        n_exits = 0

        for exit_idx, layer_idx in enumerate(wrapper.config.exit_layers):
            if layer_idx not in captured:
                continue
            hidden = captured[layer_idx]  # (1, d_model)
            exit_logits = wrapper.exit_classifiers[exit_idx](hidden)  # (1, vocab_size)
            exit_log_probs = F.log_softmax(exit_logits, dim=-1)

            # KL(target || exit) — train exit to match target distribution
            kl = F.kl_div(exit_log_probs, target_log_probs.exp(), reduction="batchmean")
            step_kl = step_kl + kl
            n_exits += 1

        if n_exits > 0:
            loss = step_kl / n_exits
            loss.backward()
            optimizer.step()
            total_kl += loss.item()
            count += 1

    mean_kl_loss = total_kl / max(count, 1)
    return {"mean_kl_loss": mean_kl_loss}
