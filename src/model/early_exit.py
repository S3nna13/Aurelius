"""Early Exit / Adaptive Computation for Aurelius transformer.

Implements intermediate exit points so that "easy" tokens can exit before all
N layers are processed, saving FLOPs at inference time.

References:
    - Graves 2016 (Adaptive Computation Time)
    - DejaVu 2023
    - SkipDecode 2024
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EarlyExitConfig:
    """Configuration for early exit behaviour."""

    n_exit_layers: int = 4  # how many intermediate exits to place
    exit_threshold: float = 0.9  # confidence threshold for early exit
    loss_weight: float = 0.3  # weight for intermediate exit losses during training
    placement: str = "uniform"  # "uniform" | "late" (last half only)


@dataclass
class ExitStats:
    """Statistics collected from an early-exit forward pass."""

    layer_exit_counts: list[int]  # how many tokens exited at each exit point
    mean_layers_used: float  # average computational depth
    flop_savings: float  # 1 - mean_layers_used / n_layers


class EarlyExitClassifier(nn.Module):
    """Lightweight exit head: LayerNorm + Linear(d_model, vocab_size).

    Returns logits and confidence (max softmax probability).
    Intentionally lightweight — no FFN, just norm + projection.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute exit logits and per-token confidence.

        Args:
            x: Hidden states of shape (B, T, d_model).

        Returns:
            logits:     (B, T, vocab_size)
            confidence: (B, T) — max softmax probability across vocabulary.
        """
        normed = self.norm(x)
        logits = self.linear(normed)
        confidence = logits.softmax(dim=-1).max(dim=-1).values  # (B, T)
        return logits, confidence


class EarlyExitTransformer(nn.Module):
    """Transformer with intermediate exit points.

    During training: all layers are always computed; intermediate exits provide
    auxiliary supervision.  Total loss = final_loss + loss_weight * mean(intermediate_losses).

    During inference with use_early_exit=True: computation halts once confidence
    exceeds exit_threshold at any exit point.

    Args:
        config:      AureliusConfig (or any object with the same fields).
        exit_config: EarlyExitConfig controlling exit placement and thresholds.
    """

    def __init__(self, config, exit_config: EarlyExitConfig | None = None) -> None:
        super().__init__()

        from src.model.attention import precompute_rope_frequencies, yarn_rope_frequencies
        from src.model.rms_norm import RMSNorm
        from src.model.transformer import TransformerBlock

        self.config = config
        self.exit_config = exit_config or EarlyExitConfig()

        # ---- core components (mirrors AureliusTransformer, built from parts) ----
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if getattr(config, "tie_embeddings", True):
            self.lm_head.weight = self.embed.weight

        # RoPE frequencies
        rope_type = getattr(config, "rope_scaling_type", "none")
        if rope_type == "yarn":
            freqs = yarn_rope_frequencies(
                config.head_dim,
                config.max_seq_len,
                config.rope_theta,
                scale=config.rope_scaling_factor,
                original_max_seq_len=config.rope_original_max_seq_len,
            )
        else:
            freqs = precompute_rope_frequencies(
                config.head_dim,
                config.max_seq_len,
                getattr(config, "rope_theta", 500_000.0),
            )
        self.register_buffer("freqs_cis", freqs, persistent=False)

        # ---- exit classifiers ----
        self.exit_positions: list[int] = self._get_exit_positions(config.n_layers)
        self.exit_classifiers = nn.ModuleList(
            [EarlyExitClassifier(config.d_model, config.vocab_size) for _ in self.exit_positions]
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Exit placement
    # ------------------------------------------------------------------

    def _get_exit_positions(self, n_layers: int) -> list[int]:
        """Return layer indices (0-indexed) where exit classifiers are placed.

        The final layer (index n_layers-1) is never an exit point — that is the
        main output head.

        uniform: evenly spaced across layers 0 ... n_layers-2.
        late:    exits in the second half of layers only.
        """
        n_exits = self.exit_config.n_exit_layers
        candidate_layers = list(range(n_layers - 1))  # exclude last layer

        placement = self.exit_config.placement
        if placement == "late":
            half = n_layers // 2
            candidate_layers = [i for i in candidate_layers if i >= half]

        if n_exits >= len(candidate_layers):
            return candidate_layers

        # Evenly spaced indices into candidate_layers
        step = len(candidate_layers) / n_exits
        positions = [candidate_layers[int(i * step)] for i in range(n_exits)]
        return positions

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T)
        labels: torch.Tensor | None = None,  # (B, T)
        use_early_exit: bool = False,
    ) -> tuple:
        """
        Training (labels provided):
            Computes all layers, collects intermediate losses, returns
            (total_loss, final_logits, None).

        Inference with use_early_exit=True:
            Stops computation once confidence > threshold at any exit point.
            Returns (None, logits, ExitStats).

        Inference with use_early_exit=False:
            Standard forward through all layers.
            Returns (None, final_logits, None).
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[:T]

        # Map from layer index -> classifier index for O(1) lookup
        exit_pos_map: dict[int, int] = {pos: idx for idx, pos in enumerate(self.exit_positions)}

        # ---- Training path --------------------------------------------------------
        if labels is not None:
            intermediate_losses: list[torch.Tensor] = []

            for layer_idx, layer in enumerate(self.layers):
                x, _ = layer(x, freqs_cis)

                if layer_idx in exit_pos_map:
                    clf_idx = exit_pos_map[layer_idx]
                    exit_logits, _ = self.exit_classifiers[clf_idx](x)
                    # Shift for next-token prediction
                    shift_logits = exit_logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss_i = F.cross_entropy(
                        shift_logits.view(-1, self.config.vocab_size),
                        shift_labels.view(-1),
                    )
                    intermediate_losses.append(loss_i)

            # Final head
            x_normed = self.norm(x)
            final_logits = self.lm_head(x_normed)
            shift_logits = final_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            final_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

            if intermediate_losses:
                aux_loss = torch.stack(intermediate_losses).mean()
                total_loss = final_loss + self.exit_config.loss_weight * aux_loss
            else:
                total_loss = final_loss

            return total_loss, final_logits, None

        # ---- Inference with early exit -------------------------------------------
        if use_early_exit:
            # exited_logits accumulates the logit vector for each (b, t) pair
            exited_logits = torch.zeros(B, T, self.config.vocab_size, device=input_ids.device)
            exited = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)

            # +1 slot for tokens that never hit an exit and use the final head
            n_exit_slots = len(self.exit_positions) + 1
            layer_exit_counts = [0] * n_exit_slots

            # Track the depth (number of layers used) for each token
            token_depth = torch.full((B, T), float(self.config.n_layers), device=input_ids.device)

            for layer_idx, layer in enumerate(self.layers):
                if exited.all():
                    break
                x, _ = layer(x, freqs_cis)

                if layer_idx in exit_pos_map:
                    clf_idx = exit_pos_map[layer_idx]
                    exit_logits, confidence = self.exit_classifiers[clf_idx](x)

                    exits_here = (~exited) & (confidence > self.exit_config.exit_threshold)

                    if exits_here.any():
                        exited_logits[exits_here] = exit_logits[exits_here]
                        token_depth[exits_here] = float(layer_idx + 1)
                        exited = exited | exits_here
                        layer_exit_counts[clf_idx] += int(exits_here.sum().item())

            # Remaining tokens go through the final head
            x_normed = self.norm(x)
            final_logits = self.lm_head(x_normed)
            remaining = ~exited
            if remaining.any():
                exited_logits[remaining] = final_logits[remaining]
                layer_exit_counts[-1] += int(remaining.sum().item())

            mean_layers_used = float(token_depth.mean().item())
            flop_savings = max(0.0, min(1.0, 1.0 - mean_layers_used / self.config.n_layers))

            stats = ExitStats(
                layer_exit_counts=layer_exit_counts,
                mean_layers_used=mean_layers_used,
                flop_savings=flop_savings,
            )
            return None, exited_logits, stats

        # ---- Standard inference (no early exit) ----------------------------------
        for layer in self.layers:
            x, _ = layer(x, freqs_cis)

        x_normed = self.norm(x)
        final_logits = self.lm_head(x_normed)
        return None, final_logits, None

    # ------------------------------------------------------------------
    # Convenience: compute exit stats
    # ------------------------------------------------------------------

    def compute_exit_stats(self, input_ids: torch.Tensor) -> ExitStats:
        """Run a single forward pass with early exit and return ExitStats."""
        self.train(False)
        with torch.no_grad():
            _, _, stats = self.forward(input_ids, use_early_exit=True)
        return stats


# ---------------------------------------------------------------------------
# Profiling utility
# ---------------------------------------------------------------------------


def profile_exit_distribution(
    model: EarlyExitTransformer,
    input_ids: torch.Tensor,
    n_runs: int = 10,
) -> dict:
    """Run n_runs forward passes and aggregate exit statistics.

    Returns:
        dict with keys:
            'mean_exit_layer':    float
            'std_exit_layer':     float
            'flop_savings':       float
            'per_layer_exit_rate': list[float]
    """
    model.train(False)

    all_mean_layers: list[float] = []
    all_flop_savings: list[float] = []
    n_exit_slots = len(model.exit_positions) + 1
    accumulated_counts = [0] * n_exit_slots
    total_tokens = 0

    with torch.no_grad():
        for _ in range(n_runs):
            stats = model.compute_exit_stats(input_ids)
            all_mean_layers.append(stats.mean_layers_used)
            all_flop_savings.append(stats.flop_savings)
            for i, c in enumerate(stats.layer_exit_counts):
                accumulated_counts[i] += c
            total_tokens += sum(stats.layer_exit_counts)

    mean_vals = torch.tensor(all_mean_layers)
    mean_exit_layer = float(mean_vals.mean().item())
    std_exit_layer = float(mean_vals.std().item()) if n_runs > 1 else 0.0
    flop_savings = float(torch.tensor(all_flop_savings).mean().item())

    if total_tokens > 0:
        per_layer_exit_rate = [c / total_tokens for c in accumulated_counts]
    else:
        per_layer_exit_rate = [0.0] * n_exit_slots

    return {
        "mean_exit_layer": mean_exit_layer,
        "std_exit_layer": std_exit_layer,
        "flop_savings": flop_savings,
        "per_layer_exit_rate": per_layer_exit_rate,
    }
