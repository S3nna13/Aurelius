"""ReMoDE — Residual Mixture of Depth and Experts.

Composite routing architecture that combines MoD and MoE:

1. MoD router decides which tokens to process (typically 30-50%)
2. MoE router within-selected-tokens routes them to experts
3. Shared experts process ALL tokens (residual stream)
4. Tokens not selected by MoD use residual connection only

This is the architecture used by DeepSeek-V4 and similar frontier models.
The "residual" part means unselected tokens skip the expensive computation
while the residual stream (shared experts) maintains information flow.

Reference: Combining MoD token routing with MoE expert routing creates
a powerful architecture where sparsity operates at both the token level
(which tokens compute) and the parameter level (which experts activate).
"""

from __future__ import annotations

from .config import ReMoDEConfig
from .mod import CapacityTracker, MoDRouter
from .moe import ExpertFFN, TopKRouter


class ReMoDELayer:
    """Residual Mixture of Depth and Experts layer.

    Architecture::

        Input
          │
          ├── MoD Router ── selects top-k tokens
          │     │
          │     ▼
          │   MoE Router ── routes selected tokens to experts
          │     │
          │     ▼
          │   Expert FFNs ── process selected tokens
          │     │
          │     ▼
          │   Weighted combine
          │     │
          ├── Shared Expert(s) ── process ALL tokens
          │     │
          └── Residual ── unselected tokens pass through unchanged
                │
                ▼
              Output
    """

    def __init__(self, config: ReMoDEConfig) -> None:
        self.config = config
        self.d_model = config.d_model

        # MoD: token-level depth routing
        self.mod_router = MoDRouter(
            d_model=config.d_model,
            router_type="learned",
            load_balance_weight=config.mod_aux_loss_coeff,
        )
        self.mod_tracker = CapacityTracker()

        # MoE: expert-level routing for selected tokens
        self.moe_router = TopKRouter(
            d_model=config.d_model,
            n_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            load_balance_alpha=config.load_balance_alpha,
        )

        # Expert FFNs
        self.experts = [
            ExpertFFN(config.d_model, config.d_ff) for _ in range(config.moe_num_experts)
        ]

        # Shared experts (process ALL tokens for residual signal)
        self.shared_experts = [
            ExpertFFN(config.d_model, config.d_ff) for _ in range(config.shared_experts)
        ]

    def forward(self, x: list[list[float]], layer_idx: int = 0) -> tuple[list[list[float]], float]:
        """Forward pass through ReMoDE layer.

        Args:
            x: token representations [n_tokens, d_model]
            layer_idx: layer index for tracking

        Returns:
            output: processed tokens
            total_aux_loss: sum of MoD aux loss + MoE load balance loss
        """
        n_tokens = len(x)
        if n_tokens == 0:
            return x, 0.0

        # 1. MoD: select which tokens to process through MoE
        selected_indices, skipped_indices, mod_aux_loss = self.mod_router.route(
            x, self.config.mod_capacity
        )

        # 2. Gather selected tokens
        selected_inputs = [x[i] for i in selected_indices]

        # 3. MoE: route selected tokens to experts
        if selected_inputs:
            expert_indices, expert_weights, moe_aux_loss = self.moe_router.route(selected_inputs)
        else:
            expert_indices, expert_weights, moe_aux_loss = [], [], 0.0

        # 4. Process each selected token through its assigned experts
        # Initialize output as copy of input (residual: unselected tokens pass through)
        output: list[list[float]] = [list(row) for row in x]

        for group_idx, orig_idx in enumerate(selected_indices):
            if group_idx >= len(expert_indices):
                continue

            token_experts = expert_indices[group_idx]
            token_weights = expert_weights[group_idx]

            token_output = [0.0] * self.d_model
            for exp_idx, weight in zip(token_experts, token_weights, strict=True):
                if exp_idx < len(self.experts):
                    expert_out = self.experts[exp_idx].forward(selected_inputs[group_idx])
                    for d in range(self.d_model):
                        token_output[d] += weight * expert_out[d]

            # Combine with residual (weighted by MoD score)
            moe_weight = 1.0 / (len(token_experts) + 1)
            for d in range(self.d_model):
                output[orig_idx][d] += moe_weight * token_output[d]

        # 5. Shared experts process ALL tokens (residual stream)
        for shared_exp in self.shared_experts:
            for tok_idx in range(n_tokens):
                shared_out = shared_exp.forward(x[tok_idx])
                for d in range(self.d_model):
                    output[tok_idx][d] += shared_out[d] / (len(self.shared_experts) + 1)

        # Track statistics
        total_aux_loss = mod_aux_loss + moe_aux_loss
        self.mod_tracker.record(layer_idx, len(selected_indices), n_tokens, self.mod_router._scores)

        return output, total_aux_loss


class ReMoDEBlock:
    """Full ReMoDE transformer block: attention + ReMoDE.

    Wraps a self-attention layer followed by a ReMoDE FFN layer.
    Attention runs on ALL tokens; only the FFN uses ReMoDE routing.
    """

    def __init__(self, config: ReMoDEConfig, attention_fn: object = None) -> None:
        self.config = config
        self.attention_fn = attention_fn if callable(attention_fn) else (lambda x: x)
        self.remode = ReMoDELayer(config)

    def forward(self, x: list[list[float]], layer_idx: int = 0) -> tuple[list[list[float]], float]:
        # Self-attention on ALL tokens
        attended = self.attention_fn(x)

        # ReMoDE FFN (MoD-routed MoE)
        output, aux_loss = self.remode.forward(attended, layer_idx)

        return output, aux_loss
