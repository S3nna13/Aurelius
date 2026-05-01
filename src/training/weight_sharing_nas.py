"""Weight-sharing NAS: Single-path One-Shot NAS and Slimmable Networks.

Implements Once-For-All (OFA) style weight sharing NAS where a single supernet
is trained and then subnets are extracted without retraining.

Key ideas:
- SlimmableLinear: a single weight matrix that can be sliced to different sizes
- OFABlock: elastic FFN block with SwiGLU and slimmable projections
- OneShotSuperNet: supernet with max-capacity blocks; forward with any SubnetSpec
- OFATrainer: trains supernet by sampling random subnets each step

References:
  Cai et al. 2020 "Once-for-All: Train One Network and Specialize it for
  Efficient Deployment" arXiv:1908.09791
  Yu et al. 2019 "Slimmable Neural Networks" arXiv:1812.08928
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config & Spec
# ---------------------------------------------------------------------------


@dataclass
class OFAConfig:
    """Configuration for the Once-For-All supernet."""

    d_model_choices: list = field(default_factory=lambda: [32, 48, 64])
    d_ff_choices: list = field(default_factory=lambda: [64, 128, 256])
    n_layer_choices: list = field(default_factory=lambda: [1, 2])
    elastic_kernel: bool = False
    vocab_size: int = 256


@dataclass
class SubnetSpec:
    """Specifies a concrete subnet configuration to extract from the supernet."""

    d_model: int
    d_ff: int
    n_layers: int


# ---------------------------------------------------------------------------
# SlimmableLinear
# ---------------------------------------------------------------------------


class SlimmableLinear(nn.Module):
    """Linear layer with a single weight matrix that can be sliced.

    Stores the full max-size weight and bias; at forward time slices down to
    the requested (in_features, out_features) dimensions.
    """

    def __init__(self, max_in: int, max_out: int) -> None:
        super().__init__()
        self.max_in = max_in
        self.max_out = max_out
        self.weight = nn.Parameter(torch.empty(max_out, max_in))
        self.bias = nn.Parameter(torch.zeros(max_out))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(
        self,
        x: Tensor,
        in_features: int | None = None,
        out_features: int | None = None,
    ) -> Tensor:
        in_f = in_features if in_features is not None else self.max_in
        out_f = out_features if out_features is not None else self.max_out
        w = self.weight[:out_f, :in_f]
        b = self.bias[:out_f]
        return F.linear(x, w, b)


# ---------------------------------------------------------------------------
# OFABlock
# ---------------------------------------------------------------------------


class OFABlock(nn.Module):
    """Elastic FFN block with SwiGLU activation and slimmable projections.

    Supports any (d_model, d_ff) up to the max dimensions it was built with.
    """

    def __init__(self, max_d_model: int, max_d_ff: int) -> None:
        super().__init__()
        self.max_d_model = max_d_model
        self.max_d_ff = max_d_ff
        # SwiGLU: gate_proj and up_proj both project d_model -> d_ff,
        # down_proj projects d_ff -> d_model.
        self.gate_proj = SlimmableLinear(max_d_model, max_d_ff)
        self.up_proj = SlimmableLinear(max_d_model, max_d_ff)
        self.down_proj = SlimmableLinear(max_d_ff, max_d_model)
        self.norm = nn.LayerNorm(max_d_model)

    def forward(
        self,
        x: Tensor,
        d_model: int | None = None,
        d_ff: int | None = None,
    ) -> Tensor:
        dm = d_model if d_model is not None else self.max_d_model
        df = d_ff if d_ff is not None else self.max_d_ff

        # Slice norm weight/bias to active d_model
        norm_w = self.norm.weight[:dm]
        norm_b = self.norm.bias[:dm]
        x_slice = x[..., :dm]
        normed = F.layer_norm(x_slice, (dm,), norm_w, norm_b, eps=1e-5)

        gate = self.gate_proj(normed, in_features=dm, out_features=df)
        up = self.up_proj(normed, in_features=dm, out_features=df)
        hidden = F.silu(gate) * up
        out = self.down_proj(hidden, in_features=df, out_features=dm)
        return x_slice + out


# ---------------------------------------------------------------------------
# OneShotSuperNet
# ---------------------------------------------------------------------------


class OneShotSuperNet(nn.Module):
    """One-shot supernet that covers all subnet configurations.

    Built at maximum capacity; forward can be restricted to any SubnetSpec.
    """

    def __init__(self, config: OFAConfig) -> None:
        super().__init__()
        self.config = config
        self.max_d_model = max(config.d_model_choices)
        self.max_d_ff = max(config.d_ff_choices)
        self.max_n_layers = max(config.n_layer_choices)
        self.vocab_size = config.vocab_size

        self.embed = nn.Embedding(config.vocab_size, self.max_d_model)
        self.blocks = nn.ModuleList(
            [OFABlock(self.max_d_model, self.max_d_ff) for _ in range(self.max_n_layers)]
        )
        # Head: always full d_model -> vocab_size (output projection fixed)
        self.head = nn.Linear(self.max_d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        spec: SubnetSpec | None = None,
    ) -> tuple:
        dm = spec.d_model if spec is not None else self.max_d_model
        df = spec.d_ff if spec is not None else self.max_d_ff
        n_layers = spec.n_layers if spec is not None else self.max_n_layers

        x = self.embed(input_ids)[..., :dm]  # (B, T, dm)
        for i in range(n_layers):
            x = self.blocks[i](x, d_model=dm, d_ff=df)

        # Project back to full d_model for head (pad if needed)
        if dm < self.max_d_model:
            B, T, _ = x.shape
            pad = x.new_zeros(B, T, self.max_d_model - dm)
            x = torch.cat([x, pad], dim=-1)

        logits = self.head(x)  # (B, T, vocab_size)
        return (None, logits, None)

    def sample_subnet(self) -> SubnetSpec:
        """Randomly sample a valid subnet specification."""
        return SubnetSpec(
            d_model=random.choice(self.config.d_model_choices),  # noqa: S311
            d_ff=random.choice(self.config.d_ff_choices),  # noqa: S311
            n_layers=random.choice(self.config.n_layer_choices),  # noqa: S311
        )

    def get_subnet_params(self, spec: SubnetSpec) -> int:
        """Estimate parameter count for a given subnet spec."""
        dm = spec.d_model
        df = spec.d_ff
        nl = spec.n_layers
        # Embedding: vocab_size * dm
        embed_params = self.vocab_size * dm
        # Each OFABlock: gate_proj + up_proj + down_proj + norm + biases
        block_params = (
            dm * df
            + df  # gate_proj weight + bias
            + dm * df
            + df  # up_proj weight + bias
            + df * dm
            + dm  # down_proj weight + bias
            + dm
            + dm  # LayerNorm weight + bias
        )
        # Head: dm * vocab_size (no bias)
        head_params = dm * self.vocab_size
        return embed_params + nl * block_params + head_params


# ---------------------------------------------------------------------------
# OFATrainer
# ---------------------------------------------------------------------------


class OFATrainer:
    """Trainer that optimizes the supernet via random subnet sampling."""

    def __init__(
        self,
        supernet: OneShotSuperNet,
        optimizer: torch.optim.Optimizer,
        config: OFAConfig,
    ) -> None:
        self.supernet = supernet
        self.optimizer = optimizer
        self.config = config

    def train_step(self, input_ids: Tensor) -> dict:
        """Sample a random subnet, forward, compute CE loss, backward, step."""
        self.supernet.train()
        spec = self.supernet.sample_subnet()

        self.optimizer.zero_grad()
        _, logits, _ = self.supernet(input_ids, spec=spec)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "subnet_spec": {"d_model": spec.d_model, "d_ff": spec.d_ff, "n_layers": spec.n_layers},
            "n_params": self.supernet.get_subnet_params(spec),
        }

    def evaluate_subnet(self, spec: SubnetSpec, data: list) -> float:
        """Evaluate a specific subnet on a list of input tensors, return mean CE loss."""
        self.supernet.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for input_ids in data:
                _, logits, _ = self.supernet(input_ids, spec=spec)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total_loss += loss.item()
                count += 1
        return total_loss / count if count > 0 else 0.0
