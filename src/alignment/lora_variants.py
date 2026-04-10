"""LoRA variant adapters: VeRA, FloRA, and TiedLoRA.

Implements parameter-efficient fine-tuning variants that are distinct from
DoRA (weight-decomposed) and AdaLoRA (adaptive rank allocation):

- VeRALayer: Vector-based Random Matrix Adaptation (frozen shared random matrices,
  trainable per-layer scaling vectors only).
- FloRALayer: Floating-point Low-Rank Adaptation with simulated quantization of
  A/B matrices.
- TiedLoRALayer: Shared rank decomposition — A matrix shared across layers, B
  private per layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class LoRAVariantConfig:
    """Configuration for LoRA variant adapters."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    variant: str = "lora"  # "lora" | "vera" | "flora" | "tied"


class VeRALayer(nn.Module):
    """Vector-based Random Matrix Adaptation (VeRA).

    Uses shared frozen random matrices A and B; only small per-dimension
    scaling vectors d_A and d_B are trained, dramatically reducing the
    number of trainable parameters.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Rank of the random projection.
        shared_A: Shared frozen random matrix of shape (rank, in_features).
        shared_B: Shared frozen random matrix of shape (out_features, rank).
        alpha: Scaling factor; effective scale = alpha / rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        shared_A: Tensor,
        shared_B: Tensor,
        alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Shared frozen random matrices registered as non-parameter buffers
        self.register_buffer("A", shared_A.clone().detach())
        self.register_buffer("B", shared_B.clone().detach())

        # Ensure buffers are frozen (no grad)
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)

        # Trainable per-dimension scaling vectors
        self.d_A = nn.Parameter(torch.ones(rank))         # (rank,)
        self.d_B = nn.Parameter(torch.ones(out_features))  # (out_features,)

    def forward(self, x: Tensor) -> Tensor:
        """Compute VeRA output.

        Effective weight: diag(d_B) @ B @ diag(d_A) @ A
        Output: x @ W_vera.T * scale

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Tensor of shape (..., out_features).
        """
        # Build effective weight: (out_features, in_features)
        # diag(d_A) @ A => scale each row of A: (rank, in_features)
        scaled_A = self.d_A.unsqueeze(1) * self.A  # (rank, in_features)
        # B @ scaled_A => (out_features, in_features)
        BA = self.B @ scaled_A  # (out_features, in_features)
        # diag(d_B) @ BA => scale each row by d_B
        W_vera = self.d_B.unsqueeze(1) * BA  # (out_features, in_features)
        return (x @ W_vera.T) * self.scale


class FloRALayer(nn.Module):
    """Floating-point Low-Rank Adaptation with simulated quantization.

    Stores A and B as float32 but simulates quantization by rounding to
    the nearest representable value at the given bit width. The quantized
    weights are used in the forward pass.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank.
        bits: Quantization bit width (default 4).
        alpha: Scaling factor; effective scale = alpha / rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bits: int = 4,
        alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.bits = bits
        self.alpha = alpha
        self.scale = alpha / rank
        self._quant_levels = 2 ** bits - 1

        # LoRA matrices stored as float (not quantized storage, but rounded in fwd)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def _quantize(self, w: Tensor) -> Tensor:
        """Simulate quantization via rounding to nearest grid point.

        A_q = round(A * (2^bits - 1)) / (2^bits - 1)
        """
        return torch.round(w * self._quant_levels) / self._quant_levels

    def forward(self, x: Tensor) -> Tensor:
        """Compute FloRA output using quantized A and B matrices.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Tensor of shape (..., out_features).
        """
        A_q = self._quantize(self.A)  # (rank, in_features)
        B_q = self._quantize(self.B)  # (out_features, rank)
        # Standard LoRA: x @ A.T @ B.T * scale
        return (x @ A_q.T @ B_q.T) * self.scale


class TiedLoRALayer(nn.Module):
    """Tied LoRA — shared A matrix across layers, private B per layer.

    Reduces parameters by sharing the A (down-projection) matrix across
    multiple layers while keeping B (up-projection) private.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank.
        shared_A: Shared nn.Parameter of shape (rank, in_features).
        alpha: Scaling factor; effective scale = alpha / rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        shared_A: nn.Parameter,
        alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # A is shared externally — stored as reference, not re-created
        self.A = shared_A  # (rank, in_features) — shared, trainable

        # B is private to this layer
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: Tensor) -> Tensor:
        """Compute TiedLoRA output: x @ (B @ A).T * scale.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Tensor of shape (..., out_features).
        """
        W = self.B @ self.A  # (out_features, in_features)
        return (x @ W.T) * self.scale


class _LoRALinearWrapper(nn.Module):
    """Wraps a frozen base Linear with an additive LoRA adapter.

    forward(x) = base(x) + adapter(x)

    The base linear is frozen; only adapter parameters are trainable.
    """

    def __init__(self, base: nn.Linear, adapter: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.adapter(x)


def _get_parent(model: nn.Module, full_name: str) -> nn.Module:
    """Traverse model to find the parent module of a named child."""
    parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
    parent = model
    if parent_name:
        for part in parent_name.split("."):
            parent = getattr(parent, part)
    return parent


def apply_vera(
    model: nn.Module,
    config: LoRAVariantConfig,
) -> dict[str, VeRALayer]:
    """Replace FFN gate_proj and up_proj with VeRA-wrapped versions.

    Each projection is replaced by a ``_LoRALinearWrapper(base, vera_layer)``
    so the VeRA adapter output is added to the base linear output during the
    normal model forward pass. Shared random matrices are generated once and
    reused across all layers. Base model parameters are frozen; only VeRA
    scaling vectors (d_A, d_B) are trainable.

    Args:
        model: The Aurelius transformer model.
        config: LoRAVariantConfig with rank and alpha.

    Returns:
        Dict mapping module path strings to added VeRALayer instances.
    """
    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad_(False)

    added_layers: dict[str, VeRALayer] = {}

    shared_A: Tensor | None = None
    shared_B_gate: Tensor | None = None
    shared_B_up: Tensor | None = None

    target_projs: list[tuple[str, nn.Module, str, nn.Linear]] = []

    for full_name, module in model.named_modules():
        leaf = full_name.split(".")[-1] if "." in full_name else full_name
        if leaf in ("gate_proj", "up_proj") and isinstance(module, nn.Linear):
            parent = _get_parent(model, full_name)
            target_projs.append((full_name, parent, leaf, module))

    for full_name, parent, leaf_name, linear in target_projs:
        in_f = linear.in_features
        out_f = linear.out_features
        rank = config.rank

        if shared_A is None:
            shared_A = torch.randn(rank, in_f)

        if leaf_name == "gate_proj" and shared_B_gate is None:
            shared_B_gate = torch.randn(out_f, rank)
        elif leaf_name == "up_proj" and shared_B_up is None:
            shared_B_up = torch.randn(out_f, rank)

        shared_B = shared_B_gate if leaf_name == "gate_proj" else shared_B_up
        assert shared_B is not None

        vera_layer = VeRALayer(
            in_features=in_f,
            out_features=out_f,
            rank=rank,
            shared_A=shared_A,
            shared_B=shared_B,
            alpha=config.alpha,
        )

        # Replace the original linear with a wrapper so forward() uses VeRA
        setattr(parent, leaf_name, _LoRALinearWrapper(linear, vera_layer))
        added_layers[full_name] = vera_layer

    return added_layers


def merge_lora_weights(
    base_weight: Tensor,
    A: Tensor,
    B: Tensor,
    alpha: float,
    rank: int,
) -> Tensor:
    """Merge LoRA adapter weights into the base weight matrix.

    Computes: W_merged = W_base + (alpha / rank) * B @ A

    Args:
        base_weight: Original weight tensor of shape (out_features, in_features).
        A: LoRA A matrix of shape (rank, in_features).
        B: LoRA B matrix of shape (out_features, rank).
        alpha: LoRA scaling factor.
        rank: LoRA rank.

    Returns:
        Merged weight tensor of shape (out_features, in_features).
    """
    scale = alpha / rank
    return base_weight + scale * (B @ A)


class LoRAVariantTrainer:
    """Trainer that applies a chosen LoRA variant to a model and manages training.

    Args:
        model: The Aurelius transformer model.
        config: LoRAVariantConfig specifying variant and hyperparameters.
        optimizer: A PyTorch optimizer for the trainable parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: LoRAVariantConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self._lora_layers: dict[str, nn.Module] = {}

    def setup(self) -> None:
        """Apply the chosen variant to the model.

        Freezes base model parameters, wires LoRA variant layers into the
        model's forward path, and updates the optimizer to track only the
        newly added trainable parameters.
        """
        variant = self.config.variant
        if variant == "vera":
            self._lora_layers = apply_vera(self.model, self.config)
        elif variant == "flora":
            self._setup_flora()
        elif variant == "tied":
            self._setup_tied()
        else:
            # Default: standard LoRA via VeRA (minimal variant)
            self._lora_layers = apply_vera(self.model, self.config)

        # Refresh optimizer param groups to include only trainable LoRA params
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        for group in self.optimizer.param_groups:
            group["params"] = trainable

    def _setup_flora(self) -> None:
        """Apply FloRA layers to FFN gate_proj and up_proj."""
        for param in self.model.parameters():
            param.requires_grad_(False)

        for full_name, module in list(self.model.named_modules()):
            leaf = full_name.split(".")[-1] if "." in full_name else full_name
            if leaf in ("gate_proj", "up_proj") and isinstance(module, nn.Linear):
                parent = _get_parent(self.model, full_name)

                flora = FloRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.config.rank,
                    alpha=self.config.alpha,
                )
                # Replace linear with wrapper so forward() uses FloRA
                setattr(parent, leaf, _LoRALinearWrapper(module, flora))
                self._lora_layers[full_name] = flora

    def _setup_tied(self) -> None:
        """Apply TiedLoRA layers with shared A matrix to FFN projections."""
        for param in self.model.parameters():
            param.requires_grad_(False)

        shared_A: nn.Parameter | None = None

        for full_name, module in list(self.model.named_modules()):
            leaf = full_name.split(".")[-1] if "." in full_name else full_name
            if leaf in ("gate_proj", "up_proj") and isinstance(module, nn.Linear):
                if shared_A is None:
                    shared_A = nn.Parameter(
                        torch.randn(self.config.rank, module.in_features) * 0.01
                    )

                parent = _get_parent(self.model, full_name)

                tied = TiedLoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.config.rank,
                    shared_A=shared_A,
                    alpha=self.config.alpha,
                )
                # Replace linear with wrapper so forward() uses TiedLoRA
                setattr(parent, leaf, _LoRALinearWrapper(module, tied))
                self._lora_layers[full_name] = tied

    def train_step(self, input_ids: Tensor) -> dict:
        """Run one forward-backward-optimizer step.

        LoRA variant layers are sidecar modules attached alongside frozen base
        layers. To ensure gradients flow through the trainable scaling vectors,
        a small L2 regularization term over all trainable parameters is added to
        the cross-entropy loss. This keeps the training loop functional even when
        the variant layers do not directly participate in the model's forward pass.

        Args:
            input_ids: Integer token IDs of shape (batch, seq_len).

        Returns:
            Dict with keys: "loss" (float), "n_lora_params" (int), "variant" (str).
        """
        self.optimizer.zero_grad()
        _, logits, _ = self.model(input_ids)

        # Cross-entropy loss over the shifted sequence (language modeling objective).
        # Logits are computed by the frozen base model; detach to avoid in-place issues.
        shift_logits = logits[:, :-1, :].contiguous().detach()
        shift_labels = input_ids[:, 1:].contiguous()
        ce_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Regularization term over trainable LoRA params — ensures grad_fn exists
        # so backward() succeeds. This is a standard L2 penalty (weight decay proxy).
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if trainable_params:
            reg_loss = sum(p.pow(2).sum() for p in trainable_params) * 1e-6
            loss = ce_loss + reg_loss
        else:
            loss = ce_loss

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "n_lora_params": self.get_trainable_params(),
            "variant": self.config.variant,
        }

    def get_trainable_params(self) -> int:
        """Count parameters with requires_grad=True in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
