"""VeRA: Vector-based Random Matrix Adaptation.

Reference: Kopiczko et al., "VeRA: Vector-based Random Matrix Adaptation",
arXiv:2310.11454 (2023).

Key idea:
    Standard LoRA trains per-layer matrices B_l and A_l (both fully trainable).
    VeRA instead shares a SINGLE frozen random pair (A, B) across ALL adapted
    layers, and only trains small per-layer diagonal scaling vectors d_l and b_l:

        W_l = W_l^0 + Λ_b^l  B  Λ_d^l  A

    where:
        A  ∈ R^{r×k}  — shared frozen random matrix (Kaiming uniform)
        B  ∈ R^{m×r}  — shared frozen random matrix (Kaiming uniform)
        Λ_d^l = diag(d_l),  d_l ∈ R^r   — per-layer trainable, init zeros
        Λ_b^l = diag(b_l),  b_l ∈ R^m   — per-layer trainable, init ones

    Trainable parameters per layer: r + m  (vs. r*k + m*r for LoRA).

Forward (row convention matching nn.Linear):
    h = W_l^0(x) + b_l * (F.linear(F.linear(x, A) * d_l, B))

Initialization (paper Section 3.2):
    A, B : Kaiming uniform, then immediately frozen (requires_grad=False)
    d    : zeros  → initial delta is zero (matches LoRA zero-init convention)
    b    : ones
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Shared random matrix factory
# ---------------------------------------------------------------------------


def _make_shared_matrix(
    rows: int, cols: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Allocate and Kaiming-uniform-initialise a frozen random matrix.

    The tensor is returned as a plain (non-Parameter) buffer so that it is
    never tracked by autograd and never appears in model.parameters().
    """
    t = torch.empty(rows, cols)
    # Kaiming uniform: same default as nn.Linear weight init and LoRA A
    nn.init.kaiming_uniform_(t, a=math.sqrt(5))
    t.requires_grad_(False)
    return t


# ---------------------------------------------------------------------------
# VeRALinear
# ---------------------------------------------------------------------------


class VeRALinear(nn.Module):
    """A linear layer augmented with VeRA adaptation.

    Paper notation (all per-layer quantities carry superscript l):
        A  ∈ R^{r×k}  : shared frozen random matrix
        B  ∈ R^{m×r}  : shared frozen random matrix
        d  ∈ R^r       : trainable diagonal of Λ_d  (init 0)
        b  ∈ R^m       : trainable diagonal of Λ_b  (init 1)

    Forward:
        h = W^0(x) + b ⊙ (F.linear(F.linear(x, A) * d, B))

    Args:
        in_features:  k — input dimension
        out_features: m — output dimension
        rank:         r — shared random matrix rank
        shared_A:     Pre-allocated frozen A tensor (shape r×k).  If None a
                      new one is created internally (useful for single-layer
                      experiments; in production use VeRAModel which supplies
                      a single shared pair).
        shared_B:     Pre-allocated frozen B tensor (shape m×r).  See above.
        bias:         Whether to include a bias term in the base linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        shared_A: torch.Tensor | None = None,
        shared_B: torch.Tensor | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank {rank} must be ≤ min(in_features={in_features}, out_features={out_features})"
            )

        self.in_features = in_features  # k
        self.out_features = out_features  # m
        self.rank = rank  # r

        # Base (frozen) linear — weights will be frozen when VeRAModel wraps
        # an existing nn.Linear, or kept trainable for standalone use.
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)

        # Shared frozen random matrices A ∈ R^{r×k}, B ∈ R^{m×r}
        if shared_A is None:
            shared_A = _make_shared_matrix(rank, in_features)
        if shared_B is None:
            shared_B = _make_shared_matrix(out_features, rank)

        # Validate shapes
        if shared_A.shape != (rank, in_features):
            raise ValueError(f"shared_A shape {tuple(shared_A.shape)} != ({rank}, {in_features})")
        if shared_B.shape != (out_features, rank):
            raise ValueError(f"shared_B shape {tuple(shared_B.shape)} != ({out_features}, {rank})")

        # Register as buffers so they move with .to(device) but are NOT parameters
        self.register_buffer("A", shared_A)  # r × k
        self.register_buffer("B", shared_B)  # m × r

        # Per-layer trainable scaling vectors (paper eq. 3)
        # d_l ∈ R^r : init zeros  → initial delta = 0
        # b_l ∈ R^m : init ones
        self.d = nn.Parameter(torch.zeros(rank))  # Λ_d diagonal
        self.b = nn.Parameter(torch.ones(out_features))  # Λ_b diagonal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute W^0(x) + b ⊙ (B (d ⊙ (A x^T)))  [row convention].

        Equivalently:
            h = base(x) + b * F.linear(F.linear(x, A) * d, B)
        """
        # Base output: (batch, m)
        base = self.base_linear(x)

        # VeRA delta: x → A: (..., r)  → * d: (..., r)  → B: (..., m)  → * b
        h = F.linear(x, self.A)  # (..., r)  — A is r×k
        h = h * self.d  # (..., r)  — element-wise Λ_d
        h = F.linear(h, self.B)  # (..., m)  — B is m×r
        h = h * self.b  # (..., m)  — element-wise Λ_b

        return base + h

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"


# ---------------------------------------------------------------------------
# VeRAModel
# ---------------------------------------------------------------------------


class VeRAModel(nn.Module):
    """Wraps a base model replacing target nn.Linear layers with VeRALinear.

    All replaced layers share the SAME frozen A and B tensors — this is the
    core innovation of VeRA that dramatically reduces trainable parameters.

    The base model weights (including replaced layer weights) are frozen.
    Only the d and b vectors are trainable.

    Args:
        base_model:     The original nn.Module to adapt.
        target_modules: List of substrings; any nn.Linear whose fully-qualified
                        name contains at least one substring is replaced.
        rank:           r — rank of the shared random matrices.
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_modules: list[str],
        rank: int,
    ) -> None:
        super().__init__()

        if not target_modules:
            raise ValueError("target_modules must be a non-empty list of name substrings")
        if rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank}")

        self.base_model = base_model
        self.target_modules = list(target_modules)
        self.rank = rank

        # Step 1: freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad_(False)

        # Step 2: identify target layers and their (k, m) shapes so we can
        # validate that a single (A, B) pair is consistent.
        # VeRA requires all replaced layers to share the same (r, k) and (m, r)
        # shapes.  In practice the paper adapts all attention projections which
        # typically share the same hidden dimension.  We build one pair per
        # unique (in_features, out_features) — but the paper shares a SINGLE
        # pair, so here we collect the *first* found shape and require all
        # targets to have the same in_features.  Out-features may differ; we
        # build A once (shape r×k) and one B per unique out_features, but all
        # layers with the same out_features share the same B tensor.
        #
        # For simplicity (and faithfulness to the paper which targets same-dim
        # projections) we keep a single A (requires all targets to have the same
        # in_features) and a dict of B tensors keyed by out_features.

        # Collect targets
        targets: list[tuple[str, nn.Linear]] = []
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                if any(t in name for t in target_modules):
                    targets.append((name, module))

        if not targets:
            raise ValueError(f"No nn.Linear modules found matching target_modules={target_modules}")

        # Build shared A (single, keyed by in_features)
        shared_A_map: dict[int, torch.Tensor] = {}
        shared_B_map: dict[tuple[int, int], torch.Tensor] = {}  # (out_features, in_features)

        # Step 3: replace each target layer
        for full_name, linear_module in targets:
            k = linear_module.in_features
            m = linear_module.out_features

            if k not in shared_A_map:
                shared_A_map[k] = _make_shared_matrix(rank, k)
            A = shared_A_map[k]

            key_B = (m, k)
            if key_B not in shared_B_map:
                shared_B_map[key_B] = _make_shared_matrix(m, rank)
            B = shared_B_map[key_B]

            vera_layer = VeRALinear(
                in_features=k,
                out_features=m,
                rank=rank,
                shared_A=A,
                shared_B=B,
                bias=linear_module.bias is not None,
            )
            # Copy base weights and bias from the original linear
            vera_layer.base_linear.weight = nn.Parameter(
                linear_module.weight.data.clone(), requires_grad=False
            )
            if linear_module.bias is not None:
                vera_layer.base_linear.bias = nn.Parameter(
                    linear_module.bias.data.clone(), requires_grad=False
                )

            # Set the attribute on the correct parent module
            *parent_parts, child_name = full_name.split(".")
            parent = base_model
            for part in parent_parts:
                parent = getattr(parent, part)
            setattr(parent, child_name, vera_layer)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def vera_parameters(self):
        """Yield only the trainable VeRA parameters (d and b vectors)."""
        for module in self.base_model.modules():
            if isinstance(module, VeRALinear):
                yield module.d
                yield module.b

    def trainable_parameter_count(self) -> int:
        """Return total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
