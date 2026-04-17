"""
S-LoRA: Scalable Serving of Many LoRA Adapters
Reference: Sheng et al., 2023. arXiv:2311.03285

Core math (per linear layer, per request i):
    output_i = x_i @ W_base^T + x_i @ A_i^T @ B_i^T * scaling_i

S-LoRA batched forward (heterogeneous adapter batch):
    1. Base pass: H_base = X @ W^T  (shared for all)
    2. LoRA pass: for each unique adapter_id, gather x[adapter_mask],
                  compute delta, scatter back
    3. Final:     H = H_base + H_lora
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRAAdapter — stores (A, B, rank r, scaling α/r) for one adapter
# ---------------------------------------------------------------------------

@dataclass
class LoRAAdapter:
    """
    Holds the low-rank weight matrices for a single LoRA adapter.

    Fields match paper notation:
        adapter_id : unique string identifier
        A          : down-projection matrix, shape (r, d_in)
        B          : up-projection matrix,   shape (d_out, r)
        rank       : r  — rank of the decomposition
        scaling    : α / r  (merged scalar; often 1.0 for pre-scaled matrices)
    """
    adapter_id: str
    A: torch.Tensor   # (r, d_in)
    B: torch.Tensor   # (d_out, r)
    rank: int
    scaling: float


# ---------------------------------------------------------------------------
# SLoRARegistry — stores/swaps adapters, enforces capacity
# ---------------------------------------------------------------------------

class SLoRARegistry:
    """
    In-memory registry of active LoRA adapters.

    Mirrors the S-LoRA "adapter store" concept:
    adapters live here while being served; eviction is explicit (swap_out).

    Parameters
    ----------
    max_adapters : int
        Maximum number of concurrently loaded adapters.
    """

    def __init__(self, max_adapters: int = 32) -> None:
        self.max_adapters = max_adapters
        self._store: dict[str, LoRAAdapter] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def swap_in(
        self,
        adapter_id: str,
        A: torch.Tensor,
        B: torch.Tensor,
        rank: int,
        scaling: float = 1.0,
    ) -> None:
        """Register adapter tensors for serving.

        Parameters
        ----------
        adapter_id : unique key for this adapter
        A          : (r, d_in)  — down-projection
        B          : (d_out, r) — up-projection
        rank       : r
        scaling    : scalar multiplier α/r
        """
        if adapter_id in self._store:
            # Re-registration is a no-op replacement; no capacity charge.
            self._store[adapter_id] = LoRAAdapter(adapter_id, A, B, rank, scaling)
            return

        if len(self._store) >= self.max_adapters:
            raise RuntimeError(
                f"SLoRARegistry at capacity ({self.max_adapters} adapters). "
                "Call swap_out() to evict an adapter before adding a new one."
            )
        self._store[adapter_id] = LoRAAdapter(adapter_id, A, B, rank, scaling)

    def swap_out(self, adapter_id: str) -> None:
        """Evict adapter from registry (free its slot)."""
        if adapter_id not in self._store:
            raise KeyError(f"Adapter '{adapter_id}' not in registry.")
        del self._store[adapter_id]

    def get(self, adapter_id: str) -> LoRAAdapter:
        """Return adapter or raise KeyError if not loaded."""
        if adapter_id not in self._store:
            raise KeyError(
                f"Adapter '{adapter_id}' is not loaded in the registry. "
                "Call swap_in() first."
            )
        return self._store[adapter_id]

    def active_ids(self) -> list[str]:
        """Return list of currently-loaded adapter IDs."""
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, adapter_id: str) -> bool:
        return adapter_id in self._store


# ---------------------------------------------------------------------------
# SLoRALinear — batched heterogeneous-adapter linear layer
# ---------------------------------------------------------------------------

class SLoRALinear(nn.Module):
    """
    Linear layer with S-LoRA batched adapter computation.

    The base weight W_base is stored as a frozen nn.Linear (bias optional).
    LoRA deltas are fetched per-request from ``registry`` at forward time.

    Parameters
    ----------
    in_features  : d_in
    out_features : d_out
    registry     : SLoRARegistry to look up adapter matrices
    bias         : whether the base linear has a bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        registry: SLoRARegistry,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.registry = registry

        # Base weight W_base — shared across all requests in a batch.
        # Gradient is intentionally kept live so downstream callers can
        # fine-tune the base model separately; LoRA matrices in registry
        # are treated as frozen inference parameters.
        self.W_base = nn.Linear(in_features, out_features, bias=bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        adapter_ids: list[Optional[str]],
    ) -> torch.Tensor:
        """
        Batched heterogeneous-adapter forward pass.

        Parameters
        ----------
        x           : (B, T, d_in) — batch of token sequences
        adapter_ids : list of length B; adapter_ids[b] is the adapter ID
                      for request b, or None to apply base weights only.

        Returns
        -------
        H : (B, T, d_out)
        """
        B, T, d_in = x.shape

        if len(adapter_ids) != B:
            raise ValueError(
                f"len(adapter_ids)={len(adapter_ids)} must equal batch size B={B}."
            )

        # ------------------------------------------------------------------
        # Step 1: Base pass  H_base = X @ W_base^T     shape (B, T, d_out)
        # ------------------------------------------------------------------
        H = self.W_base(x)  # (B, T, d_out)

        # ------------------------------------------------------------------
        # Step 2: LoRA pass — group requests by adapter_id, compute delta,
        #         scatter back into H.
        # ------------------------------------------------------------------
        # Build a mapping: adapter_id → list of batch indices
        adapter_to_indices: dict[str, list[int]] = {}
        for b, aid in enumerate(adapter_ids):
            if aid is None:
                continue
            adapter_to_indices.setdefault(aid, []).append(b)

        for aid, indices in adapter_to_indices.items():
            adapter = self.registry.get(aid)  # raises KeyError if not loaded
            idx = torch.tensor(indices, dtype=torch.long, device=x.device)

            # Gather: x_sub shape (|indices|, T, d_in)
            x_sub = x[idx]

            # LoRA delta:
            #   lora_out = x_sub @ A^T @ B^T * scaling
            #   A : (r, d_in)  =>  x_sub @ A^T  gives (|idx|, T, r)
            #   B : (d_out, r) =>  (...) @ B^T  gives (|idx|, T, d_out)
            A = adapter.A.to(x.device)   # (r, d_in)  — no gradient
            B_mat = adapter.B.to(x.device)  # (d_out, r)

            with torch.no_grad():
                # Detach A, B so gradients do not flow through adapter weights.
                delta = (x_sub @ A.detach().T) @ B_mat.detach().T  # (|idx|, T, d_out)
                delta = delta * adapter.scaling

            # Scatter delta back (in-place add on a slice of H)
            H = H.clone()          # avoid in-place on leaf / autograd tape
            H[idx] = H[idx] + delta

        return H


# ---------------------------------------------------------------------------
# SLoRALayer — convenience wrapper: augments an existing nn.Linear
# ---------------------------------------------------------------------------

class SLoRALayer(nn.Module):
    """
    Thin wrapper that augments a pre-existing ``nn.Linear`` with S-LoRA
    serving without modifying the original module.

    Usage
    -----
    >>> base_linear = nn.Linear(32, 64, bias=False)
    >>> registry    = SLoRARegistry(max_adapters=16)
    >>> layer       = SLoRALayer(base_linear, registry)
    >>> out         = layer(x, adapter_ids)
    """

    def __init__(self, linear: nn.Linear, registry: SLoRARegistry) -> None:
        super().__init__()
        self.linear = linear
        self.registry = registry

        # Build an SLoRALinear that shares the same weight matrix.
        in_f = linear.in_features
        out_f = linear.out_features
        has_bias = linear.bias is not None

        self._slora = SLoRALinear(in_f, out_f, registry, bias=has_bias)
        # Share weights: point W_base to the wrapped linear.
        self._slora.W_base = linear

    def forward(
        self,
        x: torch.Tensor,
        adapter_ids: list[Optional[str]],
    ) -> torch.Tensor:
        return self._slora(x, adapter_ids)
