"""
src/interpretability/function_vectors.py

Function Vectors (FVs) in Large Language Models.

Based on: Hendel et al., 2023. "Function Vectors in Large Language Models."
arXiv:2310.15213.

Key insight: specific attention heads ("function heads") encode the task being
performed via in-context learning (ICL). Their outputs can be extracted and
injected into zero-shot prompts to transfer the task.

Variable notation follows the paper:
  - h         : attention head index
  - t         : token position index
  - i         : demonstration index
  - out_{h,t} : output of head h at position t  (R^d)
  - FV_h      : function vector for head h = mean_{i,t} out_{h,t,i}
  - FV_task   : global task function vector (mean over heads, or top-K weighted)

Pure PyTorch — no HuggingFace, no scipy, no sklearn, no einops.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# _split_heads_from_o_proj_input
# ---------------------------------------------------------------------------


def _split_heads(x: Tensor, n_heads: int) -> Tensor:
    """Split the last dimension of x into n_heads chunks.

    Args:
        x: (..., d_model)
        n_heads: number of heads

    Returns:
        (..., n_heads, head_dim) where head_dim = d_model // n_heads
    """
    *leading, d_model = x.shape
    head_dim = d_model // n_heads
    return x.reshape(*leading, n_heads, head_dim)


# ---------------------------------------------------------------------------
# FunctionVectorExtractor
# ---------------------------------------------------------------------------


class FunctionVectorExtractor:
    """Extract function vectors from attention head outputs at a given layer.

    The function vector for head h is defined as the mean of that head's
    output across all N demonstrations and all T token positions:

        FV_h = mean_{i,t} out_{h,t,i}   ∈ R^{head_dim}

    Paper §3 (Hendel et al., 2023).

    Args:
        model: An AureliusTransformer (or any nn.Module whose transformer
               blocks are stored in model.layers[layer].attn.o_proj).
        layer: Which transformer layer to extract from (0-indexed).
    """

    def __init__(self, model: nn.Module, layer: int = 0) -> None:
        self.model = model
        self.layer = layer

        # Infer n_heads from model config when available, else from o_proj weight
        if hasattr(model, "config"):
            self._n_heads: int = model.config.n_heads
        else:
            # Fallback: inspect the output projection
            attn = self._get_attn_module()
            o_proj = attn.o_proj
            # o_proj: (d_model, d_model) — weight shape
            d_model = o_proj.weight.shape[0]
            head_dim = attn.head_dim if hasattr(attn, "head_dim") else None
            if head_dim is not None:
                self._n_heads = d_model // head_dim
            else:
                raise ValueError(
                    "Cannot infer n_heads: model has no .config and attention "
                    "module has no .head_dim attribute."
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_attn_module(self) -> nn.Module:
        """Return the attention sub-module for self.layer."""
        layers = self._get_layers()
        return layers[self.layer].attn

    def _get_layers(self):
        """Return the list/ModuleList of transformer blocks."""
        if hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError("model has no .layers attribute; cannot locate transformer blocks.")

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    def extract(
        self,
        input_ids: Tensor,
        positions: list[int] | None = None,
    ) -> Tensor:
        """Extract and average attention head outputs across demonstrations.

        For each demonstration i and each token position t, captures
        out_{h,t,i} (the pre-o_proj, per-head attention output) and
        averages to produce FV_h.

        Args:
            input_ids: (N, T) — N demonstrations (each row is one prompt).
            positions:  Which token positions to include when averaging.
                        None = all positions.

        Returns:
            fv: (n_heads, head_dim) — function vector per head.
                To get the scalar global FV_task, call .mean(0) on the result.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (N, T), got shape {tuple(input_ids.shape)}")
        N, T = input_ids.shape

        # We capture the INPUT to the o_proj rather than its output so we can
        # split per head.  The input to o_proj has shape (B, S, n_heads*head_dim)
        # after the contiguous reshape inside GroupedQueryAttention.forward.
        captured: list[Tensor] = []

        def _hook(_module: nn.Module, _inp, _out) -> None:  # noqa: ANN001
            # _inp is a tuple; _inp[0] is the pre-projection tensor
            # shape: (B, S, n_heads * head_dim)
            captured.append(_inp[0].detach())

        attn = self._get_attn_module()
        handle = attn.o_proj.register_forward_hook(_hook)

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Hook did not capture any activation.")

        # captured[0]: (N, T, n_heads * head_dim)
        pre_proj: Tensor = captured[0]  # (N, T, n_heads * head_dim)

        # Validate
        if pre_proj.shape[0] != N or pre_proj.shape[1] != T:
            raise RuntimeError(
                f"Captured tensor shape {tuple(pre_proj.shape)} does not match "
                f"expected (N={N}, T={T}, ...)."
            )

        # Split into per-head outputs: (N, T, n_heads, head_dim)
        n_heads = self._n_heads
        out_h_t_i = _split_heads(pre_proj, n_heads)  # (N, T, n_heads, head_dim)

        if positions is not None:
            # Select only the requested positions along dim=1
            pos_idx = torch.tensor(positions, dtype=torch.long, device=out_h_t_i.device)
            out_h_t_i = out_h_t_i[:, pos_idx, :, :]  # (N, |positions|, n_heads, head_dim)

        # FV_h = mean_{i,t} out_{h,t,i}
        # Average over demonstration dim (0) and position dim (1)
        fv = out_h_t_i.mean(dim=(0, 1))  # (n_heads, head_dim)

        return fv

    # ------------------------------------------------------------------
    # compute_head_importance
    # ------------------------------------------------------------------

    def compute_head_importance(
        self,
        demonstrations: Tensor,
        zero_shot: Tensor,
        metric_fn: Callable[[Tensor], float],
    ) -> Tensor:
        """Measure the causal importance of each attention head.

        For each head h, the importance is:
            importance(h) = metric(model_with_FV_h_patched) - metric(baseline)

        The baseline is the zero-shot forward pass with no intervention.

        Args:
            demonstrations: (N, T) — few-shot demonstration prompts.
            zero_shot:       (1, T) — zero-shot prompt (no demonstrations).
            metric_fn:       (logits: Tensor) -> float.  Higher = better task
                             performance on the zero-shot prompt.

        Returns:
            importance: (n_heads,) — causal importance score per head.
        """
        if zero_shot.dim() != 2 or zero_shot.shape[0] != 1:
            raise ValueError(f"zero_shot must be (1, T), got shape {tuple(zero_shot.shape)}")

        # Extract FV per head: (n_heads, head_dim)
        fv = self.extract(demonstrations)

        # Baseline metric — zero-shot without any injection
        with torch.no_grad():
            _, baseline_logits, _ = self.model(zero_shot)
        baseline_score = metric_fn(baseline_logits)

        n_heads = fv.shape[0]
        head_dim = fv.shape[1]
        importance = torch.zeros(n_heads, dtype=fv.dtype, device=fv.device)

        injector = FunctionVectorInjector(self.model, layer=self.layer)

        for h in range(n_heads):
            # Expand FV_h into a full d_model vector by placing it in the
            # correct head slot; other heads are zeroed.
            fv_full = torch.zeros(
                self.model.config.d_model if hasattr(self.model, "config") else n_heads * head_dim,
                dtype=fv.dtype,
                device=fv.device,
            )
            start = h * head_dim
            fv_full[start : start + head_dim] = fv[h]

            patched_logits = injector.inject(zero_shot, fv_full, position=-1)
            patched_score = metric_fn(patched_logits)
            importance[h] = patched_score - baseline_score

        return importance


# ---------------------------------------------------------------------------
# FunctionVectorInjector
# ---------------------------------------------------------------------------


class FunctionVectorInjector:
    """Inject a function vector into the model's residual stream.

    The injection adds fv to the hidden state at the input of the target
    layer and at the specified token position, effectively steering the
    model's computation.

    Paper §4 (Hendel et al., 2023): "Inserting the function vector into
    the residual stream at a specific layer."

    Args:
        model: AureliusTransformer (or compatible).
        layer: Which layer's residual stream to inject into (0-indexed).
    """

    def __init__(self, model: nn.Module, layer: int = 0) -> None:
        self.model = model
        self.layer = layer

    # ------------------------------------------------------------------
    # inject
    # ------------------------------------------------------------------

    def inject(
        self,
        input_ids: Tensor,
        fv: Tensor,
        position: int = -1,
    ) -> Tensor:
        """Run the model with fv added to the residual stream.

        The fv is added to the hidden state tensor x at the *input* of
        self.layer (i.e., after the previous layer's residual has been
        applied but before this layer's attn_norm / attention).

        Args:
            input_ids: (1, T) — tokenised zero-shot prompt.
            fv:        (d_model,) — function vector to inject.
            position:  Token position at which to inject. Negative indices
                       are resolved relative to T (e.g. -1 = last token).

        Returns:
            logits: (1, T, vocab_size)
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (1, T), got shape {tuple(input_ids.shape)}")
        if fv.dim() != 1:
            raise ValueError(f"fv must be 1-D (d_model,), got shape {tuple(fv.shape)}")

        T = input_ids.shape[1]
        pos = position if position >= 0 else T + position  # normalise

        if not (0 <= pos < T):
            raise ValueError(f"position {position} is out of range for sequence length {T}.")

        # We hook the *forward pre-hook* of the target layer's TransformerBlock so
        # that we can modify the hidden state x before it enters the block.
        fv_device = fv.to(input_ids.device)

        def _pre_hook(_module: nn.Module, args):  # noqa: ANN001
            # args[0] is x: (B, T, d_model)
            x = args[0]
            x = x.clone()
            x[:, pos, :] = x[:, pos, :] + fv_device
            # Return modified args tuple
            return (x,) + args[1:]

        layers = self._get_layers()
        handle = layers[self.layer].register_forward_pre_hook(_pre_hook)

        try:
            with torch.no_grad():
                _, logits, _ = self.model(input_ids)
        finally:
            handle.remove()

        return logits

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_layers(self):
        if hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError("model has no .layers attribute; cannot locate transformer blocks.")


# ---------------------------------------------------------------------------
# Convenience: build_task_fv
# ---------------------------------------------------------------------------


def build_task_fv(
    model: nn.Module,
    demonstrations: Tensor,
    layer: int = 0,
    positions: list[int] | None = None,
    top_k: int | None = None,
) -> Tensor:
    """Compute the global task function vector FV_task.

    FV_task = mean_h FV_h   (or mean over top-K heads by L2 norm if top_k given)

    Args:
        model:         AureliusTransformer.
        demonstrations:(N, T) — few-shot demonstration prompts.
        layer:         Which layer to extract from.
        positions:     Token positions to average over (None = all).
        top_k:         If provided, average only the K heads with the largest
                       ||FV_h|| norms (a simple proxy for importance).

    Returns:
        fv_task: (d_model,) — the task function vector.
                 d_model = n_heads * head_dim
    """
    extractor = FunctionVectorExtractor(model, layer=layer)
    fv_per_head = extractor.extract(demonstrations, positions=positions)
    # fv_per_head: (n_heads, head_dim)

    if top_k is not None and top_k < fv_per_head.shape[0]:
        norms = fv_per_head.norm(dim=-1)  # (n_heads,)
        top_idx = torch.topk(norms, k=top_k).indices
        fv_per_head = fv_per_head[top_idx]

    # FV_task = mean_h FV_h, then flatten to (d_model,) compatible vector
    # We reconstruct the full d_model vector by stacking head slices.
    # mean over head dim → (head_dim,) only makes sense if we want a compact
    # vector; paper instead concatenates / places in the residual stream.
    # We return the flat concatenation of FV_h across heads = (n_heads*head_dim,)
    # = (d_model,), which is the natural "residual stream" representation.
    fv_task = fv_per_head.reshape(-1)  # (n_heads * head_dim,) = (d_model,)
    return fv_task
