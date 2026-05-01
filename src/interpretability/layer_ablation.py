"""
src/interpretability/layer_ablation.py

Layer ablation studies for AureliusTransformer.

Systematically removes or zeroes out individual transformer layers/components
to understand their contribution to model behaviour. Supports zero ablation,
mean ablation, noise ablation, and parameter-freeze ablation.

Pure PyTorch — no HuggingFace.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# AblationType
# ---------------------------------------------------------------------------


class AblationType(StrEnum):
    """Strategy used to ablate a component's activations or parameters."""

    ZERO = "zero"  # replace activations with zeros
    MEAN = "mean"  # replace activations with a dataset mean tensor
    NOISE = "noise"  # add Gaussian noise to activations
    FREEZE = "freeze"  # freeze parameters (no gradient update)


# ---------------------------------------------------------------------------
# AblationResult
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Stores the outcome of ablating a single model component."""

    component: str  # e.g. "layer_0", "layer_1_attn_head_3"
    ablation_type: str
    metric_before: float
    metric_after: float
    delta: float  # metric_after - metric_before (negative = component was helpful)
    relative_impact: float  # |delta| / |metric_before|


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """Hyper-parameters for an ablation experiment."""

    ablation_type: str = AblationType.ZERO
    noise_std: float = 0.1
    n_seeds: int = 3


# ---------------------------------------------------------------------------
# LayerAblator
# ---------------------------------------------------------------------------


class LayerAblator:
    """Run layer-level ablation studies on an AureliusTransformer-like model.

    Parameters
    ----------
    model:
        An ``nn.Module`` with a ``layers`` attribute that is an
        ``nn.ModuleList`` of ``TransformerBlock``-like modules.  Each block
        must be callable as ``block(x, freqs_cis, mask, past_kv) -> (x, kv)``.
    metric_fn:
        A callable ``(logits: Tensor) -> float`` where *higher* is better
        (e.g. negative cross-entropy, accuracy, or log-probability of a
        target token).
    """

    def __init__(self, model: nn.Module, metric_fn: Callable) -> None:
        self.model = model
        self.metric_fn = metric_fn

    # ------------------------------------------------------------------
    # Internal: run a full forward pass, optionally skipping one layer
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward(
        self,
        input_ids: torch.Tensor,
        ablate_layer_idx: int | None = None,
        ablation_type: AblationType = AblationType.ZERO,
        mean_activation: torch.Tensor | None = None,
        noise_std: float = 0.1,
        ablate_head_idx: int | None = None,
    ) -> torch.Tensor:
        """Run forward pass with an optional layer-output ablation.

        Returns
        -------
        logits: Tensor of shape (batch, seq_len, vocab_size)
        """
        model = self.model
        B, S = input_ids.shape

        # Determine position offset (no KV cache here)
        past_len = 0
        x = model.embed(input_ids)
        freqs_cis = model.freqs_cis[past_len : past_len + S]

        for i, layer in enumerate(model.layers):
            x, _kv, _aux = layer(x, freqs_cis, None, None)

            if i == ablate_layer_idx:
                if ablation_type == AblationType.ZERO:
                    x = torch.zeros_like(x)

                elif ablation_type == AblationType.MEAN:
                    if mean_activation is not None:
                        x = mean_activation.expand_as(x).clone()
                    else:
                        # Fall back: use the mean over the current batch/seq
                        x = x.mean(dim=(0, 1), keepdim=True).expand_as(x).clone()

                elif ablation_type == AblationType.NOISE:
                    x = x + torch.randn_like(x) * noise_std

                elif ablation_type == AblationType.FREEZE:
                    # FREEZE means we keep activations but stop gradients.
                    # At inference time (no_grad) this is a no-op on activations;
                    # the semantic is that parameters are frozen during training.
                    pass

        x = model.norm(x)
        logits = model.lm_head(x)
        return logits

    # ------------------------------------------------------------------
    # Internal: run forward pass ablating a specific attention head
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_ablate_head(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        head_idx: int | None,
        ablation_type: AblationType = AblationType.ZERO,
        noise_std: float = 0.1,
    ) -> torch.Tensor:
        """Forward pass zeroing out a specific attention head's contribution.

        For the target layer we hook into the attention output (``o_proj``
        input) and zero/noise the columns that correspond to *head_idx*.
        If *head_idx* is None, all heads are ablated (equivalent to
        ablating the whole attention sub-layer).
        """
        model = self.model
        B, S = input_ids.shape

        hooks = []
        target_layer = model.layers[layer_idx]
        head_dim = model.config.head_dim

        # We hook into the *input* of o_proj to zero the relevant slice.
        def _make_hook(h_idx: int | None):
            def _hook(module: nn.Module, inp, out):  # noqa: ANN001
                # inp[0]: (B, S, n_heads * head_dim)
                tensor = inp[0].clone()
                if h_idx is None:
                    if ablation_type == AblationType.ZERO:
                        tensor = torch.zeros_like(tensor)
                    elif ablation_type == AblationType.NOISE:
                        tensor = tensor + torch.randn_like(tensor) * noise_std
                else:
                    start = h_idx * head_dim
                    end = start + head_dim
                    if ablation_type == AblationType.ZERO:
                        tensor[..., start:end] = 0.0
                    elif ablation_type == AblationType.NOISE:
                        tensor[..., start:end] += (
                            torch.randn_like(tensor[..., start:end]) * noise_std
                        )
                    elif ablation_type == AblationType.MEAN:
                        mean_val = tensor[..., start:end].mean()
                        tensor[..., start:end] = mean_val
                # Re-run the projection with the modified input
                return module(tensor)

            return _hook

        h = target_layer.attn.o_proj.register_forward_hook(_make_hook(head_idx))
        hooks.append(h)

        try:
            x = model.embed(input_ids)
            freqs_cis = model.freqs_cis[:S]
            for i, layer in enumerate(model.layers):
                x, _kv, _aux = layer(x, freqs_cis, None, None)
            x = model.norm(x)
            logits = model.lm_head(x)
        finally:
            for hk in hooks:
                hk.remove()

        return logits

    # ------------------------------------------------------------------
    # ablate_layer — context manager
    # ------------------------------------------------------------------

    @contextmanager
    def ablate_layer(
        self,
        layer_idx: int,
        ablation_type: AblationType,
        input_ids: torch.Tensor,
        mean_activation: torch.Tensor | None = None,
    ) -> Generator[torch.Tensor, None, None]:
        """Context manager that temporarily patches the output of *layer_idx*.

        Yields the modified logits tensor of shape ``(batch, seq, vocab)``.

        Usage::

            with ablator.ablate_layer(0, AblationType.ZERO, ids) as logits:
                score = metric_fn(logits)
        """
        logits = self._forward(
            input_ids,
            ablate_layer_idx=layer_idx,
            ablation_type=ablation_type,
            mean_activation=mean_activation,
        )
        yield logits

    # ------------------------------------------------------------------
    # ablate_attention
    # ------------------------------------------------------------------

    def ablate_attention(
        self,
        layer_idx: int,
        head_idx: int | None,
        ablation_type: AblationType,
        input_ids: torch.Tensor,
    ) -> float:
        """Ablate attention head(s) in *layer_idx* and return the metric score.

        Parameters
        ----------
        layer_idx:
            Which transformer layer to target.
        head_idx:
            The query-head index to ablate, or ``None`` to ablate all heads.
        ablation_type:
            How to ablate (ZERO / MEAN / NOISE).
        input_ids:
            Token ids of shape ``(batch, seq_len)``.

        Returns
        -------
        float — metric_fn(logits) after ablation.
        """
        logits = self._forward_ablate_head(
            input_ids,
            layer_idx=layer_idx,
            head_idx=head_idx,
            ablation_type=ablation_type,
        )
        return self.metric_fn(logits)

    # ------------------------------------------------------------------
    # run_full_ablation
    # ------------------------------------------------------------------

    def run_full_ablation(
        self,
        input_ids: torch.Tensor,
        ablation_type: AblationType = AblationType.ZERO,
    ) -> list[AblationResult]:
        """Ablate every layer in turn and record the metric impact.

        Parameters
        ----------
        input_ids:
            Token ids of shape ``(batch, seq_len)``.
        ablation_type:
            Strategy to use for all layers.

        Returns
        -------
        List of :class:`AblationResult`, one per layer, sorted by
        ``relative_impact`` descending (most important first).
        """
        # Baseline metric (no ablation)
        baseline_logits = self._forward(input_ids)
        metric_before = self.metric_fn(baseline_logits)

        results: list[AblationResult] = []
        n_layers = len(self.model.layers)

        for i in range(n_layers):
            ablated_logits = self._forward(
                input_ids,
                ablate_layer_idx=i,
                ablation_type=ablation_type,
            )
            metric_after = self.metric_fn(ablated_logits)
            delta = metric_after - metric_before
            denom = abs(metric_before) if abs(metric_before) > 1e-12 else 1e-12
            relative_impact = abs(delta) / denom

            results.append(
                AblationResult(
                    component=f"layer_{i}",
                    ablation_type=str(ablation_type),
                    metric_before=metric_before,
                    metric_after=metric_after,
                    delta=delta,
                    relative_impact=relative_impact,
                )
            )

        # Sort: most impactful first
        results.sort(key=lambda r: r.relative_impact, reverse=True)
        return results

    # ------------------------------------------------------------------
    # compute_layer_importance
    # ------------------------------------------------------------------

    def compute_layer_importance(
        self,
        results: list[AblationResult],
    ) -> torch.Tensor:
        """Derive a per-layer importance tensor from ablation results.

        Parameters
        ----------
        results:
            Output of :meth:`run_full_ablation`.

        Returns
        -------
        Tensor of shape ``(n_layers,)`` where index ``i`` corresponds to
        ``layer_i``'s importance (= ``relative_impact``).  Higher means
        more important.
        """
        # Re-order by layer index so the output tensor is layer-ordered
        layer_results: dict[int, float] = {}
        for r in results:
            # component name: "layer_{i}"
            try:
                idx = int(r.component.split("_")[1])
            except (IndexError, ValueError):
                continue
            layer_results[idx] = r.relative_impact

        n_layers = len(layer_results)
        importance = torch.zeros(n_layers)
        for idx, val in layer_results.items():
            if idx < n_layers:
                importance[idx] = val
        return importance

    # ------------------------------------------------------------------
    # get_redundant_layers
    # ------------------------------------------------------------------

    def get_redundant_layers(
        self,
        results: list[AblationResult],
        threshold: float = 0.05,
    ) -> list[int]:
        """Return layer indices whose relative impact is below *threshold*.

        These layers have minimal effect when ablated and may be safely
        removed or pruned.

        Parameters
        ----------
        results:
            Output of :meth:`run_full_ablation`.
        threshold:
            Layers with ``relative_impact < threshold`` are considered redundant.

        Returns
        -------
        Sorted list of layer indices.
        """
        redundant: list[int] = []
        for r in results:
            if r.relative_impact < threshold:
                try:
                    idx = int(r.component.split("_")[1])
                    redundant.append(idx)
                except (IndexError, ValueError):
                    pass
        redundant.sort()
        return redundant
