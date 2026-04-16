"""
src/interpretability/circuit_discovery.py

Circuit discovery for transformer models via activation patching.

Based on the Anthropic Circuits thread and Wang et al. 2022 (IOI paper).
Identifies which subset of attention heads and MLPs implement a specific
behavior by measuring each component's causal contribution.

Key idea — activation patching:
  1. Run model on "clean" input, cache all intermediate activations.
  2. Run model on "corrupted" input (same task, wrong answer).
  3. For each component, re-run the corrupted pass but *restore* that
     component's activation to its clean value.
  4. Measure how much the metric recovers:
       score = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)
     Score ≈ 1 → component is critical.  Score ≈ 0 → component is not important.

Pure PyTorch — no HuggingFace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ComponentScore
# ---------------------------------------------------------------------------

@dataclass
class ComponentScore:
    """Score for a single model component in a circuit-discovery experiment."""

    component_type: str   # 'attention_head' or 'mlp'
    layer: int
    head: int             # -1 for MLP components
    score: float          # normalised patching contribution
    description: str      # human-readable label


# ---------------------------------------------------------------------------
# CircuitConfig
# ---------------------------------------------------------------------------

@dataclass
class CircuitConfig:
    """Configuration knobs for a CircuitDiscoverer run."""

    threshold: float = 0.5
    n_patches: Optional[int] = None   # None = score every component
    normalize_scores: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return logits from *model* given *input_ids*.

    Handles the (loss, logits, kvs) tuple returned by AureliusTransformer as
    well as plain nn.Module that return a Tensor directly.
    """
    with torch.no_grad():
        output = model(input_ids)
    if isinstance(output, tuple):
        # AureliusTransformer returns (loss, logits, present_key_values)
        return output[1]
    return output


# ---------------------------------------------------------------------------
# CircuitDiscoverer
# ---------------------------------------------------------------------------

class CircuitDiscoverer:
    """Identifies circuit components via activation patching.

    Parameters
    ----------
    model      : The transformer model to analyse.  Must expose ``.layers``
                 as a ``nn.ModuleList`` of ``TransformerBlock``-like objects
                 each having ``.attn`` and ``.ffn`` sub-modules.
    metric_fn  : Callable that maps logits (Tensor) → float.  Higher values
                 should indicate *better* / more correct behaviour.
    """

    def __init__(
        self,
        model: nn.Module,
        metric_fn: Callable[[torch.Tensor], float],
    ) -> None:
        self.model = model
        self.metric_fn = metric_fn

        # Detect model shape from config if available, otherwise inspect layers
        config = getattr(model, "config", None)
        if config is not None:
            self.n_layers: int = config.n_layers
            self.n_heads: int = config.n_heads
        else:
            self.n_layers = len(model.layers)  # type: ignore[arg-type]
            # Infer n_heads from first layer's attention
            first_attn = model.layers[0].attn  # type: ignore[index]
            self.n_heads = getattr(first_attn, "n_heads", 1)

    # ------------------------------------------------------------------
    # get_clean_activations
    # ------------------------------------------------------------------

    def get_clean_activations(self, clean_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run a forward pass on *clean_ids* and cache intermediate activations.

        Cached keys:
          ``attn_layer_{i}_head_{j}``  — output slice for query head *j* at layer *i*,
                                          shape (B, S, head_dim).
          ``mlp_layer_{i}``            — FFN output at layer *i*, shape (B, S, d_model).

        Returns
        -------
        dict mapping component key → Tensor (detached, on the same device as the model).
        """
        activations: Dict[str, torch.Tensor] = {}
        hooks: list = []

        for layer_idx, layer in enumerate(self.model.layers):  # type: ignore[union-attr]
            # ---- attention hook ----------------------------------------
            def _make_attn_hook(li: int, n_heads: int, head_dim: int):
                def _hook(module, input, output):
                    # output is (attn_out, (k_cache, v_cache)) for GQA layers
                    attn_out = output[0] if isinstance(output, tuple) else output
                    # attn_out: (B, S, d_model)  —  split per head
                    B, S, d = attn_out.shape
                    hd = d // n_heads
                    per_head = attn_out.detach().view(B, S, n_heads, hd)
                    for h in range(n_heads):
                        activations[f"attn_layer_{li}_head_{h}"] = per_head[:, :, h, :]
                return _hook

            # ---- mlp hook -----------------------------------------------
            def _make_mlp_hook(li: int):
                def _hook(module, input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    activations[f"mlp_layer_{li}"] = out.detach()
                return _hook

            # Detect per-head dimension safely
            config = getattr(self.model, "config", None)
            head_dim = config.head_dim if config is not None else (
                self.model.config.d_model // self.n_heads  # type: ignore[union-attr]
                if hasattr(self.model, "config") else 64
            )

            hooks.append(layer.attn.register_forward_hook(
                _make_attn_hook(layer_idx, self.n_heads, head_dim)
            ))
            hooks.append(layer.ffn.register_forward_hook(
                _make_mlp_hook(layer_idx)
            ))

        try:
            with torch.no_grad():
                self.model(clean_ids)
        finally:
            for h in hooks:
                h.remove()

        return activations

    # ------------------------------------------------------------------
    # patch_activation
    # ------------------------------------------------------------------

    def patch_activation(
        self,
        corrupted_ids: torch.Tensor,
        patch_source: Dict[str, torch.Tensor],
        component: str,
    ) -> float:
        """Run the model on *corrupted_ids* with one component patched from *patch_source*.

        Parameters
        ----------
        corrupted_ids : Input token ids for the corrupted run.
        patch_source  : Dict produced by ``get_clean_activations``.
        component     : Key into *patch_source* identifying which component to patch.

        Returns
        -------
        metric_fn(patched_logits) as a float.
        """
        if component not in patch_source:
            raise KeyError(f"Component '{component}' not found in patch_source.")

        clean_act = patch_source[component]  # (B, S, dim)
        hook_handle = None

        is_attn = component.startswith("attn_layer_")
        is_mlp = component.startswith("mlp_layer_")

        if is_attn:
            # Parse layer and head indices
            parts = component.split("_")
            # Format: attn_layer_{i}_head_{j}
            layer_idx = int(parts[2])
            head_idx = int(parts[4])

            config = getattr(self.model, "config", None)
            head_dim = config.head_dim if config is not None else (
                clean_act.shape[-1]
            )
            n_heads = self.n_heads

            def _attn_patch_hook(module, input, output):
                attn_out = output[0] if isinstance(output, tuple) else output
                B, S, d = attn_out.shape
                hd = d // n_heads
                per_head = attn_out.clone().view(B, S, n_heads, hd)
                per_head[:, :, head_idx, :] = clean_act.to(attn_out.device)
                patched = per_head.view(B, S, d)
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            layer = list(self.model.layers)[layer_idx]  # type: ignore[union-attr]
            hook_handle = layer.attn.register_forward_hook(_attn_patch_hook)

        elif is_mlp:
            parts = component.split("_")
            # Format: mlp_layer_{i}
            layer_idx = int(parts[2])

            def _mlp_patch_hook(module, input, output):
                is_tuple = isinstance(output, tuple)
                out = output[0] if is_tuple else output
                patched = clean_act.to(out.device)
                if is_tuple:
                    return (patched,) + output[1:]
                return patched

            layer = list(self.model.layers)[layer_idx]  # type: ignore[union-attr]
            hook_handle = layer.ffn.register_forward_hook(_mlp_patch_hook)

        else:
            raise ValueError(f"Cannot parse component key: '{component}'")

        try:
            logits = _run_forward(self.model, corrupted_ids)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        return self.metric_fn(logits)

    # ------------------------------------------------------------------
    # compute_patching_scores
    # ------------------------------------------------------------------

    def compute_patching_scores(
        self,
        clean_ids: torch.Tensor,
        corrupted_ids: torch.Tensor,
    ) -> List[ComponentScore]:
        """Score every attention head and MLP by activation patching.

        For each component *c*:
            score(c) = (metric_patched(c) - metric_corrupted) /
                       (metric_clean - metric_corrupted + 1e-8)

        Returns a list of :class:`ComponentScore` sorted by score descending.
        """
        # Baseline metrics
        clean_logits = _run_forward(self.model, clean_ids)
        corrupted_logits = _run_forward(self.model, corrupted_ids)
        metric_clean = self.metric_fn(clean_logits)
        metric_corrupted = self.metric_fn(corrupted_logits)
        denom = metric_clean - metric_corrupted

        # Cache clean activations once
        clean_acts = self.get_clean_activations(clean_ids)

        scores: List[ComponentScore] = []

        for layer_idx in range(self.n_layers):
            # Attention heads
            for head_idx in range(self.n_heads):
                key = f"attn_layer_{layer_idx}_head_{head_idx}"
                metric_patched = self.patch_activation(corrupted_ids, clean_acts, key)
                raw_score = (metric_patched - metric_corrupted) / (denom + 1e-8)
                scores.append(ComponentScore(
                    component_type="attention_head",
                    layer=layer_idx,
                    head=head_idx,
                    score=float(raw_score),
                    description=f"Layer {layer_idx} Attention Head {head_idx}",
                ))

            # MLP
            key = f"mlp_layer_{layer_idx}"
            metric_patched = self.patch_activation(corrupted_ids, clean_acts, key)
            raw_score = (metric_patched - metric_corrupted) / (denom + 1e-8)
            scores.append(ComponentScore(
                component_type="mlp",
                layer=layer_idx,
                head=-1,
                score=float(raw_score),
                description=f"Layer {layer_idx} MLP",
            ))

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    # ------------------------------------------------------------------
    # discover_circuit
    # ------------------------------------------------------------------

    def discover_circuit(
        self,
        clean_ids: torch.Tensor,
        corrupted_ids: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[ComponentScore]:
        """Return only the components whose patching score >= *threshold*.

        Parameters
        ----------
        clean_ids      : Token ids for the "clean" (correct) input.
        corrupted_ids  : Token ids for the "corrupted" (wrong) input.
        threshold      : Minimum score to include in the circuit.

        Returns
        -------
        Sorted list of :class:`ComponentScore` with score >= threshold.
        """
        all_scores = self.compute_patching_scores(clean_ids, corrupted_ids)
        return [s for s in all_scores if s.score >= threshold]


# ---------------------------------------------------------------------------
# compute_indirect_effect  (module-level convenience)
# ---------------------------------------------------------------------------

def compute_indirect_effect(
    model: nn.Module,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    component_name: str,
    metric_fn: Callable[[torch.Tensor], float],
) -> float:
    """Compute the indirect causal effect of patching *component_name*.

    The indirect effect is the normalised change in the metric when that
    single component's activation is restored from the clean run:

        IE = (metric_patched - metric_corrupted) / (metric_clean - metric_corrupted + 1e-8)

    Parameters
    ----------
    model          : Transformer model.
    clean_ids      : Token ids for the clean input.
    corrupted_ids  : Token ids for the corrupted input.
    component_name : Component key, e.g. ``'attn_layer_0_head_1'``.
    metric_fn      : Callable(logits) → float.

    Returns
    -------
    Indirect effect as a float.
    """
    discoverer = CircuitDiscoverer(model, metric_fn)
    clean_acts = discoverer.get_clean_activations(clean_ids)

    corrupted_logits = _run_forward(model, corrupted_ids)
    clean_logits = _run_forward(model, clean_ids)

    metric_corrupted = metric_fn(corrupted_logits)
    metric_clean = metric_fn(clean_logits)
    metric_patched = discoverer.patch_activation(corrupted_ids, clean_acts, component_name)

    denom = metric_clean - metric_corrupted
    return (metric_patched - metric_corrupted) / (denom + 1e-8)


# ---------------------------------------------------------------------------
# visualize_circuit
# ---------------------------------------------------------------------------

def visualize_circuit(
    scores: List[ComponentScore],
    top_k: int = 10,
) -> dict:
    """Produce a structured summary of circuit components.

    Parameters
    ----------
    scores : List of :class:`ComponentScore` (typically from
             :meth:`CircuitDiscoverer.discover_circuit`).
    top_k  : How many top components to list under ``'top_components'``.

    Returns
    -------
    dict with keys:
      ``'by_layer'``       : ``{layer_idx: [ComponentScore, ...]}``
      ``'top_components'`` : list of up to *top_k* ComponentScore, sorted by
                             score descending.
    """
    by_layer: Dict[int, List[ComponentScore]] = {}
    for cs in scores:
        by_layer.setdefault(cs.layer, []).append(cs)

    # Sort each layer's components by score descending
    for layer_idx in by_layer:
        by_layer[layer_idx].sort(key=lambda s: s.score, reverse=True)

    # Global top-k (already sorted if caller passed sorted list, but be safe)
    sorted_all = sorted(scores, key=lambda s: s.score, reverse=True)
    top_components = sorted_all[:top_k]

    return {
        "by_layer": by_layer,
        "top_components": top_components,
    }
