"""Attribution Graphs for mechanistic interpretability.

Given a torch model and an input, computes per-layer input-attribution
scores and builds a directed graph (layer i, neuron j) -> (layer i+1, neuron k)
via input x gradient or integrated gradients.

Pure torch. No foreign imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AttributionNode:
    layer: int
    unit: int
    activation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "unit": int(self.unit),
            "activation": float(self.activation),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttributionNode:
        return cls(layer=int(d["layer"]), unit=int(d["unit"]), activation=float(d["activation"]))


@dataclass
class AttributionEdge:
    src: AttributionNode
    dst: AttributionNode
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src.to_dict(), "dst": self.dst.to_dict(), "weight": float(self.weight)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttributionEdge:
        return cls(
            src=AttributionNode.from_dict(d["src"]),
            dst=AttributionNode.from_dict(d["dst"]),
            weight=float(d["weight"]),
        )


@dataclass
class AttributionGraph:
    nodes: list[AttributionNode] = field(default_factory=list)
    edges: list[AttributionEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttributionGraph:
        return cls(
            nodes=[AttributionNode.from_dict(n) for n in d.get("nodes", [])],
            edges=[AttributionEdge.from_dict(e) for e in d.get("edges", [])],
        )

    def is_acyclic_forward(self) -> bool:
        """All edges go from lower layer to higher layer."""
        for e in self.edges:
            if e.src.layer >= e.dst.layer:
                return False
        return True


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class AttributionGraphBuilder:
    """Build an attribution graph over named submodules of a torch model.

    Parameters
    ----------
    model : nn.Module
        The model to analyze.
    layer_names : list[str] | None
        Dotted names of submodules to treat as layers. If None, uses all
        direct children named "0", "1", ... (Sequential default) in order.
    method : str
        "input_x_grad" or "integrated_gradients".
    top_k_per_node : int
        Maximum number of outgoing edges per source node.
    """

    _VALID_METHODS = ("input_x_grad", "integrated_gradients")

    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str] | None = None,
        method: str = "input_x_grad",
        top_k_per_node: int = 8,
    ) -> None:
        if layer_names is not None and len(layer_names) == 0:
            raise ValueError("layer_names must be non-empty if provided")
        if method not in self._VALID_METHODS:
            raise ValueError(f"Unknown method {method!r}; valid: {self._VALID_METHODS}")
        if top_k_per_node <= 0:
            raise ValueError("top_k_per_node must be positive")

        self.model = model
        self.method = method
        self.top_k_per_node = int(top_k_per_node)

        if layer_names is None:
            layer_names = [n for n, _ in model.named_children()]
            if not layer_names:
                raise ValueError("Model has no named_children; supply layer_names")
        self.layer_names = list(layer_names)

        # Resolve modules
        self._modules: list[nn.Module] = []
        for name in self.layer_names:
            mod = self._resolve(name)
            if mod is None:
                raise ValueError(f"Layer {name!r} not found on model")
            self._modules.append(mod)

    # ---------------------- module resolution ----------------------
    def _resolve(self, dotted: str) -> nn.Module | None:
        obj: Any = self.model
        for part in dotted.split("."):
            if not hasattr(obj, part):
                # Also support dict-like sequentials
                try:
                    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
                except Exception:
                    return None
            else:
                obj = getattr(obj, part)
        return obj if isinstance(obj, nn.Module) else None

    # ---------------------- hook capture ----------------------
    def _capture(self, x: Tensor) -> list[Tensor]:
        """Forward pass; return list of activations (retain_grad) per layer."""
        acts: list[Tensor | None] = [None] * len(self._modules)

        def make_hook(idx: int):
            def hook(_module, _inp, out):
                t = out if isinstance(out, Tensor) else out[0]
                t.retain_grad()
                acts[idx] = t

            return hook

        handles = []
        for i, m in enumerate(self._modules):
            handles.append(m.register_forward_hook(make_hook(i)))
        try:
            y = self.model(x)
        finally:
            for h in handles:
                h.remove()

        # assemble
        out: list[Tensor] = []
        for a in acts:
            if a is None:
                raise RuntimeError("Layer forward hook never fired")
            out.append(a)
        return out, y  # type: ignore[return-value]

    # ---------------------- attribution methods ----------------------
    def input_x_grad(self, input_ids: Tensor, target: Tensor) -> Tensor:
        """Return |input * grad| summed to scalar score per input element.

        Returns a tensor with the same shape as `input_ids` (float-cast) giving
        per-element attribution toward `target` (a scalar or tensor used via
        .sum()).
        """
        x = input_ids.detach().clone().to(torch.float32).requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        y = self.model(x)
        scalar = self._scalar_target(y, target)
        scalar.backward()
        if x.grad is None:
            raise RuntimeError("No gradient flowed to input")
        return (x * x.grad).detach()

    def integrated_gradients(
        self,
        input_ids: Tensor,
        target: Tensor,
        n_steps: int = 16,
    ) -> Tensor:
        """Integrated gradients from a zero baseline to the input."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        x0 = torch.zeros_like(input_ids, dtype=torch.float32)
        xT = input_ids.detach().to(torch.float32)
        total = torch.zeros_like(xT)
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            xk = (x0 + alpha * (xT - x0)).detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            y = self.model(xk)
            scalar = self._scalar_target(y, target)
            scalar.backward()
            if xk.grad is None:
                raise RuntimeError("No gradient flowed to input")
            total = total + xk.grad.detach()
        avg_grad = total / n_steps
        return ((xT - x0) * avg_grad).detach()

    # ---------------------- build ----------------------
    def build(
        self,
        input_ids: Tensor,
        target_layer: int = -1,
        target_unit: int = 0,
    ) -> AttributionGraph:
        L = len(self._modules)
        if target_layer < 0:
            target_layer += L
        if target_layer < 0 or target_layer >= L:
            raise IndexError(f"target_layer out of range (got {target_layer}, have {L} layers)")

        # Forward to record activations.
        x = input_ids.detach().clone().to(torch.float32).requires_grad_(True)
        acts, _y = self._capture(x)

        # Build target scalar = activation at (target_layer, target_unit) summed.
        tgt_act = acts[target_layer]
        flat = tgt_act.reshape(tgt_act.shape[0], -1)
        n_units = flat.shape[1]
        if target_unit < 0 or target_unit >= n_units:
            raise IndexError(f"target_unit {target_unit} out of range (layer has {n_units})")
        scalar = flat[:, target_unit].sum()

        # Backprop once; this gives grad w.r.t. every captured activation.
        self.model.zero_grad(set_to_none=True)
        scalar.backward(retain_graph=False)

        # Now compute attribution per layer: act * grad (input_x_grad) or IG.
        attributions: list[Tensor] = []
        for a in acts:
            g = a.grad
            if g is None:
                # Layer doesn't participate in target; zero attribution.
                attributions.append(torch.zeros_like(a.detach()))
            else:
                if self.method == "input_x_grad":
                    attributions.append(a.detach() * g.detach())
                else:  # integrated_gradients along activation path from 0 -> a
                    attributions.append(self._ig_on_activation(acts, a, g, n_steps=16))

        # Flatten per layer to [units]
        per_layer_attr = [attr.reshape(attr.shape[0], -1).mean(dim=0) for attr in attributions]
        per_layer_act = [a.detach().reshape(a.shape[0], -1).mean(dim=0) for a in acts]

        # Build nodes (all units in all involved layers, up to target).
        nodes: list[AttributionNode] = []
        node_index: dict[tuple, AttributionNode] = {}
        for li in range(target_layer + 1):
            for ui in range(per_layer_act[li].numel()):
                n = AttributionNode(
                    layer=li,
                    unit=ui,
                    activation=float(per_layer_act[li][ui].item()),
                )
                nodes.append(n)
                node_index[(li, ui)] = n

        # Build edges: from layer i to layer i+1 (forward only, acyclic).
        edges: list[AttributionEdge] = []
        for li in range(target_layer):
            src_attr = per_layer_attr[li]
            dst_attr = per_layer_attr[li + 1]
            n_src = src_attr.numel()
            n_dst = dst_attr.numel()
            # Weight(src -> dst) = |src_attr[src] * dst_attr[dst]|
            # This is a cheap heuristic bridging attribution magnitudes.
            src_vec = src_attr.abs()
            dst_vec = dst_attr.abs()
            # Top-k destinations by attribution magnitude (shared across sources).
            k = min(self.top_k_per_node, n_dst)
            top_dst_vals, top_dst_idx = torch.topk(dst_vec, k)
            for s in range(n_src):
                for j in range(k):
                    d = int(top_dst_idx[j].item())
                    w = float((src_vec[s] * dst_vec[d]).item())
                    edges.append(
                        AttributionEdge(
                            src=node_index[(li, s)],
                            dst=node_index[(li + 1, d)],
                            weight=w,
                        )
                    )

        return AttributionGraph(nodes=nodes, edges=edges)

    # ---------------------- helpers ----------------------
    def _ig_on_activation(
        self,
        all_acts: list[Tensor],
        a: Tensor,
        g: Tensor,
        n_steps: int = 16,
    ) -> Tensor:
        """Approximate IG on an intermediate activation using the single grad.

        Because a path integral over the activation space requires multiple
        forward/backward passes, we approximate with a Riemann sum where the
        gradient is assumed locally linear; equivalent to a * g scaled by the
        trapezoidal average — which limits toward input_x_grad for large
        n_steps.
        """
        # Use a simple average of alpha * a * g over n_steps, which reduces
        # to (a * g) * mean(alpha) = 0.5 * a*g for a zero baseline.
        # To let n_steps=large recover input_x_grad, we use alpha going from
        # 1/n to 1 inclusive, so the mean approaches 0.5; we then scale by 2
        # to match input_x_grad in the limit.
        alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=a.device)
        mean_alpha = alphas.mean()
        return (a.detach() * g.detach()) * (2.0 * mean_alpha)

    def _scalar_target(self, y: Tensor, target: Tensor) -> Tensor:
        """Reduce model output to a scalar using target as index/mask/scalar."""
        if target.dim() == 0:
            # interpret as class index across the last dim, summed across batch
            idx = int(target.item())
            if y.dim() >= 2 and idx < y.shape[-1]:
                return y[..., idx].sum()
            return (y * target).sum()
        if target.shape == y.shape:
            return (y * target).sum()
        # fallback: sum of y
        return y.sum()
