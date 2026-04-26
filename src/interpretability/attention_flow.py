"""
attention_flow.py — Analyzes attention flow across transformer layers.

Pure Python, stdlib-only. No torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionFlow:
    """A single directed attention flow from one position to another."""

    layer: int
    head: int
    from_pos: int
    to_pos: int
    weight: float


class AttentionFlowAnalyzer:
    """Records and analyzes attention flows across transformer layers and heads."""

    def __init__(self, num_layers: int, num_heads: int) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        # All recorded flows stored as a flat list
        self._flows: list[AttentionFlow] = []
        # Per (layer, head) -> list of AttentionFlow
        self._layer_head_flows: dict[tuple, list[AttentionFlow]] = {}

    def record(
        self,
        layer: int,
        head: int,
        attention_weights: list[list[float]],
    ) -> list[AttentionFlow]:
        """Record attention weights for one layer/head.

        For each (from_pos, to_pos) pair, creates an AttentionFlow if weight > 0.0.

        Args:
            layer: Layer index.
            head: Head index.
            attention_weights: 2-D list [from_pos][to_pos] of floats.

        Returns:
            List of AttentionFlow objects created for this layer/head.
        """
        flows: list[AttentionFlow] = []
        for from_pos, row in enumerate(attention_weights):
            for to_pos, weight in enumerate(row):
                if weight > 0.0:
                    flow = AttentionFlow(
                        layer=layer,
                        head=head,
                        from_pos=from_pos,
                        to_pos=to_pos,
                        weight=weight,
                    )
                    flows.append(flow)

        self._flows.extend(flows)
        key = (layer, head)
        if key not in self._layer_head_flows:
            self._layer_head_flows[key] = []
        self._layer_head_flows[key].extend(flows)
        return flows

    def top_flows(self, n: int = 10) -> list[AttentionFlow]:
        """Return the globally top-n AttentionFlow objects by weight descending.

        If n > number of recorded flows, returns all flows.

        Args:
            n: Number of top flows to return.

        Returns:
            Sorted list (descending by weight) of up to n AttentionFlow objects.
        """
        sorted_flows = sorted(self._flows, key=lambda f: f.weight, reverse=True)
        return sorted_flows[:n]

    def layer_summary(self, layer: int) -> dict:
        """Summarize all recorded flows for a given layer (across all heads).

        Args:
            layer: Layer index.

        Returns:
            Dict with keys: "layer", "total_flows", "mean_weight", "max_weight".
        """
        layer_flows = [f for f in self._flows if f.layer == layer]
        total = len(layer_flows)
        if total == 0:
            return {
                "layer": layer,
                "total_flows": 0,
                "mean_weight": 0.0,
                "max_weight": 0.0,
            }
        weights = [f.weight for f in layer_flows]
        mean_w = sum(weights) / total
        max_w = max(weights)
        return {
            "layer": layer,
            "total_flows": total,
            "mean_weight": mean_w,
            "max_weight": max_w,
        }

    def head_importance(self, layer: int) -> list[float]:
        """Return mean attention weight per head for a given layer.

        Args:
            layer: Layer index.

        Returns:
            List of length num_heads. Each entry is the mean weight for that
            head at the given layer, or 0.0 if no flows were recorded for it.
        """
        result = [0.0] * self.num_heads
        for head in range(self.num_heads):
            key = (layer, head)
            flows = self._layer_head_flows.get(key, [])
            if flows:
                result[head] = sum(f.weight for f in flows) / len(flows)
        return result

    def reset(self) -> None:
        """Clear all recorded flows."""
        self._flows = []
        self._layer_head_flows = {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ATTENTION_FLOW_REGISTRY = {
    "default": AttentionFlowAnalyzer,
}
