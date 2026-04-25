"""Feature visualization: activation patching, causal tracing, feature decomposition."""

from dataclasses import dataclass
from enum import Enum


class PatchTarget(str, Enum):
    RESIDUAL_STREAM = "residual_stream"
    ATTENTION_OUTPUT = "attention_output"
    MLP_OUTPUT = "mlp_output"


@dataclass
class PatchResult:
    target: PatchTarget
    layer: int
    position: int
    effect: float
    baseline_output: float
    patched_output: float


class ActivationPatcher:
    def __init__(self) -> None:
        pass

    def compute_effect(self, baseline: float, patched: float) -> float:
        return abs(patched - baseline) / (abs(baseline) + 1e-8)

    def patch(
        self,
        original_activations: list[float],
        patch_values: list[float],
        positions: list[int],
    ) -> list[float]:
        result = list(original_activations)
        for i, pos in enumerate(positions):
            result[pos] = patch_values[i]
        return result

    def causal_trace(
        self,
        target: PatchTarget,
        layer: int,
        position: int,
        baseline_output: float,
        patched_output: float,
    ) -> PatchResult:
        effect = self.compute_effect(baseline_output, patched_output)
        return PatchResult(
            target=target,
            layer=layer,
            position=position,
            effect=effect,
            baseline_output=baseline_output,
            patched_output=patched_output,
        )

    def top_effects(self, results: list[PatchResult], k: int = 5) -> list[PatchResult]:
        return sorted(results, key=lambda r: r.effect, reverse=True)[:k]


class FeatureDecomposer:
    def __init__(self, n_components: int = 10) -> None:
        self.n_components = n_components

    def pca_stub(self, activations: list[list[float]]) -> list[list[float]]:
        if not activations:
            return []
        n_samples = len(activations)
        n_feats = len(activations[0])
        # compute per-feature mean
        means = [
            sum(activations[i][j] for i in range(n_samples)) / n_samples
            for j in range(n_feats)
        ]
        # subtract mean, then take first n_components features as stub
        n_out = min(self.n_components, n_feats)
        result = []
        for sample in activations:
            centered = [sample[j] - means[j] for j in range(n_feats)]
            result.append(centered[:n_out])
        return result

    def reconstruction_error(
        self,
        original: list[list[float]],
        reconstructed: list[list[float]],
    ) -> float:
        total = 0.0
        count = 0
        for orig_row, rec_row in zip(original, reconstructed):
            for o, r in zip(orig_row, rec_row):
                diff = o - r
                total += diff * diff
                count += 1
        return total / count if count else 0.0
