"""Weight sharing / parameter tying for model compression."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SharingGroup:
    """A group of parameters that share a single set of centroid values."""

    group_id: str
    param_names: list[str]
    shared_value: list[float]


@dataclass(frozen=True)
class WeightSharingConfig:
    """Configuration for weight-sharing / k-means clustering."""

    num_clusters: int = 256
    bits: int = 8


class WeightSharing:
    """Cluster model weights into k shared centroids (Lloyd's k-means)."""

    def __init__(self, config: WeightSharingConfig | None = None) -> None:
        self.config = config or WeightSharingConfig()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def cluster_weights(self, weights: list[float], seed: int = 42) -> list[int]:
        """Assign each weight to its nearest centroid via Lloyd's k-means.

        Args:
            weights: flat list of weight values.
            seed: RNG seed used to initialise centroids.

        Returns:
            Integer cluster assignment for every weight in *weights*.
        """
        if not weights:
            return []

        k = min(self.config.num_clusters, len(weights))
        rng = random.Random(seed)  # noqa: S311

        # Initialise centroids by sampling k values (without replacement when
        # possible, otherwise with replacement).
        if k <= len(weights):
            centroids: list[float] = rng.sample(weights, k)
        else:
            centroids = [rng.choice(weights) for _ in range(k)]

        assignments: list[int] = [0] * len(weights)

        for _iteration in range(100):
            # --- Assignment step ---
            new_assignments: list[int] = []
            for w in weights:
                best_idx = 0
                best_dist = (w - centroids[0]) ** 2
                for c_idx in range(1, k):
                    dist = (w - centroids[c_idx]) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = c_idx
                new_assignments.append(best_idx)

            # --- Convergence check ---
            if new_assignments == assignments:
                assignments = new_assignments
                break
            assignments = new_assignments

            # --- Update step ---
            sums = [0.0] * k
            counts = [0] * k
            for w, a in zip(weights, assignments):
                sums[a] += w
                counts[a] += 1

            for c_idx in range(k):
                if counts[c_idx] > 0:
                    centroids[c_idx] = sums[c_idx] / counts[c_idx]
                # empty cluster: centroid unchanged (keeps previous position)

        return assignments

    def reconstruct(self, assignments: list[int], centroids: list[float]) -> list[float]:
        """Replace each assignment index with the corresponding centroid value."""
        return [centroids[a] for a in assignments]

    def compression_ratio(self, original_bits: int, weight_count: int) -> float:
        """Ratio of compressed storage to original storage.

        bits_needed = weight_count * config.bits / 8  (bytes)
        original    = original_bits                   (bits → converted internally)

        Returns bits_needed / original_bits so that a value < 1.0 means compression.
        """
        bits_needed = weight_count * self.config.bits / 8
        return bits_needed / original_bits

    def create_group(
        self,
        group_id: str,
        param_names: list[str],
        weights: list[float],
    ) -> SharingGroup:
        """Cluster *weights* and return a :class:`SharingGroup` whose
        ``shared_value`` contains the resulting centroids."""
        assignments = self.cluster_weights(weights)
        k = min(self.config.num_clusters, len(weights)) if weights else 0

        # Recompute centroids from final assignments
        sums = [0.0] * k
        counts = [0] * k
        for w, a in zip(weights, assignments):
            sums[a] += w
            counts[a] += 1

        centroids: list[float] = []
        for c_idx in range(k):
            if counts[c_idx] > 0:
                centroids.append(sums[c_idx] / counts[c_idx])
            else:
                centroids.append(0.0)

        return SharingGroup(
            group_id=group_id,
            param_names=param_names,
            shared_value=centroids,
        )


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

WEIGHT_SHARING_REGISTRY: dict[str, type[WeightSharing]] = {
    "default": WeightSharing,
}
