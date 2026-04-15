"""
Sparse Feature Circuits for mechanistic interpretability.

Inspired by "Towards Monosemanticity" (Bricken et al.) and
"Sparse Feature Circuits" (Marks et al.). Identifies minimal circuits
of features responsible for specific model behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SparseFeature:
    feature_id: int
    activation_mean: float
    activation_freq: float    # fraction of inputs that activate this feature
    max_activation: float
    top_tokens: List[int]     # token ids that most activate this feature


@dataclass
class Circuit:
    feature_ids: List[int]
    layer_name: str
    faithfulness_score: float   # how well circuit explains the output
    completeness_score: float   # how much of the effect this circuit captures
    minimality_score: float     # how minimal/sparse the circuit is


# ---------------------------------------------------------------------------
# SparseAutoencoder
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    Single-layer SAE: h = ReLU(W_enc @ x + b_enc); x_hat = W_dec @ h + b_dec.
    Trained with L1 sparsity penalty on h.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,      # typically >> input_dim (overcomplete)
        l1_coef: float = 1e-3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.l1_coef = l1_coef

        # Encoder: input_dim -> n_features
        self.W_enc = nn.Parameter(torch.empty(input_dim, n_features))
        self.b_enc = nn.Parameter(torch.full((n_features,), -1.0))

        # Decoder: n_features -> input_dim
        self.W_dec = nn.Parameter(torch.empty(n_features, input_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) -> features: (B, n_features), sparse via ReLU"""
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, n_features) -> reconstruction: (B, d)"""
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, features, loss) where loss = MSE + l1_coef * L1(features)"""
        features = self.encode(x)
        reconstruction = self.decode(features)
        mse_loss = F.mse_loss(reconstruction, x)
        l1_loss = features.abs().mean()
        loss = mse_loss + self.l1_coef * l1_loss
        return reconstruction, features, loss

    def get_active_features(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Returns (B, n_features) bool mask of active features."""
        features = self.encode(x)
        return features > threshold


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_sae_step(
    sae: SparseAutoencoder,
    activations: torch.Tensor,  # (B, d)
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """One training step. Returns {'loss': ..., 'reconstruction': ..., 'sparsity': ...}"""
    sae.train()
    optimizer.zero_grad()
    reconstruction, features, loss = sae(activations)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        mse = F.mse_loss(reconstruction.detach(), activations).item()
        sparsity = (features.detach() > 0).float().mean().item()

    return {
        "loss": loss.item(),
        "reconstruction": mse,
        "sparsity": sparsity,
    }


# ---------------------------------------------------------------------------
# FeatureAnalyzer
# ---------------------------------------------------------------------------

class FeatureAnalyzer:
    """Analyze which features activate for given inputs."""

    def __init__(self, sae: SparseAutoencoder) -> None:
        self.sae = sae

    def analyze_batch(
        self,
        activations: torch.Tensor,   # (N, d)
        token_ids: Optional[torch.Tensor] = None,  # (N,) — which token produced each activation
    ) -> List[SparseFeature]:
        """
        Compute statistics for each feature across N samples.
        Returns list of SparseFeature for features that activate at least once.
        """
        self.sae.eval()
        with torch.no_grad():
            features = self.sae.encode(activations)  # (N, n_features)

        N = activations.shape[0]
        n_features = features.shape[1]

        activation_mean = features.mean(dim=0)                    # (n_features,)
        activation_freq = (features > 0).float().mean(dim=0)      # (n_features,)
        max_activation = features.max(dim=0).values               # (n_features,)

        sparse_features: List[SparseFeature] = []
        for fid in range(n_features):
            if max_activation[fid].item() <= 0.0:
                continue  # never activated

            feat_vals = features[:, fid]  # (N,)
            top_k = min(5, N)
            if token_ids is not None:
                topk_indices = feat_vals.topk(top_k).indices.tolist()
                top_toks = [int(token_ids[i].item()) for i in topk_indices]
            else:
                top_toks = feat_vals.topk(top_k).indices.tolist()

            sparse_features.append(SparseFeature(
                feature_id=fid,
                activation_mean=activation_mean[fid].item(),
                activation_freq=activation_freq[fid].item(),
                max_activation=max_activation[fid].item(),
                top_tokens=top_toks,
            ))

        return sparse_features

    def find_top_features(
        self,
        activations: torch.Tensor,  # (B, d) activation for specific input
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Returns [(feature_id, activation_value), ...] sorted by activation descending."""
        self.sae.eval()
        with torch.no_grad():
            features = self.sae.encode(activations)  # (B, n_features)

        mean_features = features.mean(dim=0)  # (n_features,)

        active_mask = mean_features > 0
        n_active = int(active_mask.sum().item())
        k = min(top_k, n_active)

        if k == 0:
            return []

        topk_vals, topk_ids = mean_features.topk(k)
        result = [(int(topk_ids[i].item()), float(topk_vals[i].item())) for i in range(k)]
        return result


# ---------------------------------------------------------------------------
# CircuitFinder
# ---------------------------------------------------------------------------

class CircuitFinder:
    """Find minimal circuits of features explaining model behavior."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        model: nn.Module,
        layer_name: str = "hidden",
    ) -> None:
        self.sae = sae
        self.model = model
        self.layer_name = layer_name

    def find_circuit_greedy(
        self,
        activations: torch.Tensor,   # (B, d) target activations
        target_features: Optional[List[int]] = None,
        max_features: int = 20,
        threshold: float = 0.9,      # faithfulness threshold to stop
    ) -> Circuit:
        """
        Greedy circuit finding: iteratively add features that most improve
        faithfulness until threshold reached or max_features exceeded.
        Faithfulness = cosine similarity of reconstructed vs original activations.
        """
        self.sae.eval()
        with torch.no_grad():
            all_features = self.sae.encode(activations)  # (B, n_features)

        if target_features is not None:
            candidate_ids = list(target_features)
        else:
            active_anywhere = (all_features > 0).any(dim=0)  # (n_features,)
            candidate_ids = active_anywhere.nonzero(as_tuple=False).squeeze(-1).tolist()

        circuit_ids: List[int] = []
        best_faithfulness = 0.0

        for _ in range(max_features):
            if not candidate_ids:
                break

            best_fid = None
            best_score = -1.0

            for fid in candidate_ids:
                trial_ids = circuit_ids + [fid]
                score = self.compute_faithfulness(activations, trial_ids, all_features)
                if score > best_score:
                    best_score = score
                    best_fid = fid

            if best_fid is None:
                break

            circuit_ids.append(best_fid)
            candidate_ids.remove(best_fid)
            best_faithfulness = best_score

            if best_faithfulness >= threshold:
                break

        n_total_active = max(1, int((all_features > 0).any(dim=0).sum().item()))
        minimality = 1.0 - len(circuit_ids) / n_total_active
        minimality = max(0.0, min(1.0, minimality))

        completeness = best_faithfulness

        return Circuit(
            feature_ids=circuit_ids,
            layer_name=self.layer_name,
            faithfulness_score=max(0.0, min(1.0, best_faithfulness)),
            completeness_score=max(0.0, min(1.0, completeness)),
            minimality_score=minimality,
        )

    def compute_faithfulness(
        self,
        original: torch.Tensor,      # (B, d)
        circuit_features: List[int],
        all_features: torch.Tensor,  # (B, n_features)
    ) -> float:
        """
        Reconstruct using only circuit_features, measure cosine similarity to original.
        """
        if not circuit_features:
            return 0.0

        with torch.no_grad():
            masked = torch.zeros_like(all_features)
            for fid in circuit_features:
                masked[:, fid] = all_features[:, fid]

            reconstruction = self.sae.decode(masked)
            cos_sim = F.cosine_similarity(reconstruction, original, dim=-1)  # (B,)
            faithfulness = cos_sim.mean().item()

        return faithfulness

    def ablate_features(
        self,
        activations: torch.Tensor,   # (B, d)
        feature_ids: List[int],
        mode: str = "zero",          # "zero" | "mean"
    ) -> torch.Tensor:
        """
        Ablate (zero out or replace with mean) specified features.
        Returns modified reconstruction.
        """
        self.sae.eval()
        with torch.no_grad():
            features = self.sae.encode(activations)  # (B, n_features)
            ablated = features.clone()

            if mode == "zero":
                for fid in feature_ids:
                    ablated[:, fid] = 0.0
            elif mode == "mean":
                feature_means = features.mean(dim=0)  # (n_features,)
                for fid in feature_ids:
                    ablated[:, fid] = feature_means[fid]
            else:
                raise ValueError(f"Unknown ablation mode: {mode!r}. Use 'zero' or 'mean'.")

            reconstruction = self.sae.decode(ablated)

        return reconstruction


# ---------------------------------------------------------------------------
# Feature correlation
# ---------------------------------------------------------------------------

def compute_feature_correlation(
    feature_activations: torch.Tensor,  # (N, n_features)
    top_k: int = 10,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Compute pairwise feature co-activation correlation.
    Returns {feature_id: [(correlated_feature_id, correlation), ...]} for top_k pairs.
    Uses Pearson correlation (pure PyTorch).
    """
    N, n_features = feature_activations.shape

    mean = feature_activations.mean(dim=0, keepdim=True)  # (1, n_features)
    centered = feature_activations - mean                  # (N, n_features)

    std = centered.std(dim=0, unbiased=False)              # (n_features,)
    std = std.clamp(min=1e-8)

    normed = centered / std.unsqueeze(0)                   # (N, n_features)
    corr_matrix = (normed.T @ normed) / N                  # (n_features, n_features)

    result: Dict[int, List[Tuple[int, float]]] = {}

    for fid in range(n_features):
        row = corr_matrix[fid].clone()
        row[fid] = float('-inf')

        k = min(top_k, n_features - 1)
        topk_vals, topk_ids = row.topk(k)

        pairs = [
            (int(topk_ids[i].item()), float(topk_vals[i].item()))
            for i in range(k)
            if topk_vals[i].item() != float('-inf')
        ]
        result[fid] = pairs

    return result
