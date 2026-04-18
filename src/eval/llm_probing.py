"""
LLM Probing — linear/nonlinear probes that extract representations from
intermediate transformer layers to test what concepts are encoded.

Pure PyTorch only. No external dependencies beyond stdlib + torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ActivationExtractor
# ---------------------------------------------------------------------------

class ActivationExtractor:
    """
    Registers forward hooks on named sub-modules of a model and collects
    their output activations after each forward pass.

    Usage (explicit):
        extractor = ActivationExtractor(model, ["layer.0", "layer.1"])
        acts = extractor.forward(input_ids)  # {layer_name: [B, T, d]}

    Usage (context-manager):
        with ActivationExtractor(model, layer_names) as ext:
            model(input_ids)
            acts = ext.activations
    """

    def __init__(self, model: nn.Module, layer_names: List[str]) -> None:
        self.model = model
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        named_modules = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in named_modules:
                raise ValueError(
                    f"Layer '{name}' not found in model. "
                    f"Available: {list(named_modules.keys())}"
                )
            module = named_modules[name]
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook_fn

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass and return collected activations.

        Args:
            input_ids: [B, T] integer token ids.

        Returns:
            dict mapping layer_name -> [B, T, d_model] activation tensor.
        """
        self.clear()
        with torch.no_grad():
            self.model(input_ids)
        return dict(self.activations)

    def clear(self) -> None:
        """Clear all stored activations."""
        self.activations.clear()

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivationExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._remove_hooks()
        return False


# ---------------------------------------------------------------------------
# Helper: macro F1 in pure PyTorch
# ---------------------------------------------------------------------------

def _macro_f1(preds: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    f1_scores: List[float] = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


# ---------------------------------------------------------------------------
# LinearProbe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """
    Single linear layer probe for classifying internal representations.
    """

    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_in] -> logits: [B, n_classes]"""
        return self.linear(x)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int = 5,
        lr: float = 1e-3,
    ) -> List[float]:
        """Train the probe. Returns per-epoch loss history."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_history: List[float] = []

        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = self.forward(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        self.eval()
        return loss_history

    def evaluate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """Returns (accuracy, macro_f1) both in [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features)
            preds = logits.argmax(dim=-1)

        n_classes = logits.shape[-1]
        accuracy = (preds == labels).float().mean().item()
        macro_f1 = _macro_f1(preds, labels, n_classes)
        return accuracy, macro_f1


# ---------------------------------------------------------------------------
# NonlinearProbe
# ---------------------------------------------------------------------------

class NonlinearProbe(nn.Module):
    """
    Two-layer MLP probe (d_in -> hidden -> n_classes) with ReLU.
    Same fit/evaluate interface as LinearProbe.
    """

    def __init__(self, d_in: int, n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_in] -> logits: [B, n_classes]"""
        return self.net(x)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int = 5,
        lr: float = 1e-3,
    ) -> List[float]:
        """Train the probe. Returns per-epoch loss history."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_history: List[float] = []

        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = self.forward(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        self.eval()
        return loss_history

    def evaluate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """Returns (accuracy, macro_f1) both in [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features)
            preds = logits.argmax(dim=-1)

        n_classes = logits.shape[-1]
        accuracy = (preds == labels).float().mean().item()
        macro_f1 = _macro_f1(preds, labels, n_classes)
        return accuracy, macro_f1


# ---------------------------------------------------------------------------
# Helpers for RSA
# ---------------------------------------------------------------------------

def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-center a kernel matrix: H K H where H = I - (1/n) 11^T."""
    row_mean = K.mean(dim=1, keepdim=True)
    col_mean = K.mean(dim=0, keepdim=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean


def _knn_indices(X: torch.Tensor, k: int) -> torch.Tensor:
    """Return [N, k] tensor of k-NN indices for each row (excluding self)."""
    sq_norms = (X * X).sum(dim=1)
    dists = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (X @ X.T)
    dists.fill_diagonal_(float("inf"))
    _, indices = dists.topk(k, dim=1, largest=False, sorted=True)
    return indices


# ---------------------------------------------------------------------------
# RepresentationSimilarityAnalysis
# ---------------------------------------------------------------------------

class RepresentationSimilarityAnalysis:
    """
    Geometric and information-theoretic similarity measures between two sets
    of neural representations.
    """

    @staticmethod
    def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Linear CKA: HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
        HSIC(A,B) = trace(K_c L_c) / (n-1)^2  where K=AA^T, L=BB^T, centered.

        Returns value in [0, 1].
        """
        X = X.float()
        Y = Y.float()
        n = X.shape[0]

        K = X @ X.T
        L = Y @ Y.T

        Kc = _center_kernel(K)
        Lc = _center_kernel(L)

        hsic_xy = (Kc * Lc).sum() / (n - 1) ** 2
        hsic_xx = (Kc * Kc).sum() / (n - 1) ** 2
        hsic_yy = (Lc * Lc).sum() / (n - 1) ** 2

        denom = (hsic_xx * hsic_yy).sqrt()
        if denom.item() < 1e-10:
            return 0.0
        return (hsic_xy / denom).clamp(0.0, 1.0).item()

    @staticmethod
    def procrustes_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Orthogonal Procrustes distance:
            1 - sum(S) / (||X||_F * ||Y||_F)
        where U S V^T = SVD(X^T Y) and sum(S) = trace of diagonal.

        This equals 0 when X and Y are identical (up to orthogonal rotation)
        and approaches 1 when the spaces are maximally dissimilar.

        Returns value in [0, 1].
        """
        X = X.float()
        Y = Y.float()

        M = X.T @ Y
        _U, S, _Vh = torch.linalg.svd(M, full_matrices=False)
        # Procrustes similarity = sum of singular values / (||X||_F * ||Y||_F)
        trace_val = S.sum()

        norm_x = torch.norm(X, p="fro")
        norm_y = torch.norm(Y, p="fro")
        denom = norm_x * norm_y

        if denom.item() < 1e-10:
            return 0.0

        similarity = (trace_val / denom).clamp(0.0, 1.0).item()
        return 1.0 - similarity

    @staticmethod
    def mutual_knn(X: torch.Tensor, Y: torch.Tensor, k: int = 5) -> float:
        """
        Fraction of k-NN neighbors shared between X and Y spaces.
        Returns mean Jaccard overlap in [0, 1].
        """
        X = X.float()
        Y = Y.float()
        n = X.shape[0]

        knn_x = _knn_indices(X, k)
        knn_y = _knn_indices(Y, k)

        overlaps: List[float] = []
        for i in range(n):
            set_x = set(knn_x[i].tolist())
            set_y = set(knn_y[i].tolist())
            intersection = len(set_x & set_y)
            union = len(set_x | set_y)
            overlaps.append(intersection / union if union > 0 else 0.0)

        return sum(overlaps) / len(overlaps)


# ---------------------------------------------------------------------------
# ProbingConfig
# ---------------------------------------------------------------------------

@dataclass
class ProbingConfig:
    """Configuration dataclass for ProbeEvaluationSuite."""
    n_epochs: int = 5
    lr: float = 1e-3
    hidden: int = 64
    probe_type: str = "linear"
    k_knn: int = 5


# ---------------------------------------------------------------------------
# ProbeEvaluationSuite
# ---------------------------------------------------------------------------

class ProbeEvaluationSuite:
    """
    High-level interface: run probes across all captured layers and compare
    their representational geometry via CKA.
    """

    def __init__(self, model: nn.Module, layer_names: List[str]) -> None:
        self.model = model
        self.layer_names = layer_names
        self.extractor = ActivationExtractor(model, layer_names)

    def run_probe(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        probe_type: str = "linear",
        config: Optional[ProbingConfig] = None,
    ) -> Dict[str, Dict]:
        """
        For each layer: extract mean-pooled representations, fit a probe,
        evaluate it, and return results.

        Args:
            input_ids:  [N, T] integer token ids
            labels:     [N] integer class labels
            probe_type: "linear" or "nonlinear"
            config:     ProbingConfig (uses defaults if None)

        Returns:
            {layer_name: {"acc": float, "f1": float, "loss_curve": list[float]}, ...}
        """
        if config is None:
            config = ProbingConfig()

        activations = self.extractor.forward(input_ids)

        results: Dict[str, Dict] = {}
        for layer_name, acts in activations.items():
            features = acts.mean(dim=1)  # [N, d]
            d_in = features.shape[-1]
            n_classes = int(labels.max().item()) + 1

            if probe_type == "linear":
                probe: nn.Module = LinearProbe(d_in, n_classes)
            elif probe_type == "nonlinear":
                probe = NonlinearProbe(d_in, n_classes, hidden=config.hidden)
            else:
                raise ValueError(
                    f"Unknown probe_type '{probe_type}'. Use 'linear' or 'nonlinear'."
                )

            loss_curve = probe.fit(  # type: ignore[attr-defined]
                features.detach(),
                labels,
                n_epochs=config.n_epochs,
                lr=config.lr,
            )
            acc, f1 = probe.evaluate(features.detach(), labels)  # type: ignore[attr-defined]

            results[layer_name] = {"acc": acc, "f1": f1, "loss_curve": loss_curve}

        return results

    def compare_layers(
        self,
        input_ids: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise Linear CKA between all captured layers.

        Returns:
            {layer_a: {layer_b: cka_value, ...}, ...}
        """
        activations = self.extractor.forward(input_ids)
        pooled: Dict[str, torch.Tensor] = {
            name: acts.mean(dim=1) for name, acts in activations.items()
        }

        rsa = RepresentationSimilarityAnalysis()
        layer_list = list(pooled.keys())
        result: Dict[str, Dict[str, float]] = {name: {} for name in layer_list}

        for name_a in layer_list:
            for name_b in layer_list:
                result[name_a][name_b] = rsa.cka(pooled[name_a], pooled[name_b])

        return result
