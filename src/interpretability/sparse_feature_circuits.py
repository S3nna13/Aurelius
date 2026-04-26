"""
src/interpretability/sparse_feature_circuits.py

Sparse Feature Circuits for transformer interpretability.

Based on Marks et al. 2024 — "Sparse Feature Circuits: Discovering and Editing
Interpretable Causal Graphs in Language Models."

Key idea: instead of expressing circuits in terms of raw neurons (MLP outputs),
decompose activations into sparse, interpretable features via a Sparse Autoencoder
(SAE), then run activation patching at the feature level.

Steps:
  1. Train / use an SAE to decompose MLP activations into an overcomplete sparse
     feature basis.
  2. Run activation patching on individual SAE features.
  3. Rank features by their causal contribution to the target metric.

Pure PyTorch — no HuggingFace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SparseCircuitConfig
# ---------------------------------------------------------------------------


@dataclass
class SparseCircuitConfig:
    """Configuration for sparse feature circuit discovery."""

    n_features: int = 256  # SAE dictionary size (typically 4-16x d_model)
    sparsity_coef: float = 0.01  # L1 penalty coefficient for feature sparsity
    top_k: int = 10  # Number of top circuit features to return


# ---------------------------------------------------------------------------
# SparseAutoencoder
# ---------------------------------------------------------------------------


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder (SAE) that decomposes model activations into an
    overcomplete sparse feature dictionary.

    Architecture:
        Encoder : Linear(d_model, n_features) + ReLU  →  sparse feature activations
        Decoder : Linear(n_features, d_model, bias=False) with unit-norm columns

    The encoder has a bias; the decoder has no bias so that features directly
    correspond to directions in the residual-stream / MLP-output space.

    Parameters
    ----------
    d_model       : Dimensionality of the input activations.
    n_features    : Size of the overcomplete feature dictionary (typically 4–16× d_model).
    sparsity_coef : Weight of the L1 sparsity regularisation term in the loss.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        sparsity_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.sparsity_coef = sparsity_coef

        # Encoder: maps activation → sparse feature space
        self.encoder = nn.Linear(d_model, n_features)

        # Decoder: maps sparse features → reconstructed activation (no bias)
        self.decoder = nn.Linear(n_features, d_model, bias=False)

        # Initialise and immediately normalise decoder columns
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.normalize_decoder()

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations into sparse features.

        Parameters
        ----------
        x : (..., d_model)  — activation tensor (any leading batch dims).

        Returns
        -------
        features : (..., n_features)  — non-negative sparse activations.
        """
        return F.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct activations from sparse features.

        Parameters
        ----------
        features : (..., n_features)

        Returns
        -------
        reconstruction : (..., d_model)
        """
        return self.decoder(features)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encode → decode.

        Parameters
        ----------
        x : (..., d_model)

        Returns
        -------
        (reconstruction, features) : Tuple[(..., d_model), (..., n_features)]
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute SAE training loss.

        Loss = MSE reconstruction loss + sparsity_coef * mean L1 feature norm.

        Parameters
        ----------
        x : (..., d_model)  — target activations.

        Returns
        -------
        (total_loss, info_dict)
        info_dict keys:
            'recon_loss'           : scalar MSE reconstruction loss.
            'sparsity_loss'        : scalar L1 sparsity penalty.
            'mean_active_features' : average number of non-zero features per token.
        """
        reconstruction, features = self.forward(x)

        recon_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = self.sparsity_coef * features.abs().mean()
        total_loss = recon_loss + sparsity_loss

        # Number of active (> 0) features per token, averaged across the batch
        mean_active = (features > 0).float().sum(dim=-1).mean()

        info = {
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "mean_active_features": mean_active,
        }
        return total_loss, info

    # ------------------------------------------------------------------
    # normalize_decoder
    # ------------------------------------------------------------------

    def normalize_decoder(self) -> None:
        """Normalise each decoder column to unit L2 norm (in-place, no grad)."""
        with torch.no_grad():
            # decoder.weight shape: (d_model, n_features)
            # Each column is decoder.weight[:, j]
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)


# ---------------------------------------------------------------------------
# FeatureCircuitFinder
# ---------------------------------------------------------------------------


class FeatureCircuitFinder:
    """Identifies circuit-relevant SAE features via activation patching.

    For each candidate feature *f* at *layer_idx*:
      1. Run clean forward pass; encode MLP output with SAE → clean_features.
      2. Run corrupted forward pass with a hook that:
           a. Encodes the corrupted MLP output into corrupted_features.
           b. Replaces corrupted_features[:, :, f] with clean_features[:, :, f].
           c. Decodes the patched features back into the MLP output space.
      3. Record metric_fn(patched_logits) as the feature's patching score.

    Parameters
    ----------
    model     : AureliusTransformer (or compatible) — must have `.layers` as
                nn.ModuleList where each layer has a `.ffn` sub-module.
    sae       : Trained SparseAutoencoder whose d_model matches the MLP output dim.
    metric_fn : Callable(logits: Tensor) → float.  Higher = more correct.
    """

    def __init__(
        self,
        model: nn.Module,
        sae: SparseAutoencoder,
        metric_fn: Callable[[torch.Tensor], float],
    ) -> None:
        self.model = model
        self.sae = sae
        self.metric_fn = metric_fn

    # ------------------------------------------------------------------
    # _run_forward
    # ------------------------------------------------------------------

    @staticmethod
    def _run_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Run model and return logits, handling the (loss, logits, kvs) tuple."""
        with torch.no_grad():
            output = model(input_ids)
        if isinstance(output, tuple):
            return output[1]  # AureliusTransformer: (loss, logits, kv_cache)
        return output

    # ------------------------------------------------------------------
    # get_feature_activations
    # ------------------------------------------------------------------

    def get_feature_activations(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Extract SAE feature activations for the MLP output at *layer_idx*.

        Parameters
        ----------
        input_ids : (batch, seq_len)
        layer_idx : Which transformer layer's FFN output to encode.

        Returns
        -------
        features : (batch, seq_len, n_features)
        """
        mlp_output: list[torch.Tensor] = []

        def _hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            mlp_output.append(out.detach())

        layer = list(self.model.layers)[layer_idx]
        handle = layer.ffn.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            handle.remove()

        # mlp_output[0]: (batch, seq_len, d_model)
        act = mlp_output[0]
        with torch.no_grad():
            features = self.sae.encode(act)  # (batch, seq_len, n_features)
        return features

    # ------------------------------------------------------------------
    # patch_feature
    # ------------------------------------------------------------------

    def patch_feature(
        self,
        clean_ids: torch.Tensor,
        corrupted_ids: torch.Tensor,
        layer_idx: int,
        feature_idx: int,
    ) -> float:
        """Patch a single SAE feature from the clean run into the corrupted run.

        Steps:
          1. Collect clean MLP output at layer_idx and encode it → clean_features.
          2. Run corrupted forward with a hook that:
               • Encodes the corrupted MLP output → corrupted_features.
               • Replaces corrupted_features[..., feature_idx] with clean value.
               • Decodes back to d_model space and returns as patched MLP output.
          3. Return metric_fn(patched_logits).

        Parameters
        ----------
        clean_ids     : (batch, seq_len) clean token ids.
        corrupted_ids : (batch, seq_len) corrupted token ids.
        layer_idx     : Transformer layer whose FFN output to patch.
        feature_idx   : Index of the SAE feature to patch.

        Returns
        -------
        metric_fn(patched_logits) as a float.
        """
        # Step 1: collect clean features
        clean_features = self.get_feature_activations(clean_ids, layer_idx)
        # clean_features: (batch, seq_len, n_features)

        # Step 2: register hook on corrupted run
        sae = self.sae

        def _patch_hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            out = output[0] if is_tuple else output  # (B, S, d_model)

            with torch.no_grad():
                corrupted_feats = sae.encode(out)  # (B, S, n_features)
                # Patch the chosen feature dimension
                corrupted_feats[..., feature_idx] = clean_features[..., feature_idx].to(out.device)
                patched_out = sae.decode(corrupted_feats)  # (B, S, d_model)

            if is_tuple:
                return (patched_out,) + output[1:]
            return patched_out

        layer = list(self.model.layers)[layer_idx]
        handle = layer.ffn.register_forward_hook(_patch_hook)
        try:
            logits = self._run_forward(self.model, corrupted_ids)
        finally:
            handle.remove()

        return self.metric_fn(logits)

    # ------------------------------------------------------------------
    # find_circuit_features
    # ------------------------------------------------------------------

    def find_circuit_features(
        self,
        clean_ids: torch.Tensor,
        corrupted_ids: torch.Tensor,
        layer_idx: int,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Identify the top-k SAE features most causally relevant to the metric.

        For every feature index 0..n_features-1, compute the patching score
        (metric value when that feature is restored from the clean run).
        Return the top_k (feature_idx, score) pairs sorted by score descending.

        Parameters
        ----------
        clean_ids     : (batch, seq_len) clean token ids.
        corrupted_ids : (batch, seq_len) corrupted token ids.
        layer_idx     : Which layer to analyse.
        top_k         : How many top features to return.

        Returns
        -------
        List of (feature_idx, score) tuples, sorted by score descending.
        """
        n_features = self.sae.n_features
        scores: list[tuple[int, float]] = []

        for feat_idx in range(n_features):
            score = self.patch_feature(clean_ids, corrupted_ids, layer_idx, feat_idx)
            scores.append((feat_idx, score))

        # Sort by score descending
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]
