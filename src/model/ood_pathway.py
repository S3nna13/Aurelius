"""Aurelius OOD Two-Pathway Detection — arXiv:2605.00269.

Two-pathway framework for understanding how language models process out-of-distribution
inputs, deconfounding OOD signals from length artifacts via length-matched evaluation.

Architecture:
    1. Embeddings pathway — k-NN on token embeddings; effective for topic-distinctive OOD.
    2. Processing trajectory pathway — hidden-state evolution across layers;
       effective for covert-intent inputs (e.g., jailbreaks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TrajectoryScore(NamedTuple):
    """Components of a trajectory OOD score."""

    total: Tensor
    embedding_delta: Tensor
    trajectory_entropy: Tensor
    layerwise: Tensor


class CircuitAttribution(NamedTuple):
    """Circuit attribution scores per layer and head."""

    layer_scores: Tensor
    head_scores: Tensor | None
    top_layers: list[int]
    top_heads: list[int] | None


@dataclass
class OODPathwayConfig:
    """Configuration for OOD two-pathway detection.

    Attributes:
        embedding_k: Number of nearest neighbours for k-NN embedding detection.
        trajectory_layers: List of layer indices to include in trajectory.
            None/empty means all layers.
        embedding_threshold: OOD threshold for embedding pathway logits.
        trajectory_threshold: OOD threshold for trajectory pathway logits.
        length_matched_bins: Number of length bins for length-matched evaluation.
        max_length_bin: Maximum sequence length per bin (auto-computed if 0).
        crossover_learning_rate: Learning rate for crossover weighting optimiser.
        crossover_weight_init: Initial embedding pathway weight (trajectory = 1 - this).
        circuit_attr_top_k: Number of top layers/heads to report in circuit attribution.
        layerwise_trajectory: Compute per-layer trajectory scores.
        use_length_correction: Apply length confound correction.
        normalize_trajectories: L2-normalize trajectory vectors before scoring.
        embedding_aggregation: How to aggregate token-level embedding scores.
            One of: "mean", "max", "last".
        trajectory_aggregation: How to aggregate layer-level trajectory scores.
            One of: "mean", "max", "weighted".
        jailbreak_auroc_target: Target AUROC for jailbreak detection (0.85 paper).
    """

    embedding_k: int = 32
    trajectory_layers: list[int] | None = None
    embedding_threshold: float = 0.5
    trajectory_threshold: float = 0.5
    length_matched_bins: int = 10
    max_length_bin: int = 0
    crossover_learning_rate: float = 1e-3
    crossover_weight_init: float = 0.5
    circuit_attr_top_k: int = 8
    layerwise_trajectory: bool = False
    use_length_correction: bool = True
    normalize_trajectories: bool = True
    embedding_aggregation: str = "mean"
    trajectory_aggregation: str = "mean"
    jailbreak_auroc_target: float = 0.85


class EmbeddingPathway:
    """Embedding-based OOD detection via k-NN on token embeddings.

    Captures "what text is about" — effective for topic-distinctive OOD inputs
    where the semantic content differs substantially from in-distribution data.

    The pathway computes the k-NN distance statistics of input token embeddings
    against a cached reference corpus of in-distribution embeddings. Out-of-distribution
    inputs tend to have larger average k-NN distances because their embeddings
    fall in sparse regions of the embedding space.

    Length confound: k-NN scores are structurally correlated with sequence length
    because longer sequences introduce more token-level variance. Use
    :class:`LengthConfoundCorrector` to deconfound.
    """

    def __init__(
        self,
        config: OODPathwayConfig,
        reference_embeddings: Tensor | None = None,
    ) -> None:
        """
        Args:
            config: OOD pathway configuration.
            reference_embeddings: Optional pre-cached in-distribution embeddings
                of shape (N_ref, d_model). If None, must be provided via ``set_reference``.
        """
        self.config = config
        self._reference: Tensor | None = reference_embeddings
        self._reference_lengths: Tensor | None = None

    def set_reference(self, embeddings: Tensor, lengths: Tensor | None = None) -> None:
        """Set the in-distribution reference corpus.

        Args:
            embeddings: Reference embeddings of shape (N, d_model).
            lengths: Optional reference sequence lengths (N,).
        """
        self._reference = embeddings
        self._reference_lengths = lengths

    @torch.no_grad()
    def score(self, token_embeddings: Tensor, mask: Tensor | None = None) -> Tensor:
        """Compute embedding OOD scores for a batch.

        Args:
            token_embeddings: Per-token embeddings of shape (batch, seq_len, d_model).
            mask: Optional padding mask of shape (batch, seq_len), 1 = valid.

        Returns:
            OOD scores of shape (batch,). Higher = more OOD.
        """
        if self._reference is None:
            raise RuntimeError("Reference embeddings not set. Call set_reference first.")

        B, S, D = token_embeddings.shape
        flat = token_embeddings.view(-1, D)
        valid_mask = mask.bool().view(-1) if mask is not None else None

        if valid_mask is not None:
            flat = flat[valid_mask]
            if flat.shape[0] == 0:
                return token_embeddings.new_zeros(B)

        reference = self._reference.to(device=flat.device, dtype=flat.dtype)
        k = min(self.config.embedding_k, reference.shape[0])
        dists = torch.cdist(flat, reference, p=2)
        topk_dists, _ = dists.topk(k, dim=1, largest=False)

        avg_dists = topk_dists.mean(dim=1)

        if valid_mask is not None:
            scores = flat.new_zeros(B * S)
            scores[valid_mask] = avg_dists
            scores = scores.view(B, S)
            if self.config.embedding_aggregation == "mean":
                denom = mask.sum(dim=1).clamp(min=1)
                return (scores.sum(dim=1) / denom).cpu()
            elif self.config.embedding_aggregation == "max":
                scores = scores.masked_fill(~mask.bool(), float("-inf"))
                return scores.max(dim=1)[0].cpu()
            else:
                return scores[:, -1].cpu()
        else:
            scores = avg_dists.view(B, S)
            if self.config.embedding_aggregation == "mean":
                return scores.mean(dim=1).cpu()
            elif self.config.embedding_aggregation == "max":
                return scores.max(dim=1)[0].cpu()
            else:
                return scores[:, -1].cpu()

    @torch.no_grad()
    def score_batch(
        self,
        token_ids: Tensor,
        model: nn.Module,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Convenience method: compute OOD scores from token IDs via model's embed layer.

        Args:
            token_ids: Token IDs of shape (batch, seq_len).
            model: Model with an ``embed`` attribute (nn.Embedding or similar).
            mask: Optional padding mask.

        Returns:
            OOD scores of shape (batch,).
        """
        embeds = model.embed(token_ids)
        return self.score(embeds, mask)


class TrajectoryPathway:
    """Processing-trajectory-based OOD detection via hidden-state evolution across layers.

    Captures "how the model processes input" — effective for covert-intent OOD inputs
    like jailbreaks where the adversarial content is hidden within normal-looking text
    but triggers distinct processing patterns.

    The pathway tracks the trajectory of hidden states through the transformer layers,
    computing:
    1. Per-layer hidden-state deltas (change from previous layer)
    2. Trajectory entropy (how variable the processing is across layers)
    3. Final-layer deviation from the mean trajectory

    Paper finding: trajectory features achieve 0.850 AUROC for jailbreak detection,
    significantly outperforming embedding-only methods on covert-intent tasks.
    """

    def __init__(self, config: OODPathwayConfig) -> None:
        self.config = config
        self._register_buffer = True

    @torch.no_grad()
    def score(
        self,
        hidden_states: list[Tensor] | Tensor,
        mask: Tensor | None = None,
    ) -> TrajectoryScore:
        """Compute trajectory OOD scores from hidden states.

        Args:
            hidden_states: Either a list of per-layer hidden states
                [(batch, seq, d_model), ...] or a single tensor of shape
                (n_layers, batch, seq, d_model).
            mask: Optional padding mask (batch, seq).

        Returns:
            TrajectoryScore with total OOD score and components.
        """
        if isinstance(hidden_states, list):
            stacked = torch.stack(hidden_states, dim=0)
        else:
            stacked = hidden_states

        n_layers, B, S, D = stacked.shape
        layers_to_use = self.config.trajectory_layers
        if layers_to_use:
            stacked = stacked[layers_to_use]

        deltas = stacked[1:] - stacked[:-1]
        delta_norms = deltas.norm(p=2, dim=-1)

        if self.config.normalize_trajectories:
            delta_mean = delta_norms.mean(dim=0, keepdim=True)
            delta_std = delta_norms.std(dim=0, keepdim=True) + 1e-8
            delta_norms = (delta_norms - delta_mean) / delta_std

        layerwise_entropy = self._trajectory_entropy(stacked)

        emb_delta = (stacked[-1] - stacked[0]).norm(p=2, dim=-1)
        if mask is not None:
            emb_delta = emb_delta * mask
            delta_norms = delta_norms * mask.unsqueeze(0)

        emb_agg = emb_delta.mean(dim=1)
        tray_agg = delta_norms.mean(dim=0).mean(dim=1)

        if self.config.trajectory_aggregation == "weighted":
            total = 0.6 * emb_agg + 0.4 * tray_agg
        elif self.config.trajectory_aggregation == "max":
            total = torch.maximum(emb_agg, tray_agg)
        else:
            total = 0.5 * emb_agg + 0.5 * tray_agg

        return TrajectoryScore(
            total=total.cpu(),
            embedding_delta=emb_agg.cpu(),
            trajectory_entropy=layerwise_entropy.mean(dim=1).cpu(),
            layerwise=delta_norms.mean(dim=0).cpu(),
        )

    def _trajectory_entropy(self, stacked: Tensor) -> Tensor:
        """Compute per-token trajectory entropy — how variable the layer-to-layer changes are.

        Args:
            stacked: (n_layers, B, S, D)

        Returns:
            Entropy per token (B, S).
        """
        deltas = stacked[1:] - stacked[:-1]
        delta_norms = deltas.norm(p=2, dim=-1)

        probs = F.softmax(delta_norms, dim=0)
        eps = torch.finfo(probs.dtype).eps
        entropy = -(probs * (probs + eps).log()).sum(dim=0)

        entropy_max = math.log(delta_norms.shape[0])
        entropy_norm = entropy / (entropy_max + 1e-8)

        return entropy_norm

    @torch.no_grad()
    def score_from_model(
        self,
        token_ids: Tensor,
        model: nn.Module,
        mask: Tensor | None = None,
    ) -> TrajectoryScore:
        """Run a forward pass and extract trajectory scores.

        Args:
            token_ids: (batch, seq_len).
            model: AureliusTransformer or compatible model with return_hidden_states support.
            mask: Optional padding mask.

        Returns:
            TrajectoryScore.
        """
        B, S = token_ids.shape

        x = model.embed(token_ids)
        freqs_cis = model.freqs_cis[:S]

        layer_hiddens: list[Tensor] = []
        for layer in model.layers:
            x_norm = layer.attn_norm(x)
            attn_out, _, _ = layer.attn(x_norm, freqs_cis, None, None)
            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))
            layer_hiddens.append(x.clone())

        x = model.norm(x)
        layer_hiddens.append(x)

        stacked = torch.stack(layer_hiddens, dim=0)
        return self.score(stacked, mask)


class TwoPathwayOODDetector:
    """Combined OOD detector with learned crossover weighting between pathways.

    The two pathways capture complementary signals:
    - Embedding pathway: "what text is about" (topic shifts, semantic drift)
    - Trajectory pathway: "how the model processes input" (covert intent, jailbreaks)

    The crossover weighting mechanism learns to combine these signals in a task-aware
    manner — weighting trajectory signals higher for adversarial/covert tasks and
    embedding signals higher for topic-distinctive tasks.

    Usage:
        detector = TwoPathwayOODDetector(model, config)
        detector.fit_reference(id_embeddings, id_hidden_states)
        ood_score, weights = detector.score(input_ids, model)
    """

    def __init__(
        self,
        model: nn.Module,
        config: OODPathwayConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or OODPathwayConfig()
        self.embedding_pathway = EmbeddingPathway(self.config)
        self.trajectory_pathway = TrajectoryPathway(self.config)
        self.length_corrector = LengthConfoundCorrector(self.config)

        self._weight = nn.Parameter(
            torch.tensor(self.config.crossover_weight_init, dtype=torch.float32)
        )
        self._optimizer: torch.optim.Optimizer | None = None
        self._id_hidden_states: list[Tensor] | Tensor | None = None
        self._id_lengths: Tensor | None = None
        self._fitted = False

    def fit_reference(
        self,
        id_embeddings: Tensor,
        id_hidden_states: list[Tensor] | Tensor | None = None,
        id_lengths: Tensor | None = None,
    ) -> None:
        """Fit the detector on in-distribution data.

        Args:
            id_embeddings: In-distribution token embeddings (N, d_model).
            id_hidden_states: Optional in-distribution hidden states for trajectory pathway.
            id_lengths: Optional sequence lengths for length-matched evaluation.
        """
        self.embedding_pathway.set_reference(id_embeddings, id_lengths)
        self._id_hidden_states = id_hidden_states
        self._id_lengths = id_lengths
        self._fitted = True

    def fit_crossover(
        self,
        ood_embeddings: Tensor,
        ood_hidden_states: list[Tensor] | Tensor | None = None,
        id_embeddings: Tensor | None = None,
        ood_lengths: Tensor | None = None,
        id_lengths: Tensor | None = None,
        n_steps: int = 100,
    ) -> dict[str, float]:
        """Fit the crossover weighting on mixed ID/OOD data.

        Args:
            ood_embeddings: OOD embeddings (M, d_model).
            ood_hidden_states: Optional OOD hidden states.
            id_embeddings: ID embeddings (if not already fitted via fit_reference).
            ood_lengths: OOD sequence lengths.
            id_lengths: ID sequence lengths.
            n_steps: Number of optimisation steps.

        Returns:
            Dict with training metrics.
        """
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                [self._weight],
                lr=self.config.crossover_learning_rate,
            )

        id_emb = id_embeddings if id_embeddings is not None else self.embedding_pathway._reference
        if id_emb is None:
            raise RuntimeError("ID embeddings required. Call fit_reference or pass id_embeddings.")

        labels = torch.cat(
            [
                torch.zeros(id_emb.shape[0]),
                torch.ones(ood_embeddings.shape[0]),
            ]
        )
        embeddings = torch.cat([id_emb, ood_embeddings], dim=0)

        losses = []
        for step in range(n_steps):
            self._optimizer.zero_grad()
            w = torch.sigmoid(self._weight)

            emb_scores = self.embedding_pathway.score(embeddings.unsqueeze(1)).squeeze()
            ref_states = getattr(self, "_id_hidden_states", None)
            if ref_states is not None and len(ref_states) > 0:
                tray_scores = self.trajectory_pathway.score(ref_states).total
                if tray_scores.shape != emb_scores.shape:
                    tray_scores = torch.zeros_like(emb_scores)
                else:
                    tray_scores = tray_scores.to(device=emb_scores.device, dtype=emb_scores.dtype)
            else:
                tray_scores = torch.zeros_like(emb_scores)

            combined = w * emb_scores + (1 - w) * tray_scores

            loss = F.binary_cross_entropy_with_logits(combined, labels)
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())

        return {
            "crossover_loss": float(sum(losses) / len(losses)),
            "final_weight": float(torch.sigmoid(self._weight)),
        }

    @torch.no_grad()
    def score(
        self,
        token_ids: Tensor,
        mask: Tensor | None = None,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute combined OOD scores for a batch.

        Args:
            token_ids: (batch, seq_len).
            mask: Optional padding mask.
            return_weights: If True, return (scores, pathway_weights).

        Returns:
            OOD scores (batch,) or (scores, weights) if return_weights=True.
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit_reference first.")

        embeds = self.model.embed(token_ids)
        emb_score = self.embedding_pathway.score(embeds, mask)
        tray_score = self.trajectory_pathway.score_from_model(token_ids, self.model, mask).total

        w = torch.sigmoid(self._weight)
        combined = w * emb_score + (1 - w) * tray_score

        if self.config.use_length_correction and mask is not None:
            lengths = mask.sum(dim=1)
            combined = self.length_corrector.correct(combined, lengths)

        if return_weights:
            return combined, torch.stack([w, 1 - w])
        return combined

    @torch.no_grad()
    def predict(
        self,
        token_ids: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Full OOD prediction with per-pathway scores.

        Args:
            token_ids: (batch, seq_len).
            mask: Optional padding mask.

        Returns:
            Dict with keys: combined_score, embedding_score, trajectory_score,
                is_ood (bool), pathway_weights.
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit_reference first.")

        embeds = self.model.embed(token_ids)
        emb_score = self.embedding_pathway.score(embeds, mask)
        tray = self.trajectory_pathway.score_from_model(token_ids, self.model, mask)

        w = torch.sigmoid(self._weight)
        combined = w * emb_score + (1 - w) * tray.total

        if self.config.use_length_correction and mask is not None:
            lengths = mask.sum(dim=1)
            combined = self.length_corrector.correct(combined, lengths)

        is_ood = torch.logical_or(
            emb_score > self.config.embedding_threshold,
            tray.total > self.config.trajectory_threshold,
        )

        return {
            "combined_score": combined,
            "embedding_score": emb_score,
            "trajectory_score": tray.total,
            "trajectory_embedding_delta": tray.embedding_delta,
            "trajectory_entropy": tray.trajectory_entropy,
            "layerwise_trajectory": tray.layerwise,
            "is_ood": is_ood,
            "pathway_weights": torch.stack([w, 1 - w]),
        }


class LengthConfoundCorrector:
    """Applies length-matched evaluation to deconfound OOD signals from length artifacts.

    Key insight from paper: existing OOD detection methods (CED, RAUQ, WildGuard,
    attention entropy) are structurally confounded by input sequence length. They
    collapse to near-chance under length-matched evaluation.

    This corrector implements the length-matched evaluation protocol:
    1. Bin samples by sequence length
    2. Within each bin, compute OOD scores and labels
    3. Apply bin-specific threshold calibration to remove length correlation

    Per-layer analysis from the paper: Layer-0 k-NN signal is almost entirely
    a length artifact; processing constructs genuine OOD signal from near-chance embeddings.
    """

    def __init__(self, config: OODPathwayConfig | None = None) -> None:
        self.config = config or OODPathwayConfig()
        self._length_bins: list[tuple[float, float]] = []
        self._bin_thresholds: list[float] = []
        self._fitted = False

    def fit(
        self,
        scores: Tensor,
        lengths: Tensor,
        labels: Tensor,
    ) -> dict[str, float]:
        """Fit length-bin calibration on ID/OOD data with known labels.

        Args:
            scores: Raw OOD scores (N,).
            lengths: Sequence lengths (N,).
            labels: Binary labels 0=ID, 1=OOD (N,).

        Returns:
            Dict with calibration metrics including length correlation reduction.
        """
        n_bins = self.config.length_matched_bins
        max_len = self.config.max_length_bin or int(lengths.max().item())
        bin_width = max_len / n_bins

        self._length_bins = []
        self._bin_thresholds = []

        bin_correlations = []
        raw_correlations = []

        for i in range(n_bins):
            lo = i * bin_width
            hi = (i + 1) * bin_width
            mask = (lengths >= lo) & (lengths < hi)

            if mask.sum() < 2:
                self._length_bins.append((lo, hi))
                self._bin_thresholds.append(0.5)
                continue

            bin_scores = scores[mask]
            bin_labels = labels[mask]

            if bin_labels.sum() > 0 and bin_labels.sum() < len(bin_labels):
                tpr = bin_scores[bin_labels == 1].mean()
                fpr = bin_scores[bin_labels == 0].mean()
                threshold = (tpr + fpr) / 2
            else:
                threshold = 0.5

            self._length_bins.append((lo, hi))
            self._bin_thresholds.append(threshold)

            if len(bin_labels) > 1:
                length_bin_vals = torch.full_like(bin_labels, float(i))
                centered_scores = bin_scores - bin_scores.mean()
                centered_bins = length_bin_vals - length_bin_vals.mean()
                corr = centered_scores.dot(centered_bins)
                denom = bin_scores.std() * length_bin_vals.std() + 1e-8
                bin_correlations.append(corr / denom)
                raw_correlations.append(corr / denom)

        self._fitted = True

        return {
            "mean_bin_correlation": float(torch.stack(bin_correlations).mean().abs())
            if bin_correlations
            else 0.0,
            "n_bins": n_bins,
        }

    @torch.no_grad()
    def correct(self, scores: Tensor, lengths: Tensor) -> Tensor:
        """Apply length-matched correction to OOD scores.

        Args:
            scores: Raw OOD scores (batch,).
            lengths: Sequence lengths (batch,).

        Returns:
            Length-corrected OOD scores (batch,).
        """
        if not self._fitted:
            return scores

        corrected = scores.clone()

        for i, ((lo, hi), thresh) in enumerate(zip(self._length_bins, self._bin_thresholds)):
            mask = (lengths >= lo) & (lengths < hi)
            if mask.sum() == 0:
                continue

            bin_scores = corrected[mask]
            centered = bin_scores - bin_scores.mean() + thresh
            corrected[mask] = centered

        return corrected

    @torch.no_grad()
    def correct_batch(self, scores: Tensor, lengths: Tensor) -> Tensor:
        """Batch-friendly length correction.

        Args:
            scores: (N,).
            lengths: (N,).

        Returns:
            Corrected scores (N,).
        """
        return self.correct(scores, lengths)


class CircuitAttributor:
    """Attributes OOD detection signals to specific attention circuits.

    Implements circuit-level analysis to identify which layers and attention heads
    contribute most to OOD detection. The paper finds that adversarial tasks engage
    attention circuits more than semantic tasks, and that early-layer attention
    patterns differ substantially between ID and OOD inputs.

    Usage:
        attr = CircuitAttributor(model, config)
        attribution = attr.attribute(token_ids, model, is_ood=True)
    """

    def __init__(self, model: nn.Module, config: OODPathwayConfig | None = None) -> None:
        self.model = model
        self.config = config or OODPathwayConfig()
        self._hook_handles: list = []

    @torch.no_grad()
    def attribute(
        self,
        token_ids: Tensor,
        mask: Tensor | None = None,
        is_ood: bool = True,
    ) -> CircuitAttribution:
        """Attribute OOD signal to specific layers and heads.

        Args:
            token_ids: (batch, seq_len).
            mask: Optional padding mask.
            is_ood: If True, attribute OOD signal; else attribute ID signal.

        Returns:
            CircuitAttribution with per-layer and per-head scores.
        """
        B, S = token_ids.shape

        layer_attn_scores: list[Tensor] = []
        layer_hidden_deltas: list[Tensor] = []
        prev_hidden: Tensor | None = None

        x = self.model.embed(token_ids)
        freqs_cis = self.model.freqs_cis[:S]

        for layer_idx, layer in enumerate(self.model.layers):
            x_norm = layer.attn_norm(x)
            attn_out, kv = layer.attn(x_norm, freqs_cis, None, None)

            if hasattr(layer.attn, "attn_weights") and layer.attn.attn_weights is not None:
                attn_weights = layer.attn.attn_weights
                if mask is not None:
                    attn_weights = attn_weights * mask.unsqueeze(1).unsqueeze(2)
                layer_attn_scores.append(attn_weights.mean(dim=-1))

            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

            if prev_hidden is not None:
                layer_hidden_deltas.append((x - prev_hidden).norm(p=2, dim=-1).mean())
            prev_hidden = x.clone()

        layer_scores = (
            torch.stack(layer_hidden_deltas)
            if layer_hidden_deltas
            else torch.zeros(len(self.model.layers), dtype=x.dtype, device=x.device)
        )

        if layer_scores.numel() > 0:
            top_layers_list = torch.argsort(layer_scores, descending=True)[
                : self.config.circuit_attr_top_k
            ].tolist()
        else:
            top_layers_list = []

        head_scores = None
        top_heads_list = None
        if layer_attn_scores:
            avg_head_scores = torch.stack([s.mean(dim=0).mean() for s in layer_attn_scores])
            head_scores = avg_head_scores
            top_heads_list = torch.argsort(avg_head_scores, descending=True)[
                : self.config.circuit_attr_top_k
            ].tolist()

        return CircuitAttribution(
            layer_scores=layer_scores.cpu(),
            head_scores=head_scores.cpu() if head_scores is not None else None,
            top_layers=top_layers_list,
            top_heads=top_heads_list,
        )

    @torch.no_grad()
    def attribute_with_hooks(
        self,
        token_ids: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Attribute using attention hooks for more detailed head-level analysis.

        Args:
            token_ids: (batch, seq_len).
            mask: Optional padding mask.

        Returns:
            Dict with layer_scores (n_layers,) and head_scores (n_layers, n_heads).
        """
        head_scores_per_layer: list[Tensor] = []
        layer_scores: list[Tensor] = []
        prev_hidden: Tensor | None = None

        hooks = []

        def attn_hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    head_scores_per_layer.append(attn_weights.detach())

        x = self.model.embed(token_ids)
        freqs_cis = self.model.freqs_cis[: x.shape[1]]

        for layer_idx, layer in enumerate(self.model.layers):
            for handle in hooks:
                handle.remove()
            hooks.clear()

            h = layer.attn.register_forward_hook(attn_hook_fn)

            x_norm = layer.attn_norm(x)
            attn_out, kv = layer.attn(x_norm, freqs_cis, mask, None)
            h.remove()

            if prev_hidden is not None:
                delta = (x + attn_out - prev_hidden).norm(p=2, dim=-1)
                layer_scores.append(delta.mean())

            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))
            prev_hidden = x.clone()

            hooks.append(layer.attn.register_forward_hook(attn_hook_fn))

        for h in hooks:
            h.remove()

        result: dict[str, Tensor] = {}
        if layer_scores:
            result["layer_scores"] = torch.stack(layer_scores).cpu()
        if head_scores_per_layer:
            stacked = torch.stack(head_scores_per_layer)
            result["head_scores"] = stacked.mean(dim=0).cpu()

        return result
