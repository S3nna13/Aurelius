"""Integrated Gradients attribution (Sundararajan et al., 2017)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Core interpolation helper
# ---------------------------------------------------------------------------


def interpolate_inputs(baseline: Tensor, inputs: Tensor, alpha: float) -> Tensor:
    """Linear interpolation between baseline and inputs.

    Args:
        baseline: Baseline tensor, same shape as inputs.
        inputs: Input tensor.
        alpha: Interpolation coefficient in [0, 1]. alpha=0 returns baseline, alpha=1 returns inputs.

    Returns:
        Interpolated tensor of the same shape as inputs.
    """  # noqa: E501
    return baseline + alpha * (inputs - baseline)


# ---------------------------------------------------------------------------
# Embedding-hook utility
# ---------------------------------------------------------------------------


def _get_embedding_layer(model: nn.Module) -> nn.Embedding:
    """Return the token embedding layer from an AureliusTransformer (or similar)."""
    for attr in ("token_embedding", "embed_tokens", "embed"):
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, nn.Embedding):
                return layer
    # Walk named modules as fallback
    for _name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            return module
    raise AttributeError("Could not find an nn.Embedding layer in the model.")


# ---------------------------------------------------------------------------
# Core Integrated Gradients computation
# ---------------------------------------------------------------------------


def compute_integrated_gradients(
    model: nn.Module,
    input_ids: Tensor,
    target_token_idx: int,
    target_pos: int,
    n_steps: int = 50,
    baseline_token: int = 0,
) -> Tensor:
    """Compute Integrated Gradients attribution scores per input position.

    Args:
        model: AureliusTransformer (or compatible) model. Called as model(input_ids)
            returning (loss, logits, past_key_values).
        input_ids: Shape (1, T) single-batch input token ids.
        target_token_idx: Vocabulary index of the target token to explain.
        target_pos: Position in the sequence whose logit is differentiated.
        n_steps: Number of Riemann-sum integration steps.
        baseline_token: Token id used for the all-constant baseline sequence.

    Returns:
        Attribution tensor of shape (T,) with absolute attribution per position.
    """
    embedding_layer = _get_embedding_layer(model)

    # Build baseline input ids (same shape as input_ids)
    baseline_ids = torch.full_like(input_ids, fill_value=baseline_token)

    # Embed baseline and input once (no grad needed -- only for delta)
    with torch.no_grad():
        baseline_emb = embedding_layer(baseline_ids).detach()  # (1, T, d)
        inputs_emb = embedding_layer(input_ids).detach()  # (1, T, d)
    delta_emb = inputs_emb - baseline_emb  # (1, T, d)

    accumulated_grads = torch.zeros_like(inputs_emb)  # (1, T, d)

    # skip alpha=0 (gradient at baseline -- degenerate)
    alphas = torch.linspace(0.0, 1.0, steps=n_steps + 1)[1:]

    model.eval()
    for alpha in alphas:
        # Interpolated embedding: baseline_emb + alpha * delta_emb
        interp_emb = baseline_emb + alpha.item() * delta_emb  # (1, T, d)
        interp_emb = interp_emb.detach().requires_grad_(True)

        # We hook the embedding layer's forward to return our interpolated tensor
        # instead of looking up token ids.
        def _make_hook(emb_tensor):
            def _fwd_hook(module, inp, output):
                return emb_tensor

            return _fwd_hook

        handle = embedding_layer.register_forward_hook(_make_hook(interp_emb))
        try:
            _loss, logits, _pkv = model(input_ids)
        finally:
            handle.remove()

        scalar = logits[0, target_pos, target_token_idx]

        (grad,) = torch.autograd.grad(
            scalar,
            interp_emb,
            create_graph=False,
            retain_graph=False,
        )
        accumulated_grads = accumulated_grads + grad.detach()

    # Average gradients (Riemann approximation)
    avg_grads = accumulated_grads / n_steps  # (1, T, d)

    # Element-wise multiply by (inputs_emb - baseline_emb), sum over embedding dim
    ig = (avg_grads * delta_emb).sum(dim=-1)  # (1, T)
    ig = ig.squeeze(0)  # (T,)

    return ig.abs()


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class IGConfig:
    """Configuration for IntegratedGradientsExplainer."""

    n_steps: int = 50
    baseline_token: int = 0
    normalize: bool = True


# ---------------------------------------------------------------------------
# High-level explainer class
# ---------------------------------------------------------------------------


class IntegratedGradientsExplainer:
    """Wraps compute_integrated_gradients with a convenient high-level API."""

    def __init__(self, model: nn.Module, config: IGConfig) -> None:
        self.model = model
        self.config = config

    def explain(self, input_ids: Tensor, target_pos: int, target_token_idx: int) -> Tensor:
        """Compute attribution scores for every input position.

        Args:
            input_ids: Shape (1, T) token ids.
            target_pos: Sequence position whose logit is attributed.
            target_token_idx: Vocabulary index of the target token.

        Returns:
            Attribution tensor of shape (T,). If config.normalize is True the
            values are divided by their sum so they sum to 1.
        """
        attributions = compute_integrated_gradients(
            model=self.model,
            input_ids=input_ids,
            target_token_idx=target_token_idx,
            target_pos=target_pos,
            n_steps=self.config.n_steps,
            baseline_token=self.config.baseline_token,
        )

        if self.config.normalize:
            total = attributions.sum()
            if total > 0:
                attributions = attributions / total

        return attributions

    def top_k_tokens(self, attributions: Tensor, k: int) -> list[int]:
        """Return the k token positions with the highest attribution scores.

        Args:
            attributions: Shape (T,) attribution tensor.
            k: Number of top positions to return.

        Returns:
            List of k position indices sorted by attribution descending.
        """
        k = min(k, attributions.shape[0])
        _, indices = torch.topk(attributions, k=k, largest=True, sorted=True)
        return indices.tolist()


# ---------------------------------------------------------------------------
# SmoothGrad
# ---------------------------------------------------------------------------


def smooth_grad(
    model: nn.Module,
    input_ids: Tensor,
    target_token_idx: int,
    target_pos: int,
    n_samples: int = 10,
    noise_std: float = 0.01,
) -> Tensor:
    """SmoothGrad: average IG attributions over noisy embedding perturbations.

    Gaussian noise is injected into the embedding space (not token ids) for
    each sample, then integrated gradients are computed and results averaged.

    Args:
        model: AureliusTransformer (or compatible).
        input_ids: Shape (1, T).
        target_token_idx: Vocabulary index of the target token.
        target_pos: Sequence position whose logit is attributed.
        n_samples: Number of noisy samples.
        noise_std: Standard deviation of additive Gaussian noise on embeddings.

    Returns:
        Averaged attribution tensor of shape (T,).
    """
    embedding_layer = _get_embedding_layer(model)
    n_steps = 50

    with torch.no_grad():
        inputs_emb_clean = embedding_layer(input_ids).detach()  # (1, T, d)
        baseline_ids = torch.full_like(input_ids, fill_value=0)
        baseline_emb = embedding_layer(baseline_ids).detach()  # (1, T, d)

    accumulated = torch.zeros(input_ids.shape[1])  # (T,)

    model.eval()
    for _ in range(n_samples):
        noise = torch.randn_like(inputs_emb_clean) * noise_std
        noisy_inputs_emb = inputs_emb_clean + noise  # (1, T, d)
        delta_emb = noisy_inputs_emb - baseline_emb

        sample_grads = torch.zeros_like(noisy_inputs_emb)
        alphas = torch.linspace(0.0, 1.0, steps=n_steps + 1)[1:]

        for alpha in alphas:
            interp_emb = baseline_emb + alpha.item() * delta_emb
            interp_emb = interp_emb.detach().requires_grad_(True)

            def _make_hook(emb_tensor):
                def _fwd_hook(module, inp, output):
                    return emb_tensor

                return _fwd_hook

            handle = embedding_layer.register_forward_hook(_make_hook(interp_emb))
            try:
                _loss, logits, _pkv = model(input_ids)
            finally:
                handle.remove()

            scalar = logits[0, target_pos, target_token_idx]
            (grad,) = torch.autograd.grad(
                scalar, interp_emb, create_graph=False, retain_graph=False
            )
            sample_grads = sample_grads + grad.detach()

        avg_grads = sample_grads / n_steps
        ig = (avg_grads * delta_emb).sum(dim=-1).squeeze(0).abs()  # (T,)
        accumulated = accumulated + ig

    return accumulated / n_samples
