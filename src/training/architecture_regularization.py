"""
Neural Architecture Regularization techniques for stable LLM training.

Implements:
  - SpectralNorm: spectral normalization via power iteration
  - OrthogonalRegularizer: orthogonal weight regularization
  - GradientPenalty: R1 and gradient-norm penalties
  - WeightConstraints: clipping, selective weight decay, nuclear norm
  - LipschitzConstraint: Lipschitz enforcement via SVD
  - RegularizedTrainer: training loop with regularization hooks
  - RegConfig: dataclass for hyperparameters
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# RegConfig
# ---------------------------------------------------------------------------


@dataclass
class RegConfig:
    """Hyperparameter configuration for regularized training."""

    lambda_orth: float = 1e-3
    lambda_gp: float = 10.0
    lambda_nuc: float = 1e-4
    max_weight_norm: float = 1.0
    k_lipschitz: float = 1.0
    n_power_iterations: int = 1
    lr: float = 1e-4


# ---------------------------------------------------------------------------
# SpectralNorm
# ---------------------------------------------------------------------------


class SpectralNorm:
    """
    Spectral normalization for nn.Linear layers.

    Replaces the weight W with W / sigma(W) where sigma is the largest
    singular value estimated via power iteration.
    """

    def __init__(self, n_power_iterations: int = 1) -> None:
        if n_power_iterations < 1:
            raise ValueError("n_power_iterations must be >= 1")
        self.n_power_iterations = n_power_iterations

    # ------------------------------------------------------------------
    def compute_sigma(
        self,
        W: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        n_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Power iteration to estimate the spectral norm (largest singular value).

        Args:
            W: weight matrix of shape (out, in)
            u: left singular vector estimate, shape (out,)
            v: right singular vector estimate, shape (in,)
            n_iters: number of power iterations

        Returns:
            (sigma, u_new, v_new)
        """
        u_hat = u
        v_hat = v
        for _ in range(n_iters):
            # v = W^T u / ||W^T u||
            v_hat = F.normalize(W.t().mv(u_hat), dim=0, eps=1e-12)
            # u = W v / ||W v||
            u_hat = F.normalize(W.mv(v_hat), dim=0, eps=1e-12)
        sigma = u_hat.dot(W.mv(v_hat))
        return sigma, u_hat, v_hat

    # ------------------------------------------------------------------
    def apply(self, module: nn.Linear) -> nn.Module:
        """
        Register u/v buffers and wrap the module's forward with spectral norm.

        Returns the module (mutated in-place with a wrapped forward).
        """
        W = module.weight.data  # (out_features, in_features)
        h, w = W.shape

        # Initialise with random unit vectors
        u = F.normalize(W.new_empty(h).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(W.new_empty(w).normal_(0, 1), dim=0, eps=1e-12)

        module.register_buffer("_sn_u", u)
        module.register_buffer("_sn_v", v)
        module._sn_n_iters = self.n_power_iterations

        # Capture self (SpectralNorm instance) in closure
        sn = self
        original_forward = module.forward

        def _forward_with_sn(x: torch.Tensor) -> torch.Tensor:
            W_raw = module.weight
            u_buf = module._sn_u
            v_buf = module._sn_v
            sigma, u_new, v_new = sn.compute_sigma(W_raw, u_buf, v_buf, module._sn_n_iters)
            # Update buffers in-place (no grad)
            module._sn_u.data.copy_(u_new.data)
            module._sn_v.data.copy_(v_new.data)
            W_hat = W_raw / (sigma + 1e-12)
            return F.linear(x, W_hat, module.bias)

        module.forward = _forward_with_sn  # type: ignore[method-assign]
        module._sn_original_forward = original_forward
        return module

    # ------------------------------------------------------------------
    def remove(self, module: nn.Module) -> None:
        """Restore the original forward and remove SN buffers."""
        if hasattr(module, "_sn_original_forward"):
            module.forward = module._sn_original_forward  # type: ignore[method-assign]
            del module._sn_original_forward
        for attr in ("_sn_u", "_sn_v", "_sn_n_iters"):
            if hasattr(module, attr):
                delattr(module, attr)


# ---------------------------------------------------------------------------
# OrthogonalRegularizer
# ---------------------------------------------------------------------------


class OrthogonalRegularizer:
    """
    Penalise deviation from orthogonality: ||W W^T - I||_F^2.
    """

    def __init__(self, lambda_orth: float = 1e-3) -> None:
        self.lambda_orth = lambda_orth

    # ------------------------------------------------------------------
    def loss(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute ||W W^T - I||_F^2 for a 2-D weight matrix.

        For weights with more rows than columns (or equal), uses W W^T.
        For tall/wide matrices the Frobenius norm is computed directly.
        """
        W = weight
        if W.dim() != 2:
            W = W.view(W.shape[0], -1)
        m, n = W.shape
        # Use the smaller dimension to keep cost O(min(m,n)^2 * max(m,n))
        if m <= n:
            gram = W.mm(W.t())  # (m, m)
            _I = torch.eye(m, device=W.device, dtype=W.dtype)
        else:
            gram = W.t().mm(W)  # (n, n)
            _I = torch.eye(n, device=W.device, dtype=W.dtype)
        diff = gram - _I
        return (diff * diff).sum()

    # ------------------------------------------------------------------
    def apply_to_model(self, model: nn.Module) -> torch.Tensor:
        """Sum orthogonal loss over all Linear layers in the model."""
        total = torch.tensor(0.0)
        first = True
        for module in model.modules():
            if isinstance(module, nn.Linear):
                lo = self.loss(module.weight)
                if first:
                    total = lo * self.lambda_orth
                    first = False
                else:
                    total = total + lo * self.lambda_orth
        if first:
            # No linear layers found – return zero on appropriate device
            device = (
                next(model.parameters()).device
                if len(list(model.parameters())) > 0
                else torch.device("cpu")
            )
            total = torch.zeros(1, device=device).squeeze()
        return total

    # ------------------------------------------------------------------
    @staticmethod
    def orthogonal_init(weight: torch.Tensor) -> torch.Tensor:
        """
        Initialise weight with (semi-)orthogonal matrix via QR decomposition.

        Returns a tensor of the same shape as `weight`.
        """
        rows, cols = weight.shape[0], weight.numel() // weight.shape[0]
        flat = weight.new_empty(max(rows, cols), min(rows, cols)).normal_(0, 1)
        Q, _ = torch.linalg.qr(flat)
        Q = Q[:rows, :cols]  # slice to match original shape
        return Q.view_as(weight)


# ---------------------------------------------------------------------------
# GradientPenalty
# ---------------------------------------------------------------------------


class GradientPenalty:
    """
    Gradient-based regularization penalties.
    """

    def __init__(self, lambda_gp: float = 10.0) -> None:
        self.lambda_gp = lambda_gp

    # ------------------------------------------------------------------
    def r1_penalty(
        self,
        model: nn.Module,
        real_ids: torch.Tensor,
        loss_fn,
    ) -> torch.Tensor:
        """
        R1 regularization: E[||∇_embed D(x)||^2] on real samples.

        Computes gradient of loss w.r.t. the embedding of real inputs.

        Args:
            model: must expose model.embed (nn.Embedding) or model.embedding
            real_ids: [B, T] integer token ids
            loss_fn: callable(model, input_ids) -> scalar loss

        Returns:
            scalar R1 penalty
        """
        # Find the embedding layer
        embed_layer = None
        for name in ("embed", "embedding", "tok_emb", "wte"):
            if hasattr(model, name):
                embed_layer = getattr(model, name)
                break
        if embed_layer is None:
            for module in model.modules():
                if isinstance(module, nn.Embedding):
                    embed_layer = module
                    break
        if embed_layer is None:
            raise RuntimeError("No embedding layer found in model for R1 penalty")

        # Get embeddings with gradient
        embed_weight = embed_layer.weight
        embed_out = embed_weight[real_ids]  # [B, T, d_model]
        embed_out.requires_grad_(True)

        # Compute loss using a hook that replaces embedding lookup
        loss = loss_fn(model, real_ids)

        grads = torch.autograd.grad(
            outputs=loss,
            inputs=embed_weight,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        grad = grads[0]
        if grad is None:
            device = real_ids.device
            return torch.zeros(1, device=device).squeeze() * self.lambda_gp
        penalty = (grad * grad).sum()
        return self.lambda_gp * penalty

    # ------------------------------------------------------------------
    def gradient_norm_penalty(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        loss_fn,
        threshold: float = 1.0,
    ) -> torch.Tensor:
        """
        Soft hinge penalty on the gradient norm of model parameters.

        Penalizes ||∇_θ L|| when it exceeds `threshold`.

        Args:
            model: nn.Module
            inputs: input tensor (passed to loss_fn)
            loss_fn: callable(model, inputs) -> scalar loss
            threshold: gradient norm threshold

        Returns:
            non-negative scalar penalty
        """
        loss = loss_fn(model, inputs)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        grad_norm_sq = torch.zeros(1, device=loss.device).squeeze()
        for g in grads:
            if g is not None:
                grad_norm_sq = grad_norm_sq + (g * g).sum()
        grad_norm = grad_norm_sq.sqrt()
        # Soft hinge: max(0, ||∇|| - threshold)^2
        penalty = F.relu(grad_norm - threshold) ** 2
        return self.lambda_gp * penalty


# ---------------------------------------------------------------------------
# WeightConstraints
# ---------------------------------------------------------------------------


class WeightConstraints:
    """
    Various weight constraint and regularization utilities.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def clip_weights(model: nn.Module, max_norm: float = 1.0) -> None:
        """Clip all weight tensors (not biases) to [-max_norm, max_norm]."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "bias" not in name:
                    param.clamp_(-max_norm, max_norm)

    # ------------------------------------------------------------------
    @staticmethod
    def weight_decay_selective(
        model: nn.Module,
        wd: float,
        exclude: list[str],
    ) -> torch.Tensor:
        """
        L2 weight decay, skipping parameters whose names contain any string in
        `exclude` (e.g. ["bias", "norm"]).

        Returns scalar regularization loss.
        """
        total = None
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            skip = any(ex in name for ex in exclude)
            if skip:
                continue
            term = (param * param).sum()
            total = term if total is None else total + term
        if total is None:
            return torch.zeros(1).squeeze()
        return wd * total

    # ------------------------------------------------------------------
    @staticmethod
    def nuclear_norm_regularizer(
        weight: torch.Tensor,
        lambda_nuc: float,
    ) -> torch.Tensor:
        """
        Nuclear norm regularization: lambda_nuc * ||W||_*

        The nuclear norm is the sum of singular values, computed via SVD.
        """
        W = weight
        if W.dim() != 2:
            W = W.view(W.shape[0], -1)
        # torch.linalg.svdvals returns singular values in descending order
        sv = torch.linalg.svdvals(W)
        nuclear_norm = sv.sum()
        return lambda_nuc * nuclear_norm


# ---------------------------------------------------------------------------
# LipschitzConstraint
# ---------------------------------------------------------------------------


class LipschitzConstraint:
    """
    Enforce a Lipschitz constant k on nn.Linear layers via SVD clipping.
    """

    def __init__(self, k: float = 1.0) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    # ------------------------------------------------------------------
    def enforce_via_clipping(self, module: nn.Linear) -> None:
        """
        Clip singular values of the weight matrix to [0, k].

        Reconstructs W = U * clip(S, 0, k) * V^T in-place.
        """
        W = module.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S_clipped = S.clamp(0.0, self.k)
        W_new = U.mm(torch.diag(S_clipped)).mm(Vh)
        module.weight.data.copy_(W_new)

    # ------------------------------------------------------------------
    def estimate_lipschitz(self, module: nn.Linear) -> float:
        """Return the spectral norm (largest singular value) of the weight."""
        W = module.weight.data
        sv = torch.linalg.svdvals(W)
        return float(sv[0].item())


# ---------------------------------------------------------------------------
# RegularizedTrainer
# ---------------------------------------------------------------------------


class RegularizedTrainer:
    """
    A training wrapper that adds orthogonal and weight-constraint regularization
    to a base language model.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        config: RegConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.orthogonal_reg = OrthogonalRegularizer(lambda_orth=config.lambda_orth)
        self.weight_constraints = WeightConstraints()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward + backward pass with regularization.

        Args:
            input_ids: [B, T] integer token ids
            labels: [B, T] integer token ids (shifted target)

        Returns:
            (task_loss, reg_loss) both scalar tensors
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass – model is expected to return logits [B, T, vocab]
        logits = self.model(input_ids)  # [B, T, vocab]

        B, T, V = logits.shape
        task_loss = F.cross_entropy(
            logits.view(B * T, V),
            labels.view(B * T),
            ignore_index=-100,
        )

        # Regularization
        reg_loss = self.orthogonal_reg.apply_to_model(self.model)

        total_loss = task_loss + reg_loss
        total_loss.backward()
        self.optimizer.step()

        return task_loss.detach(), reg_loss.detach()

    # ------------------------------------------------------------------
    def apply_constraints(self) -> None:
        """Apply post-step weight constraints (clipping)."""
        self.weight_constraints.clip_weights(self.model, max_norm=self.config.max_weight_norm)
        lc = LipschitzConstraint(k=self.config.k_lipschitz)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                lc.enforce_via_clipping(module)
