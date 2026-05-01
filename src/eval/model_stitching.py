"""Model stitching and representational similarity analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StitchConfig:
    """Configuration for model stitching."""

    stitch_layer: int = 1  # which layer index to stitch at
    use_affine: bool = True  # use affine transform (Linear) vs identity
    freeze_bottom: bool = True  # freeze layers below stitch point
    freeze_top: bool = False  # freeze layers above stitch point


def centered_kernel_alignment(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA similarity between representation matrices.

    Args:
        X: (n, d1) representation matrix.
        Y: (n, d2) representation matrix.

    Returns:
        CKA similarity in [0, 1].
    """
    n = X.shape[0]

    # Gram matrices
    K = X @ X.T  # (n, n)
    L = Y @ Y.T  # (n, n)

    # Centering matrix H = I - 11^T / n
    H = (
        torch.eye(n, dtype=X.dtype, device=X.device)
        - torch.ones(n, n, dtype=X.dtype, device=X.device) / n
    )

    # HSIC(K, L) = trace(K H L H) / (n-1)^2
    KH = K @ H
    LH = L @ H
    hsic_kl = torch.trace(KH @ LH) / ((n - 1) ** 2)
    hsic_kk = torch.trace(KH @ KH) / ((n - 1) ** 2)
    hsic_ll = torch.trace(LH @ LH) / ((n - 1) ** 2)

    denom = (hsic_kk * hsic_ll).clamp(min=0.0).sqrt()
    if denom < 1e-10:
        return 0.0

    cka = (hsic_kl / denom).item()
    return float(max(0.0, min(1.0, cka)))


def procrustes_similarity(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Orthogonal Procrustes similarity between representation matrices.

    Finds rotation R minimizing ||X - YR||_F.
    R = V U^T from SVD of X^T Y = U S V^T.

    Args:
        X: (n, d) representation matrix.
        Y: (n, d) representation matrix (same n).

    Returns:
        Normalized similarity in [0, 1].
    """
    # SVD of X^T Y
    M = X.T @ Y  # (d1, d2)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    V = Vh.T

    # Optimal rotation
    R = V @ U.T  # (d2, d1)

    # Frobenius norms
    aligned = Y @ R  # (n, d1)
    diff_norm = torch.linalg.norm(X - aligned, ord="fro")
    x_norm = torch.linalg.norm(X, ord="fro")
    y_norm = torch.linalg.norm(Y, ord="fro")

    similarity = 1.0 - diff_norm / (x_norm + y_norm + 1e-8)
    return float(similarity.item())


def _is_transformer_block(module: nn.Module) -> bool:
    """Heuristic: detect TransformerBlock-like layers by class name."""
    cls_name = type(module).__name__
    return "TransformerBlock" in cls_name or "Block" in cls_name and "Transformer" in cls_name


class ActivationCollector:
    """Collects intermediate activations from a model via forward hooks.

    Usage (context manager):
        with ActivationCollector(model) as collector:
            acts = collector.collect(input_ids)

    Or manually:
        collector = ActivationCollector(model)
        collector.__enter__()
        acts = collector.collect(input_ids)
        collector.__exit__(None, None, None)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # output may be tuple (x, kv) for TransformerBlock
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            # Mean pool over seq dim: (B, T, D) -> (B, D)
            if act.dim() == 3:
                act = act.mean(dim=1)
            self.activations[name] = act.detach()

        return hook

    def __enter__(self) -> ActivationCollector:
        self._hooks = []
        for name, module in self.model.named_modules():
            if _is_transformer_block(module):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def collect(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run forward pass and return collected activations (mean-pooled over seq).

        Each activation has shape (B, D).
        """
        self.clear()
        with torch.no_grad():
            self.model(input_ids)
        return dict(self.activations)

    def clear(self) -> None:
        """Reset activations dict."""
        self.activations = {}


def _get_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Extract TransformerBlock children from model.layers or direct children."""
    # Try model.layers first (AureliusTransformer)
    if hasattr(model, "layers"):
        layers = model.layers
        if hasattr(layers, "__iter__"):
            blocks = [m for m in layers if _is_transformer_block(m)]
            if blocks:
                return blocks

    # Fall back: top-level children that are TransformerBlock-like
    blocks = []
    for child in model.children():
        if _is_transformer_block(child):
            blocks.append(child)
    return blocks


def _get_embedding(model: nn.Module) -> nn.Module:
    """Find the token embedding from model."""
    for attr in ("embed", "token_embedding", "embedding", "wte"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(f"Cannot find embedding in {type(model).__name__}")


class StitchedModel(nn.Module):
    """Combines bottom layers from model_a and top layers from model_b.

    Architecture:
        embed_a -> bottom_layers (from model_a) -> stitching_layer ->
        top_layers (from model_b) -> norm_b -> head_b
    """

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        config: StitchConfig,
    ) -> None:
        super().__init__()

        self.stitch_config = config

        # Extract components from model_a
        self.embed = _get_embedding(model_a)
        blocks_a = _get_transformer_blocks(model_a)
        blocks_b = _get_transformer_blocks(model_b)

        k = config.stitch_layer
        self.bottom_layers = nn.ModuleList(blocks_a[:k])
        self.top_layers = nn.ModuleList(blocks_b[k:])

        # Detect d_model from embedding
        d_model_a = self.embed.embedding_dim
        if hasattr(model_b, "norm"):
            # Infer d_model_b from norm weight shape
            d_model_b = model_b.norm.weight.shape[0]
        else:
            d_model_b = d_model_a

        if config.use_affine:
            self.stitching_layer: nn.Module = nn.Linear(d_model_a, d_model_b, bias=True)
        else:
            self.stitching_layer = nn.Identity()

        # Get final norm and head from model_b
        if hasattr(model_b, "norm"):
            self.norm = model_b.norm
        else:
            self.norm = nn.Identity()

        if hasattr(model_b, "lm_head"):
            self.head = model_b.lm_head
        elif hasattr(model_b, "output_projection"):
            self.head = model_b.output_projection
        else:
            raise AttributeError(f"Cannot find lm_head in {type(model_b).__name__}")

        # Store freqs_cis from model_a (for bottom) and model_b (for top)
        # We store references; they'll be accessed at forward time
        self._model_a = model_a
        self._model_b = model_b

        # Freeze as configured
        if config.freeze_bottom:
            for p in self.embed.parameters():
                p.requires_grad_(False)
            for p in self.bottom_layers.parameters():
                p.requires_grad_(False)

        if config.freeze_top:
            for p in self.top_layers.parameters():
                p.requires_grad_(False)
            for p in self.norm.parameters():
                p.requires_grad_(False)
            for p in self.head.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through stitched model.

        Args:
            input_ids: (B, T) token indices.

        Returns:
            logits: (B, T, V).
        """
        B, S = input_ids.shape

        # Get RoPE frequencies from model_a for bottom, model_b for top
        freqs_a = self._model_a.freqs_cis[:S]
        freqs_b = self._model_b.freqs_cis[:S]

        # Embeddings from model_a
        x = self.embed(input_ids)

        # Bottom layers from model_a
        for layer in self.bottom_layers:
            x, _, _ = layer(x, freqs_a)

        # Stitching layer (Linear or Identity)
        x = self.stitching_layer(x)

        # Top layers from model_b
        for layer in self.top_layers:
            x, _, _ = layer(x, freqs_b)

        # Final norm + head from model_b
        x = self.norm(x)
        logits = self.head(x)

        return logits


def compare_model_representations(
    model_a: nn.Module,
    model_b: nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    """Compare representations layer by layer using CKA and Procrustes.

    Args:
        model_a: First model.
        model_b: Second model.
        input_ids: (B, T) input token indices.

    Returns:
        Dict with keys like "layer_0_cka", "layer_0_procrustes", ...
    """
    with ActivationCollector(model_a) as col_a:
        acts_a = col_a.collect(input_ids)

    with ActivationCollector(model_b) as col_b:
        acts_b = col_b.collect(input_ids)

    results: dict[str, float] = {}

    keys_a = list(acts_a.keys())
    keys_b = list(acts_b.keys())
    n_layers = min(len(keys_a), len(keys_b))

    for i in range(n_layers):
        X = acts_a[keys_a[i]]
        Y = acts_b[keys_b[i]]

        # If d_model differs, truncate to smaller dim for Procrustes
        d = min(X.shape[-1], Y.shape[-1])

        cka_val = centered_kernel_alignment(X, Y)
        proc_val = procrustes_similarity(X[..., :d], Y[..., :d])

        results[f"layer_{i}_cka"] = cka_val
        results[f"layer_{i}_procrustes"] = proc_val

    return results
