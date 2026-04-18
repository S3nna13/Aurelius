"""Gradient inversion attack for the Aurelius LLM research platform.

Reconstructs approximate input embeddings from observed parameter gradients
under the deep leakage from gradients threat model for federated learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ProxyModel(nn.Module):
    """Lightweight proxy used during the inversion optimisation loop.

    Accepts a pre-computed embedding tensor directly (bypassing the token
    lookup) and passes it through a small 2-layer MLP head to produce logits,
    keeping the attack computationally cheap.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        hidden = d_model * 2
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, vocab_size, bias=False)

    def forward(
        self,
        x_embed: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Forward pass directly from an embedding tensor.

        Args:
            x_embed: (batch, seq_len, d_model) float32 embeddings.
            labels:  (batch, seq_len) int64 target tokens, or None.

        Returns:
            Tuple of (loss, logits).  loss is None when labels is None.
        """
        # Mean-pool over the sequence dimension then project to vocab
        h = F.gelu(self.fc1(x_embed))   # (B, T, hidden)
        logits = self.fc2(h)             # (B, T, vocab_size)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
            )
        return loss, logits


class GradientInverter:
    """Reconstruct input embeddings from observed parameter gradients.

    The attack minimises the cosine / L2 distance between gradients produced
    by a dummy embedding and a target gradient vector.  Only the lightweight
    proxy model (embedding + 2-layer MLP head) is involved; the full
    transformer is not required for the inversion loop.

    Args:
        model:     AureliusTransformer (or any nn.Module with .embed and
                   .config attributes).  Used to extract the embedding layer
                   and model dimensions.
        loss_fn:   A callable ``(logits, labels) -> scalar`` used when
                   computing the reference gradients in ``compute_gradients``.
                   Ignored inside ``invert`` (cross-entropy is used there).
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module | None = None) -> None:
        self.model = model
        self.loss_fn = loss_fn

        cfg = model.config
        self.d_model: int = cfg.d_model
        self.vocab_size: int = cfg.vocab_size

        # Build a small proxy model for fast gradient computation
        self._proxy = _ProxyModel(self.d_model, self.vocab_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_gradients(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a flat gradient vector for a given (input, label) pair.

        The gradient is computed w.r.t. all parameters of *model* using the
        standard language-model cross-entropy loss.

        Args:
            model:     The nn.Module whose gradients are computed.
            input_ids: (1, seq_len) int64 token indices.
            labels:    (1, seq_len) int64 target token indices.

        Returns:
            1-D float32 tensor containing the concatenated, flattened
            gradient of every parameter in *model*.
        """
        model.zero_grad()

        # Forward pass — model must return (loss, logits, ...) tuple
        output = model(input_ids, labels=labels)
        if isinstance(output, (tuple, list)):
            loss = output[0]
        else:
            loss = output

        if loss is None:
            raise ValueError(
                "model.forward() returned None loss; pass labels to compute loss."
            )

        loss.backward()

        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
            else:
                grads.append(torch.zeros(p.numel(), dtype=torch.float32))

        model.zero_grad()
        return torch.cat(grads)

    def invert(
        self,
        grads_target: torch.Tensor,
        n_tokens: int,
        n_steps: int = 300,
        lr: float = 0.01,
        _return_loss_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
        """Reconstruct the input embedding that produced *grads_target*.

        Optimises a dummy embedding tensor ``x_dummy ∈ ℝ^(1, T, d_model)``
        to minimise the squared L2 distance between the gradients it induces
        and the target gradient vector.

        Only the lightweight proxy model is used, so the full transformer is
        never evaluated during the attack loop.

        Args:
            grads_target:         1-D flat gradient vector (as returned by
                                  ``compute_gradients``).
            n_tokens:             Number of token positions T to reconstruct.
            n_steps:              Number of Adam optimisation steps.
            lr:                   Learning rate for Adam.
            _return_loss_history: Internal flag.  When True the return value is
                                  ``(x_dummy, loss_history)`` where
                                  ``loss_history`` is a list of per-step attack
                                  losses (useful for convergence tests).

        Returns:
            Reconstructed embeddings of shape (1, n_tokens, d_model),
            float32.  When ``_return_loss_history`` is True, returns a tuple
            ``(embeddings, loss_history)``.
        """
        proxy = _ProxyModel(self.d_model, self.vocab_size)
        proxy.train()

        # Initialise dummy embedding with small random values
        x_dummy = nn.Parameter(
            torch.randn(1, n_tokens, self.d_model) * 0.01
        )
        optimizer = torch.optim.Adam([x_dummy], lr=lr)

        # Pre-build dummy labels (shifted cross-entropy requires T >= 2)
        dummy_labels = torch.zeros(1, n_tokens, dtype=torch.long)

        loss_history: list[float] = []

        for _ in range(n_steps):
            optimizer.zero_grad()
            proxy.zero_grad()

            # Compute gradients from dummy embedding
            loss_dummy, _ = proxy(x_dummy, labels=dummy_labels)
            if loss_dummy is None:
                raise RuntimeError("Proxy model returned None loss during inversion.")

            dummy_grads = torch.autograd.grad(
                loss_dummy,
                proxy.parameters(),
                create_graph=True,
                allow_unused=True,
            )

            dummy_flat = torch.cat([
                g.view(-1) if g is not None else torch.zeros(p.numel())
                for g, p in zip(dummy_grads, proxy.parameters())
            ])

            # Align sizes by padding/truncating to the shorter of the two
            min_len = min(dummy_flat.numel(), grads_target.numel())
            attack_loss = (
                (dummy_flat[:min_len] - grads_target[:min_len].detach()) ** 2
            ).sum()

            loss_history.append(attack_loss.item())

            attack_loss.backward()
            optimizer.step()

        result = x_dummy.detach()
        if _return_loss_history:
            return result, loss_history
        return result

    @staticmethod
    def reconstruction_error(
        x_true_embed: torch.Tensor,
        x_reconstructed: torch.Tensor,
    ) -> float:
        """Normalised L2 distance between true and reconstructed embeddings.

        Args:
            x_true_embed:    Ground-truth embedding tensor.
            x_reconstructed: Embedding tensor returned by ``invert``.

        Returns:
            Non-negative float.  0.0 when both tensors are identical.
        """
        diff = (x_true_embed.float() - x_reconstructed.float()).norm()
        denom = x_true_embed.float().norm()
        if denom == 0.0:
            return diff.item()
        return (diff / denom).item()
