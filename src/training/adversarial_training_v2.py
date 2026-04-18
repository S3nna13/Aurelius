"""Adversarial Training for LLMs — v2.

Implements PGD attacks on token embeddings, adversarial suffix generation,
Free Adversarial Training, Virtual Adversarial Training (VAT), and
adversarial data augmentation for language models.

Classes
-------
EmbeddingPGD               – PGD / FGSM attacks in embedding space
AdversarialSuffixGenerator – Continuous suffix optimisation + greedy decode
FreeAdversarialTraining    – Free-AT (replay perturbation across mini-batches)
VirtualAdversarialTraining – VAT via power-iteration (unsupervised)
AdversarialDataAugmentation – Mix clean + adversarial batches; mixup
AdvTrainingConfig          – Dataclass of all hyperparameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AdvTrainingConfig:
    """Unified configuration for adversarial training components."""

    # PGD / FGSM
    epsilon: float = 0.1
    alpha: float = 0.01
    n_pgd_steps: int = 10

    # Adversarial suffix
    suffix_len: int = 4
    n_suffix_iters: int = 20

    # Free adversarial training
    m_free: int = 3

    # Virtual adversarial training
    epsilon_vat: float = 0.1
    n_power_iters: int = 1

    # Data augmentation
    augment_frac: float = 0.5


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_embedding_layer(model: nn.Module) -> nn.Embedding:
    """Return the embedding layer from *model*.

    Looks for attributes named ``embedding``, ``embed_tokens``, or ``wte``
    (common naming conventions).  Raises ``AttributeError`` if not found.
    """
    for attr in ("embedding", "embed_tokens", "wte"):
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, nn.Embedding):
                return layer
    raise AttributeError(
        "Cannot locate embedding layer on model.  "
        "Expose it as model.embedding (nn.Embedding)."
    )


def _forward_with_embeds(
    model: nn.Module,
    embeds: Tensor,
    labels: Tensor,
) -> Tensor:
    """Run a forward pass using pre-computed embeddings *embeds* and return
    the cross-entropy loss averaged over the batch.

    The model must accept ``inputs_embeds`` keyword or fall back to a
    ``forward_embeds`` method.  For test models we always expose
    ``forward_embeds``.
    """
    try:
        out = model(inputs_embeds=embeds)
    except TypeError:
        out = model.forward_embeds(embeds)

    # ``out`` may be a Tensor [B, T, V] or a tuple/object – normalise.
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "logits"):
        out = out.logits  # type: ignore[union-attr]

    # out: [B, T, V], labels: [B, T]
    B, T, V = out.shape
    loss = F.cross_entropy(out.reshape(B * T, V), labels.reshape(B * T))
    return loss


# ---------------------------------------------------------------------------
# EmbeddingPGD
# ---------------------------------------------------------------------------

class EmbeddingPGD:
    """Projected Gradient Descent (PGD) adversarial attack in embedding space.

    Parameters
    ----------
    epsilon : float
        L-inf perturbation budget.
    alpha : float
        Step size per PGD iteration.
    n_steps : int
        Number of PGD iterations.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        n_steps: int = 10,
    ) -> None:
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_steps = n_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attack(
        self,
        model: nn.Module,
        input_ids: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """PGD attack returning adversarial embeddings ``e + δ``.

        Algorithm
        ---------
        1. ``e = model.embedding(input_ids)``              – [B, T, d]
        2. ``δ ~ U(-ε, ε)``                               – initialise
        3. For *n_steps*: ``δ ← δ + α · sign(∇_δ loss)``; clip to [-ε, ε]
        4. Return ``e + δ``

        Returns
        -------
        Tensor of shape [B, T, d_model] – adversarial embeddings (detached).
        """
        embed_layer = _get_embedding_layer(model)

        with torch.no_grad():
            e = embed_layer(input_ids)  # [B, T, d]

        delta = torch.empty_like(e).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)

        for _ in range(self.n_steps):
            adv_emb = e.detach() + delta
            loss = _forward_with_embeds(model, adv_emb, labels)
            loss.backward()

            with torch.no_grad():
                grad_sign = delta.grad.sign()
                delta.data = delta.data + self.alpha * grad_sign
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)

            delta.grad.zero_()

        return (e.detach() + delta.detach()).detach()

    def fgsm(
        self,
        model: nn.Module,
        input_ids: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Single-step FGSM adversarial attack.

        Returns
        -------
        Tensor of shape [B, T, d_model] – adversarial embeddings.
        """
        embed_layer = _get_embedding_layer(model)

        with torch.no_grad():
            e = embed_layer(input_ids)  # [B, T, d]

        delta = torch.zeros_like(e, requires_grad=True)
        adv_emb = e.detach() + delta
        loss = _forward_with_embeds(model, adv_emb, labels)
        loss.backward()

        with torch.no_grad():
            adv_delta = self.epsilon * delta.grad.sign()

        return (e.detach() + adv_delta).detach()


# ---------------------------------------------------------------------------
# AdversarialSuffixGenerator
# ---------------------------------------------------------------------------

class AdversarialSuffixGenerator:
    """Optimise a continuous suffix in embedding space to maximise target loss.

    The suffix is stored as a learnable parameter ``suffix_embeddings`` of
    shape ``[suffix_len, d_model]`` and updated via gradient ascent.

    Parameters
    ----------
    model : nn.Module
        Language model that exposes an ``embedding`` attribute.
    suffix_len : int
        Number of suffix tokens to optimise.
    n_iters : int
        Number of gradient steps.
    """

    def __init__(
        self,
        model: nn.Module,
        suffix_len: int = 4,
        n_iters: int = 50,
    ) -> None:
        self.model = model
        self.suffix_len = suffix_len
        self.n_iters = n_iters

        embed_layer = _get_embedding_layer(model)
        d_model = embed_layer.embedding_dim
        self.suffix_embeddings: Tensor = torch.randn(suffix_len, d_model)

    # ------------------------------------------------------------------
    def optimize(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        lr: float = 0.1,
    ) -> Tensor:
        """Optimise ``suffix_embeddings`` to maximise loss on *target_ids*.

        Parameters
        ----------
        input_ids : Tensor [B, T]
        target_ids : Tensor [B, T_suffix] – labels for the suffix region
        lr : float

        Returns
        -------
        Tensor [suffix_len, d_model] – the optimised suffix embeddings.
        """
        embed_layer = _get_embedding_layer(self.model)

        with torch.no_grad():
            prefix_emb = embed_layer(input_ids)  # [B, T, d]

        B = prefix_emb.shape[0]
        d = self.suffix_embeddings.shape[-1]

        # Reinitialise suffix as a leaf tensor requiring grad.
        suffix = self.suffix_embeddings.clone().detach().requires_grad_(True)
        optim = torch.optim.Adam([suffix], lr=lr)

        for _ in range(self.n_iters):
            optim.zero_grad()
            # Expand suffix to batch dimension: [B, suffix_len, d]
            suffix_expanded = suffix.unsqueeze(0).expand(B, -1, -1)
            # Concatenate prefix + suffix embeddings: [B, T+suffix_len, d]
            full_emb = torch.cat([prefix_emb.detach(), suffix_expanded], dim=1)

            # Build labels: ignore prefix positions (-100), use target_ids for suffix.
            T = input_ids.shape[1]
            prefix_labels = torch.full(
                (B, T), fill_value=-100, dtype=torch.long, device=input_ids.device
            )
            full_labels = torch.cat([prefix_labels, target_ids], dim=1)  # [B, T+suffix_len]

            # We maximise loss (adversarial) → negate for gradient ascent.
            loss = _forward_with_embeds(self.model, full_emb, full_labels)
            (-loss).backward()  # gradient *ascent*
            optim.step()

        self.suffix_embeddings = suffix.detach()
        return self.suffix_embeddings

    # ------------------------------------------------------------------
    def greedy_decode_suffix(self) -> Tensor:
        """Round continuous suffix embeddings to nearest token ids.

        Uses cosine similarity against the full embedding weight matrix.

        Returns
        -------
        Tensor [suffix_len] of dtype long – nearest token indices.
        """
        embed_layer = _get_embedding_layer(self.model)
        W = embed_layer.weight  # [vocab_size, d_model]

        # Normalise both sides for cosine similarity.
        suffix_norm = F.normalize(self.suffix_embeddings, dim=-1)  # [suffix_len, d]
        W_norm = F.normalize(W, dim=-1)  # [V, d]

        # sim: [suffix_len, V]
        sim = suffix_norm @ W_norm.t()
        token_ids = sim.argmax(dim=-1)  # [suffix_len]
        return token_ids


# ---------------------------------------------------------------------------
# FreeAdversarialTraining
# ---------------------------------------------------------------------------

class FreeAdversarialTraining:
    """Free Adversarial Training (Free-AT).

    Reuses the perturbation ``delta`` from the previous mini-batch and updates
    it simultaneously with the model parameters (``m`` replay steps).

    Parameters
    ----------
    model : nn.Module
    epsilon : float
        L-inf perturbation budget.
    m : int
        Number of free replay steps per mini-batch.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        m: int = 3,
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.m = m
        # Accumulated perturbation; initialised lazily on first call.
        self.delta: Optional[Tensor] = None

    # ------------------------------------------------------------------
    def free_step(
        self,
        input_ids: Tensor,
        labels: Tensor,
    ) -> Tuple[float, float]:
        """One free adversarial training step (m replay sub-steps).

        Parameters
        ----------
        input_ids : [B, T]
        labels    : [B, T]

        Returns
        -------
        (loss_value, delta_norm) : Tuple[float, float]
            Mean loss across replay steps and L-inf norm of current delta.
        """
        embed_layer = _get_embedding_layer(self.model)
        with torch.no_grad():
            e = embed_layer(input_ids)  # [B, T, d]

        # Initialise or resize delta when batch shape changes.
        if self.delta is None or self.delta.shape != e.shape:
            self.delta = torch.zeros_like(e)

        total_loss = 0.0

        for _ in range(self.m):
            delta_var = self.delta.clone().detach().requires_grad_(True)
            adv_emb = e.detach() + delta_var
            loss = _forward_with_embeds(self.model, adv_emb, labels)
            loss.backward()
            total_loss += loss.item()

            with torch.no_grad():
                # Update delta (gradient ascent step).
                self.delta = self.delta + self.epsilon * delta_var.grad.sign()
                self.delta = self.delta.clamp(-self.epsilon, self.epsilon)

        mean_loss = total_loss / self.m
        delta_norm = float(self.delta.abs().max().item())
        return mean_loss, delta_norm

    # ------------------------------------------------------------------
    def train_epoch(
        self,
        data_iter: Iterator[Tuple[Tensor, Tensor]],
        optimizer: torch.optim.Optimizer,
    ) -> List[float]:
        """Train for one epoch using free adversarial training.

        Parameters
        ----------
        data_iter : iterable of (input_ids, labels) tensors
        optimizer : torch optimiser attached to *self.model*

        Returns
        -------
        List of per-batch loss values.
        """
        self.model.train()
        losses: List[float] = []

        for input_ids, labels in data_iter:
            optimizer.zero_grad()
            loss_val, _ = self.free_step(input_ids, labels)
            # Gradients were already computed inside free_step; step here.
            optimizer.step()
            losses.append(loss_val)

        return losses


# ---------------------------------------------------------------------------
# VirtualAdversarialTraining
# ---------------------------------------------------------------------------

class VirtualAdversarialTraining:
    """Virtual Adversarial Training (VAT) for LLMs.

    Computes the worst-case perturbation direction via power iteration and
    minimises the KL divergence between clean and perturbed predictions.

    Parameters
    ----------
    model : nn.Module
    epsilon : float
        L-2 perturbation magnitude for VAT.
    n_power_iters : int
        Number of power-iteration steps to approximate the adversarial direction.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        n_power_iters: int = 1,
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.n_power_iters = n_power_iters

    # ------------------------------------------------------------------
    def _model_logprobs(self, embeds: Tensor) -> Tensor:
        """Run model with *embeds* and return log-softmax over vocab.

        Returns
        -------
        Tensor [B, T, V]
        """
        try:
            out = self.model(inputs_embeds=embeds)
        except TypeError:
            out = self.model.forward_embeds(embeds)

        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "logits"):
            out = out.logits  # type: ignore[union-attr]

        return F.log_softmax(out, dim=-1)

    # ------------------------------------------------------------------
    def virtual_adversarial_direction(self, input_ids: Tensor) -> Tensor:
        """Approximate worst-case perturbation direction via power iteration.

        Algorithm
        ---------
        1. ``e = embedding(input_ids)``
        2. ``r ~ N(0, 1)``, normalise
        3. For *n_power_iters*:
           a. ``p = logprobs(e)``
           b. ``p_r = logprobs(e + ε · r̂)``
           c. Compute ``∇_r KL(p || p_r)``
           d. ``r = ∇_r / ||∇_r||``
        4. Return ``ε · r``

        Returns
        -------
        Tensor [B, T, d_model] – unit-scaled adversarial direction.
        """
        embed_layer = _get_embedding_layer(self.model)
        with torch.no_grad():
            e = embed_layer(input_ids)  # [B, T, d]

        # Random initialisation.
        r = torch.randn_like(e)
        r = F.normalize(r.view(r.shape[0], -1), dim=-1).view_as(r)

        # Clean log-probs (no grad).
        with torch.no_grad():
            log_p = self._model_logprobs(e)  # [B, T, V]
            p = log_p.exp()

        for _ in range(self.n_power_iters):
            r_var = r.clone().detach().requires_grad_(True)
            log_p_r = self._model_logprobs(e.detach() + r_var)  # [B, T, V]

            # KL(p || p_r) = Σ p · (log p - log p_r)
            kl = (p.detach() * (log_p.detach() - log_p_r)).sum(dim=-1).mean()
            kl.backward()

            with torch.no_grad():
                grad = r_var.grad
                r = F.normalize(grad.view(grad.shape[0], -1), dim=-1).view_as(grad)

        return (self.epsilon * r).detach()

    # ------------------------------------------------------------------
    def vat_loss(self, input_ids: Tensor) -> Tensor:
        """Compute VAT loss: KL divergence between clean and adversarially
        perturbed predictions.

        Returns
        -------
        Scalar Tensor – non-negative KL divergence.
        """
        embed_layer = _get_embedding_layer(self.model)
        with torch.no_grad():
            e = embed_layer(input_ids)

        # Adversarial direction.
        r_adv = self.virtual_adversarial_direction(input_ids)

        # Clean log-probs (no grad for the reference distribution).
        with torch.no_grad():
            log_p = self._model_logprobs(e)

        # Perturbed log-probs (with grad for learning).
        log_p_adv = self._model_logprobs(e + r_adv)

        p = log_p.exp()
        # KL(p_clean || p_adv) – mean over batch and sequence.
        kl = (p * (log_p - log_p_adv)).sum(dim=-1).mean()
        # Clamp to zero to guard against tiny negative values from float precision.
        return kl.clamp(min=0.0)


# ---------------------------------------------------------------------------
# AdversarialDataAugmentation
# ---------------------------------------------------------------------------

class AdversarialDataAugmentation:
    """Mix clean and adversarial examples for data augmentation.

    Parameters
    ----------
    pgd : EmbeddingPGD
        The PGD attacker to generate adversarial embeddings.
    augment_frac : float
        Fraction of the batch to replace with adversarial examples (0–1).
    """

    def __init__(
        self,
        pgd: EmbeddingPGD,
        augment_frac: float = 0.5,
    ) -> None:
        self.pgd = pgd
        self.augment_frac = augment_frac

    # ------------------------------------------------------------------
    def augment_batch(
        self,
        model: nn.Module,
        input_ids: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return augmented (input_ids, labels) with a mix of clean and PGD examples.

        ``augment_frac`` of the batch indices are replaced by PGD adversarial
        embeddings encoded back to the nearest token id (greedy).  The returned
        tensors have the *same shape* as the inputs.

        Parameters
        ----------
        model     : nn.Module
        input_ids : [B, T]
        labels    : [B, T]

        Returns
        -------
        (aug_input_ids, aug_labels) each of shape [B, T]
        """
        B = input_ids.shape[0]
        n_adv = max(1, int(B * self.augment_frac))

        adv_indices = torch.randperm(B)[:n_adv]

        # Get adversarial embeddings for the selected sub-batch.
        adv_input = input_ids[adv_indices]
        adv_labels = labels[adv_indices]
        adv_emb = self.pgd.attack(model, adv_input, adv_labels)  # [n_adv, T, d]

        # Project back to token ids via nearest embedding lookup.
        embed_layer = _get_embedding_layer(model)
        W = embed_layer.weight  # [V, d]
        W_norm = F.normalize(W, dim=-1)
        adv_emb_norm = F.normalize(adv_emb, dim=-1)  # [n_adv, T, d]
        # sim: [n_adv, T, V]
        sim = adv_emb_norm @ W_norm.t()
        adv_token_ids = sim.argmax(dim=-1)  # [n_adv, T]

        aug_input_ids = input_ids.clone()
        aug_labels = labels.clone()
        aug_input_ids[adv_indices] = adv_token_ids

        return aug_input_ids, aug_labels

    # ------------------------------------------------------------------
    @staticmethod
    def mixup_adversarial(
        clean_emb: Tensor,
        adv_emb: Tensor,
        lam: float = 0.5,
    ) -> Tensor:
        """Interpolate between clean and adversarial embeddings.

        Parameters
        ----------
        clean_emb : [B, T, d]
        adv_emb   : [B, T, d]
        lam       : float – mixing coefficient (0 = adv, 1 = clean)

        Returns
        -------
        Tensor [B, T, d]
        """
        return lam * clean_emb + (1.0 - lam) * adv_emb
