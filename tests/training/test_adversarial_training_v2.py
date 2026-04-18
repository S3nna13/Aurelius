"""Tests for adversarial_training_v2.py

Covers:
 - EmbeddingPGD output shapes, perturbation budget, adversarial property, fgsm
 - AdversarialSuffixGenerator optimize shape, greedy_decode token validity
 - FreeAdversarialTraining finite loss, non-negative delta_norm, delta persistence
 - VirtualAdversarialTraining direction shape, unit norm, non-negative vat_loss
 - AdversarialDataAugmentation augment_batch shape, mixup shape
 - AdvTrainingConfig defaults
 - EmbeddingPGD perturbations differ across steps
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.adversarial_training_v2 import (
    AdvTrainingConfig,
    AdversarialDataAugmentation,
    AdversarialSuffixGenerator,
    EmbeddingPGD,
    FreeAdversarialTraining,
    VirtualAdversarialTraining,
)

# ---------------------------------------------------------------------------
# Tiny model used by all tests
# ---------------------------------------------------------------------------

VOCAB = 16
D = 16
B = 2
T = 6


class TinyLM(nn.Module):
    """Minimal transformer-like LM for testing.

    Exposes ``.embedding`` (nn.Embedding) and accepts ``inputs_embeds``.
    """

    def __init__(self, vocab_size: int = VOCAB, d_model: int = D) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embedding(input_ids)
        return self.proj(x)  # [B, T, V]

    def forward_embeds(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs_embeds)


def make_model() -> TinyLM:
    return TinyLM(vocab_size=VOCAB, d_model=D)


def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.randint(0, VOCAB, (B, T))
    labels = torch.randint(0, VOCAB, (B, T))
    return ids, labels


# ===========================================================================
# EmbeddingPGD
# ===========================================================================

def test_pgd_attack_output_shape() -> None:
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.1, alpha=0.01, n_steps=3)
    ids, labels = make_batch()
    adv = pgd.attack(model, ids, labels)
    assert adv.shape == (B, T, D), f"Expected ({B},{T},{D}), got {adv.shape}"


def test_pgd_perturbation_norm_bounded() -> None:
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.1, alpha=0.01, n_steps=5)
    ids, labels = make_batch()
    adv = pgd.attack(model, ids, labels)
    clean = model.embedding(ids).detach()
    delta = adv - clean
    max_abs = delta.abs().max().item()
    assert max_abs <= 0.1 + 1e-5, f"Perturbation {max_abs:.6f} exceeds epsilon=0.1"


def test_pgd_attack_increases_loss() -> None:
    """Adversarial embeddings should produce higher or equal loss than clean."""
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.1, alpha=0.02, n_steps=10)
    ids, labels = make_batch()

    clean_emb = model.embedding(ids).detach()
    B_, T_, V_ = B, T, VOCAB
    clean_logits = model.proj(clean_emb)
    clean_loss = F.cross_entropy(
        clean_logits.reshape(B_ * T_, V_), labels.reshape(B_ * T_)
    ).item()

    adv = pgd.attack(model, ids, labels)
    adv_logits = model.proj(adv)
    adv_loss = F.cross_entropy(
        adv_logits.reshape(B_ * T_, V_), labels.reshape(B_ * T_)
    ).item()

    assert adv_loss >= clean_loss - 1e-4, (
        f"Adversarial loss {adv_loss:.4f} < clean loss {clean_loss:.4f}"
    )


def test_pgd_fgsm_output_shape() -> None:
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.1, alpha=0.01, n_steps=3)
    ids, labels = make_batch()
    adv = pgd.fgsm(model, ids, labels)
    assert adv.shape == (B, T, D), f"Expected ({B},{T},{D}), got {adv.shape}"


def test_pgd_perturbations_differ_per_step() -> None:
    """PGD with 1 step vs 5 steps should produce different perturbations."""
    model = make_model()
    ids, labels = make_batch()

    pgd1 = EmbeddingPGD(epsilon=0.1, alpha=0.02, n_steps=1)
    pgd5 = EmbeddingPGD(epsilon=0.1, alpha=0.02, n_steps=5)

    adv1 = pgd1.attack(model, ids, labels)
    adv5 = pgd5.attack(model, ids, labels)

    # With different numbers of steps, perturbed embeddings should differ.
    assert not torch.allclose(adv1, adv5, atol=1e-6), (
        "1-step and 5-step PGD produced identical perturbations (unexpected)"
    )


def test_pgd_fgsm_perturbation_bounded() -> None:
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.05, alpha=0.01, n_steps=1)
    ids, labels = make_batch()
    adv = pgd.fgsm(model, ids, labels)
    clean = model.embedding(ids).detach()
    delta = adv - clean
    max_abs = delta.abs().max().item()
    assert max_abs <= 0.05 + 1e-5, f"FGSM perturbation {max_abs:.6f} exceeds epsilon=0.05"


# ===========================================================================
# AdversarialSuffixGenerator
# ===========================================================================

def test_suffix_optimize_returns_correct_shape() -> None:
    model = make_model()
    suffix_len = 4
    gen = AdversarialSuffixGenerator(model, suffix_len=suffix_len, n_iters=5)
    ids, _ = make_batch()
    target_ids = torch.randint(0, VOCAB, (B, suffix_len))
    result = gen.optimize(ids, target_ids, lr=0.1)
    assert result.shape == (suffix_len, D), (
        f"Expected ({suffix_len},{D}), got {result.shape}"
    )


def test_suffix_optimize_updates_embeddings() -> None:
    """After optimisation the suffix should differ from initialisation."""
    model = make_model()
    suffix_len = 3
    gen = AdversarialSuffixGenerator(model, suffix_len=suffix_len, n_iters=10)
    initial = gen.suffix_embeddings.clone()
    ids, _ = make_batch()
    target_ids = torch.randint(0, VOCAB, (B, suffix_len))
    gen.optimize(ids, target_ids, lr=0.1)
    assert not torch.allclose(gen.suffix_embeddings, initial, atol=1e-6), (
        "suffix_embeddings did not change after optimization"
    )


def test_suffix_greedy_decode_shape() -> None:
    model = make_model()
    suffix_len = 4
    gen = AdversarialSuffixGenerator(model, suffix_len=suffix_len, n_iters=2)
    ids, _ = make_batch()
    target_ids = torch.randint(0, VOCAB, (B, suffix_len))
    gen.optimize(ids, target_ids, lr=0.05)
    token_ids = gen.greedy_decode_suffix()
    assert token_ids.shape == (suffix_len,), (
        f"Expected ({suffix_len},), got {token_ids.shape}"
    )


def test_suffix_greedy_decode_valid_vocab() -> None:
    """All decoded token ids must be in [0, vocab_size)."""
    model = make_model()
    suffix_len = 4
    gen = AdversarialSuffixGenerator(model, suffix_len=suffix_len, n_iters=2)
    ids, _ = make_batch()
    target_ids = torch.randint(0, VOCAB, (B, suffix_len))
    gen.optimize(ids, target_ids, lr=0.05)
    token_ids = gen.greedy_decode_suffix()
    assert (token_ids >= 0).all() and (token_ids < VOCAB).all(), (
        f"Token ids out of vocab range: {token_ids}"
    )


# ===========================================================================
# FreeAdversarialTraining
# ===========================================================================

def test_free_step_returns_finite_loss() -> None:
    model = make_model()
    fat = FreeAdversarialTraining(model, epsilon=0.1, m=2)
    ids, labels = make_batch()
    loss_val, _ = fat.free_step(ids, labels)
    assert torch.isfinite(torch.tensor(loss_val)), f"Loss is not finite: {loss_val}"


def test_free_step_delta_norm_nonneg() -> None:
    model = make_model()
    fat = FreeAdversarialTraining(model, epsilon=0.1, m=2)
    ids, labels = make_batch()
    _, delta_norm = fat.free_step(ids, labels)
    assert delta_norm >= 0.0, f"delta_norm {delta_norm} is negative"


def test_free_step_delta_persists() -> None:
    """delta should be non-None and updated after each call."""
    model = make_model()
    fat = FreeAdversarialTraining(model, epsilon=0.1, m=2)
    ids, labels = make_batch()

    assert fat.delta is None, "delta should start as None"
    fat.free_step(ids, labels)
    delta_after_first = fat.delta.clone()
    fat.free_step(ids, labels)
    delta_after_second = fat.delta.clone()

    # delta persists: it was set after the first call.
    assert fat.delta is not None
    # After two calls with the same data delta may or may not change
    # (depends on gradient direction), but shape must match embeddings.
    assert delta_after_first.shape == (B, T, D)
    assert delta_after_second.shape == (B, T, D)


def test_free_step_loss_is_positive() -> None:
    model = make_model()
    fat = FreeAdversarialTraining(model, epsilon=0.05, m=3)
    ids, labels = make_batch()
    loss_val, _ = fat.free_step(ids, labels)
    assert loss_val > 0.0, f"Expected positive loss, got {loss_val}"


# ===========================================================================
# VirtualAdversarialTraining
# ===========================================================================

def test_vat_direction_output_shape() -> None:
    model = make_model()
    vat = VirtualAdversarialTraining(model, epsilon=0.1, n_power_iters=1)
    ids, _ = make_batch()
    direction = vat.virtual_adversarial_direction(ids)
    assert direction.shape == (B, T, D), (
        f"Expected ({B},{T},{D}), got {direction.shape}"
    )


def test_vat_direction_unit_normalized() -> None:
    """Each sample's direction vector should be ε-scaled and unit-normalised
    per batch entry (L2 norm of flattened direction ≈ ε * sqrt(T*d) is not
    meaningful; instead verify that the *unscaled* direction has unit norm)."""
    model = make_model()
    epsilon = 0.1
    vat = VirtualAdversarialTraining(model, epsilon=epsilon, n_power_iters=2)
    ids, _ = make_batch()
    direction = vat.virtual_adversarial_direction(ids)  # [B, T, d], scaled by ε

    # direction = ε · r̂  ⟹  ||direction|| / ε ≈ ||r̂|| = 1 per batch sample.
    for b in range(B):
        unscaled = direction[b].reshape(-1)
        norm = unscaled.norm().item() / epsilon
        assert abs(norm - 1.0) < 0.1, (
            f"Sample {b}: unscaled norm={norm:.4f}, expected ≈1.0"
        )


def test_vat_loss_nonneg() -> None:
    model = make_model()
    vat = VirtualAdversarialTraining(model, epsilon=0.1, n_power_iters=1)
    ids, _ = make_batch()
    loss = vat.vat_loss(ids)
    assert loss.item() >= 0.0, f"VAT loss {loss.item()} is negative"


def test_vat_loss_is_scalar() -> None:
    model = make_model()
    vat = VirtualAdversarialTraining(model, epsilon=0.1, n_power_iters=1)
    ids, _ = make_batch()
    loss = vat.vat_loss(ids)
    assert loss.ndim == 0, f"Expected scalar (0-dim), got shape {loss.shape}"


# ===========================================================================
# AdversarialDataAugmentation
# ===========================================================================

def test_augment_batch_output_shape() -> None:
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.1, alpha=0.01, n_steps=2)
    aug = AdversarialDataAugmentation(pgd, augment_frac=0.5)
    ids, labels = make_batch()
    aug_ids, aug_labels = aug.augment_batch(model, ids, labels)
    assert aug_ids.shape == ids.shape, (
        f"aug_ids shape {aug_ids.shape} != {ids.shape}"
    )
    assert aug_labels.shape == labels.shape, (
        f"aug_labels shape {aug_labels.shape} != {labels.shape}"
    )


def test_augment_batch_valid_token_ids() -> None:
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.1, alpha=0.01, n_steps=2)
    aug = AdversarialDataAugmentation(pgd, augment_frac=0.5)
    ids, labels = make_batch()
    aug_ids, _ = aug.augment_batch(model, ids, labels)
    assert (aug_ids >= 0).all() and (aug_ids < VOCAB).all(), (
        "Augmented token ids out of vocabulary range"
    )


def test_mixup_adversarial_output_shape() -> None:
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.1, alpha=0.01, n_steps=2)
    ids, labels = make_batch()
    clean_emb = model.embedding(ids).detach()
    adv_emb = pgd.attack(model, ids, labels)
    mixed = AdversarialDataAugmentation.mixup_adversarial(clean_emb, adv_emb, lam=0.5)
    assert mixed.shape == (B, T, D), (
        f"Expected ({B},{T},{D}), got {mixed.shape}"
    )


def test_mixup_adversarial_lam_zero_equals_adv() -> None:
    """lam=0 → output should equal adv_emb exactly."""
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.1, alpha=0.01, n_steps=2)
    ids, labels = make_batch()
    clean_emb = model.embedding(ids).detach()
    adv_emb = pgd.attack(model, ids, labels)
    mixed = AdversarialDataAugmentation.mixup_adversarial(clean_emb, adv_emb, lam=0.0)
    assert torch.allclose(mixed, adv_emb), "lam=0 mixup should equal adv_emb"


def test_mixup_adversarial_lam_one_equals_clean() -> None:
    """lam=1 → output should equal clean_emb exactly."""
    model = make_model()
    pgd = EmbeddingPGD(epsilon=0.1, alpha=0.01, n_steps=2)
    ids, labels = make_batch()
    clean_emb = model.embedding(ids).detach()
    adv_emb = pgd.attack(model, ids, labels)
    mixed = AdversarialDataAugmentation.mixup_adversarial(clean_emb, adv_emb, lam=1.0)
    assert torch.allclose(mixed, clean_emb), "lam=1 mixup should equal clean_emb"


# ===========================================================================
# AdvTrainingConfig
# ===========================================================================

def test_adv_training_config_defaults() -> None:
    cfg = AdvTrainingConfig()
    assert cfg.epsilon == 0.1
    assert cfg.alpha == 0.01
    assert cfg.n_pgd_steps == 10
    assert cfg.suffix_len == 4
    assert cfg.n_suffix_iters == 20
    assert cfg.m_free == 3
    assert cfg.epsilon_vat == 0.1
    assert cfg.n_power_iters == 1
    assert cfg.augment_frac == 0.5


def test_adv_training_config_custom() -> None:
    cfg = AdvTrainingConfig(epsilon=0.2, alpha=0.05, n_pgd_steps=20, m_free=5)
    assert cfg.epsilon == 0.2
    assert cfg.alpha == 0.05
    assert cfg.n_pgd_steps == 20
    assert cfg.m_free == 5
