"""Tests for cross_lingual_transfer module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.training.cross_lingual_transfer import (
    AlignmentLoss,
    CrossLingualConfig,
    CrossLingualEncoder,
    LanguageAdversary,
    LanguageIdentifier,
    ZeroShotTransferTrainer,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 2
N_LANGUAGES = 3
B = 2
T = 6
N_CLASSES = 4
TEMPERATURE = 0.07


def make_encoder() -> CrossLingualEncoder:
    return CrossLingualEncoder(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_languages=N_LANGUAGES,
    )


def make_trainer(encoder: CrossLingualEncoder) -> ZeroShotTransferTrainer:
    task_head = nn.Linear(D_MODEL, N_CLASSES)
    return ZeroShotTransferTrainer(
        encoder=encoder,
        task_head=task_head,
        lr=1e-3,
        alpha_align=0.5,
        beta_adv=0.1,
        temperature=TEMPERATURE,
    )


def rand_ids(b: int = B, t: int = T, vocab: int = VOCAB_SIZE) -> torch.Tensor:
    return torch.randint(0, vocab, (b, t))


def rand_langs(b: int = B, n: int = N_LANGUAGES) -> torch.Tensor:
    return torch.randint(0, n, (b,))


def rand_labels(b: int = B, n_classes: int = N_CLASSES) -> torch.Tensor:
    return torch.randint(0, n_classes, (b,))


# ---------------------------------------------------------------------------
# LanguageIdentifier tests
# ---------------------------------------------------------------------------


def test_language_identifier_forward_shape():
    """LanguageIdentifier forward output has shape [B, n_languages]."""
    model = LanguageIdentifier(D_MODEL, N_LANGUAGES)
    pooled = torch.randn(B, D_MODEL)
    logits = model(pooled)
    assert logits.shape == (B, N_LANGUAGES), f"Expected ({B}, {N_LANGUAGES}), got {logits.shape}"


def test_language_identifier_predict_valid_range():
    """LanguageIdentifier.predict returns lang_ids in [0, n_languages)."""
    model = LanguageIdentifier(D_MODEL, N_LANGUAGES)
    pooled = torch.randn(B, D_MODEL)
    lang_ids = model.predict(pooled)
    assert lang_ids.shape == (B,), f"Expected ({B},), got {lang_ids.shape}"
    assert (lang_ids >= 0).all() and (lang_ids < N_LANGUAGES).all(), (
        f"lang_ids out of range: {lang_ids}"
    )


def test_language_identifier_predict_shape():
    """LanguageIdentifier.predict returns a 1-D tensor of length B."""
    model = LanguageIdentifier(D_MODEL, N_LANGUAGES)
    pooled = torch.randn(B, D_MODEL)
    lang_ids = model.predict(pooled)
    assert lang_ids.ndim == 1
    assert lang_ids.shape[0] == B


# ---------------------------------------------------------------------------
# LanguageAdversary tests
# ---------------------------------------------------------------------------


def test_language_adversary_forward_shape():
    """LanguageAdversary forward output has shape [B, n_languages]."""
    model = LanguageAdversary(D_MODEL, N_LANGUAGES, lambda_adv=0.1)
    features = torch.randn(B, D_MODEL)
    logits = model(features)
    assert logits.shape == (B, N_LANGUAGES), f"Expected ({B}, {N_LANGUAGES}), got {logits.shape}"


def test_language_adversary_gradient_reversal():
    """LanguageAdversary gradient reversal: grad w.r.t. input has opposite sign vs. direct path."""
    lam = 1.0
    d = D_MODEL

    # Direct path: linear layer, gradient flows normally
    linear = nn.Linear(d, N_LANGUAGES, bias=False)
    inp_direct = torch.randn(B, d, requires_grad=True)
    out_direct = linear(inp_direct)
    loss_direct = out_direct.sum()
    loss_direct.backward()
    grad_direct = inp_direct.grad.clone()

    # GRL path: same weights, GRL in between
    from src.training.cross_lingual_transfer import _GradReversalFn

    inp_grl = inp_direct.detach().clone().requires_grad_(True)
    reversed_inp = _GradReversalFn.apply(inp_grl, lam)
    out_grl = linear(reversed_inp)
    loss_grl = out_grl.sum()
    loss_grl.backward()
    grad_grl = inp_grl.grad.clone()

    # GRL gradient should be negation of direct gradient
    assert torch.allclose(grad_grl, -grad_direct, atol=1e-5), (
        "GRL gradient should be the negative of the direct gradient"
    )


def test_language_adversary_lambda_zero_no_reversal():
    """With lambda_adv=0 the gradient reversal has no effect (multiplied by 0)."""
    from src.training.cross_lingual_transfer import _GradReversalFn

    x = torch.randn(B, D_MODEL, requires_grad=True)
    out = _GradReversalFn.apply(x, 0.0)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.zeros_like(x.grad))


# ---------------------------------------------------------------------------
# AlignmentLoss tests
# ---------------------------------------------------------------------------


def test_alignment_loss_mean_in_range():
    """mean_alignment result is in [0, 2] (cosine distance range)."""
    src = torch.randn(B, T, D_MODEL)
    tgt = torch.randn(B, T, D_MODEL)
    loss = AlignmentLoss.mean_alignment(src, tgt)
    assert loss.ndim == 0, "Should be scalar"
    assert 0.0 <= loss.item() <= 2.0, f"Loss {loss.item()} outside [0, 2]"


def test_alignment_loss_mean_zero_for_identical():
    """mean_alignment is 0 when src and tgt embeddings are identical."""
    emb = torch.randn(B, T, D_MODEL)
    loss = AlignmentLoss.mean_alignment(emb, emb)
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"


def test_alignment_loss_contrastive_positive():
    """contrastive_alignment returns a positive scalar."""
    src = torch.randn(B, D_MODEL)
    tgt = torch.randn(B, D_MODEL)
    loss = AlignmentLoss.contrastive_alignment(src, tgt, temperature=TEMPERATURE)
    assert loss.ndim == 0, "Should be scalar"
    assert loss.item() >= 0.0, f"Contrastive loss should be non-negative, got {loss.item()}"


def test_alignment_loss_contrastive_identical_near_zero():
    """contrastive_alignment is near 0 when src and tgt embeddings are identical."""
    # Identical embeddings → diagonal scores dominate → loss approaches 0
    emb = torch.randn(B, D_MODEL)
    loss = AlignmentLoss.contrastive_alignment(emb, emb, temperature=TEMPERATURE)
    assert loss.item() < 0.5, (
        f"Contrastive loss with identical embeddings should be near 0, got {loss.item()}"
    )


def test_alignment_loss_word_alignment_attn_shape():
    """word_alignment returns attention tensor of shape [B, T_s, T_t]."""
    T_s, T_t = 5, 7
    src = torch.randn(B, T_s, D_MODEL)
    tgt = torch.randn(B, T_t, D_MODEL)
    attn, loss = AlignmentLoss.word_alignment(src, tgt)
    assert attn.shape == (B, T_s, T_t), f"Expected ({B}, {T_s}, {T_t}), got {attn.shape}"


def test_alignment_loss_word_alignment_attn_sums_to_one():
    """word_alignment attention rows sum to 1 (softmax)."""
    T_s, T_t = 4, 6
    src = torch.randn(B, T_s, D_MODEL)
    tgt = torch.randn(B, T_t, D_MODEL)
    attn, _ = AlignmentLoss.word_alignment(src, tgt)
    row_sums = attn.sum(dim=-1)  # [B, T_s]
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "Attention rows should sum to 1"
    )


def test_alignment_loss_word_alignment_loss_positive():
    """word_alignment entropy loss is non-negative."""
    src = torch.randn(B, T, D_MODEL)
    tgt = torch.randn(B, T, D_MODEL)
    _, loss = AlignmentLoss.word_alignment(src, tgt)
    assert loss.ndim == 0
    assert loss.item() >= 0.0, f"Entropy loss should be non-negative, got {loss.item()}"


# ---------------------------------------------------------------------------
# CrossLingualEncoder tests
# ---------------------------------------------------------------------------


def test_encoder_token_repr_shape():
    """CrossLingualEncoder forward returns token_repr of shape [B, T, d_model]."""
    enc = make_encoder()
    ids = rand_ids()
    langs = rand_langs()
    token_repr, pooled = enc(ids, langs)
    assert token_repr.shape == (B, T, D_MODEL), (
        f"Expected ({B}, {T}, {D_MODEL}), got {token_repr.shape}"
    )


def test_encoder_pooled_shape():
    """CrossLingualEncoder forward returns pooled of shape [B, d_model]."""
    enc = make_encoder()
    ids = rand_ids()
    langs = rand_langs()
    _, pooled = enc(ids, langs)
    assert pooled.shape == (B, D_MODEL), f"Expected ({B}, {D_MODEL}), got {pooled.shape}"


def test_encoder_different_lang_ids_produce_different_outputs():
    """Different lang_ids should produce different pooled representations."""
    enc = make_encoder()
    ids = rand_ids()

    lang_a = torch.zeros(B, dtype=torch.long)
    lang_b = torch.ones(B, dtype=torch.long)

    _, pooled_a = enc(ids, lang_a)
    _, pooled_b = enc(ids, lang_b)

    assert not torch.allclose(pooled_a, pooled_b), (
        "Different language ids should produce different encoder outputs"
    )


def test_encoder_adversary_present():
    """CrossLingualEncoder has an adversary attribute of type LanguageAdversary."""
    enc = make_encoder()
    assert hasattr(enc, "adversary"), "Encoder should have adversary attribute"
    assert isinstance(enc.adversary, LanguageAdversary), (
        "encoder.adversary should be LanguageAdversary"
    )


# ---------------------------------------------------------------------------
# ZeroShotTransferTrainer tests
# ---------------------------------------------------------------------------


def test_trainer_train_step_finite_losses():
    """ZeroShotTransferTrainer train_step returns finite total_loss."""
    enc = make_encoder()
    trainer = make_trainer(enc)

    src_ids = rand_ids()
    src_lang = rand_langs()
    tgt_ids = rand_ids()
    tgt_lang = rand_langs()
    labels = rand_labels()

    total_loss, loss_dict = trainer.train_step(src_ids, src_lang, tgt_ids, tgt_lang, labels)
    assert math.isfinite(total_loss.item()), f"total_loss is not finite: {total_loss.item()}"


def test_trainer_loss_dict_keys():
    """ZeroShotTransferTrainer train_step loss_dict has expected keys."""
    enc = make_encoder()
    trainer = make_trainer(enc)

    src_ids = rand_ids()
    src_lang = rand_langs()
    tgt_ids = rand_ids()
    tgt_lang = rand_langs()
    labels = rand_labels()

    _, loss_dict = trainer.train_step(src_ids, src_lang, tgt_ids, tgt_lang, labels)
    expected_keys = {"task", "align", "adv", "total"}
    assert set(loss_dict.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(loss_dict.keys())}"
    )


def test_trainer_loss_dict_values_finite():
    """All values in loss_dict are finite floats."""
    enc = make_encoder()
    trainer = make_trainer(enc)

    src_ids = rand_ids()
    src_lang = rand_langs()
    tgt_ids = rand_ids()
    tgt_lang = rand_langs()
    labels = rand_labels()

    _, loss_dict = trainer.train_step(src_ids, src_lang, tgt_ids, tgt_lang, labels)
    for key, val in loss_dict.items():
        assert math.isfinite(val), f"loss_dict['{key}'] = {val} is not finite"


def test_trainer_evaluate_transfer_in_range():
    """ZeroShotTransferTrainer.evaluate_transfer returns a float in [0, 1]."""
    enc = make_encoder()
    trainer = make_trainer(enc)

    src_ids = rand_ids()
    src_lang = rand_langs()
    tgt_ids = rand_ids()
    tgt_lang = rand_langs()
    labels = rand_labels()

    acc = trainer.evaluate_transfer(src_ids, src_lang, tgt_ids, tgt_lang, labels)
    assert isinstance(acc, float), f"evaluate_transfer should return float, got {type(acc)}"
    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} not in [0, 1]"


# ---------------------------------------------------------------------------
# CrossLingualConfig tests
# ---------------------------------------------------------------------------


def test_cross_lingual_config_defaults():
    """CrossLingualConfig has correct default values."""
    cfg = CrossLingualConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.n_languages == 3
    assert cfg.lambda_adv == 0.1
    assert cfg.alpha_align == 0.5
    assert cfg.beta_adv == 0.1
    assert cfg.lr == 1e-4
    assert cfg.temperature == 0.07


def test_cross_lingual_config_custom():
    """CrossLingualConfig accepts custom values."""
    cfg = CrossLingualConfig(d_model=64, n_languages=10, lr=5e-4)
    assert cfg.d_model == 64
    assert cfg.n_languages == 10
    assert cfg.lr == 5e-4
