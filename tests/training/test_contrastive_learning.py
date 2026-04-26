"""Tests for src/training/contrastive_learning.py (SimCSE-style)."""

from __future__ import annotations

import math

import torch

from src.training.contrastive_learning import (
    ContrastiveConfig,
    ContrastiveLearner,
    compute_alignment,
    compute_uniformity,
    cosine_similarity_matrix,
    hard_negative_loss,
    in_batch_negatives_loss,
    normalize_embeddings,
    nt_xent_loss,
    pool_hidden_states,
)

# ---------------------------------------------------------------------------
# Tiny dimensions used throughout
# ---------------------------------------------------------------------------
D = 16  # hidden / embedding dim
B = 4  # batch size
T = 6  # sequence length


def make_hidden(b: int = B, t: int = T, d: int = D) -> torch.Tensor:
    """Random (B, T, D) hidden states."""
    return torch.randn(b, t, d)


def make_mask(b: int = B, t: int = T, pad_last: int = 2) -> torch.Tensor:
    """Attention mask with the last ``pad_last`` tokens masked out."""
    mask = torch.ones(b, t, dtype=torch.long)
    if pad_last > 0:
        mask[:, t - pad_last :] = 0
    return mask


def tiny_encoder(ids: torch.Tensor) -> torch.Tensor:
    """Tiny deterministic-ish encoder: (B, T) -> (B, T, D)."""
    return torch.randn(ids.shape[0], ids.shape[1], D)


def make_ids(b: int = B, t: int = T) -> torch.Tensor:
    return torch.randint(0, 100, (b, t))


# ===========================================================================
# 1. ContrastiveConfig defaults
# ===========================================================================


def test_config_defaults():
    cfg = ContrastiveConfig()
    assert cfg.temperature == 0.05
    assert cfg.hard_negative_weight == 0.0
    assert cfg.pooling == "mean"


# ===========================================================================
# 2. pool_hidden_states — mean pooling shape
# ===========================================================================


def test_pool_mean_shape():
    hidden = make_hidden()
    out = pool_hidden_states(hidden, pooling="mean")
    assert out.shape == (B, D)


# ===========================================================================
# 3. pool_hidden_states — mean pooling with mask
# ===========================================================================


def test_pool_mean_with_mask():
    hidden = make_hidden()
    mask = make_mask()
    out = pool_hidden_states(hidden, attention_mask=mask, pooling="mean")
    assert out.shape == (B, D)
    # Manually verify first row
    real_tokens = mask[0].sum().item()
    expected = hidden[0, : int(real_tokens), :].mean(dim=0)
    assert torch.allclose(out[0], expected, atol=1e-5)


# ===========================================================================
# 4. pool_hidden_states — cls picks index 0
# ===========================================================================


def test_pool_cls_picks_index_0():
    hidden = make_hidden()
    out = pool_hidden_states(hidden, pooling="cls")
    assert out.shape == (B, D)
    assert torch.allclose(out, hidden[:, 0, :])


# ===========================================================================
# 5. pool_hidden_states — last token (with mask)
# ===========================================================================


def test_pool_last_with_mask():
    hidden = make_hidden()
    mask = make_mask(pad_last=2)  # real tokens: T-2 = 4
    out = pool_hidden_states(hidden, attention_mask=mask, pooling="last")
    assert out.shape == (B, D)
    # Last real token index is T - pad_last - 1 = 3
    expected = hidden[:, T - 2 - 1, :]
    assert torch.allclose(out, expected, atol=1e-5)


# ===========================================================================
# 6. pool_hidden_states — last token (no mask falls back to hidden[:, -1, :])
# ===========================================================================


def test_pool_last_no_mask():
    hidden = make_hidden()
    out = pool_hidden_states(hidden, pooling="last")
    assert torch.allclose(out, hidden[:, -1, :])


# ===========================================================================
# 7. normalize_embeddings — L2 norm ≈ 1
# ===========================================================================


def test_normalize_embeddings_unit_norm():
    emb = torch.randn(B, D) * 5  # arbitrary scale
    normed = normalize_embeddings(emb)
    norms = normed.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-6)


# ===========================================================================
# 8. cosine_similarity_matrix — shape (B, B)
# ===========================================================================


def test_cosine_similarity_matrix_shape():
    a = normalize_embeddings(torch.randn(B, D))
    b = normalize_embeddings(torch.randn(B, D))
    sim = cosine_similarity_matrix(a, b)
    assert sim.shape == (B, B)


# ===========================================================================
# 9. cosine_similarity_matrix — diagonal = 1 for identical normalized vecs
# ===========================================================================


def test_cosine_similarity_matrix_diagonal_ones():
    a = normalize_embeddings(torch.randn(B, D))
    sim = cosine_similarity_matrix(a, a)
    diag = torch.diagonal(sim)
    assert torch.allclose(diag, torch.ones(B), atol=1e-5)


# ===========================================================================
# 10. nt_xent_loss — returns a scalar tensor
# ===========================================================================


def test_nt_xent_loss_scalar():
    a = normalize_embeddings(torch.randn(B, D))
    sim = cosine_similarity_matrix(a, a)
    loss = nt_xent_loss(sim, temperature=0.05)
    assert loss.shape == ()


# ===========================================================================
# 11. nt_xent_loss — value > 0
# ===========================================================================


def test_nt_xent_loss_positive():
    a = normalize_embeddings(torch.randn(B, D))
    b = normalize_embeddings(torch.randn(B, D))
    sim = cosine_similarity_matrix(a, b)
    loss = nt_xent_loss(sim, temperature=0.05)
    assert loss.item() > 0


# ===========================================================================
# 12. in_batch_negatives_loss — returns a scalar
# ===========================================================================


def test_in_batch_negatives_loss_scalar():
    anchor = normalize_embeddings(torch.randn(B, D))
    positive = normalize_embeddings(torch.randn(B, D))
    loss = in_batch_negatives_loss(anchor, positive, temperature=0.05)
    assert loss.shape == ()
    assert loss.item() > 0


# ===========================================================================
# 13. hard_negative_loss — returns a scalar
# ===========================================================================


def test_hard_negative_loss_scalar():
    anchor = normalize_embeddings(torch.randn(B, D))
    positive = normalize_embeddings(torch.randn(B, D))
    negative = normalize_embeddings(torch.randn(B, D))
    loss = hard_negative_loss(anchor, positive, negative, temperature=0.05)
    assert loss.shape == ()
    assert loss.item() > 0


# ===========================================================================
# 14. ContrastiveLearner.encode — correct shape and unit norm
# ===========================================================================


def test_contrastive_learner_encode_shape_and_norm():
    cfg = ContrastiveConfig(pooling="mean")
    learner = ContrastiveLearner(encoder_fn=tiny_encoder, config=cfg)
    ids = make_ids()
    emb = learner.encode(ids)
    assert emb.shape == (B, D)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5)


# ===========================================================================
# 15. compute_loss — returns dict with correct keys
# ===========================================================================


def test_compute_loss_keys():
    cfg = ContrastiveConfig()
    learner = ContrastiveLearner(encoder_fn=tiny_encoder, config=cfg)
    anchor_ids = make_ids()
    positive_ids = make_ids()
    result = learner.compute_loss(anchor_ids, positive_ids)
    assert set(result.keys()) == {
        "loss",
        "similarity_matrix",
        "mean_positive_sim",
        "mean_negative_sim",
    }


# ===========================================================================
# 16. compute_loss — similarity_matrix has correct shape
# ===========================================================================


def test_compute_loss_similarity_matrix_shape():
    cfg = ContrastiveConfig()
    learner = ContrastiveLearner(encoder_fn=tiny_encoder, config=cfg)
    result = learner.compute_loss(make_ids(), make_ids())
    assert result["similarity_matrix"].shape == (B, B)


# ===========================================================================
# 17. compute_loss — with hard negatives, loss is still a scalar
# ===========================================================================


def test_compute_loss_with_hard_negatives():
    cfg = ContrastiveConfig()
    learner = ContrastiveLearner(encoder_fn=tiny_encoder, config=cfg)
    result = learner.compute_loss(make_ids(), make_ids(), negative_ids=make_ids())
    assert result["loss"].shape == ()
    assert result["loss"].item() > 0


# ===========================================================================
# 18. compute_alignment <= 0 for random embeddings
# ===========================================================================


def test_compute_alignment_le_zero():
    a = normalize_embeddings(torch.randn(B, D))
    b = normalize_embeddings(torch.randn(B, D))
    val = compute_alignment(a, b)
    assert isinstance(val, float)
    assert val <= 0.0


# ===========================================================================
# 19. compute_alignment == 0 for identical embeddings
# ===========================================================================


def test_compute_alignment_zero_for_identical():
    a = normalize_embeddings(torch.randn(B, D))
    val = compute_alignment(a, a)
    assert math.isclose(val, 0.0, abs_tol=1e-6)


# ===========================================================================
# 20. compute_uniformity — returns a float
# ===========================================================================


def test_compute_uniformity_returns_float():
    emb = normalize_embeddings(torch.randn(B, D))
    val = compute_uniformity(emb)
    assert isinstance(val, float)
    # Uniformity is log of a mean of positive values → can be any real; just check finite
    assert math.isfinite(val)
