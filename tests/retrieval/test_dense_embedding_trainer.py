"""Unit tests for the contrastive dense-embedding trainer."""

from __future__ import annotations

import pytest
import torch

from src.retrieval.dense_embedding_trainer import (
    DenseEmbedder,
    EmbedderConfig,
    EmbeddingTrainer,
    InfoNCELoss,
)


def tiny_config(**overrides) -> EmbedderConfig:
    base = dict(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        max_seq_len=8,
        dropout=0.0,
        pad_token_id=0,
        embed_dim=16,
    )
    base.update(overrides)
    return EmbedderConfig(**base)


def _random_ids(batch: int, seq: int, vocab: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # avoid pad (id 0) so every row has at least one real token
    return torch.randint(1, vocab, (batch, seq), generator=g)


def test_embedder_output_shape() -> None:
    cfg = tiny_config()
    emb = DenseEmbedder(cfg)
    ids = _random_ids(4, 8, cfg.vocab_size)
    out = emb(ids)
    assert out.shape == (4, cfg.embed_dim)


def test_embeddings_l2_normalized() -> None:
    cfg = tiny_config()
    emb = DenseEmbedder(cfg)
    ids = _random_ids(6, 8, cfg.vocab_size, seed=1)
    out = emb(ids)
    norms = out.norm(dim=-1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), f"norms were {norms}"


def test_gradient_flow_to_all_trainable_params() -> None:
    cfg = tiny_config()
    emb = DenseEmbedder(cfg)
    loss_fn = InfoNCELoss(temperature=0.1)
    ids_a = _random_ids(4, 8, cfg.vocab_size, seed=2)
    ids_p = _random_ids(4, 8, cfg.vocab_size, seed=3)
    a = emb(ids_a)
    p = emb(ids_p)
    loss = loss_fn(a, p)
    loss.backward()
    missing = []
    for name, param in emb.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
        elif torch.all(param.grad == 0):
            # padding row of token embedding is expected to be zero-grad
            if name == "token_embedding.weight":
                continue
            missing.append(f"{name}:all-zero")
    assert not missing, f"params without gradient: {missing}"


def test_infonce_identical_anchor_positive_low_loss() -> None:
    torch.manual_seed(0)
    loss_fn = InfoNCELoss(temperature=0.05)
    x = torch.randn(8, 16)
    x = torch.nn.functional.normalize(x, dim=-1)
    loss = loss_fn(x, x)
    assert loss.item() < 0.1, f"expected tiny loss, got {loss.item()}"


def test_infonce_random_pairs_higher_than_identical() -> None:
    torch.manual_seed(0)
    loss_fn = InfoNCELoss(temperature=0.05)
    x = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    y = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    loss_identical = loss_fn(x, x).item()
    loss_random = loss_fn(x, y).item()
    assert loss_random > loss_identical


def test_infonce_symmetric_under_swap() -> None:
    torch.manual_seed(0)
    loss_fn = InfoNCELoss(temperature=0.1)
    a = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
    p = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
    l_ap = loss_fn(a, p).item()
    l_pa = loss_fn(p, a).item()
    assert abs(l_ap - l_pa) < 1e-6


def test_temperature_scaling_softer_lower_loss() -> None:
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    p = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    low = InfoNCELoss(temperature=0.01)(a, p).item()
    high = InfoNCELoss(temperature=1.0)(a, p).item()
    # Higher temperature -> softer distribution -> lower CE on random pairs
    assert high < low


def test_attention_mask_excludes_pad_positions() -> None:
    cfg = tiny_config()
    emb = DenseEmbedder(cfg)
    emb.train(False)
    torch.manual_seed(0)
    real = torch.randint(1, cfg.vocab_size, (1, 4))
    # Row A: real tokens + pad padding.
    row_a = torch.cat([real, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    mask_a = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
    # Row B: same real tokens + DIFFERENT non-pad junk in masked positions.
    junk = torch.randint(1, cfg.vocab_size, (1, 4))
    row_b = torch.cat([real, junk], dim=1)
    mask_b = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        out_a = emb(row_a, mask_a)
        out_b = emb(row_b, mask_b)
    assert torch.allclose(out_a, out_b, atol=1e-5), (
        "attention_mask must fully exclude pad positions from pooling"
    )


def test_determinism_with_fixed_seed() -> None:
    cfg = tiny_config()
    torch.manual_seed(123)
    emb1 = DenseEmbedder(cfg)
    torch.manual_seed(123)
    emb2 = DenseEmbedder(cfg)
    ids = _random_ids(3, 8, cfg.vocab_size, seed=7)
    emb1.train(False)
    emb2.train(False)
    with torch.no_grad():
        assert torch.allclose(emb1(ids), emb2(ids), atol=0.0)


def test_train_step_reduces_loss() -> None:
    cfg = tiny_config()
    torch.manual_seed(0)
    emb = DenseEmbedder(cfg)
    opt = torch.optim.Adam(emb.parameters(), lr=1e-2)
    trainer = EmbeddingTrainer(emb, opt, InfoNCELoss(temperature=0.1))
    ids_a = _random_ids(8, 8, cfg.vocab_size, seed=11)
    ids_p = ids_a.clone()
    losses = []
    for _ in range(50):
        losses.append(trainer.train_step(ids_a, ids_p)["loss"])
    assert losses[-1] < losses[0] - 0.1, (
        f"training did not reduce loss: start={losses[0]} end={losses[-1]}"
    )


def test_infonce_batch_one_degenerate() -> None:
    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(1, 16), dim=-1)
    loss = InfoNCELoss(temperature=0.05)(x, x)
    assert abs(loss.item()) < 1e-6, f"batch=1 loss must be ~0, got {loss.item()}"


def test_no_nan_or_inf_in_forward_and_loss() -> None:
    cfg = tiny_config(dropout=0.1)
    emb = DenseEmbedder(cfg)
    loss_fn = InfoNCELoss(temperature=0.05)
    ids_a = _random_ids(4, 8, cfg.vocab_size, seed=21)
    ids_p = _random_ids(4, 8, cfg.vocab_size, seed=22)
    out_a = emb(ids_a)
    out_p = emb(ids_p)
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_p).all()
    loss = loss_fn(out_a, out_p)
    assert torch.isfinite(loss).all()


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        EmbedderConfig(vocab_size=0)
    with pytest.raises(ValueError):
        EmbedderConfig(d_model=15, n_heads=4)  # not divisible
    with pytest.raises(ValueError):
        EmbedderConfig(dropout=1.5)
    with pytest.raises(ValueError):
        InfoNCELoss(temperature=0.0)


def test_seq_len_exceeds_max_raises() -> None:
    cfg = tiny_config(max_seq_len=8)
    emb = DenseEmbedder(cfg)
    ids = _random_ids(2, 16, cfg.vocab_size)
    with pytest.raises(ValueError):
        emb(ids)
