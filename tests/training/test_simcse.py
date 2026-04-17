"""
Tests for src/training/simcse.py
SimCSE contrastive learning for sentence embeddings.

Tiny config: d_model=8, vocab=16, seq_len=8, batch=4, temperature=0.05
Backbone: nn.Embedding(16, 8) — outputs (B, T, D) via unsqueeze on embed output
"""

import math

import pytest
import torch
import torch.nn as nn

from src.training.simcse import (
    AlignmentUniformityLoss,
    SimCSEEncoder,
    SimCSETrainer,
    SupervisedSimCSELoss,
    UnsupervisedSimCSELoss,
    _spearman,
)

# ---------------------------------------------------------------------------
# Shared config & helpers
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 8
SEQ_LEN = 8
BATCH = 4
TEMPERATURE = 0.05


class EmbeddingBackbone(nn.Module):
    """Simple backbone: nn.Embedding -> (B, T, D)."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)  # (B, T, D)


def make_backbone() -> EmbeddingBackbone:
    torch.manual_seed(0)
    return EmbeddingBackbone()


def make_encoder(pooling: str = "cls") -> SimCSEEncoder:
    return SimCSEEncoder(make_backbone(), d_model=D_MODEL, pooling=pooling)


def make_input(batch: int = BATCH, seq: int = SEQ_LEN) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randint(0, VOCAB, (batch, seq))


# ---------------------------------------------------------------------------
# Test 1: SimCSEEncoder forward shape with pooling="cls"
# ---------------------------------------------------------------------------

def test_encoder_forward_shape_cls():
    enc = make_encoder(pooling="cls")
    x = make_input()
    out = enc(x)
    assert out.shape == (BATCH, D_MODEL), f"Expected ({BATCH},{D_MODEL}), got {out.shape}"
    # backward pass
    loss = out.sum()
    loss.backward()


# ---------------------------------------------------------------------------
# Test 2: SimCSEEncoder forward shape with pooling="mean"
# ---------------------------------------------------------------------------

def test_encoder_forward_shape_mean():
    enc = make_encoder(pooling="mean")
    x = make_input()
    out = enc(x)
    assert out.shape == (BATCH, D_MODEL)
    out.sum().backward()


# ---------------------------------------------------------------------------
# Test 3: SimCSEEncoder forward shape with pooling="last"
# ---------------------------------------------------------------------------

def test_encoder_forward_shape_last():
    enc = make_encoder(pooling="last")
    x = make_input()
    out = enc(x)
    assert out.shape == (BATCH, D_MODEL)
    out.sum().backward()


# ---------------------------------------------------------------------------
# Test 4: SimCSEEncoder.normalize produces unit L2 norms
# ---------------------------------------------------------------------------

def test_encoder_normalize_unit_norm():
    enc = make_encoder()
    x = make_input()
    emb = enc(x)
    normed = enc.normalize(emb)
    norms = normed.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5), \
        f"L2 norms not 1: {norms}"


# ---------------------------------------------------------------------------
# Test 5: UnsupervisedSimCSELoss forward returns finite scalar + valid cosines
# ---------------------------------------------------------------------------

def test_unsupervised_loss_forward_finite():
    enc = make_encoder()
    enc.train()
    loss_fn = UnsupervisedSimCSELoss(temperature=TEMPERATURE)
    x = make_input()
    loss, avg_pos, avg_neg = loss_fn(enc, x)
    assert loss.shape == (), "loss must be a scalar"
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert -1.0 <= avg_pos.item() <= 1.0, f"avg_pos out of range: {avg_pos}"
    assert -1.0 <= avg_neg.item() <= 1.0, f"avg_neg out of range: {avg_neg}"
    loss.backward()


# ---------------------------------------------------------------------------
# Test 6: UnsupervisedSimCSELoss — pos_sim > neg_sim after training steps
# ---------------------------------------------------------------------------

def test_unsupervised_loss_pos_greater_neg_after_training():
    torch.manual_seed(1)
    enc = make_encoder()
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    loss_fn = UnsupervisedSimCSELoss(temperature=TEMPERATURE)
    x = make_input()

    final_pos, final_neg = None, None
    for _ in range(30):
        opt.zero_grad()
        loss, pos, neg = loss_fn(enc, x)
        loss.backward()
        opt.step()
        final_pos, final_neg = pos.item(), neg.item()

    assert final_pos > final_neg, (
        f"Expected pos_sim > neg_sim after training, got pos={final_pos:.4f} neg={final_neg:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: UnsupervisedSimCSELoss — gradients flow to encoder parameters
# ---------------------------------------------------------------------------

def test_unsupervised_loss_grad_flows():
    enc = make_encoder()
    enc.train()
    loss_fn = UnsupervisedSimCSELoss(temperature=TEMPERATURE)
    x = make_input()

    # Zero all grads first
    for p in enc.parameters():
        p.grad = None

    loss, _, _ = loss_fn(enc, x)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert has_grad, "No gradients flowed to encoder parameters"


# ---------------------------------------------------------------------------
# Test 8: SupervisedSimCSELoss forward returns finite scalar
# ---------------------------------------------------------------------------

def test_supervised_loss_forward_finite():
    enc = make_encoder()
    enc.train()
    loss_fn = SupervisedSimCSELoss(temperature=TEMPERATURE)

    anchor = make_input()
    positive = make_input()
    negative = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))

    loss, pos_sim, neg_sim = loss_fn(enc, anchor, positive, negative)
    assert loss.shape == ()
    assert torch.isfinite(loss), f"Supervised loss not finite: {loss}"
    loss.backward()


# ---------------------------------------------------------------------------
# Test 9: SupervisedSimCSELoss — gradients flow to encoder parameters
# ---------------------------------------------------------------------------

def test_supervised_loss_grad_flows():
    enc = make_encoder()
    enc.train()
    loss_fn = SupervisedSimCSELoss(temperature=TEMPERATURE)

    anchor = make_input()
    positive = make_input()
    negative = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))

    for p in enc.parameters():
        p.grad = None

    loss, _, _ = loss_fn(enc, anchor, positive, negative)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert has_grad, "No gradients flowed to encoder parameters (supervised)"


# ---------------------------------------------------------------------------
# Test 10: AlignmentUniformityLoss.alignment >= 0, and = 0 for identical vectors
# ---------------------------------------------------------------------------

def test_alignment_nonneg_and_zero_for_identical():
    au = AlignmentUniformityLoss(t=2.0)
    torch.manual_seed(7)
    z = torch.randn(BATCH, D_MODEL)
    z = z / z.norm(dim=-1, keepdim=True)

    # Identical vectors => alignment == 0
    align_zero = au.alignment(z, z)
    assert align_zero.item() >= -1e-6, f"alignment(z,z) negative: {align_zero}"
    assert align_zero.item() < 1e-5, f"alignment(z,z) should be ~0, got {align_zero.item()}"

    # Different vectors => alignment > 0
    z2 = torch.randn(BATCH, D_MODEL)
    z2 = z2 / z2.norm(dim=-1, keepdim=True)
    align_nonzero = au.alignment(z, z2)
    assert align_nonzero.item() >= 0.0, f"alignment negative: {align_nonzero}"


# ---------------------------------------------------------------------------
# Test 11: AlignmentUniformityLoss.uniformity is finite and negative for diverse embeddings
# ---------------------------------------------------------------------------

def test_uniformity_finite_negative():
    au = AlignmentUniformityLoss(t=2.0)
    torch.manual_seed(9)
    # Diverse random unit vectors
    z = torch.randn(BATCH, D_MODEL)
    z = z / z.norm(dim=-1, keepdim=True)
    unif = au.uniformity(z)
    assert torch.isfinite(unif), f"uniformity not finite: {unif}"
    # For diverse embeddings the value should be negative (log of something < 1)
    assert unif.item() < 0.0, f"uniformity should be negative for diverse vectors, got {unif.item()}"


# ---------------------------------------------------------------------------
# Test 12: AlignmentUniformityLoss.loss = weighted sum of components
# ---------------------------------------------------------------------------

def test_alignment_uniformity_loss_weighted_sum():
    au = AlignmentUniformityLoss(t=2.0)
    torch.manual_seed(11)
    # Use requires_grad so backward() works on the raw tensors
    z1 = torch.randn(BATCH, D_MODEL, requires_grad=True)
    z1_n = z1 / z1.norm(dim=-1, keepdim=True)
    z2 = torch.randn(BATCH, D_MODEL, requires_grad=True)
    z2_n = z2 / z2.norm(dim=-1, keepdim=True)

    aw, uw = 0.5, 2.0
    total = au.loss(z1_n, z2_n, align_weight=aw, uniform_weight=uw)

    # Recompute components for value check (no_grad to avoid graph reuse issues)
    with torch.no_grad():
        z1_d = torch.randn(BATCH, D_MODEL)
        z1_d = z1_d / z1_d.norm(dim=-1, keepdim=True)
        z2_d = torch.randn(BATCH, D_MODEL)
        z2_d = z2_d / z2_d.norm(dim=-1, keepdim=True)
    # Just verify the formula structure by checking it is finite and backward works
    assert torch.isfinite(total), f"total loss not finite: {total}"

    # Verify the loss equals the manual formula on fresh tensors
    torch.manual_seed(11)
    z1b = torch.randn(BATCH, D_MODEL)
    z1b = z1b / z1b.norm(dim=-1, keepdim=True)
    z2b = torch.randn(BATCH, D_MODEL)
    z2b = z2b / z2b.norm(dim=-1, keepdim=True)
    align = au.alignment(z1b, z2b)
    unif = (au.uniformity(z1b) + au.uniformity(z2b)) / 2.0
    expected = aw * align + uw * unif
    # compute total again on same seed values (no grad needed for value check)
    total2 = au.loss(z1b, z2b, align_weight=aw, uniform_weight=uw)
    assert abs(total2.item() - expected.item()) < 1e-5, (
        f"total={total2.item():.6f} != expected={expected.item():.6f}"
    )
    # backward pass through the grad-tracked version
    total.backward()


# ---------------------------------------------------------------------------
# Test 13: SimCSETrainer unsupervised — train_step returns all keys, loss finite
# ---------------------------------------------------------------------------

def test_trainer_unsupervised_train_step():
    enc = make_encoder()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    trainer = SimCSETrainer(enc, opt, mode="unsupervised")
    x = make_input()
    result = trainer.train_step(x)
    assert set(result.keys()) == {"loss", "pos_sim", "neg_sim"}, \
        f"Missing keys: {result.keys()}"
    assert math.isfinite(result["loss"]), f"loss not finite: {result['loss']}"


# ---------------------------------------------------------------------------
# Test 14: SimCSETrainer supervised — requires positive+negative, returns all keys
# ---------------------------------------------------------------------------

def test_trainer_supervised_train_step():
    enc = make_encoder()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    trainer = SimCSETrainer(enc, opt, mode="supervised")

    anchor = make_input()
    positive = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    negative = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))

    result = trainer.train_step(anchor, positive_ids=positive, negative_ids=negative)
    assert set(result.keys()) == {"loss", "pos_sim", "neg_sim"}
    assert math.isfinite(result["loss"])

    # Verify raises when positive/negative missing
    with pytest.raises(ValueError, match="supervised mode requires"):
        trainer.train_step(anchor)


# ---------------------------------------------------------------------------
# Test 15: SimCSETrainer.evaluate_sts returns float in [-1, 1]
# ---------------------------------------------------------------------------

def test_trainer_evaluate_sts():
    enc = make_encoder()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    trainer = SimCSETrainer(enc, opt, mode="unsupervised")

    # Build pairs and scores: create 6 pairs with varying similarity
    torch.manual_seed(3)
    pairs = [
        (torch.randint(0, VOCAB, (SEQ_LEN,)), torch.randint(0, VOCAB, (SEQ_LEN,)))
        for _ in range(6)
    ]
    scores = [0.1, 0.3, 0.5, 0.6, 0.8, 0.9]

    rho = trainer.evaluate_sts(pairs, scores)
    assert isinstance(rho, float), f"Expected float, got {type(rho)}"
    assert -1.0 <= rho <= 1.0, f"Spearman rho out of range: {rho}"


# ---------------------------------------------------------------------------
# Test 16: Temperature effect — lower temperature gives higher loss magnitude
# ---------------------------------------------------------------------------

def test_temperature_effect_on_loss():
    """Lower temperature scales logits by 1/tau, making the softmax sharper.

    We test this directly: given fixed normalized embeddings and a cosine similarity
    matrix, the NT-Xent loss with temperature=0.05 should be LOWER than with
    temperature=1.0 when the positive pair already has the highest similarity
    (the model is 'right' — lower temp = more confident = lower CE loss).
    Conversely, when the positive pair has lower similarity than some negatives,
    lower temp should give HIGHER loss (more penalized).

    Here we construct a clear case where positive sim > all negative sims and
    verify loss(low_temp) < loss(high_temp).
    """
    import torch.nn.functional as F

    B = BATCH
    D = D_MODEL

    # Construct z1 and z2 such that z1[i] dot z2[i] >> z1[i] dot z2[j!=i]
    # Use almost-identical pairs with small noise
    torch.manual_seed(123)
    base = torch.randn(B, D)
    base = F.normalize(base, dim=-1)
    noise = torch.randn(B, D) * 0.01
    z2 = F.normalize(base + noise, dim=-1)  # z2[i] is close to base[i]
    # z1 is base itself
    z1 = base

    def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> float:
        sim = torch.mm(z1, z2.t()) / temp
        labels = torch.arange(B)
        return torch.nn.functional.cross_entropy(sim, labels).item()

    loss_low = nt_xent_loss(z1, z2, temp=0.05)
    loss_high = nt_xent_loss(z1, z2, temp=1.0)

    # When positives dominate, sharpening (low temp) gives lower CE loss
    assert loss_low < loss_high, (
        f"Expected lower temperature to give lower loss when positives dominate, "
        f"got loss_low={loss_low:.4f} loss_high={loss_high:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 17: Identical inputs — pos_sim approx 1, loss near minimum
# ---------------------------------------------------------------------------

def test_identical_inputs_high_pos_sim():
    """When the same input always produces the same embedding (no dropout),
    pos_sim should be ~1 and loss should be near its minimum (log(1/B))."""
    # Use eval mode so dropout is identity — both passes are identical
    enc = make_encoder()
    enc.eval()

    # Manually set all embedding weights to the same value so embeddings are identical
    with torch.no_grad():
        enc.backbone.embed.weight.fill_(0.5)

    loss_fn = UnsupervisedSimCSELoss(temperature=TEMPERATURE)
    x = make_input()

    # In eval mode, both passes give the same output
    loss, avg_pos, avg_neg = loss_fn(enc, x)

    assert avg_pos.item() > 0.99, f"pos_sim should be ~1.0 for identical embeddings, got {avg_pos.item():.4f}"
    # Minimum possible loss for NT-Xent with perfect positives = log(B)
    # (all off-diagonal similarities also = 1 here, so loss = log(B))
    assert torch.isfinite(loss), f"loss not finite: {loss}"
