"""
Tests for src/eval/factual_consistency.py

Tiny config: d_model=16, vocab=16, seq_len=8, batch=2
All tests run forward and/or backward passes.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.eval.factual_consistency import (
    ConsistencyBenchmark,
    FactConsistencyScorer,
    FactualConsistencyTrainer,
    HallucinationDetector,
    NLIClassifier,
    TextEncoder,
)

# ---------------------------------------------------------------------------
# Tiny constants
# ---------------------------------------------------------------------------
D = 16  # d_model
V = 16  # vocab size
T = 8  # seq_len
B = 2  # batch


# ---------------------------------------------------------------------------
# Minimal backbone: Embedding + Linear → (B, T, D)
# ---------------------------------------------------------------------------


class TinyBackbone(nn.Module):
    """nn.Embedding(V, D) + nn.Linear(D, D) — returns (B, T, D)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, D)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.proj(self.emb(input_ids))


def make_backbone() -> TinyBackbone:
    return TinyBackbone()


def make_encoder(pooling: str = "mean") -> TextEncoder:
    return TextEncoder(make_backbone(), pooling=pooling)


def make_nli() -> NLIClassifier:
    return NLIClassifier(D)


def rand_ids(batch: int = B, seq: int = T) -> Tensor:
    return torch.randint(0, V, (batch, seq))


# ---------------------------------------------------------------------------
# 1. NLIClassifier output shape (B, 3)
# ---------------------------------------------------------------------------


def test_nli_output_shape():
    nli = make_nli()
    p = torch.randn(B, D)
    h = torch.randn(B, D)
    logits = nli(p, h)
    assert logits.shape == (B, 3), f"Expected (B,3), got {logits.shape}"


# ---------------------------------------------------------------------------
# 2. NLIClassifier logits are finite
# ---------------------------------------------------------------------------


def test_nli_logits_finite():
    nli = make_nli()
    p = torch.randn(B, D)
    h = torch.randn(B, D)
    logits = nli(p, h)
    assert torch.isfinite(logits).all(), "NLI logits contain non-finite values"


# ---------------------------------------------------------------------------
# 3. entailment_score shape (B,) and values in (0, 1)
# ---------------------------------------------------------------------------


def test_entailment_score_shape_and_range():
    nli = make_nli()
    p = torch.randn(B, D)
    h = torch.randn(B, D)
    scores = nli.entailment_score(p, h)
    assert scores.shape == (B,), f"Expected ({B},), got {scores.shape}"
    assert (scores > 0).all() and (scores < 1).all(), "Entailment scores outside (0,1)"


# ---------------------------------------------------------------------------
# 4. NLIClassifier: grad flows through both premise and hypothesis paths
# ---------------------------------------------------------------------------


def test_nli_gradient_flows():
    nli = make_nli()
    p = torch.randn(B, D, requires_grad=True)
    h = torch.randn(B, D, requires_grad=True)
    logits = nli(p, h)
    loss = logits.sum()
    loss.backward()
    assert p.grad is not None and p.grad.abs().sum() > 0, "No grad on premise"
    assert h.grad is not None and h.grad.abs().sum() > 0, "No grad on hypothesis"


# ---------------------------------------------------------------------------
# 5. TextEncoder.encode: shape (B, D) for mean/cls/last pooling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pooling", ["mean", "cls", "last"])
def test_text_encoder_encode_shape(pooling):
    enc = make_encoder(pooling)
    ids = rand_ids()
    out = enc.encode(ids)
    assert out.shape == (B, D), f"pooling={pooling}: expected ({B},{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 6. TextEncoder.batch_encode: shape (N, D) for N sequences
# ---------------------------------------------------------------------------


def test_text_encoder_batch_encode_shape():
    enc = make_encoder()
    seqs = [torch.randint(0, V, (T + i,)) for i in range(3)]  # different lengths
    out = enc.batch_encode(seqs)
    assert out.shape == (3, D), f"Expected (3,{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 7. FactConsistencyScorer.score: returns (float, str), label in valid set
# ---------------------------------------------------------------------------


def test_fact_scorer_score_types():
    scorer = FactConsistencyScorer(make_encoder(), make_nli())
    src = rand_ids(1)
    clm = rand_ids(1)
    prob, label = scorer.score(src, clm)
    assert isinstance(prob, float), f"prob should be float, got {type(prob)}"
    assert label in {"entailment", "neutral", "contradiction"}, f"Invalid label: {label}"


# ---------------------------------------------------------------------------
# 8. FactConsistencyScorer.batch_score: length == len(claims), all in (0,1)
# ---------------------------------------------------------------------------


def test_fact_scorer_batch_score():
    scorer = FactConsistencyScorer(make_encoder(), make_nli())
    src = rand_ids(1)
    claims = [rand_ids(1).squeeze(0) for _ in range(3)]
    scores = scorer.batch_score(src, claims)
    assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
    for s in scores:
        assert 0.0 < s < 1.0, f"Score {s} outside (0,1)"


# ---------------------------------------------------------------------------
# 9. FactConsistencyScorer.consistency_threshold: returns list[bool], same len
# ---------------------------------------------------------------------------


def test_consistency_threshold():
    scorer = FactConsistencyScorer(make_encoder(), make_nli())
    scores = [0.2, 0.5, 0.7, 0.9]
    result = scorer.consistency_threshold(scores, threshold=0.5)
    assert len(result) == len(scores), "Length mismatch"
    assert all(isinstance(v, bool) for v in result), "Not all bool"
    assert result == [False, True, True, True], f"Unexpected result: {result}"


# ---------------------------------------------------------------------------
# 10. HallucinationDetector.token_entropy: shape (T,), all >= 0
# ---------------------------------------------------------------------------


def test_hallucination_detector_entropy_shape_and_nonneg():
    det = HallucinationDetector(make_backbone(), threshold=3.0)
    ids = rand_ids(1)  # (1, T)
    entropy = det.token_entropy(ids)
    assert entropy.shape == (T,), f"Expected ({T},), got {entropy.shape}"
    assert (entropy >= 0).all(), "Entropy has negative values"


# ---------------------------------------------------------------------------
# 11. HallucinationDetector.detect_hallucinations: all 3 keys present
# ---------------------------------------------------------------------------


def test_hallucination_detector_keys():
    det = HallucinationDetector(make_backbone(), threshold=3.0)
    ids = rand_ids(1)
    ctx = rand_ids(1)
    result = det.detect_hallucinations(ids, ctx)
    for key in ("high_entropy_positions", "consistency_score", "hallucination_risk"):
        assert key in result, f"Missing key: {key}"
    assert result["hallucination_risk"] in {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# 12. HallucinationDetector: high entropy positions correctly identified
# ---------------------------------------------------------------------------


def test_hallucination_detector_high_entropy_positions():
    backbone = make_backbone()
    threshold = 0.0  # almost all positions will be above this (entropy of uniform > 0)
    det = HallucinationDetector(backbone, threshold=threshold)
    ids = rand_ids(1)
    entropy = det.token_entropy(ids)
    expected = [int(i) for i, e in enumerate(entropy.tolist()) if e > threshold]

    result = det.detect_hallucinations(ids, rand_ids(1))
    assert result["high_entropy_positions"] == expected, (
        f"Mismatch in high_entropy_positions:\n"
        f"  expected {expected}\n  got {result['high_entropy_positions']}"
    )


# ---------------------------------------------------------------------------
# 13. FactualConsistencyTrainer.train_step: loss finite, accuracy in [0,1]
# ---------------------------------------------------------------------------


def test_trainer_train_step_loss_and_accuracy():
    encoder = make_encoder()
    nli = make_nli()

    # Collect all parameters
    params = list(encoder.model.parameters()) + list(nli.parameters())
    optimizer = torch.optim.SGD(params, lr=1e-3)
    trainer = FactualConsistencyTrainer(encoder, nli, optimizer)

    premise_ids = rand_ids(B)
    hyp_ids = rand_ids(B)
    labels = torch.randint(0, 3, (B,))

    result = trainer.train_step(premise_ids, hyp_ids, labels)
    assert "loss" in result and "accuracy" in result
    assert math.isfinite(result["loss"]), f"Loss is not finite: {result['loss']}"
    assert 0.0 <= result["accuracy"] <= 1.0, f"Accuracy out of [0,1]: {result['accuracy']}"


# ---------------------------------------------------------------------------
# 14. FactualConsistencyTrainer: grad flows to encoder and NLI classifier
# ---------------------------------------------------------------------------


def test_trainer_gradient_flows():
    backbone = make_backbone()
    encoder = TextEncoder(backbone, pooling="mean")
    nli = make_nli()

    params = list(backbone.parameters()) + list(nli.parameters())
    optimizer = torch.optim.SGD(params, lr=1e-3)
    trainer = FactualConsistencyTrainer(encoder, nli, optimizer)

    # Zero out grads first
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    premise_ids = rand_ids(B)
    hyp_ids = rand_ids(B)
    labels = torch.randint(0, 3, (B,))

    trainer.train_step(premise_ids, hyp_ids, labels)

    backbone_grad_norm = sum(
        p.grad.abs().sum().item() for p in backbone.parameters() if p.grad is not None
    )
    nli_grad_norm = sum(p.grad.abs().sum().item() for p in nli.parameters() if p.grad is not None)
    assert backbone_grad_norm > 0, "No gradient flowed to backbone (encoder)"
    assert nli_grad_norm > 0, "No gradient flowed to NLI classifier"


# ---------------------------------------------------------------------------
# 15. ConsistencyBenchmark: precision/recall/f1 in [0,1]; calibration_error in [0,1]
# ---------------------------------------------------------------------------


def test_benchmark_precision_recall_and_calibration():
    bench = ConsistencyBenchmark()

    preds = [True, True, False, True, False]
    truth = [True, False, False, True, True]
    metrics = bench.precision_recall(preds, truth)
    for k in ("precision", "recall", "f1"):
        assert k in metrics, f"Missing key: {k}"
        assert 0.0 <= metrics[k] <= 1.0, f"{k}={metrics[k]} outside [0,1]"

    scores = [0.1, 0.4, 0.6, 0.8, 0.9]
    labels = [False, False, True, True, True]
    ece = bench.calibration_error(scores, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0, f"ECE={ece} outside [0,1]"
