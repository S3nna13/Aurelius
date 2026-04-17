"""
Tests for src/inference/llm_watermark.py

Config: vocab=16, gamma=0.5, delta=2.0, z_threshold=2.0,
        seq_len=8, batch=2 (tiny to stay within project constraints).
"""

from __future__ import annotations

import math

import torch
import pytest

from src.inference.llm_watermark import (
    GreenRedPartitioner,
    LearnedWatermarkDetector,
    WatermarkBenchmark,
    WatermarkDetector,
    WatermarkLogitsProcessor,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
VOCAB = 16
GAMMA = 0.5
DELTA = 2.0
Z_THR = 2.0
SEQ_LEN = 8
BATCH = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _partitioner() -> GreenRedPartitioner:
    return GreenRedPartitioner(vocab_size=VOCAB, gamma=GAMMA, seed_key=42)


def _detector() -> WatermarkDetector:
    return WatermarkDetector(_partitioner(), z_threshold=Z_THR)


# ---------------------------------------------------------------------------
# Test 1 — GreenRedPartitioner.partition: green+red = full vocab, no overlap
# ---------------------------------------------------------------------------
def test_partition_union_and_no_overlap() -> None:
    p = _partitioner()
    green, red = p.partition(context_hash=7)
    # Union covers full vocab
    all_ids = torch.cat([green, red]).sort().values
    expected = torch.arange(VOCAB)
    assert torch.equal(all_ids, expected), "Green ∪ Red must equal full vocabulary"
    # No overlap
    green_set = set(green.tolist())
    red_set = set(red.tolist())
    assert green_set.isdisjoint(red_set), "Green and Red lists must not overlap"


# ---------------------------------------------------------------------------
# Test 2 — GreenRedPartitioner.partition: correct sizes
# ---------------------------------------------------------------------------
def test_partition_sizes() -> None:
    p = _partitioner()
    green, red = p.partition(context_hash=3)
    expected_green = int(math.floor(GAMMA * VOCAB))
    assert green.shape[0] == expected_green, f"Expected {expected_green} green tokens"
    assert red.shape[0] == VOCAB - expected_green, "Red size should be V - n_green"


# ---------------------------------------------------------------------------
# Test 3 — Same context_hash → same partition (deterministic)
# ---------------------------------------------------------------------------
def test_partition_deterministic() -> None:
    p = _partitioner()
    g1, r1 = p.partition(context_hash=99)
    g2, r2 = p.partition(context_hash=99)
    assert torch.equal(g1.sort().values, g2.sort().values), "Green lists must match"
    assert torch.equal(r1.sort().values, r2.sort().values), "Red lists must match"


# ---------------------------------------------------------------------------
# Test 4 — Different context_hash → different partition (high probability)
# ---------------------------------------------------------------------------
def test_partition_different_context_differs() -> None:
    p = _partitioner()
    g1, _ = p.partition(context_hash=0)
    g2, _ = p.partition(context_hash=1)
    # With vocab=16, the probability of identical permutation is astronomically low
    assert not torch.equal(g1.sort().values, g2.sort().values), (
        "Different context hashes should produce different partitions"
    )


# ---------------------------------------------------------------------------
# Test 5 — WatermarkLogitsProcessor.process: output shape (B, V)
# ---------------------------------------------------------------------------
def test_logits_processor_output_shape() -> None:
    p = _partitioner()
    proc = WatermarkLogitsProcessor(p, delta=DELTA)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    logits = torch.randn(BATCH, VOCAB)
    out = proc.process(input_ids, logits)
    assert out.shape == (BATCH, VOCAB), f"Expected ({BATCH}, {VOCAB}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 6 — WatermarkLogitsProcessor: green positions increased by delta
# ---------------------------------------------------------------------------
def test_logits_processor_green_boosted() -> None:
    p = _partitioner()
    proc = WatermarkLogitsProcessor(p, delta=DELTA)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    logits = torch.zeros(BATCH, VOCAB)
    out = proc.process(input_ids, logits)

    for b in range(BATCH):
        ctx_hash = p.context_hash(input_ids[b])
        green_ids, _ = p.partition(ctx_hash)
        # All green positions should be exactly DELTA
        assert torch.allclose(
            out[b, green_ids], torch.full_like(out[b, green_ids], DELTA)
        ), f"Batch {b}: green logits should be boosted by delta={DELTA}"


# ---------------------------------------------------------------------------
# Test 7 — WatermarkLogitsProcessor: non-green positions unchanged
# ---------------------------------------------------------------------------
def test_logits_processor_red_unchanged() -> None:
    p = _partitioner()
    proc = WatermarkLogitsProcessor(p, delta=DELTA)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    logits = torch.zeros(BATCH, VOCAB)
    out = proc.process(input_ids, logits)

    for b in range(BATCH):
        ctx_hash = p.context_hash(input_ids[b])
        _, red_ids = p.partition(ctx_hash)
        assert torch.allclose(
            out[b, red_ids], torch.zeros_like(out[b, red_ids])
        ), f"Batch {b}: red logits must remain unchanged"


# ---------------------------------------------------------------------------
# Test 8 — WatermarkDetector.count_green_tokens: returns (int, int), n_green ≤ n_total
# ---------------------------------------------------------------------------
def test_count_green_tokens_types_and_bound() -> None:
    det = _detector()
    token_ids = torch.randint(0, VOCAB, (SEQ_LEN,))
    n_green, n_total = det.count_green_tokens(token_ids)
    assert isinstance(n_green, int), "n_green must be int"
    assert isinstance(n_total, int), "n_total must be int"
    assert n_green <= n_total, "n_green must not exceed n_total"
    # n_total should be seq_len - 1 (first token skipped)
    assert n_total == SEQ_LEN - 1, f"n_total should be {SEQ_LEN - 1}"


# ---------------------------------------------------------------------------
# Test 9 — WatermarkDetector.z_score: finite float, higher for more green tokens
# ---------------------------------------------------------------------------
def test_z_score_finite_and_monotone() -> None:
    det = _detector()
    n_total = 10
    z_low = det.z_score(0, n_total)
    z_mid = det.z_score(n_total // 2, n_total)
    z_high = det.z_score(n_total, n_total)
    assert math.isfinite(z_low), "z_score must be finite"
    assert math.isfinite(z_mid), "z_score must be finite"
    assert math.isfinite(z_high), "z_score must be finite"
    assert z_low < z_mid < z_high, "z_score must increase with n_green"


# ---------------------------------------------------------------------------
# Test 10 — WatermarkDetector.detect: returns (bool, float, float), p_value in [0,1]
# ---------------------------------------------------------------------------
def test_detect_return_types_and_p_value_range() -> None:
    det = _detector()
    token_ids = torch.randint(0, VOCAB, (SEQ_LEN,))
    is_wm, z, p = det.detect(token_ids)
    assert isinstance(is_wm, bool), "is_watermarked must be bool"
    assert isinstance(z, float), "z_score must be float"
    assert isinstance(p, float), "p_value must be float"
    assert 0.0 <= p <= 1.0, f"p_value must be in [0, 1], got {p}"


# ---------------------------------------------------------------------------
# Test 11 — WatermarkDetector: heavily biased-green → is_watermarked=True
# ---------------------------------------------------------------------------
def test_detect_watermarked_sequence() -> None:
    p = _partitioner()
    det = WatermarkDetector(p, z_threshold=Z_THR)

    # Build a long sequence where every token is in the green list of its predecessor
    seq_len = 64
    tokens = [torch.randint(0, VOCAB, (1,)).item()]
    for _ in range(seq_len - 1):
        prev = torch.tensor([tokens[-1]])
        ctx_hash = p.context_hash(prev)
        green_ids, _ = p.partition(ctx_hash)
        # always pick first green token
        tokens.append(int(green_ids[0].item()))

    token_ids = torch.tensor(tokens)
    is_wm, z, _ = det.detect(token_ids)
    assert is_wm, f"Heavily green-biased sequence should be detected (z={z:.2f})"
    assert z > Z_THR, f"z={z:.2f} should exceed threshold {Z_THR}"


# ---------------------------------------------------------------------------
# Test 12 — WatermarkDetector: uniform random tokens → z_score ≈ 0
# ---------------------------------------------------------------------------
def test_detect_random_sequence_z_near_zero() -> None:
    torch.manual_seed(0)
    det = _detector()
    # Use a long sequence so the mean is close to 0 by CLT
    seq_len = 256
    z_scores = []
    for _ in range(10):
        token_ids = torch.randint(0, VOCAB, (seq_len,))
        _, z, _ = det.detect(token_ids)
        z_scores.append(z)

    mean_z = sum(z_scores) / len(z_scores)
    # With random tokens, z should centre near 0; allow generous range
    assert abs(mean_z) < 3.0, f"Mean z-score for random tokens should be ≈ 0, got {mean_z:.3f}"


# ---------------------------------------------------------------------------
# Test 13 — LearnedWatermarkDetector: output shape (B,), values in (0,1)
# ---------------------------------------------------------------------------
def test_learned_detector_output() -> None:
    model = LearnedWatermarkDetector(vocab_size=VOCAB, embed_dim=32, n_layers=2)
    token_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    probs = model(token_ids)
    assert probs.shape == (BATCH,), f"Expected ({BATCH},), got {probs.shape}"
    assert torch.all(probs > 0.0) and torch.all(probs < 1.0), "Probabilities must be in (0, 1)"


# ---------------------------------------------------------------------------
# Test 14 — LearnedWatermarkDetector.loss: scalar, finite, grad flows
# ---------------------------------------------------------------------------
def test_learned_detector_loss_and_gradients() -> None:
    model = LearnedWatermarkDetector(vocab_size=VOCAB, embed_dim=32, n_layers=2)
    token_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    labels = torch.tensor([1.0, 0.0])
    loss = model.loss(token_ids, labels)
    assert loss.shape == (), "Loss must be a scalar"
    assert torch.isfinite(loss), "Loss must be finite"
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Gradient for {name} is not finite"


# ---------------------------------------------------------------------------
# Test 15 — WatermarkBenchmark.distortion_score: ≥ 0, 0 for identical logits
# ---------------------------------------------------------------------------
def test_distortion_score_nonneg_and_zero_for_identical() -> None:
    det = _detector()
    bench = WatermarkBenchmark(det)
    logits = torch.randn(BATCH, VOCAB)
    score_same = bench.distortion_score(logits, logits)
    assert abs(score_same) < 1e-5, f"Identical logits → distortion 0, got {score_same}"

    wm_logits = logits.clone()
    wm_logits[:, :VOCAB // 2] += 3.0
    score_diff = bench.distortion_score(logits, wm_logits)
    assert score_diff >= 0.0, f"Distortion score must be ≥ 0, got {score_diff}"


# ---------------------------------------------------------------------------
# Bonus Test (16th guard) — WatermarkBenchmark.perplexity_impact: > 0
# ---------------------------------------------------------------------------
def test_perplexity_impact_positive() -> None:
    det = _detector()
    bench = WatermarkBenchmark(det)
    clean_lp = torch.full((SEQ_LEN,), -1.0)          # log probs = -1.0
    wm_lp = torch.full((SEQ_LEN,), -2.0)             # worse log probs
    impact = bench.perplexity_impact(wm_lp, clean_lp)
    assert impact > 0.0, f"perplexity_impact must be positive, got {impact}"
    assert impact > 1.0, "Watermarked perplexity should be higher than clean when lp is worse"


# ---------------------------------------------------------------------------
# Bonus Test (17th guard) — WatermarkBenchmark.tpr_at_fpr: float in [0,1]
# ---------------------------------------------------------------------------
def test_tpr_at_fpr_range() -> None:
    p_obj = _partitioner()
    det = WatermarkDetector(p_obj, z_threshold=Z_THR)
    bench = WatermarkBenchmark(det)

    # Build watermarked sequences (always pick green token)
    def make_wm_seq(length: int = 32) -> torch.Tensor:
        tokens = [torch.randint(0, VOCAB, (1,)).item()]
        for _ in range(length - 1):
            prev = torch.tensor([tokens[-1]])
            ctx_hash = p_obj.context_hash(prev)
            green_ids, _ = p_obj.partition(ctx_hash)
            tokens.append(int(green_ids[0].item()))
        return torch.tensor(tokens)

    wm_seqs = [make_wm_seq(32) for _ in range(5)]
    clean_seqs = [torch.randint(0, VOCAB, (32,)) for _ in range(10)]

    tpr = bench.tpr_at_fpr(wm_seqs, clean_seqs, fpr_target=0.5)
    assert 0.0 <= tpr <= 1.0, f"tpr_at_fpr must be in [0,1], got {tpr}"
