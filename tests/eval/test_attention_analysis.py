"""Tests for src/eval/attention_analysis.py — head importance, clustering, induction heads."""

from __future__ import annotations

import pytest
import torch

from src.eval.attention_analysis import (
    AttentionAnalysisConfig,
    AttentionPattern,
    HeadAnalyzer,
    classify_attention_pattern,
    cluster_attention_heads,
    compute_head_entropy,
    compute_mean_attention_distance,
    detect_induction_heads,
    extract_attention_weights,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


@pytest.fixture
def input_ids():
    return torch.randint(0, 256, (1, 16))


def make_diagonal_attn(T: int) -> torch.Tensor:
    """Attention matrix that is nearly the identity (diagonal pattern)."""
    attn = torch.eye(T) * 0.9 + torch.ones(T, T) * (0.1 / T)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


def make_uniform_attn(T: int) -> torch.Tensor:
    """Uniform attention matrix (global pattern)."""
    return torch.full((T, T), 1.0 / T)


def make_peaked_attn(T: int) -> torch.Tensor:
    """Attention concentrated entirely on position 0 (low entropy)."""
    attn = torch.zeros(T, T)
    attn[:, 0] = 1.0
    return attn


def make_subdiag_attn(T: int, offset: int = 3) -> torch.Tensor:
    """Attention concentrating on sub-diagonal at given offset (induction-like)."""
    attn = torch.zeros(T, T)
    for i in range(T):
        j = i - offset
        if 0 <= j < T:
            attn[i, j] = 0.8
        # remaining weight spread uniformly to avoid zero rows
        remaining = 1.0 - attn[i].sum().item()
        attn[i] += remaining / T
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


def make_sample_patterns(n_layers: int = 2, n_heads: int = 2) -> list[AttentionPattern]:
    """Create a diverse set of AttentionPattern objects for testing."""
    T = 16
    patterns = []
    for layer in range(n_layers):
        for head in range(n_heads):
            attn = make_uniform_attn(T) if (layer + head) % 2 == 0 else make_peaked_attn(T)
            entropy = compute_head_entropy(attn)
            mean_dist = compute_mean_attention_distance(attn)
            patterns.append(
                AttentionPattern(
                    layer=layer,
                    head=head,
                    pattern_type="global" if (layer + head) % 2 == 0 else "local",
                    entropy=entropy,
                    mean_distance=mean_dist,
                )
            )
    return patterns


# ---------------------------------------------------------------------------
# 1. AttentionAnalysisConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = AttentionAnalysisConfig()
    assert cfg.n_heads == 2
    assert cfg.n_layers == 2
    assert cfg.induction_threshold == 0.5
    assert cfg.cluster_k == 3


# ---------------------------------------------------------------------------
# 2. AttentionPattern fields
# ---------------------------------------------------------------------------


def test_attention_pattern_fields():
    p = AttentionPattern(layer=1, head=0, pattern_type="local", entropy=0.7, mean_distance=2.5)
    assert p.layer == 1
    assert p.head == 0
    assert p.pattern_type == "local"
    assert p.entropy == pytest.approx(0.7)
    assert p.mean_distance == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# 3. extract_attention_weights returns dict
# ---------------------------------------------------------------------------


def test_extract_attention_weights_returns_dict(small_model, input_ids):
    result = extract_attention_weights(small_model, input_ids)
    assert isinstance(result, dict)
    assert len(result) > 0
    for key in result:
        assert isinstance(key, tuple)
        assert len(key) == 2


# ---------------------------------------------------------------------------
# 4. extract_attention_weights attention sums to ~1 per row
# ---------------------------------------------------------------------------


def test_extract_attention_weights_rows_sum_to_one(small_model, input_ids):
    result = extract_attention_weights(small_model, input_ids)
    for (layer, head), attn in result.items():
        # attn: (B, T, T) or (T, T)
        if attn.dim() == 3:
            attn = attn.squeeze(0)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
            f"Layer {layer}, head {head}: rows don't sum to 1 (got {row_sums})"
        )


# ---------------------------------------------------------------------------
# 5. classify_attention_pattern — diagonal matrix → "diagonal"
# ---------------------------------------------------------------------------


def test_classify_diagonal():
    T = 10
    attn = make_diagonal_attn(T)
    result = classify_attention_pattern(attn)
    assert result == "diagonal"


# ---------------------------------------------------------------------------
# 6. classify_attention_pattern — uniform matrix → "global"
# ---------------------------------------------------------------------------


def test_classify_global():
    T = 20
    attn = make_uniform_attn(T)
    result = classify_attention_pattern(attn)
    assert result == "global"


# ---------------------------------------------------------------------------
# 7. compute_head_entropy — uniform attention → high entropy (near 1.0)
# ---------------------------------------------------------------------------


def test_entropy_uniform_is_high():
    T = 16
    attn = make_uniform_attn(T)
    entropy = compute_head_entropy(attn)
    # Uniform attention has maximum entropy = log(T), normalized to 1.0
    assert entropy > 0.95, f"Expected entropy near 1.0, got {entropy}"


# ---------------------------------------------------------------------------
# 8. compute_head_entropy — peaked attention → low entropy (near 0.0)
# ---------------------------------------------------------------------------


def test_entropy_peaked_is_low():
    T = 16
    attn = make_peaked_attn(T)
    entropy = compute_head_entropy(attn)
    assert entropy < 0.2, f"Expected entropy near 0.0, got {entropy}"


# ---------------------------------------------------------------------------
# 9. compute_mean_attention_distance — diagonal attention → ~0.0
# ---------------------------------------------------------------------------


def test_mean_distance_diagonal_near_zero():
    T = 16
    attn = torch.eye(T)
    dist = compute_mean_attention_distance(attn)
    assert dist < 0.1, f"Expected near-zero distance for diagonal attn, got {dist}"


# ---------------------------------------------------------------------------
# 10. detect_induction_heads returns list of tuples
# ---------------------------------------------------------------------------


def test_detect_induction_heads_returns_list_of_tuples():
    T = 20
    attn_weights = {
        (0, 0): make_subdiag_attn(T, offset=3).unsqueeze(0),
        (0, 1): make_uniform_attn(T).unsqueeze(0),
    }
    result = detect_induction_heads(attn_weights, threshold=0.5)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2


# ---------------------------------------------------------------------------
# 11. cluster_attention_heads returns k clusters
# ---------------------------------------------------------------------------


def test_cluster_returns_k_clusters():
    patterns = make_sample_patterns(n_layers=4, n_heads=3)
    k = 3
    clusters = cluster_attention_heads(patterns, k=k)
    assert isinstance(clusters, dict)
    # Should have at most k distinct cluster ids
    assert len(clusters) <= k


# ---------------------------------------------------------------------------
# 12. HeadAnalyzer.analyze returns list of AttentionPattern
# ---------------------------------------------------------------------------


def test_head_analyzer_analyze_returns_patterns(small_model, input_ids, small_cfg):
    config = AttentionAnalysisConfig(n_heads=small_cfg.n_heads, n_layers=small_cfg.n_layers)
    analyzer = HeadAnalyzer(small_model, config)
    patterns = analyzer.analyze(input_ids)
    assert isinstance(patterns, list)
    assert len(patterns) > 0
    for p in patterns:
        assert isinstance(p, AttentionPattern)
        assert p.pattern_type in ("local", "global", "induction", "diagonal")


# ---------------------------------------------------------------------------
# 13. HeadAnalyzer.rank_by_importance sorted descending
# ---------------------------------------------------------------------------


def test_rank_by_importance_sorted_descending(small_model, input_ids, small_cfg):
    config = AttentionAnalysisConfig(n_heads=small_cfg.n_heads, n_layers=small_cfg.n_layers)
    analyzer = HeadAnalyzer(small_model, config)
    patterns = analyzer.analyze(input_ids)
    ranked = analyzer.rank_by_importance(patterns)
    assert isinstance(ranked, list)
    assert len(ranked) == len(patterns)
    importances = [r[0] for r in ranked]
    assert importances == sorted(importances, reverse=True), (
        f"Importances not sorted descending: {importances}"
    )


# ---------------------------------------------------------------------------
# 14. HeadAnalyzer.report returns required keys
# ---------------------------------------------------------------------------


def test_report_has_required_keys(small_model, input_ids, small_cfg):
    config = AttentionAnalysisConfig(n_heads=small_cfg.n_heads, n_layers=small_cfg.n_layers)
    analyzer = HeadAnalyzer(small_model, config)
    patterns = analyzer.analyze(input_ids)
    report = analyzer.report(patterns)
    required_keys = {"n_local", "n_global", "n_induction", "n_diagonal", "mean_entropy"}
    assert required_keys.issubset(set(report.keys())), (
        f"Missing keys: {required_keys - set(report.keys())}"
    )
    assert isinstance(report["mean_entropy"], float)
    total = report["n_local"] + report["n_global"] + report["n_induction"] + report["n_diagonal"]
    assert total == len(patterns)


# ---------------------------------------------------------------------------
# 15. cluster_attention_heads — all heads assigned to some cluster
# ---------------------------------------------------------------------------


def test_cluster_all_heads_assigned():
    patterns = make_sample_patterns(n_layers=3, n_heads=2)
    k = 3
    clusters = cluster_attention_heads(patterns, k=k)
    assigned = []
    for members in clusters.values():
        assigned.extend(members)
    # All heads should be assigned
    assert len(assigned) == len(patterns), f"Expected {len(patterns)} assigned, got {len(assigned)}"
    # No duplicates
    assert len(set(assigned)) == len(assigned)
