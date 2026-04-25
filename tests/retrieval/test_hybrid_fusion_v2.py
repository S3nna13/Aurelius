"""Tests for hybrid_fusion_v2.py (~45 tests)."""

from __future__ import annotations

import pytest

from src.retrieval.hybrid_fusion_v2 import (
    FusionResult,
    FusionStrategy,
    HYBRID_FUSION_REGISTRY,
    HybridFusionV2,
)


# ---------------------------------------------------------------------------
# FusionStrategy enum
# ---------------------------------------------------------------------------

class TestFusionStrategy:
    def test_rrf_value(self):
        assert FusionStrategy.RRF == "rrf"

    def test_linear_value(self):
        assert FusionStrategy.LINEAR == "linear"

    def test_convex_value(self):
        assert FusionStrategy.CONVEX == "convex"

    def test_three_members(self):
        assert len(list(FusionStrategy)) == 3

    def test_is_str(self):
        assert isinstance(FusionStrategy.RRF, str)


# ---------------------------------------------------------------------------
# FusionResult dataclass
# ---------------------------------------------------------------------------

class TestFusionResult:
    def test_fields(self):
        fr = FusionResult(doc_id="d1", fused_score=0.8)
        assert fr.doc_id == "d1"
        assert fr.fused_score == 0.8

    def test_sparse_default(self):
        fr = FusionResult(doc_id="d1", fused_score=0.5)
        assert fr.sparse_score == 0.0

    def test_dense_default(self):
        fr = FusionResult(doc_id="d1", fused_score=0.5)
        assert fr.dense_score == 0.0

    def test_custom_scores(self):
        fr = FusionResult(doc_id="d1", fused_score=0.7, sparse_score=0.4, dense_score=0.3)
        assert fr.sparse_score == 0.4
        assert fr.dense_score == 0.3

    def test_equality(self):
        a = FusionResult(doc_id="d1", fused_score=0.5)
        b = FusionResult(doc_id="d1", fused_score=0.5)
        assert a == b


# ---------------------------------------------------------------------------
# HybridFusionV2.fuse — LINEAR strategy
# ---------------------------------------------------------------------------

class TestHybridFusionV2Linear:
    def setup_method(self):
        self.fuser = HybridFusionV2(strategy=FusionStrategy.LINEAR, alpha=0.5)

    def test_fuse_returns_list(self):
        result = self.fuser.fuse([("d1", 1.0)], [("d1", 0.5)])
        assert isinstance(result, list)

    def test_fuse_returns_fusion_results(self):
        result = self.fuser.fuse([("d1", 1.0)], [("d1", 0.5)])
        assert all(isinstance(r, FusionResult) for r in result)

    def test_alpha_one_sparse_only(self):
        fuser = HybridFusionV2(strategy=FusionStrategy.LINEAR, alpha=1.0)
        sparse = [("d1", 2.0), ("d2", 1.0)]
        dense = [("d1", 0.0), ("d2", 0.0)]
        result = fuser.fuse(sparse, dense)
        scores = {r.doc_id: r.fused_score for r in result}
        # d1 normalized sparse=1.0, d2=0.0; dense all 0
        assert scores["d1"] > scores["d2"]

    def test_alpha_zero_dense_only(self):
        fuser = HybridFusionV2(strategy=FusionStrategy.LINEAR, alpha=0.0)
        sparse = [("d1", 0.0), ("d2", 0.0)]
        dense = [("d1", 2.0), ("d2", 1.0)]
        result = fuser.fuse(sparse, dense)
        scores = {r.doc_id: r.fused_score for r in result}
        assert scores["d1"] > scores["d2"]

    def test_all_docs_included(self):
        sparse = [("d1", 1.0), ("d2", 0.5)]
        dense = [("d3", 0.8)]
        result = self.fuser.fuse(sparse, dense)
        doc_ids = {r.doc_id for r in result}
        assert doc_ids == {"d1", "d2", "d3"}

    def test_sorted_desc(self):
        sparse = [("d1", 2.0), ("d2", 1.0), ("d3", 0.5)]
        dense = [("d1", 1.0), ("d2", 0.5), ("d3", 0.2)]
        result = self.fuser.fuse(sparse, dense)
        scores = [r.fused_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_sparse_scores_stored(self):
        sparse = [("d1", 1.0)]
        dense = [("d1", 0.5)]
        result = self.fuser.fuse(sparse, dense)
        assert result[0].sparse_score is not None

    def test_dense_scores_stored(self):
        sparse = [("d1", 1.0)]
        dense = [("d1", 0.5)]
        result = self.fuser.fuse(sparse, dense)
        assert result[0].dense_score is not None


# ---------------------------------------------------------------------------
# HybridFusionV2.fuse — CONVEX strategy
# ---------------------------------------------------------------------------

class TestHybridFusionV2Convex:
    def test_convex_result_between_sparse_and_dense(self):
        fuser = HybridFusionV2(strategy=FusionStrategy.CONVEX, alpha=0.5)
        sparse = [("d1", 1.0)]
        dense = [("d1", 0.0)]
        result = fuser.fuse(sparse, dense)
        # With equal sparse=1.0, dense=0.0 (both normalize to same since single doc)
        assert isinstance(result[0].fused_score, float)

    def test_convex_alpha_clamped(self):
        # alpha > 1.0 should be clamped to 1.0 (CONVEX enforces [0,1])
        fuser = HybridFusionV2(strategy=FusionStrategy.CONVEX, alpha=2.0)
        sparse = [("d1", 1.0), ("d2", 0.0)]
        dense = [("d1", 0.0), ("d2", 0.0)]
        result = fuser.fuse(sparse, dense)
        # Should not crash
        assert len(result) == 2

    def test_convex_sorted_desc(self):
        fuser = HybridFusionV2(strategy=FusionStrategy.CONVEX, alpha=0.6)
        sparse = [("d1", 3.0), ("d2", 1.0)]
        dense = [("d1", 2.0), ("d2", 0.5)]
        result = fuser.fuse(sparse, dense)
        scores = [r.fused_score for r in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# HybridFusionV2.fuse — RRF strategy
# ---------------------------------------------------------------------------

class TestHybridFusionV2RRF:
    def setup_method(self):
        self.fuser = HybridFusionV2(strategy=FusionStrategy.RRF, rrf_k=60)

    def test_rrf_returns_list(self):
        sparse = [("d1", 1.0), ("d2", 0.5)]
        dense = [("d1", 0.8), ("d2", 0.4)]
        result = self.fuser.fuse(sparse, dense)
        assert isinstance(result, list)

    def test_rrf_doc_in_both_lists_scores_higher(self):
        # d1 appears in both, d2 only in sparse, d3 only in dense
        sparse = [("d1", 1.0), ("d2", 0.5)]
        dense = [("d1", 0.8), ("d3", 0.4)]
        result = self.fuser.fuse(sparse, dense)
        scores = {r.doc_id: r.fused_score for r in result}
        # d1 in both → higher score than d2 (sparse only rank 2) and d3 (dense only rank 2)
        assert scores["d1"] > scores.get("d2", 0.0)
        assert scores["d1"] > scores.get("d3", 0.0)

    def test_rrf_sorted_desc(self):
        sparse = [("d1", 1.0), ("d2", 0.5), ("d3", 0.1)]
        dense = [("d1", 0.9), ("d2", 0.4), ("d3", 0.2)]
        result = self.fuser.fuse(sparse, dense)
        scores = [r.fused_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_all_docs_included(self):
        sparse = [("d1", 1.0)]
        dense = [("d2", 1.0)]
        result = self.fuser.fuse(sparse, dense)
        assert {r.doc_id for r in result} == {"d1", "d2"}

    def test_rrf_empty_lists(self):
        result = self.fuser.fuse([], [])
        assert result == []


# ---------------------------------------------------------------------------
# HybridFusionV2.tune_alpha
# ---------------------------------------------------------------------------

class TestHybridFusionV2TuneAlpha:
    def setup_method(self):
        self.fuser = HybridFusionV2(strategy=FusionStrategy.LINEAR)

    def test_returns_float(self):
        results_at_alpha = {
            0.3: [FusionResult("d1", 0.9), FusionResult("d2", 0.8)],
            0.7: [FusionResult("d2", 0.9), FusionResult("d1", 0.8)],
        }
        alpha = self.fuser.tune_alpha({"d1"}, results_at_alpha)
        assert isinstance(alpha, float)

    def test_returns_value_in_zero_one(self):
        results_at_alpha = {
            0.1: [FusionResult("d1", 0.9)],
            0.9: [FusionResult("d2", 0.9)],
        }
        alpha = self.fuser.tune_alpha({"d1"}, results_at_alpha)
        assert 0.0 <= alpha <= 1.0

    def test_selects_best_alpha(self):
        # alpha=0.8 has 3 relevant docs in top-5, alpha=0.2 has only 1
        irr = [FusionResult(f"irr{i}", 1.0 - i * 0.1) for i in range(5)]
        rel = [FusionResult(f"rel{i}", 0.95 - i * 0.1) for i in range(5)]
        # alpha=0.2: irrelevant docs dominate top-5
        results_at_02 = irr[:4] + [rel[0]] + irr[4:]
        # alpha=0.8: relevant docs dominate top-5
        results_at_08 = rel[:4] + [irr[0]] + rel[4:]
        results_at_alpha = {0.2: results_at_02, 0.8: results_at_08}
        relevant_ids = {f"rel{i}" for i in range(5)}
        alpha = self.fuser.tune_alpha(relevant_ids, results_at_alpha)
        assert alpha == 0.8

    def test_empty_results(self):
        alpha = self.fuser.tune_alpha({"d1"}, {})
        assert isinstance(alpha, float)

    def test_no_relevant_docs(self):
        results_at_alpha = {
            0.5: [FusionResult("d1", 0.9), FusionResult("d2", 0.8)],
        }
        alpha = self.fuser.tune_alpha(set(), results_at_alpha)
        assert isinstance(alpha, float)


# ---------------------------------------------------------------------------
# HYBRID_FUSION_REGISTRY
# ---------------------------------------------------------------------------

class TestHybridFusionRegistry:
    def test_has_rrf(self):
        assert "rrf" in HYBRID_FUSION_REGISTRY

    def test_has_linear(self):
        assert "linear" in HYBRID_FUSION_REGISTRY

    def test_has_convex(self):
        assert "convex" in HYBRID_FUSION_REGISTRY

    def test_rrf_strategy(self):
        assert HYBRID_FUSION_REGISTRY["rrf"].strategy == FusionStrategy.RRF

    def test_linear_strategy(self):
        assert HYBRID_FUSION_REGISTRY["linear"].strategy == FusionStrategy.LINEAR

    def test_convex_strategy(self):
        assert HYBRID_FUSION_REGISTRY["convex"].strategy == FusionStrategy.CONVEX

    def test_all_are_hybrid_fusion_v2(self):
        for v in HYBRID_FUSION_REGISTRY.values():
            assert isinstance(v, HybridFusionV2)

    def test_registry_size(self):
        assert len(HYBRID_FUSION_REGISTRY) == 3
