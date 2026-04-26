"""Integration tests for DPAwareMoERouter — GLM-5 §3.2 consistent hashing.

Verifies that the class is correctly wired into MODEL_COMPONENT_REGISTRY
via src/model/__init__.py, and that the routing API works end-to-end.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. "dp_aware_moe_routing" present in MODEL_COMPONENT_REGISTRY
# ---------------------------------------------------------------------------


def test_dp_aware_moe_routing_in_registry():
    from src.model import MODEL_COMPONENT_REGISTRY

    assert "dp_aware_moe_routing" in MODEL_COMPONENT_REGISTRY, (
        "MODEL_COMPONENT_REGISTRY missing 'dp_aware_moe_routing' key"
    )


# ---------------------------------------------------------------------------
# 2. Construct from registry, rank_for_session returns valid rank
# ---------------------------------------------------------------------------


def test_registry_construct_and_rank_for_session():
    from src.model import MODEL_COMPONENT_REGISTRY

    cls = MODEL_COMPONENT_REGISTRY["dp_aware_moe_routing"]
    router = cls(num_dp_ranks=8)
    rank = router.rank_for_session("integration-test-session-42")
    assert isinstance(rank, int), "rank_for_session must return an int"
    assert 0 <= rank < 8, f"rank {rank} out of [0, 8)"


# ---------------------------------------------------------------------------
# 3. route_batch of 3 sessions → 3 results with ranks in range
# ---------------------------------------------------------------------------


def test_registry_route_batch_three_sessions():
    from src.model import MODEL_COMPONENT_REGISTRY

    cls = MODEL_COMPONENT_REGISTRY["dp_aware_moe_routing"]
    router = cls(num_dp_ranks=4)

    session_ids = ["alpha-session", "beta-session", "gamma-session"]
    expert_ids_batch = [[0, 1], [2, 3], [4, 5]]

    results = router.route_batch(session_ids, expert_ids_batch)
    assert len(results) == 3, "route_batch must return one result per input"

    for i, (rank, experts) in enumerate(results):
        assert 0 <= rank < 4, f"Result {i}: rank {rank} out of [0, 4)"
        assert experts == expert_ids_batch[i], (
            f"Result {i}: experts modified — expected {expert_ids_batch[i]}, got {experts}"
        )


# ---------------------------------------------------------------------------
# 4. Regression guard: existing MODEL_COMPONENT_REGISTRY keys still present
# ---------------------------------------------------------------------------


def test_existing_registry_keys_regression():
    """Ensure adding dp_aware_moe_routing did not remove pre-existing keys."""
    from src.model import MODEL_COMPONENT_REGISTRY

    assert isinstance(MODEL_COMPONENT_REGISTRY, dict)
    # Keys introduced by earlier cycles must remain
    assert "dsa_attention" in MODEL_COMPONENT_REGISTRY, (
        "Regression: 'dsa_attention' was removed from MODEL_COMPONENT_REGISTRY"
    )
    assert "mtp_shared" in MODEL_COMPONENT_REGISTRY, (
        "Regression: 'mtp_shared' was removed from MODEL_COMPONENT_REGISTRY"
    )
