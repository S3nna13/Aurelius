"""Unit tests for DPAwareMoERouter — GLM-5 §3.2 consistent hashing."""

from __future__ import annotations

import random
import string
import uuid

from aurelius.model.dp_aware_moe_routing import DPAwareMoERouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_session_id(length: int = 16) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


# ---------------------------------------------------------------------------
# Test 1: same session always gets same rank
# ---------------------------------------------------------------------------


def test_same_session_same_rank():
    router = DPAwareMoERouter(num_dp_ranks=8)
    session_id = "session-abc-123"
    rank1 = router.rank_for_session(session_id)
    rank2 = router.rank_for_session(session_id)
    assert rank1 == rank2, "Same session_id must always map to the same rank"


# ---------------------------------------------------------------------------
# Test 2: rank is always within valid range for a single call
# ---------------------------------------------------------------------------


def test_rank_in_valid_range():
    router = DPAwareMoERouter(num_dp_ranks=8)
    rank = router.rank_for_session("test-session")
    assert 0 <= rank < 8


# ---------------------------------------------------------------------------
# Test 3: 1000 random sessions — all ranks in [0, 8)
# ---------------------------------------------------------------------------


def test_rank_in_range_many():
    router = DPAwareMoERouter(num_dp_ranks=8)
    for _ in range(1000):
        sid = _random_session_id()
        rank = router.rank_for_session(sid)
        assert 0 <= rank < 8, f"rank {rank} out of [0, 8) for session {sid!r}"


# ---------------------------------------------------------------------------
# Test 4: num_dp_ranks=1 always returns rank 0
# ---------------------------------------------------------------------------


def test_num_dp_ranks_1():
    router = DPAwareMoERouter(num_dp_ranks=1)
    for _ in range(50):
        sid = _random_session_id()
        assert router.rank_for_session(sid) == 0


# ---------------------------------------------------------------------------
# Test 5: distribution is roughly uniform over 10000 sessions
# ---------------------------------------------------------------------------


def test_distribution_uniform():
    num_ranks = 8
    n_sessions = 10_000
    router = DPAwareMoERouter(num_dp_ranks=num_ranks)

    counts = [0] * num_ranks
    for _ in range(n_sessions):
        sid = str(uuid.uuid4())
        counts[router.rank_for_session(sid)] += 1

    n_sessions / num_ranks
    for i, count in enumerate(counts):
        fraction = count / n_sessions
        assert 0.05 <= fraction <= 0.20, (
            f"Rank {i} has fraction {fraction:.3f} — not in [5%, 20%]. Distribution may be skewed."
        )


# ---------------------------------------------------------------------------
# Test 6: empty string session_id is deterministic and does not crash
# ---------------------------------------------------------------------------


def test_empty_session_id():
    router = DPAwareMoERouter(num_dp_ranks=8)
    rank1 = router.rank_for_session("")
    rank2 = router.rank_for_session("")
    assert rank1 == rank2
    assert 0 <= rank1 < 8


# ---------------------------------------------------------------------------
# Test 7: unicode session_id returns a valid rank
# ---------------------------------------------------------------------------


def test_unicode_session_id():
    router = DPAwareMoERouter(num_dp_ranks=8)
    rank = router.rank_for_session("用户-42")
    assert 0 <= rank < 8


# ---------------------------------------------------------------------------
# Test 8: route() returns expert_ids unchanged
# ---------------------------------------------------------------------------


def test_route_returns_experts_unchanged():
    router = DPAwareMoERouter(num_dp_ranks=8)
    experts = [3, 7, 12]
    _, returned_experts = router.route("s1", experts)
    assert returned_experts == experts, "route() must not modify expert_ids"


# ---------------------------------------------------------------------------
# Test 9: route_batch length matches input length
# ---------------------------------------------------------------------------


def test_route_batch_length():
    router = DPAwareMoERouter(num_dp_ranks=8)
    session_ids = [f"session-{i}" for i in range(5)]
    expert_ids_batch = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 3]]
    results = router.route_batch(session_ids, expert_ids_batch)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Test 10: route_batch ranks match individual route() calls
# ---------------------------------------------------------------------------


def test_route_batch_consistent():
    router = DPAwareMoERouter(num_dp_ranks=8)
    session_ids = [f"batch-session-{i}" for i in range(10)]
    expert_ids_batch = [[i % 4] for i in range(10)]

    batch_results = router.route_batch(session_ids, expert_ids_batch)

    for i, (sid, eids) in enumerate(zip(session_ids, expert_ids_batch)):
        expected_rank, _ = router.route(sid, eids)
        actual_rank, actual_experts = batch_results[i]
        assert actual_rank == expected_rank, (
            f"Batch rank {actual_rank} != individual rank {expected_rank} for session {sid!r}"
        )
        assert actual_experts == eids


# ---------------------------------------------------------------------------
# Test 11: 100 distinct sessions — not all mapped to the same rank (probabilistic)
# ---------------------------------------------------------------------------


def test_different_sessions_may_differ():
    router = DPAwareMoERouter(num_dp_ranks=8)
    sessions = [f"unique-session-{i}" for i in range(100)]
    ranks = {router.rank_for_session(s) for s in sessions}
    # With 100 sessions and 8 ranks, probability all land on 1 rank is negligible
    assert len(ranks) > 1, (
        "All 100 distinct sessions mapped to the same rank — hashing appears broken"
    )


# ---------------------------------------------------------------------------
# Test 12: num_dp_ranks=16 — all ranks in [0, 16)
# ---------------------------------------------------------------------------


def test_num_dp_ranks_16():
    router = DPAwareMoERouter(num_dp_ranks=16)
    for _ in range(1000):
        sid = _random_session_id()
        rank = router.rank_for_session(sid)
        assert 0 <= rank < 16, f"rank {rank} out of [0, 16)"


# ---------------------------------------------------------------------------
# Bonus test 13: route() rank matches rank_for_session()
# ---------------------------------------------------------------------------


def test_route_rank_matches_rank_for_session():
    router = DPAwareMoERouter(num_dp_ranks=8)
    sid = "consistency-check"
    expected_rank = router.rank_for_session(sid)
    actual_rank, _ = router.route(sid, [1, 2])
    assert actual_rank == expected_rank


# ---------------------------------------------------------------------------
# Bonus test 14: sha256 hash_algo also works
# ---------------------------------------------------------------------------


def test_sha256_hash_algo():
    router = DPAwareMoERouter(num_dp_ranks=8, hash_algo="sha256")
    sid = "sha256-session"
    rank1 = router.rank_for_session(sid)
    rank2 = router.rank_for_session(sid)
    assert rank1 == rank2
    assert 0 <= rank1 < 8
