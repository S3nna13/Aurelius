"""DP-aware MoE Routing via Consistent Hashing — GLM-5 §3.2 (arXiv:2602.15763).

Maps session_id → fixed DP rank to preserve KV cache locality in multi-turn agents.
Prevents cross-rank KV cache misses that degrade multi-turn latency.

In multi-turn agentic workloads with Data Parallelism (DP), different conversation turns
from the same session may be processed by different DP ranks. This causes KV cache misses
because the cached KV from turn 1 (on rank 0) is not available when turn 2 routes to rank 3.

Solution: Consistent hashing maps session_id → fixed DP rank.
All turns from the same session always go to the same rank → KV cache hits guaranteed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class DPAwareMoERouter:
    """Data-Parallel aware MoE router using consistent hashing for session affinity.

    Attributes:
        num_dp_ranks: Number of DP ranks in the cluster.
        hash_algo: Hash algorithm for session_id → rank mapping. Defaults to
                   "md5" for speed and uniform distribution.
    """

    num_dp_ranks: int = 8
    hash_algo: str = "md5"  # fast, uniform distribution

    def rank_for_session(self, session_id: str) -> int:
        """Deterministically map session_id to a DP rank via consistent hashing.

        The same session_id always maps to the same rank, guaranteeing KV cache
        locality across all turns of a multi-turn conversation.

        Args:
            session_id: Unique identifier for the conversation/session.

        Returns:
            Integer rank in [0, num_dp_ranks).
        """
        h = hashlib.new(self.hash_algo)
        h.update(session_id.encode("utf-8"))
        digest = h.digest()
        return int.from_bytes(digest[:4], "little") % self.num_dp_ranks

    def route(
        self,
        session_id: str,
        expert_ids: list[int],
    ) -> tuple[int, list[int]]:
        """Route a single request to a DP rank, preserving expert selection.

        Expert selection is the responsibility of the upstream MoE gate and is
        passed through unchanged. This router only determines the DP rank.

        Args:
            session_id: Unique identifier for the conversation/session.
            expert_ids: Pre-selected expert indices from the MoE gate.

        Returns:
            Tuple of (assigned_rank, expert_ids). Expert selection unchanged.
        """
        rank = self.rank_for_session(session_id)
        return rank, expert_ids

    def route_batch(
        self,
        session_ids: list[str],
        expert_ids_batch: list[list[int]],
    ) -> list[tuple[int, list[int]]]:
        """Route a batch of requests, one per session.

        Args:
            session_ids: List of session identifiers.
            expert_ids_batch: Corresponding list of expert id lists.

        Returns:
            List of (rank, expert_ids) tuples, one per input pair.
        """
        return [self.route(sid, eids) for sid, eids in zip(session_ids, expert_ids_batch)]
