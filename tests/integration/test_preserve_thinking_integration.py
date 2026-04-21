"""Integration test for preserve_thinking module.

Scenario:
  - Build a PreserveThinkingBuffer with max_snapshots=3.
  - Add 5 snapshots (triggering 2 evictions).
  - Verify len==3, eviction_count==2.
  - Verify get_prepend_tokens() returns the last snapshot's tokens.
  - Verify AGENT_LOOP_REGISTRY["preserve_thinking"] is wired correctly.
"""

from __future__ import annotations

import pytest

from src.agent import AGENT_LOOP_REGISTRY
from src.agent.preserve_thinking import (
    PreserveThinkingBuffer,
    PreserveThinkingConfig,
)


def test_preserve_thinking_full_integration():
    cfg = PreserveThinkingConfig(
        max_snapshots=3,
        max_tokens_per_snapshot=4096,
        use_summary=False,
        max_prepend_tokens=2048,
    )
    buf = PreserveThinkingBuffer(cfg)

    # Add 5 snapshots → 2 evictions, 3 retained
    snapshots_added = []
    for i in range(5):
        tokens = list(range(i * 10, i * 10 + 5))
        snap = buf.add_snapshot(
            turn_id=i,
            thinking_tokens=tokens,
            summary_tokens=[i],
            metadata={"turn": i},
        )
        snapshots_added.append(snap)

    # Basic length and eviction invariants
    assert len(buf) == 3, f"Expected 3, got {len(buf)}"
    assert buf.eviction_count() == 2, f"Expected 2 evictions, got {buf.eviction_count()}"

    # Oldest surviving snapshot is turn_id=2
    all_snaps = buf.get_all_snapshots()
    assert len(all_snaps) == 3
    assert all_snaps[0].turn_id == 2
    assert all_snaps[-1].turn_id == 4

    # get_prepend_tokens returns last snapshot's thinking_tokens
    last_snap = snapshots_added[-1]
    result = buf.get_prepend_tokens()
    assert result == last_snap.thinking_tokens, (
        f"Expected {last_snap.thinking_tokens}, got {result}"
    )

    # Registry wiring
    assert "preserve_thinking" in AGENT_LOOP_REGISTRY, (
        "AGENT_LOOP_REGISTRY missing 'preserve_thinking' key"
    )
    cls = AGENT_LOOP_REGISTRY["preserve_thinking"]
    assert cls is PreserveThinkingBuffer

    # Verify registry entry can be instantiated
    buf2 = cls()
    assert len(buf2) == 0
