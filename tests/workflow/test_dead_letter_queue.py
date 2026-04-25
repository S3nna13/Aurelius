"""Tests for workflow dead letter queue."""
from __future__ import annotations

import pytest

from src.workflow.dead_letter_queue import DeadLetterQueue, DeadLetteredStep


class TestDeadLetterQueue:
    def test_enqueue_and_pending(self):
        dlq = DeadLetterQueue()
        dlq.enqueue(DeadLetteredStep("s1", "w1", "timeout", {}))
        assert dlq.pending() == 1

    def test_requeue_returns_matching(self):
        dlq = DeadLetterQueue()
        dlq.enqueue(DeadLetteredStep("s1", "w1", "err", {}))
        dlq.enqueue(DeadLetteredStep("s2", "w2", "err", {}))
        requeued = dlq.requeue("w1")
        assert len(requeued) == 1
        assert requeued[0].step_id == "s1"
        assert dlq.pending() == 1

    def test_requeue_increments_retry(self):
        dlq = DeadLetterQueue()
        dlq.enqueue(DeadLetteredStep("s1", "w1", "err", {}))
        requeued = dlq.requeue("w1")
        assert requeued[0].retry_count == 1

    def test_clear(self):
        dlq = DeadLetterQueue()
        dlq.enqueue(DeadLetteredStep("s1", "w1", "err", {}))
        dlq.clear()
        assert dlq.pending() == 0