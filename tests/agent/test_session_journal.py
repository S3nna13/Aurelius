"""Unit tests for ``src.agent.session_journal``."""

from __future__ import annotations

import json

from src.agent.session_journal import SessionJournal


def test_session_journal_branch_and_compaction_round_trip_is_json_safe():
    journal = SessionJournal.create("session-1")

    first = journal.append(
        kind="session.created",
        summary="Created session",
        payload={"workspace": "/tmp/workspace"},
        tags=("setup",),
    )
    second = journal.append(
        kind="thread.registered",
        summary="Registered thread",
        thread_id="thread-1",
        payload={"thread_id": "thread-1", "mode": "code"},
    )
    branch = journal.branch("review", from_entry_id=second.entry_id)
    branch_note = journal.append(
        kind="journal.note",
        summary="Branch note",
        branch_id=branch.branch_id,
        thread_id="thread-1",
        payload={"note": "forked"},
    )
    compaction = journal.compact(branch_id=branch.branch_id, keep_last_n=1)

    snapshot = journal.snapshot()
    json.dumps(snapshot)

    loaded = SessionJournal.from_dict(snapshot)

    assert journal.describe()["entries"] == len(journal.entries)
    assert journal.describe()["branches"] >= 2
    assert branch.branch_id in journal.branches
    assert first.entry_id in {entry.entry_id for entry in journal.entries}
    assert branch_note.entry_id in {entry.entry_id for entry in journal.entries}
    assert journal.get_branch(branch.branch_id).head_entry_id == compaction.summary_entry_id
    assert compaction.summary_entry_id is not None
    assert loaded.get_entry(second.entry_id) is not None
    assert loaded.get_branch(branch.branch_id) is not None
    assert loaded.describe()["latest_compaction_id"] == compaction.compaction_id
