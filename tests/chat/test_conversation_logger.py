"""Tests for src.chat.conversation_logger."""

from __future__ import annotations

import json
import time

import pytest

from src.chat.conversation_logger import (
    CONVERSATION_LOGGER_REGISTRY,
    ConversationLogger,
)


class TestConversationLogger:
    def test_log_and_retrieve(self, tmp_path):
        logger = ConversationLogger(str(tmp_path), max_files=10)
        logger.log_turn("sess-1", "user", "hello", timestamp=1.0)
        logger.log_turn("sess-1", "assistant", "hi there", timestamp=2.0)

        history = logger.get_history("sess-1")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "hello", "timestamp": 1.0}
        assert history[1] == {"role": "assistant", "content": "hi there", "timestamp": 2.0}

    def test_list_sessions(self, tmp_path):
        logger = ConversationLogger(str(tmp_path), max_files=10)
        logger.log_turn("alpha", "user", "a", timestamp=1.0)
        logger.log_turn("beta", "user", "b", timestamp=2.0)
        logger.log_turn("gamma", "user", "c", timestamp=3.0)

        sessions = logger.list_sessions()
        assert sessions == ["alpha", "beta", "gamma"]

    def test_delete_session(self, tmp_path):
        logger = ConversationLogger(str(tmp_path), max_files=10)
        logger.log_turn("sess-del", "user", "bye", timestamp=1.0)
        assert logger.get_history("sess-del")

        logger.delete_session("sess-del")
        assert logger.get_history("sess-del") == []
        assert "sess-del" not in logger.list_sessions()

    def test_delete_session_idempotent(self, tmp_path):
        logger = ConversationLogger(str(tmp_path), max_files=10)
        # Should not raise when file does not exist
        logger.delete_session("nonexistent")

    def test_invalid_session_id_raises(self, tmp_path):
        logger = ConversationLogger(str(tmp_path), max_files=10)
        invalid_ids = ["", "sess/with/slash", "sess.space", "sess:colon", "sess*star"]
        for sid in invalid_ids:
            with pytest.raises(ValueError):
                logger.log_turn(sid, "user", "x")
            with pytest.raises(ValueError):
                logger.get_history(sid)
            with pytest.raises(ValueError):
                logger.delete_session(sid)

    def test_rotation_deletes_oldest(self, tmp_path):
        logger = ConversationLogger(str(tmp_path), max_files=3)
        # Create three sessions with staggered mtimes
        logger.log_turn("old-1", "user", "1", timestamp=1.0)
        time.sleep(0.05)
        logger.log_turn("old-2", "user", "2", timestamp=2.0)
        time.sleep(0.05)
        logger.log_turn("old-3", "user", "3", timestamp=3.0)

        assert logger.list_sessions() == ["old-1", "old-2", "old-3"]

        # Adding a 4th session should evict the oldest (old-1)
        time.sleep(0.05)
        logger.log_turn("new-4", "user", "4", timestamp=4.0)
        sessions = logger.list_sessions()
        assert "old-1" not in sessions
        assert "old-2" in sessions
        assert "old-3" in sessions
        assert "new-4" in sessions
        assert len(sessions) == 3

    def test_rotation_ignores_existing_file(self, tmp_path):
        logger = ConversationLogger(str(tmp_path), max_files=2)
        logger.log_turn("a", "user", "1", timestamp=1.0)
        time.sleep(0.05)
        logger.log_turn("b", "user", "2", timestamp=2.0)

        # Both slots are full, but appending to an existing file is fine
        logger.log_turn("a", "user", "3", timestamp=3.0)
        assert logger.list_sessions() == ["a", "b"]
        assert len(logger.get_history("a")) == 2

    def test_get_history_file_not_found(self, tmp_path):
        logger = ConversationLogger(str(tmp_path), max_files=10)
        history = logger.get_history("missing")
        assert history == []

    def test_json_lines_format(self, tmp_path):
        logger = ConversationLogger(str(tmp_path), max_files=10)
        logger.log_turn("fmt", "system", "start", timestamp=5.0)

        path = tmp_path / "fmt.jsonl"
        with path.open("r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0]) == {
            "role": "system",
            "content": "start",
            "timestamp": 5.0,
        }

    def test_default_timestamp_populated(self, tmp_path):
        before = time.time()
        logger = ConversationLogger(str(tmp_path), max_files=10)
        logger.log_turn("ts", "user", "hello")
        after = time.time()

        history = logger.get_history("ts")
        assert len(history) == 1
        assert before <= history[0]["timestamp"] <= after

    def test_registry_contains_default_logger(self):
        assert "default" in CONVERSATION_LOGGER_REGISTRY
        assert isinstance(CONVERSATION_LOGGER_REGISTRY["default"], ConversationLogger)
