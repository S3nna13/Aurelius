"""Tests for src/cli/log_tailer.py."""

from __future__ import annotations

import threading
import time

import pytest

from src.cli.log_tailer import LogTailer, LogTailerError


class TestTail:
    def test_tail_returns_correct_lines(self, tmp_path):
        path = tmp_path / "test.log"
        path.write_text("line1\nline2\nline3\nline4\n")
        tailer = LogTailer(str(path))
        assert tailer.tail(2) == ["line3", "line4"]

    def test_tail_default_n(self, tmp_path):
        path = tmp_path / "test.log"
        path.write_text("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\n")
        tailer = LogTailer(str(path))
        result = tailer.tail()
        assert len(result) == 10
        assert result == ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

    def test_tail_empty_file(self, tmp_path):
        path = tmp_path / "empty.log"
        path.write_text("")
        tailer = LogTailer(str(path))
        assert tailer.tail(5) == []

    def test_tail_n_larger_than_file(self, tmp_path):
        path = tmp_path / "test.log"
        path.write_text("one\ntwo\n")
        tailer = LogTailer(str(path))
        assert tailer.tail(10) == ["one", "two"]


class TestFollow:
    def test_follow_yields_new_lines(self, tmp_path):
        path = tmp_path / "follow.log"
        path.write_text("initial\n")
        tailer = LogTailer(str(path))

        collected: list[str] = []

        def writer():
            time.sleep(0.1)
            with open(path, "a") as f:
                f.write("new1\n")
                f.write("new2\n")

        t = threading.Thread(target=writer)
        t.start()

        for line in tailer.follow(timeout=1.0):
            collected.append(line)

        t.join()
        assert collected == ["new1", "new2"]

    def test_follow_respects_timeout(self, tmp_path):
        path = tmp_path / "follow.log"
        path.write_text("start\n")
        tailer = LogTailer(str(path))

        start = time.time()
        list(tailer.follow(timeout=0.2))
        elapsed = time.time() - start
        assert elapsed < 0.5


class TestSearch:
    def test_search_regex(self, tmp_path):
        path = tmp_path / "test.log"
        path.write_text("error: disk full\ninfo: started\nerror: timeout\n")
        tailer = LogTailer(str(path))
        results = tailer.search(r"^error:")
        assert len(results) == 2
        assert results[0] == "error: disk full"
        assert results[1] == "error: timeout"

    def test_search_no_matches(self, tmp_path):
        path = tmp_path / "test.log"
        path.write_text("foo\nbar\n")
        tailer = LogTailer(str(path))
        assert tailer.search(r"baz") == []


class TestPathValidation:
    def test_path_traversal_rejected(self):
        with pytest.raises(LogTailerError):
            LogTailer("../outside.log")

    def test_file_not_found(self):
        with pytest.raises(LogTailerError):
            LogTailer("/nonexistent/path/file.log")

    def test_directory_rejected(self, tmp_path):
        with pytest.raises(LogTailerError):
            LogTailer(str(tmp_path))


class TestCache:
    def test_clear_cache(self, tmp_path):
        path = tmp_path / "test.log"
        path.write_text("a\nb\nc\n")
        tailer = LogTailer(str(path))
        tailer.tail(1)
        assert len(tailer._lines) > 0
        tailer.clear_cache()
        assert len(tailer._lines) == 0


class TestMaxLines:
    def test_max_lines_limits_cache(self, tmp_path):
        path = tmp_path / "test.log"
        path.write_text("\n".join(str(i) for i in range(15)) + "\n")
        tailer = LogTailer(str(path), max_lines=5)
        tailer.tail(10)
        assert len(tailer._lines) == 5
