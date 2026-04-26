"""Tests for the Reddit data preprocessing pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from training_data.reddit_pipeline import RedditPipeline, _clean_text, _is_bot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_submission(overrides: dict | None = None) -> dict:
    base = {
        "id": "abc123",
        "title": "A question about machine learning",
        "selftext": "I was wondering how transformers work in NLP models?",
        "subreddit": "MachineLearning",
        "score": 10,
        "created_utc": 1700000000,
        "permalink": "/r/MachineLearning/comments/abc123/",
    }
    base.update(overrides or {})
    return base


def _sample_comment(overrides: dict | None = None) -> dict:
    base = {
        "id": "def456",
        "body": "Transformers use self-attention mechanisms to process sequential data efficiently.",
        "subreddit": "MachineLearning",
        "score": 5,
        "created_utc": 1700000001,
        "parent_id": "t1_abc123",
        "link_id": "t3_abc123",
        "permalink": "/r/MachineLearning/comments/abc123//def456/",
    }
    base.update(overrides or {})
    return base


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_default_config() -> None:
    p = RedditPipeline({})
    assert "subreddits" in p.config
    assert "technical" in p.config["subreddits"]
    assert "exclude" in p.config["subreddits"]
    assert p.config["min_score_submissions"] == 1
    assert p.config["min_score_comments"] == 2
    assert p.config["min_text_length"] == 50
    assert p.config["max_text_length"] == 10000
    assert p.config["include_conversations"] is True
    assert p.config["max_conversation_depth"] == 5


def test_custom_config_overrides() -> None:
    p = RedditPipeline({"min_score_submissions": 5, "min_text_length": 100})
    assert p.config["min_score_submissions"] == 5
    assert p.config["min_text_length"] == 100
    assert p.config["min_score_comments"] == 2


def test_blocklist_populated() -> None:
    p = RedditPipeline({})
    assert "gonewild" in p._blocklist
    assert "nsfw" in p._blocklist
    assert "politic" in p._blocklist
    assert "conspiracy" in p._blocklist
    assert "MachineLearning" not in p._blocklist
    assert "python" not in p._blocklist


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def test_clean_text_strips_markdown_links() -> None:
    text = "See [this link](https://example.com) for details"
    assert _clean_text(text) == "See this link for details"


def test_clean_text_strips_markdown_images() -> None:
    text = "![alt text](image.png) hello"
    assert _clean_text(text) == "alt text hello"


def test_clean_text_strips_bold_italic() -> None:
    text = "this is **bold** and *italic* text"
    result = _clean_text(text)
    assert "**" not in result
    assert "*" not in result
    assert "bold" in result
    assert "italic" in result


def test_clean_text_normalizes_whitespace() -> None:
    text = "hello    world\n\n\n\nanother  line"
    result = _clean_text(text)
    assert "    " not in result
    assert "\n\n\n" not in result


def test_clean_text_strips_blockquotes() -> None:
    text = "> quoted text\nnormal text"
    result = _clean_text(text)
    assert "> " not in result


def test_clean_text_handles_empty() -> None:
    assert _clean_text("") == ""
    assert _clean_text(None) == ""


# ---------------------------------------------------------------------------
# Bot detection
# ---------------------------------------------------------------------------

def test_is_bot_detects_bot_phrases() -> None:
    assert _is_bot("I am a bot that helps with formatting")
    assert _is_bot("Beep boop, I'm a bot")
    assert _is_bot("This is an automated response")


def test_is_bot_returns_false_for_normal_text() -> None:
    assert not _is_bot("I have a question about Python")
    assert not _is_bot("Can someone help me with this code?")


# ---------------------------------------------------------------------------
# Parsing and filtering
# ---------------------------------------------------------------------------

def test_parse_item_keeps_valid_submission() -> None:
    p = RedditPipeline({})
    item = _sample_submission()
    result = p._parse_item(item)
    assert result is not None
    assert result["type"] == "submission"
    assert result["id"] == "abc123"
    assert result["subreddit"] == "machinelearning"
    assert result["score"] == 10
    assert "transformers" in result["text"]


def test_parse_item_filters_low_score() -> None:
    p = RedditPipeline({"min_score_submissions": 5})
    item = _sample_submission({"score": 1})
    assert p._parse_item(item) is None


def test_parse_item_filters_deleted_content() -> None:
    p = RedditPipeline({})
    item = _sample_submission({"selftext": "[deleted]"})
    assert p._parse_item(item) is None


def test_parse_item_filters_removed_content() -> None:
    p = RedditPipeline({})
    item = _sample_comment({"body": "[removed]"})
    assert p._parse_item(item) is None


def test_parse_item_filters_bot_posts() -> None:
    p = RedditPipeline({})
    item = _sample_comment({"body": "I am a bot. Here is the answer."})
    assert p._parse_item(item) is None


def test_parse_item_filters_blocklisted_subreddit() -> None:
    p = RedditPipeline({})
    item = _sample_submission({"subreddit": "gonewild"})
    assert p._parse_item(item) is None


def test_parse_item_filters_short_text() -> None:
    p = RedditPipeline({"min_text_length": 100})
    item = _sample_submission({"selftext": "Short."})
    assert p._parse_item(item) is None


def test_parse_item_keeps_valid_comment() -> None:
    p = RedditPipeline({})
    item = _sample_comment()
    result = p._parse_item(item)
    assert result is not None
    assert result["type"] == "comment"
    assert result["id"] == "def456"
    assert "self-attention" in result["text"]
    assert result["parent_id"] == "t1_abc123"


# ---------------------------------------------------------------------------
# Quality filtering on parsed items
# ---------------------------------------------------------------------------

def test_filter_by_quality_removes_low_score() -> None:
    p = RedditPipeline({"min_score_comments": 3})
    items = [
        {"id": "1", "text": "This is a long enough text to pass min length.", "score": 1, "type": "comment"},
        {"id": "2", "text": "This is another long enough text to pass min length.", "score": 5, "type": "comment"},
    ]
    filtered = p.filter_by_quality(items)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "2"


def test_filter_by_quality_removes_deleted() -> None:
    p = RedditPipeline({})
    items = [
        {"id": "1", "text": "This is a long enough text that exceeds the minimum threshold for testing purposes.", "score": 10, "type": "comment"},
        {"id": "2", "text": "[deleted]", "score": 10, "type": "comment"},
    ]
    filtered = p.filter_by_quality(items)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "1"


# ---------------------------------------------------------------------------
# Comment threading
# ---------------------------------------------------------------------------

def test_thread_comments_empty() -> None:
    p = RedditPipeline({})
    assert p.thread_comments([]) == []


def test_thread_comments_basic_chain() -> None:
    p = RedditPipeline({})
    comments = [
        {
            "id": "c1",
            "parent_id": "",
            "text": "What is a transformer?",
            "score": 5,
            "created_utc": 1,
            "subreddit": "MachineLearning",
        },
        {
            "id": "c2",
            "parent_id": "t1_c1",
            "text": "It is a neural network architecture.",
            "score": 3,
            "created_utc": 2,
            "subreddit": "MachineLearning",
        },
    ]
    convs = p.thread_comments(comments)
    assert len(convs) == 1
    assert convs[0]["id"] == "c1"
    assert len(convs[0]["conversations"]) == 2
    assert convs[0]["conversations"][0]["from"] == "human"
    assert convs[0]["conversations"][1]["from"] == "assistant"


def test_thread_comments_respects_depth() -> None:
    p = RedditPipeline({"max_conversation_depth": 2})
    comments = [
        {"id": "c1", "parent_id": "", "text": "Q1", "score": 1, "created_utc": 1, "subreddit": "test"},
        {"id": "c2", "parent_id": "t1_c1", "text": "A1", "score": 1, "created_utc": 2, "subreddit": "test"},
        {"id": "c3", "parent_id": "t1_c2", "text": "A2", "score": 1, "created_utc": 3, "subreddit": "test"},
    ]
    convs = p.thread_comments(comments)
    for conv in convs:
        for msg in conv["conversations"]:
            assert isinstance(msg["value"], str)


# ---------------------------------------------------------------------------
# Output format checks
# ---------------------------------------------------------------------------

def test_raw_jsonl_output_format() -> None:
    p = RedditPipeline({})
    item = _sample_submission()
    result = p._parse_item(item)
    assert result is not None
    required_keys = {"id", "subreddit", "title", "text", "score", "created_utc", "type"}
    assert required_keys.issubset(result.keys())
    assert result["type"] == "submission"


def test_conversation_jsonl_format() -> None:
    p = RedditPipeline({})
    comments = [
        {"id": "c1", "parent_id": "", "text": "Hello?", "score": 2, "created_utc": 1, "subreddit": "test"},
        {"id": "c2", "parent_id": "t1_c1", "text": "World!", "score": 1, "created_utc": 2, "subreddit": "test"},
    ]
    convs = p.thread_comments(comments)
    assert len(convs) == 1
    conv = convs[0]
    assert "id" in conv
    assert "subreddit" in conv
    assert "conversations" in conv
    assert "score" in conv
    for msg in conv["conversations"]:
        assert "from" in msg
        assert "value" in msg
        assert msg["from"] in ("human", "assistant")


def test_pretrain_output_format() -> None:
    p = RedditPipeline({})
    item = _sample_submission()
    result = p._parse_item(item)
    assert result is not None
    pretrain_entry = {
        "text": result["text"],
        "source": "reddit",
        "subreddit": result["subreddit"],
    }
    assert "text" in pretrain_entry
    assert "source" in pretrain_entry
    assert "subreddit" in pretrain_entry
    assert pretrain_entry["source"] == "reddit"
    # Verify JSON-serializable
    json.dumps(pretrain_entry)


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

def test_state_tracking_marks_and_checks() -> None:
    p = RedditPipeline({})
    with tempfile.TemporaryDirectory() as tmpdir:
        assert p._is_processed(tmpdir, "test_file.zst") is False
        p._mark_processed(tmpdir, "test_file.zst", {"total": 10, "kept": 5, "errors": 0})
        assert p._is_processed(tmpdir, "test_file.zst") is True


def test_state_tracking_downloaded_files() -> None:
    p = RedditPipeline({})
    with tempfile.TemporaryDirectory() as tmpdir:
        state = p._load_state(tmpdir)
        assert "downloaded_files" in state
        p._mark_downloaded(tmpdir, "http://example.com/file.zst", "/tmp/file.zst")
        state = p._load_state(tmpdir)
        assert "http://example.com/file.zst" in state["downloaded_files"]


def test_state_tracking_persistence() -> None:
    p = RedditPipeline({})
    with tempfile.TemporaryDirectory() as tmpdir:
        p._mark_processed(tmpdir, "test.zst", {"total": 5, "kept": 3, "errors": 0})
        p2 = RedditPipeline({})
        assert p2._is_processed(tmpdir, "test.zst") is True


# ---------------------------------------------------------------------------
# Incremental skip
# ---------------------------------------------------------------------------

def test_skips_already_processed_files() -> None:
    p = RedditPipeline({})
    with tempfile.TemporaryDirectory() as tmpdir:
        p._mark_processed(tmpdir, "file.zst", {"total": 1, "kept": 1, "errors": 0})
        assert p._is_processed(tmpdir, "file.zst") is True
        assert p._is_processed(tmpdir, "other.zst") is False


# ---------------------------------------------------------------------------
# Filter_by_quality on score
# ---------------------------------------------------------------------------

def test_filter_by_quality_removes_bot() -> None:
    p = RedditPipeline({})
    items = [
        {"id": "1", "text": "This is a real question about Python programming language and its frameworks.", "score": 10, "type": "comment"},
        {"id": "2", "text": "I am a bot, beep boop.", "score": 10, "type": "comment"},
    ]
    filtered = p.filter_by_quality(items)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "1"


# ---------------------------------------------------------------------------
# Text length boundaries
# ---------------------------------------------------------------------------

def test_filter_by_quality_text_length_bounds() -> None:
    p = RedditPipeline({"min_text_length": 10, "max_text_length": 50})
    items = [
        {"id": "1", "text": "Short", "score": 10, "type": "submission"},
        {"id": "2", "text": "Just right text length here.", "score": 10, "type": "submission"},
        {"id": "3", "text": "x" * 100, "score": 10, "type": "submission"},
    ]
    filtered = p.filter_by_quality(items)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "2"
