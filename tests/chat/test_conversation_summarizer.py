"""Tests for conversation_summarizer.py"""

from src.chat.conversation_summarizer import (
    ConversationSummarizer,
    ConversationSummary,
    SummaryMode,
)

# --- SummaryMode enum ---


def test_summary_mode_extractive():
    assert SummaryMode.EXTRACTIVE == "extractive"


def test_summary_mode_abstractive_stub():
    assert SummaryMode.ABSTRACTIVE_STUB == "abstractive_stub"


def test_summary_mode_bullets():
    assert SummaryMode.BULLETS == "bullets"


def test_summary_mode_count():
    assert len(SummaryMode) == 3


def test_summary_mode_is_str():
    assert isinstance(SummaryMode.EXTRACTIVE, str)


# --- ConversationSummary dataclass ---


def test_conversation_summary_fields():
    cs = ConversationSummary(
        original_turn_count=10,
        summary_text="hello",
        key_topics=["topic1"],
        compression_ratio=0.5,
        mode=SummaryMode.EXTRACTIVE,
    )
    assert cs.original_turn_count == 10
    assert cs.summary_text == "hello"
    assert cs.key_topics == ["topic1"]
    assert cs.compression_ratio == 0.5
    assert cs.mode == SummaryMode.EXTRACTIVE


def test_conversation_summary_mode_stored():
    cs = ConversationSummary(
        original_turn_count=1,
        summary_text="x",
        key_topics=[],
        compression_ratio=1.0,
        mode=SummaryMode.BULLETS,
    )
    assert cs.mode == SummaryMode.BULLETS


# --- ConversationSummarizer.extract_topics ---


def test_extract_topics_returns_list():
    cs = ConversationSummarizer()
    result = cs.extract_topics(["hello world hello"])
    assert isinstance(result, list)


def test_extract_topics_excludes_stopwords():
    cs = ConversationSummarizer()
    result = cs.extract_topics(
        ["the a an is in to of and or for with that this it be was are were has have"]
    )
    assert all(
        w
        not in {
            "the",
            "a",
            "an",
            "is",
            "in",
            "to",
            "of",
            "and",
            "or",
            "for",
            "with",
            "that",
            "this",
            "it",
            "be",
            "was",
            "are",
            "were",
            "has",
            "have",
        }
        for w in result
    )


def test_extract_topics_at_most_5():
    cs = ConversationSummarizer()
    msgs = ["apple banana cherry date elderberry fig grape hazelnut iris jasmine"]
    result = cs.extract_topics(msgs)
    assert len(result) <= 5


def test_extract_topics_most_frequent_first():
    cs = ConversationSummarizer()
    msgs = ["python python python java java ruby"]
    result = cs.extract_topics(msgs)
    assert result[0] == "python"


def test_extract_topics_case_insensitive():
    cs = ConversationSummarizer()
    msgs = ["Python PYTHON python"]
    result = cs.extract_topics(msgs)
    assert "python" in result


def test_extract_topics_empty_messages():
    cs = ConversationSummarizer()
    result = cs.extract_topics([])
    assert result == []


def test_extract_topics_only_stopwords():
    cs = ConversationSummarizer()
    result = cs.extract_topics(["the and is a"])
    assert result == []


def test_extract_topics_returns_strings():
    cs = ConversationSummarizer()
    result = cs.extract_topics(["machine learning neural network"])
    assert all(isinstance(t, str) for t in result)


# --- summarize_extractive ---


def test_summarize_extractive_nonempty():
    cs = ConversationSummarizer()
    result = cs.summarize_extractive(["Hello there", "How are you"])
    assert result != ""


def test_summarize_extractive_uses_pipe_separator():
    cs = ConversationSummarizer()
    result = cs.summarize_extractive(["msg1", "msg2"])
    assert " | " in result or result in ["msg1", "msg2", "msg1 | msg2"]


def test_summarize_extractive_single_message():
    cs = ConversationSummarizer()
    result = cs.summarize_extractive(["only one"])
    assert "only one" in result


def test_summarize_extractive_empty():
    cs = ConversationSummarizer()
    result = cs.summarize_extractive([])
    assert result == ""


def test_summarize_extractive_includes_first():
    cs = ConversationSummarizer()
    msgs = [f"message {i}" for i in range(20)]
    result = cs.summarize_extractive(msgs)
    assert "message 0" in result


def test_summarize_extractive_includes_last():
    cs = ConversationSummarizer()
    msgs = [f"message {i}" for i in range(20)]
    result = cs.summarize_extractive(msgs)
    assert "message 19" in result


def test_summarize_extractive_max_sentences():
    cs = ConversationSummarizer(max_sentences=3)
    msgs = [f"message {i}" for i in range(10)]
    result = cs.summarize_extractive(msgs)
    parts = result.split(" | ")
    assert len(parts) <= 3


# --- summarize_bullets ---


def test_summarize_bullets_each_line_starts_with_bullet():
    cs = ConversationSummarizer(mode=SummaryMode.BULLETS)
    result = cs.summarize_bullets(["Hello world", "Goodbye moon"])
    for line in result.strip().split("\n"):
        assert line.startswith("•")


def test_summarize_bullets_nonempty():
    cs = ConversationSummarizer(mode=SummaryMode.BULLETS)
    result = cs.summarize_bullets(["some message"])
    assert result != ""


def test_summarize_bullets_truncates_to_60():
    cs = ConversationSummarizer(mode=SummaryMode.BULLETS)
    long_msg = "x" * 100
    result = cs.summarize_bullets([long_msg])
    # "• " + first 60 chars = 62 chars
    line = result.strip().split("\n")[0]
    assert len(line) <= 62


def test_summarize_bullets_empty():
    cs = ConversationSummarizer(mode=SummaryMode.BULLETS)
    result = cs.summarize_bullets([])
    assert result == ""


# --- summarize() ---


def test_summarize_extractive_mode():
    cs = ConversationSummarizer(mode=SummaryMode.EXTRACTIVE)
    msgs = ["hello world"] * 3
    result = cs.summarize(msgs)
    assert result.mode == SummaryMode.EXTRACTIVE


def test_summarize_extractive_compression_ratio_less_than_1():
    cs = ConversationSummarizer(mode=SummaryMode.EXTRACTIVE, max_sentences=3)
    msgs = [
        f"This is a longer message number {i} with extra words to make it long enough"
        for i in range(20)
    ]
    result = cs.summarize(msgs)
    assert result.compression_ratio < 1.0


def test_summarize_abstractive_stub_contains_summary_of():
    cs = ConversationSummarizer(mode=SummaryMode.ABSTRACTIVE_STUB)
    result = cs.summarize(["Hello world", "Goodbye moon"])
    assert "Summary of" in result.summary_text


def test_summarize_abstractive_stub_contains_count():
    cs = ConversationSummarizer(mode=SummaryMode.ABSTRACTIVE_STUB)
    result = cs.summarize(["Hello world", "Goodbye moon"])
    assert "2" in result.summary_text


def test_summarize_bullets_summary_contains_bullet():
    cs = ConversationSummarizer(mode=SummaryMode.BULLETS)
    result = cs.summarize(["Hello world"])
    assert "•" in result.summary_text


def test_summarize_key_topics_populated():
    cs = ConversationSummarizer(mode=SummaryMode.EXTRACTIVE)
    result = cs.summarize(["machine learning neural network machine"])
    assert len(result.key_topics) > 0


def test_summarize_key_topics_is_list():
    cs = ConversationSummarizer()
    result = cs.summarize(["python code testing"])
    assert isinstance(result.key_topics, list)


def test_summarize_original_turn_count():
    cs = ConversationSummarizer()
    msgs = ["one", "two", "three"]
    result = cs.summarize(msgs)
    assert result.original_turn_count == 3


def test_summarize_compression_ratio_is_float():
    cs = ConversationSummarizer()
    result = cs.summarize(["hello world"])
    assert isinstance(result.compression_ratio, float)


def test_summarize_empty_messages():
    cs = ConversationSummarizer()
    result = cs.summarize([])
    assert result.original_turn_count == 0
    assert isinstance(result.summary_text, str)


def test_summarize_abstractive_stub_mode_stored():
    cs = ConversationSummarizer(mode=SummaryMode.ABSTRACTIVE_STUB)
    result = cs.summarize(["test"])
    assert result.mode == SummaryMode.ABSTRACTIVE_STUB


def test_summarize_bullets_mode_stored():
    cs = ConversationSummarizer(mode=SummaryMode.BULLETS)
    result = cs.summarize(["test message"])
    assert result.mode == SummaryMode.BULLETS
