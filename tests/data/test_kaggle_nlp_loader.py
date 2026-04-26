"""Tests for src/data/kaggle_nlp_loader.py.

All tests are pure Python — no network, no torch.
"""

from __future__ import annotations

from src.data.kaggle_nlp_loader import (
    DISCOURSE_TYPES,
    TOXICITY_LABEL_NAMES,
    DiscourseElement,
    JigsawSample,
    QALabelSample,
    discourse_to_instruction,
    filter_toxic,
    jigsaw_to_instruction,
    mock_discourse_data,
    mock_jigsaw_data,
    mock_qa_data,
    parse_discourse_element,
    parse_jigsaw_sample,
    parse_qa_sample,
    qa_to_instruction,
)

# ---------------------------------------------------------------------------
# 1. mock_jigsaw_data has correct fields
# ---------------------------------------------------------------------------


def test_mock_jigsaw_data_fields():
    data = mock_jigsaw_data(4)
    assert len(data) == 4
    required = {
        "id",
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    }
    for row in data:
        assert required.issubset(row.keys()), f"Missing fields: {required - row.keys()}"


# ---------------------------------------------------------------------------
# 2. parse_jigsaw_sample: toxic field is 0 or 1
# ---------------------------------------------------------------------------


def test_parse_jigsaw_sample_toxic_binary():
    data = mock_jigsaw_data(8)
    for row in data:
        sample = parse_jigsaw_sample(row)
        assert isinstance(sample, JigsawSample)
        assert sample.toxic in (0, 1), f"Expected 0 or 1, got {sample.toxic}"
        for label in TOXICITY_LABEL_NAMES:
            assert getattr(sample, label) in (0, 1)


# ---------------------------------------------------------------------------
# 3. JigsawSample.any_toxic works correctly
# ---------------------------------------------------------------------------


def test_any_toxic_true_when_label_set():
    s = JigsawSample(
        id="t1", text="bad", toxic=1, severe_toxic=0, obscene=0, threat=0, insult=0, identity_hate=0
    )
    assert s.any_toxic is True


def test_any_toxic_false_when_all_zero():
    s = JigsawSample(
        id="t2",
        text="fine",
        toxic=0,
        severe_toxic=0,
        obscene=0,
        threat=0,
        insult=0,
        identity_hate=0,
    )
    assert s.any_toxic is False


def test_any_toxic_true_for_non_primary_label():
    s = JigsawSample(
        id="t3",
        text="hateful",
        toxic=0,
        severe_toxic=0,
        obscene=0,
        threat=0,
        insult=0,
        identity_hate=1,
    )
    assert s.any_toxic is True


# ---------------------------------------------------------------------------
# 4. JigsawSample.toxicity_labels returns active label names
# ---------------------------------------------------------------------------


def test_toxicity_labels_returns_active_names():
    s = JigsawSample(
        id="t4",
        text="very bad",
        toxic=1,
        severe_toxic=1,
        obscene=0,
        threat=0,
        insult=1,
        identity_hate=0,
    )
    active = s.toxicity_labels
    assert "toxic" in active
    assert "severe_toxic" in active
    assert "insult" in active
    assert "obscene" not in active
    assert "threat" not in active
    assert "identity_hate" not in active


# ---------------------------------------------------------------------------
# 5. mock_qa_data has qa_id, question_title, question_body, answer
# ---------------------------------------------------------------------------


def test_mock_qa_data_fields():
    data = mock_qa_data(4)
    assert len(data) == 4
    required = {"qa_id", "question_title", "question_body", "answer"}
    for row in data:
        assert required.issubset(row.keys()), f"Missing fields: {required - row.keys()}"


# ---------------------------------------------------------------------------
# 6. parse_qa_sample: labels is a dict of floats
# ---------------------------------------------------------------------------


def test_parse_qa_sample_labels_are_floats():
    data = mock_qa_data(4)
    for row in data:
        sample = parse_qa_sample(row)
        assert isinstance(sample, QALabelSample)
        assert isinstance(sample.labels, dict)
        assert len(sample.labels) == 30
        for key, val in sample.labels.items():
            assert isinstance(val, float), f"Label {key} is not float: {type(val)}"


# ---------------------------------------------------------------------------
# 7. mock_discourse_data has discourse_type
# ---------------------------------------------------------------------------


def test_mock_discourse_data_fields():
    data = mock_discourse_data(4)
    assert len(data) == 4
    for row in data:
        assert "discourse_type" in row
        assert row["discourse_type"] in DISCOURSE_TYPES


# ---------------------------------------------------------------------------
# 8. parse_discourse_element: type in valid set
# ---------------------------------------------------------------------------


def test_parse_discourse_element_type_valid():
    data = mock_discourse_data(8)
    for row in data:
        elem = parse_discourse_element(row)
        assert isinstance(elem, DiscourseElement)
        assert elem.discourse_type in DISCOURSE_TYPES, f"Unexpected type: {elem.discourse_type!r}"


# ---------------------------------------------------------------------------
# 9. jigsaw_to_instruction output is 'toxic' or 'non-toxic'
# ---------------------------------------------------------------------------


def test_jigsaw_to_instruction_output_values():
    data = mock_jigsaw_data(8)
    for row in data:
        sample = parse_jigsaw_sample(row)
        inst = jigsaw_to_instruction(sample)
        assert "instruction" in inst and "input" in inst and "output" in inst
        assert inst["output"] in ("toxic", "non-toxic"), f"Bad output: {inst['output']!r}"
        assert inst["input"] == sample.text


def test_jigsaw_to_instruction_consistent_with_any_toxic():
    """Output == 'toxic' iff sample.any_toxic is True."""
    data = mock_jigsaw_data(8)
    for row in data:
        sample = parse_jigsaw_sample(row)
        inst = jigsaw_to_instruction(sample)
        if sample.any_toxic:
            assert inst["output"] == "toxic"
        else:
            assert inst["output"] == "non-toxic"


# ---------------------------------------------------------------------------
# 10. qa_to_instruction output matches answer
# ---------------------------------------------------------------------------


def test_qa_to_instruction_output_matches_answer():
    data = mock_qa_data(4)
    for row in data:
        sample = parse_qa_sample(row)
        inst = qa_to_instruction(sample)
        assert "instruction" in inst and "input" in inst and "output" in inst
        assert inst["output"] == sample.answer
        assert sample.question_title in inst["input"]
        assert sample.question_body in inst["input"]


# ---------------------------------------------------------------------------
# 11. discourse_to_instruction output is discourse_type
# ---------------------------------------------------------------------------


def test_discourse_to_instruction_output_is_type():
    data = mock_discourse_data(4)
    for row in data:
        elem = parse_discourse_element(row)
        inst = discourse_to_instruction(elem)
        assert "instruction" in inst and "input" in inst and "output" in inst
        assert inst["output"] == elem.discourse_type
        assert inst["input"] == elem.text


# ---------------------------------------------------------------------------
# 12. filter_toxic with "toxic" returns only toxic=1 samples
# ---------------------------------------------------------------------------


def test_filter_toxic_by_toxic_label():
    data = mock_jigsaw_data(8)
    samples = [parse_jigsaw_sample(r) for r in data]
    filtered = filter_toxic(samples, min_label="toxic")
    assert all(s.toxic == 1 for s in filtered), "Non-toxic sample slipped through"
    # Ensure we didn't accidentally drop all results when some are toxic
    toxic_count = sum(1 for s in samples if s.toxic == 1)
    assert len(filtered) == toxic_count


# ---------------------------------------------------------------------------
# 13. filter_toxic with non-existent label handles gracefully
# ---------------------------------------------------------------------------


def test_filter_toxic_unknown_label_returns_empty():
    data = mock_jigsaw_data(4)
    samples = [parse_jigsaw_sample(r) for r in data]
    result = filter_toxic(samples, min_label="nonexistent_label")
    assert result == [], f"Expected empty list, got {result}"


# ---------------------------------------------------------------------------
# 14. toxicity_labels returns empty list for all-zero sample
# ---------------------------------------------------------------------------


def test_toxicity_labels_empty_for_all_zero():
    s = JigsawSample(
        id="clean",
        text="great comment",
        toxic=0,
        severe_toxic=0,
        obscene=0,
        threat=0,
        insult=0,
        identity_hate=0,
    )
    assert s.toxicity_labels == []
