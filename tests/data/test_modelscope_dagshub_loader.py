"""Tests for modelscope_dagshub_loader — no network required."""
from __future__ import annotations
import pytest
from src.data.modelscope_dagshub_loader import (
    # mock generators
    mock_modelscope_samples,
    mock_modelscope_model_cards,
    mock_dagshub_mlruns,
    mock_dagshub_datafiles,
    # parsers
    parse_modelscope_sample,
    parse_modelscope_model_card,
    parse_dagshub_mlrun,
    parse_dagshub_datafile,
    # filters
    filter_by_quality,
    filter_by_language,
    # converters
    modelscope_to_sft_format,
    mlrun_to_summary,
)


# ---------------------------------------------------------------------------
# 1. mock_modelscope_samples has all required fields
# ---------------------------------------------------------------------------

def test_mock_modelscope_samples_required_fields():
    samples = mock_modelscope_samples(4)
    assert len(samples) == 4
    required = {"id", "instruction", "input", "output", "history", "quality_score", "token_count"}
    for s in samples:
        for key in required:
            assert key in s, f"Missing field '{key}' in sample"


# ---------------------------------------------------------------------------
# 2. parse_modelscope_sample quality_score is float
# ---------------------------------------------------------------------------

def test_parse_modelscope_sample_quality_score_is_float():
    raw = mock_modelscope_samples(1)[0]
    sample = parse_modelscope_sample(raw)
    assert isinstance(sample.quality_score, float)


# ---------------------------------------------------------------------------
# 3. parse_modelscope_sample history is list
# ---------------------------------------------------------------------------

def test_parse_modelscope_sample_history_is_list():
    raw = mock_modelscope_samples(2)
    for r in raw:
        sample = parse_modelscope_sample(r)
        assert isinstance(sample.history, list)


# ---------------------------------------------------------------------------
# 4. mock_modelscope_model_cards has model_id, task, metrics
# ---------------------------------------------------------------------------

def test_mock_modelscope_model_cards_required_fields():
    cards = mock_modelscope_model_cards(4)
    assert len(cards) == 4
    for c in cards:
        assert "model_id" in c
        assert "task" in c
        assert "metrics" in c


# ---------------------------------------------------------------------------
# 5. parse_modelscope_model_card languages is list
# ---------------------------------------------------------------------------

def test_parse_modelscope_model_card_languages_is_list():
    raws = mock_modelscope_model_cards(3)
    for raw in raws:
        card = parse_modelscope_model_card(raw)
        assert isinstance(card.languages, list)


# ---------------------------------------------------------------------------
# 6. mock_dagshub_mlruns has run_id, metrics, params, start/end time
# ---------------------------------------------------------------------------

def test_mock_dagshub_mlruns_required_fields():
    runs = mock_dagshub_mlruns(4)
    assert len(runs) == 4
    for r in runs:
        assert "run_id" in r
        assert "metrics" in r
        assert "params" in r
        assert "start_time" in r
        assert "end_time" in r


# ---------------------------------------------------------------------------
# 7. parse_dagshub_mlrun duration_seconds = end - start
# ---------------------------------------------------------------------------

def test_parse_dagshub_mlrun_duration_seconds():
    raw = mock_dagshub_mlruns(1)[0]
    run = parse_dagshub_mlrun(raw)
    expected = float(raw["end_time"] - raw["start_time"])
    assert run.duration_seconds == expected


# ---------------------------------------------------------------------------
# 8. mock_dagshub_datafiles has path, md5, size
# ---------------------------------------------------------------------------

def test_mock_dagshub_datafiles_required_fields():
    files = mock_dagshub_datafiles(4)
    assert len(files) == 4
    for f in files:
        assert "path" in f
        assert "md5" in f
        assert "size" in f


# ---------------------------------------------------------------------------
# 9. parse_dagshub_datafile size_bytes is int
# ---------------------------------------------------------------------------

def test_parse_dagshub_datafile_size_bytes_is_int():
    raws = mock_dagshub_datafiles(2)
    for raw in raws:
        df = parse_dagshub_datafile(raw)
        assert isinstance(df.size_bytes, int)


# ---------------------------------------------------------------------------
# 10. filter_by_quality removes low-score samples
# ---------------------------------------------------------------------------

def test_filter_by_quality_removes_low_scores():
    raws = mock_modelscope_samples(4)
    samples = [parse_modelscope_sample(r) for r in raws]
    # force one sample to have a very low score
    samples[0].quality_score = 0.3
    filtered = filter_by_quality(samples, min_score=0.8)
    assert all(s.quality_score >= 0.8 for s in filtered)
    assert len(filtered) < len(samples)


# ---------------------------------------------------------------------------
# 11. filter_by_language returns only matching language
# ---------------------------------------------------------------------------

def test_filter_by_language_returns_matching_only():
    raws = mock_modelscope_samples(4)
    samples = [parse_modelscope_sample(r) for r in raws]
    # At least one sample should have language "zh" (index 0 and 4)
    zh_samples = filter_by_language(samples, "zh")
    assert all(s.language == "zh" for s in zh_samples)
    assert len(zh_samples) > 0


# ---------------------------------------------------------------------------
# 12. modelscope_to_sft_format returns dict with 'messages' list
# ---------------------------------------------------------------------------

def test_modelscope_to_sft_format_structure():
    raw = mock_modelscope_samples(1)[0]
    sample = parse_modelscope_sample(raw)
    result = modelscope_to_sft_format(sample)
    assert isinstance(result, dict)
    assert "messages" in result
    assert isinstance(result["messages"], list)
    roles = [m["role"] for m in result["messages"]]
    assert "user" in roles
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# 13. mlrun_to_summary has duration_min and best_metric
# ---------------------------------------------------------------------------

def test_mlrun_to_summary_has_required_keys():
    raw = mock_dagshub_mlruns(1)[0]
    run = parse_dagshub_mlrun(raw)
    summary = mlrun_to_summary(run)
    assert "run_id" in summary
    assert "status" in summary
    assert "duration_min" in summary
    assert "best_metric" in summary


# ---------------------------------------------------------------------------
# 14. mlrun_to_summary duration_min = duration_seconds / 60
# ---------------------------------------------------------------------------

def test_mlrun_to_summary_duration_min_calculation():
    raw = mock_dagshub_mlruns(1)[0]
    run = parse_dagshub_mlrun(raw)
    summary = mlrun_to_summary(run)
    expected_min = run.duration_seconds / 60.0
    assert summary["duration_min"] == pytest.approx(expected_min)


# ---------------------------------------------------------------------------
# Bonus: best_metric is the key with the highest value
# ---------------------------------------------------------------------------

def test_mlrun_to_summary_best_metric_is_highest():
    raw = mock_dagshub_mlruns(1)[0]
    run = parse_dagshub_mlrun(raw)
    summary = mlrun_to_summary(run)
    best_key = summary["best_metric"]
    best_val = run.metrics[best_key]
    assert all(best_val >= v for v in run.metrics.values())
