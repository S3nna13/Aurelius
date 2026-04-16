"""Tests for src/data/platform_loaders.py — no network calls, pure Python."""
from __future__ import annotations

import pytest

from src.data.platform_loaders import (
    DagsHubRun,
    KaggleReadabilitySample,
    KaggleSentimentSample,
    ModelScopeInstructSample,
    PlatformDataMixer,
    ReplicatePrediction,
    kaggle_sentiment_to_instruction,
    mock_dagshub_runs,
    mock_kaggle_readability_data,
    mock_kaggle_sentiment_data,
    mock_modelscope_data,
    mock_replicate_predictions,
    modelscope_to_instruction,
    parse_dagshub_run,
    parse_kaggle_readability,
    parse_kaggle_sentiment,
    parse_modelscope_instruct,
    parse_replicate_prediction,
    readability_to_regression_sample,
)


# ---------------------------------------------------------------------------
# Test 1: mock_kaggle_sentiment_data has target and text fields
# ---------------------------------------------------------------------------
def test_mock_kaggle_sentiment_has_target_and_text():
    rows = mock_kaggle_sentiment_data(4)
    assert len(rows) == 4
    for row in rows:
        assert "target" in row, "Row missing 'target' field"
        assert "text" in row, "Row missing 'text' field"
        assert row["target"] in (0, 4), f"target must be 0 or 4, got {row['target']}"


# ---------------------------------------------------------------------------
# Test 2: parse_kaggle_sentiment target=0 → label=0
# ---------------------------------------------------------------------------
def test_parse_kaggle_sentiment_label_zero():
    row = {"target": 0, "id": "1", "date": "2024-01-01", "flag": "NO_QUERY",
           "user": "alice", "text": "Terrible product."}
    sample = parse_kaggle_sentiment(row)
    assert isinstance(sample, KaggleSentimentSample)
    assert sample.label == 0
    assert sample.text == "Terrible product."


# ---------------------------------------------------------------------------
# Test 3: parse_kaggle_sentiment target=4 → label=4
# ---------------------------------------------------------------------------
def test_parse_kaggle_sentiment_label_four():
    row = {"target": 4, "id": "2", "date": "2024-01-02", "flag": "NO_QUERY",
           "user": "bob", "text": "Love it!"}
    sample = parse_kaggle_sentiment(row)
    assert isinstance(sample, KaggleSentimentSample)
    assert sample.label == 4
    assert sample.text == "Love it!"


# ---------------------------------------------------------------------------
# Test 4: mock_kaggle_readability_data has excerpt, target, standard_error
# ---------------------------------------------------------------------------
def test_mock_kaggle_readability_has_required_fields():
    rows = mock_kaggle_readability_data(4)
    assert len(rows) == 4
    for row in rows:
        assert "excerpt" in row, "Row missing 'excerpt'"
        assert "target" in row, "Row missing 'target'"
        assert "standard_error" in row, "Row missing 'standard_error'"
        assert isinstance(row["excerpt"], str)
        assert isinstance(row["target"], float)
        assert isinstance(row["standard_error"], float)


# ---------------------------------------------------------------------------
# Test 5: parse_kaggle_readability returns float target
# ---------------------------------------------------------------------------
def test_parse_kaggle_readability_float_target():
    row = {"id": "r1", "url_legal": "", "license": "CC",
           "excerpt": "Some text here.", "target": -1.5, "standard_error": 0.45}
    sample = parse_kaggle_readability(row)
    assert isinstance(sample, KaggleReadabilitySample)
    assert isinstance(sample.target, float)
    assert sample.target == -1.5
    assert sample.excerpt == "Some text here."


# ---------------------------------------------------------------------------
# Test 6: mock_modelscope_data has instruction, input, output, history
# ---------------------------------------------------------------------------
def test_mock_modelscope_has_required_fields():
    rows = mock_modelscope_data(4)
    assert len(rows) == 4
    for row in rows:
        assert "instruction" in row, "Row missing 'instruction'"
        assert "input" in row, "Row missing 'input'"
        assert "output" in row, "Row missing 'output'"
        assert "history" in row, "Row missing 'history'"
        assert isinstance(row["history"], list)


# ---------------------------------------------------------------------------
# Test 7: parse_modelscope_instruct handles empty history
# ---------------------------------------------------------------------------
def test_parse_modelscope_instruct_empty_history():
    raw = {"instruction": "Summarize this.", "input": "Some text.", "output": "Summary.", "history": []}
    sample = parse_modelscope_instruct(raw)
    assert isinstance(sample, ModelScopeInstructSample)
    assert sample.history == []
    assert sample.instruction == "Summarize this."


def test_parse_modelscope_instruct_with_history():
    raw = {
        "instruction": "Answer the question.",
        "input": "What is AI?",
        "output": "Artificial Intelligence.",
        "history": [["What is ML?", "Machine Learning."]],
    }
    sample = parse_modelscope_instruct(raw)
    assert sample.history == [["What is ML?", "Machine Learning."]]


# ---------------------------------------------------------------------------
# Test 8: mock_dagshub_runs has run_id, metrics, params
# ---------------------------------------------------------------------------
def test_mock_dagshub_runs_has_required_fields():
    rows = mock_dagshub_runs(4)
    assert len(rows) == 4
    for row in rows:
        assert "run_id" in row, "Row missing 'run_id'"
        assert "metrics" in row, "Row missing 'metrics'"
        assert "params" in row, "Row missing 'params'"
        assert isinstance(row["metrics"], dict)
        assert isinstance(row["params"], dict)


# ---------------------------------------------------------------------------
# Test 9: parse_dagshub_run metrics is dict
# ---------------------------------------------------------------------------
def test_parse_dagshub_run_metrics_is_dict():
    raw = {
        "run_id": "run_0001",
        "experiment_id": "exp_001",
        "status": "FINISHED",
        "metrics": {"loss": 0.25, "accuracy": 0.92},
        "params": {"lr": "0.001"},
        "tags": {},
    }
    run = parse_dagshub_run(raw)
    assert isinstance(run, DagsHubRun)
    assert isinstance(run.metrics, dict)
    assert run.metrics["loss"] == 0.25
    assert run.metrics["accuracy"] == 0.92


# ---------------------------------------------------------------------------
# Test 10: mock_replicate_predictions has id, version, status
# ---------------------------------------------------------------------------
def test_mock_replicate_predictions_has_required_fields():
    rows = mock_replicate_predictions(4)
    assert len(rows) == 4
    for row in rows:
        assert "id" in row, "Row missing 'id'"
        assert "version" in row, "Row missing 'version'"
        assert "status" in row, "Row missing 'status'"
        assert isinstance(row["id"], str)
        assert isinstance(row["version"], str)
        assert isinstance(row["status"], str)


# ---------------------------------------------------------------------------
# Test 11: parse_replicate_prediction predict_time is float
# ---------------------------------------------------------------------------
def test_parse_replicate_prediction_predict_time_is_float():
    raw = {
        "id": "pred_00000001",
        "version": "v1.0.0",
        "status": "succeeded",
        "input": {"prompt": "Hello"},
        "output": "World",
        "error": None,
        "metrics": {"predict_time": 1.23},
    }
    pred = parse_replicate_prediction(raw)
    assert isinstance(pred, ReplicatePrediction)
    assert isinstance(pred.predict_time, float)
    assert pred.predict_time == 1.23


def test_parse_replicate_prediction_no_metrics():
    """predict_time defaults to 0.0 when metrics key is absent."""
    raw = {
        "id": "pred_00000002",
        "version": "v2.0.0",
        "status": "failed",
        "input": {},
        "output": None,
        "error": "OOM",
    }
    pred = parse_replicate_prediction(raw)
    assert isinstance(pred.predict_time, float)
    assert pred.predict_time == 0.0
    assert pred.error == "OOM"


# ---------------------------------------------------------------------------
# Test 12: kaggle_sentiment_to_instruction has instruction/input/output keys
# ---------------------------------------------------------------------------
def test_kaggle_sentiment_to_instruction_keys():
    sample = KaggleSentimentSample(text="Great day!", label=4)
    result = kaggle_sentiment_to_instruction(sample)
    assert "instruction" in result
    assert "input" in result
    assert "output" in result
    assert result["input"] == "Great day!"
    assert result["output"] == "positive"


def test_kaggle_sentiment_to_instruction_negative():
    sample = KaggleSentimentSample(text="Awful day.", label=0)
    result = kaggle_sentiment_to_instruction(sample)
    assert result["output"] == "negative"


# ---------------------------------------------------------------------------
# Test 13: readability_to_regression_sample has text/score/weight
# ---------------------------------------------------------------------------
def test_readability_to_regression_sample_keys():
    sample = KaggleReadabilitySample(excerpt="Some passage.", target=-1.0, standard_error=0.5)
    result = readability_to_regression_sample(sample)
    assert "text" in result
    assert "score" in result
    assert "weight" in result
    assert result["text"] == "Some passage."
    assert result["score"] == -1.0
    assert isinstance(result["weight"], float)
    assert result["weight"] > 0


def test_readability_to_regression_sample_weight_formula():
    """weight = 1 / (std_err^2 + 1e-8)"""
    std_err = 0.5
    sample = KaggleReadabilitySample(excerpt="x", target=0.0, standard_error=std_err)
    result = readability_to_regression_sample(sample)
    expected_weight = 1.0 / (std_err ** 2 + 1e-8)
    assert abs(result["weight"] - expected_weight) < 1e-6


# ---------------------------------------------------------------------------
# Test 14: PlatformDataMixer.add_source + stats = correct counts
# ---------------------------------------------------------------------------
def test_platform_data_mixer_stats():
    mixer = PlatformDataMixer()
    sentiment_rows = [parse_kaggle_sentiment(r) for r in mock_kaggle_sentiment_data(6)]
    read_rows = [parse_kaggle_readability(r) for r in mock_kaggle_readability_data(3)]

    mixer.add_source("sentiment", sentiment_rows)
    mixer.add_source("readability", read_rows)

    stats = mixer.stats()
    assert stats == {"sentiment": 6, "readability": 3}


# ---------------------------------------------------------------------------
# Test 15: PlatformDataMixer.mix returns combined list
# ---------------------------------------------------------------------------
def test_platform_data_mixer_mix_equal():
    mixer = PlatformDataMixer()
    a = list(range(3))
    b = list("abc")
    mixer.add_source("nums", a)
    mixer.add_source("letters", b)

    result = mixer.mix()
    assert len(result) == 6
    # All original items are present
    for item in a:
        assert item in result
    for item in b:
        assert item in result


def test_platform_data_mixer_mix_empty():
    mixer = PlatformDataMixer()
    assert mixer.mix() == []


def test_platform_data_mixer_mix_single_source():
    mixer = PlatformDataMixer()
    samples = [1, 2, 3]
    mixer.add_source("only", samples)
    result = mixer.mix()
    assert result == [1, 2, 3]


def test_platform_data_mixer_mix_with_real_samples():
    mixer = PlatformDataMixer()
    sentiment = [parse_kaggle_sentiment(r) for r in mock_kaggle_sentiment_data(4)]
    instruct = [parse_modelscope_instruct(r) for r in mock_modelscope_data(4)]
    mixer.add_source("sentiment", sentiment)
    mixer.add_source("instruct", instruct)

    result = mixer.mix()
    assert len(result) == 8
    sentiment_count = sum(1 for x in result if isinstance(x, KaggleSentimentSample))
    instruct_count = sum(1 for x in result if isinstance(x, ModelScopeInstructSample))
    assert sentiment_count == 4
    assert instruct_count == 4
