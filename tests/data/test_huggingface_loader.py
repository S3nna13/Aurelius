"""Tests for src/data/huggingface_loader.py

All tests use mock data generators — no network requests are made.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.data.huggingface_loader import (
    HFDatasetConfig,
    HuggingFaceLoader,
    InstructionSample,
    PreferencePair,
    mock_alpaca_data,
    mock_hh_rlhf_data,
    mock_lima_data,
    mock_oasst_data,
    mock_ultrafeedback_data,
    parse_alpaca_sample,
    parse_hh_rlhf_sample,
    parse_lima_sample,
    parse_oasst_sample,
    parse_ultrafeedback_sample,
)


# ---------------------------------------------------------------------------
# 1. mock_hh_rlhf_data has correct fields
# ---------------------------------------------------------------------------

def test_mock_hh_rlhf_data_fields():
    data = mock_hh_rlhf_data(4)
    assert len(data) == 4
    for record in data:
        assert "chosen" in record, "hh-rlhf records must have a 'chosen' field"
        assert "rejected" in record, "hh-rlhf records must have a 'rejected' field"
        assert isinstance(record["chosen"], str)
        assert isinstance(record["rejected"], str)


# ---------------------------------------------------------------------------
# 2. parse_hh_rlhf_sample returns PreferencePair with non-empty fields
# ---------------------------------------------------------------------------

def test_parse_hh_rlhf_sample_returns_preference_pair():
    raw = mock_hh_rlhf_data(1)[0]
    result = parse_hh_rlhf_sample(raw)
    assert isinstance(result, PreferencePair)
    assert len(result.prompt) > 0, "prompt should be non-empty"
    assert len(result.chosen) > 0, "chosen should be non-empty"
    assert len(result.rejected) > 0, "rejected should be non-empty"


# ---------------------------------------------------------------------------
# 3. parse_hh_rlhf_sample: prompt does NOT include the final response text
# ---------------------------------------------------------------------------

def test_parse_hh_rlhf_sample_prompt_excludes_response():
    raw = {
        "chosen": "\n\nHuman: What is 2+2?\n\nAssistant: The answer is 4.",
        "rejected": "\n\nHuman: What is 2+2?\n\nAssistant: I don't know.",
    }
    result = parse_hh_rlhf_sample(raw)
    # The chosen response text should NOT appear inside the prompt
    assert "The answer is 4." not in result.prompt
    assert "I don't know." not in result.prompt
    # The chosen response should be what follows the prompt
    assert "The answer is 4." in result.chosen
    assert "I don't know." in result.rejected


# ---------------------------------------------------------------------------
# 4. mock_alpaca_data has correct fields
# ---------------------------------------------------------------------------

def test_mock_alpaca_data_fields():
    data = mock_alpaca_data(4)
    assert len(data) == 4
    for record in data:
        assert "instruction" in record
        assert "input" in record
        assert "output" in record
        assert "text" in record


# ---------------------------------------------------------------------------
# 5. parse_alpaca_sample returns InstructionSample
# ---------------------------------------------------------------------------

def test_parse_alpaca_sample_returns_instruction_sample():
    raw = mock_alpaca_data(1)[0]
    result = parse_alpaca_sample(raw)
    assert isinstance(result, InstructionSample)
    assert result.instruction == raw["instruction"]
    assert result.output == raw["output"]


# ---------------------------------------------------------------------------
# 6. parse_alpaca_sample: empty input → input_context=""
# ---------------------------------------------------------------------------

def test_parse_alpaca_sample_empty_input_context():
    raw = {
        "instruction": "Translate 'hello' to French.",
        "input": "",
        "output": "Bonjour.",
        "text": "...",
    }
    result = parse_alpaca_sample(raw)
    assert result.input_context == ""


# ---------------------------------------------------------------------------
# 7. mock_oasst_data has correct fields
# ---------------------------------------------------------------------------

def test_mock_oasst_data_fields():
    data = mock_oasst_data(4)
    assert len(data) == 4
    for record in data:
        assert "message_id" in record
        assert "role" in record
        assert "text" in record
        assert "deleted" in record
        assert isinstance(record["deleted"], bool)


# ---------------------------------------------------------------------------
# 8. parse_oasst_sample returns None for deleted=True
# ---------------------------------------------------------------------------

def test_parse_oasst_sample_returns_none_for_deleted():
    raw = {
        "message_id": "msg-0001",
        "parent_id": None,
        "text": "Some assistant text.",
        "role": "assistant",
        "lang": "en",
        "deleted": True,
        "rank": 0,
        "message_tree_id": "tree-0001",
    }
    assert parse_oasst_sample(raw) is None


# ---------------------------------------------------------------------------
# 9. parse_oasst_sample returns InstructionSample for role=assistant, deleted=False
# ---------------------------------------------------------------------------

def test_parse_oasst_sample_assistant_not_deleted():
    raw = {
        "message_id": "msg-0002",
        "parent_id": "msg-0001",
        "text": "Here is my answer.",
        "role": "assistant",
        "lang": "en",
        "deleted": False,
        "rank": 0,
        "message_tree_id": "tree-0001",
    }
    result = parse_oasst_sample(raw)
    assert result is not None
    assert isinstance(result, InstructionSample)
    assert result.instruction == "Here is my answer."


# ---------------------------------------------------------------------------
# 10. mock_lima_data has correct fields
# ---------------------------------------------------------------------------

def test_mock_lima_data_fields():
    data = mock_lima_data(4)
    assert len(data) == 4
    for record in data:
        assert "source" in record
        assert "conversations" in record
        assert isinstance(record["conversations"], list)


# ---------------------------------------------------------------------------
# 11. parse_lima_sample conversation list is non-empty
# ---------------------------------------------------------------------------

def test_parse_lima_sample_conversation_non_empty():
    raw = mock_lima_data(1)[0]
    result = parse_lima_sample(raw)
    assert isinstance(result, InstructionSample)
    assert len(result.conversation) > 0, "conversation list should not be empty"
    # First turn is the user, second is the assistant
    assert result.conversation[0]["role"] == "user"
    assert result.conversation[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# 12. mock_ultrafeedback_data has completions list
# ---------------------------------------------------------------------------

def test_mock_ultrafeedback_data_has_completions():
    data = mock_ultrafeedback_data(4)
    assert len(data) == 4
    for record in data:
        assert "completions" in record
        assert isinstance(record["completions"], list)
        assert len(record["completions"]) >= 2, "need at least 2 completions"
        # Each completion should have model and response keys
        for comp in record["completions"]:
            assert "model" in comp
            assert "response" in comp


# ---------------------------------------------------------------------------
# 13. parse_ultrafeedback_sample returns list[PreferencePair]
# ---------------------------------------------------------------------------

def test_parse_ultrafeedback_sample_returns_list_of_preference_pairs():
    raw = mock_ultrafeedback_data(1)[0]
    results = parse_ultrafeedback_sample(raw)
    assert isinstance(results, list)
    assert len(results) >= 1
    for item in results:
        assert isinstance(item, PreferencePair)
        assert len(item.prompt) > 0
        assert len(item.chosen) > 0
        assert len(item.rejected) > 0


# ---------------------------------------------------------------------------
# 14. HuggingFaceLoader with patched _data → as_preference_pairs works
# ---------------------------------------------------------------------------

def test_huggingface_loader_as_preference_pairs_with_patched_data():
    config = HFDatasetConfig(dataset_name="Anthropic/hh-rlhf", max_samples=4)
    loader = HuggingFaceLoader(config)
    loader._data = mock_hh_rlhf_data(4)  # bypass load()

    pairs = loader.as_preference_pairs()
    assert isinstance(pairs, list)
    assert len(pairs) == 4
    for pair in pairs:
        assert isinstance(pair, PreferencePair)
        assert len(pair.prompt) > 0
        assert len(pair.chosen) > 0
        assert len(pair.rejected) > 0


# ---------------------------------------------------------------------------
# 15. HuggingFaceLoader.to_tokenized returns list of dicts with tensor values
# ---------------------------------------------------------------------------

def test_huggingface_loader_to_tokenized_returns_tensors():
    config = HFDatasetConfig(dataset_name="Anthropic/hh-rlhf", max_samples=4)
    loader = HuggingFaceLoader(config)
    loader._data = mock_hh_rlhf_data(4)  # bypass load()

    # Minimal character-level tokenizer for testing
    def simple_tokenizer(text: str) -> list[int]:
        return [ord(c) % 256 for c in text]

    results = loader.to_tokenized(simple_tokenizer, max_len=64)

    assert isinstance(results, list)
    assert len(results) == 4
    for item in results:
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], Tensor)
        assert isinstance(item["labels"], Tensor)
        assert item["input_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.long
        assert item["input_ids"].shape[0] <= 64
        assert item["labels"].shape == item["input_ids"].shape
