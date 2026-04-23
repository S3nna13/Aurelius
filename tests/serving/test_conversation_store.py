"""Tests for ConversationStore."""

import pytest
from src.serving.conversation_store import ConversationStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(storage_dir=str(tmp_path))


MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


def test_instantiates_with_custom_storage_dir(tmp_path):
    s = ConversationStore(storage_dir=str(tmp_path))
    assert s.storage_dir == tmp_path.resolve()


def test_save_creates_file(store, tmp_path):
    store.save("conv1", MESSAGES)
    assert (tmp_path / "conv1.json").exists()


def test_load_returns_saved_messages(store):
    store.save("conv1", MESSAGES)
    loaded = store.load("conv1")
    assert loaded == MESSAGES


def test_load_returns_empty_list_for_nonexistent(store):
    assert store.load("does_not_exist") == []


def test_exists_true_after_save_false_before(store):
    assert not store.exists("conv1")
    store.save("conv1", MESSAGES)
    assert store.exists("conv1")


def test_list_conversations_returns_saved_id(store):
    store.save("conv42", MESSAGES)
    assert "conv42" in store.list_conversations()


def test_delete_returns_true_if_existed(store):
    store.save("conv1", MESSAGES)
    assert store.delete("conv1") is True


def test_delete_returns_false_if_not_existed(store):
    assert store.delete("never_saved") is False


def test_after_delete_load_returns_empty(store):
    store.save("conv1", MESSAGES)
    store.delete("conv1")
    assert store.load("conv1") == []


def test_messages_round_trip_content_preserved(store):
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "metadata": {"tokens": 1}},
    ]
    store.save("roundtrip", messages)
    loaded = store.load("roundtrip")
    assert loaded[0]["content"] == "What is 2+2?"
    assert loaded[1]["metadata"]["tokens"] == 1
