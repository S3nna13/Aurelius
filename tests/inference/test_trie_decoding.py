"""Tests for src/inference/trie_decoding.py — trie/prefix-tree constrained decoding."""

from __future__ import annotations

import pytest
import torch

from src.inference.trie_decoding import (
    TrieDecodingConfig,
    TrieNode,
    Trie,
    constrained_logits,
    TrieDecoder,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
EOS = 0


# ---------------------------------------------------------------------------
# Tiny fake model
# ---------------------------------------------------------------------------

class TinyModel:
    """Returns uniform logits; greedy picks token 1 by slight bias."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, preferred_token: int = 1) -> None:
        self.vocab_size = vocab_size
        self.preferred_token = preferred_token

    def __call__(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[:, :, self.preferred_token] = 2.0  # strong bias
        loss = torch.tensor(0.0)
        pkv = None
        return loss, logits, pkv


# ---------------------------------------------------------------------------
# Test 1: TrieNode starts with no children
# ---------------------------------------------------------------------------

def test_trie_node_starts_empty():
    node = TrieNode()
    assert node.children == {}
    assert node.is_terminal is False


# ---------------------------------------------------------------------------
# Test 2: TrieNode.insert adds children
# ---------------------------------------------------------------------------

def test_trie_node_insert_adds_children():
    node = TrieNode()
    node.insert([10, 20, 30])
    assert node.has_child(10)
    child = node.get_child(10)
    assert child is not None
    assert child.has_child(20)


# ---------------------------------------------------------------------------
# Test 3: TrieNode.valid_next_tokens returns correct ids
# ---------------------------------------------------------------------------

def test_trie_node_valid_next_tokens():
    node = TrieNode()
    node.insert([5])
    node.insert([7])
    valid = node.valid_next_tokens()
    assert set(valid) == {5, 7}


# ---------------------------------------------------------------------------
# Test 4: TrieNode.is_terminal set correctly at end of sequence
# ---------------------------------------------------------------------------

def test_trie_node_is_terminal_at_end():
    node = TrieNode()
    node.insert([3, 4])
    # Root should not be terminal
    assert not node.is_terminal
    # Child of 3 should not be terminal
    child_3 = node.get_child(3)
    assert child_3 is not None
    assert not child_3.is_terminal
    # Child of 3 -> 4 should be terminal
    child_4 = child_3.get_child(4)
    assert child_4 is not None
    assert child_4.is_terminal


# ---------------------------------------------------------------------------
# Test 5: Trie.insert stores sequence
# ---------------------------------------------------------------------------

def test_trie_insert_stores_sequence():
    trie = Trie()
    trie.insert([10, 20])
    assert trie.root.has_child(10)
    child = trie.root.get_child(10)
    assert child is not None
    assert child.has_child(20)


# ---------------------------------------------------------------------------
# Test 6: Trie.__contains__ True for inserted, False for others
# ---------------------------------------------------------------------------

def test_trie_contains_true_for_inserted():
    trie = Trie()
    trie.insert([1, 2, 3])
    assert [1, 2, 3] in trie


def test_trie_contains_false_for_prefix_only():
    trie = Trie()
    trie.insert([1, 2, 3])
    # Prefix [1, 2] is NOT a complete sequence
    assert [1, 2] not in trie


def test_trie_contains_false_for_absent():
    trie = Trie()
    trie.insert([1, 2, 3])
    assert [4, 5] not in trie


# ---------------------------------------------------------------------------
# Test 7: Trie.prefix_allowed_tokens returns valid next tokens
# ---------------------------------------------------------------------------

def test_trie_prefix_allowed_tokens_from_root():
    trie = Trie()
    trie.insert([5, 10])
    trie.insert([5, 20])
    # From root, allowed after [] is [5]
    allowed = trie.prefix_allowed_tokens([])
    assert set(allowed) == {5}


def test_trie_prefix_allowed_tokens_after_first_token():
    trie = Trie()
    trie.insert([5, 10])
    trie.insert([5, 20])
    allowed = trie.prefix_allowed_tokens([5])
    assert set(allowed) == {10, 20}


# ---------------------------------------------------------------------------
# Test 8: Trie.prefix_allowed_tokens returns [] for invalid prefix
# ---------------------------------------------------------------------------

def test_trie_prefix_allowed_tokens_invalid_prefix():
    trie = Trie()
    trie.insert([5, 10])
    # Token 99 is not in the trie at all
    allowed = trie.prefix_allowed_tokens([99])
    assert allowed == []


# ---------------------------------------------------------------------------
# Test 9: Trie.insert_batch inserts multiple sequences
# ---------------------------------------------------------------------------

def test_trie_insert_batch():
    trie = Trie()
    trie.insert_batch([[1, 2], [3, 4], [5, 6]])
    assert [1, 2] in trie
    assert [3, 4] in trie
    assert [5, 6] in trie
    assert [7, 8] not in trie


# ---------------------------------------------------------------------------
# Test 10: Trie.__len__ counts sequences
# ---------------------------------------------------------------------------

def test_trie_len():
    trie = Trie()
    assert len(trie) == 0
    trie.insert([1, 2])
    assert len(trie) == 1
    trie.insert([3, 4])
    assert len(trie) == 2
    trie.insert_batch([[5, 6], [7, 8]])
    assert len(trie) == 4


# ---------------------------------------------------------------------------
# Test 11: constrained_logits masks non-allowed tokens to -inf
# ---------------------------------------------------------------------------

def test_constrained_logits_non_allowed_are_neginf():
    logits = torch.zeros(VOCAB_SIZE)
    allowed = [3, 7]
    out = constrained_logits(logits, allowed)
    for i in range(VOCAB_SIZE):
        if i not in allowed:
            assert out[i].item() == float("-inf"), f"Token {i} should be -inf"


# ---------------------------------------------------------------------------
# Test 12: constrained_logits allowed tokens have finite values
# ---------------------------------------------------------------------------

def test_constrained_logits_allowed_are_finite():
    logits = torch.ones(VOCAB_SIZE)
    allowed = [3, 7]
    out = constrained_logits(logits, allowed)
    for tok in allowed:
        assert torch.isfinite(out[tok]), f"Token {tok} should be finite"


def test_constrained_logits_empty_allowed_returns_unconstrained():
    """Empty allowed list falls back to unconstrained (no masking)."""
    logits = torch.randn(VOCAB_SIZE)
    out = constrained_logits(logits, [])
    # With temperature=1.0, should equal logits unchanged
    assert torch.allclose(out, logits)


def test_constrained_logits_temperature_scaling():
    logits = torch.ones(VOCAB_SIZE)
    allowed = [5]
    out = constrained_logits(logits, allowed, temperature=2.0)
    # logits / 2.0 = 0.5 for the allowed token
    assert out[5].item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 13: TrieDecoder instantiates
# ---------------------------------------------------------------------------

def test_trie_decoder_instantiates():
    model = TinyModel()
    trie = Trie()
    trie.insert([1, 2, 3])
    cfg = TrieDecodingConfig(eos_token_id=EOS, max_new_tokens=4)
    decoder = TrieDecoder(model, trie, cfg)
    assert decoder.model is model
    assert decoder.trie is trie
    assert decoder.config is cfg


# ---------------------------------------------------------------------------
# Test 14: TrieDecoder.generate returns (Tensor, dict) with correct keys
# ---------------------------------------------------------------------------

def test_trie_decoder_generate_returns_tensor_and_dict():
    model = TinyModel(preferred_token=1)
    trie = Trie()
    trie.insert([1, 2])
    cfg = TrieDecodingConfig(eos_token_id=EOS, max_new_tokens=8)
    decoder = TrieDecoder(model, trie, cfg)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    result = decoder.generate(input_ids)
    assert isinstance(result, tuple)
    assert len(result) == 2
    gen_ids, stats = result
    assert isinstance(gen_ids, torch.Tensor)
    assert isinstance(stats, dict)
    for key in ("n_tokens", "constrained", "trie_terminal_reached"):
        assert key in stats, f"Missing stats key: {key}"


# ---------------------------------------------------------------------------
# Test 15: TrieDecoder.generate output tokens are subset of trie paths
# ---------------------------------------------------------------------------

def test_trie_decoder_generate_tokens_follow_trie():
    """Model prefers token 1; trie only allows [2, 3]. Decoder must follow trie."""
    model = TinyModel(preferred_token=1)
    trie = Trie()
    trie.insert([2, 3])
    cfg = TrieDecodingConfig(eos_token_id=EOS, max_new_tokens=8, allow_eos_anywhere=False)
    decoder = TrieDecoder(model, trie, cfg)
    input_ids = torch.zeros(1, 2, dtype=torch.long)
    gen_ids, stats = decoder.generate(input_ids)
    # The first generated token must be 2 (the only valid start in trie)
    generated_list = gen_ids.tolist()
    assert generated_list[0] == 2, f"First token should be 2, got {generated_list[0]}"
    # Second must be 3
    assert generated_list[1] == 3, f"Second token should be 3, got {generated_list[1]}"
    # After [2, 3], trie is terminal with no children — stops
    assert stats["trie_terminal_reached"] is True


# ---------------------------------------------------------------------------
# Test 16: TrieDecoder.batch_generate returns list of n_sequences tensors
# ---------------------------------------------------------------------------

def test_trie_decoder_batch_generate_returns_n_sequences():
    model = TinyModel(preferred_token=1)
    trie = Trie()
    trie.insert([1, 2])
    trie.insert([1, 3])
    cfg = TrieDecodingConfig(eos_token_id=EOS, max_new_tokens=6)
    decoder = TrieDecoder(model, trie, cfg)
    input_ids = torch.zeros(1, 2, dtype=torch.long)
    n = 4
    results = decoder.batch_generate(input_ids, n_sequences=n)
    assert isinstance(results, list)
    assert len(results) == n
    for r in results:
        assert isinstance(r, torch.Tensor)
        assert r.shape == (cfg.max_new_tokens,)


# ---------------------------------------------------------------------------
# Extra: generated_ids has correct length (max_new_tokens)
# ---------------------------------------------------------------------------

def test_trie_decoder_generate_output_length():
    model = TinyModel(preferred_token=2)
    trie = Trie()
    trie.insert([2, 3, 4, 5, 6, 7, 8, 9])
    cfg = TrieDecodingConfig(eos_token_id=EOS, max_new_tokens=10)
    decoder = TrieDecoder(model, trie, cfg)
    input_ids = torch.zeros(1, 2, dtype=torch.long)
    gen_ids, _ = decoder.generate(input_ids)
    assert gen_ids.shape == (cfg.max_new_tokens,)
