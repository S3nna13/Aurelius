"""Tests for src/model/vocab_pruning.py."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.model.vocab_pruning import (
    VocabPruningResult,
    collect_token_frequencies,
    prune_vocabulary,
    remap_token_ids,
    select_top_k_tokens,
)

VOCAB_SIZE = 256
D_MODEL = 64
N_HEADS = 2
N_KV_HEADS = 2
HEAD_DIM = 32
D_FF = 128
N_LAYERS = 2
MAX_SEQ_LEN = 32


def make_small_config() -> AureliusConfig:
    return AureliusConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        tie_embeddings=True,
    )


@pytest.fixture(scope="module")
def small_model() -> AureliusTransformer:
    config = make_small_config()
    model = AureliusTransformer(config)
    model.eval()
    return model


def test_collect_token_frequencies():
    sequences = [
        [1, 2, 3, 2, 1],
        [2, 4, 4, 4],
        [1],
    ]
    freq = collect_token_frequencies(sequences)
    assert freq[1] == 3
    assert freq[2] == 3
    assert freq[3] == 1
    assert freq[4] == 3
    assert 0 not in freq


def test_select_top_k_tokens():
    frequencies = {10: 100, 20: 50, 30: 10, 40: 5, 50: 1}
    result = select_top_k_tokens(frequencies, k=3)
    assert len(result) == 3
    assert result == sorted(result)
    assert set(result) == {10, 20, 30}


def test_select_top_k_always_keep():
    frequencies = {10: 100, 20: 50, 30: 10, 40: 5, 99: 1}
    result = select_top_k_tokens(frequencies, k=3, always_keep=[99])
    assert len(result) == 3
    assert 99 in result
    assert result == sorted(result)


def test_remap_token_ids_basic():
    id_map = {0: 0, 5: 1, 10: 2, 20: 3}
    tokens = [0, 5, 10, 20, 10, 5]
    remapped = remap_token_ids(tokens, id_map)
    assert remapped == [0, 1, 2, 3, 2, 1]


def test_remap_token_ids_unknown():
    id_map = {1: 0, 2: 1}
    tokens = [1, 99, 2, 999]
    remapped = remap_token_ids(tokens, id_map, unk_id=0)
    assert remapped == [0, 0, 1, 0]


def test_pruning_result_fields():
    id_map = {i: i for i in range(100)}
    result = VocabPruningResult(
        old_vocab_size=200,
        new_vocab_size=100,
        tokens_removed=100,
        embedding_params_saved=100 * 64,
        id_map=id_map,
    )
    assert result.old_vocab_size == 200
    assert result.new_vocab_size == 100
    assert result.tokens_removed == 100
    assert result.compression_ratio == pytest.approx(2.0)
    assert result.tokens_kept == 100
    assert result.embedding_params_saved == 100 * 64


def test_prune_vocabulary_reduces_embed(small_model):
    tokens_to_keep = list(range(100))
    pruned, result = prune_vocabulary(small_model, tokens_to_keep, d_model=D_MODEL)
    assert pruned.embed.num_embeddings == 100
    assert pruned.embed.num_embeddings < VOCAB_SIZE


def test_prune_vocabulary_output_shape(small_model):
    tokens_to_keep = list(range(100))
    pruned, result = prune_vocabulary(small_model, tokens_to_keep, d_model=D_MODEL)
    pruned.eval()

    input_ids = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        _, logits, _ = pruned(input_ids)

    assert logits.shape == (2, 8, 100)


def test_prune_vocabulary_no_inplace(small_model):
    original_vocab = small_model.embed.num_embeddings
    tokens_to_keep = list(range(100))
    _, _ = prune_vocabulary(small_model, tokens_to_keep, d_model=D_MODEL)
    assert small_model.embed.num_embeddings == original_vocab


def test_prune_vocabulary_id_map_valid(small_model):
    tokens_to_keep = list(range(50, 150))
    _, result = prune_vocabulary(small_model, tokens_to_keep, d_model=D_MODEL)
    new_vocab = result.new_vocab_size
    for old_id, new_id in result.id_map.items():
        assert 0 <= new_id < new_vocab, f"id_map[{old_id}]={new_id} is outside [0, {new_vocab})"
