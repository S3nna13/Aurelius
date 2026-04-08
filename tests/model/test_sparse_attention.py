import pytest
import torch
from src.model.sparse_attention import (
    SparseAttentionConfig, make_longformer_mask, make_causal_longformer_mask,
    count_attended_tokens
)

def test_local_window_shape():
    mask = make_longformer_mask(16, window_size=3)
    assert mask.shape == (16, 16)

def test_local_window_symmetric():
    mask = make_longformer_mask(10, window_size=2, causal=False)
    # Center token (5) should see tokens 3,4,5,6,7
    assert (mask[5, 3:8] == 0.0).all()
    assert mask[5, 0] == float("-inf")
    assert mask[5, 9] == float("-inf")

def test_causal_window():
    mask = make_causal_longformer_mask(8, window_size=2)
    # Token 4 sees tokens 2,3,4 (backward window) but NOT 5,6,7
    assert (mask[4, 2:5] == 0.0).all()
    assert (mask[4, 5:] == float("-inf")).all()

def test_global_token_attends_all():
    mask = make_longformer_mask(10, window_size=1, global_token_indices=[0])
    # Token 0 (global) should attend to all tokens
    assert (mask[0, :] == 0.0).all()

def test_all_attend_global():
    mask = make_longformer_mask(10, window_size=1, global_token_indices=[0])
    # All tokens should attend to token 0 (global)
    assert (mask[:, 0] == 0.0).all()

def test_multiple_global_tokens():
    mask = make_longformer_mask(12, window_size=2, global_token_indices=[0, 5])
    assert (mask[0, :] == 0.0).all()   # global 0 attends all
    assert (mask[5, :] == 0.0).all()   # global 5 attends all
    assert (mask[:, 0] == 0.0).all()   # all attend global 0
    assert (mask[:, 5] == 0.0).all()   # all attend global 5

def test_count_attended_tokens():
    mask = make_longformer_mask(8, window_size=2, causal=False)
    counts = count_attended_tokens(mask)
    assert counts.shape == (8,)
    # Middle token (4) attends to 5 tokens (window=2 each side + self)
    assert counts[4].item() == 5

def test_causal_window_no_future():
    mask = make_causal_longformer_mask(10, window_size=3)
    # No token should attend to future positions
    for i in range(10):
        for j in range(i+1, 10):
            assert mask[i, j] == float("-inf"), f"Future token ({i},{j}) should be blocked"
