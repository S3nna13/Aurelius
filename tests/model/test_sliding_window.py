from src.model.sliding_window import (
    combine_with_prefix_mask,
    make_sliding_window_mask,
    make_sliding_window_mask_batched,
)


def test_mask_shape():
    mask = make_sliding_window_mask(16, window_size=4)
    assert mask.shape == (16, 16)


def test_mask_causal():
    mask = make_sliding_window_mask(8, window_size=8)
    # Full window = full causal mask
    for i in range(8):
        for j in range(i + 1, 8):
            assert mask[i, j] == float("-inf"), f"Future token ({i},{j}) should be blocked"


def test_mask_window_limit():
    mask = make_sliding_window_mask(10, window_size=3)
    # Token 5 should see tokens 3,4,5 but NOT 0,1,2
    assert mask[5, 5] == 0.0  # self
    assert mask[5, 4] == 0.0  # window
    assert mask[5, 3] == 0.0  # window
    assert mask[5, 2] == float("-inf")  # beyond window


def test_mask_window_1():
    """Window size 1 = only self-attention."""
    mask = make_sliding_window_mask(5, window_size=1)
    for i in range(5):
        assert mask[i, i] == 0.0
        for j in range(5):
            if j != i:
                assert mask[i, j] == float("-inf")


def test_batched_mask_shape():
    mask = make_sliding_window_mask_batched(12, window_size=4, batch_size=3)
    assert mask.shape == (3, 1, 12, 12)


def test_combine_with_prefix_mask():
    from src.training.prefix_lm import make_prefix_lm_mask

    S = 8
    prefix_mask = make_prefix_lm_mask(S, prefix_len=3)
    window_mask = make_sliding_window_mask(S, window_size=2)
    combined = combine_with_prefix_mask(window_mask, prefix_mask)
    assert combined.shape == (S, S)
    # Prefix tokens should be visible from any suffix token (prefix mask allows it)
    # Combined should be at least as permissive as prefix_mask
    allowed_in_prefix = prefix_mask == 0.0
    assert (combined[allowed_in_prefix] == 0.0).all()


def test_first_token_sees_only_itself():
    mask = make_sliding_window_mask(10, window_size=3)
    assert mask[0, 0] == 0.0
    assert (mask[0, 1:] == float("-inf")).all()
