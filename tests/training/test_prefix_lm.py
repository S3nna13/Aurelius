import torch

from src.training.prefix_lm import PrefixLMConfig, make_prefix_lm_mask, prepare_prefix_lm_batch


def test_mask_shape():
    mask = make_prefix_lm_mask(10, 4)
    assert mask.shape == (10, 10)


def test_prefix_is_bidirectional():
    mask = make_prefix_lm_mask(10, 4)
    # All prefix-to-prefix positions should be 0.0
    assert (mask[:4, :4] == 0.0).all()


def test_suffix_is_causal():
    mask = make_prefix_lm_mask(10, 4)
    # Suffix row i (>=4) should see cols 0..i only
    for i in range(4, 10):
        assert (mask[i, : i + 1] == 0.0).all()
        if i + 1 < 10:
            assert (mask[i, i + 1 :] == float("-inf")).all()


def test_no_future_prefix_from_suffix():
    mask = make_prefix_lm_mask(8, 3)
    # Suffix tokens should NOT be able to see other suffix tokens in their future
    for i in range(3, 8):
        for j in range(i + 1, 8):
            assert mask[i, j] == float("-inf")


def test_prepare_batch_shapes():
    input_ids = torch.randint(0, 100, (2, 16))
    result = prepare_prefix_lm_batch(input_ids, prefix_len=4)
    assert result["input_ids"].shape == (2, 16)
    assert result["labels"].shape == (2, 16)
    assert result["attention_mask"].shape == (16, 16)


def test_prefix_labels_masked():
    input_ids = torch.randint(0, 100, (2, 16))
    result = prepare_prefix_lm_batch(
        input_ids, cfg=PrefixLMConfig(mask_prefix_labels=True), prefix_len=6
    )
    # First 6 label positions should be -100
    assert (result["labels"][:, :6] == -100).all()


def test_suffix_labels_not_masked():
    input_ids = torch.arange(16).unsqueeze(0)  # (1, 16)
    result = prepare_prefix_lm_batch(
        input_ids, cfg=PrefixLMConfig(mask_prefix_labels=True), prefix_len=4
    )
    # Suffix labels (positions 4..14) should equal input_ids[:, 5..15]
    assert (result["labels"][0, 4:15] == input_ids[0, 5:16]).all()


def test_random_prefix_in_range():
    input_ids = torch.randint(0, 100, (1, 20))
    cfg = PrefixLMConfig(min_prefix_fraction=0.2, max_prefix_fraction=0.8)
    for _ in range(10):
        result = prepare_prefix_lm_batch(input_ids, cfg)
        # Check that some prefix was masked
        n_masked = (result["labels"][0] == -100).sum().item()
        assert 4 <= n_masked  # at least 20% of 20 = 4 masked (prefix + last)
