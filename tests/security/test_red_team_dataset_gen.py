"""Tests for the red-team dataset generator."""

from __future__ import annotations

import torch

from src.security.red_team_dataset_gen import (
    RedTeamConfig,
    RedTeamGenerator,
)

CFG = RedTeamConfig(n_benign=4, n_adversarial=4, max_len=16, vocab_size=64, seed=0)


def make_gen() -> RedTeamGenerator:
    return RedTeamGenerator(CFG)


# 1. Generator instantiates without error
def test_generator_instantiates():
    gen = make_gen()
    assert isinstance(gen, RedTeamGenerator)


# 2. generate_benign returns list of correct length
def test_generate_benign_length():
    gen = make_gen()
    samples = gen.generate_benign(CFG.n_benign)
    assert isinstance(samples, list)
    assert len(samples) == CFG.n_benign


# 3. All benign samples have label='benign'
def test_generate_benign_labels():
    gen = make_gen()
    samples = gen.generate_benign(CFG.n_benign)
    for s in samples:
        assert s.label == "benign"
        assert s.perturbation_type == "none"


# 4. generate_prefix_injection returns list of correct length
def test_prefix_injection_length():
    gen = make_gen()
    samples = gen.generate_prefix_injection(CFG.n_adversarial)
    assert isinstance(samples, list)
    assert len(samples) == CFG.n_adversarial


# 5. Prefix samples have correct label and perturbation_type
def test_prefix_injection_label_and_type():
    gen = make_gen()
    samples = gen.generate_prefix_injection(CFG.n_adversarial)
    for s in samples:
        assert s.label == "adversarial"
        assert s.perturbation_type == "prefix_injection"


# 6. First 4 tokens of prefix samples are [1, 2, 3, 4]
def test_prefix_injection_pattern():
    gen = make_gen()
    samples = gen.generate_prefix_injection(CFG.n_adversarial)
    expected = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    for s in samples:
        ids = s.prompt_ids.squeeze(0)
        assert torch.equal(ids[:4], expected), f"Prefix mismatch: {ids[:4]}"


# 7. Last 4 tokens of suffix samples are [5, 6, 7, 8]
def test_suffix_injection_pattern():
    gen = make_gen()
    samples = gen.generate_suffix_injection(CFG.n_adversarial)
    expected = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    for s in samples:
        ids = s.prompt_ids.squeeze(0)
        assert torch.equal(ids[-4:], expected), f"Suffix mismatch: {ids[-4:]}"


# 8. generate_random_token_attack returns list with label='adversarial'
def test_random_token_attack_labels():
    gen = make_gen()
    samples = gen.generate_random_token_attack(CFG.n_adversarial)
    assert isinstance(samples, list)
    assert len(samples) == CFG.n_adversarial
    for s in samples:
        assert s.label == "adversarial"
        assert s.perturbation_type == "token_noise"


# 9. generate_dataset returns a list
def test_generate_dataset_returns_list():
    gen = make_gen()
    dataset = gen.generate_dataset()
    assert isinstance(dataset, list)
    assert len(dataset) > 0


# 10. Dataset contains both 'benign' and 'adversarial' labels
def test_generate_dataset_contains_both_labels():
    gen = make_gen()
    dataset = gen.generate_dataset()
    labels = {s.label for s in dataset}
    assert "benign" in labels
    assert "adversarial" in labels


# 11. to_tensor_batch returns a 2-D Tensor of shape (N, max_len)
def test_to_tensor_batch_shape():
    gen = make_gen()
    samples = gen.generate_benign(CFG.n_benign)
    batch = gen.to_tensor_batch(samples)
    assert isinstance(batch, torch.Tensor)
    assert batch.ndim == 2
    assert batch.shape == (CFG.n_benign, CFG.max_len)


# 12. All tokens in batch are in [0, vocab_size)
def test_to_tensor_batch_token_range():
    gen = make_gen()
    dataset = gen.generate_dataset()
    batch = gen.to_tensor_batch(dataset)
    assert batch.min().item() >= 0
    assert batch.max().item() < CFG.vocab_size


# 13. Same seed → same benign sequence (deterministic)
def test_deterministic_benign_sequence():
    gen1 = RedTeamGenerator(CFG)
    gen2 = RedTeamGenerator(CFG)
    s1 = gen1.generate_benign(1)[0]
    s2 = gen2.generate_benign(1)[0]
    assert torch.equal(s1.prompt_ids, s2.prompt_ids)
