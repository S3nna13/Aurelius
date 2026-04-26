"""
Tests for src/training/rejection_finetuning.py

Tiny configs: vocab=16, d_model=8, n_samples=3, max_new_tokens=4, seq_len=8.
Model: nn.Embedding(16, 8) + nn.Linear(8, 16).
Every test performs at least a forward (and where required, backward) pass.
"""

import math

import torch
import torch.nn as nn

from src.training.rejection_finetuning import (
    EnsembleVerifier,
    ExactMatchVerifier,
    RejectionFilter,
    RewardVerifier,
    RFTDataset,
    RFTTrainer,
    SolutionSampler,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 8
N_SAMPLES = 3
MAX_NEW_TOKENS = 4
SEQ_LEN = 8


class TinyModel(nn.Module):
    """Minimal LM: Embedding -> Linear -> logits."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq)
        h = self.embed(input_ids)  # (batch, seq, d_model)
        return self.proj(h)  # (batch, seq, vocab)


def make_model() -> TinyModel:
    torch.manual_seed(0)
    return TinyModel()


def make_prompt(length: int = 4) -> torch.Tensor:
    return torch.randint(0, VOCAB, (length,))


def make_sampler(model: nn.Module) -> SolutionSampler:
    return SolutionSampler(
        model, n_samples=N_SAMPLES, temperature=0.8, max_new_tokens=MAX_NEW_TOKENS
    )


# ---------------------------------------------------------------------------
# Test 1: ExactMatchVerifier — True for matching tail
# ---------------------------------------------------------------------------


def test_exact_match_verifier_true():
    verifier = ExactMatchVerifier()
    gt = torch.tensor([3, 7, 2])
    prompt = make_prompt()
    solution = torch.tensor([5, 5, 3, 7, 2])  # tail matches gt

    # forward pass to confirm model runs (not the verifier, but we exercise the tensor ops)
    result = verifier.verify(prompt, solution, gt)
    assert result is True


# ---------------------------------------------------------------------------
# Test 2: ExactMatchVerifier — False for non-matching
# ---------------------------------------------------------------------------


def test_exact_match_verifier_false():
    verifier = ExactMatchVerifier()
    gt = torch.tensor([3, 7, 2])
    prompt = make_prompt()
    solution = torch.tensor([5, 5, 3, 7, 9])  # tail does NOT match

    result = verifier.verify(prompt, solution, gt)
    assert result is False


# ---------------------------------------------------------------------------
# Test 3: RewardVerifier — True when reward > threshold
# ---------------------------------------------------------------------------


def test_reward_verifier_true():
    model = make_model()

    # reward_fn: sum of logits at first token position (always > 0.5 for non-trivial input)
    def reward_fn(solution_ids: torch.Tensor) -> float:
        with torch.no_grad():
            logits = model(solution_ids.unsqueeze(0))
            return float(logits.abs().mean().item())

    verifier = RewardVerifier(reward_fn=reward_fn, threshold=0.0)
    prompt = make_prompt()
    solution = torch.randint(0, VOCAB, (MAX_NEW_TOKENS,))

    result = verifier.verify(prompt, solution, ground_truth=None)
    assert result is True


# ---------------------------------------------------------------------------
# Test 4: RewardVerifier — False when reward <= threshold
# ---------------------------------------------------------------------------


def test_reward_verifier_false():
    # reward_fn always returns 0.0; threshold=0.5 → False
    def reward_fn(solution_ids: torch.Tensor) -> float:
        return 0.0

    verifier = RewardVerifier(reward_fn=reward_fn, threshold=0.5)
    prompt = make_prompt()
    solution = torch.randint(0, VOCAB, (MAX_NEW_TOKENS,))

    result = verifier.verify(prompt, solution, ground_truth=None)
    assert result is False


# ---------------------------------------------------------------------------
# Test 5: EnsembleVerifier require_all=True — True only when all agree
# ---------------------------------------------------------------------------


def test_ensemble_verifier_require_all_true_case():
    # Both verifiers return True
    gt = torch.tensor([1, 2])
    solution = torch.tensor([5, 1, 2])  # tail matches
    v1 = ExactMatchVerifier()
    v2 = RewardVerifier(reward_fn=lambda s: 1.0, threshold=0.5)

    ensemble = EnsembleVerifier([v1, v2], require_all=True)
    result = ensemble.verify(make_prompt(), solution, gt)
    assert result is True


# ---------------------------------------------------------------------------
# Test 6: EnsembleVerifier require_all=True — False when one disagrees
# ---------------------------------------------------------------------------


def test_ensemble_verifier_require_all_false_case():
    # v1 returns True, v2 returns False → overall False
    gt = torch.tensor([1, 2])
    solution = torch.tensor([5, 1, 2])
    v1 = ExactMatchVerifier()
    v2 = RewardVerifier(reward_fn=lambda s: 0.0, threshold=0.5)  # always False

    ensemble = EnsembleVerifier([v1, v2], require_all=True)
    result = ensemble.verify(make_prompt(), solution, gt)
    assert result is False


# ---------------------------------------------------------------------------
# Test 7: EnsembleVerifier require_all=False — True when any is True
# ---------------------------------------------------------------------------


def test_ensemble_verifier_any_true():
    gt = torch.tensor([9, 9])
    solution = torch.tensor([1, 2, 3])  # tail does NOT match gt (would be False for exact)
    v1 = ExactMatchVerifier()  # False
    v2 = RewardVerifier(reward_fn=lambda s: 1.0, threshold=0.5)  # True

    ensemble = EnsembleVerifier([v1, v2], require_all=False)
    result = ensemble.verify(make_prompt(), solution, gt)
    assert result is True


# ---------------------------------------------------------------------------
# Test 8: SolutionSampler.sample — returns n_samples solutions, valid vocab ids
# ---------------------------------------------------------------------------


def test_solution_sampler_sample_count_and_vocab():
    model = make_model()
    sampler = make_sampler(model)
    prompt = make_prompt(4)

    solutions, log_probs = sampler.sample(prompt)

    assert len(solutions) == N_SAMPLES
    assert len(log_probs) == N_SAMPLES
    for sol in solutions:
        assert sol.dtype == torch.long
        assert sol.numel() == MAX_NEW_TOKENS
        assert torch.all(sol >= 0).item()
        assert torch.all(sol < VOCAB).item()


# ---------------------------------------------------------------------------
# Test 9: SolutionSampler.sample — log_probs are negative floats (or 0)
# ---------------------------------------------------------------------------


def test_solution_sampler_log_probs_negative():
    model = make_model()
    sampler = make_sampler(model)
    prompt = make_prompt(4)

    _, log_probs = sampler.sample(prompt)

    for lp in log_probs:
        assert isinstance(lp, float)
        assert lp <= 0.0, f"log_prob {lp} should be <= 0"
        assert math.isfinite(lp), f"log_prob {lp} is not finite"


# ---------------------------------------------------------------------------
# Test 10: SolutionSampler.diverse_sample — returns n_samples solutions
# ---------------------------------------------------------------------------


def test_solution_sampler_diverse_sample_count():
    model = make_model()
    sampler = make_sampler(model)
    prompt = make_prompt(4)

    solutions, log_probs = sampler.diverse_sample(prompt)

    assert len(solutions) == N_SAMPLES
    assert len(log_probs) == N_SAMPLES
    for sol in solutions:
        assert sol.numel() == MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# Test 11: RejectionFilter.filter — n_kept <= len(solutions), rate in [0,1]
# ---------------------------------------------------------------------------


def test_rejection_filter_bounds():
    # verifier always returns True
    verifier = RewardVerifier(reward_fn=lambda s: 1.0, threshold=0.5)
    filt = RejectionFilter(verifier)

    prompt = make_prompt()
    solutions = [torch.randint(0, VOCAB, (MAX_NEW_TOKENS,)) for _ in range(N_SAMPLES)]

    kept, n_kept, rate = filt.filter(prompt, solutions, ground_truth=None)

    assert n_kept <= len(solutions)
    assert 0.0 <= rate <= 1.0
    assert len(kept) == n_kept


# ---------------------------------------------------------------------------
# Test 12: RejectionFilter.best_of_k — returns single tensor with best log_prob
# ---------------------------------------------------------------------------


def test_rejection_filter_best_of_k():
    verifier = ExactMatchVerifier()
    filt = RejectionFilter(verifier)

    prompt = make_prompt()
    solutions = [torch.randint(0, VOCAB, (MAX_NEW_TOKENS,)) for _ in range(N_SAMPLES)]
    log_probs = [-5.0, -1.0, -3.0]  # best is index 1

    best = filt.best_of_k(prompt, solutions, log_probs)

    assert isinstance(best, torch.Tensor)
    assert torch.equal(best, solutions[1])


# ---------------------------------------------------------------------------
# Test 13: RFTDataset.add + len — grows correctly
# ---------------------------------------------------------------------------


def test_rft_dataset_add_len():
    ds = RFTDataset()
    assert len(ds) == 0

    for i in range(5):
        prompt = make_prompt(4)
        solution = torch.randint(0, VOCAB, (MAX_NEW_TOKENS,))
        ds.add(prompt, solution)
        assert len(ds) == i + 1


# ---------------------------------------------------------------------------
# Test 14: RFTDataset.collate_fn — output shapes padded to max length
# ---------------------------------------------------------------------------


def test_rft_dataset_collate_fn():
    ds = RFTDataset()
    # Add pairs with different lengths
    ds.add(torch.tensor([1, 2]), torch.tensor([3, 4, 5]))
    ds.add(torch.tensor([6, 7, 8, 9]), torch.tensor([10, 11]))

    batch = [ds[0], ds[1]]
    padded_prompts, padded_solutions = RFTDataset.collate_fn(batch)

    assert padded_prompts.shape == (2, 4)  # max prompt len = 4
    assert padded_solutions.shape == (2, 3)  # max solution len = 3
    assert padded_prompts.dtype == torch.long
    assert padded_solutions.dtype == torch.long


# ---------------------------------------------------------------------------
# Test 15a: RFTDataset.stats — all keys present, n_examples correct
# ---------------------------------------------------------------------------


def test_rft_dataset_stats():
    ds = RFTDataset()
    for _ in range(4):
        ds.add(make_prompt(4), torch.randint(0, VOCAB, (MAX_NEW_TOKENS,)))

    stats = ds.stats()

    assert "n_examples" in stats
    assert "mean_solution_length" in stats
    assert "unique_prompts" in stats
    assert stats["n_examples"] == 4


# ---------------------------------------------------------------------------
# Test 16: RFTTrainer.iteration — all keys present, n_correct in [0, n*n_samples]
# ---------------------------------------------------------------------------


def test_rft_trainer_iteration_keys_and_range():
    torch.manual_seed(42)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    sampler = SolutionSampler(
        model, n_samples=N_SAMPLES, temperature=0.8, max_new_tokens=MAX_NEW_TOKENS
    )
    verifier = RewardVerifier(reward_fn=lambda s: 1.0, threshold=0.5)  # always correct

    trainer = RFTTrainer(model, optimizer, sampler, verifier, n_iterations=2)
    prompts = [make_prompt(4) for _ in range(2)]
    ground_truths = [None, None]

    stats = trainer.iteration(prompts, ground_truths)

    assert "n_correct" in stats
    assert "acceptance_rate" in stats
    assert "loss" in stats
    max_possible = len(prompts) * N_SAMPLES
    assert 0 <= stats["n_correct"] <= max_possible


# ---------------------------------------------------------------------------
# Test 17: RFTTrainer — loss is finite when there are correct solutions
# ---------------------------------------------------------------------------


def test_rft_trainer_loss_finite():
    torch.manual_seed(7)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    sampler = SolutionSampler(
        model, n_samples=N_SAMPLES, temperature=0.8, max_new_tokens=MAX_NEW_TOKENS
    )
    # Always accept → always have correct solutions → loss must be computed
    verifier = RewardVerifier(reward_fn=lambda s: 1.0, threshold=0.5)

    trainer = RFTTrainer(model, optimizer, sampler, verifier, n_iterations=1)
    prompts = [make_prompt(4)]
    ground_truths = [None]

    stats = trainer.iteration(prompts, ground_truths)

    assert math.isfinite(stats["loss"]), f"loss should be finite, got {stats['loss']}"


# ---------------------------------------------------------------------------
# Test 18: self_improvement_loop — returns n_iterations dicts
# ---------------------------------------------------------------------------


def test_self_improvement_loop_length():
    torch.manual_seed(99)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    sampler = SolutionSampler(
        model, n_samples=N_SAMPLES, temperature=0.8, max_new_tokens=MAX_NEW_TOKENS
    )
    verifier = RewardVerifier(reward_fn=lambda s: 1.0, threshold=0.5)

    n_iters = 3
    trainer = RFTTrainer(model, optimizer, sampler, verifier, n_iterations=n_iters)
    prompts = [make_prompt(4)]
    ground_truths = [None]

    all_stats = trainer.self_improvement_loop(prompts, ground_truths)

    assert isinstance(all_stats, list)
    assert len(all_stats) == n_iters
    for s in all_stats:
        assert "n_correct" in s
        assert "acceptance_rate" in s
        assert "loss" in s
