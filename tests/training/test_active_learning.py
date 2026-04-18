import math
import pytest
import torch

from src.training.active_learning import ALConfig, ActiveLearner

N = 20
C = 10


@pytest.fixture
def config():
    return ALConfig(strategy="entropy", n_select=5)


@pytest.fixture
def learner(config):
    return ActiveLearner(config)


@pytest.fixture
def uniform_probs():
    return torch.full((N, C), 1.0 / C)


@pytest.fixture
def random_probs():
    torch.manual_seed(0)
    raw = torch.rand(N, C)
    return raw / raw.sum(dim=-1, keepdim=True)


# Test 1: ALConfig instantiates
def test_alconfig_instantiates():
    cfg = ALConfig(strategy="entropy", n_select=5)
    assert cfg.strategy == "entropy"
    assert cfg.n_select == 5
    assert cfg.seed == 42


# Test 2: ActiveLearner instantiates
def test_active_learner_instantiates(config):
    learner = ActiveLearner(config)
    assert learner.config is config


# Test 3: least_confidence returns shape (N,)
def test_least_confidence_shape(learner, random_probs):
    scores = learner.least_confidence(random_probs)
    assert scores.shape == (N,)


# Test 4: least_confidence values in [0, 1]
def test_least_confidence_range(learner, random_probs):
    scores = learner.least_confidence(random_probs)
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()


# Test 5: uniform distribution -> least_confidence ≈ 1 - 1/C
def test_least_confidence_uniform(learner, uniform_probs):
    scores = learner.least_confidence(uniform_probs)
    expected = 1.0 - 1.0 / C
    assert torch.allclose(scores, torch.full((N,), expected), atol=1e-5)


# Test 6: margin_score returns shape (N,)
def test_margin_score_shape(learner, random_probs):
    scores = learner.margin_score(random_probs)
    assert scores.shape == (N,)


# Test 7: one-hot distribution -> margin_score ≈ 1.0
def test_margin_score_one_hot(learner):
    one_hot = torch.zeros(N, C)
    one_hot[:, 0] = 1.0
    scores = learner.margin_score(one_hot)
    assert torch.allclose(scores, torch.ones(N), atol=1e-5)


# Test 8: entropy_score returns shape (N,)
def test_entropy_score_shape(learner, random_probs):
    scores = learner.entropy_score(random_probs)
    assert scores.shape == (N,)


# Test 9: uniform -> max entropy ≈ log(C)
def test_entropy_score_uniform(learner, uniform_probs):
    scores = learner.entropy_score(uniform_probs)
    expected = math.log(C)
    assert torch.allclose(scores, torch.full((N,), expected), atol=1e-4)


# Test 10: select returns LongTensor of length n_select
def test_select_returns_long_tensor(learner, random_probs):
    indices = learner.select(random_probs)
    assert indices.dtype == torch.long
    assert indices.shape == (learner.config.n_select,)


# Test 11: selected indices are unique
def test_select_unique_indices(learner, random_probs):
    indices = learner.select(random_probs)
    assert len(indices) == len(indices.unique())


# Test 12: update_pool removes selected ids
def test_update_pool_removes_selected():
    config = ALConfig(strategy="entropy", n_select=5)
    learner = ActiveLearner(config)
    pool = list(range(20))
    selected = [0, 3, 7, 12, 19]
    remaining = learner.update_pool(pool, selected)
    assert set(remaining) == set(pool) - set(selected)
    assert len(remaining) == 15
