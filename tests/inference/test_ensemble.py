"""Tests for model ensemble inference."""

import math

import pytest
import torch

from src.inference.ensemble import EnsembleConfig, EnsembleMode, ModelEnsemble
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


def _make_model(seed: int = 0) -> AureliusTransformer:
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=64,
        max_seq_len=64,
    )
    torch.manual_seed(seed)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture
def ensemble():
    models = [_make_model(i) for i in range(3)]
    return ModelEnsemble(models, EnsembleConfig(mode=EnsembleMode.PROB_MEAN))


def test_ensemble_forward_shape(ensemble):
    input_ids = torch.randint(0, 64, (2, 8))
    _, logits, _ = ensemble(input_ids)
    assert logits.shape == (2, 8, 64)


def test_ensemble_returns_none_without_labels(ensemble):
    input_ids = torch.randint(0, 64, (1, 6))
    loss, logits, _ = ensemble(input_ids)
    assert loss is None
    assert logits is not None


def test_ensemble_returns_loss_with_labels(ensemble):
    input_ids = torch.randint(0, 64, (1, 6))
    loss, _, _ = ensemble(input_ids, labels=input_ids)
    assert loss is not None
    assert math.isfinite(loss.item())


def test_ensemble_weights_normalized():
    models = [_make_model(i) for i in range(2)]
    cfg = EnsembleConfig(weights=[3.0, 1.0])
    ens = ModelEnsemble(models, cfg)
    assert abs(sum(ens._weights) - 1.0) < 1e-5
    assert abs(ens._weights[0] - 0.75) < 0.01


def test_ensemble_logit_mean_mode():
    models = [_make_model(i) for i in range(2)]
    cfg = EnsembleConfig(mode=EnsembleMode.LOGIT_MEAN)
    ens = ModelEnsemble(models, cfg)
    input_ids = torch.randint(0, 64, (1, 4))
    _, logits, _ = ens(input_ids)
    assert logits.shape == (1, 4, 64)


def test_ensemble_log_prob_mode():
    models = [_make_model(i) for i in range(2)]
    cfg = EnsembleConfig(mode=EnsembleMode.LOG_PROB_MEAN)
    ens = ModelEnsemble(models, cfg)
    input_ids = torch.randint(0, 64, (1, 4))
    _, logits, _ = ens(input_ids)
    assert logits.shape == (1, 4, 64)


def test_ensemble_generate_shape(ensemble):
    input_ids = torch.randint(0, 64, (1, 4))
    output = ensemble.generate(input_ids, max_new_tokens=5)
    assert output.shape == (1, 9)  # 4 + 5


def test_single_model_ensemble():
    """Ensemble of 1 model should be equivalent to single model."""
    model = _make_model()
    ens = ModelEnsemble([model], EnsembleConfig(mode=EnsembleMode.LOGIT_MEAN))
    input_ids = torch.randint(0, 64, (1, 6))
    _, ens_logits, _ = ens(input_ids)
    _, solo_logits, _ = model(input_ids)
    assert torch.allclose(ens_logits, solo_logits, atol=1e-5)
