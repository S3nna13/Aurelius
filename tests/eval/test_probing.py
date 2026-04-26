import pytest
import torch

from src.eval.probing import LinearProbe, ProbeConfig, extract_layer_hiddens, probe_all_layers
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    return AureliusTransformer(cfg)


def test_linear_probe_fit_and_evaluate():
    cfg = ProbeConfig(n_classes=3, d_model=16, n_epochs=5)
    probe = LinearProbe(cfg)
    torch.manual_seed(42)
    X = torch.randn(50, 16)
    y = torch.randint(0, 3, (50,))
    losses = probe.fit(X, y)
    assert len(losses) == 5
    acc = probe.evaluate(X, y)
    assert 0.0 <= acc <= 1.0


def test_linear_probe_predict_shape():
    cfg = ProbeConfig(n_classes=2, d_model=8)
    probe = LinearProbe(cfg)
    X = torch.randn(10, 8)
    preds = probe.predict(X)
    assert preds.shape == (10,)


def test_linear_probe_learns_linearly_separable():
    """Probe should achieve > 90% on linearly separable data."""
    cfg = ProbeConfig(n_classes=2, d_model=4, n_epochs=50, lr=0.01)
    probe = LinearProbe(cfg)
    torch.manual_seed(1)
    X = torch.cat([torch.randn(50, 4) + 3, torch.randn(50, 4) - 3])
    y = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    probe.fit(X, y)
    acc = probe.evaluate(X, y)
    assert acc > 0.9


def test_extract_layer_hiddens_shape(small_model):
    input_ids = torch.randint(0, 256, (2, 8))
    hiddens = extract_layer_hiddens(small_model, input_ids, layer_idx=0)
    assert hiddens.shape == (2, 8, 64)


def test_extract_layer_hiddens_layer1(small_model):
    input_ids = torch.randint(0, 256, (1, 6))
    h0 = extract_layer_hiddens(small_model, input_ids, layer_idx=0)
    h1 = extract_layer_hiddens(small_model, input_ids, layer_idx=1)
    assert h0.shape == h1.shape
    assert not torch.equal(h0, h1)  # different layers -> different hiddens


def test_probe_all_layers(small_model):
    torch.manual_seed(0)
    input_ids = torch.randint(0, 256, (4, 8))
    labels = torch.randint(0, 2, (4,))
    cfg = ProbeConfig(n_classes=2, d_model=64, n_epochs=3)
    results = probe_all_layers(small_model, input_ids, labels, n_layers=2, probe_cfg=cfg)
    assert set(results.keys()) == {0, 1}
    for acc in results.values():
        assert 0.0 <= acc <= 1.0
