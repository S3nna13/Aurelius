"""Tests for structural probes and MDL probing (probing_v2)."""

import math

import pytest
import torch
import torch.nn as nn

from src.eval.probing_v2 import (
    MDLProbeDataset,
    MDLProbeTrainer,
    ProbingBenchmark,
    StructuralProbe,
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
D_MODEL = 16
N_CLASSES = 4
N = 40
T = 10  # sequence length for structural probe tests
BATCH = 8
RANK = 8


@pytest.fixture
def probe():
    torch.manual_seed(0)
    return StructuralProbe(d_model=D_MODEL, rank=RANK)


@pytest.fixture
def hidden_pair():
    torch.manual_seed(1)
    h_i = torch.randn(BATCH, D_MODEL)
    h_j = torch.randn(BATCH, D_MODEL)
    return h_i, h_j


@pytest.fixture
def hidden_seq():
    torch.manual_seed(2)
    return torch.randn(T, D_MODEL)


@pytest.fixture
def target_distances():
    """Symmetric non-negative distance matrix with zero diagonal."""
    torch.manual_seed(3)
    raw = torch.rand(T, T)
    sym = (raw + raw.T) / 2
    sym.fill_diagonal_(0.0)
    return sym


@pytest.fixture
def repr_and_labels():
    torch.manual_seed(4)
    repr_ = torch.randn(N, D_MODEL)
    labels = torch.randint(0, N_CLASSES, (N,))
    return repr_, labels


@pytest.fixture
def mdl_dataset(repr_and_labels):
    repr_, labels = repr_and_labels
    return MDLProbeDataset(repr_, labels, n_splits=4)


@pytest.fixture
def mdl_trainer():
    return MDLProbeTrainer(d_model=D_MODEL, n_classes=N_CLASSES, n_epochs_per_split=5)


@pytest.fixture
def benchmark():
    return ProbingBenchmark()


@pytest.fixture
def train_test_split(repr_and_labels):
    repr_, labels = repr_and_labels
    n_train = int(0.8 * N)
    return (
        repr_[:n_train], labels[:n_train],
        repr_[n_train:], labels[n_train:],
    )


# -------------------------------------------------------------------
# StructuralProbe tests
# -------------------------------------------------------------------

def test_distance_output_shape(probe, hidden_pair):
    """distance() should return (B,) shaped tensor."""
    h_i, h_j = hidden_pair
    dist = probe.distance(h_i, h_j)
    assert dist.shape == (BATCH,), f"Expected ({BATCH},), got {dist.shape}"


def test_distance_non_negative(probe, hidden_pair):
    """Squared distances must be non-negative."""
    h_i, h_j = hidden_pair
    dist = probe.distance(h_i, h_j)
    assert (dist >= 0).all(), "distance() returned negative values"


def test_distance_matrix_shape(probe, hidden_seq):
    """distance_matrix() should return (T, T) tensor."""
    dm = probe.distance_matrix(hidden_seq)
    assert dm.shape == (T, T), f"Expected ({T}, {T}), got {dm.shape}"


def test_distance_matrix_diagonal_zero(probe, hidden_seq):
    """Diagonal of distance_matrix should be ~0 (distance from a point to itself)."""
    dm = probe.distance_matrix(hidden_seq)
    diag = dm.diagonal()
    assert (diag.abs() < 1e-5).all(), f"Diagonal not ~0: max={diag.abs().max().item():.2e}"


def test_structural_probe_loss_scalar(probe, hidden_seq, target_distances):
    """loss() should return a scalar tensor."""
    loss = probe.loss(hidden_seq, target_distances)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0


def test_loss_gradients_flow(probe, hidden_seq, target_distances):
    """Gradients should flow back to the B parameter."""
    loss = probe.loss(hidden_seq, target_distances)
    loss.backward()
    assert probe.B.grad is not None, "No gradient on probe.B"
    assert not torch.isnan(probe.B.grad).any(), "NaN in probe.B gradient"


# -------------------------------------------------------------------
# MDLProbeDataset tests
# -------------------------------------------------------------------

def test_split_fractions_length(mdl_dataset):
    """split_fractions() should return a list of length n_splits."""
    fracs = mdl_dataset.split_fractions()
    assert len(fracs) == mdl_dataset.n_splits


def test_split_fractions_values(mdl_dataset):
    """Fractions should be [1/2, 1/4, 1/8, 1/16] for n_splits=4."""
    fracs = mdl_dataset.split_fractions()
    expected = [1.0 / (2 ** k) for k in range(1, mdl_dataset.n_splits + 1)]
    for f, e in zip(fracs, expected):
        assert abs(f - e) < 1e-9


def test_get_split_shapes(mdl_dataset):
    """get_split should return tensors with consistent shapes."""
    fraction = 0.5
    tr, trl, te, tel = mdl_dataset.get_split(fraction)
    n_train = max(1, int(fraction * N))
    n_test = N - n_train
    assert tr.shape == (n_train, D_MODEL)
    assert trl.shape == (n_train,)
    assert te.shape == (n_test, D_MODEL)
    assert tel.shape == (n_test,)


# -------------------------------------------------------------------
# MDLProbeTrainer tests
# -------------------------------------------------------------------

def test_train_on_split_returns_module(mdl_trainer, repr_and_labels):
    """train_on_split() should return an nn.Module."""
    repr_, labels = repr_and_labels
    n_train = 20
    probe = mdl_trainer.train_on_split(repr_[:n_train], labels[:n_train])
    assert isinstance(probe, nn.Module)


def test_codelength_positive(mdl_trainer, repr_and_labels):
    """codelength() should return a positive float."""
    repr_, labels = repr_and_labels
    n_train = 20
    probe = mdl_trainer.train_on_split(repr_[:n_train], labels[:n_train])
    cl = mdl_trainer.codelength(probe, repr_[n_train:], labels[n_train:])
    assert isinstance(cl, float)
    assert cl > 0.0, f"Expected positive codelength, got {cl}"


def test_mdl_score_keys(mdl_trainer, mdl_dataset):
    """mdl_score() should return dict with required keys."""
    result = mdl_trainer.mdl_score(mdl_dataset)
    assert "total_codelength" in result
    assert "uniform_codelength" in result
    assert "compression" in result


def test_compression_range(mdl_trainer, mdl_dataset):
    """compression should be <= 1.0 (could be negative for random data)."""
    result = mdl_trainer.mdl_score(mdl_dataset)
    assert result["compression"] <= 1.0, f"compression > 1: {result['compression']}"


# -------------------------------------------------------------------
# ProbingBenchmark tests
# -------------------------------------------------------------------

def test_run_linear_probe_keys(benchmark, train_test_split):
    """run_linear_probe() should return dict with expected keys."""
    tr, trl, te, tel = train_test_split
    result = benchmark.run_linear_probe(tr, trl, te, tel, n_epochs=5)
    assert "train_acc" in result
    assert "test_acc" in result
    assert "n_classes" in result


def test_train_acc_range(benchmark, train_test_split):
    """train_acc should be in [0, 1]."""
    tr, trl, te, tel = train_test_split
    result = benchmark.run_linear_probe(tr, trl, te, tel, n_epochs=5)
    assert 0.0 <= result["train_acc"] <= 1.0, f"train_acc out of range: {result['train_acc']}"


def test_compare_layers_length(benchmark, repr_and_labels):
    """compare_layers() should return a list with the same length as layer_reprs."""
    repr_, labels = repr_and_labels
    n_layers = 4
    layer_reprs = [repr_ + torch.randn_like(repr_) * 0.1 for _ in range(n_layers)]
    results = benchmark.compare_layers(layer_reprs, labels)
    assert len(results) == n_layers, f"Expected {n_layers} results, got {len(results)}"
