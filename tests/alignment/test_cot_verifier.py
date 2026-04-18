"""
Tests for src/alignment/cot_verifier.py

Dimensions used throughout
--------------------------
B        = 2   (batch size)
T_step   = 8   (tokens per step)
n_steps  = 3   (steps in a chain)
d_model  = 16
vocab_size = 16
n_layers = 2
"""

import math
import torch
import pytest

from src.alignment.cot_verifier import (
    StepEncoder,
    ChainEncoder,
    VerifierHead,
    CoTVerifierModel,
    VerifierTrainer,
    ProcessRewardModel,
    VerifierConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

B = 2
T = 8
S = 3
D = 16
V = 16
L = 2


def make_step_ids(b=B, t=T, v=V):
    return torch.randint(0, v, (b, t))


def make_step_ids_list(s=S, b=B, t=T, v=V):
    return [make_step_ids(b, t, v) for _ in range(s)]


# ---------------------------------------------------------------------------
# StepEncoder tests
# ---------------------------------------------------------------------------

class TestStepEncoder:
    def setup_method(self):
        self.enc = StepEncoder(d_model=D, vocab_size=V, n_layers=L)

    def test_output_shape(self):
        ids = make_step_ids()
        out = self.enc(ids)
        assert out.shape == (B, D), f"Expected ({B}, {D}), got {out.shape}"

    def test_different_inputs_different_encodings(self):
        torch.manual_seed(0)
        ids_a = torch.zeros(B, T, dtype=torch.long)
        ids_b = torch.ones(B, T, dtype=torch.long)
        out_a = self.enc(ids_a)
        out_b = self.enc(ids_b)
        # The two inputs differ → output representations must differ
        assert not torch.allclose(out_a, out_b), \
            "Different token inputs should produce different encodings"

    def test_output_is_float(self):
        ids = make_step_ids()
        out = self.enc(ids)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self):
        ids = make_step_ids()
        out = self.enc(ids)
        assert not torch.isnan(out).any(), "StepEncoder output contains NaN"

    def test_batch_size_one(self):
        ids = make_step_ids(b=1)
        out = self.enc(ids)
        assert out.shape == (1, D)


# ---------------------------------------------------------------------------
# ChainEncoder tests
# ---------------------------------------------------------------------------

class TestChainEncoder:
    def setup_method(self):
        self.enc = ChainEncoder(d_model=D, n_layers=L)

    def test_output_shape(self):
        x = torch.randn(B, S, D)
        out = self.enc(x)
        assert out.shape == (B, S, D), f"Expected ({B}, {S}, {D}), got {out.shape}"

    def test_contextual_encoding_differs_from_input(self):
        x = torch.randn(B, S, D)
        out = self.enc(x)
        # Transformer with positional embeddings should alter the representations
        assert not torch.allclose(out, x), \
            "ChainEncoder output should differ from its input after transformation"

    def test_no_nan_in_output(self):
        x = torch.randn(B, S, D)
        out = self.enc(x)
        assert not torch.isnan(out).any()

    def test_different_step_counts(self):
        for n in [1, 4, 8]:
            x = torch.randn(B, n, D)
            out = self.enc(x)
            assert out.shape == (B, n, D)


# ---------------------------------------------------------------------------
# VerifierHead tests
# ---------------------------------------------------------------------------

class TestVerifierHead:
    def setup_method(self):
        self.head = VerifierHead(d_model=D)

    def _make_chain_enc(self):
        return torch.randn(B, S, D)

    def test_step_scores_shape(self):
        step_scores, _ = self.head(self._make_chain_enc())
        assert step_scores.shape == (B, S), \
            f"step_scores: expected ({B}, {S}), got {step_scores.shape}"

    def test_chain_score_shape(self):
        _, chain_score = self.head(self._make_chain_enc())
        assert chain_score.shape == (B,), \
            f"chain_score: expected ({B},), got {chain_score.shape}"

    def test_step_scores_in_unit_interval(self):
        step_scores, _ = self.head(self._make_chain_enc())
        assert (step_scores >= 0.0).all() and (step_scores <= 1.0).all(), \
            "step_scores should be in [0, 1]"

    def test_chain_score_in_unit_interval(self):
        _, chain_score = self.head(self._make_chain_enc())
        assert (chain_score >= 0.0).all() and (chain_score <= 1.0).all(), \
            "chain_score should be in [0, 1]"

    def test_no_nan(self):
        step_scores, chain_score = self.head(self._make_chain_enc())
        assert not torch.isnan(step_scores).any()
        assert not torch.isnan(chain_score).any()


# ---------------------------------------------------------------------------
# CoTVerifierModel tests
# ---------------------------------------------------------------------------

class TestCoTVerifierModel:
    def setup_method(self):
        self.model = CoTVerifierModel(d_model=D, vocab_size=V, n_layers=L)

    def test_forward_output_shapes(self):
        ids_list = make_step_ids_list()
        step_scores, chain_score = self.model(ids_list)
        assert step_scores.shape == (B, S), \
            f"step_scores: expected ({B}, {S}), got {step_scores.shape}"
        assert chain_score.shape == (B,), \
            f"chain_score: expected ({B},), got {chain_score.shape}"

    def test_step_scores_in_unit_interval(self):
        ids_list = make_step_ids_list()
        step_scores, _ = self.model(ids_list)
        assert (step_scores >= 0.0).all() and (step_scores <= 1.0).all()

    def test_chain_score_in_unit_interval(self):
        ids_list = make_step_ids_list()
        _, chain_score = self.model(ids_list)
        assert (chain_score >= 0.0).all() and (chain_score <= 1.0).all()

    def test_backward_gradients_flow(self):
        ids_list = make_step_ids_list()
        step_scores, chain_score = self.model(ids_list)
        loss = step_scores.sum() + chain_score.sum()
        loss.backward()
        # At least one parameter should have a non-None, non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in self.model.parameters()
        )
        assert has_grad, "No gradients flowed through CoTVerifierModel"

    def test_single_step_chain(self):
        ids_list = make_step_ids_list(s=1)
        step_scores, chain_score = self.model(ids_list)
        assert step_scores.shape == (B, 1)
        assert chain_score.shape == (B,)


# ---------------------------------------------------------------------------
# VerifierTrainer tests
# ---------------------------------------------------------------------------

class TestVerifierTrainer:
    def setup_method(self):
        self.model = CoTVerifierModel(d_model=D, vocab_size=V, n_layers=L)
        self.trainer = VerifierTrainer(self.model, lr=1e-3)

    def _scores_labels(self):
        ids_list = make_step_ids_list()
        step_scores, chain_score = self.model(ids_list)
        step_labels = torch.randint(0, 2, (B, S)).float()
        chain_labels = torch.randint(0, 2, (B,)).float()
        return step_scores, chain_score, step_labels, chain_labels

    def test_step_loss_finite_scalar(self):
        step_scores, _, step_labels, _ = self._scores_labels()
        loss = self.trainer.step_loss(step_scores, step_labels)
        assert loss.shape == torch.Size([]), "step_loss should be a scalar"
        assert math.isfinite(loss.item()), "step_loss should be finite"

    def test_chain_loss_finite_scalar(self):
        _, chain_score, _, chain_labels = self._scores_labels()
        loss = self.trainer.chain_loss(chain_score, chain_labels)
        assert loss.shape == torch.Size([]), "chain_loss should be a scalar"
        assert math.isfinite(loss.item()), "chain_loss should be finite"

    def test_combined_loss_between_individual_losses(self):
        step_scores, chain_score, step_labels, chain_labels = self._scores_labels()
        sl = self.trainer.step_loss(step_scores, step_labels).item()
        cl = self.trainer.chain_loss(chain_score, chain_labels).item()
        alpha = 0.5
        combined = self.trainer.combined_loss(
            step_scores, chain_score, step_labels, chain_labels, alpha
        ).item()
        expected = alpha * sl + (1.0 - alpha) * cl
        assert math.isclose(combined, expected, rel_tol=1e-5), \
            f"combined_loss {combined} != {expected}"

    def test_train_step_returns_finite_loss(self):
        ids_list = make_step_ids_list()
        step_labels = torch.randint(0, 2, (B, S)).float()
        chain_labels = torch.randint(0, 2, (B,)).float()
        loss = self.trainer.train_step(ids_list, step_labels, chain_labels)
        assert math.isfinite(loss.item()), "train_step loss should be finite"

    def test_train_step_returns_scalar(self):
        ids_list = make_step_ids_list()
        step_labels = torch.randint(0, 2, (B, S)).float()
        chain_labels = torch.randint(0, 2, (B,)).float()
        loss = self.trainer.train_step(ids_list, step_labels, chain_labels)
        assert loss.shape == torch.Size([])

    def test_combined_loss_alpha_zero_equals_chain_loss(self):
        step_scores, chain_score, step_labels, chain_labels = self._scores_labels()
        cl = self.trainer.chain_loss(chain_score, chain_labels).item()
        combined = self.trainer.combined_loss(
            step_scores, chain_score, step_labels, chain_labels, alpha=0.0
        ).item()
        assert math.isclose(combined, cl, rel_tol=1e-5)

    def test_combined_loss_alpha_one_equals_step_loss(self):
        step_scores, chain_score, step_labels, chain_labels = self._scores_labels()
        sl = self.trainer.step_loss(step_scores, step_labels).item()
        combined = self.trainer.combined_loss(
            step_scores, chain_score, step_labels, chain_labels, alpha=1.0
        ).item()
        assert math.isclose(combined, sl, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# ProcessRewardModel tests
# ---------------------------------------------------------------------------

class TestProcessRewardModel:
    def setup_method(self):
        self.prm = ProcessRewardModel(d_model=D, vocab_size=V, n_layers=L)

    def test_forward_output_shape(self):
        ids = make_step_ids()
        out = self.prm(ids)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"

    def test_forward_output_in_unit_interval(self):
        ids = make_step_ids()
        out = self.prm(ids)
        assert (out >= 0.0).all() and (out <= 1.0).all(), \
            "ProcessRewardModel output should be in [0, 1]"

    def test_outcome_from_steps_shape(self):
        step_scores = torch.rand(B, S)
        out = self.prm.outcome_from_steps(step_scores)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"

    def test_outcome_from_steps_in_unit_interval(self):
        step_scores = torch.rand(B, S)
        out = self.prm.outcome_from_steps(step_scores)
        assert (out >= 0.0).all() and (out <= 1.0).all(), \
            "outcome_from_steps should be in [0, 1]"

    def test_outcome_from_steps_product_correctness(self):
        step_scores = torch.tensor([[0.5, 0.4, 0.8], [1.0, 1.0, 1.0]])
        out = self.prm.outcome_from_steps(step_scores)
        expected_0 = 0.5 * 0.4 * 0.8
        expected_1 = 1.0
        assert math.isclose(out[0].item(), expected_0, rel_tol=1e-5), \
            f"outcome[0]: {out[0].item()} != {expected_0}"
        assert math.isclose(out[1].item(), expected_1, rel_tol=1e-5), \
            f"outcome[1]: {out[1].item()} != {expected_1}"

    def test_no_nan_in_forward(self):
        ids = make_step_ids()
        out = self.prm(ids)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# VerifierConfig tests
# ---------------------------------------------------------------------------

class TestVerifierConfig:
    def test_defaults(self):
        cfg = VerifierConfig()
        assert cfg.d_model == 32
        assert cfg.vocab_size == 64
        assert cfg.n_layers == 2
        assert cfg.lr == 1e-4
        assert cfg.alpha == 0.5
        assert cfg.max_steps == 8
        assert cfg.max_step_len == 16

    def test_custom_values(self):
        cfg = VerifierConfig(d_model=64, vocab_size=128, lr=1e-3)
        assert cfg.d_model == 64
        assert cfg.vocab_size == 128
        assert math.isclose(cfg.lr, 1e-3)

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(VerifierConfig)
