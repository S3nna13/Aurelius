"""Tests for flow_matching_v2.py

Uses small dimensions so tests run quickly on CPU.
  d_model=16, vocab_size=16, seq_len=4, hidden=32, B=4, n_steps=4
"""

import math
import torch
import pytest

from src.model.flow_matching_v2 import (
    VectorField,
    ConditionalVectorField,
    FlowMatchingLoss,
    EulerFlowSampler,
    SequenceFlowModel,
    FlowMatchingConfig,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
D_MODEL   = 16
VOCAB     = 16
SEQ_LEN   = 4
HIDDEN    = 32
B         = 4
N_STEPS   = 4
D_COND    = 8


# ===========================================================================
# VectorField tests
# ===========================================================================

class TestVectorField:

    def test_output_shape(self):
        """VectorField forward returns [B, d_model]."""
        vf = VectorField(D_MODEL, hidden=HIDDEN)
        x = torch.randn(B, D_MODEL)
        t = torch.rand(B, 1)
        v = vf(x, t)
        assert v.shape == (B, D_MODEL), f"Expected {(B, D_MODEL)}, got {v.shape}"

    def test_output_depends_on_t(self):
        """Different time inputs produce different velocity outputs."""
        vf = VectorField(D_MODEL, hidden=HIDDEN)
        x = torch.randn(B, D_MODEL)
        t1 = torch.zeros(B, 1)
        t2 = torch.ones(B, 1)
        v1 = vf(x, t1)
        v2 = vf(x, t2)
        assert not torch.allclose(v1, v2), "Velocity should differ for t=0 vs t=1"

    def test_n_layers_respected(self):
        """Model with n_layers=3 still produces correct output shape."""
        vf = VectorField(D_MODEL, hidden=HIDDEN, n_layers=3)
        x = torch.randn(B, D_MODEL)
        t = torch.rand(B, 1)
        v = vf(x, t)
        assert v.shape == (B, D_MODEL)

    def test_output_is_float_tensor(self):
        """Output dtype matches input dtype (float32)."""
        vf = VectorField(D_MODEL, hidden=HIDDEN)
        x = torch.randn(B, D_MODEL)
        t = torch.rand(B, 1)
        v = vf(x, t)
        assert v.dtype == torch.float32


# ===========================================================================
# ConditionalVectorField tests
# ===========================================================================

class TestConditionalVectorField:

    def test_output_shape(self):
        """ConditionalVectorField forward returns [B, d_model]."""
        cvf = ConditionalVectorField(D_MODEL, D_COND, hidden=HIDDEN)
        x    = torch.randn(B, D_MODEL)
        t    = torch.rand(B, 1)
        cond = torch.randn(B, D_COND)
        v = cvf(x, t, cond)
        assert v.shape == (B, D_MODEL), f"Expected {(B, D_MODEL)}, got {v.shape}"

    def test_output_depends_on_cond(self):
        """Different conditioning vectors produce different velocities."""
        cvf  = ConditionalVectorField(D_MODEL, D_COND, hidden=HIDDEN)
        x    = torch.randn(B, D_MODEL)
        t    = torch.rand(B, 1)
        c1   = torch.zeros(B, D_COND)
        c2   = torch.ones(B, D_COND)
        v1 = cvf(x, t, c1)
        v2 = cvf(x, t, c2)
        assert not torch.allclose(v1, v2), "Velocity should differ for different cond"


# ===========================================================================
# FlowMatchingLoss tests
# ===========================================================================

class TestFlowMatchingLoss:

    def setup_method(self):
        self.loss_fn = FlowMatchingLoss()

    def test_ot_path_at_t0_equals_x0(self):
        """xt at t=0 should equal x0."""
        x0 = torch.randn(B, D_MODEL)
        x1 = torch.randn(B, D_MODEL)
        t  = torch.zeros(B, 1)
        xt, _ = FlowMatchingLoss.optimal_transport_path(x0, x1, t)
        assert torch.allclose(xt, x0, atol=1e-6), "xt(t=0) must equal x0"

    def test_ot_path_at_t1_equals_x1(self):
        """xt at t=1 should equal x1."""
        x0 = torch.randn(B, D_MODEL)
        x1 = torch.randn(B, D_MODEL)
        t  = torch.ones(B, 1)
        xt, _ = FlowMatchingLoss.optimal_transport_path(x0, x1, t)
        assert torch.allclose(xt, x1, atol=1e-6), "xt(t=1) must equal x1"

    def test_ot_path_ut_equals_x1_minus_x0(self):
        """Target velocity ut must equal x1 - x0."""
        x0 = torch.randn(B, D_MODEL)
        x1 = torch.randn(B, D_MODEL)
        t  = torch.rand(B, 1)
        _, ut = FlowMatchingLoss.optimal_transport_path(x0, x1, t)
        expected = x1 - x0
        assert torch.allclose(ut, expected, atol=1e-6), "ut must be x1 - x0"

    def test_cfm_loss_finite_nonneg(self):
        """cfm_loss returns a finite non-negative scalar."""
        vf   = VectorField(D_MODEL, hidden=HIDDEN)
        x1   = torch.randn(B, D_MODEL)
        loss = self.loss_fn.cfm_loss(vf, x1)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert math.isfinite(loss.item()), "Loss must be finite"
        assert loss.item() >= 0.0, "Loss must be non-negative"

    def test_conditional_cfm_loss_finite(self):
        """conditional_cfm_loss returns a finite scalar."""
        cvf  = ConditionalVectorField(D_MODEL, D_COND, hidden=HIDDEN)
        x1   = torch.randn(B, D_MODEL)
        cond = torch.randn(B, D_COND)
        loss = self.loss_fn.conditional_cfm_loss(cvf, x1, cond)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert math.isfinite(loss.item()), "Conditional loss must be finite"

    def test_cfm_loss_gradients_flow(self):
        """Gradients must flow through cfm_loss to model parameters."""
        vf   = VectorField(D_MODEL, hidden=HIDDEN)
        x1   = torch.randn(B, D_MODEL)
        loss = self.loss_fn.cfm_loss(vf, x1)
        loss.backward()
        grad_found = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in vf.parameters()
        )
        assert grad_found, "At least one parameter must have a non-zero gradient"


# ===========================================================================
# EulerFlowSampler tests
# ===========================================================================

class TestEulerFlowSampler:

    def test_sample_output_shape(self):
        """sample() returns [B, d_model]."""
        vf      = VectorField(D_MODEL, hidden=HIDDEN)
        sampler = EulerFlowSampler(n_steps=N_STEPS)
        x0      = torch.randn(B, D_MODEL)
        x1      = sampler.sample(vf, x0)
        assert x1.shape == (B, D_MODEL), f"Expected {(B, D_MODEL)}, got {x1.shape}"

    def test_sample_output_differs_from_input(self):
        """Integrating the ODE should change x (unless vf is identically 0)."""
        vf      = VectorField(D_MODEL, hidden=HIDDEN)
        sampler = EulerFlowSampler(n_steps=N_STEPS)
        x0      = torch.randn(B, D_MODEL)
        x1      = sampler.sample(vf, x0)
        assert not torch.allclose(x0, x1), "x should change after Euler integration"

    def test_sample_trajectory_length(self):
        """sample_trajectory returns n_steps+1 states."""
        vf      = VectorField(D_MODEL, hidden=HIDDEN)
        sampler = EulerFlowSampler(n_steps=N_STEPS)
        x0      = torch.randn(B, D_MODEL)
        traj    = sampler.sample_trajectory(vf, x0, n_steps=N_STEPS)
        expected_len = N_STEPS + 1
        assert len(traj) == expected_len, (
            f"Trajectory length should be {expected_len}, got {len(traj)}"
        )

    def test_sample_trajectory_first_element_is_x0(self):
        """First element of the trajectory must be x0."""
        vf      = VectorField(D_MODEL, hidden=HIDDEN)
        sampler = EulerFlowSampler(n_steps=N_STEPS)
        x0      = torch.randn(B, D_MODEL)
        traj    = sampler.sample_trajectory(vf, x0, n_steps=N_STEPS)
        assert torch.allclose(traj[0], x0), "First trajectory element must equal x0"

    def test_sample_trajectory_each_element_correct_shape(self):
        """Every trajectory element has shape [B, d_model]."""
        vf      = VectorField(D_MODEL, hidden=HIDDEN)
        sampler = EulerFlowSampler(n_steps=N_STEPS)
        x0      = torch.randn(B, D_MODEL)
        traj    = sampler.sample_trajectory(vf, x0, n_steps=N_STEPS)
        for i, state in enumerate(traj):
            assert state.shape == (B, D_MODEL), (
                f"Trajectory[{i}] shape {state.shape} != {(B, D_MODEL)}"
            )


# ===========================================================================
# SequenceFlowModel tests
# ===========================================================================

class TestSequenceFlowModel:

    def setup_method(self):
        self.model = SequenceFlowModel(D_MODEL, VOCAB, SEQ_LEN)

    def test_flow_loss_finite_scalar(self):
        """flow_loss should return a finite scalar."""
        ids  = torch.randint(0, VOCAB, (B, SEQ_LEN))
        loss = self.model.flow_loss(ids)
        assert loss.ndim == 0, "flow_loss should be a scalar"
        assert math.isfinite(loss.item()), "flow_loss must be finite"

    def test_sample_sequence_output_shape(self):
        """sample_sequence returns [n_samples, T, vocab_size]."""
        n = B
        logits = self.model.sample_sequence(n)
        expected = (n, SEQ_LEN, VOCAB)
        assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"

    def test_encode_shape(self):
        """encode returns [B, d_model * seq_len]."""
        ids = torch.randint(0, VOCAB, (B, SEQ_LEN))
        z   = self.model.encode(ids)
        assert z.shape == (B, D_MODEL * SEQ_LEN)

    def test_decode_shape(self):
        """decode returns [B, T, vocab_size]."""
        z      = torch.randn(B, D_MODEL * SEQ_LEN)
        logits = self.model.decode(z)
        assert logits.shape == (B, SEQ_LEN, VOCAB)


# ===========================================================================
# FlowMatchingConfig test
# ===========================================================================

class TestFlowMatchingConfig:

    def test_defaults(self):
        """FlowMatchingConfig has the specified default values."""
        cfg = FlowMatchingConfig()
        assert cfg.d_model    == 32
        assert cfg.vocab_size == 64
        assert cfg.seq_len    == 8
        assert cfg.hidden     == 64
        assert cfg.n_layers   == 2
        assert cfg.n_steps    == 10
        assert cfg.d_cond     == 16

    def test_custom_values(self):
        """FlowMatchingConfig accepts custom values."""
        cfg = FlowMatchingConfig(d_model=64, vocab_size=128, seq_len=16)
        assert cfg.d_model    == 64
        assert cfg.vocab_size == 128
        assert cfg.seq_len    == 16
