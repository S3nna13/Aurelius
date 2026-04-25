"""Tests for src/alignment/rlhf_pipeline.py."""

import pytest
from src.alignment.rlhf_pipeline import (
    RLHFPhase,
    PhaseConfig,
    PhaseResult,
    RLHFPipeline,
    RLHF_PIPELINE,
)


# ---------------------------------------------------------------------------
# RLHFPhase enum
# ---------------------------------------------------------------------------

class TestRLHFPhase:
    def test_sft_value(self):
        assert RLHFPhase.SFT == "sft"

    def test_reward_model_value(self):
        assert RLHFPhase.REWARD_MODEL == "reward_model"

    def test_ppo_value(self):
        assert RLHFPhase.PPO == "ppo"

    def test_eval_value(self):
        assert RLHFPhase.EVAL == "eval"

    def test_is_str_subclass(self):
        assert isinstance(RLHFPhase.SFT, str)

    def test_enum_count(self):
        assert len(RLHFPhase) == 4


# ---------------------------------------------------------------------------
# PhaseConfig dataclass
# ---------------------------------------------------------------------------

class TestPhaseConfig:
    def test_required_phase_field(self):
        cfg = PhaseConfig(phase=RLHFPhase.SFT)
        assert cfg.phase == RLHFPhase.SFT

    def test_default_n_epochs(self):
        cfg = PhaseConfig(phase=RLHFPhase.SFT)
        assert cfg.n_epochs == 1

    def test_default_batch_size(self):
        cfg = PhaseConfig(phase=RLHFPhase.SFT)
        assert cfg.batch_size == 32

    def test_default_learning_rate(self):
        cfg = PhaseConfig(phase=RLHFPhase.SFT)
        assert cfg.learning_rate == pytest.approx(1e-4)

    def test_default_metadata_empty_dict(self):
        cfg = PhaseConfig(phase=RLHFPhase.SFT)
        assert cfg.metadata == {}

    def test_metadata_not_shared(self):
        a = PhaseConfig(phase=RLHFPhase.SFT)
        b = PhaseConfig(phase=RLHFPhase.PPO)
        a.metadata["x"] = 1
        assert "x" not in b.metadata

    def test_override_fields(self):
        cfg = PhaseConfig(phase=RLHFPhase.PPO, n_epochs=5, batch_size=64, learning_rate=3e-4)
        assert cfg.n_epochs == 5
        assert cfg.batch_size == 64
        assert cfg.learning_rate == pytest.approx(3e-4)


# ---------------------------------------------------------------------------
# PhaseResult dataclass
# ---------------------------------------------------------------------------

class TestPhaseResult:
    def test_fields_present(self):
        r = PhaseResult(phase=RLHFPhase.SFT, epoch=0, loss=1.5)
        assert r.phase == RLHFPhase.SFT
        assert r.epoch == 0
        assert r.loss == pytest.approx(1.5)

    def test_default_metrics_empty(self):
        r = PhaseResult(phase=RLHFPhase.SFT, epoch=0, loss=1.0)
        assert r.metrics == {}

    def test_metrics_not_shared(self):
        a = PhaseResult(phase=RLHFPhase.SFT, epoch=0, loss=1.0)
        b = PhaseResult(phase=RLHFPhase.PPO, epoch=1, loss=0.5)
        a.metrics["acc"] = 0.9
        assert "acc" not in b.metrics

    def test_custom_metrics(self):
        r = PhaseResult(phase=RLHFPhase.PPO, epoch=2, loss=0.3, metrics={"reward": 4.2})
        assert r.metrics["reward"] == pytest.approx(4.2)


# ---------------------------------------------------------------------------
# RLHFPipeline
# ---------------------------------------------------------------------------

class TestRLHFPipelineDefaults:
    def setup_method(self):
        self.pipeline = RLHFPipeline()

    def test_default_has_three_phases(self):
        assert len(self.pipeline._phases) == 3

    def test_default_phases_order(self):
        phases = [pc.phase for pc in self.pipeline._phases]
        assert phases == [RLHFPhase.SFT, RLHFPhase.REWARD_MODEL, RLHFPhase.PPO]

    def test_current_phase_starts_at_sft(self):
        assert self.pipeline.current_phase() == RLHFPhase.SFT

    def test_history_starts_empty(self):
        assert self.pipeline.history() == []


class TestRLHFPipelineAdvance:
    def setup_method(self):
        self.pipeline = RLHFPipeline()

    def test_advance_moves_to_reward_model(self):
        self.pipeline.advance_phase()
        assert self.pipeline.current_phase() == RLHFPhase.REWARD_MODEL

    def test_advance_moves_to_ppo(self):
        self.pipeline.advance_phase()
        self.pipeline.advance_phase()
        assert self.pipeline.current_phase() == RLHFPhase.PPO

    def test_advance_returns_true_when_not_at_end(self):
        result = self.pipeline.advance_phase()
        assert result is True

    def test_advance_at_end_returns_false(self):
        self.pipeline.advance_phase()
        self.pipeline.advance_phase()
        result = self.pipeline.advance_phase()
        assert result is False

    def test_current_phase_none_after_completion(self):
        self.pipeline.advance_phase()
        self.pipeline.advance_phase()
        self.pipeline.advance_phase()
        assert self.pipeline.current_phase() is None


class TestRLHFPipelineHistory:
    def setup_method(self):
        self.pipeline = RLHFPipeline()

    def test_log_result_appends(self):
        r = PhaseResult(phase=RLHFPhase.SFT, epoch=0, loss=1.0)
        self.pipeline.log_result(r)
        assert len(self.pipeline.history()) == 1

    def test_log_multiple_results(self):
        for i in range(3):
            self.pipeline.log_result(PhaseResult(phase=RLHFPhase.SFT, epoch=i, loss=float(i)))
        assert len(self.pipeline.history()) == 3

    def test_history_returns_all_without_filter(self):
        self.pipeline.log_result(PhaseResult(phase=RLHFPhase.SFT, epoch=0, loss=1.0))
        self.pipeline.log_result(PhaseResult(phase=RLHFPhase.PPO, epoch=0, loss=0.5))
        assert len(self.pipeline.history()) == 2

    def test_history_filters_by_phase(self):
        self.pipeline.log_result(PhaseResult(phase=RLHFPhase.SFT, epoch=0, loss=1.0))
        self.pipeline.log_result(PhaseResult(phase=RLHFPhase.PPO, epoch=0, loss=0.5))
        sft_history = self.pipeline.history(phase=RLHFPhase.SFT)
        assert len(sft_history) == 1
        assert sft_history[0].phase == RLHFPhase.SFT

    def test_history_filter_returns_empty_for_unlogged_phase(self):
        self.pipeline.log_result(PhaseResult(phase=RLHFPhase.SFT, epoch=0, loss=1.0))
        assert self.pipeline.history(phase=RLHFPhase.PPO) == []

    def test_history_returns_copy(self):
        r = PhaseResult(phase=RLHFPhase.SFT, epoch=0, loss=1.0)
        self.pipeline.log_result(r)
        h = self.pipeline.history()
        h.clear()
        assert len(self.pipeline.history()) == 1


class TestRLHFPipelineSummary:
    def setup_method(self):
        self.pipeline = RLHFPipeline()

    def test_summary_has_expected_keys(self):
        s = self.pipeline.summary()
        assert set(s.keys()) == {"phases_completed", "current_phase", "total_results"}

    def test_summary_phases_completed_initial(self):
        assert self.pipeline.summary()["phases_completed"] == 0

    def test_summary_current_phase_initial(self):
        assert self.pipeline.summary()["current_phase"] == "sft"

    def test_summary_total_results_initial(self):
        assert self.pipeline.summary()["total_results"] == 0

    def test_summary_phases_completed_after_advance(self):
        self.pipeline.advance_phase()
        assert self.pipeline.summary()["phases_completed"] == 1

    def test_summary_current_phase_after_advance(self):
        self.pipeline.advance_phase()
        assert self.pipeline.summary()["current_phase"] == "reward_model"

    def test_summary_total_results_after_log(self):
        self.pipeline.log_result(PhaseResult(phase=RLHFPhase.SFT, epoch=0, loss=1.0))
        assert self.pipeline.summary()["total_results"] == 1

    def test_summary_current_phase_done_after_completion(self):
        self.pipeline.advance_phase()
        self.pipeline.advance_phase()
        self.pipeline.advance_phase()
        assert self.pipeline.summary()["current_phase"] == "done"

    def test_summary_phases_completed_after_full_run(self):
        self.pipeline.advance_phase()
        self.pipeline.advance_phase()
        self.pipeline.advance_phase()
        assert self.pipeline.summary()["phases_completed"] == 3


# ---------------------------------------------------------------------------
# Custom phases
# ---------------------------------------------------------------------------

class TestRLHFPipelineCustomPhases:
    def test_custom_phases_respected(self):
        phases = [PhaseConfig(phase=RLHFPhase.SFT), PhaseConfig(phase=RLHFPhase.EVAL)]
        pipeline = RLHFPipeline(phases=phases)
        assert len(pipeline._phases) == 2

    def test_single_phase_pipeline(self):
        pipeline = RLHFPipeline(phases=[PhaseConfig(phase=RLHFPhase.EVAL)])
        assert pipeline.current_phase() == RLHFPhase.EVAL
        result = pipeline.advance_phase()
        assert result is False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestRLHFPipelineSingleton:
    def test_rlhf_pipeline_exists(self):
        assert RLHF_PIPELINE is not None

    def test_rlhf_pipeline_is_instance(self):
        assert isinstance(RLHF_PIPELINE, RLHFPipeline)
