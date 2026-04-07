"""Tests for the autonomous training pipeline."""
import torch
import pytest
from src.training.pipeline import TrainingPipeline, PipelineConfig, PipelineResult
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    return AureliusTransformer(cfg)


class FakeTok:
    def encode(self, text): return [1, 2, 3]
    def decode(self, ids): return "response"


def _make_sft_batch(n=4, seq_len=16):
    ids = torch.randint(0, 256, (seq_len,))
    return [{"input_ids": ids.clone(), "labels": ids.clone()} for _ in range(n)]


def test_pipeline_run_eval_only(tmp_path, small_model):
    """Pipeline with only eval phase must return PPL results."""
    cfg = PipelineConfig(
        output_dir=str(tmp_path),
        run_sft=False,
        run_grpo=False,
        run_eval=True,
        n_iterations=1,
        eval_n_sequences=3,
        eval_seq_len=16,
    )
    pipeline = TrainingPipeline(small_model, FakeTok(), cfg)
    results = pipeline.run()

    assert len(results) == 1
    assert results[0].eval_ppl is not None
    import math
    assert math.isfinite(results[0].eval_ppl)
    assert results[0].sft_loss is None


def test_pipeline_sft_phase_returns_loss(tmp_path, small_model):
    """Pipeline SFT phase must return a finite loss."""
    cfg = PipelineConfig(
        output_dir=str(tmp_path),
        run_sft=True,
        run_grpo=False,
        run_eval=False,
        n_iterations=1,
        sft_epochs=1,
        sft_batch_size=2,
    )
    pipeline = TrainingPipeline(small_model, FakeTok(), cfg)
    train_data = _make_sft_batch(n=4, seq_len=16)
    results = pipeline.run(train_data=train_data)

    assert len(results) == 1
    assert results[0].sft_loss is not None
    import math
    assert math.isfinite(results[0].sft_loss)


def test_pipeline_creates_checkpoint(tmp_path, small_model):
    """Pipeline must create a checkpoint after each iteration."""
    cfg = PipelineConfig(
        output_dir=str(tmp_path),
        run_sft=False,
        run_grpo=False,
        run_eval=True,
        n_iterations=2,
        eval_n_sequences=2,
        eval_seq_len=8,
    )
    pipeline = TrainingPipeline(small_model, FakeTok(), cfg)
    results = pipeline.run()

    assert all(r.checkpoint_path is not None for r in results)
    # Checkpoint directories must exist
    from pathlib import Path
    for r in results:
        assert Path(r.checkpoint_path).exists()


def test_pipeline_history_accumulates(tmp_path, small_model):
    """Pipeline history must contain one entry per iteration."""
    cfg = PipelineConfig(
        output_dir=str(tmp_path),
        run_sft=False,
        run_grpo=False,
        run_eval=False,
        n_iterations=3,
    )
    pipeline = TrainingPipeline(small_model, FakeTok(), cfg)
    results = pipeline.run()

    assert len(results) == 3
    assert [r.iteration for r in results] == [1, 2, 3]
    assert len(pipeline.history) == 3
