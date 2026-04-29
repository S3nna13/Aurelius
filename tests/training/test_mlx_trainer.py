"""Tests for MLX-based training backend (src.training.mlx_trainer)."""

import logging
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.mlx_trainer import MLXConfig, MLXTrainer, main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cleanup_mlx_modules():
    """Remove any mlx None entries from sys.modules between tests."""
    yield
    for key in ("mlx", "mlx.core", "mlx.nn", "mlx.optim"):
        if sys.modules.get(key) is None:
            sys.modules.pop(key, None)


@pytest.fixture
def mock_mlx():
    """Provide mocked mlx modules via sys.modules patch."""
    mx_mock = MagicMock()
    nn_mock = MagicMock()
    optim_mock = MagicMock()
    mx_mock.default_device.return_value = "cpu"

    mlx_mock = MagicMock()
    mlx_mock.core = mx_mock
    mlx_mock.nn = nn_mock
    mlx_mock.optim = optim_mock

    modules = {
        "mlx": mlx_mock,
        "mlx.core": mx_mock,
        "mlx.nn": nn_mock,
        "mlx.optim": optim_mock,
    }

    with patch.dict(sys.modules, modules):
        yield {"mx": mx_mock, "nn": nn_mock, "optim": optim_mock}


@pytest.fixture
def mock_safetensors():
    """Provide mocked safetensors module via sys.modules patch."""
    safe_open_mock = MagicMock()
    save_file_mock = MagicMock()

    st_mock = MagicMock()
    st_mock.safe_open = safe_open_mock

    st_torch_mock = MagicMock()
    st_torch_mock.save_file = save_file_mock
    st_mock.torch = st_torch_mock

    modules = {
        "safetensors": st_mock,
        "safetensors.torch": st_torch_mock,
    }

    with patch.dict(sys.modules, modules):
        yield {"safe_open": safe_open_mock, "save_file": save_file_mock}


# ---------------------------------------------------------------------------
# MLXConfig defaults and validation
# ---------------------------------------------------------------------------


def test_mlx_config_defaults():
    """MLXConfig must expose the expected defaults."""
    cfg = MLXConfig()
    assert cfg.model_path == ""
    assert cfg.learning_rate == 3e-4
    assert cfg.min_lr == 3e-5
    assert cfg.warmup_steps == 2000
    assert cfg.total_steps == 143000
    assert cfg.batch_size == 1
    assert cfg.seq_len == 8192
    assert cfg.global_batch_tokens == 2_097_152
    assert cfg.max_grad_norm == 1.0
    assert cfg.save_interval_steps == 4800
    assert cfg.save_dir == "checkpoints/aurelius-mlx"
    assert cfg.log_interval_steps == 10
    assert cfg.train_data_dir == "data/pretrain/train"
    assert cfg.val_data_dir == "data/pretrain/val"
    assert cfg.seed == 42
    assert cfg.gradient_checkpointing is True
    assert cfg.num_workers == 4
    assert cfg.weight_decay == 0.1
    assert cfg.beta1 == 0.9
    assert cfg.beta2 == 0.95
    assert cfg.eps == 1e-8


def test_mlx_config_custom_values():
    """MLXConfig must accept and retain overridden values."""
    cfg = MLXConfig(
        model_path="/models/test",
        learning_rate=1e-3,
        min_lr=1e-6,
        warmup_steps=100,
        total_steps=1000,
        batch_size=2,
        seq_len=2048,
        global_batch_tokens=524_288,
        max_grad_norm=0.5,
        save_interval_steps=100,
        save_dir="checkpoints/custom",
        log_interval_steps=5,
        train_data_dir="data/train",
        val_data_dir="data/val",
        seed=123,
        gradient_checkpointing=False,
        num_workers=8,
        weight_decay=0.01,
        beta1=0.95,
        beta2=0.999,
        eps=1e-6,
    )
    assert cfg.model_path == "/models/test"
    assert cfg.learning_rate == 1e-3
    assert cfg.min_lr == 1e-6
    assert cfg.warmup_steps == 100
    assert cfg.total_steps == 1000
    assert cfg.batch_size == 2
    assert cfg.seq_len == 2048
    assert cfg.global_batch_tokens == 524_288
    assert cfg.max_grad_norm == 0.5
    assert cfg.save_interval_steps == 100
    assert cfg.save_dir == "checkpoints/custom"
    assert cfg.log_interval_steps == 5
    assert cfg.train_data_dir == "data/train"
    assert cfg.val_data_dir == "data/val"
    assert cfg.seed == 123
    assert cfg.gradient_checkpointing is False
    assert cfg.num_workers == 8
    assert cfg.weight_decay == 0.01
    assert cfg.beta1 == 0.95
    assert cfg.beta2 == 0.999
    assert cfg.eps == 1e-6


# ---------------------------------------------------------------------------
# Graceful skip when mlx is not installed
# ---------------------------------------------------------------------------


def test_mlx_trainer_skips_when_mlx_unavailable():
    """Demonstrate graceful skip when mlx is not installed."""
    mlx = pytest.importorskip("mlx", reason="MLX not installed")
    # This line is only reached when mlx is present.
    assert mlx is not None


# ---------------------------------------------------------------------------
# MLXTrainer initialization (with mocked mlx modules)
# ---------------------------------------------------------------------------


def test_mlx_trainer_init_with_mocked_mlx(mock_mlx):
    """MLXTrainer must initialize and store mocked mlx deps."""
    cfg = MLXConfig(model_path="test_model")
    trainer = MLXTrainer(cfg)
    assert trainer.cfg == cfg
    assert trainer.mx == mock_mlx["mx"]
    assert trainer.mlx_nn == mock_mlx["nn"]
    assert trainer.mlx_optim == mock_mlx["optim"]


def test_mlx_trainer_init_no_mlx_raises():
    """MLXTrainer must raise ImportError when mlx is absent."""
    with patch.dict(
        sys.modules, {"mlx": None, "mlx.core": None, "mlx.nn": None, "mlx.optim": None}
    ):
        with pytest.raises(ImportError, match="MLX is required"):
            MLXTrainer()


def test_mlx_trainer_init_default_config(mock_mlx):
    """MLXTrainer must use MLXConfig defaults when no config is passed."""
    trainer = MLXTrainer()
    assert isinstance(trainer.cfg, MLXConfig)
    assert trainer.cfg.model_path == ""


def test_mlx_trainer_check_deps_logs_device(mock_mlx, caplog):
    """_check_deps must log the default device."""
    with caplog.at_level(logging.INFO, logger="src.training.mlx_trainer"):
        MLXTrainer()
    assert "MLX available on" in caplog.text


# ---------------------------------------------------------------------------
# Training step logic (with mocked mlx modules)
# ---------------------------------------------------------------------------


def test_mlx_trainer_train_runs_with_mocked_mlx(mock_mlx, caplog):
    """train() must run without error and log config when mlx is mocked."""
    cfg = MLXConfig(total_steps=100, learning_rate=1e-3, batch_size=2, seq_len=512)
    trainer = MLXTrainer(cfg)
    with caplog.at_level(logging.INFO, logger="src.training.mlx_trainer"):
        trainer.train()
    assert "MLX Trainer initialized (experimental)" in caplog.text
    assert "Config: steps=100, lr=1.00e-03, batch_size=2, seq_len=512" in caplog.text


# ---------------------------------------------------------------------------
# Learning rate scheduling (with mocked mlx modules)
# ---------------------------------------------------------------------------


def test_mlx_trainer_lr_schedule_fields_present(mock_mlx):
    """Learning rate scheduling parameters must be accessible on config."""
    cfg = MLXConfig(
        learning_rate=5e-4,
        min_lr=5e-5,
        warmup_steps=500,
        total_steps=50000,
        weight_decay=0.05,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    trainer = MLXTrainer(cfg)
    assert trainer.cfg.learning_rate == 5e-4
    assert trainer.cfg.min_lr == 5e-5
    assert trainer.cfg.warmup_steps == 500
    assert trainer.cfg.total_steps == 50000
    assert trainer.cfg.weight_decay == 0.05
    assert trainer.cfg.beta1 == 0.9
    assert trainer.cfg.beta2 == 0.95
    assert trainer.cfg.eps == 1e-8


# ---------------------------------------------------------------------------
# Checkpoint saving/loading (with mocked mlx modules)
# ---------------------------------------------------------------------------


def test_mlx_trainer_build_model_with_checkpoint(mock_mlx, mock_safetensors, tmp_path):
    """build_model must load tensors from model.safetensors when present."""
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").touch()

    mock_file = MagicMock()
    mock_file.keys.return_value = ["layer1.weight", "layer2.bias"]
    tensor_mock = MagicMock()
    tensor_mock.numpy.return_value = np.array([1.0, 2.0])
    mock_file.get_tensor.return_value = tensor_mock

    mock_safetensors["safe_open"].return_value.__enter__.return_value = mock_file

    cfg = MLXConfig(model_path=str(ckpt_dir))
    trainer = MLXTrainer(cfg)
    result = trainer.build_model()

    assert isinstance(result, dict)
    assert len(result) == 2
    assert "layer1.weight" in result
    assert "layer2.bias" in result
    np.testing.assert_array_equal(result["layer1.weight"], np.array([1.0, 2.0]))


def test_mlx_trainer_build_model_missing_dir(mock_mlx):
    """build_model must return None when model_path does not exist."""
    cfg = MLXConfig(model_path="/nonexistent/path")
    trainer = MLXTrainer(cfg)
    assert trainer.build_model() is None


def test_mlx_trainer_build_model_missing_safetensors(mock_mlx, tmp_path):
    """build_model must return None when model_path exists but model.safetensors is missing."""
    ckpt_dir = tmp_path / "empty_ckpt"
    ckpt_dir.mkdir()
    cfg = MLXConfig(model_path=str(ckpt_dir))
    trainer = MLXTrainer(cfg)
    assert trainer.build_model() is None


def test_mlx_trainer_build_model_load_failure(mock_mlx, mock_safetensors, tmp_path, caplog):
    """build_model must log a warning and return None when safetensors loading fails."""
    ckpt_dir = tmp_path / "bad_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").touch()

    mock_safetensors["safe_open"].side_effect = RuntimeError("corrupted file")

    cfg = MLXConfig(model_path=str(ckpt_dir))
    trainer = MLXTrainer(cfg)
    with caplog.at_level(logging.WARNING, logger="src.training.mlx_trainer"):
        result = trainer.build_model()

    assert result is None
    assert "Failed to load checkpoint" in caplog.text


def test_mlx_trainer_convert_from_pytorch(mock_mlx, mock_safetensors, tmp_path):
    """convert_from_pytorch must read PyTorch safetensors and write an npz file."""
    pt_dir = tmp_path / "pt_ckpt"
    pt_dir.mkdir()
    (pt_dir / "model.safetensors").touch()

    out_dir = tmp_path / "mlx_ckpt"

    mock_file = MagicMock()
    mock_file.keys.return_value = ["w1", "w2"]
    tensor_mock = MagicMock()
    tensor_mock.float.return_value.numpy.return_value = np.array([3.0, 4.0])
    mock_file.get_tensor.return_value = tensor_mock

    mock_safetensors["safe_open"].return_value.__enter__.return_value = mock_file

    trainer = MLXTrainer()
    trainer.convert_from_pytorch(str(pt_dir), str(out_dir))

    assert (out_dir / "weights.npz").exists()
    data = np.load(out_dir / "weights.npz")
    assert "w1" in data
    assert "w2" in data
    np.testing.assert_array_equal(data["w1"], np.array([3.0, 4.0]))


def test_mlx_trainer_convert_from_pytorch_missing_file(mock_mlx, tmp_path):
    """convert_from_pytorch must raise FileNotFoundError when model.safetensors is missing."""
    pt_dir = tmp_path / "pt_ckpt"
    pt_dir.mkdir()
    out_dir = tmp_path / "mlx_ckpt"

    trainer = MLXTrainer()
    with pytest.raises(FileNotFoundError, match="No model.safetensors found"):
        trainer.convert_from_pytorch(str(pt_dir), str(out_dir))


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def test_mlx_trainer_estimate_memory_savings(mock_mlx):
    """estimate_memory_savings must return correct memory estimates."""
    trainer = MLXTrainer()
    result = trainer.estimate_memory_savings(n_params_billions=2.0)

    n_params = 2.0e9
    expected_model_bf16 = (n_params * 2) / 1e9
    expected_model_fp32 = (n_params * 4) / 1e9
    expected_adamw = (n_params * 4 * 2) / 1e9
    expected_muon = (n_params * 4) / 1e9
    expected_peak_mps = (n_params * 2 * 3) / 1e9
    expected_peak_mlx = (n_params * 2 * 2.2) / 1e9

    assert result["model_bf16_gb"] == pytest.approx(expected_model_bf16)
    assert result["model_fp32_gb"] == pytest.approx(expected_model_fp32)
    assert result["optimizer_adamw_gb"] == pytest.approx(expected_adamw)
    assert result["optimizer_muon_gb"] == pytest.approx(expected_muon)
    assert result["peak_memory_mps_gb"] == pytest.approx(expected_peak_mps)
    assert result["peak_memory_mlx_estimate_gb"] == pytest.approx(expected_peak_mlx)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_mlx_trainer_main_convert(mock_mlx, mock_safetensors, tmp_path, monkeypatch):
    """main with --convert must call convert_from_pytorch."""
    pt_dir = tmp_path / "pt_ckpt"
    pt_dir.mkdir()
    (pt_dir / "model.safetensors").touch()

    mock_file = MagicMock()
    mock_file.keys.return_value = ["w"]
    tensor_mock = MagicMock()
    tensor_mock.float.return_value.numpy.return_value = np.array([1.0])
    mock_file.get_tensor.return_value = tensor_mock
    mock_safetensors["safe_open"].return_value.__enter__.return_value = mock_file

    monkeypatch.setattr(
        sys, "argv", ["mlx_trainer", "--convert", str(pt_dir), "--output", str(tmp_path / "out")]
    )

    main()

    assert (tmp_path / "out" / "weights.npz").exists()


def test_mlx_trainer_main_train(mock_mlx, monkeypatch, caplog):
    """main without --convert must instantiate trainer and call train."""
    monkeypatch.setattr(sys, "argv", ["mlx_trainer", "--model-path", "/tmp/model"])

    with caplog.at_level(logging.INFO, logger="src.training.mlx_trainer"):
        main()

    assert "MLX Trainer initialized (experimental)" in caplog.text
