"""Tests for optimizer state offloading to SSD."""

from __future__ import annotations

import tempfile

import torch  # noqa: F401
import torch._dynamo  # noqa: F401
import torch._inductor.test_operators  # noqa: F401
from pathlib import Path

import pytest
import torch.nn as nn

from src.training.optimizer_offload import OptimizerOffloader
from src.training.muon import Muon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adamw_with_state() -> tuple[torch.optim.AdamW, nn.Parameter]:
    """Create an AdamW optimizer that has stepped at least once so states exist."""
    param = nn.Parameter(torch.randn(8, 8))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    loss = param.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return optimizer, param


def _make_muon_with_state() -> tuple[Muon, nn.Parameter]:
    """Create a Muon optimizer that has stepped at least once so states exist."""
    param = nn.Parameter(torch.randn(8, 8))
    optimizer = Muon([param], lr=0.02, momentum=0.95)
    loss = param.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return optimizer, param


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_init_creates_offload_directory():
    """OptimizerOffloader creates the offload directory on init."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offload_dir = Path(tmpdir) / "offload"
        assert not offload_dir.exists()
        offloader = OptimizerOffloader(optimizer, offload_dir=str(offload_dir))
        assert offload_dir.exists()
        assert offload_dir.is_dir()
        offloader.cleanup()


def test_init_uses_tempdir_when_none():
    """When offload_dir is None, a temporary directory is created."""
    optimizer, _ = _make_adamw_with_state()
    offloader = OptimizerOffloader(optimizer)
    assert offloader.offload_dir.exists()
    assert offloader.offload_dir.is_dir()
    offloader.cleanup()


# ---------------------------------------------------------------------------
# offload_to_disk
# ---------------------------------------------------------------------------


def test_offload_saves_tensors_to_disk():
    """offload_to_disk writes optimizer state tensors to the offload directory."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)
        offloader.offload_to_disk()

        files = list(Path(tmpdir).glob("*.pt"))
        assert len(files) > 0, "No state files were written to disk"
        offloader.cleanup()


def test_offload_removes_tensors_from_ram():
    """After offload, state tensors in RAM are replaced with tiny placeholders."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)

        # Record original state shapes
        original_shapes = {}
        for param_id, state in optimizer.state.items():
            for key, tensor in state.items():
                if isinstance(tensor, torch.Tensor):
                    original_shapes[(param_id, key)] = tensor.shape

        offloader.offload_to_disk()

        for param_id, state in optimizer.state.items():
            for key, tensor in state.items():
                if isinstance(tensor, torch.Tensor) and (param_id, key) in original_shapes:
                    orig = original_shapes[(param_id, key)]
                    if orig != (1,):
                        assert tensor.numel() == 1, (
                            f"State tensor ({param_id}, {key}) was not replaced with placeholder"
                        )
        offloader.cleanup()


def test_offload_records_offloaded_keys():
    """_offloaded tracks which (param_id, key) pairs were moved to disk."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)
        offloader.offload_to_disk()
        assert len(offloader._offloaded) > 0
        offloader.cleanup()


# ---------------------------------------------------------------------------
# load_from_disk
# ---------------------------------------------------------------------------


def test_load_restores_tensors_to_ram():
    """load_from_disk restores original state tensors from disk."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)

        # Record original state values
        original_values = {}
        for param_id, state in optimizer.state.items():
            for key, tensor in state.items():
                if isinstance(tensor, torch.Tensor):
                    original_values[(param_id, key)] = tensor.clone()

        offloader.offload_to_disk()
        offloader.load_from_disk()

        for param_id, state in optimizer.state.items():
            for key, tensor in state.items():
                if isinstance(tensor, torch.Tensor) and (param_id, key) in original_values:
                    assert torch.allclose(tensor, original_values[(param_id, key)]), (
                        f"State tensor ({param_id}, {key}) was not restored correctly"
                    )
        offloader.cleanup()


def test_load_clears_offloaded_list():
    """After load_from_disk, the _offloaded list is cleared."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)
        offloader.offload_to_disk()
        assert len(offloader._offloaded) > 0
        offloader.load_from_disk()
        assert len(offloader._offloaded) == 0
        offloader.cleanup()


# ---------------------------------------------------------------------------
# Round-trip with AdamW
# ---------------------------------------------------------------------------


def test_adamw_round_trip_allows_step():
    """After offload + load, AdamW can take another optimizer step."""
    param = nn.Parameter(torch.randn(8, 8))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)

        # First step to initialize states
        loss = param.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        initial = param.data.clone()

        # Offload, load, and step again
        offloader.offload_to_disk()
        offloader.load_from_disk()

        loss2 = param.sum()
        loss2.backward()
        optimizer.step()

        assert not torch.allclose(param.data, initial), "AdamW did not update after round-trip"
        offloader.cleanup()


# ---------------------------------------------------------------------------
# Round-trip with Muon
# ---------------------------------------------------------------------------


def test_muon_round_trip_allows_step():
    """After offload + load, Muon can take another optimizer step."""
    param = nn.Parameter(torch.randn(8, 8))
    optimizer = Muon([param], lr=0.02, momentum=0.95)
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)

        # First step to initialize states
        loss = param.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        initial = param.data.clone()

        # Offload, load, and step again
        offloader.offload_to_disk()
        offloader.load_from_disk()

        loss2 = param.sum()
        loss2.backward()
        optimizer.step()

        assert not torch.allclose(param.data, initial), "Muon did not update after round-trip"
        offloader.cleanup()


# ---------------------------------------------------------------------------
# bytes_on_disk
# ---------------------------------------------------------------------------


def test_bytes_on_disk_increases_after_offload():
    """bytes_on_disk returns a positive value after offloading."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)
        assert offloader.bytes_on_disk() == 0
        offloader.offload_to_disk()
        assert offloader.bytes_on_disk() > 0
        offloader.cleanup()


def test_bytes_on_disk_decreases_after_cleanup():
    """After cleanup, bytes_on_disk returns 0."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)
        offloader.offload_to_disk()
        assert offloader.bytes_on_disk() > 0
        offloader.cleanup()
        assert offloader.bytes_on_disk() == 0


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


def test_cleanup_removes_files():
    """cleanup removes all offloaded files from disk."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)
        offloader.offload_to_disk()
        assert len(list(Path(tmpdir).glob("*.pt"))) > 0
        offloader.cleanup()
        assert len(list(Path(tmpdir).glob("*.pt"))) == 0


def test_cleanup_idempotent():
    """cleanup can be called multiple times without error."""
    optimizer, _ = _make_adamw_with_state()
    with tempfile.TemporaryDirectory() as tmpdir:
        offloader = OptimizerOffloader(optimizer, offload_dir=tmpdir)
        offloader.cleanup()
        offloader.cleanup()  # should not raise
