"""Checkpoint save/load utilities for Aurelius training.

Saves model weights + optimizer state + training metadata.
Tracks best checkpoint (lowest validation loss) automatically.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMeta:
    """Metadata saved alongside each checkpoint."""

    step: int
    epoch: int
    train_loss: float
    val_loss: float | None
    config: dict  # model config as dict


def _save_tensor_state_dict(model: nn.Module, path: Path) -> None:
    """Persist model weights with safetensors instead of executable pickle data."""
    from safetensors.torch import save_model

    save_model(model, str(path))


def _serialize_state(value: object) -> object:
    """Convert optimizer state to a JSON-serializable structure."""
    if isinstance(value, torch.Tensor):
        return {
            "__type__": "tensor",
            "dtype": str(value.dtype).split(".")[-1],
            "data": value.detach().cpu().tolist(),
        }
    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "items": [
                [_serialize_state(key), _serialize_state(item)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, list):
        return {"__type__": "list", "items": [_serialize_state(item) for item in value]}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_serialize_state(item) for item in value]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported checkpoint value type: {type(value)!r}")


def _deserialize_state(value: object, *, device: str | torch.device = "cpu") -> object:
    """Reconstruct a checkpoint state structure from JSON."""
    if not isinstance(value, dict) or "__type__" not in value:
        return value

    kind = value.get("__type__")
    if kind == "tensor":
        dtype_name = str(value.get("dtype", "float32"))
        dtype = getattr(torch, dtype_name, torch.float32)
        return torch.tensor(value.get("data", []), dtype=dtype, device=device)
    if kind == "dict":
        items = value.get("items", [])
        return {
            _deserialize_state(key, device=device): _deserialize_state(item, device=device)
            for key, item in items
        }
    if kind == "list":
        return [_deserialize_state(item, device=device) for item in value.get("items", [])]
    if kind == "tuple":
        return tuple(_deserialize_state(item, device=device) for item in value.get("items", []))
    return value


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    epoch: int,
    train_loss: float,
    output_dir: str | Path,
    val_loss: float | None = None,
    keep_last_n: int = 3,
) -> Path:
    """Save model + optimizer state + metadata to a numbered checkpoint directory.

    Creates: output_dir/checkpoint-{step:07d}/
        model.safetensors — model state dict
        optimizer.json    — optimizer state dict (if optimizer provided)
        meta.json         — CheckpointMeta as JSON

    Also updates output_dir/best/ symlink if val_loss is the lowest seen.
    Deletes oldest checkpoints beyond keep_last_n.

    Args:
        model: The model to checkpoint.
        optimizer: Optional optimizer (not saved if None).
        step: Current training step.
        epoch: Current epoch.
        train_loss: Training loss at this step.
        output_dir: Root directory for all checkpoints.
        val_loss: Validation loss (used for best-checkpoint tracking).
        keep_last_n: Number of most recent checkpoints to keep. 0 = keep all.

    Returns:
        Path to the created checkpoint directory.
    """
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / f"checkpoint-{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    _save_tensor_state_dict(model, ckpt_dir / "model.safetensors")

    # Save optimizer state dict
    if optimizer is not None:
        with open(ckpt_dir / "optimizer.json", "w", encoding="utf-8") as f:
            json.dump(_serialize_state(optimizer.state_dict()), f, indent=2)

    # Save metadata
    config = {}
    if hasattr(model, "config"):
        try:
            config = (
                asdict(model.config)
                if hasattr(model.config, "__dataclass_fields__")
                else vars(model.config)
            )
        except Exception:
            config = {}

    meta = CheckpointMeta(
        step=step,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        config=config,
    )
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2)

    logger.info("Saved checkpoint to %s (step=%d, loss=%.4f)", ckpt_dir, step, train_loss)

    # Update best checkpoint symlink
    if val_loss is not None:
        _maybe_update_best(output_dir, ckpt_dir, val_loss)

    # Clean up old checkpoints
    if keep_last_n > 0:
        _cleanup_old_checkpoints(output_dir, keep_last_n)

    return ckpt_dir


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> CheckpointMeta:
    """Load model (and optionally optimizer) state from a checkpoint directory.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint directory (contains model.safetensors, meta.json).
        optimizer: Optional optimizer to restore state into.
        strict: If True, model.load_state_dict is called with strict=True.
        map_location: Device to load tensors onto.

    Returns:
        CheckpointMeta loaded from meta.json.

    Raises:
        FileNotFoundError: If model.safetensors or meta.json not found.
    """
    ckpt_dir = Path(checkpoint_path)

    model_path = ckpt_dir / "model.safetensors"
    meta_path = ckpt_dir / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {ckpt_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {ckpt_dir}")

    # Load model weights
    from safetensors.torch import load_model

    load_model(model, model_path, strict=strict, device=map_location)
    logger.info("Loaded model weights from %s", model_path)

    # Load optimizer state if requested
    opt_path = ckpt_dir / "optimizer.json"
    if optimizer is not None and opt_path.exists():
        with open(opt_path, encoding="utf-8") as f:
            opt_state = json.load(f)
        optimizer.load_state_dict(_deserialize_state(opt_state, device=map_location))
        logger.info("Loaded optimizer state from %s", opt_path)

    # Load metadata
    with open(meta_path) as f:
        meta_dict = json.load(f)

    return CheckpointMeta(**meta_dict)


def load_best_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointMeta:
    """Load the best checkpoint (lowest val_loss) from output_dir/best/.

    Args:
        model: Model to load weights into.
        output_dir: Root checkpoint directory containing a 'best' symlink or subdir.
        optimizer: Optional optimizer to restore.
        map_location: Device to load tensors onto.

    Returns:
        CheckpointMeta of the best checkpoint.

    Raises:
        FileNotFoundError: If no best checkpoint exists.
    """
    best_path = Path(output_dir) / "best"
    if not best_path.exists():
        raise FileNotFoundError(f"No best checkpoint found at {best_path}")
    return load_checkpoint(model, best_path, optimizer, map_location=map_location)


def list_checkpoints(output_dir: str | Path) -> list[tuple[Path, CheckpointMeta]]:
    """List all checkpoints in output_dir, sorted by step ascending.

    Returns:
        List of (path, meta) tuples.
    """
    output_dir = Path(output_dir)
    checkpoints = []

    for ckpt_dir in sorted(output_dir.glob("checkpoint-*")):
        meta_path = ckpt_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta_dict = json.load(f)
            checkpoints.append((ckpt_dir, CheckpointMeta(**meta_dict)))

    return checkpoints


def _maybe_update_best(output_dir: Path, new_ckpt_dir: Path, val_loss: float) -> None:
    """Update output_dir/best to point to new_ckpt_dir if val_loss is lowest."""
    best_meta_path = output_dir / "best" / "meta.json"

    update = True
    if best_meta_path.exists():
        with open(best_meta_path) as f:
            best_meta = json.load(f)
        current_best_loss = best_meta.get("val_loss", float("inf"))
        if current_best_loss is not None and val_loss >= current_best_loss:
            update = False

    if update:
        best_link = output_dir / "best"
        if best_link.exists() or best_link.is_symlink():
            if best_link.is_symlink():
                best_link.unlink()
            elif best_link.is_dir():
                shutil.rmtree(best_link)
        shutil.copytree(new_ckpt_dir, best_link)
        logger.info("Updated best checkpoint (val_loss=%.4f)", val_loss)


def _cleanup_old_checkpoints(output_dir: Path, keep_last_n: int) -> None:
    """Delete all but the keep_last_n most recent checkpoint directories."""
    ckpt_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.name)
    to_delete = ckpt_dirs[:-keep_last_n] if len(ckpt_dirs) > keep_last_n else []
    for old_dir in to_delete:
        shutil.rmtree(old_dir)
        logger.debug("Deleted old checkpoint %s", old_dir)
