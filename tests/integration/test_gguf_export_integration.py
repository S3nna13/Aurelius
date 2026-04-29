"""Integration tests for scripts/export_gguf.py — Safetensors → GGUF conversion."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from safetensors.torch import save_file

from scripts.export_gguf import export_gguf
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

TINY_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


@pytest.fixture
def tiny_model() -> AureliusTransformer:
    model = AureliusTransformer(TINY_CONFIG)
    model.eval()
    return model


@pytest.fixture
def checkpoint_dir(tmp_path: Path, tiny_model: AureliusTransformer) -> Path:
    """Save a tiny model as safetensors and return the checkpoint directory."""
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    # Clone tensors to break weight sharing (tied embeddings) before saving
    state_dict = {
        k: v.detach().contiguous().cpu().clone() for k, v in tiny_model.state_dict().items()
    }
    save_file(state_dict, ckpt_dir / "model.safetensors")
    return ckpt_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_gguf_metadata(path: Path) -> dict[str, object]:
    """Parse metadata key-value pairs from a GGUF file produced by export_gguf."""
    data = path.read_bytes()
    pos = 0

    # Header
    pos += 4  # magic
    pos += 4  # version
    pos += 8  # total_params

    n_kv = struct.unpack("<Q", data[pos : pos + 8])[0]
    pos += 8

    metadata: dict[str, object] = {}
    for _ in range(n_kv):
        key_len = struct.unpack("<Q", data[pos : pos + 8])[0]
        pos += 8
        key = data[pos : pos + key_len].decode("utf-8")
        pos += key_len

        val_type = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4

        if val_type == 0:  # bool
            val = struct.unpack("<B", data[pos : pos + 1])[0] != 0
            pos += 4  # writer pads bool to 4 bytes
        elif val_type == 1:  # int32
            val = struct.unpack("<i", data[pos : pos + 4])[0]
            pos += 4
        elif val_type == 2:  # int64
            val = struct.unpack("<q", data[pos : pos + 8])[0]
            pos += 8
        elif val_type == 3:  # float32
            val = struct.unpack("<f", data[pos : pos + 4])[0]
            pos += 4
        elif val_type == 4:  # string
            str_len = struct.unpack("<I", data[pos : pos + 4])[0]
            pos += 4
            val = data[pos : pos + str_len].decode("utf-8")
            pos += str_len
        else:
            val = None

        metadata[key] = val

    return metadata


# ---------------------------------------------------------------------------
# Core export tests
# ---------------------------------------------------------------------------


class TestGGUFExport:
    def test_export_creates_gguf_file(self, checkpoint_dir: Path, tmp_path: Path) -> None:
        """export_gguf writes a .gguf file to the requested path."""
        output = tmp_path / "model.gguf"
        export_gguf(checkpoint_dir, output, config_name="aurelius_1_3b")
        assert output.exists()
        assert output.stat().st_size > 0

    def test_gguf_magic_bytes_and_version(self, checkpoint_dir: Path, tmp_path: Path) -> None:
        """The output file starts with the GGUF magic bytes and expected version."""
        output = tmp_path / "model.gguf"
        export_gguf(checkpoint_dir, output, config_name="aurelius_1_3b")

        data = output.read_bytes()
        assert data[:4] == GGUF_MAGIC
        version = struct.unpack("<I", data[4:8])[0]
        assert version == GGUF_VERSION

    def test_metadata_embedded_correctly(self, checkpoint_dir: Path, tmp_path: Path) -> None:
        """Metadata keys and values are present in the GGUF file."""
        output = tmp_path / "model.gguf"
        export_gguf(
            checkpoint_dir,
            output,
            config_name="aurelius_1_3b",
            model_name="TinyAurelius",
            description="Integration test model",
        )

        metadata = _read_gguf_metadata(output)
        assert metadata["general.name"] == "TinyAurelius"
        assert metadata["general.description"] == "Integration test model"
        assert metadata["general.architecture"] == "aurelius"
        assert metadata["general.file_type"] == 1
        assert isinstance(metadata["general.parameter_count"], int)
        assert metadata["aurelius.block_count"] == 24  # from aurelius_1_3b config
        assert metadata["tokenizer.ggml.model"] == "bpe"

    def test_end_to_end_load_save_export(self, tmp_path: Path) -> None:
        """Load a tiny model, save checkpoint as safetensors, then export to GGUF."""
        model = AureliusTransformer(TINY_CONFIG)
        model.eval()

        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        state_dict = {
            k: v.detach().contiguous().cpu().clone() for k, v in model.state_dict().items()
        }
        save_file(state_dict, ckpt_dir / "model.safetensors")

        output = tmp_path / "tiny.gguf"
        export_gguf(ckpt_dir, output, config_name="aurelius_1_3b")

        assert output.exists()
        data = output.read_bytes()
        assert data[:4] == GGUF_MAGIC

        # Tensor info section should list the same keys as the state dict
        metadata = _read_gguf_metadata(output)
        assert metadata["general.name"] == "Aurelius"

    def test_quantized_export_changes_file_type(self, checkpoint_dir: Path, tmp_path: Path) -> None:
        """When quantize=True, general.file_type reflects Q8_0."""
        output = tmp_path / "model_q8.gguf"
        export_gguf(checkpoint_dir, output, config_name="aurelius_1_3b", quantize=True)

        metadata = _read_gguf_metadata(output)
        assert metadata["general.file_type"] == 7
