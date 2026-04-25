from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.deployment.model_packager import MODEL_PACKAGER_REGISTRY, ModelPackager


@pytest.fixture()
def src_file(tmp_path: Path) -> Path:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"fake model weights data")
    return f


@pytest.fixture()
def packager(tmp_path: Path) -> ModelPackager:
    out = tmp_path / "pkg"
    return ModelPackager(out)


def test_add_file_returns_name(packager, src_file):
    name = packager.add_file(src_file)
    assert name == "weights.bin"


def test_add_file_sha256_correct(packager, src_file):
    packager.add_file(src_file)
    expected = hashlib.sha256(b"fake model weights data").hexdigest()
    assert packager._sha256["weights.bin"] == expected


def test_add_file_custom_dest(packager, src_file):
    name = packager.add_file(src_file, dest_name="model.bin")
    assert name == "model.bin"
    assert (packager.output_dir / "model.bin").exists()


def test_build_manifest(packager, src_file):
    packager.add_file(src_file)
    manifest = packager.build_manifest("aurelius", "1.0.0", extra_meta={"tag": "v1"})
    assert manifest.model_name == "aurelius"
    assert manifest.version == "1.0.0"
    assert "weights.bin" in manifest.files
    assert "weights.bin" in manifest.sha256
    assert manifest.metadata["tag"] == "v1"


def test_write_manifest_creates_json(packager, src_file):
    packager.add_file(src_file)
    manifest = packager.build_manifest("aurelius", "1.0.0")
    path = packager.write_manifest(manifest)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["model_name"] == "aurelius"
    assert data["version"] == "1.0.0"
    assert "weights.bin" in data["files"]


def test_verify_manifest_passes(packager, src_file):
    packager.add_file(src_file)
    manifest = packager.build_manifest("aurelius", "1.0.0")
    failures = packager.verify_manifest(manifest)
    assert failures == []


def test_verify_manifest_detects_corruption(packager, src_file):
    packager.add_file(src_file)
    manifest = packager.build_manifest("aurelius", "1.0.0")
    (packager.output_dir / "weights.bin").write_bytes(b"corrupted!")
    failures = packager.verify_manifest(manifest)
    assert "weights.bin" in failures


def test_verify_manifest_detects_missing(packager, src_file):
    packager.add_file(src_file)
    manifest = packager.build_manifest("aurelius", "1.0.0")
    (packager.output_dir / "weights.bin").unlink()
    failures = packager.verify_manifest(manifest)
    assert "weights.bin" in failures


def test_registry_key():
    assert "default" in MODEL_PACKAGER_REGISTRY
    assert MODEL_PACKAGER_REGISTRY["default"] is ModelPackager
