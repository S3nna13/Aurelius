from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PackageManifest:
    model_name: str
    version: str
    files: list[str]
    sha256: dict[str, str]
    metadata: dict


class ModelPackager:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._files: list[str] = []
        self._sha256: dict[str, str] = {}

    def add_file(self, src: Path, dest_name: str | None = None) -> str:
        if dest_name is None:
            dest_name = src.name
        dest = self.output_dir / dest_name
        shutil.copy2(src, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        self._files.append(dest_name)
        self._sha256[dest_name] = digest
        return dest_name

    def build_manifest(
        self,
        model_name: str,
        version: str,
        extra_meta: dict | None = None,
    ) -> PackageManifest:
        metadata: dict = extra_meta.copy() if extra_meta else {}
        return PackageManifest(
            model_name=model_name,
            version=version,
            files=list(self._files),
            sha256=dict(self._sha256),
            metadata=metadata,
        )

    def write_manifest(self, manifest: PackageManifest) -> Path:
        path = self.output_dir / "manifest.json"
        data = {
            "model_name": manifest.model_name,
            "version": manifest.version,
            "files": manifest.files,
            "sha256": manifest.sha256,
            "metadata": manifest.metadata,
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    def verify_manifest(self, manifest: PackageManifest) -> list[str]:
        failures: list[str] = []
        for fname, expected in manifest.sha256.items():
            fpath = self.output_dir / fname
            if not fpath.exists():
                failures.append(fname)
                continue
            actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(fname)
        return failures


MODEL_PACKAGER_REGISTRY: dict[str, type[ModelPackager]] = {"default": ModelPackager}
