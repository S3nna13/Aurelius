"""File integrity verifier — detect unauthorized modifications.

Trail of Bits: validate file integrity before trusting inputs.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field


@dataclass
class IntegrityRecord:
    path: str
    sha256: str
    size_bytes: int
    mtime: float = 0.0


@dataclass
class IntegrityVerifier:
    """Verify file integrity against recorded SHA-256 hashes."""

    _records: dict[str, IntegrityRecord] = field(default_factory=dict, repr=False)

    def snapshot(self, path: str) -> IntegrityRecord:
        with open(path, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        stat = os.stat(path)
        record = IntegrityRecord(
            path=path,
            sha256=sha,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
        )
        self._records[path] = record
        return record

    def verify(self, path: str) -> tuple[bool, str]:
        record = self._records.get(path)
        if record is None:
            return False, "no snapshot recorded"
        if not os.path.exists(path):
            return False, "file missing"
        with open(path, "rb") as f:
            current_sha = hashlib.sha256(f.read()).hexdigest()
        if current_sha != record.sha256:
            return False, f"SHA-256 mismatch (expected {record.sha256[:16]}..., got {current_sha[:16]}...)"
        return True, "ok"

    def verify_all(self) -> dict[str, tuple[bool, str]]:
        return {path: self.verify(path) for path in self._records}


INTEGRITY_VERIFIER = IntegrityVerifier()