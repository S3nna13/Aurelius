"""Memory snapshot — persistent serialization of memory stores."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SnapshotCorruptedError(ValueError):
    """Raised when a snapshot fails integrity or schema validation."""


@dataclass(frozen=True)
class SnapshotHeader:
    version: str
    hash: str
    compressed: bool
    store_type: str


class MemorySnapshot:
    """Serialize and deserialize memory stores with integrity checks."""

    CURRENT_VERSION = "1.0"
    SUPPORTED_STORES = ("EpisodicMemory", "WorkingMemory", "SemanticMemory")

    @staticmethod
    def _sanitize_path(path: str) -> Path:
        p = Path(path).resolve()
        cwd = Path(os.getcwd()).resolve()
        if not str(p).startswith(str(cwd)) and ".." in str(Path(path)):
            raise ValueError("path traversal not allowed")
        return p

    @staticmethod
    def from_memory(memory: Any) -> dict[str, Any]:
        """Extract serializable payload from a memory store."""
        store_type = type(memory).__name__
        if store_type not in MemorySnapshot.SUPPORTED_STORES:
            raise SnapshotCorruptedError(f"unsupported store type: {store_type}")

        entries: list[dict[str, Any]] = []
        raw_entries: list[Any] = []
        if hasattr(memory, "_entries"):
            raw_entries = list(memory._entries)
        elif hasattr(memory, "entries"):
            raw_entries = list(memory.entries)
        # Stores like WorkingMemory (_slots) or SemanticMemory (_concepts/_relations)
        # may not have entries; that is valid.

        for entry in raw_entries:
            if hasattr(entry, "__dict__"):
                entries.append(entry.__dict__)
            elif isinstance(entry, dict):
                entries.append(entry)
            else:
                entries.append({"value": str(entry)})

        slots: list[dict[str, Any]] = []
        if hasattr(memory, "_slots"):
            slots = [s.__dict__ if hasattr(s, "__dict__") else dict(s) for s in memory._slots.values()]

        concepts: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        if hasattr(memory, "_concepts"):
            concepts = [c.__dict__ if hasattr(c, "__dict__") else dict(c) for c in memory._concepts.values()]
        if hasattr(memory, "_relations"):
            relations = [
                r.__dict__ if hasattr(r, "__dict__") else dict(r) for r in memory._relations
            ]

        return {
            "store_type": store_type,
            "entries": entries,
            "slots": slots,
            "concepts": concepts,
            "relations": relations,
        }

    @staticmethod
    def to_memory(payload: dict[str, Any]) -> Any:
        """Reconstruct a memory store from payload."""
        store_type = payload.get("store_type")
        if store_type == "EpisodicMemory":
            from src.memory.episodic_memory import EpisodicMemory, MemoryEntry

            inst = EpisodicMemory()
            for e in payload.get("entries", []):
                inst.store(e.get("role", ""), e.get("content", ""), e.get("importance", 1.0))
            return inst
        elif store_type == "WorkingMemory":
            from src.memory.working_memory import WorkingMemory

            inst = WorkingMemory()
            for s in payload.get("slots", []):
                inst.set(s.get("key", ""), s.get("value"), s.get("ttl_seconds", 60.0))
            return inst
        elif store_type == "SemanticMemory":
            from src.memory.semantic_memory import SemanticMemory

            inst = SemanticMemory()
            for c in payload.get("concepts", []):
                inst.add_concept(c.get("name", ""), c.get("attributes", {}))
            for r in payload.get("relations", []):
                inst.add_relation(r.get("source", ""), r.get("relation_type", ""), r.get("target", ""))
            return inst
        else:
            raise SnapshotCorruptedError(f"unsupported store type: {store_type}")

    @classmethod
    def save(cls, memory: Any, path: str, *, compressed: bool = True) -> None:
        """Save a memory snapshot to disk."""
        p = cls._sanitize_path(path)
        payload = cls.from_memory(memory)
        raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        envelope = {
            "schema_version": cls.CURRENT_VERSION,
            "compressed": compressed,
            "hash": digest,
            "payload": raw.decode("utf-8") if not compressed else None,
        }
        if compressed:
            envelope["payload_compressed"] = gzip.compress(raw).hex()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(envelope), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> Any:
        """Load and verify a memory snapshot from disk."""
        p = cls._sanitize_path(path)
        if not p.exists():
            raise SnapshotCorruptedError(f"snapshot file not found: {path}")
        try:
            envelope = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SnapshotCorruptedError("invalid JSON envelope") from exc

        version = envelope.get("schema_version")
        if version != cls.CURRENT_VERSION:
            raise SnapshotCorruptedError(f"unsupported schema version: {version}")

        compressed = envelope.get("compressed", False)
        stored_hash = envelope.get("hash", "")

        if compressed:
            hex_payload = envelope.get("payload_compressed", "")
            raw = gzip.decompress(bytes.fromhex(hex_payload))
        else:
            raw = envelope.get("payload", "").encode("utf-8")

        actual_hash = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(stored_hash, actual_hash):
            raise SnapshotCorruptedError("integrity hash mismatch")

        payload = json.loads(raw.decode("utf-8"))
        return cls.to_memory(payload)


# Avoid importing hmac at top to satisfy lazy-import style; used above via compare_digest
import hmac
