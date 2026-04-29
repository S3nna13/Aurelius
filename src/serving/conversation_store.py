"""Persist and load conversation history to/from JSON files."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

try:
    from filelock import FileLock

    _HAS_FILELOCK = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_FILELOCK = False

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")
_INDEX_FILE = "index.json"


class ConversationStore:
    """Store and retrieve conversation histories as JSON files on disk.

    Writes are atomic (write-to-temp-then-rename) and optionally file-locked
    when the ``filelock`` package is available.
    """

    def __init__(self, storage_dir: str = "~/.aurelius/conversations") -> None:
        self.storage_dir = Path(os.path.expanduser(storage_dir)).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _conversation_digest(self, conversation_id: str) -> str:
        if not _SAFE_ID.fullmatch(conversation_id):
            raise ValueError(f"Invalid conversation_id: {conversation_id!r}")
        return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()

    def _index_path(self) -> Path:
        return self.storage_dir / _INDEX_FILE

    def _load_index(self) -> dict[str, str]:
        path = self._index_path()
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        index: dict[str, str] = {}
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, str):
                index[key] = value
        return index

    def _save_index(self, index: dict[str, str]) -> None:
        payload = json.dumps(index, ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(
            dir=self.storage_dir,
            prefix=".tmp-index-",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self._index_path())
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def _path(self, conversation_id: str) -> Path:
        digest = self._conversation_digest(conversation_id)
        return self.storage_dir / f"{digest}.json"

    def save(self, conversation_id: str, messages: list[dict]) -> None:
        """Atomically persist *messages* to disk."""
        path = self._path(conversation_id)
        payload = json.dumps(messages, ensure_ascii=False, indent=2)

        def _write() -> None:
            # Atomic write: write to temp file in the same directory, then rename
            fd, tmp = tempfile.mkstemp(
                dir=self.storage_dir,
                prefix=f".tmp-{conversation_id}-",
                suffix=".json",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp, path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
                raise

        if _HAS_FILELOCK:
            lock_path = path.with_suffix(".json.lock")
            with FileLock(str(lock_path), timeout=10):
                _write()
        else:
            _write()

        index = self._load_index()
        index[conversation_id] = path.name
        self._save_index(index)

    def load(self, conversation_id: str) -> list[dict]:
        path = self._path(conversation_id)
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "messages" in data:
            messages = data.get("messages", [])
            return messages if isinstance(messages, list) else []
        if isinstance(data, list):
            return data
        return []

    def list_conversations(self) -> list[str]:
        index = self._load_index()
        if index:
            return sorted(index.keys())
        return [
            p.stem
            for p in self.storage_dir.iterdir()
            if p.suffix == ".json" and p.name != _INDEX_FILE
        ]

    def delete(self, conversation_id: str) -> bool:
        path = self._path(conversation_id)
        if path.exists():
            path.unlink()
            index = self._load_index()
            if conversation_id in index:
                del index[conversation_id]
                self._save_index(index)
            return True
        return False

    def exists(self, conversation_id: str) -> bool:
        return self._path(conversation_id).exists()
