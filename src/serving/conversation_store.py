"""Persist and load conversation history to/from JSON files."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


class ConversationStore:
    """Store and retrieve conversation histories as JSON files on disk."""

    def __init__(self, storage_dir: str = "~/.aurelius/conversations") -> None:
        self.storage_dir = Path(os.path.expanduser(storage_dir)).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        if not _SAFE_ID.match(conversation_id):
            raise ValueError(f"Invalid conversation_id: {conversation_id!r}")
        path = (self.storage_dir / f"{conversation_id}.json").resolve()
        if not str(path).startswith(str(self.storage_dir)):
            raise ValueError("conversation_id escapes storage directory")
        return path

    def save(self, conversation_id: str, messages: List[Dict]) -> None:
        with open(self._path(conversation_id), "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    def load(self, conversation_id: str) -> List[Dict]:
        path = self._path(conversation_id)
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def list_conversations(self) -> List[str]:
        return [p.stem for p in self.storage_dir.iterdir() if p.suffix == ".json"]

    def delete(self, conversation_id: str) -> bool:
        path = self._path(conversation_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, conversation_id: str) -> bool:
        return self._path(conversation_id).exists()
