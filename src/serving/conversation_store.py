"""Persist and load conversation history to/from JSON files."""

import json
import os
from typing import Dict, List


class ConversationStore:
    """Store and retrieve conversation histories as JSON files on disk."""

    def __init__(self, storage_dir: str = "~/.aurelius/conversations") -> None:
        self.storage_dir = os.path.expanduser(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

    def _path(self, conversation_id: str) -> str:
        return os.path.join(self.storage_dir, f"{conversation_id}.json")

    def save(self, conversation_id: str, messages: List[Dict]) -> None:
        with open(self._path(conversation_id), "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    def load(self, conversation_id: str) -> List[Dict]:
        path = self._path(conversation_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_conversations(self) -> List[str]:
        names = []
        for entry in os.listdir(self.storage_dir):
            if entry.endswith(".json"):
                names.append(entry[:-5])
        return names

    def delete(self, conversation_id: str) -> bool:
        path = self._path(conversation_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def exists(self, conversation_id: str) -> bool:
        return os.path.exists(self._path(conversation_id))
