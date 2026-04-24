import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CheckpointData:
    workflow_id: str
    step_name: str
    step_index: int
    state: dict
    timestamp_s: float = field(default_factory=time.monotonic)
    metadata: dict = field(default_factory=dict)


class CheckpointManager:
    def __init__(self, storage_dir: str = ".workflow_checkpoints") -> None:
        self._root = Path(storage_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _workflow_dir(self, workflow_id: str) -> Path:
        return self._root / workflow_id

    def _filename(self, step_name: str, step_index: int) -> str:
        return f"{step_index:06d}_{step_name}.json"

    def save(self, data: CheckpointData) -> str:
        wdir = self._workflow_dir(data.workflow_id)
        wdir.mkdir(parents=True, exist_ok=True)
        path = wdir / self._filename(data.step_name, data.step_index)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(data), f)
        return str(path)

    def _load_file(self, path: Path) -> CheckpointData:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return CheckpointData(
            workflow_id=raw["workflow_id"],
            step_name=raw["step_name"],
            step_index=int(raw["step_index"]),
            state=raw.get("state", {}),
            timestamp_s=float(raw.get("timestamp_s", 0.0)),
            metadata=raw.get("metadata", {}),
        )

    def load(self, workflow_id: str, step_index: int | None = None) -> CheckpointData | None:
        wdir = self._workflow_dir(workflow_id)
        if not wdir.exists():
            return None
        checkpoints = self.list_checkpoints(workflow_id)
        if not checkpoints:
            return None
        if step_index is None:
            return checkpoints[-1]
        for cp in checkpoints:
            if cp.step_index == step_index:
                return cp
        return None

    def list_checkpoints(self, workflow_id: str) -> list[CheckpointData]:
        wdir = self._workflow_dir(workflow_id)
        if not wdir.exists():
            return []
        out: list[CheckpointData] = []
        for path in sorted(wdir.glob("*.json")):
            try:
                out.append(self._load_file(path))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        out.sort(key=lambda c: c.step_index)
        return out

    def delete(self, workflow_id: str, step_index: int | None = None) -> None:
        wdir = self._workflow_dir(workflow_id)
        if not wdir.exists():
            return
        if step_index is None:
            for path in wdir.glob("*.json"):
                path.unlink(missing_ok=True)
            try:
                wdir.rmdir()
            except OSError:
                pass
            return
        for path in wdir.glob("*.json"):
            try:
                cp = self._load_file(path)
            except (json.JSONDecodeError, KeyError, OSError):
                continue
            if cp.step_index == step_index:
                path.unlink(missing_ok=True)

    def latest_step(self, workflow_id: str) -> int:
        cps = self.list_checkpoints(workflow_id)
        if not cps:
            return -1
        return cps[-1].step_index


CHECKPOINT_MANAGER_REGISTRY: dict[str, type[CheckpointManager]] = {"default": CheckpointManager}
