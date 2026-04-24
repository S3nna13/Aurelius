from pathlib import Path

from src.workflow.checkpoint_manager import (
    CheckpointData,
    CheckpointManager,
    CHECKPOINT_MANAGER_REGISTRY,
)


def _mgr(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(storage_dir=str(tmp_path / "cp"))


def _cp(wf: str, idx: int, name: str = "step", state: dict | None = None) -> CheckpointData:
    return CheckpointData(
        workflow_id=wf,
        step_name=name,
        step_index=idx,
        state=state or {"v": idx},
    )


def test_registry_has_default():
    assert "default" in CHECKPOINT_MANAGER_REGISTRY
    assert CHECKPOINT_MANAGER_REGISTRY["default"] is CheckpointManager


def test_creates_storage_dir(tmp_path):
    root = tmp_path / "new_root"
    assert not root.exists()
    CheckpointManager(storage_dir=str(root))
    assert root.exists() and root.is_dir()


def test_save_returns_path(tmp_path):
    m = _mgr(tmp_path)
    path = m.save(_cp("wf1", 0))
    assert Path(path).exists()


def test_save_filename_format(tmp_path):
    m = _mgr(tmp_path)
    path = m.save(_cp("wf1", 5, name="train"))
    assert Path(path).name == "000005_train.json"


def test_save_creates_workflow_dir(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf_abc", 0))
    assert (tmp_path / "cp" / "wf_abc").is_dir()


def test_load_round_trip(tmp_path):
    m = _mgr(tmp_path)
    original = _cp("wf1", 3, name="s", state={"x": 42})
    m.save(original)
    loaded = m.load("wf1", 3)
    assert loaded is not None
    assert loaded.workflow_id == "wf1"
    assert loaded.step_index == 3
    assert loaded.step_name == "s"
    assert loaded.state == {"x": 42}


def test_load_missing_workflow(tmp_path):
    m = _mgr(tmp_path)
    assert m.load("nope") is None


def test_load_missing_step(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0))
    assert m.load("wf1", 99) is None


def test_load_latest_when_none(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0))
    m.save(_cp("wf1", 1))
    m.save(_cp("wf1", 2))
    latest = m.load("wf1")
    assert latest is not None
    assert latest.step_index == 2


def test_load_latest_empty(tmp_path):
    m = _mgr(tmp_path)
    assert m.load("wf1") is None


def test_list_checkpoints_sorted(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 5))
    m.save(_cp("wf1", 1))
    m.save(_cp("wf1", 3))
    cps = m.list_checkpoints("wf1")
    assert [c.step_index for c in cps] == [1, 3, 5]


def test_list_checkpoints_empty(tmp_path):
    m = _mgr(tmp_path)
    assert m.list_checkpoints("nope") == []


def test_list_checkpoints_only_that_workflow(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0))
    m.save(_cp("wf2", 0))
    assert len(m.list_checkpoints("wf1")) == 1
    assert len(m.list_checkpoints("wf2")) == 1


def test_delete_specific_step(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0))
    m.save(_cp("wf1", 1))
    m.delete("wf1", 0)
    remaining = [c.step_index for c in m.list_checkpoints("wf1")]
    assert remaining == [1]


def test_delete_all_for_workflow(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0))
    m.save(_cp("wf1", 1))
    m.delete("wf1")
    assert m.list_checkpoints("wf1") == []


def test_delete_unknown_workflow_no_error(tmp_path):
    m = _mgr(tmp_path)
    m.delete("nope")
    m.delete("nope", 5)


def test_delete_nonexistent_step(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0))
    m.delete("wf1", 99)
    assert len(m.list_checkpoints("wf1")) == 1


def test_latest_step_unknown_returns_neg1(tmp_path):
    m = _mgr(tmp_path)
    assert m.latest_step("nope") == -1


def test_latest_step(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0))
    m.save(_cp("wf1", 7))
    m.save(_cp("wf1", 3))
    assert m.latest_step("wf1") == 7


def test_latest_step_empty_dir_returns_neg1(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0))
    m.delete("wf1")
    assert m.latest_step("wf1") == -1


def test_save_overwrites_same_index(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0, state={"v": 1}))
    m.save(_cp("wf1", 0, state={"v": 2}))
    loaded = m.load("wf1", 0)
    assert loaded.state == {"v": 2}


def test_metadata_round_trip(tmp_path):
    m = _mgr(tmp_path)
    cp = CheckpointData(
        workflow_id="wf1",
        step_name="s",
        step_index=0,
        state={},
        metadata={"user": "alice"},
    )
    m.save(cp)
    loaded = m.load("wf1", 0)
    assert loaded.metadata == {"user": "alice"}


def test_timestamp_round_trip(tmp_path):
    m = _mgr(tmp_path)
    cp = _cp("wf1", 0)
    m.save(cp)
    loaded = m.load("wf1", 0)
    assert isinstance(loaded.timestamp_s, float)


def test_multiple_workflows_isolated(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("a", 0))
    m.save(_cp("b", 0))
    m.delete("a")
    assert m.list_checkpoints("a") == []
    assert len(m.list_checkpoints("b")) == 1


def test_default_storage_dir_param(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = CheckpointManager()
    assert Path(".workflow_checkpoints").exists()
    m.save(_cp("wf1", 0))
    assert m.load("wf1", 0) is not None


def test_list_returns_checkpoint_data_objects(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 0))
    cps = m.list_checkpoints("wf1")
    assert all(isinstance(c, CheckpointData) for c in cps)


def test_large_step_index(tmp_path):
    m = _mgr(tmp_path)
    m.save(_cp("wf1", 999999))
    assert m.latest_step("wf1") == 999999
    loaded = m.load("wf1", 999999)
    assert loaded is not None


def test_state_complex_nested(tmp_path):
    m = _mgr(tmp_path)
    state = {"list": [1, 2, 3], "dict": {"k": "v"}, "n": 1.5}
    m.save(_cp("wf1", 0, state=state))
    loaded = m.load("wf1", 0)
    assert loaded.state == state
