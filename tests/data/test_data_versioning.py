"""Tests for src/data/data_versioning.py"""

import time
import pytest

from src.data.data_versioning import (
    DATA_VERSION_REGISTRY,
    DataDiff,
    DatasetVersion,
    DataVersionRegistry,
)


# ---------------------------------------------------------------------------
# DatasetVersion
# ---------------------------------------------------------------------------

def test_dataset_version_auto_id():
    v = DatasetVersion(name="test", n_samples=100)
    assert v.version_id is not None
    assert len(v.version_id) == 8

def test_dataset_version_ids_are_unique():
    v1 = DatasetVersion(name="a", n_samples=10)
    v2 = DatasetVersion(name="b", n_samples=20)
    assert v1.version_id != v2.version_id

def test_dataset_version_created_at_is_iso():
    v = DatasetVersion(name="test", n_samples=100)
    assert "T" in v.created_at or "-" in v.created_at  # ISO8601 format

def test_dataset_version_default_parent_id_none():
    v = DatasetVersion(name="test", n_samples=100)
    assert v.parent_id is None

def test_dataset_version_default_description_empty():
    v = DatasetVersion(name="test", n_samples=100)
    assert v.description == ""

def test_dataset_version_default_metadata_empty():
    v = DatasetVersion(name="test", n_samples=100)
    assert v.metadata == {}

def test_dataset_version_stores_name():
    v = DatasetVersion(name="my_dataset", n_samples=50)
    assert v.name == "my_dataset"

def test_dataset_version_stores_n_samples():
    v = DatasetVersion(name="test", n_samples=999)
    assert v.n_samples == 999

def test_dataset_version_custom_parent_id():
    v = DatasetVersion(name="child", n_samples=10, parent_id="abc12345")
    assert v.parent_id == "abc12345"

def test_dataset_version_custom_description():
    v = DatasetVersion(name="test", n_samples=10, description="my desc")
    assert v.description == "my desc"

def test_dataset_version_custom_metadata():
    v = DatasetVersion(name="test", n_samples=10, metadata={"source": "web"})
    assert v.metadata["source"] == "web"


# ---------------------------------------------------------------------------
# DataVersionRegistry.create_version
# ---------------------------------------------------------------------------

def test_create_version_returns_dataset_version():
    reg = DataVersionRegistry()
    v = reg.create_version("ds1", 100)
    assert isinstance(v, DatasetVersion)

def test_create_version_stores_name():
    reg = DataVersionRegistry()
    v = reg.create_version("my_ds", 200)
    assert v.name == "my_ds"

def test_create_version_stores_n_samples():
    reg = DataVersionRegistry()
    v = reg.create_version("ds", 500)
    assert v.n_samples == 500

def test_create_version_with_parent():
    reg = DataVersionRegistry()
    v1 = reg.create_version("root", 100)
    v2 = reg.create_version("child", 150, parent_id=v1.version_id)
    assert v2.parent_id == v1.version_id

def test_create_version_with_description():
    reg = DataVersionRegistry()
    v = reg.create_version("ds", 100, description="initial version")
    assert v.description == "initial version"

def test_create_version_with_metadata_kwargs():
    reg = DataVersionRegistry()
    v = reg.create_version("ds", 100, source="web", lang="en")
    assert v.metadata.get("source") == "web"
    assert v.metadata.get("lang") == "en"


# ---------------------------------------------------------------------------
# DataVersionRegistry.get
# ---------------------------------------------------------------------------

def test_get_returns_version():
    reg = DataVersionRegistry()
    v = reg.create_version("ds", 100)
    result = reg.get(v.version_id)
    assert result is v

def test_get_returns_none_for_unknown():
    reg = DataVersionRegistry()
    assert reg.get("nonexistent") is None

def test_get_returns_correct_version_among_multiple():
    reg = DataVersionRegistry()
    v1 = reg.create_version("ds1", 100)
    v2 = reg.create_version("ds2", 200)
    assert reg.get(v1.version_id) is v1
    assert reg.get(v2.version_id) is v2


# ---------------------------------------------------------------------------
# DataVersionRegistry.diff
# ---------------------------------------------------------------------------

def test_diff_returns_data_diff():
    reg = DataVersionRegistry()
    v1 = reg.create_version("ds", 100)
    v2 = reg.create_version("ds", 200)
    d = reg.diff(v1.version_id, v2.version_id)
    assert isinstance(d, DataDiff)

def test_diff_added():
    reg = DataVersionRegistry()
    v1 = reg.create_version("ds", 100)
    v2 = reg.create_version("ds", 200)
    d = reg.diff(v1.version_id, v2.version_id)
    assert d.added == 100

def test_diff_removed():
    reg = DataVersionRegistry()
    v1 = reg.create_version("ds", 200)
    v2 = reg.create_version("ds", 100)
    d = reg.diff(v1.version_id, v2.version_id)
    assert d.removed == 100

def test_diff_added_zero_when_shrink():
    reg = DataVersionRegistry()
    v1 = reg.create_version("ds", 200)
    v2 = reg.create_version("ds", 100)
    d = reg.diff(v1.version_id, v2.version_id)
    assert d.added == 0

def test_diff_removed_zero_when_grow():
    reg = DataVersionRegistry()
    v1 = reg.create_version("ds", 100)
    v2 = reg.create_version("ds", 200)
    d = reg.diff(v1.version_id, v2.version_id)
    assert d.removed == 0

def test_diff_modified_is_ten_percent():
    reg = DataVersionRegistry()
    v1 = reg.create_version("ds", 100)
    v2 = reg.create_version("ds", 200)
    d = reg.diff(v1.version_id, v2.version_id)
    # min(100, 200) // 10 = 10
    assert d.modified == 10

def test_diff_none_for_unknown_from():
    reg = DataVersionRegistry()
    v = reg.create_version("ds", 100)
    assert reg.diff("nonexistent", v.version_id) is None

def test_diff_none_for_unknown_to():
    reg = DataVersionRegistry()
    v = reg.create_version("ds", 100)
    assert reg.diff(v.version_id, "nonexistent") is None

def test_diff_none_for_both_unknown():
    reg = DataVersionRegistry()
    assert reg.diff("foo", "bar") is None

def test_diff_from_version_field():
    reg = DataVersionRegistry()
    v1 = reg.create_version("ds", 100)
    v2 = reg.create_version("ds", 200)
    d = reg.diff(v1.version_id, v2.version_id)
    assert d.from_version == v1.version_id
    assert d.to_version == v2.version_id


# ---------------------------------------------------------------------------
# DataVersionRegistry.lineage
# ---------------------------------------------------------------------------

def test_lineage_single_version():
    reg = DataVersionRegistry()
    v = reg.create_version("root", 100)
    chain = reg.lineage(v.version_id)
    assert chain == [v]

def test_lineage_two_levels():
    reg = DataVersionRegistry()
    v1 = reg.create_version("root", 100)
    v2 = reg.create_version("child", 150, parent_id=v1.version_id)
    chain = reg.lineage(v2.version_id)
    assert chain == [v1, v2]

def test_lineage_three_levels():
    reg = DataVersionRegistry()
    v1 = reg.create_version("root", 100)
    v2 = reg.create_version("child", 150, parent_id=v1.version_id)
    v3 = reg.create_version("grandchild", 200, parent_id=v2.version_id)
    chain = reg.lineage(v3.version_id)
    assert chain == [v1, v2, v3]

def test_lineage_root_to_version_order():
    reg = DataVersionRegistry()
    v1 = reg.create_version("root", 100)
    v2 = reg.create_version("child", 200, parent_id=v1.version_id)
    chain = reg.lineage(v2.version_id)
    assert chain[0] is v1
    assert chain[-1] is v2

def test_lineage_unknown_returns_empty():
    reg = DataVersionRegistry()
    chain = reg.lineage("nonexistent")
    assert chain == []


# ---------------------------------------------------------------------------
# DataVersionRegistry.list_versions
# ---------------------------------------------------------------------------

def test_list_versions_returns_list():
    reg = DataVersionRegistry()
    reg.create_version("a", 100)
    reg.create_version("b", 200)
    result = reg.list_versions()
    assert isinstance(result, list)

def test_list_versions_sorted_by_created_at():
    reg = DataVersionRegistry()
    v1 = reg.create_version("a", 100)
    v2 = reg.create_version("b", 200)
    v3 = reg.create_version("c", 300)
    result = reg.list_versions()
    ids = [v.version_id for v in result]
    assert v1.version_id in ids
    assert v2.version_id in ids
    assert v3.version_id in ids
    # should be sorted ascending by created_at string
    assert result == sorted(result, key=lambda v: v.created_at)

def test_list_versions_empty_registry():
    reg = DataVersionRegistry()
    assert reg.list_versions() == []

def test_list_versions_count():
    reg = DataVersionRegistry()
    for i in range(5):
        reg.create_version(f"ds{i}", i * 100)
    assert len(reg.list_versions()) == 5


# ---------------------------------------------------------------------------
# DataVersionRegistry.delete
# ---------------------------------------------------------------------------

def test_delete_returns_true_for_known():
    reg = DataVersionRegistry()
    v = reg.create_version("ds", 100)
    assert reg.delete(v.version_id) is True

def test_delete_returns_false_for_unknown():
    reg = DataVersionRegistry()
    assert reg.delete("nonexistent") is False

def test_delete_removes_version():
    reg = DataVersionRegistry()
    v = reg.create_version("ds", 100)
    reg.delete(v.version_id)
    assert reg.get(v.version_id) is None

def test_delete_idempotent():
    reg = DataVersionRegistry()
    v = reg.create_version("ds", 100)
    reg.delete(v.version_id)
    assert reg.delete(v.version_id) is False


# ---------------------------------------------------------------------------
# DATA_VERSION_REGISTRY global
# ---------------------------------------------------------------------------

def test_data_version_registry_exists():
    assert DATA_VERSION_REGISTRY is not None

def test_data_version_registry_is_instance():
    assert isinstance(DATA_VERSION_REGISTRY, DataVersionRegistry)

def test_data_version_registry_usable():
    v = DATA_VERSION_REGISTRY.create_version("global_test", 42)
    assert DATA_VERSION_REGISTRY.get(v.version_id) is v
    DATA_VERSION_REGISTRY.delete(v.version_id)
