"""Tests for checkpoint metadata migration orchestrator."""

from __future__ import annotations

import pytest

from src.model.checkpoint_migration import (
    MIGRATION_REGISTRY,
    CheckpointMigrator,
    MigrationError,
    MigrationStep,
    get_migration_path,
    register_migration,
)


def test_single_step_migration_applies() -> None:
    migrator = CheckpointMigrator()
    state = {"checkpoint_format_version": "1.0.0"}
    result = migrator.migrate(state, "1.0.0", "1.1.0")
    assert result["checkpoint_format_version"] == "1.1.0"
    assert result["training_recipe"] is None


def test_multi_step_path_composes() -> None:
    migrator = CheckpointMigrator()
    state = {
        "checkpoint_format_version": "1.0.0",
        "tokenizer_name": "aurelius-bpe",
    }
    result = migrator.migrate(state, "1.0.0", "2.0.0")
    assert result["checkpoint_format_version"] == "2.0.0"
    assert result["tokenizer_id"] == "aurelius-bpe"
    assert "tokenizer_name" not in result
    assert result["training_recipe"] is None


def test_same_version_migration_is_noop() -> None:
    migrator = CheckpointMigrator()
    state = {"checkpoint_format_version": "1.0.0", "foo": "bar"}
    result = migrator.migrate(state, "1.0.0", "1.0.0")
    assert result == state
    assert result is not state  # new dict, not mutated input


def test_unknown_source_raises() -> None:
    with pytest.raises(MigrationError):
        get_migration_path("0.0.1", "1.0.0")


def test_unknown_destination_raises() -> None:
    with pytest.raises(MigrationError):
        get_migration_path("1.0.0", "9.9.9")


def test_register_migration_duplicate_raises() -> None:
    step = MigrationStep(
        from_version="1.0.0",
        to_version="1.1.0",
        description="dup",
        apply=lambda s: s,
    )
    with pytest.raises(MigrationError):
        register_migration(step)


def test_bfs_finds_shortest_path() -> None:
    path = get_migration_path("1.0.0", "2.0.0")
    assert len(path) == 2
    assert path[0].from_version == "1.0.0"
    assert path[0].to_version == "1.1.0"
    assert path[1].from_version == "1.1.0"
    assert path[1].to_version == "2.0.0"


def test_version_after_applying_matches_to_version() -> None:
    migrator = CheckpointMigrator()
    state = {"checkpoint_format_version": "1.0.0"}
    assert migrator.migrate(state, "1.0.0", "1.1.0")["checkpoint_format_version"] == "1.1.0"
    assert migrator.migrate(state, "1.0.0", "2.0.0")["checkpoint_format_version"] == "2.0.0"


def test_apply_never_mutates_input_dict() -> None:
    migrator = CheckpointMigrator()
    state = {
        "checkpoint_format_version": "1.0.0",
        "tokenizer_name": "aurelius-bpe",
    }
    snapshot = dict(state)
    migrator.migrate(state, "1.0.0", "2.0.0")
    assert state == snapshot


def test_dry_run_returns_descriptions_without_mutation() -> None:
    migrator = CheckpointMigrator()
    descriptions = migrator.dry_run("1.0.0", "2.0.0")
    assert descriptions == [
        "add optional 'training_recipe' field",
        "rename 'tokenizer_name' to 'tokenizer_id'",
    ]


def test_can_migrate_true_for_known() -> None:
    migrator = CheckpointMigrator()
    assert migrator.can_migrate("1.0.0", "2.0.0") is True
    assert migrator.can_migrate("1.0.0", "1.1.0") is True
    assert migrator.can_migrate("1.0.0", "1.0.0") is True


def test_can_migrate_false_for_disconnected() -> None:
    migrator = CheckpointMigrator()
    assert migrator.can_migrate("1.0.0", "99.0.0") is False
    assert migrator.can_migrate("unknown", "1.1.0") is False


def test_two_seed_migrations_registered() -> None:
    assert ("1.0.0", "1.1.0") in MIGRATION_REGISTRY
    assert ("1.1.0", "2.0.0") in MIGRATION_REGISTRY


def test_round_trip_1_0_0_to_1_1_0_to_2_0_0() -> None:
    migrator = CheckpointMigrator()
    state = {
        "checkpoint_format_version": "1.0.0",
        "tokenizer_name": "bpe-v1",
        "weights_sha": "abc123",
    }
    step1 = migrator.migrate(state, "1.0.0", "1.1.0")
    assert step1["checkpoint_format_version"] == "1.1.0"
    assert step1["tokenizer_name"] == "bpe-v1"
    step2 = migrator.migrate(step1, "1.1.0", "2.0.0")
    assert step2["checkpoint_format_version"] == "2.0.0"
    assert step2["tokenizer_id"] == "bpe-v1"
    assert step2["weights_sha"] == "abc123"


def test_renaming_migration_preserves_value_under_new_key() -> None:
    migrator = CheckpointMigrator()
    state = {
        "checkpoint_format_version": "1.1.0",
        "tokenizer_name": "sentencepiece-32k",
    }
    result = migrator.migrate(state, "1.1.0", "2.0.0")
    assert result["tokenizer_id"] == "sentencepiece-32k"
    assert "tokenizer_name" not in result
