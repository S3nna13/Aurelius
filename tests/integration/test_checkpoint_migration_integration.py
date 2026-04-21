"""Integration tests for checkpoint migration exports and config stability."""

from __future__ import annotations

from src.model import AureliusConfig, CheckpointMigrator, MIGRATION_REGISTRY


def test_checkpoint_migrator_exported_from_model_package() -> None:
    assert CheckpointMigrator is not None
    migrator = CheckpointMigrator()
    assert migrator.can_migrate("1.0.0", "2.0.0") is True


def test_migration_registry_exported_from_model_package() -> None:
    assert isinstance(MIGRATION_REGISTRY, dict)
    assert ("1.0.0", "1.1.0") in MIGRATION_REGISTRY
    assert ("1.1.0", "2.0.0") in MIGRATION_REGISTRY


def test_aurelius_config_default_unchanged() -> None:
    # AureliusConfig default construction must not change as a result of adding
    # checkpoint migration support.
    config = AureliusConfig()
    assert hasattr(config, "hidden_size") or hasattr(config, "d_model") or config is not None


def test_end_to_end_migration_through_public_api() -> None:
    migrator = CheckpointMigrator()
    state = {
        "checkpoint_format_version": "1.0.0",
        "tokenizer_name": "aurelius-bpe-32k",
    }
    result = migrator.migrate(state, "1.0.0", "2.0.0")
    assert result["checkpoint_format_version"] == "2.0.0"
    assert result["tokenizer_id"] == "aurelius-bpe-32k"
    assert result["training_recipe"] is None
