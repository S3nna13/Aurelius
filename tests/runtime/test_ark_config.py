"""Tests for ark_config."""

from __future__ import annotations

import os
from unittest import mock

from src.runtime.ark_config import ArkConfig


class TestArkConfig:
    """Test ArkConfig functionality."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        cfg = ArkConfig()
        assert cfg.default_model == "aurelius-1.3b"
        assert cfg.fallback_model == "aurelius-1.3b"
        assert cfg.embedding_model == "local-hash-embedding"
        assert cfg.embedding_dim == 1536
        assert cfg.max_tokens_default == 4096
        assert cfg.temperature_default == 0.7

    def test_api_key_auto_generated(self) -> None:
        """Test that api_key is auto-generated if not set."""
        cfg = ArkConfig()
        assert cfg.api_key.startswith("ark-")
        assert len(cfg.api_key) > 16

    def test_jwt_secret_auto_generated(self) -> None:
        """Test that jwt_secret is auto-generated if not set."""
        cfg = ArkConfig()
        assert len(cfg.jwt_secret) == 64  # 32 bytes hex = 64 chars

    def test_coerce_true_values(self) -> None:
        """Test _coerce handles true-like values."""
        assert ArkConfig._coerce("true") is True
        assert ArkConfig._coerce("True") is True
        assert ArkConfig._coerce("yes") is True
        assert ArkConfig._coerce("YES") is True

    def test_coerce_false_values(self) -> None:
        """Test _coerce handles false-like values."""
        assert ArkConfig._coerce("false") is False
        assert ArkConfig._coerce("False") is False
        assert ArkConfig._coerce("no") is False
        assert ArkConfig._coerce("NO") is False

    def test_coerce_int(self) -> None:
        """Test _coerce converts to int."""
        assert ArkConfig._coerce("42") == 42
        assert ArkConfig._coerce("-10") == -10

    def test_coerce_float(self) -> None:
        """Test _coerce converts to float."""
        assert ArkConfig._coerce("3.14") == 3.14
        assert ArkConfig._coerce("-0.5") == -0.5

    def test_coerce_string(self) -> None:
        """Test _coerce returns string for non-numeric values."""
        assert ArkConfig._coerce("hello") == "hello"
        assert ArkConfig._coerce("some-value") == "some-value"

    def test_to_dict_no_redact(self) -> None:
        """Test to_dict without redaction."""
        cfg = ArkConfig()
        d = cfg.to_dict(redact=False)
        assert "api_key" in d
        assert "jwt_secret" in d
        # Should not be redacted
        assert d["api_key"].startswith("ark-")

    def test_to_dict_with_redact(self) -> None:
        """Test to_dict with redaction of sensitive fields."""
        cfg = ArkConfig()
        d = cfg.to_dict(redact=True)
        assert d["api_key"] == "[REDACTED]"
        assert d["jwt_secret"] == "[REDACTED]"
        assert d["openai_api_key"] == "[REDACTED]"
        assert d["neo4j_password"] == "[REDACTED]"
        # Non-sensitive fields not redacted
        assert d["default_model"] == "aurelius-1.3b"

    def test_load_env_override(self) -> None:
        """Test loading config with environment variable overrides."""
        with mock.patch.dict(os.environ, {"ARK_DEFAULT_MODEL": "my-model"}):
            cfg = ArkConfig()
            assert cfg.default_model == "my-model"

    def test_load_env_int_override(self) -> None:
        """Test environment variable int coercion."""
        with mock.patch.dict(os.environ, {"ARK_PORT": "9000"}):
            cfg = ArkConfig()
            assert cfg.port == 9000

    def test_load_env_float_override(self) -> None:
        """Test environment variable float coercion."""
        with mock.patch.dict(os.environ, {"ARK_TEMPERATURE_DEFAULT": "0.5"}):
            cfg = ArkConfig()
            assert cfg.temperature_default == 0.5

    def test_load_env_bool_override(self) -> None:
        """Test environment variable bool coercion."""
        with mock.patch.dict(os.environ, {"ARK_RATE_LIMIT_RPS": "200"}):
            cfg = ArkConfig()
            assert cfg.rate_limit_rps == 200

    def test_load_class_method(self) -> None:
        """Test the load() class method."""
        with mock.patch.dict(os.environ, {"ARK_LOG_LEVEL": "DEBUG"}):
            cfg = ArkConfig.load()
            assert cfg.log_level == "DEBUG"

    def test_load_with_overrides(self) -> None:
        """Test load() with keyword argument overrides."""
        cfg = ArkConfig.load(default_model="custom-model")
        assert cfg.default_model == "custom-model"

    def test_custom_prefix(self) -> None:
        """Test custom environment variable prefix."""
        with mock.patch.dict(os.environ, {"CUSTOM_DEFAULT_MODEL": "prefixed-model"}):
            cfg = ArkConfig(_prefix="CUSTOM_")
            cfg.load_env(prefix="CUSTOM_")
            assert cfg.default_model == "prefixed-model"

    def test_sensitive_fields_set(self) -> None:
        """Test that sensitive fields are properly tracked."""
        cfg = ArkConfig()
        sensitive = cfg._sensitive_fields
        assert "api_key" in sensitive
        assert "jwt_secret" in sensitive
        assert "openai_api_key" in sensitive
        assert "anthropic_api_key" in sensitive
        assert "neo4j_password" in sensitive

    def test_load_env_only_public_fields(self) -> None:
        """Test that load_env only processes public fields."""
        with mock.patch.dict(os.environ, {"ARK_API_KEY": "my-secret-key"}):
            cfg = ArkConfig()
            # api_key is loaded via env override
            assert cfg.api_key == "my-secret-key"

    def test_to_dict_excludes_private_fields(self) -> None:
        """Test that to_dict excludes private fields."""
        cfg = ArkConfig()
        d = cfg.to_dict()
        assert "_prefix" not in d
        assert "_sensitive_fields" not in d
        # Ensure public fields are present
        assert "default_model" in d
        assert "port" in d

    def test_qdrant_defaults(self) -> None:
        """Test Qdrant configuration defaults."""
        cfg = ArkConfig()
        assert cfg.qdrant_url == "http://localhost:6333"
        assert cfg.qdrant_api_key == ""

    def test_neo4j_defaults(self) -> None:
        """Test Neo4j configuration defaults."""
        cfg = ArkConfig()
        assert cfg.neo4j_uri == "bolt://localhost:7687"
        assert cfg.neo4j_user == "neo4j"
        assert cfg.neo4j_password == ""

    def test_rate_limit_defaults(self) -> None:
        """Test rate limiting defaults."""
        cfg = ArkConfig()
        assert cfg.rate_limit_rps == 100
        assert cfg.rate_limit_burst == 200

    def test_memory_defaults(self) -> None:
        """Test memory configuration defaults."""
        cfg = ArkConfig()
        assert cfg.memory_ttl_days == 30
        assert cfg.memory_max_entries == 1000

    def test_cache_defaults(self) -> None:
        """Test cache configuration defaults."""
        cfg = ArkConfig()
        assert cfg.cache_ttl_seconds == 3600
        assert cfg.cache_max_size == 1000

    def test_host_port_defaults(self) -> None:
        """Test host and port defaults."""
        cfg = ArkConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080

    def test_metrics_port_default(self) -> None:
        """Test metrics port default."""
        cfg = ArkConfig()
        assert cfg.metrics_port == 9090

    def test_log_level_default(self) -> None:
        """Test log level default."""
        cfg = ArkConfig()
        assert cfg.log_level == "INFO"

    def test_token_expiry_default(self) -> None:
        """Test token expiry default."""
        cfg = ArkConfig()
        assert cfg.token_expiry_minutes == 60
        assert cfg.jwt_algorithm == "HS256"
