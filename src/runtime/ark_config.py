from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArkConfig:
    """Central configuration with environment variable overrides.

    All fields can be overridden via ``ARK_<UPPER_CASE_KEY>`` env vars.
    The ``_prefix`` controls the env var prefix (default ``ARK_``).

    Usage::

        config = ArkConfig.load()
        model = config.default_model

    Security notes:
    - ``api_key`` and ``jwt_secret`` MUST be set via env vars in production.
      They default to generating a random key at startup (with a warning).
    - ``to_dict()`` redacts sensitive fields.
    """

    _prefix: str = "ARK_"
    _sensitive_fields: set[str] = field(
        default_factory=lambda: {
            "api_key",
            "jwt_secret",
            "openai_api_key",
            "anthropic_api_key",
            "cohere_api_key",
            "qdrant_api_key",
            "neo4j_password",
        },
        init=False,
        repr=False,
    )

    # Model Serving
    default_model: str = "gpt-4o"
    fallback_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    max_tokens_default: int = 4096
    temperature_default: float = 0.7

    # API Keys — set via env vars only
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # Vector Database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Search
    elasticsearch_url: str = "http://localhost:9200"

    # Graph
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Auth — auto-generate if not set via env
    api_key: str = ""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    token_expiry_minutes: int = 60

    # Rate Limiting
    rate_limit_rps: int = 100
    rate_limit_burst: int = 200

    # Memory
    memory_ttl_days: int = 30
    memory_max_entries: int = 1000

    # Cache
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000

    # Observability
    log_level: str = "INFO"
    metrics_port: int = 9090

    # Internal
    host: str = "127.0.0.1"
    port: int = 8080

    def __post_init__(self) -> None:
        self.load_env()
        import logging

        logger = logging.getLogger("ark.config")
        if not self.api_key:
            self.api_key = f"ark-{secrets.token_hex(24)}"
            logger.warning(
                "No ARK_API_KEY set. Generated ephemeral key: %s... Use env var for persistence.",
                self.api_key[:16],
            )
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_hex(32)
            logger.warning(
                "No ARK_JWT_SECRET set. Generated random secret. Set via env var in production."
            )

    def load_env(self, prefix: str | None = None) -> None:
        pfx = prefix or self._prefix
        for key in dir(self):
            if key.startswith("_"):
                continue
            env_key = f"{pfx}{key.upper()}"
            if env_key in os.environ:
                val = os.environ[env_key]
                setattr(self, key, self._coerce(val))

    @staticmethod
    def _coerce(val: str) -> Any:
        if val.lower() in ("true", "yes", "1"):
            return True
        if val.lower() in ("false", "no", "0"):
            return False
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val

    def to_dict(self, redact: bool = True) -> dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if redact and k in self._sensitive_fields:
                v = "[REDACTED]" if v else ""
            result[k] = v
        return result

    @classmethod
    def load(cls, **overrides: Any) -> ArkConfig:
        cfg = cls(**overrides)
        cfg.load_env()
        return cfg
