"""Canary token pipeline for data-exfiltration detection. AUR-SEC-2026-0013. STRIDE: Information Disclosure."""

import logging
import secrets
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CanaryToken:
    token_id: str
    value: str
    label: str
    created_at: float = field(default_factory=time.time)


@dataclass
class CanaryConfig:
    token_length: int = 32
    prefix: str = "AURELIUS_CANARY_"
    alert_on_output: bool = True


class CanaryPipeline:
    """Pipeline for generating, injecting, and scanning canary tokens."""

    def __init__(self, config: CanaryConfig | None = None) -> None:
        self._config = config or CanaryConfig()
        self._tokens: dict[str, CanaryToken] = {}

    def generate(self, label: str = "") -> CanaryToken:
        """Generate a cryptographically random canary token and register it."""
        token_id = secrets.token_hex(16)
        value = self._config.prefix + secrets.token_hex(self._config.token_length)
        token = CanaryToken(token_id=token_id, value=value, label=label)
        self._tokens[token_id] = token
        return token

    def inject(self, text: str, token: CanaryToken) -> str:
        """Inject the canary token value into text as a hidden XML comment."""
        return text + f"<!-- {token.value} -->"

    def scan(self, text: str) -> list[CanaryToken]:
        """Scan text for any registered canary token values.

        Uses simple substring matching (not regex) to avoid ReDoS.
        Triggered tokens are revoked after detection (one-shot alarm).
        """
        triggered: list[CanaryToken] = []
        # Iterate over a snapshot of items so we can revoke during scan
        for token_id, token in list(self._tokens.items()):
            if token.value in text:
                triggered.append(token)
                self.revoke(token_id)
        return triggered

    def revoke(self, token_id: str) -> bool:
        """Remove a token from the active set. Returns True if it existed."""
        if token_id in self._tokens:
            del self._tokens[token_id]
            return True
        return False

    def revoke_all(self) -> None:
        """Remove all active tokens."""
        self._tokens.clear()

    def active_count(self) -> int:
        """Return the number of currently active (registered) tokens."""
        return len(self._tokens)

    def alert(self, triggered: list[CanaryToken], source: str) -> None:
        """Log a WARNING for each triggered canary. Never logs the token VALUE."""
        for token in triggered:
            logger.warning(
                "Canary triggered — token_id=%s label=%r source=%r",
                token.token_id,
                token.label,
                source,
            )


# Module-level registry and default instance
CANARY_PIPELINE_REGISTRY: dict[str, "CanaryPipeline"] = {}
DEFAULT_CANARY_PIPELINE = CanaryPipeline()
CANARY_PIPELINE_REGISTRY["default"] = DEFAULT_CANARY_PIPELINE
