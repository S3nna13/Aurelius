"""Finding AUR-SEC-2026-0012 mitigation: authentication for serving surface.
No spoofing/EoP via unauthenticated API calls.

Pure-stdlib, backend-agnostic authentication middleware. Keys are stored as
SHA-256 hashes only — raw keys are never retained. All comparisons use
``hmac.compare_digest`` to prevent timing side-channels.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class APIKey:
    """Registered API key record. The raw secret is never stored here."""

    key_id: str
    key_hash: str          # SHA-256 hex digest of the raw key
    scopes: frozenset[str]
    rate_limit_rps: float


@dataclass
class AuthConfig:
    """Configuration for an :class:`AuthMiddleware` instance."""

    keys: dict[str, APIKey]
    require_auth: bool = True
    bearer_header: str = "Authorization"
    api_key_header: str = "X-API-Key"


@dataclass
class AuthResult:
    """Result returned by :meth:`AuthMiddleware.authenticate`."""

    authenticated: bool
    key_id: Optional[str]
    scopes: frozenset[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_hex(raw_key: str) -> str:
    """Return the SHA-256 hex digest of *raw_key* (UTF-8 encoded)."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _timing_safe_equal(a: str, b: str) -> bool:
    """Constant-time string comparison using ``hmac.compare_digest``."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class AuthMiddleware:
    """Authenticate incoming requests against a registry of hashed API keys.

    Usage::

        config = AuthConfig(keys={}, require_auth=True)
        mw = AuthMiddleware(config)
        mw.add_key("svc-a", raw_key, frozenset({"read", "write"}))

        result = mw.authenticate({"Authorization": "Bearer <raw_key>"})
        if not result.authenticated:
            raise PermissionError(result.error)
    """

    def __init__(self, config: AuthConfig) -> None:
        self._config = config
        # Work on a mutable copy so mutations don't affect the caller's dict.
        self._keys: dict[str, APIKey] = dict(config.keys)

    # ------------------------------------------------------------------
    # Core authentication
    # ------------------------------------------------------------------

    def authenticate(self, headers: dict[str, str]) -> AuthResult:
        """Check *headers* for a valid Bearer token or X-API-Key value.

        Lookup is case-insensitive for header names. The provided raw key is
        hashed with SHA-256 and compared to the stored hash using a
        timing-safe digest.

        Returns:
            :class:`AuthResult` — always returns an object; never raises.
        """
        # Normalise header keys to lower-case for case-insensitive lookup.
        normalised = {k.lower(): v for k, v in headers.items()}

        raw_key: Optional[str] = None

        # 1. Try Bearer token in Authorization header.
        bearer_header = self._config.bearer_header.lower()
        auth_value = normalised.get(bearer_header, "")
        if auth_value.lower().startswith("bearer "):
            raw_key = auth_value[len("bearer "):]

        # 2. Fall back to X-API-Key header.
        if raw_key is None:
            api_key_header = self._config.api_key_header.lower()
            header_value = normalised.get(api_key_header)
            if header_value is not None:
                raw_key = header_value

        # 3. No credential supplied.
        if raw_key is None:
            if not self._config.require_auth:
                # Dev / open mode — pass through.
                return AuthResult(
                    authenticated=True,
                    key_id=None,
                    scopes=frozenset(),
                    error=None,
                )
            return AuthResult(
                authenticated=False,
                key_id=None,
                scopes=frozenset(),
                error="No authentication credential provided.",
            )

        # 4. Hash the supplied key and compare against all registered keys.
        #    We always hash and compare every registered key to avoid leaking
        #    information about which key_ids exist via early-exit timing.
        supplied_hash = _sha256_hex(raw_key)
        matched_key: Optional[APIKey] = None

        for api_key in self._keys.values():
            if _timing_safe_equal(supplied_hash, api_key.key_hash):
                matched_key = api_key
                # Do NOT break — continue iterating for constant-time behaviour.

        if matched_key is None:
            return AuthResult(
                authenticated=False,
                key_id=None,
                scopes=frozenset(),
                error="Invalid API key.",
            )

        return AuthResult(
            authenticated=True,
            key_id=matched_key.key_id,
            scopes=matched_key.scopes,
            error=None,
        )

    # ------------------------------------------------------------------
    # Scope enforcement
    # ------------------------------------------------------------------

    def require_scope(self, result: AuthResult, scope: str) -> bool:
        """Return ``True`` iff *result* is authenticated and has *scope*."""
        return result.authenticated and scope in result.scopes

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def add_key(
        self,
        key_id: str,
        raw_key: str,
        scopes: frozenset[str],
        rate_limit_rps: float = 100.0,
    ) -> APIKey:
        """Hash *raw_key* and register it under *key_id*.

        The raw key is hashed immediately and not retained. Returns the
        :class:`APIKey` record so the caller can persist it if needed; the
        raw key must be stored separately by the caller.
        """
        api_key = APIKey(
            key_id=key_id,
            key_hash=_sha256_hex(raw_key),
            scopes=scopes,
            rate_limit_rps=rate_limit_rps,
        )
        self._keys[key_id] = api_key
        return api_key

    def remove_key(self, key_id: str) -> bool:
        """Remove *key_id* from the registry. Returns ``True`` if it existed."""
        if key_id in self._keys:
            del self._keys[key_id]
            return True
        return False

    def has_key(self, key_id: str) -> bool:
        """Return ``True`` if *key_id* is currently registered."""
        return key_id in self._keys


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

#: Global registry mapping logical names to :class:`AuthMiddleware` instances.
AUTH_MIDDLEWARE_REGISTRY: dict[str, AuthMiddleware] = {}

#: Default middleware — **fail-closed** (``require_auth=True``).
#: Production deployments MUST call ``add_key()`` before accepting traffic.
#: The previous default (``require_auth=False``) was changed after security
#: review AUR-SEC-2026-0028 to prevent accidental open-api deployments.
DEFAULT_AUTH_MIDDLEWARE = AuthMiddleware(AuthConfig(keys={}, require_auth=True))
