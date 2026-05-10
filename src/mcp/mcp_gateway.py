"""Aurelius MCP Gateway — centralized security proxy for all MCP traffic.

Enforces tool allowlisting, capability-level permissions, egress filtering,
schema hash validation, audit logging, human-in-the-loop gates, rate limiting,
and input sanitization.  Transparent proxy: wraps an existing handler callable.

Inspired by OWASP and Coalition for Secure AI (2026) recommendations for
MCP centralized gateway architecture.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.mcp.tool_rate_limiter import ToolRateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instruction-injection patterns to strip from tool outputs
# ---------------------------------------------------------------------------

_INSTRUCTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(?:all\s+)?(?:your\s+)?instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:an?\s+)?", re.IGNORECASE),
    re.compile(r"system\s*:\s*new\s+instruction", re.IGNORECASE),
    re.compile(r"<!--\s*ignore\s*-->", re.IGNORECASE),
    re.compile(r"\[\s*system\s*override\s*\]", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"ignore\s+above\s+instructions?", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Destructive-operation name heuristics
# ---------------------------------------------------------------------------

_DESTRUCTIVE_TOOLS: set[str] = {
    "delete",
    "remove",
    "drop",
    "write",
    "modify",
    "update",
    "exec",
    "shell",
    "execute",
    "wipe",
    "destroy",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SecurityException(Exception):
    """Raised when a security check in the MCP gateway fails."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MCPGatewayConfig:
    """Configuration for the MCP security gateway.

    Attributes:
        enabled: Master switch; when ``False`` the gateway is a no-op.
        tool_allowlist: List of regex patterns.  Empty list = allow all.
        role_capabilities: Mapping of role name → list of regex patterns.
            A role with no entry implicitly has no restrictions.
        allowed_egress_domains: Domain patterns allowed in tool params.
            Supports ``*`` wildcards, e.g. ``*.example.com``.
        allowed_egress_ips: CIDR blocks allowed for resolved IPs.
        require_hitl_for_destructive: If ``True``, destructive tools need
            an explicit approval token before execution.
        rate_limit_per_tool: Max calls per tool per window.
        rate_limit_window: Sliding-window length in seconds.
        rate_limit_per_user: Max calls per user per window.
        sanitize_outputs: If ``True``, strip instruction-like strings
            from tool outputs before returning them to the LLM.
        audit_log_path: Optional filesystem path to append newline-JSON.
    """

    enabled: bool = False
    tool_allowlist: list[str] = field(default_factory=list)
    role_capabilities: dict[str, list[str]] = field(default_factory=dict)
    allowed_egress_domains: list[str] = field(default_factory=list)
    allowed_egress_ips: list[str] = field(default_factory=list)
    require_hitl_for_destructive: bool = True
    rate_limit_per_tool: int = 60
    rate_limit_window: float = 60.0
    rate_limit_per_user: int = 120
    sanitize_outputs: bool = True
    audit_log_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPGatewayConfig":
        """Build a config from a plain dict (e.g. parsed YAML)."""
        return cls(
            enabled=bool(data.get("enabled", False)),
            tool_allowlist=list(data.get("tool_allowlist", [])),
            role_capabilities=dict(data.get("role_capabilities", {})),
            allowed_egress_domains=list(data.get("allowed_egress_domains", [])),
            allowed_egress_ips=list(data.get("allowed_egress_ips", [])),
            require_hitl_for_destructive=bool(
                data.get("require_hitl_for_destructive", True)
            ),
            rate_limit_per_tool=int(data.get("rate_limit_per_tool", 60)),
            rate_limit_window=float(data.get("rate_limit_window", 60.0)),
            rate_limit_per_user=int(data.get("rate_limit_per_user", 120)),
            sanitize_outputs=bool(data.get("sanitize_outputs", True)),
            audit_log_path=data.get("audit_log_path") or None,
        )


@dataclass
class AuditLogEntry:
    """Single row in the gateway audit trail."""

    timestamp: float
    caller_id: str
    tool_name: str
    params_hash: str
    allowed: bool
    reason: str | None = None


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class MCPGateway:
    """Centralized security gateway that proxies all MCP traffic.

    Usage::

        config = MCPGatewayConfig(enabled=True, tool_allowlist=["^search$"])
        gw = MCPGateway(config)
        result = gw.intercept(
            handler=my_tool_handler,
            tool_name="search",
            caller_id="user-42",
            caller_role="read_only",
            params={"query": "hello"},
        )
    """

    def __init__(self, config: MCPGatewayConfig) -> None:
        self.config = config
        self._schema_hashes: dict[str, str] = {}
        self._tool_rate_limiter = ToolRateLimiter(
            max_calls=config.rate_limit_per_tool,
            window=config.rate_limit_window,
        )
        self._user_rate_limiter = ToolRateLimiter(
            max_calls=config.rate_limit_per_user,
            window=config.rate_limit_window,
        )
        self._audit_log: list[AuditLogEntry] = []
        self._allowed_domains: list[re.Pattern[str]] = [
            re.compile(r"^" + re.escape(d).replace(r"\*", ".*") + r"$")
            for d in config.allowed_egress_domains
        ]
        self._allowed_nets: list[
            ipaddress.IPv4Network | ipaddress.IPv6Network
        ] = []
        for ip in config.allowed_egress_ips:
            self._allowed_nets.append(ipaddress.ip_network(ip, strict=False))
        self._allowlist_patterns: list[re.Pattern[str]] = [
            re.compile(p) for p in config.tool_allowlist
        ]
        self._hitl_approvals: set[str] = set()

    # ------------------------------------------------------------------
    # Schema hash validation
    # ------------------------------------------------------------------

    def register_schema(self, tool_name: str, schema: dict[str, Any]) -> None:
        """Compute and store the SHA-256 hash of a tool's schema on first load."""
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        self._schema_hashes[tool_name] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    def validate_schema(self, tool_name: str, schema: dict[str, Any]) -> bool:
        """Re-validate a tool's schema hash before calling.

        If the tool has never been seen, its hash is stored and the call
        is allowed.  Subsequent calls must match the stored hash.
        """
        if tool_name not in self._schema_hashes:
            self.register_schema(tool_name, schema)
            return True
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        current_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self._schema_hashes[tool_name] == current_hash

    # ------------------------------------------------------------------
    # Internal security checks
    # ------------------------------------------------------------------

    def _check_allowlist(self, tool_name: str) -> bool:
        if not self._allowlist_patterns:
            return True
        return any(p.search(tool_name) for p in self._allowlist_patterns)

    def _check_capabilities(self, tool_name: str, caller_role: str) -> bool:
        caps = self.config.role_capabilities.get(caller_role)
        if caps is None:
            return False
        if not caps:
            return False
        return any(re.compile(c).search(tool_name) for c in caps)

    def _is_destructive(self, tool_name: str) -> bool:
        return tool_name.lower() in _DESTRUCTIVE_TOOLS

    def _check_egress(self, params: dict[str, Any]) -> bool:
        """Block external network calls to non-allowed IPs / domains."""
        urls: list[str] = []
        self._extract_urls(params, urls)

        for url in urls:
            host = self._extract_host(url)
            if not host:
                continue

            if self._allowed_domains:
                if any(p.match(host) for p in self._allowed_domains):
                    continue
                if not self._allowed_nets:
                    return False

            if self._allowed_nets:
                try:
                    addr_info = socket.getaddrinfo(host, None)
                except socket.gaierror:
                    return False
                for _fam, _type, _proto, _canon, sockaddr in addr_info:
                    ip_str = sockaddr[0]
                    try:
                        ip = ipaddress.ip_address(ip_str)
                    except ValueError:
                        continue
                    if any(ip in net for net in self._allowed_nets):
                        break
                else:
                    return False
            else:
                # If neither domains nor IPs are configured, allow everything.
                pass

        return True

    def _extract_urls(self, obj: Any, urls: list[str]) -> None:
        if isinstance(obj, str):
            for m in re.finditer(r"https?://[^\s\"'<>]+", obj):
                urls.append(m.group(0))
        elif isinstance(obj, dict):
            for v in obj.values():
                self._extract_urls(v, urls)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_urls(item, urls)

    @staticmethod
    def _extract_host(url: str) -> str | None:
        m = re.match(r"https?://([^/:=]+)", url)
        return m.group(1) if m else None

    def _check_rate_limits(self, tool_name: str, caller_id: str) -> tuple[bool, str]:
        tool_result = self._tool_rate_limiter.check_call(tool_name)
        if not tool_result.allowed:
            return False, f"tool rate limit exceeded for {tool_name}"
        user_result = self._user_rate_limiter.check_call(caller_id)
        if not user_result.allowed:
            return False, f"user rate limit exceeded for {caller_id}"
        return True, ""

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def _log(
        self,
        caller_id: str,
        tool_name: str,
        params: dict[str, Any],
        allowed: bool,
        reason: str | None = None,
    ) -> None:
        params_canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        params_hash = hashlib.sha256(params_canonical.encode("utf-8")).hexdigest()
        entry = AuditLogEntry(
            timestamp=time.time(),
            caller_id=caller_id,
            tool_name=tool_name,
            params_hash=params_hash,
            allowed=allowed,
            reason=reason,
        )
        self._audit_log.append(entry)
        logger.info(
            "MCP_GATEWAY %s caller=%s tool=%s params_hash=%s reason=%s",
            "ALLOW" if allowed else "DENY",
            caller_id,
            tool_name,
            params_hash,
            reason or "",
        )
        if self.config.audit_log_path:
            self._append_audit_log(entry)

    def _append_audit_log(self, entry: AuditLogEntry) -> None:
        try:
            with open(self.config.audit_log_path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "timestamp": entry.timestamp,
                            "caller_id": entry.caller_id,
                            "tool_name": entry.tool_name,
                            "params_hash": entry.params_hash,
                            "allowed": entry.allowed,
                            "reason": entry.reason,
                        }
                    )
                    + "\n"
                )
        except OSError as exc:
            logger.error("Failed to write audit log: %s", exc)

    # ------------------------------------------------------------------
    # Human-in-the-loop
    # ------------------------------------------------------------------

    def request_hitl_approval(
        self, tool_name: str, caller_id: str, params: dict[str, Any]
    ) -> str:
        """Generate a one-time approval token for a destructive operation."""
        params_canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        token = hashlib.sha256(
            f"{caller_id}:{tool_name}:{params_canonical}:{time.time()}".encode(
                "utf-8"
            )
        ).hexdigest()
        logger.warning(
            "HITL_APPROVAL_REQUIRED caller=%s tool=%s token=%s",
            caller_id,
            tool_name,
            token,
        )
        return token

    def approve_hitl(self, token: str) -> None:
        """Approve a pending HITL request (one-time use)."""
        self._hitl_approvals.add(token)

    def _is_hitl_approved(self, token: str | None) -> bool:
        return token is not None and token in self._hitl_approvals

    # ------------------------------------------------------------------
    # Output sanitization
    # ------------------------------------------------------------------

    def sanitize(self, output: Any) -> Any:
        if not self.config.sanitize_outputs:
            return {"__trusted": False, "content": output}
        if isinstance(output, str):
            for pattern in _INSTRUCTION_PATTERNS:
                output = pattern.sub("[SANITIZED]", output)
            return {"__trusted": True, "content": output}
        if isinstance(output, dict):
            return {"__trusted": False, "content": {k: self.sanitize(v) for k, v in output.items()}}
        if isinstance(output, list):
            return {"__trusted": False, "content": [self.sanitize(item) for item in output]}
        return {"__trusted": False, "content": output}

    # ------------------------------------------------------------------
    # Main interception API
    # ------------------------------------------------------------------

    def intercept(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        tool_name: str,
        caller_id: str,
        caller_role: str = "default",
        params: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
        hitl_token: str | None = None,
    ) -> dict[str, Any]:
        """Proxy a tool call through every security layer.

        Raises:
            SecurityException: If any policy check fails.
        """
        params = dict(params) if params else {}

        # 1. Tool allowlisting
        if not self._check_allowlist(tool_name):
            self._log(
                caller_id, tool_name, params, allowed=False, reason="not in allowlist"
            )
            raise SecurityException(
                f"Tool {tool_name!r} is not in the allowlist"
            )

        # 2. Capability scoping
        if not self._check_capabilities(tool_name, caller_role):
            self._log(
                caller_id,
                tool_name,
                params,
                allowed=False,
                reason="capability denied",
            )
            raise SecurityException(
                f"Role {caller_role!r} lacks capability for {tool_name!r}"
            )

        # 3. Egress filtering
        if not self._check_egress(params):
            self._log(
                caller_id, tool_name, params, allowed=False, reason="egress blocked"
            )
            raise SecurityException("Egress filtering blocked this request")

        # 4. Rate limiting
        allowed, reason = self._check_rate_limits(tool_name, caller_id)
        if not allowed:
            self._log(caller_id, tool_name, params, allowed=False, reason=reason)
            raise SecurityException(reason)

        # 5. Schema hash validation (prevents rug-pull updates)
        if schema is not None and not self.validate_schema(tool_name, schema):
            self._log(
                caller_id,
                tool_name,
                params,
                allowed=False,
                reason="schema hash mismatch",
            )
            raise SecurityException(
                f"Schema hash mismatch for {tool_name!r} (possible rug-pull update)"
            )

        # 6. Human-in-the-loop for destructive operations
        if self.config.require_hitl_for_destructive and self._is_destructive(
            tool_name
        ):
            if not self._is_hitl_approved(hitl_token):
                token = self.request_hitl_approval(tool_name, caller_id, params)
                self._log(
                    caller_id,
                    tool_name,
                    params,
                    allowed=False,
                    reason=f"hitl required: {token}",
                )
                raise SecurityException(
                    f"Human approval required. Token: {token}"
                )
            # Consume the token (one-time use)
            self._hitl_approvals.discard(hitl_token)

        # Execute
        self._log(caller_id, tool_name, params, allowed=True)
        result = handler(params)

        # 7. Sanitize output
        return self.sanitize(result)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MCP_GATEWAY_REGISTRY: dict[str, type[MCPGateway]] = {"default": MCPGateway}

__all__ = [
    "MCPGateway",
    "MCPGatewayConfig",
    "SecurityException",
    "AuditLogEntry",
    "MCP_GATEWAY_REGISTRY",
]
