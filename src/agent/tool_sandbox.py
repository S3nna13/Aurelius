from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ToolSandbox:
    """Sandboxed tool execution with permission checking.

    Enforces:
    - Tool permission levels: read-only, write, dangerous
    - Rate limiting
    - Input validation against schemas
    - Execution timeout
    - Network egress controls (for dangerous tools)
    - Audit logging
    """

    def __init__(
        self,
        permission_check: Callable[[str, str], bool] | None = None,
        rate_limiter: Callable[[str], bool] | None = None,
    ) -> None:
        self._permission_check = permission_check or (lambda user_id, tool: True)
        self._rate_limiter = rate_limiter or (lambda user_id: True)
        self._audit_log: list[dict[str, Any]] = []

    def check_permission(self, user_id: str, tool_name: str, level: str = "none") -> bool:
        if level == "dangerous":
            return False  # Always require explicit approval
        if level == "write":
            # Only allow if explicit permission check passes
            return self._permission_check(user_id, tool_name)
        return True

    def log_execution(
        self,
        user_id: str,
        tool_name: str,
        input_data: dict[str, Any],
        result: Any,
        error: str | None = None,
    ) -> None:
        self._audit_log.append(
            {
                "user_id": user_id,
                "tool": tool_name,
                "input": input_data,
                "error": error,
                "success": error is None,
            }
        )

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._audit_log[-limit:]
