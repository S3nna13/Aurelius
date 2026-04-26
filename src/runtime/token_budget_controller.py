from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetConfig:
    max_input_tokens: int = 8192
    max_output_tokens: int = 4096
    max_total_tokens: int = 16384
    cost_per_input_token: float = 0.000003
    cost_per_output_token: float = 0.000015


@dataclass
class BudgetState:
    session_id: str
    input_tokens_used: int
    output_tokens_used: int
    total_tokens_used: int
    cost_usd: float
    hard_limit_hit: bool


class TokenBudgetController:
    """Per-session token budget enforcement with cost tracking."""

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self._config = config or BudgetConfig()
        self._sessions: dict[str, BudgetState] = {}

    def create_session(self, session_id: str) -> BudgetState:
        state = BudgetState(
            session_id=session_id,
            input_tokens_used=0,
            output_tokens_used=0,
            total_tokens_used=0,
            cost_usd=0.0,
            hard_limit_hit=False,
        )
        self._sessions[session_id] = state
        return state

    def _get_or_raise(self, session_id: str) -> BudgetState:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session: {session_id}")
        return self._sessions[session_id]

    def record_usage(self, session_id: str, input_tokens: int, output_tokens: int) -> BudgetState:
        state = self._get_or_raise(session_id)
        state.input_tokens_used += input_tokens
        state.output_tokens_used += output_tokens
        state.total_tokens_used += input_tokens + output_tokens
        state.cost_usd = (
            state.input_tokens_used * self._config.cost_per_input_token
            + state.output_tokens_used * self._config.cost_per_output_token
        )
        if (
            state.input_tokens_used > self._config.max_input_tokens
            or state.output_tokens_used > self._config.max_output_tokens
            or state.total_tokens_used > self._config.max_total_tokens
        ):
            state.hard_limit_hit = True
        return state

    def check_budget(
        self,
        session_id: str,
        requested_input: int = 0,
        requested_output: int = 0,
    ) -> tuple[bool, str]:
        state = self._get_or_raise(session_id)
        cfg = self._config
        if state.input_tokens_used + requested_input > cfg.max_input_tokens:
            return False, "input token limit would be exceeded"
        if state.output_tokens_used + requested_output > cfg.max_output_tokens:
            return False, "output token limit would be exceeded"
        if state.total_tokens_used + requested_input + requested_output > cfg.max_total_tokens:
            return False, "total token limit would be exceeded"
        return True, "ok"

    def get_remaining(self, session_id: str) -> dict:
        state = self._get_or_raise(session_id)
        cfg = self._config
        total_used = state.total_tokens_used
        budget_pct = (total_used / cfg.max_total_tokens * 100) if cfg.max_total_tokens else 0.0
        return {
            "input_remaining": max(0, cfg.max_input_tokens - state.input_tokens_used),
            "output_remaining": max(0, cfg.max_output_tokens - state.output_tokens_used),
            "total_remaining": max(0, cfg.max_total_tokens - total_used),
            "cost_usd": state.cost_usd,
            "budget_pct_used": budget_pct,
        }

    def reset_session(self, session_id: str) -> BudgetState:
        self._get_or_raise(session_id)
        return self.create_session(session_id)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self._sessions.values())

    def prune_sessions(self, max_sessions: int = 1000) -> int:
        overflow = len(self._sessions) - max_sessions
        if overflow <= 0:
            return 0
        to_remove = list(self._sessions.keys())[:overflow]
        for sid in to_remove:
            del self._sessions[sid]
        return len(to_remove)
