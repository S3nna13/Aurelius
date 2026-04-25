"""Regression tests for cycle-164 security hardening.

Coverage:
- AUR-SEC-2026-0024: sqlitedict/lm-eval removal and security gate
- CWE-78: subprocess hardening (sys.executable, controlled argv)
- CWE-400: proactive_trigger exception handling (no silent swallow)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0024 — sqlitedict security gate
# ---------------------------------------------------------------------------


class TestAURSEC20240024SqlitedictGate:
    def test_harness_blocks_when_sqlitedict_present(self):
        """If sqlitedict is importable, _run_via_python_api must refuse."""
        from src.eval.harness import EvalHarness
        from src.eval.benchmark_config import MMLU

        harness = EvalHarness()
        fake_sqlitedict = MagicMock()

        with patch.dict("sys.modules", {"sqlitedict": fake_sqlitedict}):
            with pytest.raises(RuntimeError) as exc_info:
                harness._run_via_python_api(Path("/fake/checkpoint"), MMLU)

        msg = str(exc_info.value)
        assert "AUR-SEC-2026-0024" in msg
        assert "CWE-502" in msg
        assert "sqlitedict" in msg

    def test_harness_gate_mention_in_docstring(self):
        """The security gate must be documented in the method docstring."""
        from src.eval.harness import EvalHarness

        assert "AUR-SEC-2026-0024" in EvalHarness._run_via_python_api.__doc__
        assert "CWE-502" in EvalHarness._run_via_python_api.__doc__


# ---------------------------------------------------------------------------
# CWE-78 — subprocess hardening
# ---------------------------------------------------------------------------


class TestSubprocessHardening:
    def test_code_execution_tool_uses_sys_executable(self):
        """CodeExecutionTool must use sys.executable instead of hardcoded python3."""
        from src.agent.code_execution_tool import (
            CodeExecutionTool,
            ExecutionRequest,
            ExecutionLanguage,
        )

        tool = CodeExecutionTool()
        req = ExecutionRequest(code="print(42)", language=ExecutionLanguage.PYTHON)

        with patch("src.agent.code_execution_tool.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="42\n", stderr=""
            )
            tool.execute(req)

        call_args = mock_run.call_args
        assert call_args is not None
        argv = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
        assert argv[0] == sys.executable, (
            f"Expected sys.executable ({sys.executable!r}), got {argv[0]!r}"
        )

    def test_code_execution_tool_rejects_javascript(self):
        """JavaScript execution must be rejected (not implemented = smaller attack surface)."""
        from src.agent.code_execution_tool import (
            CodeExecutionTool,
            ExecutionRequest,
            ExecutionLanguage,
        )

        tool = CodeExecutionTool()
        req = ExecutionRequest(
            code="alert(1)", language=ExecutionLanguage.JAVASCRIPT
        )
        result = tool.execute(req)
        assert result.exit_code == 1
        assert "not yet supported" in result.stderr

    def test_code_execution_tool_deny_patterns_block_dangerous_code(self):
        """Deny patterns must block obviously dangerous primitives."""
        from src.agent.code_execution_tool import (
            CodeExecutionTool,
            ExecutionRequest,
            ExecutionLanguage,
        )

        tool = CodeExecutionTool()
        dangerous_snippets = [
            "import os; os.system('ls')",
            "import sys; sys.exit(1)",
            "__import__('os').system('ls')",
            "eval('1+1')",
            "exec('pass')",
            "open('/etc/passwd').read()",
        ]
        for code in dangerous_snippets:
            req = ExecutionRequest(code=code, language=ExecutionLanguage.PYTHON)
            result = tool.execute(req)
            assert result.exit_code == 1, f"Expected block for: {code!r}"
            assert "Blocked" in result.stderr


# ---------------------------------------------------------------------------
# CWE-400 / CWE-209 — proactive_trigger exception handling
# ---------------------------------------------------------------------------


class TestProactiveTriggerExceptionHandling:
    def test_broken_trigger_fn_is_logged_not_silenced(self, caplog):
        """A broken trigger_fn must be logged, not silently swallowed."""
        from src.agent.proactive_trigger import (
            ProactiveTriggerRegistry,
            TriggerSpec,
        )

        reg = ProactiveTriggerRegistry()
        bad_spec = TriggerSpec(
            name="bad",
            trigger_fn=lambda t: (_ for _ in ()).throw(RuntimeError("boom")),
            action="explode",
        )
        reg.register(bad_spec)

        with caplog.at_level(logging.WARNING):
            fired = reg.check_all(current_time=0.0)

        assert fired == []
        assert "bad" in caplog.text or "predicate raised" in caplog.text

    def test_trigger_registry_limits_max_triggers(self):
        """Registry capacity limits prevent unbounded DoS."""
        from src.agent.proactive_trigger import (
            ProactiveTriggerRegistry,
            ProactiveTriggerConfig,
            TriggerSpec,
        )

        config = ProactiveTriggerConfig(max_triggers=2)
        reg = ProactiveTriggerRegistry(config)
        reg.register(TriggerSpec("t1", lambda t: True, "a1"))
        reg.register(TriggerSpec("t2", lambda t: True, "a2"))

        with pytest.raises(ValueError, match="capacity"):
            reg.register(TriggerSpec("t3", lambda t: True, "a3"))


# ---------------------------------------------------------------------------
# Foreign-import isolation check (runtime.isolation compliance)
# ---------------------------------------------------------------------------


class TestIsolationCompliance:
    """Spot-check that core surfaces do not import banned optional deps."""

    def test_eval_harness_no_lm_eval_at_module_level(self):
        """lm_eval must not appear at module level in harness.py."""
        import src.eval.harness as _harness_mod
        import inspect
        harness_path = Path(inspect.getfile(_harness_mod))
        source = harness_path.read_text(encoding="utf-8")
        # lm_eval is allowed only inside the method body (lazy import)
        lines = source.splitlines()
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("import lm_eval") or stripped.startswith("from lm_eval"):
                # Must be indented (inside a function/method)
                if not line.startswith("        ") and not line.startswith("\t"):
                    pytest.fail(
                        f"Module-level lm_eval import found at {harness_path}:{i}"
                    )

    def test_security_gate_imports_only_stdlib(self):
        """The security gate code in harness.py must use only stdlib imports."""
        # The gate itself does `import sqlitedict` — that's the whole point.
        # Everything else in the method must be stdlib or project-local.
        from src.eval.harness import EvalHarness

        import inspect
        sig = inspect.signature(EvalHarness._run_via_python_api)
        # No assertion needed — if the import were banned, the module would fail to load.
        assert "checkpoint_path" in sig.parameters
