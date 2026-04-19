"""Integration tests for the unified-diff generator surface.

Verifies that :class:`UnifiedDiffGenerator` and :class:`DiffResult` are
exposed via :mod:`src.agent`, that prior agent registry entries remain
intact, and that generated diffs round-trip through the production
``apply_patch_via_python`` helper from :mod:`src.eval.swebench_lite_scorer`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src import agent
from src.agent import DiffResult, UnifiedDiffGenerator
from src.eval.swebench_lite_scorer import apply_patch_via_python


def test_surface_exports_present():
    assert "DiffResult" in agent.__all__
    assert "UnifiedDiffGenerator" in agent.__all__
    assert agent.DiffResult is DiffResult
    assert agent.UnifiedDiffGenerator is UnifiedDiffGenerator


def test_prior_agent_entries_intact():
    # Sanity-check a sampling of registrations from earlier surfaces.
    assert "react" in agent.AGENT_LOOP_REGISTRY
    assert "beam_plan" in agent.AGENT_LOOP_REGISTRY
    assert "safe_dispatch" in agent.AGENT_LOOP_REGISTRY
    assert "xml" in agent.TOOL_CALL_PARSER_REGISTRY
    assert "json" in agent.TOOL_CALL_PARSER_REGISTRY
    # A handful of earlier symbols should still re-export.
    for name in ("ReActLoop", "BeamPlanner", "RepoContextPacker"):
        assert hasattr(agent, name)


def test_generated_diff_applies_via_scorer_helper():
    gen = UnifiedDiffGenerator()
    before = {"pkg/a.py": "def foo():\n    return 1\n"}
    after = {"pkg/a.py": "def foo():\n    return 42\n"}
    result = gen.from_file_pairs(before, after)
    assert result.line_changes > 0

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel, content in before.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

        ok = apply_patch_via_python(result.diff, str(root))
        assert ok is True
        assert (root / "pkg/a.py").read_text() == after["pkg/a.py"]


def test_round_trip_via_public_api_matches_direct_apply():
    gen = UnifiedDiffGenerator(context_lines=2)
    before = {
        "a.txt": "hello\n",
        "b/c.txt": "stay\n",
        "d.txt": "remove me\n",
    }
    after = {
        "a.txt": "HELLO\n",
        "b/c.txt": "stay\n",
        "e.txt": "brand new\n",
    }
    diff = gen.from_file_pairs(before, after).diff
    recovered = gen.apply_round_trip(before, diff)

    # d.txt deleted, e.txt created, a.txt modified, b/c.txt untouched.
    assert recovered.get("a.txt") == after["a.txt"]
    assert recovered.get("e.txt") == after["e.txt"]
    assert recovered.get("b/c.txt") == "stay\n"
    assert "d.txt" not in recovered


def test_validate_exposed_through_surface():
    gen = agent.UnifiedDiffGenerator()
    good = gen.from_single_edit("x.py", "1\n", "2\n").diff
    assert gen.validate(good) is True
    assert gen.validate("junk\n") is False
