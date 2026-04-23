"""Integration tests for AST-aware FIM and patch synthesis pipelines."""

from __future__ import annotations

from src.agent.ast_fim import ASTAnalyzer, FIMFormat, FIMSpan, FIMTokenizer
from src.agent.patch_synthesis import PatchSynthesizer


_SAMPLE_SOURCE = """\
import os

def compute(x, y):
    result = x + y
    return result

class Calculator:
    def add(self, a, b):
        return a + b
"""


def test_ast_fim_pipeline():
    """Parse Python source, extract context at line 5, format PSM, verify token structure."""
    analyzer = ASTAnalyzer()
    tokenizer = FIMTokenizer()

    # Parse produces nodes
    nodes = analyzer.parse_python(_SAMPLE_SOURCE)
    assert len(nodes) > 0

    # extract_context at line 5 (inside compute)
    prefix, suffix = analyzer.extract_context(_SAMPLE_SOURCE, cursor_line=5, context_lines=20)
    assert isinstance(prefix, str)
    assert isinstance(suffix, str)

    # Build a FIMSpan from extracted context
    span = FIMSpan(
        prefix=prefix,
        suffix=suffix,
        middle="",
        language="python",
        cursor_line=5,
    )

    # Format as PSM and verify token structure
    formatted = tokenizer.format_psm(span)
    assert formatted.startswith(tokenizer.FIM_PREFIX_TOKEN)
    assert tokenizer.FIM_SUFFIX_TOKEN in formatted
    assert tokenizer.FIM_MIDDLE_TOKEN in formatted

    # Verify SPM format starts differently
    formatted_spm = tokenizer.format_spm(span)
    assert formatted_spm.startswith(tokenizer.FIM_SUFFIX_TOKEN)

    # format_span with RANDOM doesn't crash
    for _ in range(5):
        result = tokenizer.format_span(span, FIMFormat.RANDOM)
        assert isinstance(result, str)
        assert tokenizer.FIM_MIDDLE_TOKEN in result


def test_patch_synthesis_roundtrip():
    """make_patch → patch_to_str → apply_str_patch → verify recovered text."""
    synth = PatchSynthesizer()

    original = _SAMPLE_SOURCE
    modified = original.replace(
        "result = x + y",
        "result = x + y  # computed",
    ).replace(
        "return result",
        "return int(result)",
    )

    # Produce a patch
    patch = synth.make_patch(original, modified, filename="compute.py")
    assert len(patch.hunks) >= 1

    # Render to string
    patch_str = synth.patch_to_str(patch)
    assert patch_str.startswith("---")
    assert "+++" in patch_str
    assert "@@" in patch_str

    # Apply via string path
    recovered = synth.apply_str_patch(original, patch_str)
    assert recovered == modified

    # Idempotency: a zero-hunk patch leaves source untouched
    zero_patch = synth.make_patch(original, original)
    assert len(zero_patch.hunks) == 0
    unchanged = synth.apply_patch(original, zero_patch)
    assert unchanged == original


# ---------------------------------------------------------------------------
# Code execution tool integration (additive)
# Code execution: Inspired by Gemini 2.5 code execution tool (Google DeepMind 2025).
# ---------------------------------------------------------------------------

from src.agent.code_execution_tool import CodeExecutionTool, ExecutionRequest  # noqa: E402


def test_code_execution_safe() -> None:
    """Execute simple safe Python and verify stdout is captured."""
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="print('aurelius')")
    result = tool.execute(req)
    assert result.exit_code == 0
    assert "aurelius" in result.stdout
    assert result.timed_out is False
    assert result.error is None


def test_code_execution_blocked() -> None:
    """Blocked code (import os) must return exit_code=1 with 'Blocked' in stderr."""
    tool = CodeExecutionTool()
    req = ExecutionRequest(code="import os\nprint(os.listdir('.'))")
    result = tool.execute(req)
    assert result.exit_code == 1
    assert "Blocked" in result.stderr
