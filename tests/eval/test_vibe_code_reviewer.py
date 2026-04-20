"""Unit tests for src/eval/vibe_code_reviewer.py."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.eval.vibe_code_reviewer import (
    NEGATIVE_EXAMPLES,
    REQUIRED_FIELDS,
    VULNHUNTER_SYSTEM_PROMPT,
    VibeCodeReviewer,
    VibeFinding,
    VibeReviewReport,
    stub_judge_fn,
)


def test_system_prompt_contains_skepticism_rule():
    assert (
        "Don't flag just because it touches user input; you must articulate "
        "attacker identity + controlled input + sink line + one-line fix "
        "before emitting a finding."
    ) in VULNHUNTER_SYSTEM_PROMPT
    # ≥30 lines
    assert len(VULNHUNTER_SYSTEM_PROMPT.splitlines()) >= 30


def test_system_prompt_contains_negative_list_substrings():
    assert "Do NOT flag" in VULNHUNTER_SYSTEM_PROMPT
    assert "style issues" in VULNHUNTER_SYSTEM_PROMPT
    assert "test fixtures with test creds" in VULNHUNTER_SYSTEM_PROMPT
    assert "env-var secrets" in VULNHUNTER_SYSTEM_PROMPT


def test_negative_examples_non_empty():
    assert isinstance(NEGATIVE_EXAMPLES, tuple)
    assert len(NEGATIVE_EXAMPLES) >= 5


def test_stub_judge_produces_findings_with_required_fields():
    code = "x = 1\n# VIBE_SSRF_SINK\nrequests.get(url)\n"
    findings = stub_judge_fn(code, "f.py")
    assert findings
    for f in findings:
        for field in REQUIRED_FIELDS:
            assert field in f
            assert isinstance(f[field], str)


def test_low_quality_finding_dropped():
    code = "VIBE_SSRF_SINK\nVIBE_LOW_QUALITY\n"
    reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="low")
    report = reviewer.review_file("x.py", code=code)
    # The low-quality marker has empty why_real/fix_hint -> dropped
    assert report.dropped >= 1
    for f in report.findings:
        assert f.why_real.strip()
        assert f.fix_hint.strip()
        assert f.cwe.strip()


def test_review_file_reads_when_code_not_supplied():
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write("# VIBE_SSRF_SINK\nrequests.get(u)\n")
        path = tmp.name
    try:
        reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="low")
        report = reviewer.review_file(path)
        assert any("CWE-918" == f.cwe for f in report.findings)
    finally:
        os.unlink(path)


def test_review_corpus_multiple_files():
    reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="low")
    paths = []
    try:
        for marker in ("VIBE_SSRF_SINK", "VIBE_IDOR_SINK"):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(f"# {marker}\n")
                paths.append(tmp.name)
        out = reviewer.review_corpus(paths)
        assert set(out.keys()) == set(paths)
        assert all(isinstance(v, VibeReviewReport) for v in out.values())
    finally:
        for p in paths:
            os.unlink(p)


def test_empty_code_yields_empty_findings():
    reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="low")
    report = reviewer.review_file("x.py", code="")
    assert report.findings == []
    assert report.total_lines == 0


def test_judge_raising_yields_empty_findings_and_warning():
    def bad_judge(code, path):
        raise RuntimeError("boom")

    reviewer = VibeCodeReviewer(bad_judge, min_severity="low")
    report = reviewer.review_file("x.py", code="print(1)\n")
    assert report.findings == []
    assert report.warning and "boom" in report.warning


def test_min_severity_filters_below_threshold():
    code = "# VIBE_LOW_SEVERITY\n# VIBE_SSRF_SINK\n"
    reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="high")
    report = reviewer.review_file("x.py", code=code)
    # low severity dropped, high kept
    sevs = [f.severity for f in report.findings]
    assert "low" not in sevs
    assert "high" in sevs


def test_line_number_present_and_snippet_surrounds_line():
    code = "a = 1\nb = 2\n# VIBE_SSRF_SINK\nc = 3\nd = 4\n"
    reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="low")
    report = reviewer.review_file("x.py", code=code)
    assert report.findings
    f = report.findings[0]
    assert f.line == 3
    assert "VIBE_SSRF_SINK" in f.snippet
    assert "b = 2" in f.snippet or "c = 3" in f.snippet


def test_elapsed_s_non_negative():
    reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="low")
    report = reviewer.review_file("x.py", code="# VIBE_SSRF_SINK\n")
    assert report.elapsed_s >= 0.0


def test_determinism_with_stub():
    code = "# VIBE_SSRF_SINK\n# VIBE_IDOR_SINK\n"
    reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="low")
    r1 = reviewer.review_file("x.py", code=code)
    r2 = reviewer.review_file("x.py", code=code)
    assert [(f.cwe, f.line) for f in r1.findings] == [
        (f.cwe, f.line) for f in r2.findings
    ]


def test_unicode_code_handled():
    code = "# héllo VIBE_SSRF_SINK こんにちは\n"
    reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="low")
    report = reviewer.review_file("x.py", code=code)
    assert any(f.cwe == "CWE-918" for f in report.findings)


def test_path_does_not_exist_raises():
    reviewer = VibeCodeReviewer(stub_judge_fn, min_severity="low")
    with pytest.raises(FileNotFoundError):
        reviewer.review_file("/nonexistent/definitely/not/here.py")


def test_vibefinding_and_report_dataclass_shape():
    f = VibeFinding(
        why_real="x", fix_hint="y", cwe="CWE-1", severity="high",
        line=1, snippet="s", file_path="p",
    )
    assert f.severity == "high"
    r = VibeReviewReport(
        file_path="p", findings=[f], total_lines=1, elapsed_s=0.0, dropped=0,
    )
    assert r.findings[0].cwe == "CWE-1"
