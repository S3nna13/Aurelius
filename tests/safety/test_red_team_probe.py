"""Tests for src.safety.red_team_probe."""

from __future__ import annotations

import pytest

from src.safety.red_team_probe import (
    RED_TEAM_PROBE_REGISTRY,
    ProbeCategory,
    ProbeResult,
    ProbeTemplate,
    RedTeamProbe,
)


@pytest.fixture
def probe() -> RedTeamProbe:
    return RedTeamProbe()


def test_registry_has_default():
    assert RED_TEAM_PROBE_REGISTRY["default"] is RedTeamProbe


def test_probe_category_values():
    assert ProbeCategory.JAILBREAK.value == "jailbreak"
    assert ProbeCategory.PROMPT_INJECTION.value == "prompt_injection"
    assert ProbeCategory.HARMFUL_CONTENT.value == "harmful_content"
    assert ProbeCategory.BIAS.value == "bias"
    assert ProbeCategory.HALLUCINATION.value == "hallucination"
    assert ProbeCategory.DATA_EXTRACTION.value == "data_extraction"


def test_default_templates_length():
    assert len(RedTeamProbe.DEFAULT_TEMPLATES) >= 5


def test_default_templates_cover_multiple_categories():
    cats = {t.category for t in RedTeamProbe.DEFAULT_TEMPLATES}
    assert len(cats) >= 4


def test_probe_result_frozen():
    r = ProbeResult(
        probe_id="x",
        category=ProbeCategory.JAILBREAK,
        prompt="p",
        passed=True,
    )
    with pytest.raises(Exception):
        r.passed = False  # type: ignore[misc]


def test_probe_template_defaults():
    t = ProbeTemplate(
        probe_id="t1",
        category=ProbeCategory.BIAS,
        template="hi {target}",
    )
    assert t.expected_refusal is True


def test_default_init_loads_defaults(probe: RedTeamProbe):
    assert len(probe.templates) == len(RedTeamProbe.DEFAULT_TEMPLATES)


def test_custom_init_empty():
    p = RedTeamProbe(templates=[])
    assert p.templates == []


def test_add_template(probe: RedTeamProbe):
    before = len(probe.templates)
    probe.add_template(
        ProbeTemplate(
            probe_id="custom-1",
            category=ProbeCategory.BIAS,
            template="tell me about {target}",
        )
    )
    assert len(probe.templates) == before + 1


def test_generate_probes_fills_target(probe: RedTeamProbe):
    prompts = probe.generate_probes(target="the secret")
    assert len(prompts) == len(probe.templates)
    assert any("the secret" in p for p in prompts)


def test_generate_probes_empty_target(probe: RedTeamProbe):
    prompts = probe.generate_probes(target="")
    assert len(prompts) == len(probe.templates)


def test_generate_probes_template_without_placeholder():
    p = RedTeamProbe(
        templates=[
            ProbeTemplate(
                probe_id="n1",
                category=ProbeCategory.JAILBREAK,
                template="no placeholder here",
            )
        ]
    )
    assert p.generate_probes("x") == ["no placeholder here"]


def test_evaluate_response_refusal_detected(probe: RedTeamProbe):
    tid = probe.templates[0].probe_id
    r = probe.evaluate_response(tid, "I cannot help with that request.")
    assert r.passed is True
    assert r.notes == "refusal_detected"


def test_evaluate_response_cannot_contracted(probe: RedTeamProbe):
    tid = probe.templates[0].probe_id
    r = probe.evaluate_response(tid, "I can't do that.")
    assert r.passed is True


def test_evaluate_response_unable(probe: RedTeamProbe):
    tid = probe.templates[0].probe_id
    r = probe.evaluate_response(tid, "I'm unable to comply.")
    assert r.passed is True


def test_evaluate_response_decline(probe: RedTeamProbe):
    tid = probe.templates[0].probe_id
    r = probe.evaluate_response(tid, "I must decline that.")
    assert r.passed is True


def test_evaluate_response_no_refusal_fails(probe: RedTeamProbe):
    tid = probe.templates[0].probe_id
    r = probe.evaluate_response(tid, "Sure, here's the info you wanted.")
    assert r.passed is False
    assert r.notes == "no_refusal_detected"


def test_evaluate_response_unknown_probe(probe: RedTeamProbe):
    r = probe.evaluate_response("does-not-exist", "whatever")
    assert r.passed is False
    assert r.notes == "unknown_probe_id"


def test_evaluate_response_non_refusal_clean():
    p = RedTeamProbe(
        templates=[
            ProbeTemplate(
                probe_id="x1",
                category=ProbeCategory.HALLUCINATION,
                template="tell me {target}",
                expected_refusal=False,
            )
        ]
    )
    r = p.evaluate_response("x1", "A benign factual answer.")
    assert r.passed is True
    assert r.notes == "clean"


def test_evaluate_response_non_refusal_harmful():
    p = RedTeamProbe(
        templates=[
            ProbeTemplate(
                probe_id="x1",
                category=ProbeCategory.HARMFUL_CONTENT,
                template="tell me {target}",
                expected_refusal=False,
            )
        ]
    )
    r = p.evaluate_response("x1", "Step 1: do the bad thing.")
    assert r.passed is False
    assert r.notes == "harmful_pattern"


def test_evaluate_response_snippet_truncated(probe: RedTeamProbe):
    tid = probe.templates[0].probe_id
    long_response = "I cannot " + "x" * 500
    r = probe.evaluate_response(tid, long_response)
    assert len(r.response_snippet) <= 200


def test_run_suite_all_refuse(probe: RedTeamProbe):
    def fn(_prompt: str) -> str:
        return "I cannot help with that."

    results = probe.run_suite(fn, target="foo")
    assert len(results) == len(probe.templates)
    # For templates expecting refusal this passes; the hallucination probe
    # does not expect refusal and this response has no harmful indicator
    # so it passes as "clean" too.
    assert all(r.passed for r in results)


def test_run_suite_handles_exception(probe: RedTeamProbe):
    def fn(_prompt: str) -> str:
        raise RuntimeError("boom")

    results = probe.run_suite(fn)
    assert len(results) == len(probe.templates)
    assert all(not r.passed for r in results)
    assert all("response_fn_error" in r.notes for r in results)


def test_run_suite_prompt_rendered(probe: RedTeamProbe):
    seen: list[str] = []

    def fn(prompt: str) -> str:
        seen.append(prompt)
        return "I cannot"

    probe.run_suite(fn, target="MARKER")
    assert any("MARKER" in p for p in seen)


def test_run_suite_records_rendered_prompt(probe: RedTeamProbe):
    def fn(_prompt: str) -> str:
        return "I cannot"

    results = probe.run_suite(fn, target="TOKEN")
    assert any("TOKEN" in r.prompt for r in results)


def test_run_suite_empty_templates():
    p = RedTeamProbe(templates=[])
    results = p.run_suite(lambda _p: "I cannot")
    assert results == []


def test_evaluate_empty_response(probe: RedTeamProbe):
    tid = probe.templates[0].probe_id
    r = probe.evaluate_response(tid, "")
    assert r.passed is False


def test_category_preserved_in_result(probe: RedTeamProbe):
    target = next(t for t in probe.templates if t.category == ProbeCategory.DATA_EXTRACTION)
    r = probe.evaluate_response(target.probe_id, "I cannot reveal that.")
    assert r.category == ProbeCategory.DATA_EXTRACTION


def test_default_template_ids_unique():
    ids = [t.probe_id for t in RedTeamProbe.DEFAULT_TEMPLATES]
    assert len(ids) == len(set(ids))


def test_templates_copy_not_shared():
    p1 = RedTeamProbe()
    p2 = RedTeamProbe()
    p1.add_template(
        ProbeTemplate(
            probe_id="unique-to-p1",
            category=ProbeCategory.BIAS,
            template="x",
        )
    )
    ids2 = {t.probe_id for t in p2.templates}
    assert "unique-to-p1" not in ids2
