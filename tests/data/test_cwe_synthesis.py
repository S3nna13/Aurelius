"""Tests for src/data/cwe_synthesis.py -- synthetic CWE pair generator.

These tests NEVER execute generated templates. They only verify string-level
shape/content properties.
"""

from __future__ import annotations

import pytest

from src.data.cwe_synthesis import (
    CWE_CATALOG,
    CWERecipe,
    CWESyntheticGenerator,
)

REQUIRED_CWES = {
    "CWE-78",
    "CWE-89",
    "CWE-79",
    "CWE-22",
    "CWE-798",
    "CWE-327",
    "CWE-502",
    "CWE-94",
}


def test_catalog_has_at_least_ten_entries():
    assert len(CWE_CATALOG) >= 10


def test_required_cwes_present():
    ids = {r.cwe_id for r in CWE_CATALOG}
    missing = REQUIRED_CWES - ids
    assert not missing, f"missing required CWEs: {missing}"


def test_all_recipes_well_formed():
    seen_ids: set[str] = set()
    for r in CWE_CATALOG:
        assert isinstance(r, CWERecipe)
        assert r.cwe_id.startswith("CWE-")
        assert r.cwe_id not in seen_ids, f"duplicate {r.cwe_id}"
        seen_ids.add(r.cwe_id)
        assert r.name
        assert r.description
        assert r.persona_hints and all(isinstance(h, str) and h for h in r.persona_hints)
        assert r.vulnerable_template
        assert r.secure_template
        assert r.vulnerable_template != r.secure_template, r.cwe_id


def test_render_produces_both_variants():
    gen = CWESyntheticGenerator(rng_seed=1)
    for r in CWE_CATALOG:
        v, s = gen.render(r)
        assert v and s
        assert v != s
        # Assert the chosen placeholder values for non-arg keys appear in output.
        # (Rendered f-string braces like {address} in generated source are fine.)
        # Sanity: literal template markers {func}, {var}, {env} should be gone.
        for marker in ("{func}", "{var}", "{env}"):
            assert marker not in v, (r.cwe_id, marker, v)
            assert marker not in s, (r.cwe_id, marker, s)


def test_placeholder_substitution_varies_output():
    """Different placeholder choices -> different rendered outputs."""
    recipe = next(r for r in CWE_CATALOG if r.cwe_id == "CWE-78")
    gen = CWESyntheticGenerator(rng_seed=7)
    outputs = {gen.render(recipe)[0] for _ in range(20)}
    assert len(outputs) >= 2


def test_determinism_under_seed():
    a = CWESyntheticGenerator(rng_seed=42).generate_batch(8)
    b = CWESyntheticGenerator(rng_seed=42).generate_batch(8)
    assert a == b


def test_generate_pair_shape():
    gen = CWESyntheticGenerator(rng_seed=3)
    pair = gen.generate_pair()
    for key in (
        "cwe_id",
        "vulnerable_code",
        "secure_code",
        "rationale",
        "persona_hint",
        "chosen_placeholders",
    ):
        assert key in pair
    assert pair["cwe_id"].startswith("CWE-")
    assert pair["vulnerable_code"] != pair["secure_code"]
    assert isinstance(pair["chosen_placeholders"], dict)


def test_generate_batch_length():
    gen = CWESyntheticGenerator(rng_seed=5)
    out = gen.generate_batch(12)
    assert len(out) == 12
    gen2 = CWESyntheticGenerator(rng_seed=5)
    assert gen2.generate_batch(0) == []


def test_cwe_ids_filter_works():
    gen = CWESyntheticGenerator(rng_seed=11)
    batch = gen.generate_batch(20, cwe_ids=["CWE-89", "CWE-22"])
    assert {p["cwe_id"] for p in batch} <= {"CWE-89", "CWE-22"}
    assert len(batch) == 20


def test_unknown_cwe_id_raises():
    gen = CWESyntheticGenerator(rng_seed=0)
    with pytest.raises(KeyError):
        gen.generate_pair(cwe_id="CWE-999999")
    with pytest.raises(KeyError):
        gen.generate_batch(3, cwe_ids=["CWE-does-not-exist"])


def test_empty_placeholders_handled():
    recipe = CWERecipe(
        cwe_id="CWE-TEST",
        name="t",
        description="d",
        persona_hints=("h",),
        vulnerable_template="VULN_BODY\n",
        secure_template="SAFE_BODY\n",
        placeholders={},
    )
    gen = CWESyntheticGenerator(catalog=(recipe,), rng_seed=0)
    v, s = gen.render(recipe)
    assert v == "VULN_BODY\n"
    assert s == "SAFE_BODY\n"
    pair = gen.generate_pair()
    assert pair["chosen_placeholders"] == {}


def test_rationale_non_empty_and_mentions_cwe():
    gen = CWESyntheticGenerator(rng_seed=2)
    for r in CWE_CATALOG:
        pair = gen.generate_pair(cwe_id=r.cwe_id)
        assert pair["rationale"]
        assert r.cwe_id in pair["rationale"]


def test_templates_differ_per_recipe():
    for r in CWE_CATALOG:
        assert r.vulnerable_template.strip() != r.secure_template.strip(), r.cwe_id


def test_rendering_does_not_execute_templates():
    """Critical safety: render() returns strings, never runs them."""
    gen = CWESyntheticGenerator(rng_seed=0)
    # CWE-94 contains a dynamic-eval call as TEXT. Verify it is text.
    recipe = next(r for r in CWE_CATALOG if r.cwe_id == "CWE-94")
    v, s = gen.render(recipe)
    assert isinstance(v, str)
    assert isinstance(s, str)
    # Text should reference the dynamic-eval primitive but NOT raise.
    assert "eval" in v
    # The secure counterpart points at ast.literal_eval.
    assert "ast.literal_eval" in s


def test_no_foreign_imports_in_generated_code():
    """Generated templates must not import any banned ML frameworks."""
    banned = (
        "transformers",
        "einops",
        "trl",
        "xformers",
        "flash_attn",
        "bitsandbytes",
        "peft",
        "diffusers",
        "datasets",
        "accelerate",
        "deepspeed",
        "langchain",
        "llamaindex",
    )
    gen = CWESyntheticGenerator(rng_seed=9)
    batch = gen.generate_batch(40)
    for pair in batch:
        blob = pair["vulnerable_code"] + "\n" + pair["secure_code"]
        for name in banned:
            assert f"import {name}" not in blob
            assert f"from {name}" not in blob


def test_render_override_wins():
    gen = CWESyntheticGenerator(rng_seed=0)
    recipe = next(r for r in CWE_CATALOG if r.cwe_id == "CWE-89")
    v, s = gen.render(
        recipe,
        placeholders_override={
            "func": "FIXED_FUNC",
            "arg": "FIXED_ARG",
        },
    )
    assert "FIXED_FUNC" in v and "FIXED_ARG" in v
    assert "FIXED_FUNC" in s and "FIXED_ARG" in s


def test_empty_catalog_rejected():
    with pytest.raises(ValueError):
        CWESyntheticGenerator(catalog=(), rng_seed=0)


def test_negative_batch_rejected():
    gen = CWESyntheticGenerator(rng_seed=0)
    with pytest.raises(ValueError):
        gen.generate_batch(-1)
