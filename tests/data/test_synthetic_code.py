"""Tests for src/data/synthetic_code.py"""
import pytest

from src.data.synthetic_code import (
    CodeExample,
    MutationConfig,
    count_cyclomatic_complexity,
    classify_difficulty,
    extract_function_names,
    extract_imports,
    rename_variables,
    add_type_hints,
    add_docstring,
    apply_mutations,
    build_oss_instruct_prompt,
    parse_test_cases_from_text,
    generate_test_cases,
    SyntheticCodePipeline,
    SEED_SNIPPETS,
)


# ---------------------------------------------------------------------------
# Mock generate_fn
# ---------------------------------------------------------------------------

def mock_generate(prompt: str) -> str:
    return (
        "Write a Python function that processes data and returns a result.\n"
        "def test_example():\n    assert add(1, 2) == 3\n\n"
        "def test_edge():\n    assert add(0, 0) == 0\n"
    )


# ---------------------------------------------------------------------------
# Complexity heuristics
# ---------------------------------------------------------------------------

def test_complexity_simple_function():
    code = "def f():\n    pass\n"
    assert count_cyclomatic_complexity(code) == 1


def test_complexity_with_branches():
    code = "def f(x):\n    if x > 0:\n        for i in range(x):\n            pass\n"
    # 1 + if + for = 3
    assert count_cyclomatic_complexity(code) == 3


def test_classify_easy():
    code = "def f():\n    return 1\n"
    assert classify_difficulty(code) == "easy"


def test_classify_hard():
    # 1 + if + elif + else(implicit) + for + while + try + except + and + or = 9
    code = (
        "def f(x):\n"
        "    if x > 0:\n"
        "        if x > 10:\n"
        "            for i in range(x):\n"
        "                while i > 0:\n"
        "                    try:\n"
        "                        if i and x or x:\n"
        "                            pass\n"
        "                    except:\n"
        "                        pass\n"
    )
    assert classify_difficulty(code) == "hard"


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------

def test_extract_function_names_simple():
    code = "def foo():\n    pass\ndef bar():\n    pass\n"
    names = extract_function_names(code)
    assert "foo" in names
    assert "bar" in names


def test_extract_function_names_empty_on_error():
    names = extract_function_names("def (broken syntax!!!")
    assert names == []


def test_extract_imports_simple():
    code = "import os\nimport sys\n"
    imports = extract_imports(code)
    assert "os" in imports
    assert "sys" in imports


def test_extract_imports_empty_on_error():
    imports = extract_imports("import !!!broken")
    assert imports == []


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def test_rename_variables_with_mapping():
    code = "x = 1\ny = x + 2\n"
    result = rename_variables(code, mapping={"x": "alpha", "y": "beta"})
    assert "alpha" in result
    assert "beta" in result


def test_add_type_hints_inserts_return_type():
    code = "def foo(a, b):\n    return a + b\n"
    result = add_type_hints(code)
    assert "-> None" in result


def test_add_type_hints_preserves_existing():
    code = "def foo(a: int) -> int:\n    return a\n"
    result = add_type_hints(code)
    # Should not add another return annotation
    assert result.count("->") == 1


def test_add_docstring_inserts_after_def():
    code = "def foo():\n    return 1\n"
    result = add_docstring(code, "My docstring.")
    assert "My docstring." in result


def test_apply_mutations_runs_without_error():
    code = SEED_SNIPPETS[0]
    config = MutationConfig()
    result = apply_mutations(code, config)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Instruction / prompt generation
# ---------------------------------------------------------------------------

def test_build_oss_instruct_prompt_nonempty():
    prompt = build_oss_instruct_prompt("def foo(): pass", "python")
    assert len(prompt) > 0
    assert "python" in prompt.lower()


def test_build_oss_instruct_prompt_contains_code():
    seed = "def foo(): pass"
    prompt = build_oss_instruct_prompt(seed, "python")
    assert "foo" in prompt


# ---------------------------------------------------------------------------
# Test case parsing
# ---------------------------------------------------------------------------

def test_parse_test_cases_extracts_def_test_blocks():
    text = (
        "def test_one():\n    assert add(1, 2) == 3\n\n"
        "def test_two():\n    assert add(0, 0) == 0\n"
    )
    cases = parse_test_cases_from_text(text)
    assert len(cases) >= 1
    assert any("test_one" in c for c in cases)


def test_generate_test_cases_returns_list():
    cases = generate_test_cases("add", mock_generate, n_cases=2)
    assert isinstance(cases, list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def test_pipeline_run_returns_code_examples():
    pipeline = SyntheticCodePipeline(generate_fn=mock_generate, seed_snippets=SEED_SNIPPETS[:3])
    examples = pipeline.run(n_examples=3)
    assert isinstance(examples, list)
    assert all(isinstance(e, CodeExample) for e in examples)


def test_pipeline_run_returns_correct_count():
    pipeline = SyntheticCodePipeline(generate_fn=mock_generate, seed_snippets=SEED_SNIPPETS[:5])
    examples = pipeline.run(n_examples=4)
    assert len(examples) == 4


def test_pipeline_export_returns_dicts():
    pipeline = SyntheticCodePipeline(generate_fn=mock_generate, seed_snippets=SEED_SNIPPETS[:2])
    examples = pipeline.run(n_examples=2)
    exported = pipeline.export(examples)
    assert all(isinstance(d, dict) for d in exported)
    assert all("instruction" in d for d in exported)


def test_pipeline_deduplicate_removes_duplicates():
    pipeline = SyntheticCodePipeline(generate_fn=mock_generate)
    ex = CodeExample(instruction="Do X", code="def foo(): pass")
    duplicate = CodeExample(instruction="Do Y", code="def foo(): pass")
    unique = pipeline.deduplicate([ex, duplicate, ex])
    assert len(unique) == 1
