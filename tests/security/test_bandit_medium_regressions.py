"""Regression tests for medium-severity Bandit findings normalized in CI."""

from __future__ import annotations

import pytest

from aurelius_cli.pipeline_commands import _compile_expr
from gateway.aurelius_api import _resolve_tokenizer_revision
from gateway.native_tools import SandboxedFilesystem


def test_pipeline_expression_supports_common_json_transforms() -> None:
    fn = _compile_expr('x["age"] >= 18 and x["name"].upper().startswith("A")')

    assert fn({"age": 21, "name": "ada"}) is True
    assert fn({"age": 16, "name": "ada"}) is False


def test_pipeline_expression_supports_lambda_form_without_eval() -> None:
    fn = _compile_expr('lambda x: x.get("score", 0) + 2')

    assert fn({"score": 3}) == 5


def test_pipeline_expression_blocks_import_escape() -> None:
    fn = _compile_expr('__import__("os").system("id")')

    with pytest.raises(ValueError, match="(function|method) .* is not allowed"):
        fn({})


def test_remote_tokenizer_requires_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURELIUS_TOKENIZER_REVISION", raising=False)
    monkeypatch.delenv("AURELIUS_MODEL_REVISION", raising=False)

    with pytest.raises(RuntimeError, match="require .*REVISION"):
        _resolve_tokenizer_revision("org/model")


def test_tokenizer_revision_uses_explicit_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURELIUS_TOKENIZER_REVISION", "0123456789abcdef")

    assert _resolve_tokenizer_revision("org/model") == "0123456789abcdef"


def test_local_tokenizer_path_does_not_require_revision(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert _resolve_tokenizer_revision(str(model_dir)) is None


def test_sandbox_filesystem_default_root_uses_tempdir() -> None:
    fs = SandboxedFilesystem()

    assert fs.root.name == "aurelius_sandbox"
