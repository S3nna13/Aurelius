"""Unit tests for :mod:`src.agent.repo_context_packer`."""

from __future__ import annotations

import os
import time

import pytest

from src.agent.repo_context_packer import (
    FileSnippet,
    RepoContext,
    RepoContextPacker,
)


def _mkfile(root, rel, content):
    path = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def test_invalid_repo_root_raises(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(ValueError):
        RepoContextPacker(str(missing))


def test_build_tree_respects_max_depth(tmp_path):
    _mkfile(tmp_path, "a/b/c/deep.py", "x = 1\n")
    _mkfile(tmp_path, "top.py", "y = 2\n")
    p = RepoContextPacker(str(tmp_path))
    t0 = p.build_tree(max_depth=0)
    t1 = p.build_tree(max_depth=1)
    t3 = p.build_tree(max_depth=4)
    assert "top.py" not in t0
    assert "top.py" in t1
    assert "a/" in t1
    # depth 1 shows 'a/' as a directory but not its children.
    assert "b/" not in t1
    assert "b/" in t3
    assert "deep.py" in t3


def test_build_tree_excludes_exclude_dirs(tmp_path):
    _mkfile(tmp_path, "src/main.py", "pass\n")
    _mkfile(tmp_path, ".git/config", "x\n")
    _mkfile(tmp_path, "node_modules/foo/index.js", "x\n")
    p = RepoContextPacker(str(tmp_path))
    tree = p.build_tree(max_depth=4)
    assert ".git" not in tree
    assert "node_modules" not in tree
    assert "main.py" in tree


def test_pack_returns_repo_context_within_budget(tmp_path):
    _mkfile(tmp_path, "a.py", "def foo():\n    return 1\n" * 5)
    _mkfile(tmp_path, "b.py", "def bar():\n    return 2\n" * 5)
    p = RepoContextPacker(str(tmp_path), max_tokens=500)
    ctx = p.pack("foo")
    assert isinstance(ctx, RepoContext)
    assert ctx.token_estimate <= 500
    assert isinstance(ctx.snippets, list)


def test_pack_selects_top_scoring_file_for_targeted_query(tmp_path):
    _mkfile(tmp_path, "math_utils.py", "def calculate_total(xs):\n    return sum(xs)\n")
    _mkfile(tmp_path, "unrelated.py", "def greet(name):\n    return 'hi ' + name\n")
    _mkfile(tmp_path, "readme.md", "# project readme\n")
    p = RepoContextPacker(str(tmp_path), max_tokens=2000)
    ctx = p.pack("calculate_total")
    assert len(ctx.snippets) >= 1
    assert ctx.snippets[0].path == "math_utils.py"


def test_extract_imports_python(tmp_path):
    p = RepoContextPacker(str(tmp_path))
    src = (
        "import os\n"
        "import sys, json\n"
        "from collections import defaultdict\n"
        "from .local import thing\n"
    )
    imports = p.extract_imports(src, language="python")
    assert "os" in imports
    assert "sys" in imports
    assert "json" in imports
    assert "collections" in imports


def test_extensions_filter_respected(tmp_path):
    _mkfile(tmp_path, "keep.py", "def calculate(): pass\n")
    _mkfile(tmp_path, "skip.txt", "calculate is here\n")
    p = RepoContextPacker(str(tmp_path), extensions=(".py",))
    ctx = p.pack("calculate")
    paths = [s.path for s in ctx.snippets]
    assert "keep.py" in paths
    assert "skip.txt" not in paths


def test_empty_repo_returns_empty_context(tmp_path):
    p = RepoContextPacker(str(tmp_path))
    ctx = p.pack("anything")
    assert ctx.tree == ""
    assert ctx.snippets == []
    assert ctx.token_estimate == 0


def test_oov_query_returns_empty_snippets(tmp_path):
    _mkfile(tmp_path, "a.py", "def foo(): return 1\n")
    p = RepoContextPacker(str(tmp_path))
    ctx = p.pack("zzzzzzqqqqxyzzzz_never_occurs")
    assert ctx.snippets == []


def test_determinism(tmp_path):
    _mkfile(tmp_path, "a.py", "def calculate(x): return x * 2\n")
    _mkfile(tmp_path, "b.py", "def other(): pass\n")
    p = RepoContextPacker(str(tmp_path))
    ctx1 = p.pack("calculate")
    ctx2 = p.pack("calculate")
    assert ctx1.tree == ctx2.tree
    assert [s.path for s in ctx1.snippets] == [s.path for s in ctx2.snippets]
    assert [s.content for s in ctx1.snippets] == [s.content for s in ctx2.snippets]
    assert ctx1.token_estimate == ctx2.token_estimate


def test_respects_max_tokens_budget(tmp_path):
    big = "def calculate(x):\n    return x\n" * 200
    _mkfile(tmp_path, "big.py", big)
    _mkfile(tmp_path, "other.py", "def calculate_other(): pass\n")
    p = RepoContextPacker(str(tmp_path), max_tokens=200)
    ctx = p.pack("calculate")
    assert ctx.token_estimate <= 200


def test_custom_token_counter_honored(tmp_path):
    _mkfile(tmp_path, "a.py", "def calculate(x): return x\n")
    calls = {"n": 0}

    def counter(text: str) -> int:
        calls["n"] += 1
        return len(text)  # 1 token per char

    p = RepoContextPacker(str(tmp_path), token_counter=counter, max_tokens=10_000)
    ctx = p.pack("calculate")
    assert calls["n"] > 0
    # With char-counter, token_estimate should be substantially larger than
    # the default /4 heuristic would produce.
    assert ctx.token_estimate > 10


def test_large_repo_packs_quickly(tmp_path):
    for i in range(200):
        _mkfile(tmp_path, f"pkg{i // 20}/mod_{i}.py", f"def fn_{i}(x):\n    return x + {i}\n")
    p = RepoContextPacker(str(tmp_path), max_tokens=4000)
    start = time.perf_counter()
    ctx = p.pack("fn_17")
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"pack() took {elapsed:.2f}s (expected <2s)"
    assert isinstance(ctx, RepoContext)


def test_binary_file_skipped(tmp_path):
    bin_path = os.path.join(str(tmp_path), "blob.py")
    with open(bin_path, "wb") as f:
        f.write(b"\x00\x01\x02" * 100 + b"\xff\xfe")
    _mkfile(tmp_path, "ok.py", "def calculate(): return 42\n")
    p = RepoContextPacker(str(tmp_path))
    ctx = p.pack("calculate")
    paths = [s.path for s in ctx.snippets]
    assert "blob.py" not in paths
    assert "ok.py" in paths


def test_file_snippet_dataclass_fields():
    s = FileSnippet(path="x.py", content="abc", score=1.25, lines_selected=(1, 3))
    assert s.path == "x.py"
    assert s.score == 1.25
    assert s.lines_selected == (1, 3)


def test_extract_imports_js(tmp_path):
    p = RepoContextPacker(str(tmp_path))
    src = "import React from 'react';\nconst fs = require('fs');\n"
    imports = p.extract_imports(src, language="js")
    assert "react" in imports
    assert "fs" in imports
