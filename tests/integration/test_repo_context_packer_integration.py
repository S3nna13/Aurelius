"""Integration test: repo_context_packer exposed via :mod:`src.agent`."""

from __future__ import annotations

import os

import src.agent as agent


def _mkfile(root, rel, content):
    path = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def test_repo_context_packer_surface_exports():
    assert hasattr(agent, "RepoContextPacker")
    assert hasattr(agent, "RepoContext")
    assert hasattr(agent, "FileSnippet")


def test_pack_toy_repo_three_py_files(tmp_path):
    _mkfile(
        tmp_path,
        "totals.py",
        "def calculate_total(items):\n"
        "    '''Return the sum of items.'''\n"
        "    return sum(items)\n",
    )
    _mkfile(
        tmp_path,
        "greet.py",
        "def say_hello(name):\n    return f'hello {name}'\n",
    )
    _mkfile(
        tmp_path,
        "io_utils.py",
        "import os\n\n"
        "def read_text(path):\n"
        "    with open(path) as f:\n"
        "        return f.read()\n",
    )

    packer = agent.RepoContextPacker(str(tmp_path), max_tokens=2000)
    ctx = packer.pack("calculate")

    assert isinstance(ctx, agent.RepoContext)
    assert ctx.snippets, "expected at least one snippet for query 'calculate'"
    assert ctx.snippets[0].path == "totals.py"
    # imports_summary keyed by relative path.
    assert "totals.py" in ctx.imports_summary
    # Tree shows all three files.
    assert "totals.py" in ctx.tree
    assert "greet.py" in ctx.tree
    assert "io_utils.py" in ctx.tree
    assert ctx.token_estimate <= 2000
