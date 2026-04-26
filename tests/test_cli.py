"""
tests/test_cli.py

Smoke tests for the aurelius CLI entry point.
"""

import pytest

from src.cli.main import DEFAULT_SYSTEM, __version__, _build_parser, main


def test_version_string():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_parser_creates_subcommands():
    parser = _build_parser()
    assert parser is not None


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


def test_help_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "chat" in captured.out
    assert "serve" in captured.out


def test_chat_help(capsys):
    with pytest.raises(SystemExit):
        main(["chat", "--help"])
    captured = capsys.readouterr()
    assert "system" in captured.out.lower() or "model" in captured.out.lower()


def test_serve_help(capsys):
    with pytest.raises(SystemExit):
        main(["serve", "--help"])
    captured = capsys.readouterr()
    assert "port" in captured.out.lower()


def test_eval_no_checkpoint(capsys):
    result = main(["eval"])
    assert result == 1


def test_default_system_nonempty():
    assert len(DEFAULT_SYSTEM) > 10


def test_mock_generate():
    from src.cli.main import _mock_generate

    result = _mock_generate("<|user|>\nhello<|end|>\n<|assistant|>\n")
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_parser_returns_parser():
    import argparse

    parser = _build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
