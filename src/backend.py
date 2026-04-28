"""Aurelius backend runner.

This entrypoint starts the OpenAI-compatible inference server from
``src.serving.api_server`` instead of the cockpit-oriented FastAPI app.
That keeps the default backend contract aligned with the rest of the
integration surface.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

from src.serving.api_server import _load_auth_from_env

logger = logging.getLogger(__name__)


def _build_generate_fn(
    model_path: str | None,
    max_tokens: int,
    temperature: float,
):
    """Load a real model if requested, otherwise fall back to the mock server."""

    from src.serving.api_server import make_mock_generate_fn

    if not model_path:
        return make_mock_generate_fn()

    try:
        from src.cli.main import _load_generate_fn

        return _load_generate_fn(model_path, max_tokens, temperature)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning(
            "Could not load model from %s (%s); falling back to mock completions",
            model_path,
            exc,
        )
        return make_mock_generate_fn()


def _validate_host_auth(host: str) -> None:
    """Abort if non-loopback binding lacks configured auth."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        auth_mw = _load_auth_from_env()
        if not auth_mw.is_configured:
            logger.error(
                "Aborting: non-loopback host %s requires at least one configured API key "
                "(set AURELIUS_API_KEYS env var).",
                host,
            )
            sys.exit(1)


def create_api_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    model_path: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    """Create the OpenAI-compatible Aurelius API server."""

    from src.serving.api_server import create_server

    _validate_host_auth(host)
    auth_mw = _load_auth_from_env()

    generate_fn = _build_generate_fn(model_path, max_tokens, temperature)
    server = create_server(
        host,
        port,
        generate_fn,
        auth_middleware=auth_mw,
    )
    server._started_at = time.time()
    server._is_mock = model_path is None
    server.model_id = os.environ.get("AURELIUS_MODEL_ID", "aurelius")
    return server


def start_api(
    host: str = "127.0.0.1",
    port: int = 8080,
    reload: bool = False,
    *,
    model_path: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> None:
    """Start the Aurelius API server.

    ``reload`` is accepted for CLI compatibility, but it is not supported by
    the underlying ``HTTPServer`` implementation.
    """

    if reload:
        logger.warning("reload is ignored by the HTTPServer-based API server")
    server = create_api_server(
        host,
        port,
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    server.serve_forever()


def start_shell() -> None:
    """Start the Aurelius interactive shell."""

    from src.cli.aurelius_shell import AureliusShell

    shell = AureliusShell()
    shell.run()


def start_all(
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    model_path: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> None:
    """Start the API server in the background, then launch the shell."""

    thread = threading.Thread(
        target=start_api,
        kwargs={
            "host": host,
            "port": port,
            "model_path": model_path,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        daemon=True,
    )
    thread.start()
    start_shell()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurelius Backend")
    parser.add_argument("mode", nargs="?", default="shell", choices=["shell", "api", "all"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.mode == "api":
        start_api(
            args.host,
            args.port,
            args.reload,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.mode == "all":
        start_all(
            args.host,
            args.port,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        start_shell()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
