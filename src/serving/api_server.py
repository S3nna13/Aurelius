"""
OpenAI-compatible HTTP API server for Aurelius.

Run: python -m src.serving.api_server --port 8080

Endpoints:
  POST /v1/chat/completions   — chat completion (streaming or non-streaming)
  GET  /v1/models             — list available models
  GET  /health                — health check (simple)
  GET  /healthz               — liveness probe
  GET  /readyz                — readiness probe
  GET  /openapi.json          — OpenAPI 3.1 specification
  GET  /docs                  — Swagger UI documentation
"""

import hashlib
import json
import logging
import math
import os
import signal
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

from .auth_middleware import AuthMiddleware
from .cors_middleware import CORS
from .metrics_middleware import METRICS
from .openapi_spec import openapi_spec, render_docs_page
from .rate_limiter import RateLimiterChain, TokenBucketLimiter
from .request_coalescer import RequestCoalescer

logger = logging.getLogger(__name__)

#: Maximum allowed request body size (1 MiB) to prevent memory-exhaustion DoS.
_MAX_CONTENT_LENGTH = 1_048_576
#: Maximum number of messages per request.
_MAX_MESSAGES = 1_024
#: Maximum characters per message content.
_MAX_MESSAGE_CHARS = 65_536


@dataclass
class ChatRequest:
    model: str
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False
    system: str | None = None


@dataclass
class ChatResponse:
    id: str
    object: str
    created: int
    model: str
    choices: list[dict]
    usage: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
            "usage": self.usage,
        }


def _check_auth_and_rate_limit(handler: BaseHTTPRequestHandler) -> bool:
    """Return True if the request may proceed.

    Sends 401 or 429 response and returns *False* when blocked.
    """
    server = handler.server
    auth_mw = getattr(server, "auth_middleware", None)
    rate_limiter = getattr(server, "rate_limiter", None)

    auth_result = None
    if auth_mw is not None:
        auth_result = auth_mw.authenticate(dict(handler.headers))
        if not auth_result.authenticated:
            handler._send_error(401, auth_result.error or "Unauthorized")
            return False

    if rate_limiter is not None:
        identifier = (
            auth_result.key_id
            if auth_result is not None and auth_result.key_id
            else handler.client_address[0]
        )
        if isinstance(rate_limiter, RateLimiterChain):
            result = rate_limiter.check_all(
                key=identifier,
                ip=handler.client_address[0],
                route=handler.path,
            )
        else:
            result = rate_limiter.check(identifier)

        if not result.allowed:
            handler.send_response(429)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Retry-After", str(int(result.retry_after_s) + 1))
            handler.end_headers()
            handler.wfile.write(
                json.dumps(
                    {"error": "Rate limit exceeded", "retry_after": result.retry_after_s}
                ).encode("utf-8")
            )
            return False

    return True


def _validate_chat_request(body: dict) -> ChatRequest:
    """Validate and construct a :class:`ChatRequest` from the raw JSON body.

    Enforces caps on message count, message size, temperature range, and
    max_tokens bounds to prevent CPU / memory exhaustion.
    """
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if len(messages) > _MAX_MESSAGES:
        raise ValueError(f"messages list exceeds maximum length ({_MAX_MESSAGES})")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be an object")
        content = msg.get("content")
        if content is not None and (
            not isinstance(content, str) or len(content) > _MAX_MESSAGE_CHARS
        ):
            raise ValueError(
                f"messages[{i}].content must be a string <= {_MAX_MESSAGE_CHARS} chars"
            )
        role = msg.get("role")
        if role is not None and (not isinstance(role, str) or not role.strip()):
            raise ValueError(f"messages[{i}].role must be a non-empty string")

    temperature = float(body.get("temperature", 0.7))
    if math.isnan(temperature) or math.isinf(temperature):
        raise ValueError("temperature must be a finite number")
    if not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")

    max_tokens = int(body.get("max_tokens", 512))
    if not (1 <= max_tokens <= 32_768):
        raise ValueError("max_tokens must be between 1 and 32768")

    return ChatRequest(
        model=str(body.get("model", "aurelius")),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=bool(body.get("stream", False)),
        system=body.get("system"),
    )


class AureliusRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.debug("%s - %s", self.address_string(), format % args)

    def send_response(self, code: int, message: str | None = None) -> None:
        super().send_response(code, message)
        CORS.add_headers(self)

    def do_OPTIONS(self):
        CORS.handle_preflight(self)

    def _record_metrics(self, start_time: float, status: int, error: str | None = None) -> None:
        latency = (time.perf_counter() - start_time) * 1000
        METRICS.record_request(
            method=self.command,
            path=self.path.split("?")[0],
            status=status,
            latency_ms=latency,
            error_type=error,
        )

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": "api_error"}})

    def _read_body(self) -> bytes:
        """Read the request body enforcing :data:`_MAX_CONTENT_LENGTH`."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc
        if content_length < 0:
            raise ValueError("Negative Content-Length")
        if content_length > _MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Content-Length {content_length} exceeds maximum {_MAX_CONTENT_LENGTH}"
            )
        return self.rfile.read(content_length)

    def _get_server_info(self) -> dict:
        uptime = 0.0
        started = getattr(self.server, "_started_at", None)
        if started is not None:
            uptime = time.time() - started
        version = os.environ.get("AURELIUS_VERSION", "0.1.0")
        try:
            import psutil
            mem = psutil.Process().memory_info()
            rss = mem.rss
        except ImportError:
            rss = 0
        return {
            "version": version,
            "uptime": round(uptime, 2),
            "memory": {"rss": rss},
        }

    def do_GET(self):
        start = time.perf_counter()
        if self.path == "/metrics":
            METRICS.connection_opened()
            try:
                text = METRICS.prometheus_text()
                body = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self._record_metrics(start, 200)
            finally:
                METRICS.connection_closed()
            return
        if self.path == "/health":
            info = self._get_server_info()
            self._send_json(200, {"status": "ok", **info})
            self._record_metrics(start, 200)
            return
        if self.path == "/healthz":
            METRICS.connection_opened()
            try:
                self._send_json(200, {"alive": True})
                self._record_metrics(start, 200)
            finally:
                METRICS.connection_closed()
            return
        if self.path == "/readyz":
            started = getattr(self.server, "_started_at", None)
            ready = started is not None
            status = 200 if ready else 503
            self._send_json(status, {"ready": ready})
            self._record_metrics(start, status)
            return
        if self.path == "/openapi.json":
            host = self.server.server_address[0] if self.server else "localhost"
            port = self.server.server_address[1] if self.server else 8080
            spec = openapi_spec(host, port)
            self._send_json(200, spec)
            self._record_metrics(start, 200)
            return
        if self.path == "/docs":
            host = self.server.server_address[0] if self.server else "localhost"
            port = self.server.server_address[1] if self.server else 8080
            page = render_docs_page(host, port)
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self._record_metrics(start, 200)
            return
        if not _check_auth_and_rate_limit(self):
            self._record_metrics(start, 401)
            return
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "aurelius",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "aurelius",
                        }
                    ],
                },
            )
            self._record_metrics(start, 200)
        else:
            self._send_error(404, "Not found")
            self._record_metrics(start, 404)

    def do_POST(self):
        start = time.perf_counter()
        if self.path != "/v1/chat/completions":
            self._send_error(404, "Not found")
            self._record_metrics(start, 404)
            return
        if not _check_auth_and_rate_limit(self):
            self._record_metrics(start, 401)
            return

        try:
            raw_body = self._read_body()
        except ValueError as exc:
            self._send_error(413, str(exc))
            self._record_metrics(start, 413)
            return

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            self._record_metrics(start, 400)
            return

        if "messages" not in body:
            self._send_error(400, "Missing required field: messages")
            self._record_metrics(start, 400)
            return

        try:
            request = _validate_chat_request(body)
        except ValueError as exc:
            self._send_error(400, f"Invalid request parameters: {exc}")
            self._record_metrics(start, 400)
            return

        def _generate() -> str:
            return self.server.generate_fn(request)

        coalescer = getattr(self.server, "coalescer", None)
        if coalescer is not None:
            # Only coalesce reasonably-sized bodies to avoid CPU-DoS from
            # sorting huge JSON objects for the hash key.
            try:
                if len(raw_body) <= 10_240:
                    coalesce_key = hashlib.sha256(
                        json.dumps(body, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                else:
                    coalesce_key = None
            except Exception:
                coalesce_key = None
            if coalesce_key is not None:
                try:
                    content = coalescer.coalesce(coalesce_key, _generate)
                except Exception:
                    logger.exception("generate_fn raised an exception")
                    self._send_error(500, "Internal server error")
                    self._record_metrics(start, 500, "generation_error")
                    return
            else:
                try:
                    content = _generate()
                except Exception:
                    logger.exception("generate_fn raised an exception")
                    self._send_error(500, "Internal server error")
                    self._record_metrics(start, 500, "generation_error")
                    return
        else:
            try:
                content = _generate()
            except Exception:
                logger.exception("generate_fn raised an exception")
                self._send_error(500, "Internal server error")
                self._record_metrics(start, 500, "generation_error")
                return

        prompt_tokens = sum(len(m.get("content", "").split()) for m in request.messages)
        completion_tokens = len(content.split())

        response = ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

        self._send_json(200, response.to_dict())
        self._record_metrics(start, 200)


class AureliusServer(HTTPServer):
    def __init__(
        self,
        host: str,
        port: int,
        generate_fn: Callable[["ChatRequest"], str],
        *,
        auth_middleware: AuthMiddleware | None = None,
        rate_limiter: TokenBucketLimiter | RateLimiterChain | None = None,
        coalescer: RequestCoalescer | None = None,
        bind_and_activate: bool = True,
    ):
        super().__init__(
            (host, port),
            AureliusRequestHandler,
            bind_and_activate=bind_and_activate,
        )
        self.generate_fn = generate_fn
        self.auth_middleware = auth_middleware
        self.rate_limiter = rate_limiter
        self.coalescer = coalescer


def create_server(
    host: str,
    port: int,
    generate_fn: Callable[["ChatRequest"], str],
    *,
    auth_middleware: AuthMiddleware | None = None,
    rate_limiter: TokenBucketLimiter | RateLimiterChain | None = None,
    coalescer: RequestCoalescer | None = None,
    bind_and_activate: bool = True,
) -> AureliusServer:
    return AureliusServer(
        host,
        port,
        generate_fn,
        auth_middleware=auth_middleware,
        rate_limiter=rate_limiter,
        coalescer=coalescer,
        bind_and_activate=bind_and_activate,
    )


def make_mock_generate_fn() -> Callable[["ChatRequest"], str]:
    def _generate(request: ChatRequest) -> str:
        last_user_message = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break
        return f"Mock response to: {last_user_message}"

    return _generate


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Aurelius API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--timeout", type=int, default=30, help="Graceful shutdown timeout (default: 30s)")
    args = parser.parse_args()

    generate_fn = make_mock_generate_fn()
    server = create_server(args.host, args.port, generate_fn)
    server._started_at = time.time()
    logger.info("Aurelius API server listening on http://%s:%d", args.host, args.port)

    shutdown_event = threading.Event()

    def _handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down gracefully...", sig_name)
        shutdown_event.set()
        server.shutdown()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        logger.info("Server stopped.")
