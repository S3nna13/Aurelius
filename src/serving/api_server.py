"""
OpenAI-compatible HTTP API server for Aurelius.

Run: python -m src.serving.api_server --port 8080

Endpoints:
  POST /v1/chat/completions   — chat completion (streaming or non-streaming)
  GET  /v1/models             — list available models
  GET  /health                — health check
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatRequest:
    model: str
    messages: List[Dict]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False
    system: Optional[str] = None


@dataclass
class ChatResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Dict

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
            "usage": self.usage,
        }


class AureliusRequestHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, data: Dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": "api_error"}})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
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
        else:
            self._send_error(404, f"Not found: {self.path}")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_error(404, f"Not found: {self.path}")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return

        if "messages" not in body:
            self._send_error(400, "Missing required field: messages")
            return

        try:
            request = ChatRequest(
                model=body.get("model", "aurelius"),
                messages=body["messages"],
                temperature=float(body.get("temperature", 0.7)),
                max_tokens=int(body.get("max_tokens", 512)),
                stream=bool(body.get("stream", False)),
                system=body.get("system"),
            )
        except (TypeError, ValueError) as exc:
            self._send_error(400, f"Invalid request parameters: {exc}")
            return

        try:
            content = self.server.generate_fn(request)
        except Exception as exc:
            logger.exception("generate_fn raised an exception")
            self._send_error(500, f"Generation error: {exc}")
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


class AureliusServer(HTTPServer):

    def __init__(
        self,
        host: str,
        port: int,
        generate_fn: Callable[["ChatRequest"], str],
        *,
        bind_and_activate: bool = True,
    ):
        super().__init__(
            (host, port),
            AureliusRequestHandler,
            bind_and_activate=bind_and_activate,
        )
        self.generate_fn = generate_fn


def create_server(
    host: str,
    port: int,
    generate_fn: Callable[["ChatRequest"], str],
    *,
    bind_and_activate: bool = True,
) -> AureliusServer:
    return AureliusServer(
        host,
        port,
        generate_fn,
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
    args = parser.parse_args()

    generate_fn = make_mock_generate_fn()
    server = create_server(args.host, args.port, generate_fn)
    logger.info("Aurelius API server listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.server_close()
