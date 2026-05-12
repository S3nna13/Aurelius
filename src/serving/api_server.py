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
import ipaddress
import json
import logging
import math
import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .auth_middleware import DEFAULT_AUTH_MIDDLEWARE, AuthConfig, AuthMiddleware

try:
    from src.longcontext.paged_kv_cache import PagedKVCache
except Exception:  # pragma: no cover
    PagedKVCache = None  # type: ignore[misc,assignment]

try:
    from src.longcontext.prefix_cache import PrefixCache
except Exception:  # pragma: no cover
    PrefixCache = None  # type: ignore[misc,assignment]

try:
    from src.longcontext.chunk_prefill import ChunkPrefillScheduler
except Exception:  # pragma: no cover
    ChunkPrefillScheduler = None  # type: ignore[misc,assignment]

try:
    from src.routing.model_router import ModelRouter, TaskProfile
except Exception:  # pragma: no cover
    ModelRouter = None  # type: ignore[misc,assignment]
    TaskProfile = None  # type: ignore[misc,assignment]

try:
    from src.inference.slora import SLoRARegistry
except Exception:  # pragma: no cover
    SLoRARegistry = None  # type: ignore[misc,assignment]

try:
    from src.retrieval.pipeline import RetrievalPipeline
except Exception:  # pragma: no cover
    RetrievalPipeline = None  # type: ignore[misc,assignment]

from .cors_middleware import CORS
from .metrics_middleware import METRICS
from .openapi_spec import openapi_spec, render_docs_page
from .rate_limiter import RateLimiterChain, TokenBucketLimiter
from .request_coalescer import RequestCoalescer

logger = logging.getLogger(__name__)


def _merge_lora_into_model(model: Any, adapter_id: str, registry: Any) -> bool:
    """Temporarily merge a LoRA adapter's delta into matching linear layers.

    Uses additive weight merge (W += B @ A * scaling) so standard nn.Linear
    forward calls benefit from the adapter without any code change downstream.
    Returns True if at least one layer was updated.
    """
    try:
        import torch
        import torch.nn as nn
        adapter = registry.get(adapter_id)
        with torch.no_grad():
            delta = (adapter.B @ adapter.A) * adapter.scaling  # (d_out, d_in)
        applied = False
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.shape == delta.shape:
                module.weight.data.add_(delta.to(module.weight.device, non_blocking=True))
                applied = True
        return applied
    except Exception:
        logger.debug("LoRA merge failed for %s", adapter_id, exc_info=True)
        return False


def _unmerge_lora_from_model(model: Any, adapter_id: str, registry: Any) -> None:
    """Undo the weight merge applied by _merge_lora_into_model."""
    try:
        import torch
        import torch.nn as nn
        adapter = registry.get(adapter_id)
        with torch.no_grad():
            delta = (adapter.B @ adapter.A) * adapter.scaling
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.shape == delta.shape:
                module.weight.data.sub_(delta.to(module.weight.device, non_blocking=True))
    except Exception:
        logger.debug("LoRA unmerge failed for %s", adapter_id, exc_info=True)


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

    if auth_mw is not None and not auth_mw.is_configured:
        host = server.server_address[0] if server else "localhost"
        if host not in ("127.0.0.1", "localhost", "::1"):
            handler._send_error(401, "Auth middleware removed on non-loopback interface")
            return False

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

    model_name = str(body.get("model", "")).strip()
    if not model_name:
        raise ValueError("model field cannot be empty")

    stop = body.get("stop")
    if stop is not None:
        if isinstance(stop, str) and stop:
            pass  # single stop sequence — allowed
        elif isinstance(stop, list):
            if not all(isinstance(s, str) and s for s in stop):
                raise ValueError("stop must be a list of non-empty strings")
            if len(stop) > 4:
                raise ValueError("stop list cannot exceed 4 entries")
        else:
            raise ValueError("stop must be a string or list of strings")

    return ChatRequest(
        model=model_name,
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
        host = self.server.server_address[0] if self.server else "localhost"
        if host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning("CORS enabled on non-loopback host %s", host)
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
        connection = getattr(self, "connection", None)
        if connection is not None and hasattr(connection, "settimeout"):
            connection.settimeout(30.0)
        try:
            return self.rfile.read(content_length)
        finally:
            if connection is not None and hasattr(connection, "settimeout"):
                connection.settimeout(None)

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

    def _generate_with_chunks(self, request, token_ids):
        """Chunked prefill with KV forwarding, then decode."""
        model = getattr(self.server, "model", None)
        tokenizer_decode = getattr(self.server, "tokenizer_decode", None)
        scheduler = getattr(self.server, "chunk_scheduler", None)
        if model is None or scheduler is None or len(token_ids) <= 512:
            return self.server.generate_fn(request)

        import torch

        from src.longcontext.chunk_prefill import ChunkPrefillConfig

        config = ChunkPrefillConfig(chunk_size=512, overlap=64, max_chunks=32)
        chunks = scheduler.split(token_ids, config)
        device = next(model.parameters()).device
        past_key_values = None
        logits = None

        for i, chunk_result in enumerate(chunks):
            chunk_ids = chunk_result.token_ids
            if i > 0:
                chunk_ids = chunk_ids[config.overlap:]
            if not chunk_ids:
                continue
            chunk_tensor = torch.tensor([chunk_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _, logits, past_key_values = model(chunk_tensor, past_key_values=past_key_values)

        if logits is None:
            return self.server.generate_fn(request)

        max_new_tokens = request.max_tokens
        temperature = request.temperature
        generated_ids = []

        def _sample_next_token(next_logits):
            if temperature <= 0.01:
                return torch.argmax(next_logits, dim=-1, keepdim=True)
            next_logits = next_logits / temperature
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_mask = cumulative_probs <= (1.0 - 0.9)
            sorted_mask[..., -1:] = False
            mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            next_logits = next_logits.masked_fill(mask, float("-inf"))
            return torch.multinomial(next_logits.softmax(dim=-1), num_samples=1)

        next_token = _sample_next_token(logits[:, -1, :])
        generated_ids.append(next_token.item())
        cur_ids = next_token
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                _, logits, past_key_values = model(cur_ids, past_key_values=past_key_values)
            next_token = _sample_next_token(logits[:, -1, :])
            generated_ids.append(next_token.item())
            cur_ids = next_token

        if tokenizer_decode is not None:
            return tokenizer_decode(generated_ids)
        return "".join(str(i) for i in generated_ids)

    def _generate_via_model(self, request, token_ids):
        """Generate text using the attached AureliusTransformer model."""
        model = getattr(self.server, "model", None)
        tokenizer_decode = getattr(self.server, "tokenizer_decode", None)
        paged_kv = getattr(self.server, "paged_kv", None)
        if model is None:
            raise RuntimeError("Model not available")
        import torch
        device = next(model.parameters()).device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        # Quest page-level sparse attention takes priority when enabled.
        if hasattr(model, "quest_attention"):
            with torch.no_grad():
                output_ids = model.generate_with_quest(
                    input_ids,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
        elif paged_kv is not None and hasattr(model, "generate_paged"):
            with torch.no_grad():
                output_ids = model.generate_paged(
                    input_ids,
                    paged_kv=paged_kv,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
        else:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
        generated_ids = output_ids[0, len(token_ids):].tolist()
        if tokenizer_decode is not None:
            return tokenizer_decode(generated_ids)
        return "".join(str(i) for i in generated_ids)

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
            model_id = getattr(self.server, "model_id", "aurelius")
            is_mock = getattr(self.server, "_is_mock", False)
            entry = {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "aurelius",
            }
            if is_mock:
                entry["metadata"] = {"mock": True}
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [entry],
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

        content: str | None = None

        # S-LoRA adapter handling
        adapter_id = body.get("adapter_id")
        slora_registry = getattr(self.server, "slora_registry", None)
        _lora_merged = False
        if adapter_id and slora_registry is not None:
            if adapter_id not in slora_registry:
                adapter_dir = self.server.config.get("serving", {}).get("adapter_dir", "adapters")
                adapter_path = os.path.join(adapter_dir, adapter_id)
                a_path = os.path.join(adapter_path, "A.pt")
                b_path = os.path.join(adapter_path, "B.pt")
                if os.path.exists(a_path) and os.path.exists(b_path):
                    try:
                        import torch
                        A = torch.load(a_path, map_location="cpu", weights_only=True)
                        B = torch.load(b_path, map_location="cpu", weights_only=True)
                        rank = A.shape[0]
                        # LRU eviction: evict oldest if at capacity before loading new adapter.
                        if len(slora_registry) >= slora_registry.max_adapters:
                            oldest = slora_registry.active_ids()[0]
                            model_obj = getattr(self.server, "model", None)
                            if model_obj is not None:
                                _unmerge_lora_from_model(model_obj, oldest, slora_registry)
                            slora_registry.swap_out(oldest)
                        slora_registry.swap_in(adapter_id, A, B, rank)
                    except Exception as exc:
                        logger.warning("Failed to load S-LoRA adapter %s: %s", adapter_id, exc)
            # Merge the adapter weights into the model for this request.
            model_obj = getattr(self.server, "model", None)
            if model_obj is not None and adapter_id in slora_registry:
                _lora_merged = _merge_lora_into_model(model_obj, adapter_id, slora_registry)

        # Build prompt text for routing and prefix caching
        prompt_parts: list[str] = []
        for msg in request.messages:
            role = msg.get("role", "user")
            msg_content = msg.get("content", "")
            tok = (
                "<|system|>"
                if role == "system"
                else "<|user|>"
                if role == "user"
                else "<|assistant|>"
            )
            prompt_parts.append(f"{tok}\n{msg_content}<|end|>\n")
        prompt_parts.append("<|assistant|>\n")
        prompt_text = "".join(prompt_parts)

        # Model routing
        router = getattr(self.server, "model_router", None)
        if router is not None and TaskProfile is not None:
            try:
                decision = router.route(TaskProfile(user_id="anonymous", content=prompt_text))
            except Exception:
                logger.debug("ModelRouter.route failed", exc_info=True)
                decision = None
            else:
                if decision is not None:
                    if getattr(decision, "requires_retrieval", False):
                        pipeline = getattr(self.server, "retrieval_pipeline", None)
                        if pipeline is not None:
                            try:
                                # Use the raw user message text, not the chat-formatted prompt.
                                raw_query = ""
                                for msg in reversed(request.messages):
                                    if msg.get("role") == "user":
                                        raw_query = msg.get("content", "")
                                        break
                                rag_result = pipeline.run(raw_query or prompt_text)
                                retrieved = getattr(rag_result, "compressed_context", "") or ""
                                if retrieved:
                                    request.messages.insert(0, {
                                        "role": "system",
                                        "content": f"Retrieved context:\n{retrieved}",
                                    })
                            except Exception:
                                logger.debug("Retrieval pipeline failed", exc_info=True)
                    if getattr(decision, "requires_tools", False):
                        try:
                            from src.agent.react_loop import ReActLoop

                            def _llm_generate(messages: list[dict]) -> str:
                                req = ChatRequest(
                                    model=request.model,
                                    messages=messages,
                                    temperature=request.temperature,
                                    max_tokens=request.max_tokens,
                                )
                                return self.server.generate_fn(req)

                            def _safe_calc(expression: str) -> str:
                                allowed = set("0123456789.+-*/() ")
                                if not all(c in allowed for c in expression):
                                    return "Error: invalid characters in expression"
                                try:
                                    # B307 suppress: arithmetic-only eval; input pre-validated to digit/dot/dash/slash/parens/space
                                    return str(eval(expression, {"__builtins__": {}}, {}))  # nosec B307  # noqa: S307
                                except Exception as e:
                                    return f"Error: {e}"

                            tools = {
                                "calculator": _safe_calc,
                                "datetime": lambda: __import__("datetime").datetime.now().isoformat(),
                                "uppercase": lambda text: text.upper(),
                                "lowercase": lambda text: text.lower(),
                            }
                            loop = ReActLoop(
                                generate_fn=_llm_generate,
                                tool_registry=tools,
                                max_steps=8,
                                max_tool_seconds=5.0,
                            )
                            result = loop.run(prompt_text)
                            if result.final_answer is not None:
                                content = result.final_answer
                        except Exception as exc:
                            logger.warning("ReActLoop failed, falling back to normal generation: %s", exc)

        # Prefix cache + tokenization
        tokenizer_encode = getattr(self.server, "tokenizer_encode", None)
        prefix_cache = getattr(self.server, "prefix_cache", None)
        token_ids: list[int] = []
        if tokenizer_encode is not None:
            try:
                token_ids = tokenizer_encode(prompt_text)
            except Exception:
                logger.debug("tokenizer_encode failed", exc_info=True)

        if prefix_cache is not None and token_ids:
            try:
                matched_len, entry = prefix_cache.find_longest_prefix(token_ids)
                if entry is not None and matched_len == len(token_ids):
                    cached = entry.get("content") if isinstance(entry, dict) else None
                    if cached:
                        logger.debug("PrefixCache exact hit: %d tokens", matched_len)
                        content = cached
                elif entry is not None:
                    logger.debug("PrefixCache partial hit: %d/%d tokens", matched_len, len(token_ids))
            except Exception:
                logger.debug("PrefixCache lookup failed", exc_info=True)

        def _generate() -> str:
            if content is not None:
                return content
            chunk_scheduler = getattr(self.server, "chunk_scheduler", None)
            model = getattr(self.server, "model", None)
            if model is not None:
                if chunk_scheduler is not None and len(token_ids) > 512:
                    try:
                        return self._generate_with_chunks(request, token_ids)
                    except Exception:
                        logger.debug("Chunked prefill failed, falling back", exc_info=True)
                try:
                    return self._generate_via_model(request, token_ids)
                except Exception:
                    logger.debug("Model generation failed, falling back", exc_info=True)
            return self.server.generate_fn(request)

        try:
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
                    logger.debug("Failed to compute coalesce key", exc_info=True)
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
        finally:
            # Unmerge LoRA adapter weights — runs even if an exception caused an early return.
            if _lora_merged and adapter_id and slora_registry is not None:
                model_obj = getattr(self.server, "model", None)
                if model_obj is not None:
                    _unmerge_lora_from_model(model_obj, adapter_id, slora_registry)

        # Update prefix cache with the generated content so exact-match hits skip generation.
        if prefix_cache is not None and token_ids and content:
            try:
                prefix_cache.insert(token_ids, {"content": content})
            except Exception:
                logger.debug("PrefixCache insert failed", exc_info=True)

        if tokenizer_encode is not None:
            prompt_tokens = sum(len(tokenizer_encode(m.get("content") or "")) for m in request.messages)
            completion_tokens = len(tokenizer_encode(content))
        else:
            # NOTE: counts are approximate word counts because a tokenizer is not available.
            prompt_tokens = sum(len((m.get("content") or "").split()) for m in request.messages)
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

        if request.stream:
            self._send_error(501, "Streaming is not yet implemented")
            self._record_metrics(start, 501, "streaming_not_implemented")
            return

        self._send_json(200, response.to_dict())
        self._record_metrics(start, 200)


class AureliusServer(HTTPServer):
    def __init__(
        self,
        host: str,
        port: int,
        generate_fn: Callable[["ChatRequest"], str],
        *,
        model_id: str = "aurelius",
        auth_middleware: AuthMiddleware | None = None,
        rate_limiter: TokenBucketLimiter | RateLimiterChain | None = None,
        coalescer: RequestCoalescer | None = None,
        paged_kv: Any | None = None,
        prefix_cache: Any | None = None,
        chunk_scheduler: Any | None = None,
        model_router: Any | None = None,
        slora_registry: Any | None = None,
        retrieval_pipeline: Any | None = None,
        tokenizer_encode: Callable[[str], list[int]] | None = None,
        tokenizer_decode: Callable[[list[int]], str] | None = None,
        model: Any | None = None,
        config: dict[str, Any] | None = None,
        bind_and_activate: bool = True,
    ):
        super().__init__(
            (host, port),
            AureliusRequestHandler,
            bind_and_activate=bind_and_activate,
        )
        self.generate_fn = generate_fn
        self.model_id = model_id
        self.auth_middleware = auth_middleware
        self.rate_limiter = rate_limiter
        self.coalescer = coalescer
        self.paged_kv = paged_kv
        self.prefix_cache = prefix_cache
        self.chunk_scheduler = chunk_scheduler
        self.model_router = model_router
        self.slora_registry = slora_registry
        self.retrieval_pipeline = retrieval_pipeline
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.model = model
        self.config = config or {}


def create_server(
    host: str,
    port: int,
    generate_fn: Callable[["ChatRequest"], str],
    *,
    model_id: str = "aurelius",
    auth_middleware: AuthMiddleware | None = None,
    rate_limiter: TokenBucketLimiter | RateLimiterChain | None = None,
    coalescer: RequestCoalescer | None = None,
    paged_kv: Any | None = None,
    prefix_cache: Any | None = None,
    chunk_scheduler: Any | None = None,
    model_router: Any | None = None,
    slora_registry: Any | None = None,
    retrieval_pipeline: Any | None = None,
    tokenizer_encode: Callable[[str], list[int]] | None = None,
    tokenizer_decode: Callable[[list[int]], str] | None = None,
    model: Any | None = None,
    config: dict[str, Any] | None = None,
    bind_and_activate: bool = True,
) -> AureliusServer:
    try:
        ip_addr = ipaddress.ip_address(host)
        is_loopback = ip_addr.is_loopback
    except ValueError:
        is_loopback = host in ("localhost",)
    if not is_loopback and (
        auth_middleware is None or not auth_middleware.is_configured
    ):
        raise RuntimeError(
            f"Non-loopback host {host} requires at least one configured API key "
            f"(set AURELIUS_API_KEYS or AURELIUS_API_KEY env var)."
        )
    if paged_kv is None and PagedKVCache is not None:
        try:
            paged_kv = PagedKVCache(n_heads=16, head_dim=128, page_size=16, num_pages=4096)
        except Exception:
            logger.debug("Failed to instantiate PagedKVCache", exc_info=True)
    if prefix_cache is None and PrefixCache is not None:
        try:
            prefix_cache = PrefixCache(max_entries=2048, min_prefix_tokens=16, block_size=16)
        except Exception:
            logger.debug("Failed to instantiate PrefixCache", exc_info=True)
    if chunk_scheduler is None and ChunkPrefillScheduler is not None:
        try:
            chunk_scheduler = ChunkPrefillScheduler()
        except Exception:
            logger.debug("Failed to instantiate ChunkPrefillScheduler", exc_info=True)
    if model_router is None and ModelRouter is not None:
        try:
            model_router = ModelRouter()
        except Exception:
            logger.debug("Failed to instantiate ModelRouter", exc_info=True)
    cfg = config or {}
    if slora_registry is None and SLoRARegistry is not None:
        if cfg.get("serving", {}).get("slora_enabled", False):
            try:
                slora_registry = SLoRARegistry(max_adapters=4)
            except Exception:
                logger.debug("Failed to instantiate SLoRARegistry", exc_info=True)
    if retrieval_pipeline is None and RetrievalPipeline is not None:
        try:
            retrieval_pipeline = RetrievalPipeline.from_defaults()
        except Exception:
            logger.debug("Failed to instantiate RetrievalPipeline", exc_info=True)
    return AureliusServer(
        host,
        port,
        generate_fn,
        model_id=model_id,
        auth_middleware=auth_middleware,
        rate_limiter=rate_limiter,
        coalescer=coalescer,
        paged_kv=paged_kv,
        prefix_cache=prefix_cache,
        chunk_scheduler=chunk_scheduler,
        model_router=model_router,
        slora_registry=slora_registry,
        retrieval_pipeline=retrieval_pipeline,
        tokenizer_encode=tokenizer_encode,
        tokenizer_decode=tokenizer_decode,
        model=model,
        config=cfg,
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


def _load_auth_from_env() -> AuthMiddleware:
    """Load AuthMiddleware from AURELIUS_API_KEYS environment variable.

    Format: key_id:raw_key:scope1,scope2;...
    Falls back to DEFAULT_AUTH_MIDDLEWARE if the variable is unset.
    """
    api_keys_env = os.environ.get("AURELIUS_API_KEYS", "")
    if not api_keys_env:
        single_key = os.environ.get("AURELIUS_API_KEY", "")
        if single_key:
            auth_config = AuthConfig(keys={}, require_auth=True)
            auth_mw = AuthMiddleware(auth_config)
            auth_mw.add_key("default", single_key, frozenset())
            return auth_mw
        return DEFAULT_AUTH_MIDDLEWARE
    auth_config = AuthConfig(keys={}, require_auth=True)
    auth_mw = AuthMiddleware(auth_config)
    for key_def in api_keys_env.split(";"):
        key_def = key_def.strip()
        if not key_def:
            continue
        parts = key_def.split(":")
        if len(parts) < 2:
            logger.warning("Skipping malformed AURELIUS_API_KEYS entry: %s", key_def)
            continue
        key_id = parts[0]
        raw_key = parts[1]
        scopes = frozenset(parts[2].split(",")) if len(parts) > 2 and parts[2] else frozenset()
        auth_mw.add_key(key_id, raw_key, scopes)
    return auth_mw


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Aurelius API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port (default: 8080)",
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Graceful shutdown timeout (default: 30s)"
    )
    args = parser.parse_args()

    auth_mw = _load_auth_from_env()

    # Security: require at least one API key when binding non-loopback
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        if not auth_mw.is_configured:
            logger.error(
                "Aborting: non-loopback host %s requires at least one configured API key "
                "(set AURELIUS_API_KEYS env var).",
                args.host,
            )
            sys.exit(1)

    generate_fn = make_mock_generate_fn()

    paged_kv = None
    if PagedKVCache is not None:
        try:
            paged_kv = PagedKVCache(n_heads=16, head_dim=128, page_size=16, num_pages=4096)
        except Exception:
            logger.debug("Failed to instantiate PagedKVCache", exc_info=True)

    prefix_cache = None
    if PrefixCache is not None:
        try:
            prefix_cache = PrefixCache(max_entries=2048, min_prefix_tokens=16, block_size=16)
        except Exception:
            logger.debug("Failed to instantiate PrefixCache", exc_info=True)

    chunk_scheduler = None
    if ChunkPrefillScheduler is not None:
        try:
            chunk_scheduler = ChunkPrefillScheduler()
        except Exception:
            logger.debug("Failed to instantiate ChunkPrefillScheduler", exc_info=True)

    model_router = None
    if ModelRouter is not None:
        try:
            model_router = ModelRouter()
        except Exception:
            logger.debug("Failed to instantiate ModelRouter", exc_info=True)

    server = create_server(
        args.host,
        args.port,
        generate_fn,
        auth_middleware=auth_mw,
        paged_kv=paged_kv,
        prefix_cache=prefix_cache,
        chunk_scheduler=chunk_scheduler,
        model_router=model_router,
    )
    server._is_mock = True
    server._started_at = time.time()
    logger.info("Aurelius API server listening on http://%s:%d", args.host, args.port)

    shutdown_event = threading.Event()

    def _handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, shutting down gracefully...", sig_name)
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.shutdown()
        logger.info("Server stopped.")
