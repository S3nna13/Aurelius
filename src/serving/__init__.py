# Canonical serving package. Backward-compat shim lives in gateway/__init__.py

from .agent_cockpit import *  # noqa: F401, F403
from .agentic_runtime import *  # noqa: F401, F403
from .api_server import *  # noqa: F401, F403
from .aurelius_api import *  # noqa: F401, F403
from .aurelius_server import *  # noqa: F401, F403
from .auth_middleware import *  # noqa: F401, F403
from .batch_predictor import *  # noqa: F401, F403
from .batch_processor import *  # noqa: F401, F403
from .chat_client import *  # noqa: F401, F403
from .chat_session import *  # noqa: F401, F403
from .circuit_breaker import *  # noqa: F401, F403
from .context_compressor import *  # noqa: F401, F403
from .continuous_batching import *  # noqa: F401, F403
from .conversation_store import *  # noqa: F401, F403
from .cors_middleware import *  # noqa: F401, F403
from .engine_loader import *  # noqa: F401, F403
from .function_calling_api import *  # noqa: F401, F403
from .guardrail_middleware import *  # noqa: F401, F403
from .guardrails import *  # noqa: F401, F403
from .health_probe import *  # noqa: F401, F403
from .hermes_notifier import *  # noqa: F401, F403
from .load_balancer import *  # noqa: F401, F403
from .load_shedder import *  # noqa: F401, F403
from .memory_manager import *  # noqa: F401, F403
from .metrics_middleware import *  # noqa: F401, F403
from .mission_control import *  # noqa: F401, F403
from .model_multiplexer import *  # noqa: F401, F403
from .model_router import *  # noqa: F401, F403
from .model_warmer import *  # noqa: F401, F403
from .openai_api_validator import *  # noqa: F401, F403
from .openapi_spec import *  # noqa: F401, F403
from .paged_kv_cache import *  # noqa: F401, F403
from .prompt_cache import *  # noqa: F401, F403
from .quick_instruction import *  # noqa: F401, F403
from .rate_limiter import *  # noqa: F401, F403
from .rate_limiter_v2 import *  # noqa: F401, F403
from .request_batcher import *  # noqa: F401, F403
from .request_coalescer import *  # noqa: F401, F403
from .request_orchestrator import *  # noqa: F401, F403
from .request_priority_queue import *  # noqa: F401, F403
from .request_queue import *  # noqa: F401, F403
from .request_tracker import *  # noqa: F401, F403
from .response_cache import *  # noqa: F401, F403
from .response_dedup import *  # noqa: F401, F403
from .response_formatter import *  # noqa: F401, F403
from .response_streamer import *  # noqa: F401, F403
from .responses_api import *  # noqa: F401, F403
from .session_router import *  # noqa: F401, F403
from .sse_chat_stream import *  # noqa: F401, F403
from .sse_stream_encoder import *  # noqa: F401, F403
from .streaming import *  # noqa: F401, F403
from .streaming_handler import *  # noqa: F401, F403
from .structured_output_decoder import *  # noqa: F401, F403
from .system_prompts import *  # noqa: F401, F403
from .task_api import *  # noqa: F401, F403
from .terminal_chat import *  # noqa: F401, F403
from .tool_call_streaming import *  # noqa: F401, F403
from .tool_chain import *  # noqa: F401, F403
from .tool_executor import *  # noqa: F401, F403
from .vllm_engine import *  # noqa: F401, F403
from .web_ui import *  # noqa: F401, F403
from .websocket import *  # noqa: F401, F403
from .xgrammar_decoder import *  # noqa: F401, F403

# ---------------------------------------------------------------------------
# Backwards-compatible registry aliases for integration tests
# ---------------------------------------------------------------------------

# DECODER_REGISTRY is the historical name for the structured-output decoder
# registry.  New code should use STRUCTURED_OUTPUT_REGISTRY directly.
from .structured_output_decoder import STRUCTURED_OUTPUT_REGISTRY  # noqa: E402

DECODER_REGISTRY = STRUCTURED_OUTPUT_REGISTRY

# STREAM_HANDLER_REGISTRY maps stream-type keys to handler/encoder classes.
# "sse"       → generic SSE wire encoder (SSEStreamEncoder)
# "sse_chat"  → chat-completion SSE streaming (SSEChatStream)
from .sse_stream_encoder import SSEStreamEncoder  # noqa: E402
from .sse_chat_stream import SSEChatStream  # noqa: E402

STREAM_HANDLER_REGISTRY: dict[str, type] = {
    "sse": SSEStreamEncoder,
    "sse_chat": SSEChatStream,
}

# ---------------------------------------------------------------------------
# Resilience registry — additive registry for resilience patterns
# ---------------------------------------------------------------------------
from .circuit_breaker import CircuitBreaker  # noqa: E402

RESILIENCE_REGISTRY: dict[str, type] = {
    "circuit_breaker": CircuitBreaker,
}


# Backwards-compatible namespace aliases.  The project historically allowed
# `serving`, `src.serving`, and `aurelius.serving`; make every spelling resolve
# to this same module object, including deep submodule imports.
from src.namespace_aliases import register_namespace_aliases as _register_aliases  # noqa: E402

_register_aliases("src.serving", ("serving", "aurelius.serving"))

