"""Security regression tests for findings AUR-SEC-2026-0001 through AUR-SEC-2026-0011."""

from __future__ import annotations

import inspect
import json
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0001 — torch.load weights_only=True for optimizer state
# ---------------------------------------------------------------------------

def test_load_checkpoint_optimizer_uses_weights_only(tmp_path):
    """AUR-SEC-2026-0001: load_checkpoint must call torch.load with weights_only=True.

    Pre-fix: torch.load(..., weights_only=False) on optimizer.pt allowed arbitrary
    pickle execution.  Post-fix: weights_only=True is passed at both call sites.
    """
    import torch
    import torch.nn as nn
    from src.training.checkpoint import load_checkpoint, CheckpointMeta

    # Build a minimal checkpoint directory.
    ckpt_dir = tmp_path / "checkpoint-0000001"
    ckpt_dir.mkdir()

    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(opt.state_dict(), ckpt_dir / "optimizer.pt")
    meta = {
        "step": 1, "epoch": 0, "train_loss": 0.5,
        "val_loss": None, "config": {},
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))

    calls: list[dict] = []
    real_torch_load = torch.load

    def tracking_load(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return real_torch_load(*args, **kwargs)

    with patch("src.training.checkpoint.torch.load", side_effect=tracking_load):
        result = load_checkpoint(model, ckpt_dir, optimizer=opt)

    assert isinstance(result, CheckpointMeta)
    # Both model.pt and optimizer.pt must be loaded with weights_only=True.
    for call in calls:
        assert call["kwargs"].get("weights_only") is True, (
            f"torch.load called without weights_only=True: {call}"
        )


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0002 — ConversationStore path traversal prevention
# ---------------------------------------------------------------------------

class TestConversationStorePathTraversal:
    """AUR-SEC-2026-0002: _path() must reject traversal and overly-long IDs."""

    def _store(self, tmp_path):
        from src.serving.conversation_store import ConversationStore
        return ConversationStore(storage_dir=str(tmp_path))

    def test_dotdot_etc_passwd_raises(self, tmp_path):
        """AUR-SEC-2026-0002: '../../etc/passwd' must raise ValueError."""
        store = self._store(tmp_path)
        with pytest.raises(ValueError):
            store._path("../../etc/passwd")

    def test_dotdot_sibling_raises(self, tmp_path):
        """AUR-SEC-2026-0002: '../sibling' must raise ValueError."""
        store = self._store(tmp_path)
        with pytest.raises(ValueError):
            store._path("../sibling")

    def test_too_long_id_raises(self, tmp_path):
        """AUR-SEC-2026-0002: IDs longer than 128 chars must raise ValueError."""
        store = self._store(tmp_path)
        with pytest.raises(ValueError):
            store._path("a" * 129)

    def test_valid_id_accepted(self, tmp_path):
        """AUR-SEC-2026-0002: A well-formed ID must not raise."""
        store = self._store(tmp_path)
        path = store._path("abc-123_ok")
        assert path.name == "abc-123_ok.json"


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0003 — ReDoS in repo_context_packer regexes
# ---------------------------------------------------------------------------

def test_js_import_regex_no_redos():
    """AUR-SEC-2026-0003: _JS_IMPORT_RE must not catastrophically backtrack on adversarial input."""
    from src.agent.repo_context_packer import _JS_IMPORT_RE

    adversarial = "import " + "a" * 50_000
    start = time.perf_counter()
    _JS_IMPORT_RE.findall(adversarial)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"_JS_IMPORT_RE took {elapsed:.3f}s on adversarial input (ReDoS suspected)"


def test_go_import_regex_no_redos():
    """AUR-SEC-2026-0003: _GO_IMPORT_RE must not catastrophically backtrack on adversarial input."""
    from src.agent.repo_context_packer import _GO_IMPORT_RE

    adversarial = "import " + "a" * 50_000
    start = time.perf_counter()
    _GO_IMPORT_RE.findall(adversarial)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"_GO_IMPORT_RE took {elapsed:.3f}s on adversarial input (ReDoS suspected)"


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0004 — MD5 replaced by blake2b in session router
# ---------------------------------------------------------------------------

def test_session_router_hash_not_md5():
    """AUR-SEC-2026-0004: ConsistentHashRouter._hash() must NOT use MD5."""
    from src.serving.session_router import ConsistentHashRouter
    src = inspect.getsource(ConsistentHashRouter._hash)
    assert "md5" not in src.lower(), "_hash() still references MD5 — fix not applied"


def test_session_router_hash_deterministic():
    """AUR-SEC-2026-0004: _hash() must return the same int for the same key."""
    from src.serving.session_router import ConsistentHashRouter, SessionConfig
    router = ConsistentHashRouter(SessionConfig(n_workers=4))
    assert router._hash("test-session") == router._hash("test-session")


def test_session_router_routes_consistently():
    """AUR-SEC-2026-0004: route() must return the same worker for the same session_id."""
    from src.serving.session_router import ConsistentHashRouter, SessionConfig
    router = ConsistentHashRouter(SessionConfig(n_workers=4))
    sid = "stable-session-id"
    assert router.route(sid) == router.route(sid)


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0005 — Guardrails: single-substring harm detection
# ---------------------------------------------------------------------------

class TestGuardrailsHarmDetection:
    """AUR-SEC-2026-0005: _harm_score must fire on ANY single suspicious substring."""

    def _guardrails(self):
        from src.serving.guardrails import ContentGuardrails, GuardrailPolicy
        return ContentGuardrails(GuardrailPolicy(harm_threshold=0.5))

    def test_inject_alone_blocked(self):
        """AUR-SEC-2026-0005: Input containing only 'INJECT' must be blocked."""
        result = self._guardrails().check_input("please INJECT this")
        assert result.allowed is False

    def test_override_alone_blocked(self):
        """AUR-SEC-2026-0005: Input containing only 'OVERRIDE' must be blocked."""
        result = self._guardrails().check_input("OVERRIDE all rules")
        assert result.allowed is False

    def test_system_bypass_alone_blocked(self):
        """AUR-SEC-2026-0005: Input containing only 'SYSTEM_BYPASS' must be blocked."""
        result = self._guardrails().check_input("attempt SYSTEM_BYPASS now")
        assert result.allowed is False

    def test_clean_input_allowed(self):
        """AUR-SEC-2026-0005: Clean input must pass guardrails."""
        result = self._guardrails().check_input("Hello, how are you?")
        assert result.allowed is True


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0006 — MCP server: no exception detail leaked to callers
# ---------------------------------------------------------------------------

def test_mcp_server_exception_not_leaked():
    """AUR-SEC-2026-0006: StdioMCPServer must return generic error, not the exception message."""
    from src.mcp.mcp_server import StdioMCPServer, MCPServerConfig

    server = StdioMCPServer(MCPServerConfig(transport="stdio"))

    def _exploding_handler(self, params):
        raise RuntimeError("secret_token=abcdef")

    server._METHOD_HANDLERS = dict(server._METHOD_HANDLERS)
    server._METHOD_HANDLERS["boom"] = _exploding_handler

    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "boom",
        "params": {},
    })

    error_field = response.get("error", {})
    error_msg = error_field.get("message", "") if isinstance(error_field, dict) else str(error_field)
    assert "secret_token" not in error_msg, "Exception detail leaked in MCP error response"
    assert error_msg == "Internal server error", f"Unexpected error message: {error_msg!r}"


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0007 — ChatSession: robust model output unpacking (4-tuple)
# ---------------------------------------------------------------------------

def test_chat_session_4tuple_model_output():
    """AUR-SEC-2026-0007: logits must be taken from output[1] so 4-tuples work."""
    import torch
    import torch.nn as nn
    from src.serving.chat_session import ChatSession, GenerationConfig

    vocab_size = 16

    class _FakeCfg:
        max_seq_len = 512

    class FourTupleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Dummy parameter so ChatSession can call next(model.parameters()).device
            self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
            # Minimal config stub required by _generate_with_penalty
            self.config = _FakeCfg()

        def forward(self, x):
            batch, seq = x.shape
            logits = torch.zeros(batch, seq, vocab_size)
            # Return a 4-tuple; index 1 is the logits tensor.
            return ("extra_before", logits, "extra_after", "extra_fourth")

    class MockTokenizer:
        """Minimal tokenizer stub satisfying ChatSession interface."""
        def encode(self, text: str):
            return [0] * max(1, len(text) // 4)
        def decode(self, ids):
            return "ok"

    cfg = GenerationConfig(max_new_tokens=2, temperature=1.0)
    session = ChatSession(
        model=FourTupleModel(),
        tokenizer=MockTokenizer(),
        gen_cfg=cfg,
    )

    input_ids = torch.zeros(1, 3, dtype=torch.long)
    # Should not raise; must extract logits from index 1 of the 4-tuple.
    output = session._generate_with_penalty(input_ids, cfg)
    assert output.shape[0] == 1  # batch dimension preserved


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0008 — SSE server: Content-Length cap returns 413
# ---------------------------------------------------------------------------

def test_sse_server_large_content_length_returns_413():
    """AUR-SEC-2026-0008: POST with Content-Length > 1 MiB must return 413 without reading body."""
    from src.mcp.sse_mcp_server import SSEMCPServer, SSEMCPServerConfig, _SSEHandler

    config = SSEMCPServerConfig(host="127.0.0.1", port=0, path="/events")
    sse_server = SSEMCPServer(config=config)

    sent_responses: list[int] = []
    sent_headers: list[tuple] = []

    handler = _SSEHandler.__new__(_SSEHandler)
    handler._sse_server = sse_server
    handler.path = "/events"
    handler.wfile = BytesIO()
    handler.rfile = BytesIO(b"")  # body should NOT be read

    handler.headers = {"Content-Length": "2000000"}

    handler.send_response = lambda code: sent_responses.append(code)
    handler.send_header = lambda name, value: sent_headers.append((name, value))
    handler.end_headers = lambda: None

    handler.do_POST()

    assert 413 in sent_responses, f"Expected 413 response, got: {sent_responses}"


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0009 — SSE server: CORS wildcard default removed
# ---------------------------------------------------------------------------

class TestSSECORSPolicy:
    """AUR-SEC-2026-0009: Default cors_origins must be [] (no wildcard CORS)."""

    def test_default_cors_origins_empty(self):
        """AUR-SEC-2026-0009: SSEMCPServerConfig() default cors_origins must be []."""
        from src.mcp.sse_mcp_server import SSEMCPServerConfig
        cfg = SSEMCPServerConfig()
        assert cfg.cors_origins == [], f"Expected [], got {cfg.cors_origins!r}"

    def test_empty_cors_sends_no_acao_header(self):
        """AUR-SEC-2026-0009: cors_origins=[] must not send Access-Control-Allow-Origin."""
        from src.mcp.sse_mcp_server import SSEMCPServer, SSEMCPServerConfig, _SSEHandler

        config = SSEMCPServerConfig(cors_origins=[])
        sse_server = SSEMCPServer(config=config)

        sent_headers: list[tuple] = []

        handler = _SSEHandler.__new__(_SSEHandler)
        handler._sse_server = sse_server
        handler.headers = {"Origin": "https://evil.com"}
        handler.send_header = lambda name, value: sent_headers.append((name, value))

        handler._send_cors_headers()

        acao_headers = [h for h in sent_headers if h[0] == "Access-Control-Allow-Origin"]
        assert acao_headers == [], f"Unexpected CORS headers sent: {acao_headers}"

    def test_allowed_origin_gets_exact_origin_not_wildcard(self):
        """AUR-SEC-2026-0009: Matching origin must get exact origin echo, not '*'."""
        from src.mcp.sse_mcp_server import SSEMCPServer, SSEMCPServerConfig, _SSEHandler

        allowed = "https://app.example.com"
        config = SSEMCPServerConfig(cors_origins=[allowed])
        sse_server = SSEMCPServer(config=config)

        sent_headers: list[tuple] = []

        handler = _SSEHandler.__new__(_SSEHandler)
        handler._sse_server = sse_server
        handler.headers = {"Origin": allowed}
        handler.send_header = lambda name, value: sent_headers.append((name, value))

        handler._send_cors_headers()

        acao_headers = [v for n, v in sent_headers if n == "Access-Control-Allow-Origin"]
        assert acao_headers, "No Access-Control-Allow-Origin header sent for allowed origin"
        assert "*" not in acao_headers, "Wildcard '*' must not be sent"
        assert acao_headers[0] == allowed, f"Expected {allowed!r}, got {acao_headers[0]!r}"


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0010 — SSE tool handler: exception not leaked
# ---------------------------------------------------------------------------

def test_sse_tool_handler_exception_not_leaked():
    """AUR-SEC-2026-0010: SSEMCPServer.handle_request() must not leak exception details."""
    from src.mcp.sse_mcp_server import SSEMCPServer, SSEMCPServerConfig

    config = SSEMCPServerConfig()
    sse_server = SSEMCPServer(config=config)

    def leaky_handler(payload: dict) -> dict:
        raise RuntimeError("internal_credential=xyz")

    sse_server.register_tool_handler("secret_tool", leaky_handler)

    response = sse_server.handle_request({"tool": "secret_tool", "params": {}})

    error_msg = response.get("error", "")
    assert "internal_credential" not in error_msg, (
        f"Exception detail leaked in SSE error response: {error_msg!r}"
    )
    assert error_msg == "handler error", (
        f"Expected generic 'handler error', got: {error_msg!r}"
    )


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0011 — warm_start.py: both torch.load call sites use weights_only=True
# ---------------------------------------------------------------------------

def test_warm_start_torch_load_weights_only():
    """AUR-SEC-2026-0011: Both torch.load call sites in warm_start.py must pass weights_only=True.

    torch.load calls span multiple lines, so we scan logical call blocks by collecting
    lines from 'torch.load(' until the closing ')' and check that block for the kwarg.
    """
    source_path = Path(__file__).parents[2] / "src" / "training" / "warm_start.py"
    lines = source_path.read_text().splitlines()

    call_blocks: list[str] = []
    i = 0
    while i < len(lines):
        if "torch.load(" in lines[i]:
            # Collect the full call: accumulate lines until paren depth returns to 0.
            block = lines[i]
            depth = lines[i].count("(") - lines[i].count(")")
            j = i + 1
            while depth > 0 and j < len(lines):
                block += " " + lines[j].strip()
                depth += lines[j].count("(") - lines[j].count(")")
                j += 1
            call_blocks.append(block.strip())
            i = j
        else:
            i += 1

    assert call_blocks, "No torch.load calls found in warm_start.py"
    for block in call_blocks:
        assert "weights_only=True" in block, (
            f"torch.load call missing weights_only=True:\n  {block!r}"
        )
