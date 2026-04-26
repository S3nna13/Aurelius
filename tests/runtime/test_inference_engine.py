import pytest

from src.runtime.inference_engine import (
    INFERENCE_ENGINE_REGISTRY,
    InferenceBackend,
    InferenceConfig,
    InferenceEngine,
    InferenceRequest,
    InferenceResponse,
)


class TestInferenceBackend:
    def test_members(self):
        assert InferenceBackend.EAGER.value == "eager"
        assert InferenceBackend.COMPILED.value == "compiled"
        assert InferenceBackend.JIT_CACHED.value == "jit_cached"
        assert InferenceBackend.ONNX.value == "onnx"

    def test_count(self):
        assert len(list(InferenceBackend)) == 4


class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.backend is InferenceBackend.EAGER
        assert cfg.max_batch_size == 32
        assert cfg.max_seq_len == 2048
        assert cfg.timeout_ms == 5000.0
        assert cfg.warmup_steps == 3
        assert cfg.use_fp16 is False

    def test_custom(self):
        cfg = InferenceConfig(
            backend=InferenceBackend.COMPILED,
            max_batch_size=16,
            max_seq_len=1024,
            timeout_ms=1000.0,
            warmup_steps=5,
            use_fp16=True,
        )
        assert cfg.backend is InferenceBackend.COMPILED
        assert cfg.max_batch_size == 16
        assert cfg.use_fp16 is True


class TestInferenceRequest:
    def test_auto_request_id_length(self):
        req = InferenceRequest(prompt="hello")
        assert len(req.request_id) == 8

    def test_unique_request_ids(self):
        reqs = [InferenceRequest(prompt=f"p{i}") for i in range(20)]
        ids = {r.request_id for r in reqs}
        assert len(ids) == 20

    def test_defaults(self):
        req = InferenceRequest(prompt="hi")
        assert req.max_new_tokens == 256
        assert req.temperature == 1.0
        assert req.top_p == 1.0

    def test_frozen(self):
        req = InferenceRequest(prompt="hi")
        with pytest.raises(Exception):
            req.prompt = "other"  # type: ignore[misc]


class TestInferenceResponse:
    def test_fields(self):
        resp = InferenceResponse(
            request_id="abc",
            text="out",
            latency_ms=1.5,
            tokens_generated=1,
            backend_used=InferenceBackend.EAGER,
        )
        assert resp.request_id == "abc"
        assert resp.text == "out"
        assert resp.latency_ms == 1.5
        assert resp.tokens_generated == 1
        assert resp.backend_used is InferenceBackend.EAGER

    def test_frozen(self):
        resp = InferenceResponse("a", "b", 1.0, 1, InferenceBackend.EAGER)
        with pytest.raises(Exception):
            resp.text = "x"  # type: ignore[misc]


class TestInferenceEngine:
    def test_generate_returns_response(self):
        engine = InferenceEngine(InferenceConfig())
        resp = engine.generate(InferenceRequest(prompt="hello world"))
        assert isinstance(resp, InferenceResponse)

    def test_generate_stub_text(self):
        engine = InferenceEngine(InferenceConfig())
        resp = engine.generate(InferenceRequest(prompt="hello"))
        assert resp.text.startswith("Generated:")

    def test_generate_uses_fn(self):
        def fn(prompt, max_new_tokens, temperature, top_p):
            return f"fn:{prompt}"

        engine = InferenceEngine(InferenceConfig(), generate_fn=fn)
        resp = engine.generate(InferenceRequest(prompt="abc"))
        assert resp.text == "fn:abc"

    def test_generate_fn_receives_args(self):
        seen = {}

        def fn(prompt, max_new_tokens, temperature, top_p):
            seen["prompt"] = prompt
            seen["mnt"] = max_new_tokens
            seen["temp"] = temperature
            seen["top_p"] = top_p
            return "ok"

        engine = InferenceEngine(InferenceConfig(), generate_fn=fn)
        engine.generate(InferenceRequest(prompt="p", max_new_tokens=10, temperature=0.7, top_p=0.9))
        assert seen == {"prompt": "p", "mnt": 10, "temp": 0.7, "top_p": 0.9}

    def test_generate_latency_positive(self):
        engine = InferenceEngine(InferenceConfig())
        resp = engine.generate(InferenceRequest(prompt="hi"))
        assert resp.latency_ms >= 0.0

    def test_generate_tokens_count(self):
        engine = InferenceEngine(InferenceConfig(), generate_fn=lambda *a: "one two three")
        resp = engine.generate(InferenceRequest(prompt="x"))
        assert resp.tokens_generated == 3

    def test_generate_backend_used(self):
        cfg = InferenceConfig(backend=InferenceBackend.COMPILED)
        engine = InferenceEngine(cfg)
        resp = engine.generate(InferenceRequest(prompt="x"))
        assert resp.backend_used is InferenceBackend.COMPILED

    def test_generate_request_id_preserved(self):
        engine = InferenceEngine(InferenceConfig())
        req = InferenceRequest(prompt="hi")
        resp = engine.generate(req)
        assert resp.request_id == req.request_id

    def test_batch_generate_length(self):
        engine = InferenceEngine(InferenceConfig())
        reqs = [InferenceRequest(prompt=f"p{i}") for i in range(5)]
        resps = engine.batch_generate(reqs)
        assert len(resps) == 5

    def test_batch_generate_empty(self):
        engine = InferenceEngine(InferenceConfig())
        assert engine.batch_generate([]) == []

    def test_batch_generate_preserves_ids(self):
        engine = InferenceEngine(InferenceConfig())
        reqs = [InferenceRequest(prompt=f"p{i}") for i in range(3)]
        resps = engine.batch_generate(reqs)
        for req, resp in zip(reqs, resps):
            assert req.request_id == resp.request_id

    def test_warmup_returns_list(self):
        engine = InferenceEngine(InferenceConfig(warmup_steps=4))
        lats = engine.warmup("sample")
        assert isinstance(lats, list)
        assert len(lats) == 4

    def test_warmup_zero_steps(self):
        engine = InferenceEngine(InferenceConfig(warmup_steps=0))
        assert engine.warmup() == []

    def test_warmup_latencies_non_negative(self):
        engine = InferenceEngine(InferenceConfig(warmup_steps=2))
        lats = engine.warmup("sample")
        assert all(line >= 0.0 for line in lats)

    def test_throughput_zero_on_empty(self):
        engine = InferenceEngine(InferenceConfig())
        assert engine.throughput_tokens_per_s([]) == 0.0

    def test_throughput_positive(self):
        engine = InferenceEngine(InferenceConfig(), generate_fn=lambda *a: "a b c d")
        resps = engine.batch_generate([InferenceRequest(prompt=f"p{i}") for i in range(3)])
        tp = engine.throughput_tokens_per_s(resps)
        assert tp >= 0.0

    def test_throughput_formula(self):
        engine = InferenceEngine(InferenceConfig())
        resps = [
            InferenceResponse("1", "x", 1000.0, 10, InferenceBackend.EAGER),
            InferenceResponse("2", "y", 1000.0, 10, InferenceBackend.EAGER),
        ]
        tp = engine.throughput_tokens_per_s(resps)
        assert tp == pytest.approx(10.0, abs=0.01)

    def test_throughput_zero_latency(self):
        engine = InferenceEngine(InferenceConfig())
        resps = [InferenceResponse("1", "x", 0.0, 5, InferenceBackend.EAGER)]
        assert engine.throughput_tokens_per_s(resps) == 0.0

    def test_stub_truncates_long_prompt(self):
        engine = InferenceEngine(InferenceConfig())
        long_prompt = "x" * 200
        resp = engine.generate(InferenceRequest(prompt=long_prompt))
        assert "..." in resp.text

    def test_generate_fn_none_uses_stub(self):
        engine = InferenceEngine(InferenceConfig(), generate_fn=None)
        resp = engine.generate(InferenceRequest(prompt="hi"))
        assert "Generated:" in resp.text

    def test_unique_response_ids(self):
        engine = InferenceEngine(InferenceConfig())
        resps = engine.batch_generate([InferenceRequest(prompt=f"p{i}") for i in range(10)])
        ids = {r.request_id for r in resps}
        assert len(ids) == 10

    def test_config_stored(self):
        cfg = InferenceConfig(max_batch_size=7)
        engine = InferenceEngine(cfg)
        assert engine.config.max_batch_size == 7

    def test_generate_fn_stored(self):
        def fn(*a):
            return "y"

        engine = InferenceEngine(InferenceConfig(), generate_fn=fn)
        assert engine.generate_fn is fn


class TestRegistry:
    def test_default_present(self):
        assert "default" in INFERENCE_ENGINE_REGISTRY

    def test_default_constructs(self):
        cls = INFERENCE_ENGINE_REGISTRY["default"]
        engine = cls(InferenceConfig())
        assert isinstance(engine, InferenceEngine)
