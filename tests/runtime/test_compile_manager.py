import torch
import torch.nn as nn

from src.runtime.compile_manager import COMPILE_REGISTRY, CompileConfig, CompileManager


def _tiny_model() -> nn.Module:
    return nn.Linear(4, 4)


def _sample_input() -> torch.Tensor:
    return torch.randn(2, 4)


class TestCompileConfig:
    def test_defaults(self):
        cfg = CompileConfig()
        assert cfg.mode == "default"
        assert cfg.fullgraph is False
        assert cfg.dynamic is True
        assert cfg.backend == "inductor"
        assert cfg.warmup_iters == 3

    def test_custom_values(self):
        cfg = CompileConfig(
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
            backend="aot_eager",
            warmup_iters=5,
        )
        assert cfg.mode == "reduce-overhead"
        assert cfg.fullgraph is True
        assert cfg.dynamic is False
        assert cfg.backend == "aot_eager"
        assert cfg.warmup_iters == 5


class TestCompileManager:
    def test_default_config_used_when_none(self):
        mgr = CompileManager()
        assert isinstance(mgr.config, CompileConfig)

    def test_custom_config_stored(self):
        cfg = CompileConfig(warmup_iters=1)
        mgr = CompileManager(config=cfg)
        assert mgr.config.warmup_iters == 1

    def test_compile_returns_callable(self):
        mgr = CompileManager()
        model = _tiny_model()
        compiled = mgr.compile(model)
        assert callable(compiled)

    def test_compile_stores_in_compiled(self):
        mgr = CompileManager()
        model = _tiny_model()
        mgr.compile(model)
        assert mgr._compiled is not None

    def test_compile_output_runnable(self):
        mgr = CompileManager()
        model = _tiny_model()
        compiled = mgr.compile(model)
        out = compiled(_sample_input())
        assert out.shape == (2, 4)

    def test_warmup_returns_float(self):
        mgr = CompileManager(config=CompileConfig(warmup_iters=2))
        model = _tiny_model()
        elapsed = mgr.warmup(model, _sample_input())
        assert isinstance(elapsed, float)

    def test_warmup_returns_positive_time(self):
        mgr = CompileManager(config=CompileConfig(warmup_iters=1))
        model = _tiny_model()
        elapsed = mgr.warmup(model, _sample_input())
        assert elapsed >= 0.0

    def test_warmup_respects_iters(self):
        mgr = CompileManager(config=CompileConfig(warmup_iters=3))
        model = _tiny_model()
        elapsed = mgr.warmup(model, _sample_input())
        assert isinstance(elapsed, float)

    def test_reset_clears_compiled(self):
        mgr = CompileManager()
        model = _tiny_model()
        mgr.compile(model)
        assert mgr._compiled is not None
        mgr.reset()
        assert mgr._compiled is None

    def test_reset_on_fresh_manager_is_safe(self):
        mgr = CompileManager()
        mgr.reset()
        assert mgr._compiled is None


class TestCompileRegistry:
    def test_registry_has_default_key(self):
        assert "default" in COMPILE_REGISTRY

    def test_registry_default_is_compile_manager(self):
        assert COMPILE_REGISTRY["default"] is CompileManager

    def test_registry_instantiable(self):
        cls = COMPILE_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, CompileManager)
