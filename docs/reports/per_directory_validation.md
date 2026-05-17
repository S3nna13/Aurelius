# Per-directory validation report

Command shape: `.venv/bin/python -m pytest <paths> -q --tb=line --maxfail=5`

## eval

- Status: PASS
- Elapsed: 88.1s
- Paths: `tests/eval/`

```text
  /Users/christienantonio/aurelius/src/memory/episodic_memory.py:2: DeprecationWarning: Importing from 'plugins' is deprecated. Use 'src.plugins' instead.
    from plugins.memory.episodic_memory import *  # noqa: F401, F403

tests/eval/test_ab_comparison.py::test_metric_accuracy
  /Users/christienantonio/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: DeprecationWarning: Importing from 'aurelius' is deprecated. Use 'src' instead.
    return _bootstrap._gcd_import(name[level:], package, level)

tests/eval/test_ab_comparison.py::test_metric_accuracy
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```

## memory

- Status: PASS
- Elapsed: 3.1s
- Paths: `tests/memory/`

```text
  /Users/christienantonio/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: DeprecationWarning: Importing from 'aurelius' is deprecated. Use 'src' instead.
    return _bootstrap._gcd_import(name[level:], package, level)

tests/memory/test_associative_memory.py::TestPattern::test_pattern_creation
  /Users/christienantonio/aurelius/src/serving/aurelius_server.py:50: DeprecationWarning: Importing from 'agent' is deprecated. Use 'src.agent' instead.
    from agent.command_dispatcher import CommandDispatcher

tests/memory/test_associative_memory.py::TestPattern::test_pattern_creation
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```

## runtime

- Status: PASS
- Elapsed: 8.2s
- Paths: `tests/runtime/`

```text
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

tests/runtime/test_compile_manager.py: 14 warnings
  /Users/christienantonio/aurelius/.venv/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

tests/runtime/test_torch_profiler_wrapper.py::TestAureliusProfiler::test_context_manager_runs
  /Users/christienantonio/aurelius/.venv/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
    _warn_once(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```

## serving_gateway

- Status: FAIL exit=1
- Elapsed: 4.1s
- Paths: `tests/serving/ tests/gateway/`

```text
tests/serving/test_agent_cockpit.py::test_health_endpoint
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/serving/test_aurelius_server.py::test_unknown_api_route_fallback_to_spa
FAILED tests/serving/test_engine_loader.py::TestBuildEngineMock::test_build_engine_mock
FAILED tests/serving/test_engine_loader.py::TestBuildEngineEmptyBackend::test_build_engine_empty_backend_uses_mock
FAILED tests/serving/test_engine_loader.py::TestBuildEngineAgentic::test_build_engine_agentic_returns_callable_and_label
FAILED tests/serving/test_engine_loader.py::TestBuildEngineVLLM::test_build_engine_vllm_passes_speculative_decoding
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 5 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## agent

- Status: FAIL exit=1
- Elapsed: 28.2s
- Paths: `tests/agent/`

```text
  /Users/christienantonio/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: DeprecationWarning: Importing from 'aurelius' is deprecated. Use 'src' instead.
    return _bootstrap._gcd_import(name[level:], package, level)

tests/agent/test_absolute_zero.py::test_extract_code_python
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/agent/test_plugin_sandbox.py::TestPluginSandboxRun::test_safe_callable
FAILED tests/agent/test_plugin_sandbox.py::TestPluginSandboxRun::test_exception_caught
FAILED tests/agent/test_plugin_sandbox.py::TestPluginSandboxRunHook::test_safe_hooks
```

## alignment

- Status: PASS
- Elapsed: 32.5s
- Paths: `tests/alignment/`

```text
  /Users/christienantonio/aurelius/src/serving/aurelius_server.py:50: DeprecationWarning: Importing from 'agent' is deprecated. Use 'src.agent' instead.
    from agent.command_dispatcher import CommandDispatcher

tests/alignment/test_absolute_zero.py::test_config_defaults
  /Users/christienantonio/aurelius/src/memory/episodic_memory.py:2: DeprecationWarning: Importing from 'plugins' is deprecated. Use 'src.plugins' instead.
    from plugins.memory.episodic_memory import *  # noqa: F401, F403

tests/alignment/test_absolute_zero.py::test_config_defaults
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```

## inference

- Status: FAIL exit=1
- Elapsed: 38.8s
- Paths: `tests/inference/`

```text
  /Users/christienantonio/aurelius/src/memory/episodic_memory.py:2: DeprecationWarning: Importing from 'plugins' is deprecated. Use 'src.plugins' instead.
    from plugins.memory.episodic_memory import *  # noqa: F401, F403

tests/inference/test_abstention.py::test_token_entropy_shape
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/inference/test_code_execution.py::test_execute_python_simple - A...
FAILED tests/inference/test_code_execution.py::test_execute_python_success_flag
FAILED tests/inference/test_code_execution.py::test_code_generation_evaluator_pass_rate
```

## model_quick

- Status: PASS
- Elapsed: 74.5s
- Paths: `tests/model/ --ignore=tests/model/test_transformer.py`

```text
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

tests/model/test_norm_tracking.py::TestLayerStatsHook::test_get_stats_has_gradient_key
  /Users/christienantonio/aurelius/tests/model/test_norm_tracking.py:41: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
    loss.backward()

tests/model/test_recurrent_memory_v3.py::test_memory_detached_between_segments
  /Users/christienantonio/aurelius/tests/model/test_recurrent_memory_v3.py:313: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/build/aten/src/ATen/core/TensorBody.h:499.)
    assert seg0.grad is None or seg0.grad.abs().sum().item() == 0.0, (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```

# Affected-batch rerun after baseline fixes

## rerun: serving_gateway

- Status: FAIL exit=1
- Elapsed: 24.1s
- Paths: `tests/serving/ tests/gateway/`

```text

tests/serving/test_agent_cockpit.py::test_health_endpoint
  /Users/christienantonio/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: DeprecationWarning: Importing from 'aurelius' is deprecated. Use 'src' instead.
    return _bootstrap._gcd_import(name[level:], package, level)

tests/serving/test_agent_cockpit.py::test_health_endpoint
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/serving/test_web_ui.py::test_post_api_chat_proxies_to_configured_upstream
```

## rerun: agent

- Status: PASS
- Elapsed: 27.1s
- Paths: `tests/agent/`

```text
  /Users/christienantonio/aurelius/src/memory/episodic_memory.py:2: DeprecationWarning: Importing from 'plugins' is deprecated. Use 'src.plugins' instead.
    from plugins.memory.episodic_memory import *  # noqa: F401, F403

tests/agent/test_absolute_zero.py::test_extract_code_python
  /Users/christienantonio/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/lib/python3.12/importlib/__init__.py:90: DeprecationWarning: Importing from 'aurelius' is deprecated. Use 'src' instead.
    return _bootstrap._gcd_import(name[level:], package, level)

tests/agent/test_absolute_zero.py::test_extract_code_python
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```

## rerun: inference

- Status: PASS
- Elapsed: 43.8s
- Paths: `tests/inference/`

```text
  /Users/christienantonio/aurelius/src/serving/aurelius_server.py:50: DeprecationWarning: Importing from 'agent' is deprecated. Use 'src.agent' instead.
    from agent.command_dispatcher import CommandDispatcher

tests/inference/test_abstention.py::test_token_entropy_shape
  /Users/christienantonio/aurelius/src/memory/episodic_memory.py:2: DeprecationWarning: Importing from 'plugins' is deprecated. Use 'src.plugins' instead.
    from plugins.memory.episodic_memory import *  # noqa: F401, F403

tests/inference/test_abstention.py::test_token_entropy_shape
  /Users/christienantonio/aurelius/src/tools/code_runner_tool.py:2: DeprecationWarning: Importing from 'tools' is deprecated. Use 'src.tools' instead.
    from tools.code_runner_tool import *  # noqa: F401, F403

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```
# Final affected-batch status

- `tests/serving/ tests/gateway/`: PASS after web-ui proxy test update and serving baseline fixes.
- `tests/agent/`: PASS after legacy agent sandbox denied-import check was narrowed to actually used globals.
- `tests/inference/`: PASS after moving code-execution worker to module scope for macOS spawn pickling.

```text
serving/gateway: passed
agent: passed
inference: passed
```
