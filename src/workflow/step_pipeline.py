import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepResult:
    step_name: str
    success: bool
    output: Any
    duration_ms: float
    error: str = ""


PipelineHook = Callable[[str, StepResult], None]


@dataclass
class PipelineStep:
    name: str
    fn: Callable
    retry_count: int = 0
    timeout_ms: float | None = None


class StepPipeline:
    def __init__(self) -> None:
        self._steps: list[PipelineStep] = []
        self._on_complete: list[PipelineHook] = []
        self._on_failure: list[PipelineHook] = []

    def add_step(
        self,
        name: str,
        fn: Callable,
        retry_count: int = 0,
        timeout_ms: float | None = None,
    ) -> "StepPipeline":
        self._steps.append(
            PipelineStep(name=name, fn=fn, retry_count=retry_count, timeout_ms=timeout_ms)
        )
        return self

    def on_step_complete(self, hook: PipelineHook) -> "StepPipeline":
        self._on_complete.append(hook)
        return self

    def on_failure(self, hook: PipelineHook) -> "StepPipeline":
        self._on_failure.append(hook)
        return self

    def run(self, initial_input: Any = None) -> list[StepResult]:
        results: list[StepResult] = []
        current = initial_input

        for step in self._steps:
            attempts = step.retry_count + 1
            last_error = ""
            start = time.monotonic()
            output: Any = None
            success = False

            for _ in range(attempts):
                try:
                    output = step.fn(current)
                    success = True
                    last_error = ""
                    break
                except Exception as exc:
                    last_error = str(exc)

            duration = (time.monotonic() - start) * 1000.0
            result = StepResult(
                step_name=step.name,
                success=success,
                output=output,
                duration_ms=duration,
                error=last_error,
            )
            results.append(result)

            for hook in self._on_complete:
                hook(step.name, result)

            if not success:
                for hook in self._on_failure:
                    hook(step.name, result)
                break

            current = output

        return results

    def summary(self, results: list[StepResult]) -> dict:
        passed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        total_duration = sum(r.duration_ms for r in results)
        return {
            "total_steps": len(results),
            "passed": passed,
            "failed": failed,
            "total_duration_ms": total_duration,
        }


STEP_PIPELINE_REGISTRY: dict[str, type[StepPipeline]] = {"default": StepPipeline}
