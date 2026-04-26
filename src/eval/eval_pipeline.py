"""Evaluation pipeline: preprocessing, batch evaluation, aggregation, caching."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class EvalStage(StrEnum):
    PREPROCESS = "preprocess"
    EVALUATE = "evaluate"
    POSTPROCESS = "postprocess"
    AGGREGATE = "aggregate"


@dataclass
class PipelineResult:
    stage_results: dict
    final_score: float
    metadata: dict = field(default_factory=dict)


class EvalPipeline:
    def __init__(self, cache_results: bool = True):
        self._cache_results = cache_results
        self._stages: list = []  # list of (name, fn, stage_type)
        self._cache: dict = {}

    def add_stage(self, name: str, fn: Callable, stage_type: EvalStage) -> None:
        self._stages.append((name, fn, stage_type))

    def run(self, inputs: dict) -> PipelineResult:
        cache_key = None
        if self._cache_results:
            cache_key = json.dumps(sorted(inputs.items()))
            if cache_key in self._cache:
                return self._cache[cache_key]

        stage_results = {}
        current = inputs
        for name, fn, stage_type in self._stages:
            try:
                output = fn(current)
            except Exception as e:
                output = {"error": str(e)}
            stage_results[name] = output if isinstance(output, dict) else {"result": output}
            current = stage_results[name]

        # final_score from last stage result if present
        final_score = 0.0
        if self._stages:
            last_name = self._stages[-1][0]
            last_result = stage_results.get(last_name, {})
            if isinstance(last_result, dict) and "score" in last_result:
                final_score = float(last_result["score"])

        result = PipelineResult(
            stage_results=stage_results,
            final_score=final_score,
        )

        if self._cache_results and cache_key is not None:
            self._cache[cache_key] = result

        return result

    def cached_result(self, inputs: dict):
        if not self._cache_results:
            return None
        cache_key = json.dumps(sorted(inputs.items()))
        return self._cache.get(cache_key, None)

    def stages(self) -> list:
        return [name for name, fn, stage_type in self._stages]

    def clear_cache(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count


EVAL_PIPELINE_REGISTRY: dict = {"default": EvalPipeline()}
