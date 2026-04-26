"""Tests for src/eval/eval_pipeline.py"""

from src.eval.eval_pipeline import (
    EVAL_PIPELINE_REGISTRY,
    EvalPipeline,
    EvalStage,
    PipelineResult,
)

# ---------------------------------------------------------------------------
# EvalStage enum values
# ---------------------------------------------------------------------------


def test_eval_stage_preprocess():
    assert EvalStage.PREPROCESS == "preprocess"


def test_eval_stage_evaluate():
    assert EvalStage.EVALUATE == "evaluate"


def test_eval_stage_postprocess():
    assert EvalStage.POSTPROCESS == "postprocess"


def test_eval_stage_aggregate():
    assert EvalStage.AGGREGATE == "aggregate"


def test_eval_stage_count():
    assert len(EvalStage) == 4


# ---------------------------------------------------------------------------
# PipelineResult fields
# ---------------------------------------------------------------------------


def test_pipeline_result_fields():
    pr = PipelineResult(stage_results={"s1": {"score": 0.5}}, final_score=0.5)
    assert pr.stage_results == {"s1": {"score": 0.5}}
    assert pr.final_score == 0.5


def test_pipeline_result_default_metadata():
    pr = PipelineResult(stage_results={}, final_score=0.0)
    assert isinstance(pr.metadata, dict)
    assert pr.metadata == {}


def test_pipeline_result_custom_metadata():
    pr = PipelineResult(stage_results={}, final_score=1.0, metadata={"key": "val"})
    assert pr.metadata["key"] == "val"


# ---------------------------------------------------------------------------
# EvalPipeline construction
# ---------------------------------------------------------------------------


def test_eval_pipeline_default_cache_enabled():
    ep = EvalPipeline()
    assert ep._cache_results is True


def test_eval_pipeline_cache_disabled():
    ep = EvalPipeline(cache_results=False)
    assert ep._cache_results is False


def test_eval_pipeline_starts_with_no_stages():
    ep = EvalPipeline()
    assert ep.stages() == []


# ---------------------------------------------------------------------------
# add_stage and stages()
# ---------------------------------------------------------------------------


def test_add_stage_then_stages_round_trip():
    ep = EvalPipeline()
    ep.add_stage("preproc", lambda x: x, EvalStage.PREPROCESS)
    assert ep.stages() == ["preproc"]


def test_add_multiple_stages_preserves_order():
    ep = EvalPipeline()
    ep.add_stage("first", lambda x: x, EvalStage.PREPROCESS)
    ep.add_stage("second", lambda x: x, EvalStage.EVALUATE)
    ep.add_stage("third", lambda x: x, EvalStage.AGGREGATE)
    assert ep.stages() == ["first", "second", "third"]


def test_stages_returns_list():
    ep = EvalPipeline()
    assert isinstance(ep.stages(), list)


def test_stages_empty_pipeline():
    ep = EvalPipeline()
    assert ep.stages() == []


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_returns_pipeline_result():
    ep = EvalPipeline()
    ep.add_stage("s1", lambda x: {"score": 0.5}, EvalStage.EVALUATE)
    result = ep.run({"input": "val"})
    assert isinstance(result, PipelineResult)


def test_run_final_score_from_last_stage():
    ep = EvalPipeline()
    ep.add_stage("preproc", lambda x: {"data": x}, EvalStage.PREPROCESS)
    ep.add_stage("eval", lambda x: {"score": 0.75}, EvalStage.EVALUATE)
    result = ep.run({"input": "val"})
    assert result.final_score == 0.75


def test_run_final_score_zero_when_no_score_key():
    ep = EvalPipeline()
    ep.add_stage("s1", lambda x: {"result": "ok"}, EvalStage.EVALUATE)
    result = ep.run({"input": "val"})
    assert result.final_score == 0.0


def test_run_final_score_zero_with_no_stages():
    ep = EvalPipeline()
    result = ep.run({"input": "val"})
    assert result.final_score == 0.0


def test_run_stage_results_contains_stage_names():
    ep = EvalPipeline()
    ep.add_stage("alpha", lambda x: {"score": 0.1}, EvalStage.PREPROCESS)
    ep.add_stage("beta", lambda x: {"score": 0.9}, EvalStage.EVALUATE)
    result = ep.run({"x": 1})
    assert "alpha" in result.stage_results
    assert "beta" in result.stage_results


def test_run_executes_stages_in_order():
    execution_order = []

    def stage_fn_factory(name):
        def fn(x):
            execution_order.append(name)
            return {"score": 0.5}

        return fn

    ep = EvalPipeline()
    ep.add_stage("first", stage_fn_factory("first"), EvalStage.PREPROCESS)
    ep.add_stage("second", stage_fn_factory("second"), EvalStage.EVALUATE)
    ep.add_stage("third", stage_fn_factory("third"), EvalStage.AGGREGATE)
    ep.run({"x": 1})
    assert execution_order == ["first", "second", "third"]


def test_run_passes_output_to_next_stage():
    results = []

    def stage_a(x):
        return {"value": 42}

    def stage_b(x):
        results.append(x.get("value"))
        return {"score": 0.5}

    ep = EvalPipeline()
    ep.add_stage("a", stage_a, EvalStage.PREPROCESS)
    ep.add_stage("b", stage_b, EvalStage.EVALUATE)
    ep.run({"start": True})
    assert results == [42]


def test_run_handles_stage_exception_gracefully():
    def bad_stage(x):
        raise RuntimeError("stage error")

    ep = EvalPipeline()
    ep.add_stage("bad", bad_stage, EvalStage.EVALUATE)
    # should not raise
    result = ep.run({"x": 1})
    assert isinstance(result, PipelineResult)


# ---------------------------------------------------------------------------
# cached_result()
# ---------------------------------------------------------------------------


def test_cached_result_none_before_run():
    ep = EvalPipeline()
    ep.add_stage("s", lambda x: {"score": 0.5}, EvalStage.EVALUATE)
    assert ep.cached_result({"key": "val"}) is None


def test_cached_result_returns_result_after_run():
    ep = EvalPipeline()
    ep.add_stage("s", lambda x: {"score": 0.5}, EvalStage.EVALUATE)
    inputs = {"key": "val"}
    ep.run(inputs)
    cached = ep.cached_result(inputs)
    assert cached is not None
    assert isinstance(cached, PipelineResult)


def test_cached_result_same_as_run_result():
    ep = EvalPipeline()
    ep.add_stage("s", lambda x: {"score": 0.88}, EvalStage.EVALUATE)
    inputs = {"k": "v"}
    run_result = ep.run(inputs)
    cached = ep.cached_result(inputs)
    assert cached is run_result


def test_cached_result_none_when_cache_disabled():
    ep = EvalPipeline(cache_results=False)
    ep.add_stage("s", lambda x: {"score": 0.5}, EvalStage.EVALUATE)
    ep.run({"x": 1})
    assert ep.cached_result({"x": 1}) is None


def test_cached_result_different_inputs_not_cached():
    ep = EvalPipeline()
    ep.add_stage("s", lambda x: {"score": 0.5}, EvalStage.EVALUATE)
    ep.run({"a": 1})
    assert ep.cached_result({"b": 2}) is None


# ---------------------------------------------------------------------------
# clear_cache()
# ---------------------------------------------------------------------------


def test_clear_cache_returns_count():
    ep = EvalPipeline()
    ep.add_stage("s", lambda x: {"score": 0.5}, EvalStage.EVALUATE)
    ep.run({"x": 1})
    ep.run({"y": 2})
    count = ep.clear_cache()
    assert count == 2


def test_clear_cache_empties_cache():
    ep = EvalPipeline()
    ep.add_stage("s", lambda x: {"score": 0.5}, EvalStage.EVALUATE)
    ep.run({"x": 1})
    ep.clear_cache()
    assert ep.cached_result({"x": 1}) is None


def test_clear_cache_returns_zero_when_empty():
    ep = EvalPipeline()
    count = ep.clear_cache()
    assert count == 0


def test_clear_cache_allows_re_run():
    call_count = [0]

    def counting_stage(x):
        call_count[0] += 1
        return {"score": 0.5}

    ep = EvalPipeline()
    ep.add_stage("s", counting_stage, EvalStage.EVALUATE)
    inputs = {"x": 1}
    ep.run(inputs)
    ep.clear_cache()
    ep.run(inputs)
    assert call_count[0] == 2  # called twice because cache was cleared


# ---------------------------------------------------------------------------
# EVAL_PIPELINE_REGISTRY
# ---------------------------------------------------------------------------


def test_eval_pipeline_registry_has_default():
    assert "default" in EVAL_PIPELINE_REGISTRY


def test_eval_pipeline_registry_default_is_pipeline():
    assert isinstance(EVAL_PIPELINE_REGISTRY["default"], EvalPipeline)


def test_eval_pipeline_registry_is_dict():
    assert isinstance(EVAL_PIPELINE_REGISTRY, dict)
