"""Integration tests for the dispatch-task primitive via the agent surface."""

from __future__ import annotations


def test_dispatch_task_exposed_on_agent_surface():
    import src.agent as agent

    for name in (
        "DispatchOutcome",
        "DispatchReport",
        "DispatchTask",
        "Dispatcher",
        "classify_error",
    ):
        assert hasattr(agent, name), name
        assert name in agent.__all__, name

    assert "dispatch_task" in agent.AGENT_LOOP_REGISTRY
    assert agent.AGENT_LOOP_REGISTRY["dispatch_task"] is agent.Dispatcher


def test_prior_agent_entries_intact():
    import src.agent as agent

    assert "xml" in agent.TOOL_CALL_PARSER_REGISTRY
    assert "json" in agent.TOOL_CALL_PARSER_REGISTRY
    for key in ("react", "safe_dispatch", "beam_plan", "task_decompose"):
        assert key in agent.AGENT_LOOP_REGISTRY, key


def test_config_flag_defaults_off():
    from src.model.config import AureliusConfig

    cfg = AureliusConfig()
    assert cfg.agent_dispatch_task_enabled is False


def test_end_to_end_dispatch_three_inputs():
    from src.agent import Dispatcher, DispatchTask

    class UpperTask(DispatchTask):
        name = "upper"

        def build_prompt(self, input_item):
            return f"upper:{input_item}"

        def process_result(self, input_item, result):
            return result.upper()

        def finalize(self, processed_results):
            return "|".join(processed_results)

    def llm(prompt, schema):
        return prompt.split(":", 1)[1]

    disp = Dispatcher(max_workers=3, per_task_timeout_s=2.0)
    report = disp.dispatch(UpperTask(), ["alpha", "beta", "gamma"], llm)

    assert report.task_name == "upper"
    assert len(report.outcomes) == 3
    assert report.status_counts.get("success") == 3
    # Order in finalize matches success order, but all should be uppercased.
    assert set(report.finalized.split("|")) == {"ALPHA", "BETA", "GAMMA"}
    assert report.circuit_open is False
