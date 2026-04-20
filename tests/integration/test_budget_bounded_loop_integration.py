"""Integration: budget-bounded agent loop registry + config flags."""

from __future__ import annotations


def test_budget_bounded_registered():
    import src.agent as agent

    assert "budget_bounded" in agent.AGENT_LOOP_REGISTRY
    assert agent.AGENT_LOOP_REGISTRY["budget_bounded"] is agent.BudgetBoundedLoop


def test_prior_agent_registries_intact():
    import src.agent as agent

    assert "react" in agent.AGENT_LOOP_REGISTRY
    assert "dispatch_task" in agent.AGENT_LOOP_REGISTRY


def test_config_flags_default_off():
    from src.model.config import AureliusConfig

    cfg = AureliusConfig()
    assert cfg.agent_budget_bounded_loop_enabled is False
    assert cfg.agent_budget_max_tool_invocations == 8


def test_smoke_run_with_flag_agnostic():
    from src.agent import BudgetBoundedLoop

    loop = BudgetBoundedLoop(
        lambda m: "<final_answer>ok</final_answer>",
        {},
        max_llm_turns=4,
        max_tool_invocations=1,
    )
    assert loop.run("hi").final_answer == "ok"
