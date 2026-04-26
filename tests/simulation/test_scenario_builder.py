"""Tests for scenario builder."""

from __future__ import annotations

from src.simulation.scenario_builder import SCENARIO_BUILDER_REGISTRY, ScenarioBuilder


class TestScenarioBuilder:
    def test_build_scenario(self):
        builder = ScenarioBuilder("scenario_1")
        builder.add_actor("agent_1", "buyer", {"budget": 100})
        builder.add_event(0, "agent_1", "bid", {"amount": 50})
        builder.set_environment("market", "open")
        scenario = builder.build()
        assert scenario["scenario_id"] == "scenario_1"
        assert "agent_1" in scenario["actors"]
        assert scenario["actors"]["agent_1"]["role"] == "buyer"
        assert scenario["actors"]["agent_1"]["params"] == {"budget": 100}
        assert len(scenario["events"]) == 1
        assert scenario["events"][0]["timestamp"] == 0
        assert scenario["events"][0]["actor"] == "agent_1"
        assert scenario["events"][0]["action"] == "bid"
        assert scenario["events"][0]["payload"] == {"amount": 50}
        assert scenario["environment"]["market"] == "open"

    def test_validate_pass(self):
        builder = ScenarioBuilder("scenario_1")
        builder.add_actor("agent_1", "buyer")
        builder.add_event(0, "agent_1", "bid")
        assert builder.validate() == []

    def test_validate_fail_empty_scenario_id(self):
        builder = ScenarioBuilder("")
        assert builder.validate() == ["scenario_id must be non-empty"]

    def test_validate_missing_actor(self):
        builder = ScenarioBuilder("scenario_1")
        builder.add_actor("agent_1", "buyer")
        builder.add_event(0, "agent_2", "bid")
        errors = builder.validate()
        assert any("Unknown actor: agent_2" in e for e in errors)

    def test_validate_negative_timestamp(self):
        builder = ScenarioBuilder("scenario_1")
        builder.add_actor("agent_1", "buyer")
        builder.add_event(-1, "agent_1", "bid")
        errors = builder.validate()
        assert any("Negative timestamp: -1" in e for e in errors)

    def test_validate_duplicate_actors(self):
        builder = ScenarioBuilder("scenario_1")
        builder.add_actor("agent_1", "buyer")
        builder.add_actor("agent_1", "seller")
        errors = builder.validate()
        assert any("Duplicate actor: agent_1" in e for e in errors)


class TestScenarioBuilderRegistry:
    def test_is_dict(self):
        assert isinstance(SCENARIO_BUILDER_REGISTRY, dict)

    def test_has_builder_key(self):
        assert "builder" in SCENARIO_BUILDER_REGISTRY

    def test_builder_maps_to_class(self):
        assert SCENARIO_BUILDER_REGISTRY["builder"] is ScenarioBuilder
