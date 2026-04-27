"""Tests for src/persona/unified_persona.py."""

import pytest

from src.persona.unified_persona import (
    Guardrail,
    GuardrailScope,
    GuardrailSeverity,
    IntentMapping,
    OutputContract,
    PersonaDomain,
    PersonaFacet,
    PersonaTone,
    ResponseStyle,
    UnifiedPersona,
    WorkflowStage,
)


# ---------------------------------------------------------------------------
# Enum smoke tests
# ---------------------------------------------------------------------------


def test_persona_domain_values():
    assert PersonaDomain.GENERAL == "general"
    assert PersonaDomain.SECURITY == "security"
    assert PersonaDomain.THREAT_INTEL == "threat_intel"
    assert PersonaDomain.AGENT == "agent"
    assert PersonaDomain.CODING == "coding"


def test_persona_tone_values():
    assert PersonaTone.FORMAL == "formal"
    assert PersonaTone.CASUAL == "casual"
    assert PersonaTone.TECHNICAL == "technical"
    assert PersonaTone.EMPATHETIC == "empathetic"
    assert PersonaTone.CONCISE == "concise"


def test_response_style_values():
    assert ResponseStyle.CONCISE == "concise"
    assert ResponseStyle.STRUCTURED == "structured"
    assert ResponseStyle.VERBOSE == "verbose"
    assert ResponseStyle.METHODICAL == "methodical"
    assert ResponseStyle.ADAPTIVE == "adaptive"


def test_guardrail_severity_values():
    assert GuardrailSeverity.LOW == "low"
    assert GuardrailSeverity.MEDIUM == "medium"
    assert GuardrailSeverity.HIGH == "high"
    assert GuardrailSeverity.CRITICAL == "critical"


def test_guardrail_scope_values():
    assert GuardrailScope.ALL == "all"
    assert GuardrailScope.OFFENSIVE == "offensive"
    assert GuardrailScope.DEFENSIVE == "defensive"
    assert GuardrailScope.GENERAL == "general"


# ---------------------------------------------------------------------------
# UnifiedPersona creation and defaults
# ---------------------------------------------------------------------------


def test_minimal_persona():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="A test persona",
        system_prompt="Be helpful.",
    )
    assert p.id == "test"
    assert p.name == "Test"
    assert p.domain == PersonaDomain.GENERAL
    assert p.description == "A test persona"
    assert p.system_prompt == "Be helpful."
    assert p.tone == PersonaTone.FORMAL
    assert p.response_style == ResponseStyle.CONCISE
    assert p.temperature == 0.7
    assert p.workflow_stages == ()
    assert p.output_contracts == ()
    assert p.guardrails == ()
    assert p.intent_mappings == ()
    assert p.facets == ()
    assert p.allowed_tools == ()
    assert p.priority == 4
    assert p.immutable_prompt is False


def test_full_persona():
    p = UnifiedPersona(
        id="full-test",
        name="Full Test",
        domain=PersonaDomain.SECURITY,
        description="Full config",
        system_prompt="Secure.",
        tone=PersonaTone.TECHNICAL,
        response_style=ResponseStyle.STRUCTURED,
        temperature=0.2,
        workflow_stages=(WorkflowStage("stage1", "desc"),),
        output_contracts=(OutputContract("contract1", {"a": "b"}, ("a",)),),
        guardrails=(
            Guardrail("g1", "Never do X.", GuardrailSeverity.CRITICAL, GuardrailScope.ALL),
        ),
        intent_mappings=(IntentMapping("intent1", "behavior1", "contract1"),),
        facets=(PersonaFacet("security", {"mode": "defensive"}),),
        allowed_tools=("read", "write"),
        priority=1,
        immutable_prompt=True,
    )
    assert p.tone == PersonaTone.TECHNICAL
    assert p.response_style == ResponseStyle.STRUCTURED
    assert p.temperature == 0.2
    assert len(p.workflow_stages) == 1
    assert len(p.output_contracts) == 1
    assert len(p.guardrails) == 1
    assert len(p.intent_mappings) == 1
    assert len(p.facets) == 1
    assert p.allowed_tools == ("read", "write")
    assert p.priority == 1
    assert p.immutable_prompt is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_id_empty():
    with pytest.raises(ValueError):
        UnifiedPersona(
            id="",
            name="Test",
            domain=PersonaDomain.GENERAL,
            description="desc",
            system_prompt="prompt",
        )


def test_invalid_id_too_long():
    with pytest.raises(ValueError):
        UnifiedPersona(
            id="a" * 129,
            name="Test",
            domain=PersonaDomain.GENERAL,
            description="desc",
            system_prompt="prompt",
        )


def test_invalid_id_special_chars():
    with pytest.raises(ValueError):
        UnifiedPersona(
            id="test@id",
            name="Test",
            domain=PersonaDomain.GENERAL,
            description="desc",
            system_prompt="prompt",
        )


def test_invalid_name_empty():
    with pytest.raises(ValueError):
        UnifiedPersona(
            id="test",
            name="   ",
            domain=PersonaDomain.GENERAL,
            description="desc",
            system_prompt="prompt",
        )


def test_temperature_too_low():
    with pytest.raises(ValueError):
        UnifiedPersona(
            id="test",
            name="Test",
            domain=PersonaDomain.GENERAL,
            description="desc",
            system_prompt="prompt",
            temperature=-0.1,
        )


def test_temperature_too_high():
    with pytest.raises(ValueError):
        UnifiedPersona(
            id="test",
            name="Test",
            domain=PersonaDomain.GENERAL,
            description="desc",
            system_prompt="prompt",
            temperature=2.1,
        )


def test_temperature_boundary_zero():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        temperature=0.0,
    )
    assert p.temperature == 0.0


def test_temperature_boundary_two():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        temperature=2.0,
    )
    assert p.temperature == 2.0


def test_priority_too_low():
    with pytest.raises(ValueError):
        UnifiedPersona(
            id="test",
            name="Test",
            domain=PersonaDomain.GENERAL,
            description="desc",
            system_prompt="prompt",
            priority=-1,
        )


def test_priority_too_high():
    with pytest.raises(ValueError):
        UnifiedPersona(
            id="test",
            name="Test",
            domain=PersonaDomain.GENERAL,
            description="desc",
            system_prompt="prompt",
            priority=5,
        )


def test_priority_boundary_zero():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        priority=0,
    )
    assert p.priority == 0


def test_priority_boundary_four():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        priority=4,
    )
    assert p.priority == 4


# ---------------------------------------------------------------------------
# Facet helpers
# ---------------------------------------------------------------------------


def test_has_facet():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        facets=(PersonaFacet("security", {"mode": "defensive"}),),
    )
    assert p.has_facet("security") is True
    assert p.has_facet("constitution") is False


def test_get_facet_found():
    facet = PersonaFacet("security", {"mode": "defensive"})
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        facets=(facet,),
    )
    assert p.get_facet("security") == facet


def test_get_facet_not_found():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
    )
    assert p.get_facet("security") is None


def test_with_facets_adds_new():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
    )
    new_facet = PersonaFacet("security", {"mode": "offensive"})
    p2 = p.with_facets(new_facet)
    assert p2.has_facet("security")
    assert not p.has_facet("security")


def test_with_facets_replaces_existing():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        facets=(PersonaFacet("security", {"mode": "defensive"}),),
    )
    new_facet = PersonaFacet("security", {"mode": "offensive"})
    p2 = p.with_facets(new_facet)
    assert p2.get_facet("security").config["mode"] == "offensive"
    assert p.get_facet("security").config["mode"] == "defensive"


def test_with_facets_preserves_others():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        facets=(
            PersonaFacet("security", {"mode": "defensive"}),
            PersonaFacet("constitution", {"dimensions": ["helpfulness"]}),
        ),
    )
    new_facet = PersonaFacet("harm_filter", {"categories": "all"})
    p2 = p.with_facets(new_facet)
    assert p2.has_facet("security")
    assert p2.has_facet("constitution")
    assert p2.has_facet("harm_filter")


# ---------------------------------------------------------------------------
# Output contract helpers
# ---------------------------------------------------------------------------


def test_get_output_contract_found():
    contract = OutputContract("findings", {"id": "str"}, ("id",))
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        output_contracts=(contract,),
    )
    assert p.get_output_contract("findings") == contract


def test_get_output_contract_not_found():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
    )
    assert p.get_output_contract("missing") is None


# ---------------------------------------------------------------------------
# Guardrail helpers
# ---------------------------------------------------------------------------


def test_has_guardrail():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        guardrails=(Guardrail("g1", "text"),),
    )
    assert p.has_guardrail("g1") is True
    assert p.has_guardrail("g2") is False


# ---------------------------------------------------------------------------
# Truncated system prompt
# ---------------------------------------------------------------------------


def test_truncated_system_prompt_short():
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="short",
    )
    assert p.truncated_system_prompt == "short"


def test_truncated_system_prompt_long():
    long_prompt = "x" * 50_000
    p = UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt=long_prompt,
    )
    assert len(p.truncated_system_prompt) == 32_768
    assert p.truncated_system_prompt == long_prompt[:32_768]


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------


def test_workflow_stage_is_frozen():
    stage = WorkflowStage("name", "desc")
    with pytest.raises(Exception):
        stage.name = "new"


def test_output_contract_is_frozen():
    contract = OutputContract("name", {}, ())
    with pytest.raises(Exception):
        contract.name = "new"


def test_guardrail_is_frozen():
    guardrail = Guardrail("id", "text")
    with pytest.raises(Exception):
        guardrail.id = "new"


def test_intent_mapping_is_frozen():
    mapping = IntentMapping("intent", "behavior")
    with pytest.raises(Exception):
        mapping.intent = "new"


def test_persona_facet_is_frozen():
    facet = PersonaFacet("type", {})
    with pytest.raises(Exception):
        facet.facet_type = "new"


# ---------------------------------------------------------------------------
# Valid id patterns
# ---------------------------------------------------------------------------


def test_valid_ids():
    for valid_id in ("a", "A", "test-123", "test_123", "x" * 128):
        p = UnifiedPersona(
            id=valid_id,
            name="Test",
            domain=PersonaDomain.GENERAL,
            description="desc",
            system_prompt="prompt",
        )
        assert p.id == valid_id
