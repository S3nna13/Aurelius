"""Tests for src/persona/persona_registry.py and src/persona/persona_router.py."""

import pytest

from src.persona.persona_registry import (
    PersonaAlreadyRegisteredError,
    PersonaNotFoundError,
    UnifiedPersonaRegistry,
)
from src.persona.persona_router import PersonaRouter, RoutingResult
from src.persona.unified_persona import (
    PersonaDomain,
    PersonaFacet,
    PersonaTone,
    UnifiedPersona,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return UnifiedPersonaRegistry()


@pytest.fixture
def sample_persona():
    return UnifiedPersona(
        id="test-general",
        name="Test General",
        domain=PersonaDomain.GENERAL,
        description="A general test persona",
        system_prompt="Be helpful.",
    )


@pytest.fixture
def coding_persona():
    return UnifiedPersona(
        id="test-coding",
        name="Test Coding",
        domain=PersonaDomain.CODING,
        description="A coding test persona",
        system_prompt="Write code.",
        tone=PersonaTone.TECHNICAL,
        allowed_tools=("read", "write", "run"),
        facets=(PersonaFacet("agent_mode", {"mode": "code"}),),
    )


@pytest.fixture
def security_persona():
    return UnifiedPersona(
        id="test-security",
        name="Test Security",
        domain=PersonaDomain.SECURITY,
        description="A security test persona",
        system_prompt="Be secure.",
        facets=(PersonaFacet("security", {"mode": "defensive"}),),
    )


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


def test_register_and_get(registry, sample_persona):
    registry.register(sample_persona)
    assert registry.get("test-general") == sample_persona


def test_register_raises_on_duplicate(registry, sample_persona):
    registry.register(sample_persona)
    with pytest.raises(PersonaAlreadyRegisteredError):
        registry.register(sample_persona)


def test_get_raises_when_missing(registry):
    with pytest.raises(PersonaNotFoundError):
        registry.get("missing")


def test_get_error_includes_known_ids(registry, sample_persona):
    registry.register(sample_persona)
    with pytest.raises(PersonaNotFoundError) as exc_info:
        registry.get("missing")
    assert "test-general" in str(exc_info.value)


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------


def test_unregister_removes_persona(registry, sample_persona):
    registry.register(sample_persona)
    registry.unregister("test-general")
    with pytest.raises(PersonaNotFoundError):
        registry.get("test-general")


def test_unregister_raises_when_missing(registry):
    with pytest.raises(PersonaNotFoundError):
        registry.unregister("missing")


def test_unregister_clears_current(registry, sample_persona):
    registry.register(sample_persona)
    registry.switch_to("test-general")
    registry.unregister("test-general")
    assert registry.current() is None


# ---------------------------------------------------------------------------
# get_or_default
# ---------------------------------------------------------------------------


def test_get_or_default_returns_persona(registry, sample_persona):
    registry.register(sample_persona)
    assert registry.get_or_default("test-general") == sample_persona


def test_get_or_default_returns_default(registry, sample_persona):
    registry.register(sample_persona)
    result = registry.get_or_default("missing")
    assert result == sample_persona


def test_get_or_default_raises_when_empty(registry):
    with pytest.raises(PersonaNotFoundError):
        registry.get_or_default("missing")


# ---------------------------------------------------------------------------
# list_personas
# ---------------------------------------------------------------------------


def test_list_personas_empty(registry):
    assert registry.list_personas() == []


def test_list_personas_returns_all(registry, sample_persona, coding_persona):
    registry.register(sample_persona)
    registry.register(coding_persona)
    personas = registry.list_personas()
    assert len(personas) == 2
    assert sample_persona in personas
    assert coding_persona in personas


def test_list_personas_by_domain(registry, sample_persona, coding_persona):
    registry.register(sample_persona)
    registry.register(coding_persona)
    general = registry.list_personas(domain=PersonaDomain.GENERAL)
    coding = registry.list_personas(domain=PersonaDomain.CODING)
    assert general == [sample_persona]
    assert coding == [coding_persona]


def test_list_personas_by_unknown_domain(registry):
    assert registry.list_personas(domain=PersonaDomain.THREAT_INTEL) == []


# ---------------------------------------------------------------------------
# list_persona_ids
# ---------------------------------------------------------------------------


def test_list_persona_ids_empty(registry):
    assert registry.list_persona_ids() == []


def test_list_persona_ids_sorted(registry, sample_persona, coding_persona):
    registry.register(sample_persona)
    registry.register(coding_persona)
    ids = registry.list_persona_ids()
    assert ids == ["test-coding", "test-general"]


def test_list_persona_ids_by_domain(registry, sample_persona, coding_persona):
    registry.register(sample_persona)
    registry.register(coding_persona)
    assert registry.list_persona_ids(domain=PersonaDomain.CODING) == ["test-coding"]


# ---------------------------------------------------------------------------
# default
# ---------------------------------------------------------------------------


def test_default_prefers_aurelius_general(registry):
    general = UnifiedPersona(
        id="aurelius-general",
        name="Aurelius",
        domain=PersonaDomain.GENERAL,
        description="General",
        system_prompt="Help.",
    )
    other = UnifiedPersona(
        id="other",
        name="Other",
        domain=PersonaDomain.GENERAL,
        description="Other",
        system_prompt="Other.",
    )
    registry.register(other)
    registry.register(general)
    assert registry.default() == general


def test_default_falls_back_to_first(registry, sample_persona):
    registry.register(sample_persona)
    assert registry.default() == sample_persona


def test_default_raises_when_empty(registry):
    with pytest.raises(PersonaNotFoundError):
        registry.default()


# ---------------------------------------------------------------------------
# switch_to / current
# ---------------------------------------------------------------------------


def test_switch_to_sets_current(registry, sample_persona):
    registry.register(sample_persona)
    result = registry.switch_to("test-general")
    assert result == sample_persona
    assert registry.current() == sample_persona


def test_switch_to_raises_when_missing(registry):
    with pytest.raises(PersonaNotFoundError):
        registry.switch_to("missing")


def test_current_returns_none_when_no_switch(registry):
    assert registry.current() is None


# ---------------------------------------------------------------------------
# find_by_tool
# ---------------------------------------------------------------------------


def test_find_by_tool_with_allowed(registry, sample_persona, coding_persona):
    registry.register(sample_persona)
    registry.register(coding_persona)
    results = registry.find_by_tool("read")
    assert coding_persona in results


def test_find_by_tool_no_restrictions(registry, sample_persona):
    registry.register(sample_persona)
    results = registry.find_by_tool("any_tool")
    assert sample_persona in results


def test_find_by_tool_no_match(registry, coding_persona):
    registry.register(coding_persona)
    results = registry.find_by_tool("delete")
    assert coding_persona not in results


# ---------------------------------------------------------------------------
# find_by_facet
# ---------------------------------------------------------------------------


def test_find_by_facet_found(registry, sample_persona, coding_persona):
    registry.register(sample_persona)
    registry.register(coding_persona)
    results = registry.find_by_facet("agent_mode")
    assert coding_persona in results
    assert sample_persona not in results


def test_find_by_facet_none(registry, sample_persona):
    registry.register(sample_persona)
    assert registry.find_by_facet("security") == []


# ---------------------------------------------------------------------------
# find_by_domain
# ---------------------------------------------------------------------------


def test_find_by_domain(registry, sample_persona, coding_persona):
    registry.register(sample_persona)
    registry.register(coding_persona)
    assert registry.find_by_domain(PersonaDomain.CODING) == [coding_persona]


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_basic(registry, sample_persona):
    registry.register(sample_persona)
    messages = registry.build_messages("test-general", "Hello")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Be helpful."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


def test_build_messages_with_history(registry, sample_persona):
    registry.register(sample_persona)
    history = [{"role": "user", "content": "Previous"}, {"role": "assistant", "content": "Answer"}]
    messages = registry.build_messages("test-general", "Hello", history=history)
    assert len(messages) == 4
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Previous"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Answer"


def test_build_messages_raises_when_missing(registry):
    with pytest.raises(PersonaNotFoundError):
        registry.build_messages("missing", "Hello")


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------


def test_validate_response_no_contracts(registry, sample_persona):
    registry.register(sample_persona)
    ok, errors = registry.validate_response("test-general", {"foo": "bar"})
    assert ok is True
    assert errors == []


def test_validate_response_string_input(registry, sample_persona):
    from src.persona.unified_persona import OutputContract
    persona = UnifiedPersona(
        id="contract-test",
        name="Contract Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        output_contracts=(OutputContract("general", {}, ("field",)),),
    )
    registry.register(persona)
    ok, errors = registry.validate_response("contract-test", "plain text")
    assert ok is True
    assert errors == []


def test_validate_response_with_contract(registry):
    from src.persona.unified_persona import OutputContract
    persona = UnifiedPersona(
        id="contract-test",
        name="Contract Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        output_contracts=(OutputContract("general", {}, ("field",)),),
    )
    registry.register(persona)
    ok, errors = registry.validate_response("contract-test", {"field": "value"})
    assert ok is True
    assert errors == []


def test_validate_response_missing_field(registry):
    from src.persona.unified_persona import OutputContract
    persona = UnifiedPersona(
        id="contract-test",
        name="Contract Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        output_contracts=(OutputContract("general", {}, ("field",)),),
    )
    registry.register(persona)
    ok, errors = registry.validate_response("contract-test", {})
    assert ok is False
    assert errors == ["missing required field: field"]


def test_validate_response_with_query_type(registry):
    from src.persona.unified_persona import OutputContract
    persona = UnifiedPersona(
        id="contract-test",
        name="Contract Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        output_contracts=(
            OutputContract("type_a", {}, ("a",)),
            OutputContract("type_b", {}, ("b",)),
        ),
    )
    registry.register(persona)
    ok, errors = registry.validate_response("contract-test", {"a": 1}, query_type="type_a")
    assert ok is True
    assert errors == []


def test_validate_response_unknown_query_type(registry):
    from src.persona.unified_persona import OutputContract
    persona = UnifiedPersona(
        id="contract-test",
        name="Contract Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="prompt",
        output_contracts=(OutputContract("type_a", {}, ("a",)),),
    )
    registry.register(persona)
    ok, errors = registry.validate_response("contract-test", {}, query_type="missing")
    assert ok is False
    assert "unknown output contract" in errors[0]


# ---------------------------------------------------------------------------
# Dunder methods
# ---------------------------------------------------------------------------


def test_len(registry, sample_persona):
    assert len(registry) == 0
    registry.register(sample_persona)
    assert len(registry) == 1


def test_contains(registry, sample_persona):
    registry.register(sample_persona)
    assert "test-general" in registry
    assert "missing" not in registry


def test_repr(registry, sample_persona, coding_persona):
    registry.register(sample_persona)
    registry.register(coding_persona)
    r = repr(registry)
    assert "UnifiedPersonaRegistry" in r
    assert "2 personas" in r
    assert "general" in r
    assert "coding" in r


# ---------------------------------------------------------------------------
# PersonaRouter
# ---------------------------------------------------------------------------


def _setup_router(registry):
    from src.persona.builtins import ALL_BUILTINS
    for p in ALL_BUILTINS:
        if p.id not in registry:
            registry.register(p)
    return PersonaRouter(registry)


def test_router_cve_pattern(registry):
    router = _setup_router(registry)
    result = router.analyze("What about CVE-2024-1234?")
    assert result.persona_id == "aurelius-threatintel"
    assert result.domain == PersonaDomain.THREAT_INTEL
    assert result.confidence == 0.95
    assert result.fallback_used is False


def test_router_mitre_pattern(registry):
    router = _setup_router(registry)
    result = router.analyze("Explain T1059.001")
    assert result.persona_id == "aurelius-threatintel"
    assert result.confidence == 0.95


def test_router_threat_actor(registry):
    router = _setup_router(registry)
    result = router.analyze("Tell me about Lazarus Group")
    assert result.persona_id == "aurelius-threatintel"
    assert result.confidence == 0.85


def test_router_hash(registry):
    router = _setup_router(registry)
    result = router.analyze("Check hash d41d8cd98f00b204e9800998ecf8427e")
    assert result.persona_id == "aurelius-threatintel"
    assert result.confidence == 0.7


def test_router_redteam_keywords(registry):
    router = _setup_router(registry)
    result = router.analyze("I need a red team penetration test exploit")
    assert result.persona_id == "aurelius-redteam"
    assert result.domain == PersonaDomain.SECURITY
    assert result.confidence == 0.8


def test_router_blueteam_keywords(registry):
    router = _setup_router(registry)
    result = router.analyze("Our SOC detected an incident, need detection rules")
    assert result.persona_id == "aurelius-blueteam"
    assert result.domain == PersonaDomain.SECURITY
    assert result.confidence == 0.8


def test_router_purpleteam_keywords(registry):
    router = _setup_router(registry)
    result = router.analyze("vulnerability and threat actor analysis")
    assert result.persona_id == "aurelius-purpleteam"
    assert result.domain == PersonaDomain.SECURITY


def test_router_coding_keywords(registry):
    router = _setup_router(registry)
    result = router.analyze("How do I implement this function and debug the class?")
    assert result.persona_id == "aurelius-coding"
    assert result.fallback_used is False


def test_router_fallback(registry):
    router = _setup_router(registry)
    result = router.analyze("hello world")
    assert result.persona_id == "aurelius-general"
    assert result.fallback_used is True
    assert result.confidence == 0.0


def test_router_returns_unified_persona(registry):
    router = _setup_router(registry)
    persona = router.route("hello")
    assert isinstance(persona, UnifiedPersona)
    assert persona.id == "aurelius-general"


def test_router_matched_keywords_populated(registry):
    router = _setup_router(registry)
    result = router.analyze("red team pentest")
    assert "red team" in result.matched_keywords or "pentest" in result.matched_keywords


def test_router_matched_patterns_for_cve(registry):
    router = _setup_router(registry)
    result = router.analyze("CVE-2023-1234")
    assert any("CVE-2023-1234" in p for p in result.matched_patterns)


def test_routing_result_is_frozen():
    result = RoutingResult(
        persona_id="test",
        domain=PersonaDomain.GENERAL,
        confidence=0.5,
        matched_keywords=[],
        matched_patterns=[],
        fallback_used=False,
    )
    with pytest.raises(Exception):
        result.persona_id = "new"
