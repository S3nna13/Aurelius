from __future__ import annotations

from src.routing.intent import Intent
from src.routing.model_router import (
    DEFAULT_CODE_MODEL,
    DEFAULT_LARGE_MODEL,
    DEFAULT_MEDIUM_MODEL,
    DEFAULT_PRIVATE_MODEL,
    DEFAULT_SMALL_MODEL,
    DEFAULT_VISION_MODEL,
    ModelRouter,
    RouteAction,
    TaskProfile,
)


def _profile(
    *,
    user_tier: str = "free",
    content: str = "please explain the mechanism",
    message_length: int | None = None,
    intent: Intent | None = None,
    has_code: bool = False,
    has_image: bool = False,
    has_pii: bool = False,
) -> TaskProfile:
    return TaskProfile(
        user_id="user-123",
        user_tier=user_tier,
        content=content,
        intent=intent,
        message_length=len(content) if message_length is None else message_length,
        has_code=has_code,
        has_image=has_image,
        has_pii=has_pii,
    )


def test_default_catalog_is_local_first() -> None:
    router = ModelRouter()
    catalog = router.list_available_models()

    assert set(catalog) == {
        DEFAULT_SMALL_MODEL,
        DEFAULT_MEDIUM_MODEL,
        DEFAULT_LARGE_MODEL,
        DEFAULT_PRIVATE_MODEL,
        DEFAULT_VISION_MODEL,
        DEFAULT_CODE_MODEL,
    }
    assert all(entry["provider"] == "local" for entry in catalog.values())
    assert catalog[DEFAULT_VISION_MODEL]["modalities"] == ["text", "image"]
    assert catalog[DEFAULT_CODE_MODEL]["tier"] == "code"


def test_simple_free_query_routes_to_small_local_model() -> None:
    router = ModelRouter()
    decision = router.route(
        _profile(
            user_tier="free",
            content="please explain the mechanism",
            intent=Intent(category="general", confidence=0.2),
        )
    )

    assert decision.action == RouteAction.ROUTE_TO_SMALL
    assert decision.model == DEFAULT_SMALL_MODEL
    assert decision.cost_estimate == 0.0


def test_pro_query_routes_to_medium_local_model() -> None:
    router = ModelRouter()
    decision = router.route(
        _profile(
            user_tier="pro",
            content="please explain the mechanism in more detail",
            intent=Intent(category="general", confidence=0.2),
        )
    )

    assert decision.action == RouteAction.ROUTE_TO_MEDIUM
    assert decision.model == DEFAULT_MEDIUM_MODEL
    assert decision.cost_estimate == 0.0


def test_long_pro_query_routes_to_large_local_model() -> None:
    router = ModelRouter()
    content = " ".join(["analysis"] * 30)
    decision = router.route(
        _profile(
            user_tier="pro",
            content=content,
            message_length=200,
            intent=Intent(category="analysis", confidence=0.9),
        )
    )

    assert decision.action == RouteAction.ROUTE_TO_MEDIUM
    assert decision.model == DEFAULT_LARGE_MODEL
    assert decision.cost_estimate == 0.0


def test_code_tasks_use_code_specialist_model() -> None:
    router = ModelRouter()
    decision = router.route(
        _profile(
            user_tier="free",
            content="refactor this function with type hints",
            intent=Intent(
                category="code",
                confidence=0.95,
                requires_tools=True,
                requires_reasoning=True,
                reasoning_depth="medium",
            ),
            has_code=True,
        )
    )

    assert decision.action == RouteAction.ROUTE_TO_CODE
    assert decision.model == DEFAULT_CODE_MODEL
    assert decision.requires_tools is True
    assert decision.requires_reasoning is True


def test_multimodal_queries_route_to_local_vision_model() -> None:
    router = ModelRouter()
    decision = router.route(
        _profile(
            user_tier="free",
            content="look at this diagram and explain it",
            intent=Intent(
                category="multimodal",
                confidence=0.95,
                requires_multimodal=True,
                modalities=["text", "image"],
            ),
            has_image=True,
        )
    )

    assert decision.action == RouteAction.ROUTE_TO_VISION
    assert decision.model == DEFAULT_VISION_MODEL
    assert decision.cost_estimate == 0.0


def test_privacy_sensitive_enterprise_uses_private_local_model() -> None:
    router = ModelRouter()
    decision = router.route(
        _profile(
            user_tier="enterprise",
            content="here is a confidential customer summary",
            intent=Intent(category="sensitive", confidence=0.95, privacy_sensitive=True),
            has_pii=True,
        )
    )

    assert decision.action == RouteAction.ROUTE_TO_LARGE
    assert decision.model == DEFAULT_PRIVATE_MODEL
    assert decision.requires_security_check is True
