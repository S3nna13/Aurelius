from src.multiagent.character_personality import (
    ARCHITECT,
    DETECTIVE,
    GUARDIAN,
    CharacterPersonality,
    PersonalityConfig,
    PersonalityRouter,
)


class TestPersonalityConfig:
    def test_architect_defaults(self):
        assert ARCHITECT.name == "architect"
        assert ARCHITECT.temperature == 0.3
        assert "Leonardo" in ARCHITECT.character

    def test_detective_defaults(self):
        assert DETECTIVE.name == "detective"
        assert DETECTIVE.temperature == 0.2

    def test_guardian_defaults(self):
        assert GUARDIAN.name == "guardian"
        assert GUARDIAN.temperature == 0.1


class TestCharacterPersonality:
    def test_create_from_config(self):
        cfg = PersonalityConfig(name="test", character="TestBot", role="Tester")
        p = CharacterPersonality(cfg)
        assert p.name == "test"
        assert p.task_count == 0

    def test_build_system_prompt(self):
        cfg = PersonalityConfig(
            name="sage",
            character="Sage",
            role="Advisor",
            traits=["wise", "patient"],
            expertise=["strategy", "planning"],
            communication_style="thoughtful",
        )
        p = CharacterPersonality(cfg)
        prompt = p.build_system_prompt()
        assert "Sage" in prompt
        assert "Advisor" in prompt
        assert "wise" in prompt
        assert "strategy" in prompt

    def test_build_system_prompt_explicit(self):
        cfg = PersonalityConfig(name="custom", character="Custom", role="Helper", system_prompt="You are a helper.")
        p = CharacterPersonality(cfg)
        assert p.build_system_prompt() == "You are a helper."

    def test_add_message(self):
        p = CharacterPersonality(PersonalityConfig(name="bot", character="Bot", role="Assistant"))
        p.add_message("user", "hello")
        p.add_message("assistant", "hi")
        ctx = p.get_context()
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"

    def test_status(self):
        p = CharacterPersonality(PersonalityConfig(name="monitor", character="Monitor", role="Watcher"))
        status = p.status()
        assert status["name"] == "monitor"
        assert status["task_count"] == 0


class TestPersonalityRouter:
    def test_register(self):
        router = PersonalityRouter()
        p = CharacterPersonality(PersonalityConfig(name="test_bot", character="TB", role="Test"))
        router.register(p)
        assert router.get("test_bot") is p

    def test_route_by_keyword(self):
        router = PersonalityRouter()
        router.register(CharacterPersonality(ARCHITECT))
        router.register(CharacterPersonality(DETECTIVE))
        router.register(CharacterPersonality(GUARDIAN))
        result = router.route("debug this bug")
        assert result.name == "detective"

    def test_route_default(self):
        router = PersonalityRouter()
        router.register(CharacterPersonality(ARCHITECT))
        result = router.route("unknown task")
        assert result.name == "architect"

    def test_route_empty_registry(self):
        router = PersonalityRouter()
        result = router.route("create something")
        assert result is not None

    def test_list_personalities(self):
        router = PersonalityRouter()
        router.register(CharacterPersonality(ARCHITECT))
        router.register(CharacterPersonality(DETECTIVE))
        names = router.list_personalities()
        assert "architect" in names
        assert "detective" in names

    def test_reset(self):
        router = PersonalityRouter()
        router.register(CharacterPersonality(ARCHITECT))
        router.reset()
        assert router.list_personalities() == []
