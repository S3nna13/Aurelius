"""Tests for extensibility manager."""

from src.agent.extensibility_manager import (
    ExtensibilityManager,
    Hook,
    HookEvent,
    MCPServerConfig,
    MCPTool,
    Plugin,
    Skill,
)


def test_hook_registration():
    mgr = ExtensibilityManager()
    hook = Hook(event=HookEvent.PRE_TOOL_USE, handler=lambda ctx: None, name="test_hook")
    mgr.register_hook(hook)
    assert mgr.hooks.total_hooks > 0


def test_hook_dispatch():
    mgr = ExtensibilityManager()
    results: list[str] = []

    def handler(ctx):
        results.append("called")
        return {"status": "ok"}

    mgr.register_hook(Hook(event=HookEvent.TURN_START, handler=handler))
    mgr.dispatch_hook(HookEvent.TURN_START, {})
    assert len(results) == 1


def test_blocking_hook():
    mgr = ExtensibilityManager()
    blocked = False

    def blocking(ctx):
        nonlocal blocked
        blocked = True
        return {"block": True}

    mgr.register_hook(Hook(event=HookEvent.PRE_TOOL_USE, handler=blocking, blocking=True))
    mgr.dispatch_hook(HookEvent.PRE_TOOL_USE, {})
    assert blocked


def test_mcp_server():
    mgr = ExtensibilityManager()
    cfg = MCPServerConfig(name="test_server", transport="stdio", command="echo")
    server = mgr.register_mcp(cfg)
    assert server.connect() is True
    server.disconnect()
    assert not server.connected


def test_skill_matching():
    skill = Skill(name="code_review", trigger="review", instructions="Review this code carefully.")
    assert skill.matches("Please review this PR") is True
    assert skill.matches("Hello world") is False


def test_skill_apply():
    skill = Skill(name="debug", trigger="debug", instructions="Add print statements.")
    result = skill.apply("debug the app")
    assert "print statements" in result


def test_plugin_activation():
    mgr = ExtensibilityManager()
    plugin = Plugin("test_plugin")
    plugin.add_hook(Hook(event=HookEvent.TURN_END, handler=lambda ctx: None, name="plugin_hook"))
    mgr.register_plugin(plugin)
    assert "test_plugin" in mgr.plugins


def test_tool_pool_assembly():
    mgr = ExtensibilityManager()
    cfg = MCPServerConfig(name="tools", command="echo")
    server = mgr.register_mcp(cfg)
    server.connect()
    server.tools.append(MCPTool(name="search", description="Search", input_schema={}))
    plugin = Plugin("tools_plugin")
    plugin.add_tool(MCPTool(name="review", description="Review", input_schema={}))
    mgr.register_plugin(plugin)
    tools = mgr.assemble_tool_pool()
    assert isinstance(tools, list)
    assert len(tools) == 2
    assert mgr.n_tools_available == 2


def test_apply_skills():
    mgr = ExtensibilityManager()
    mgr.register_skill(Skill("review", "review", "Conduct a thorough review."))
    result = mgr.apply_skills("Can you review this code?")
    assert "review" in result


def test_plugin_hooks_are_registered():
    mgr = ExtensibilityManager()
    plugin = Plugin("hook_plugin")
    plugin.add_hook(Hook(event=HookEvent.TURN_START, handler=lambda ctx: None, name="turn"))
    mgr.register_plugin(plugin)
    assert mgr.hooks.total_hooks == 1
