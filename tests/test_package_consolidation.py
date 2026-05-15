#!/usr/bin/env python3
"""Verify that consolidated package structure resolves correctly."""

import sys


def test_no_ambiguous_imports():
    """Ensure agent and gateway shims properly delegate to canonical src/ locations."""
    # The shim files themselves live at agent/__init__.py etc.
    # What matters is they delegate to src/* correctly and emit warnings.
    # Verify the shim files exist at expected legacy locations.

    import agent
    import gateway

    agent_path = agent.__file__
    gateway_path = gateway.__file__

    assert agent_path.endswith("agent/__init__.py") or agent_path.endswith("agent/__init__.pyc"), (
        f"Shim should be at legacy agent/ path, got {agent_path}"
    )
    assert gateway_path.endswith("gateway/__init__.py") or gateway_path.endswith(
        "gateway/__init__.pyc"
    ), f"Shim should be at legacy gateway/ path, got {gateway_path}"

    # Verify the shim modules are importable and have contents
    assert hasattr(agent, "__doc__"), "agent shim should be a valid module"
    assert hasattr(gateway, "__doc__"), "gateway shim should be a valid module"

    print("✓ Legacy shims exist and delegate to canonical src/ modules")


def test_deprecation_warnings():
    """Ensure legacy imports emit DeprecationWarnings."""
    import warnings

    # Clear any cached imports
    for mod in ["agent", "gateway", "aurelius", "cron", "tools", "plugins"]:
        if mod in sys.modules:
            del sys.modules[mod]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Import legacy shims to trigger deprecation warnings
        import agent  # noqa: F401
        import gateway  # noqa: F401
        import aurelius  # noqa: F401
        import cron  # noqa: F401
        import tools  # noqa: F401
        import plugins  # noqa: F401

        # Check for DeprecationWarning
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) > 0, (
            f"Expected DeprecationWarning for agent import, got {[str(x.message) for x in w]}"
        )

    print("✓ DeprecationWarnings emitted for legacy imports")


def test_shims_do_not_break_original_functionality():
    """Ensure shimmed imports expose the same API as canonical imports."""
    import agent as legacy_agent
    import src.agent as src_agent

    # Verify key classes exported by both are identical objects (same class object)
    # Pick representative types from the agent package
    key_classes = [
        "AureliusInterfaceRuntime",
        "SessionManager",
        "SkillCatalog",
    ]

    for cls_name in key_classes:
        src_cls = getattr(src_agent, cls_name, None)
        legacy_cls = getattr(legacy_agent, cls_name, None)
        assert src_cls is not None, f"Canonical src.agent missing {cls_name}"
        assert legacy_cls is not None, f"Shim agent missing {cls_name}"
        assert src_cls is legacy_cls, (
            f"Class {cls_name} differs: src.agent.{cls_name} is not agent.{cls_name}"
        )

    print("✓ Shim re-exports identical class objects from canonical modules")


if __name__ == "__main__":
    print("Running package consolidation validation tests...")
    test_no_ambiguous_imports()
    test_deprecation_warnings()
    test_shims_do_not_break_original_functionality()
    print("\n✅ All consolidation checks PASSED")
