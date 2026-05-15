#!/usr/bin/env python3
"""GREEN test: verify no circular imports after fix."""

import sys
import traceback

# Clear all agent-related modules
for mod_name in list(sys.modules.keys()):
    if "agent" in mod_name:
        del sys.modules[mod_name]

print("=== GREEN TEST: Verify circular imports are fixed ===\n")

# Test 1: Import agent and check registrations
print("Test 1: Import agent and check registrations...")
try:
    import agent

    # Check AGENT_LOOP_REGISTRY has plan_and_execute
    assert "plan_and_execute" in agent.AGENT_LOOP_REGISTRY, (
        f"Missing 'plan_and_execute' in AGENT_LOOP_REGISTRY. "
        f"Keys: {list(agent.AGENT_LOOP_REGISTRY.keys())}"
    )
    print("[OK] plan_and_execute registered in AGENT_LOOP_REGISTRY")

    # Check TOOL_REGISTRY has code_execution
    assert "code_execution" in agent.TOOL_REGISTRY, (
        f"Missing 'code_execution' in TOOL_REGISTRY. Keys: {list(agent.TOOL_REGISTRY.keys())}"
    )
    print("[OK] code_execution registered in TOOL_REGISTRY")

    # Verify PlanAndExecuteAgent is accessible from agent namespace
    assert hasattr(agent, "PlanAndExecuteAgent"), "PlanAndExecuteAgent not in agent namespace"
    print("[OK] PlanAndExecuteAgent accessible from agent namespace")

    # Verify CodeExecutionTool is accessible from agent namespace
    assert hasattr(agent, "CodeExecutionTool"), "CodeExecutionTool not in agent namespace"
    print("[OK] CodeExecutionTool accessible from agent namespace")

except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print()

# Test 2: Direct imports (as specified in task)
print("Test 2: from src.agent.plan_and_execute import PlanAndExecuteAgent...")
try:
    from src.agent.plan_and_execute import PlanAndExecuteAgent

    print(f"[OK] PlanAndExecuteAgent imported successfully: {PlanAndExecuteAgent}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Test 3: from src.agent.code_execution_tool import CodeExecutionTool...")
try:
    from src.agent.code_execution_tool import CodeExecutionTool

    print(f"[OK] CodeExecutionTool imported successfully: {CodeExecutionTool}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Combined import as specified in the task
print("Test 4: Combined import (task verification command)...")
from src.agent.code_execution_tool import CodeExecutionTool
from src.agent.plan_and_execute import PlanAndExecuteAgent

print("OK")

print()

# Test 5: Import via src.agent shim (backward compat)
print("Test 5: Import via src.agent shim (backward compat)...")
for mod_name in list(sys.modules.keys()):
    if "agent" in mod_name:
        del sys.modules[mod_name]

try:
    import src.agent

    assert hasattr(src.agent, "PlanAndExecuteAgent"), "PlanAndExecuteAgent not in src.agent"
    assert hasattr(src.agent, "CodeExecutionTool"), "CodeExecutionTool not in src.agent"
    assert "plan_and_execute" in src.agent.AGENT_LOOP_REGISTRY, (
        "plan_and_execute not in AGENT_LOOP_REGISTRY via src.agent"
    )
    assert "code_execution" in src.agent.TOOL_REGISTRY, (
        "code_execution not in TOOL_REGISTRY via src.agent"
    )
    print("[OK] All registrations available through src.agent shim")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=== ALL TESTS PASSED (GREEN) ===")
