"""Aurelius agent surface: tool-call parsing and agent-loop registry.

This module is deliberately minimal: it exposes two registries other
surfaces populate and registers this surface's own tool-call parsers.
No heavyweight imports; no side effects beyond registry population.
"""

from __future__ import annotations

from typing import Callable

from .tool_call_parser import (
    JSONToolCallParser,
    ParsedToolCall,
    ParseResult,
    ToolCallParseError,
    UnifiedToolCallParser,
    XMLToolCallParser,
    detect_format,
    format_json,
    format_xml,
    parse_json,
    parse_xml,
)

#: Registry of named tool-call parser callables (``text -> list[ParsedToolCall]``).
#: Keys are format identifiers ("xml", "json"); additional siblings may
#: register more at import time.
TOOL_CALL_PARSER_REGISTRY: dict[str, Callable[[str], list[ParsedToolCall]]] = {}

#: Registry for agent-loop implementations. Populated by downstream
#: modules (react loop, plan-and-execute, etc.); left empty here.
AGENT_LOOP_REGISTRY: dict[str, Callable[..., object]] = {}


# Register this surface's parsers under their canonical keys.
TOOL_CALL_PARSER_REGISTRY["xml"] = parse_xml
TOOL_CALL_PARSER_REGISTRY["json"] = parse_json


# Register the ReAct loop under its canonical key. Import is deferred
# until after the registries exist so that cycles cannot form.
from .react_loop import AgentStep, AgentTrace, ReActLoop  # noqa: E402

AGENT_LOOP_REGISTRY["react"] = ReActLoop


__all__ = [
    "AGENT_LOOP_REGISTRY",
    "AgentStep",
    "AgentTrace",
    "ReActLoop",
    "JSONToolCallParser",
    "ParsedToolCall",
    "ParseResult",
    "TOOL_CALL_PARSER_REGISTRY",
    "ToolCallParseError",
    "UnifiedToolCallParser",
    "XMLToolCallParser",
    "detect_format",
    "format_json",
    "format_xml",
    "parse_json",
    "parse_xml",
]


# --- safe tool dispatcher (additive) ----------------------------------------
from .tool_registry_dispatcher import (  # noqa: E402
    SessionBudget,
    ToolInvocationResult,
    ToolRegistryDispatcher,
    ToolSpec,
)

AGENT_LOOP_REGISTRY["safe_dispatch"] = ToolRegistryDispatcher

__all__ += [
    "SessionBudget",
    "ToolInvocationResult",
    "ToolRegistryDispatcher",
    "ToolSpec",
]


# --- tree-of-thoughts beam planner (additive) -------------------------------
from .agent_planner import BeamPlanner, PlanNode  # noqa: E402

AGENT_LOOP_REGISTRY["beam_plan"] = BeamPlanner

__all__ += [
    "BeamPlanner",
    "PlanNode",
]


# --- tool error recovery (additive) -----------------------------------------
from .tool_error_recovery import (  # noqa: E402
    RecoveringDispatcher,
    RecoveryDecision,
    RecoveryPolicy,
    RecoveryStrategy,
)

__all__ += [
    "RecoveringDispatcher",
    "RecoveryDecision",
    "RecoveryPolicy",
    "RecoveryStrategy",
]


# --- repo context packer (additive) -----------------------------------------
from .repo_context_packer import (  # noqa: E402
    FileSnippet,
    RepoContext,
    RepoContextPacker,
)

__all__ += [
    "FileSnippet",
    "RepoContext",
    "RepoContextPacker",
]


# --- unified diff generator (additive) --------------------------------------
from .unified_diff_generator import (  # noqa: E402
    DiffResult,
    UnifiedDiffGenerator,
)

__all__ += [
    "DiffResult",
    "UnifiedDiffGenerator",
]


# --- shell command planner (additive) ---------------------------------------
from .shell_command_planner import (  # noqa: E402
    ALLOWLIST as SHELL_ALLOWLIST,
    DENYLIST as SHELL_DENYLIST,
    ShellCommand,
    ShellCommandPlanner,
    ShellPlan,
)

__all__ += [
    "SHELL_ALLOWLIST",
    "SHELL_DENYLIST",
    "ShellCommand",
    "ShellCommandPlanner",
    "ShellPlan",
]


# --- code execution sandbox (additive) --------------------------------------
from .code_execution_sandbox import (  # noqa: E402
    CodeExecutionSandbox,
    ExecutionResult,
)

__all__ += [
    "CodeExecutionSandbox",
    "ExecutionResult",
]


# --- code execution tool (additive) -----------------------------------------
from .code_execution_tool import (  # noqa: E402
    CODE_EXECUTION_TOOL_REGISTRY,
    CodeExecutionTool,
    ExecutionLanguage,
    ExecutionRequest,
    ExecutionResult as CodeExecutionResult,
)

__all__ += [
    "CODE_EXECUTION_TOOL_REGISTRY",
    "CodeExecutionTool",
    "ExecutionLanguage",
    "ExecutionRequest",
    "CodeExecutionResult",
]


# --- code test runner (additive) --------------------------------------------
from .code_test_runner import (  # noqa: E402
    CodeTestRunner,
    TestResult,
)

__all__ += [
    "CodeTestRunner",
    "TestResult",
]


# --- task decomposer (additive) ---------------------------------------------
from .task_decomposer import (  # noqa: E402
    SubTask,
    TaskDAG,
    TaskDecomposer,
    TaskDecompositionError,
)

AGENT_LOOP_REGISTRY["task_decompose"] = TaskDecomposer

__all__ += [
    "SubTask",
    "TaskDAG",
    "TaskDecomposer",
    "TaskDecompositionError",
]


# --- MCP client (additive) --------------------------------------------------
from .mcp_client import (  # noqa: E402
    MCP_PROTOCOL_VERSION,
    MCPClient,
    MCPPrompt,
    MCPProtocolError,
    MCPResource,
    MCPToolCallResult,
    MCPToolSpec,
)

__all__ += [
    "MCP_PROTOCOL_VERSION",
    "MCPClient",
    "MCPPrompt",
    "MCPProtocolError",
    "MCPResource",
    "MCPToolCallResult",
    "MCPToolSpec",
]


# --- code refactor tool (additive) ------------------------------------------
from .code_refactor_tool import (  # noqa: E402
    CodeRefactorTool,
    RefactorResult,
)

__all__ += [
    "CodeRefactorTool",
    "RefactorResult",
]


# --- toolformer data generation (additive) ----------------------------------
from .toolformer_data_gen import (  # noqa: E402
    Tool as ToolformerTool,
    ToolCallAnnotation,
    ToolformerConfig,
    ToolformerDataGenerator,
)

#: Registry for Toolformer data-generation classes.
TOOLFORMER_DATA_REGISTRY: dict[str, type] = {
    "toolformer": ToolformerDataGenerator,
}

__all__ += [
    "TOOLFORMER_DATA_REGISTRY",
    "ToolCallAnnotation",
    "ToolformerConfig",
    "ToolformerDataGenerator",
    "ToolformerTool",
]


# --- dispatch task (additive) -----------------------------------------------
from .dispatch_task import (  # noqa: E402
    DispatchOutcome,
    DispatchReport,
    DispatchTask,
    Dispatcher,
    classify_error,
)

AGENT_LOOP_REGISTRY["dispatch_task"] = Dispatcher

__all__ += [
    "DispatchOutcome",
    "DispatchReport",
    "DispatchTask",
    "Dispatcher",
    "classify_error",
]


# --- budget-bounded ReAct wrapper (additive) --------------------------------
from .budget_bounded_loop import (  # noqa: E402
    BudgetBoundedLoop,
    BudgetState,
    ToolInvocationBudgetError,
)

AGENT_LOOP_REGISTRY["budget_bounded"] = BudgetBoundedLoop

__all__ += [
    "BudgetBoundedLoop",
    "BudgetState",
    "ToolInvocationBudgetError",
]


# --- tool sandbox denylist (additive) ---------------------------------------
from .tool_sandbox_denylist import (  # noqa: E402
    DEFAULT_DENYLIST as TOOL_SANDBOX_DEFAULT_DENYLIST,
    DenyVerdict,
    DenylistCategory,
    DenylistRule,
    ToolSandboxDenylist,
)

#: Registry of pre-tool-execution policy guards. Keys are identifiers;
#: values are guard classes exposing ``evaluate(tool_name, tool_args)``.
TOOL_GUARD_REGISTRY: dict[str, type] = {
    "sandbox_denylist": ToolSandboxDenylist,
}

__all__ += [
    "DenyVerdict",
    "DenylistCategory",
    "DenylistRule",
    "TOOL_GUARD_REGISTRY",
    "TOOL_SANDBOX_DEFAULT_DENYLIST",
    "ToolSandboxDenylist",
]


# --- five-state lifecycle controller (additive) -----------------------------
from .five_state_controller import (  # noqa: E402
    AgentState,
    ControlContext,
    ControllerEvent,
    FiveStateController,
)

#: Registry of agent lifecycle supervisors (orthogonal to AGENT_LOOP_REGISTRY
#: which holds message-iteration loops). Keys are identifiers; values are the
#: supervisor classes.
AGENT_LIFECYCLE_REGISTRY: dict[str, type] = {
    "five_state": FiveStateController,
}

__all__ += [
    "AGENT_LIFECYCLE_REGISTRY",
    "AgentState",
    "ControlContext",
    "ControllerEvent",
    "FiveStateController",
]


# --- agent swarm (additive) -------------------------------------------------
from .agent_swarm import AgentSwarm, CriticalPathAnalyzer, SubAgentResult  # noqa: E402

AGENT_LOOP_REGISTRY["agent_swarm"] = AgentSwarm

__all__ += [
    "AgentSwarm",
    "CriticalPathAnalyzer",
    "SubAgentResult",
]


# --- web browse tool (additive) ---------------------------------------------
from .web_browse_tool import (  # noqa: E402
    DEFAULT_TOOL_DESCRIPTOR as WEB_BROWSE_TOOL_DESCRIPTOR,
    PrivateHostBlocked,
    UrlValidationError,
    WebBrowseTool,
    WebFetchResult,
    WebRequestSpec,
)

#: Registry of agent-invocable tool descriptors (JSON-schema style).
#: Keys are the tool name; values are the descriptor dict consumable by
#: the agent runtime / serving layer. Populated additively by sibling
#: modules.
TOOL_REGISTRY: dict[str, dict] = {
    WEB_BROWSE_TOOL_DESCRIPTOR["name"]: WEB_BROWSE_TOOL_DESCRIPTOR,
}

__all__ += [
    "PrivateHostBlocked",
    "TOOL_REGISTRY",
    "UrlValidationError",
    "WEB_BROWSE_TOOL_DESCRIPTOR",
    "WebBrowseTool",
    "WebFetchResult",
    "WebRequestSpec",
]


# --- plugin hook registry (additive) ----------------------------------------
from .plugin_hook import (  # noqa: E402
    HOOK_POINTS,
    HOOK_REGISTRY,
    PluginHookRegistry,
)

AGENT_LOOP_REGISTRY["plugin_hook"] = PluginHookRegistry

__all__ += [
    "HOOK_POINTS",
    "HOOK_REGISTRY",
    "PluginHookRegistry",
]


# --- preserve thinking ring buffer (additive) --------------------------------
from .preserve_thinking import (  # noqa: E402
    PreserveThinkingBuffer,
    PreserveThinkingConfig,
    ThinkingSnapshot,
)

AGENT_LOOP_REGISTRY["preserve_thinking"] = PreserveThinkingBuffer

__all__ += [
    "PreserveThinkingBuffer",
    "PreserveThinkingConfig",
    "ThinkingSnapshot",
]


# --- background executor (additive) -----------------------------------------
from .background_executor import (  # noqa: E402
    BackgroundExecutor,
    BackgroundExecutorConfig,
    BackgroundTask,
    TaskStatus,
)

AGENT_LOOP_REGISTRY["background_executor"] = BackgroundExecutor

__all__ += [
    "BackgroundExecutor",
    "BackgroundExecutorConfig",
    "BackgroundTask",
    "TaskStatus",
]


# --- swarm scaler (additive) ------------------------------------------------
from .swarm_scaler import (  # noqa: E402
    SwarmScaler,
    SwarmScalerConfig,
    WorkerStats,
    WorkItem,
)

AGENT_LOOP_REGISTRY["swarm_scaler"] = SwarmScaler

__all__ += [
    "SwarmScaler",
    "SwarmScalerConfig",
    "WorkerStats",
    "WorkItem",
]


# --- proactive trigger registry (additive) ----------------------------------
from .proactive_trigger import (  # noqa: E402
    ProactiveTriggerConfig,
    ProactiveTriggerRegistry,
    TriggerSpec,
    condition_trigger,
    interval_trigger,
)

AGENT_LOOP_REGISTRY["proactive_trigger"] = ProactiveTriggerRegistry

__all__ += [
    "ProactiveTriggerConfig",
    "ProactiveTriggerRegistry",
    "TriggerSpec",
    "condition_trigger",
    "interval_trigger",
]


# --- reflexion verbal RL agent (additive) -----------------------------------
from .reflexion_agent import (  # noqa: E402
    ReflexionAgent,
    ReflexionAttempt,
    ReflexionConfig,
    ReflexionMemory,
    ReflexionResult,
)

# AGENT_LOOP_REGISTRY["reflexion"] is set inside reflexion_agent.py itself.

__all__ += [
    "ReflexionAgent",
    "ReflexionAttempt",
    "ReflexionConfig",
    "ReflexionMemory",
    "ReflexionResult",
]


# --- multi-resource budget ledger (additive) --------------------------------
from .budget_ledger import (  # noqa: E402
    BUDGET_LEDGER_REGISTRY,
    BudgetExhaustedError,
    BudgetLedger,
    BudgetSeverity,
    LedgerError,
    LedgerSnapshot,
    ResourceLimit,
    get_ledger,
    list_ledgers,
    make_default_ledger,
    register_ledger,
)

__all__ += [
    "BUDGET_LEDGER_REGISTRY",
    "BudgetExhaustedError",
    "BudgetLedger",
    "BudgetSeverity",
    "LedgerError",
    "LedgerSnapshot",
    "ResourceLimit",
    "get_ledger",
    "list_ledgers",
    "make_default_ledger",
    "register_ledger",
]


# --- interface runtime and shell layer (additive) ---------------------------
_LAZY_INTERFACE_EXPORTS = {
    "AureliusInterfaceRuntime": (".interface_runtime", "AureliusInterfaceRuntime"),
    "SessionManager": (".session_manager", "SessionManager"),
    "SessionRecord": (".session_manager", "SessionRecord"),
    "WorkItem": (".session_manager", "WorkItem"),
    "SessionWorkItem": (".session_manager", "WorkItem"),
    "SessionJournal": (".session_journal", "SessionJournal"),
    "SessionJournalEntry": (".session_journal", "SessionJournalEntry"),
    "SessionJournalBranch": (".session_journal", "SessionJournalBranch"),
    "SessionJournalCompaction": (".session_journal", "SessionJournalCompaction"),
    "SkillCatalog": (".skill_catalog", "SkillCatalog"),
    "SkillCatalogEntry": (".skill_catalog", "SkillCatalogEntry"),
    "WorkflowRun": (".workflow_shell", "WorkflowRun"),
    "WorkflowShell": (".workflow_shell", "WorkflowShell"),
    "WorkflowStep": (".workflow_shell", "WorkflowStep"),
}

__all__ += list(_LAZY_INTERFACE_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_INTERFACE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_INTERFACE_EXPORTS[name]
    module = __import__(f"{__name__}{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


# --- AST-aware FIM processor (additive) -------------------------------------
from .ast_fim import (  # noqa: E402
    ASTAnalyzer,
    ASTNode,
    AST_FIM_REGISTRY,
    FIMFormat,
    FIMSpan,
    FIMTokenizer,
)

__all__ += [
    "ASTAnalyzer",
    "ASTNode",
    "AST_FIM_REGISTRY",
    "FIMFormat",
    "FIMSpan",
    "FIMTokenizer",
]


# --- patch synthesis engine (additive) --------------------------------------
from .patch_synthesis import (  # noqa: E402
    Patch,
    PatchError,
    PatchHunk,
    PATCH_REGISTRY,
    PatchSynthesizer,
)

__all__ += [
    "Patch",
    "PatchError",
    "PatchHunk",
    "PATCH_REGISTRY",
    "PatchSynthesizer",
]


# --- plan executor (additive) -----------------------------------------------
from .plan_executor import (  # noqa: E402
    PLAN_EXECUTOR_REGISTRY,
    PlanExecutor,
    PlanStep,
    StepStatus,
)

AGENT_LOOP_REGISTRY["plan_executor"] = PlanExecutor

__all__ += [
    "PLAN_EXECUTOR_REGISTRY",
    "PlanExecutor",
    "PlanStep",
    "StepStatus",
]


# --- tool orchestrator (additive) -------------------------------------------
from .tool_orchestrator import (  # noqa: E402
    TOOL_ORCHESTRATOR,
    ToolCall,
    ToolCallOutcome,
    ToolOrchestrator,
)

__all__ += [
    "TOOL_ORCHESTRATOR",
    "ToolCall",
    "ToolCallOutcome",
    "ToolOrchestrator",
]

# --- skill executor (cycle-204) ---------------------------------------------
from .skill_executor import (  # noqa: E402
    SkillExecutionError,
    SkillContext,
    ExecutionResult as SkillExecutionResult,
    SkillExecutor,
    DEFAULT_SKILL_EXECUTOR,
    SKILL_EXECUTOR_REGISTRY,
)

__all__ += [
    "SkillExecutionError",
    "SkillContext",
    "SkillExecutionResult",
    "SkillExecutor",
    "DEFAULT_SKILL_EXECUTOR",
    "SKILL_EXECUTOR_REGISTRY",
]

# --- plugin loader (cycle-204) -----------------------------------------------
from .plugin_loader import (  # noqa: E402
    PluginLoadError,
    LoadedPlugin,
    PluginLoader,
    DEFAULT_PLUGIN_LOADER,
    PLUGIN_LOADER_REGISTRY,
)

__all__ += [
    "PluginLoadError",
    "LoadedPlugin",
    "PluginLoader",
    "DEFAULT_PLUGIN_LOADER",
    "PLUGIN_LOADER_REGISTRY",
]

# --- plugin dependency resolver (cycle-204) ----------------------------------
from .plugin_dependency_resolver import (  # noqa: E402
    DependencyCycleError,
    DependencyResolver,
    DEFAULT_DEPENDENCY_RESOLVER,
    DEPENDENCY_RESOLVER_REGISTRY,
)

__all__ += [
    "DependencyCycleError",
    "DependencyResolver",
    "DEFAULT_DEPENDENCY_RESOLVER",
    "DEPENDENCY_RESOLVER_REGISTRY",
]

# --- skill composer (cycle-204b) --------------------------------------------
from .skill_composer import (  # noqa: E402
    SkillCompositionError,
    CompositionStep,
    CompositionResult,
    SkillComposer,
    DEFAULT_SKILL_COMPOSER,
    SKILL_COMPOSER_REGISTRY,
)

__all__ += [
    "SkillCompositionError",
    "CompositionStep",
    "CompositionResult",
    "SkillComposer",
    "DEFAULT_SKILL_COMPOSER",
    "SKILL_COMPOSER_REGISTRY",
]

# --- plugin sandbox (cycle-204b) ---------------------------------------------
from .plugin_sandbox import (  # noqa: E402
    SandboxViolationError,
    SandboxConfig,
    SandboxResult,
    PluginSandbox,
    DEFAULT_PLUGIN_SANDBOX,
    PLUGIN_SANDBOX_REGISTRY,
)

__all__ += [
    "SandboxViolationError",
    "SandboxConfig",
    "SandboxResult",
    "PluginSandbox",
    "DEFAULT_PLUGIN_SANDBOX",
    "PLUGIN_SANDBOX_REGISTRY",
]

# --- skill trigger engine (cycle-204b) ---------------------------------------
from .skill_trigger_engine import (  # noqa: E402
    TriggerEngineError,
    MatchedSkill,
    TriggerResult,
    SkillTriggerEngine,
    DEFAULT_TRIGGER_ENGINE,
    TRIGGER_ENGINE_REGISTRY,
)

__all__ += [
    "TriggerEngineError",
    "MatchedSkill",
    "TriggerResult",
    "SkillTriggerEngine",
    "DEFAULT_TRIGGER_ENGINE",
    "TRIGGER_ENGINE_REGISTRY",
]

# --- natural language command parser (cycle-208) -----------------------------
from .nl_command_parser import (  # noqa: E402
    NLCommandParseError,
    ParsedCommand,
    NLCommandParser,
    DEFAULT_NL_PARSER,
    NL_PARSER_REGISTRY,
)

__all__ += [
    "NLCommandParseError",
    "ParsedCommand",
    "NLCommandParser",
    "DEFAULT_NL_PARSER",
    "NL_PARSER_REGISTRY",
]

# --- command dispatcher (cycle-208) ------------------------------------------
from .command_dispatcher import (  # noqa: E402
    CommandDispatchError,
    DispatchResult,
    CommandDispatcher,
    DEFAULT_COMMAND_DISPATCHER,
    COMMAND_DISPATCHER_REGISTRY,
)

__all__ += [
    "CommandDispatchError",
    "DispatchResult",
    "CommandDispatcher",
    "DEFAULT_COMMAND_DISPATCHER",
    "COMMAND_DISPATCHER_REGISTRY",
]

# --- agent mode registry (cycle-209) -----------------------------------------
from .agent_mode_registry import (  # noqa: E402
    AgentModeError,
    AgentMode,
    AgentModeRegistry,
    DEFAULT_MODE_REGISTRY,
    AGENT_MODE_REGISTRY,
)

__all__ += [
    "AgentModeError",
    "AgentMode",
    "AgentModeRegistry",
    "DEFAULT_MODE_REGISTRY",
    "AGENT_MODE_REGISTRY",
]

# --- workflow engine (cycle-209) ---------------------------------------------
from .workflow_engine import (  # noqa: E402
    WorkflowError,
    WorkflowCheckpoint,
    WorkflowNode,
    WorkflowDAG,
    WorkflowExecutor,
    DEFAULT_WORKFLOW_EXECUTOR,
    WORKFLOW_EXECUTOR_REGISTRY,
)

__all__ += [
    "WorkflowError",
    "WorkflowCheckpoint",
    "WorkflowNode",
    "WorkflowDAG",
    "WorkflowExecutor",
    "DEFAULT_WORKFLOW_EXECUTOR",
    "WORKFLOW_EXECUTOR_REGISTRY",
]

# --- trace analyzer (cycle-209 subagent) --------------------------------------
from .trace_analyzer import (  # noqa: E402
    TraceAnalysis,
    TraceAnalyzer,
    ToolSummary,
    TRACE_ANALYZER_REGISTRY,
)

__all__ += [
    "TraceAnalysis",
    "TraceAnalyzer",
    "ToolSummary",
    "TRACE_ANALYZER_REGISTRY",
]
