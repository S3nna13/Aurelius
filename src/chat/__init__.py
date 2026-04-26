"""Chat template surface for Aurelius.

Exposes registries so other surfaces (serving, inference, eval) can
look up a named template without importing its module directly.
"""

from __future__ import annotations

from .chatml_template import ChatMLFormatError, ChatMLTemplate, Message

CHAT_TEMPLATE_REGISTRY: dict = {}
MESSAGE_FORMAT_REGISTRY: dict = {}

_chatml = ChatMLTemplate()
CHAT_TEMPLATE_REGISTRY["chatml"] = _chatml
MESSAGE_FORMAT_REGISTRY["chatml"] = Message

__all__ = [
    "CHAT_TEMPLATE_REGISTRY",
    "MESSAGE_FORMAT_REGISTRY",
    "ChatMLFormatError",
    "ChatMLTemplate",
    "Message",
]

from .llama3_template import Llama3Template  # noqa: E402
CHAT_TEMPLATE_REGISTRY["llama3"] = Llama3Template()

from .tool_message_formatter import ToolMessageFormatter, ToolResult  # noqa: E402
MESSAGE_FORMAT_REGISTRY["tool_result"] = ToolMessageFormatter

from .harmony_template import (  # noqa: E402
    HarmonyFormatError,
    HarmonyMessage,
    HarmonyTemplate,
)
CHAT_TEMPLATE_REGISTRY["harmony"] = HarmonyTemplate()
MESSAGE_FORMAT_REGISTRY["harmony"] = HarmonyMessage

from .conversation_memory import (  # noqa: E402
    ConversationMemory,
    Fact,
    InMemoryStore,
    JSONFileStore,
)

__all__ += [
    "ConversationMemory",
    "Fact",
    "InMemoryStore",
    "JSONFileStore",
]

from .instruction_tuning_data import (  # noqa: E402
    EvolInstructGenerator,
    InstructionSample,
    MagpieGenerator,
    SelfInstructGenerator,
)

__all__ += [
    "EvolInstructGenerator",
    "InstructionSample",
    "MagpieGenerator",
    "SelfInstructGenerator",
]

from .role_attention_mask import (  # noqa: E402
    MASK_VALUE,
    RoleSpan,
    RoleSpanError,
    build_loss_mask,
    build_role_mask,
    validate_spans,
)

__all__ += [
    "MASK_VALUE",
    "RoleSpan",
    "RoleSpanError",
    "build_loss_mask",
    "build_role_mask",
    "validate_spans",
]

from .fim_formatter import (  # noqa: E402
    FIM_MIDDLE,
    FIM_PAD,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIMExample,
    FIMFormatError,
    FIMFormatter,
)

__all__ += [
    "FIM_MIDDLE",
    "FIM_PAD",
    "FIM_PREFIX",
    "FIM_SUFFIX",
    "FIMExample",
    "FIMFormatError",
    "FIMFormatter",
]

from .multi_turn_state import ConversationState, ConversationTurn  # noqa: E402

__all__ += [
    "ConversationState",
    "ConversationTurn",
]

from .message_truncation_policy import (  # noqa: E402
    MessageTruncationPolicy,
    TruncatedResult,
    VALID_STRATEGIES as MESSAGE_TRUNCATION_STRATEGIES,
)

__all__ += [
    "MessageTruncationPolicy",
    "TruncatedResult",
    "MESSAGE_TRUNCATION_STRATEGIES",
]

from .threat_intel_persona import (  # noqa: E402
    ACTOR_SCHEMA,
    CVE_SCHEMA,
    IOC_SCHEMA,
    MITRE_SCHEMA,
    THREAT_INTEL_SYSTEM_PROMPT,
    ThreatIntelPersona,
)
CHAT_TEMPLATE_REGISTRY["threat_intel_persona"] = ThreatIntelPersona()
MESSAGE_FORMAT_REGISTRY["threat_intel_persona"] = ThreatIntelPersona

__all__ += [
    "ACTOR_SCHEMA",
    "CVE_SCHEMA",
    "IOC_SCHEMA",
    "MITRE_SCHEMA",
    "THREAT_INTEL_SYSTEM_PROMPT",
    "ThreatIntelPersona",
]

from .security_personas import (  # noqa: E402
    BLUE_TEAM_PERSONA,
    DEFAULT_SECURITY_PERSONA_REGISTRY,
    PURPLE_TEAM_PERSONA,
    RED_TEAM_PERSONA,
    SecurityPersona,
    SecurityPersonaRegistry,
)
CHAT_TEMPLATE_REGISTRY["security_personas"] = DEFAULT_SECURITY_PERSONA_REGISTRY
MESSAGE_FORMAT_REGISTRY["security_persona"] = SecurityPersona

__all__ += [
    "BLUE_TEAM_PERSONA",
    "DEFAULT_SECURITY_PERSONA_REGISTRY",
    "PURPLE_TEAM_PERSONA",
    "RED_TEAM_PERSONA",
    "SecurityPersona",
    "SecurityPersonaRegistry",
]

from .token_budget_allocator import TokenAllocation, TokenBudgetAllocator  # noqa: E402

CHAT_TOKEN_ALLOCATOR_REGISTRY: dict[str, type] = {
    "token_budget": TokenBudgetAllocator,
}

__all__ += [
    "CHAT_TOKEN_ALLOCATOR_REGISTRY",
    "TokenAllocation",
    "TokenBudgetAllocator",
]

from .system_prompt_priority import (  # noqa: E402
    PrincipalHierarchyConflict,
    SystemPromptFragment,
    SystemPromptPriority,
    SystemPromptPriorityEncoder,
)
CHAT_TEMPLATE_REGISTRY["system_prompt_priority"] = SystemPromptPriorityEncoder()
MESSAGE_FORMAT_REGISTRY["system_prompt_fragment"] = SystemPromptFragment

__all__ += [
    "PrincipalHierarchyConflict",
    "SystemPromptFragment",
    "SystemPromptPriority",
    "SystemPromptPriorityEncoder",
]

from .multi_agent_conversation import (  # noqa: E402
    AgentProfile,
    AgentRole,
    ConversationMessage,
    MultiAgentConversation,
)

__all__ += [
    "AgentProfile",
    "AgentRole",
    "ConversationMessage",
    "MultiAgentConversation",
]

from .message_threading import (  # noqa: E402
    MessageThreader,
    Thread,
    ThreadStatus,
)

__all__ += [
    "MessageThreader",
    "Thread",
    "ThreadStatus",
]

from .conversation_summarizer import (  # noqa: E402
    ConversationSummary,
    ConversationSummarizer,
    SummaryMode,
)

__all__ += [
    "ConversationSummary",
    "ConversationSummarizer",
    "SummaryMode",
]

from .conversation_logger import (  # noqa: E402
    CONVERSATION_LOGGER_REGISTRY,
    ConversationLogger,
)

CHAT_TEMPLATE_REGISTRY["conversation_logger"] = ConversationLogger
MESSAGE_FORMAT_REGISTRY["conversation_logger"] = ConversationLogger

__all__ += [
    "CONVERSATION_LOGGER_REGISTRY",
    "ConversationLogger",
]
