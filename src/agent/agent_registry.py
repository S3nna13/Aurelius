"""Agent registry — formal agent type definitions with capabilities and tools.

Ported from Aurelius's aurelius/agent_registry.py.
22 agent types across 8 categories, each with capabilities and tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentType:
    """Formal definition of an agent type."""

    agent_id: str
    name: str
    description: str
    category: str  # research, code, data, devops, communication, creative, education, meta
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    icon: str = "🤖"
    color: str = "gray"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "icon": self.icon,
        }


# --- Research Category ---
RESEARCH_AGENT = AgentType(
    agent_id="research",
    name="Research Agent",
    description="Conducts web searches, paper analysis, and literature reviews",
    category="research",
    capabilities=["web_search", "paper_analysis", "literature_review", "fact_checking"],
    tools=["web_search", "fetch_url"],
    icon="🔬",
    color="blue",
)

DATA_ANALYST = AgentType(
    agent_id="data_analyst",
    name="Data Analyst",
    description="Analyzes data, creates visualizations, finds patterns",
    category="research",
    capabilities=["data_analysis", "visualization", "statistics", "pattern_discovery"],
    tools=["python", "sql", "visualization"],
    icon="📊",
    color="blue",
)

# --- Code Category ---
CODE_AGENT = AgentType(
    agent_id="code",
    name="Code Agent",
    description="Writes, reviews, debugs, and refactors code",
    category="code",
    capabilities=["code_generation", "code_review", "debugging", "refactoring"],
    tools=["python", "shell", "file_ops", "git"],
    icon="💻",
    color="green",
)

ARCHITECT_AGENT = AgentType(
    agent_id="architect",
    name="Architect Agent",
    description="Designs system architecture and software structure",
    category="code",
    capabilities=["architecture_design", "design_patterns", "api_design", "database_design"],
    tools=["diagram", "documentation"],
    icon="🏗️",
    color="green",
)

REVIEWER_AGENT = AgentType(
    agent_id="reviewer",
    name="Reviewer Agent",
    description="Reviews code, finds bugs, suggests improvements",
    category="code",
    capabilities=["code_review", "static_analysis", "best_practices", "security_audit"],
    tools=["code_review", "git_diff"],
    icon="👁️",
    color="green",
)

# --- Data Category ---
DATA_ENGINEER = AgentType(
    agent_id="data_engineer",
    name="Data Engineer",
    description="Builds data pipelines, ETL processes, data warehouses",
    category="data",
    capabilities=["etl", "data_pipeline", "data_warehouse", "data_modeling"],
    tools=["sql", "python", "spark"],
    icon="⚙️",
    color="orange",
)

ML_ENGINEER = AgentType(
    agent_id="ml_engineer",
    name="ML Engineer",
    description="Trains models, optimizes hyperparameters, deploys ML systems",
    category="data",
    capabilities=["model_training", "hyperparameter_tuning", "model_deployment", "mlops"],
    tools=["python", "ml_frameworks"],
    icon="🧠",
    color="orange",
)

# --- DevOps Category ---
DEVOPS_AGENT = AgentType(
    agent_id="devops",
    name="DevOps Agent",
    description="Manages infrastructure, CI/CD, monitoring, deployments",
    category="devops",
    capabilities=["ci_cd", "infrastructure", "monitoring", "kubernetes"],
    tools=["shell", "docker", "kubernetes", "terraform"],
    icon="🛠️",
    color="red",
)

SECURITY_AGENT = AgentType(
    agent_id="security",
    name="Security Agent",
    description="Security audits, vulnerability scanning, threat modeling",
    category="devops",
    capabilities=["security_audit", "vulnerability_scan", "threat_modeling", "compliance"],
    tools=["security_scanner", "dependency_check"],
    icon="🔒",
    color="red",
)

# --- Communication Category ---
COMM_AGENT = AgentType(
    agent_id="communicator",
    name="Communicator Agent",
    description="Drafts emails, reports, presentations, and documentation",
    category="communication",
    capabilities=["writing", "editing", "formatting", "translation"],
    tools=["documentation", "email"],
    icon="💬",
    color="purple",
)

SUPPORT_AGENT = AgentType(
    agent_id="support",
    name="Support Agent",
    description="Customer support, FAQ, ticketing, issue resolution",
    category="communication",
    capabilities=["customer_support", "faq", "ticketing", "issue_resolution"],
    tools=["search", "knowledge_base"],
    icon="🎧",
    color="purple",
)

# --- Creative Category ---
CREATIVE_AGENT = AgentType(
    agent_id="creative",
    name="Creative Agent",
    description="Generates creative content, stories, marketing copy",
    category="creative",
    capabilities=["creative_writing", "copywriting", "storytelling", "brainstorming"],
    tools=["writing", "image_generation"],
    icon="🎨",
    color="pink",
)

DESIGN_AGENT = AgentType(
    agent_id="designer",
    name="Design Agent",
    description="UI/UX design, visual design, prototyping",
    category="creative",
    capabilities=["ui_design", "ux_design", "prototyping", "visual_design"],
    tools=["design", "prototyping"],
    icon="🖌️",
    color="pink",
)

# --- Education Category ---
TUTOR_AGENT = AgentType(
    agent_id="tutor",
    name="Tutor Agent",
    description="Explains concepts, creates learning materials, quizzes",
    category="education",
    capabilities=["teaching", "explanation", "quiz_creation", "curriculum_design"],
    tools=["knowledge_base", "quiz"],
    icon="📚",
    color="yellow",
)

RESEARCHER_AGENT = AgentType(
    agent_id="researcher",
    name="Researcher Agent",
    description="Academic research, paper writing, literature surveys",
    category="education",
    capabilities=["academic_research", "paper_writing", "literature_survey", "citation"],
    tools=["search", "paper_database"],
    icon="🎓",
    color="yellow",
)

# --- Meta Category ---
META_AGENT = AgentType(
    agent_id="meta",
    name="Meta Agent",
    description="Manages other agents, delegates tasks, coordinates workflows",
    category="meta",
    capabilities=[
        "agent_management",
        "task_delegation",
        "workflow_coordination",
        "quality_control",
    ],
    tools=["agent_registry", "task_queue", "workflow_engine"],
    icon="🧠",
    color="white",
)

PLANNER_AGENT = AgentType(
    agent_id="planner",
    name="Planner Agent",
    description="Creates plans, breaks down tasks, estimates effort",
    category="meta",
    capabilities=["planning", "task_breakdown", "estimation", "scheduling"],
    tools=["planner", "calendar"],
    icon="📋",
    color="white",
)

CRITIC_AGENT = AgentType(
    agent_id="critic",
    name="Critic Agent",
    description="Provides constructive criticism, finds flaws, suggests improvements",
    category="meta",
    capabilities=["critique", "analysis", "improvement_suggestions", "quality_check"],
    tools=["review", "analysis"],
    icon="⚡",
    color="white",
)

MEMORY_AGENT_TYPE = AgentType(
    agent_id="memory",
    name="Memory Agent",
    description="Manages memory, retrieval, storage, consolidation",
    category="meta",
    capabilities=["memory_management", "retrieval", "storage", "consolidation"],
    tools=["memory_store", "search"],
    icon="💾",
    color="white",
)

# Registry of all agent types
AGENT_REGISTRY: dict[str, AgentType] = {
    a.agent_id: a
    for a in [
        RESEARCH_AGENT,
        DATA_ANALYST,
        CODE_AGENT,
        ARCHITECT_AGENT,
        REVIEWER_AGENT,
        DATA_ENGINEER,
        ML_ENGINEER,
        DEVOPS_AGENT,
        SECURITY_AGENT,
        COMM_AGENT,
        SUPPORT_AGENT,
        CREATIVE_AGENT,
        DESIGN_AGENT,
        TUTOR_AGENT,
        RESEARCHER_AGENT,
        META_AGENT,
        PLANNER_AGENT,
        CRITIC_AGENT,
        MEMORY_AGENT_TYPE,
    ]
}
