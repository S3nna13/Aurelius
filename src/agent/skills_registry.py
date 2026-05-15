"""Skills registry — 32 skills across 8 categories.

Ported from Aurelius's aurelius/skills_registry.py.
Skills define what an agent can DO, mapped to agent types.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Skill:
    name: str
    description: str
    category: str
    agent_types: list[str] = field(default_factory=list)
    difficulty: str = "beginner"  # beginner | intermediate | advanced | expert
    prerequisites: list[str] = field(default_factory=list)


# Coding skills
PYTHON_SKILL = Skill(
    "python",
    "Python programming",
    "coding",
    ["code", "data_engineer", "ml_engineer"],
    "intermediate",
)
CODE_REVIEW_SKILL = Skill(
    "code_review", "Review and critique code", "coding", ["code", "reviewer"], "advanced"
)
DEBUGGING_SKILL = Skill("debugging", "Debug and fix code", "coding", ["code"], "intermediate")
API_DESIGN_SKILL = Skill(
    "api_design", "Design REST/GraphQL APIs", "coding", ["architect"], "advanced"
)
GIT_SKILL = Skill("git", "Git version control operations", "coding", ["code", "devops"], "beginner")

# Research skills
WEB_SEARCH_SKILL = Skill(
    "web_search",
    "Search the web for information",
    "research",
    ["research", "researcher"],
    "beginner",
)
PAPER_ANALYSIS_SKILL = Skill(
    "paper_analysis", "Analyze academic papers", "research", ["research", "researcher"], "advanced"
)
LITERATURE_REVIEW_SKILL = Skill(
    "literature_review",
    "Review literature on a topic",
    "research",
    ["research", "researcher"],
    "advanced",
)
FACT_CHECKING_SKILL = Skill(
    "fact_checking", "Verify facts and claims", "research", ["research", "support"], "intermediate"
)

# Data skills
DATA_ANALYSIS_SKILL = Skill(
    "data_analysis",
    "Analyze data for insights",
    "data",
    ["data_analyst", "data_engineer"],
    "intermediate",
)
DATA_VISUALIZATION_SKILL = Skill(
    "data_visualization", "Create data visualizations", "data", ["data_analyst"], "intermediate"
)
SQL_SKILL = Skill(
    "sql",
    "Write and optimize SQL queries",
    "data",
    ["data_engineer", "data_analyst"],
    "intermediate",
)
ML_SKILL = Skill(
    "machine_learning", "Train and deploy ML models", "data", ["ml_engineer"], "expert"
)

# DevOps skills
DOCKER_SKILL = Skill(
    "docker", "Container management with Docker", "devops", ["devops"], "intermediate"
)
KUBERNETES_SKILL = Skill(
    "kubernetes", "Orchestrate containers with K8s", "devops", ["devops"], "expert"
)
CI_CD_SKILL = Skill(
    "ci_cd",
    "Set up continuous integration and delivery",
    "devops",
    ["devops", "deployment"],
    "advanced",
)
INFRASTRUCTURE_SKILL = Skill(
    "infrastructure",
    "Design and manage infrastructure",
    "devops",
    ["devops", "infrastructure"],
    "expert",
)

# Communication skills
WRITING_SKILL = Skill(
    "writing",
    "Write clear and effective content",
    "communication",
    ["communicator", "support", "creative"],
    "intermediate",
)
EDITING_SKILL = Skill(
    "editing",
    "Edit and improve content",
    "communication",
    ["communicator", "reviewer"],
    "intermediate",
)
TRANSLATION_SKILL = Skill(
    "translation", "Translate between languages", "communication", ["communicator"], "intermediate"
)
DOCUMENTATION_SKILL = Skill(
    "documentation",
    "Create comprehensive documentation",
    "communication",
    ["communicator", "code", "architect"],
    "intermediate",
)

# Creative skills
CREATIVE_WRITING_SKILL = Skill(
    "creative_writing", "Write creative content", "creative", ["creative"], "intermediate"
)
STORYTELLING_SKILL = Skill(
    "storytelling", "Craft compelling narratives", "creative", ["creative"], "advanced"
)
BRAINSTORMING_SKILL = Skill(
    "brainstorming", "Generate creative ideas", "creative", ["creative", "planner"], "beginner"
)
UI_DESIGN_SKILL = Skill("ui_design", "Design user interfaces", "creative", ["designer"], "advanced")

# Education skills
TEACHING_SKILL = Skill(
    "teaching", "Explain concepts effectively", "education", ["tutor"], "advanced"
)
CURRICULUM_DESIGN_SKILL = Skill(
    "curriculum_design", "Design learning curricula", "education", ["tutor"], "expert"
)
QUIZ_CREATION_SKILL = Skill(
    "quiz_creation", "Create quizzes and assessments", "education", ["tutor"], "intermediate"
)

# Meta skills
PLANNING_SKILL = Skill(
    "planning", "Create and execute plans", "meta", ["planner", "meta"], "intermediate"
)
TASK_DELEGATION_SKILL = Skill(
    "task_delegation", "Delegate tasks to agents", "meta", ["meta", "planner"], "advanced"
)
QUALITY_CONTROL_SKILL = Skill(
    "quality_control", "Ensure output quality", "meta", ["critic", "reviewer", "meta"], "advanced"
)
MEMORY_MANAGEMENT_SKILL = Skill(
    "memory_management", "Manage memory systems", "meta", ["memory"], "intermediate"
)

# --- Registry ---
SKILLS_REGISTRY: dict[str, Skill] = {
    s.name: s
    for s in [
        PYTHON_SKILL,
        CODE_REVIEW_SKILL,
        DEBUGGING_SKILL,
        API_DESIGN_SKILL,
        GIT_SKILL,
        WEB_SEARCH_SKILL,
        PAPER_ANALYSIS_SKILL,
        LITERATURE_REVIEW_SKILL,
        FACT_CHECKING_SKILL,
        DATA_ANALYSIS_SKILL,
        DATA_VISUALIZATION_SKILL,
        SQL_SKILL,
        ML_SKILL,
        DOCKER_SKILL,
        KUBERNETES_SKILL,
        CI_CD_SKILL,
        INFRASTRUCTURE_SKILL,
        WRITING_SKILL,
        EDITING_SKILL,
        TRANSLATION_SKILL,
        DOCUMENTATION_SKILL,
        CREATIVE_WRITING_SKILL,
        STORYTELLING_SKILL,
        BRAINSTORMING_SKILL,
        UI_DESIGN_SKILL,
        TEACHING_SKILL,
        CURRICULUM_DESIGN_SKILL,
        QUIZ_CREATION_SKILL,
        PLANNING_SKILL,
        TASK_DELEGATION_SKILL,
        QUALITY_CONTROL_SKILL,
        MEMORY_MANAGEMENT_SKILL,
    ]
}
