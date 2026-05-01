"""Complete Aurelius Agent Registry — every agent type registered.

This is the single source of truth for all agent types in the system.
Each agent has capabilities, default tools, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentType:
    id: str
    name: str
    description: str
    category: str
    capabilities: list[str] = field(default_factory=list)
    default_tools: list[str] = field(default_factory=list)
    icon: str = "bot"
    color: str = "#4fc3f7"
    parameters: dict[str, Any] = field(default_factory=dict)


# ── Coding Agents ──────────────────────────────────────────────────────

CODING_AGENT = AgentType(
    id="coding", name="Coding Agent",
    description="Writes, reviews, and debugs code across all major languages.",
    category="coding",
    capabilities=["code", "python", "typescript", "javascript", "rust", "go", "java", "review", "debug", "refactor", "test"],
    default_tools=["read_file", "write_file", "run_command", "search_code", "git_operations"],
    icon="code",
)

CODE_REVIEWER = AgentType(
    id="code-reviewer", name="Code Reviewer",
    description="Reviews pull requests for bugs, style, security, and performance.",
    category="coding",
    capabilities=["code", "review", "security", "static_analysis", "linting"],
    default_tools=["read_file", "diff_viewer", "static_analysis", "security_scan"],
    icon="git-pull-request",
)

REFACTORING_AGENT = AgentType(
    id="refactoring", name="Refactoring Agent",
    description="Restructures code for better maintainability without changing behavior.",
    category="coding",
    capabilities=["code", "refactor", "analyze", "pattern_detection"],
    default_tools=["read_file", "write_file", "ast_parser", "diff_generator"],
    icon="shuffle",
)

TEST_AGENT = AgentType(
    id="test-writer", name="Test Writer",
    description="Generates unit, integration, and e2e tests for existing code.",
    category="coding",
    capabilities=["code", "test", "python", "typescript", "coverage"],
    default_tools=["read_file", "run_command", "coverage_analyzer"],
    icon="check-square",
)

# ── Research Agents ─────────────────────────────────────────────────────

RESEARCH_AGENT = AgentType(
    id="research", name="Research Agent",
    description="Searches the web, analyzes findings, and generates comprehensive reports.",
    category="research",
    capabilities=["search", "research", "analyze", "summarize", "cite", "synthesize"],
    default_tools=["search_web", "read_url", "extract_content", "cite_sources"],
    icon="book-open",
)

DATA_ANALYST = AgentType(
    id="data-analyst", name="Data Analyst",
    description="Analyzes datasets, creates visualizations, and produces data-driven insights.",
    category="research",
    capabilities=["analyze", "visualize", "statistics", "data_processing", "report"],
    default_tools=["query_database", "analyze_csv", "create_chart", "export_data"],
    icon="bar-chart",
)

FACT_CHECKER = AgentType(
    id="fact-checker", name="Fact Checker",
    description="Verifies claims against reliable sources and assigns confidence scores.",
    category="research",
    capabilities=["research", "verify", "cross_reference", "cite"],
    default_tools=["search_web", "read_url", "cross_reference", "confidence_scoring"],
    icon="shield",
)

# ── DevOps & System Agents ──────────────────────────────────────────────

DEVOPS_AGENT = AgentType(
    id="devops", name="DevOps Agent",
    description="Monitors infrastructure, manages deployments, and handles incidents.",
    category="devops",
    capabilities=["monitor", "deploy", "incident_response", "infrastructure", "logging"],
    default_tools=["run_command", "query_logs", "check_health", "deploy_service"],
    icon="server",
)

MONITORING_AGENT = AgentType(
    id="monitoring", name="Monitoring Agent",
    description="Watches system metrics, alerts on anomalies, and tracks SLAs.",
    category="devops",
    capabilities=["monitor", "alert", "metrics", "anomaly_detection", "reporting"],
    default_tools=["query_metrics", "check_health", "alert_on_condition", "generate_report"],
    icon="activity",
)

SECURITY_AGENT = AgentType(
    id="security", name="Security Agent",
    description="Scans for vulnerabilities, analyzes threats, and hardens systems.",
    category="devops",
    capabilities=["security", "scan", "audit", "compliance", "threat_analysis"],
    default_tools=["security_scan", "audit_logs", "compliance_check", "threat_intel"],
    icon="shield-alert",
)

# ── Communication Agents ────────────────────────────────────────────────

COMMUNICATION_AGENT = AgentType(
    id="communication", name="Communication Agent",
    description="Drafts emails, messages, and manages communication workflows.",
    category="communication",
    capabilities=["write", "edit", "email", "message", "summarize"],
    default_tools=["send_email", "draft_message", "summarize_thread", "schedule_message"],
    icon="message-square",
)

SOCIAL_MEDIA_AGENT = AgentType(
    id="social-media", name="Social Media Agent",
    description="Creates, schedules, and manages social media content across platforms.",
    category="communication",
    capabilities=["write", "edit", "schedule", "analyze_engagement", "content_planning"],
    default_tools=["draft_post", "schedule_content", "analyze_metrics", "generate_hashtags"],
    icon="share-2",
)

SUPPORT_AGENT = AgentType(
    id="support", name="Support Agent",
    description="Handles customer inquiries, triages issues, and provides solutions.",
    category="communication",
    capabilities=["support", "triage", "respond", "document", "escalate"],
    default_tools=["search_knowledge_base", "draft_response", "triage_issue", "escalate_ticket"],
    icon="headphones",
)

# ── Creative Agents ─────────────────────────────────────────────────────

CREATIVE_AGENT = AgentType(
    id="creative", name="Creative Agent",
    description="Generates stories, marketing copy, poems, and creative content.",
    category="creative",
    capabilities=["write", "edit", "brainstorm", "storytelling", "copywriting"],
    default_tools=["generate_text", "edit_text", "brainstorm_ideas", "rewrite"],
    icon="pen-tool",
)

CONTENT_STRATEGIST = AgentType(
    id="content-strategist", name="Content Strategist",
    description="Plans content calendars, SEO strategy, and audience analysis.",
    category="creative",
    capabilities=["plan", "analyze", "research", "write", "optimize"],
    default_tools=["research_keywords", "analyze_audience", "plan_calendar", "optimize_seo"],
    icon="trending-up",
)

# ── Education Agents ────────────────────────────────────────────────────

TUTOR_AGENT = AgentType(
    id="tutor", name="AI Tutor",
    description="Teaches concepts through adaptive Socratic dialogue and exercises.",
    category="education",
    capabilities=["teach", "explain", "assess", "adapt", "quiz"],
    default_tools=["generate_explanation", "create_quiz", "assess_understanding", "adapt_difficulty"],
    icon="graduation-cap",
)

LANGUAGE_TUTOR = AgentType(
    id="language-tutor", name="Language Tutor",
    description="Helps learn new languages with conversation practice and grammar lessons.",
    category="education",
    capabilities=["teach", "translate", "correct", "converse", "explain_grammar"],
    default_tools=["translate_text", "correct_grammar", "generate_exercise", "converse"],
    icon="languages",
)

# ── Data & Analytics Agents ─────────────────────────────────────────────

SQL_AGENT = AgentType(
    id="sql-agent", name="SQL Analyst",
    description="Writes and optimizes SQL queries, explores database schemas, generates reports.",
    category="data",
    capabilities=["sql", "analyze", "query", "optimize", "schema_design"],
    default_tools=["query_database", "explain_plan", "optimize_query", "describe_schema"],
    icon="database",
)

VISUALIZATION_AGENT = AgentType(
    id="visualization", name="Visualization Agent",
    description="Creates charts, dashboards, and data visualizations from any dataset.",
    category="data",
    capabilities=["visualize", "analyze", "chart", "dashboard", "report"],
    default_tools=["create_chart", "generate_dashboard", "analyze_data", "export_viz"],
    icon="bar-chart-3",
)

# ── Productivity Agents ─────────────────────────────────────────────────

SCHEDULING_AGENT = AgentType(
    id="scheduling", name="Scheduling Agent",
    description="Manages calendars, schedules meetings, and resolves conflicts.",
    category="productivity",
    capabilities=["schedule", "calendar", "coordinate", "plan", "remind"],
    default_tools=["check_calendar", "find_slots", "schedule_meeting", "send_reminder"],
    icon="calendar",
)

PROJECT_MANAGER = AgentType(
    id="project-manager", name="Project Manager",
    description="Tracks tasks, manages sprints, and generates status reports.",
    category="productivity",
    capabilities=["plan", "track", "report", "coordinate", "prioritize"],
    default_tools=["list_tasks", "update_status", "generate_report", "track_progress"],
    icon="clipboard-list",
)

NOTES_AGENT = AgentType(
    id="notes", name="Notes Agent",
    description="Takes, organizes, and summarizes meeting notes and documents.",
    category="productivity",
    capabilities=["write", "summarize", "organize", "transcribe", "search"],
    default_tools=["create_note", "summarize_text", "search_notes", "organize_by_topic"],
    icon="file-text",
)

# ── All Agents Registry ─────────────────────────────────────────────────

ALL_AGENTS: list[AgentType] = [
    CODING_AGENT, CODE_REVIEWER, REFACTORING_AGENT, TEST_AGENT,
    RESEARCH_AGENT, DATA_ANALYST, FACT_CHECKER,
    DEVOPS_AGENT, MONITORING_AGENT, SECURITY_AGENT,
    COMMUNICATION_AGENT, SOCIAL_MEDIA_AGENT, SUPPORT_AGENT,
    CREATIVE_AGENT, CONTENT_STRATEGIST,
    TUTOR_AGENT, LANGUAGE_TUTOR,
    SQL_AGENT, VISUALIZATION_AGENT,
    SCHEDULING_AGENT, PROJECT_MANAGER, NOTES_AGENT,
]

AGENT_REGISTRY: dict[str, AgentType] = {a.id: a for a in ALL_AGENTS}
AGENTS_BY_CATEGORY: dict[str, list[AgentType]] = {}
for a in ALL_AGENTS:
    AGENTS_BY_CATEGORY.setdefault(a.category, []).append(a)


def agent_to_dict(agent: AgentType) -> dict[str, Any]:
    return {
        "id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "category": agent.category,
        "capabilities": agent.capabilities,
        "default_tools": agent.default_tools,
        "icon": agent.icon,
        "color": agent.color,
        "parameters": agent.parameters,
    }

ALL_SKILLS = [
    "code_review", "debugging", "refactoring", "testing", "code_generation",
    "web_search", "fact_checking", "data_analysis", "visualization", "report_generation",
    "infrastructure_monitoring", "deployment", "incident_response", "security_scanning",
    "email_drafting", "content_creation", "social_media_management",
    "teaching", "translation", "quiz_generation",
    "sql_querying", "database_design", "chart_creation",
    "scheduling", "task_tracking", "note_taking",
    "prompt_engineering", "tool_creation", "workflow_automation",
]
