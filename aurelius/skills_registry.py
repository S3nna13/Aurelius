"""Complete skills registry exposed via CLI and API.

Every skill in the system, organized by category with descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Skill:
    id: str
    name: str
    description: str
    category: str
    agent_types: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

ALL_SKILLS: list[Skill] = [
    # Coding
    Skill("code_generation", "Code Generation", "Generate code in any language from natural language descriptions.", "coding", ["coding", "code-reviewer"], ["python", "typescript", "rust"]),
    Skill("code_review", "Code Review", "Review pull requests for bugs, style issues, and security.", "coding", ["code-reviewer"], ["quality", "security"]),
    Skill("debugging", "Debugging", "Find and fix bugs with systematic root cause analysis.", "coding", ["coding"], ["testing", "quality"]),
    Skill("refactoring", "Refactoring", "Restructure code for better maintainability.", "coding", ["refactoring"], ["quality", "patterns"]),
    Skill("unit_testing", "Unit Testing", "Generate and run unit tests for existing code.", "coding", ["test-writer"], ["testing", "coverage"]),
    Skill("static_analysis", "Static Analysis", "Analyze code without running it to find issues.", "coding", ["code-reviewer"], ["quality", "security"]),

    # Research
    Skill("web_search", "Web Search", "Search the web and synthesize findings.", "research", ["research"], ["search", "information"]),
    Skill("fact_checking", "Fact Checking", "Verify claims against authoritative sources.", "research", ["fact-checker"], ["accuracy", "verification"]),
    Skill("data_analysis", "Data Analysis", "Analyze datasets and extract insights.", "research", ["data-analyst"], ["statistics", "analytics"]),
    Skill("summarization", "Summarization", "Condense long texts into concise summaries.", "research", ["research"], ["reading", "efficiency"]),
    Skill("citation", "Citation Management", "Find and format citations from sources.", "research", ["research"], ["academic", "writing"]),

    # DevOps
    Skill("infrastructure_monitoring", "Infrastructure Monitoring", "Monitor system health, metrics, and uptime.", "devops", ["monitoring"], ["observability", "alerts"]),
    Skill("deployment", "Deployment", "Deploy services and manage releases.", "devops", ["devops"], ["ci/cd", "kubernetes"]),
    Skill("incident_response", "Incident Response", "Respond to and resolve system incidents.", "devops", ["devops"], ["reliability", "sre"]),
    Skill("security_scanning", "Security Scanning", "Scan for vulnerabilities and misconfigurations.", "devops", ["security"], ["vulnerability", "compliance"]),
    Skill("log_analysis", "Log Analysis", "Parse and analyze log files for patterns.", "devops", ["monitoring"], ["observability", "debugging"]),

    # Communication
    Skill("email_drafting", "Email Drafting", "Draft professional emails for any context.", "communication", ["communication"], ["writing", "professional"]),
    Skill("content_creation", "Content Creation", "Create engaging content for any platform.", "communication", ["social-media"], ["marketing", "writing"]),
    Skill("customer_support", "Customer Support", "Handle customer inquiries and resolve issues.", "communication", ["support"], ["service", "ticketing"]),
    Skill("translation", "Translation", "Translate text between languages.", "communication", ["language-tutor"], ["languages", "international"]),

    # Creative
    Skill("creative_writing", "Creative Writing", "Write stories, poems, and creative content.", "creative", ["creative"], ["writing", "storytelling"]),
    Skill("copywriting", "Copywriting", "Write persuasive marketing and ad copy.", "creative", ["content-strategist"], ["marketing", "writing"]),
    Skill("brainstorming", "Brainstorming", "Generate creative ideas and solutions.", "creative", ["creative"], ["ideation", "creativity"]),

    # Education
    Skill("teaching", "Teaching", "Explain concepts with adaptive pedagogy.", "education", ["tutor"], ["learning", "pedagogy"]),
    Skill("quiz_generation", "Quiz Generation", "Create quizzes to test knowledge.", "education", ["tutor"], ["assessment", "learning"]),
    Skill("language_learning", "Language Learning", "Practice conversations and learn grammar.", "education", ["language-tutor"], ["languages", "practice"]),

    # Data
    Skill("sql_querying", "SQL Querying", "Write and optimize SQL queries.", "data", ["sql-agent"], ["database", "analytics"]),
    Skill("data_visualization", "Data Visualization", "Create charts and visual representations of data.", "data", ["visualization"], ["charts", "dashboard"]),
    Skill("database_design", "Database Design", "Design database schemas and optimize queries.", "data", ["sql-agent"], ["schema", "optimization"]),

    # Productivity
    Skill("scheduling", "Scheduling", "Manage calendars and schedule events.", "productivity", ["scheduling"], ["calendar", "time"]),
    Skill("task_management", "Task Management", "Track and manage tasks across projects.", "productivity", ["project-manager"], ["agile", "organization"]),
    Skill("note_taking", "Note Taking", "Take and organize meeting notes.", "productivity", ["notes"], ["documentation", "organization"]),
    Skill("meeting_summarization", "Meeting Summarization", "Summarize meeting transcripts into action items.", "productivity", ["notes"], ["meetings", "efficiency"]),

    # Meta
    Skill("prompt_engineering", "Prompt Engineering", "Design and optimize prompts for LLMs.", "meta", [], ["optimization", "llm"]),
    Skill("tool_creation", "Tool Creation", "Create new tools and integrations for the agent platform.", "meta", [], ["development", "extensibility"]),
    Skill("workflow_automation", "Workflow Automation", "Automate multi-step workflows across tools.", "meta", [], ["automation", "efficiency"]),
]

SKILL_REGISTRY: dict[str, Skill] = {s.id: s for s in ALL_SKILLS}
SKILLS_BY_CATEGORY: dict[str, list[Skill]] = {}
for s in ALL_SKILLS:
    SKILLS_BY_CATEGORY.setdefault(s.category, []).append(s)


def skill_to_dict(skill: Skill) -> dict[str, object]:
    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "category": skill.category,
        "agent_types": skill.agent_types,
        "tags": skill.tags,
    }
