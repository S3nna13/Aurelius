from ..unified_persona import OutputContract

GENERAL_CONTRACT = OutputContract(
    name="general",
    schema={
        "type": "object",
        "required": ["summary", "context", "recommended_actions"],
        "properties": {
            "summary": {"type": "string"},
            "context": {"type": "string"},
            "recommended_actions": {"type": "array", "items": {"type": "string"}},
            "references": {"type": "array", "items": {"type": "string"}},
        },
    },
    required_fields=("summary", "recommended_actions"),
)
