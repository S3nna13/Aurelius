"""Aurelius model serving infrastructure."""

# Additive: expose OpenAI API shape validators for pre-flight request/response
# checking. Imported lazily-safe (no heavy deps) so adding it here does not
# affect existing module import ordering.
from .openai_api_validator import (  # noqa: E402,F401
    APIValidationError,
    OpenAIChatRequestValidator,
    OpenAIChatResponseValidator,
    API_SHAPE_REGISTRY,
)
