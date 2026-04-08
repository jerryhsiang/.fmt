from __future__ import annotations


class FmtError(Exception):
    def __init__(self, message: str, suggestion: str | None = None) -> None:
        self.suggestion = suggestion
        full_message = message
        if suggestion:
            full_message += f"\n\n\U0001f4a1 Suggestion: {suggestion}"
        super().__init__(full_message)


class BackendNotAvailableError(FmtError):
    def __init__(self, backend: str, available: list[str] | None = None) -> None:
        available_str = ", ".join(available) if available else "none"
        super().__init__(
            f"Backend '{backend}' is not installed or available.",
            suggestion=(
                f"Install it with: pip install fmtgen[{backend}]\n"
                f"Available backends: {available_str}"
            ),
        )


class NoBackendAvailableError(FmtError):
    def __init__(self) -> None:
        super().__init__(
            "No constrained decoding backends are available.",
            suggestion=(
                "Install at least one backend:\n"
                "  pip install fmtgen[outlines]     # Most mature\n"
                "  pip install fmtgen[xgrammar]     # Fastest\n"
                "  pip install fmtgen[llguidance]   # Low latency\n"
                "  pip install fmtgen[lmfe]         # Best for debugging"
            ),
        )


class UnsupportedConstraintError(FmtError):
    def __init__(
        self, backend: str, constraint_type: str, supported: list[str] | None = None
    ) -> None:
        supported_str = ", ".join(supported) if supported else "unknown"
        super().__init__(
            f"Backend '{backend}' does not support constraint type '{constraint_type}'.",
            suggestion=(
                f"Supported constraint types for '{backend}': {supported_str}\n"
                "Try using backend='auto' to let fmtgen select a compatible backend."
            ),
        )


class ProviderError(FmtError):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(
            f"Provider '{provider}' returned an error: {message}",
            suggestion=(
                "Check your API key and network connection. "
                "For local providers (Ollama, vLLM), ensure the server is running."
            ),
        )


class SchemaValidationError(FmtError):
    def __init__(self, schema_name: str, raw_output: str, validation_error: str) -> None:
        super().__init__(
            (
                f"Output failed schema validation for '{schema_name}'.\n"
                f"Raw output: {raw_output[:200]}{'...' if len(raw_output) > 200 else ''}\n"
                f"Validation error: {validation_error}"
            ),
            suggestion=(
                "This usually means the model ignored the schema constraint. Try:\n"
                "  1. Using a more capable model\n"
                "  2. Adding schema hints to your prompt\n"
                "  3. Switching to a local backend with true constrained decoding"
            ),
        )


class GenerationError(FmtError):
    pass
