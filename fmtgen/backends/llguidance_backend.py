from __future__ import annotations

from typing import Any

from fmtgen.backends.base import BaseBackend


class LlguidanceBackend(BaseBackend):
    name = "llguidance"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import llguidance  # noqa: F401

            return True
        except ImportError:
            return False

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        import json

        import llguidance

        constraint = llguidance.JsonCompiler(json.dumps(schema))
        return self._run_with_constraint(prompt, model, constraint, temperature, max_tokens)

    def generate_regex(
        self,
        prompt: str,
        pattern: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        import llguidance

        constraint = llguidance.RegexCompiler(pattern)
        return self._run_with_constraint(prompt, model, constraint, temperature, max_tokens)

    def generate_grammar(
        self,
        prompt: str,
        grammar: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        import llguidance

        constraint = llguidance.LarkCompiler(grammar)
        return self._run_with_constraint(prompt, model, constraint, temperature, max_tokens)

    def generate_choice(
        self,
        prompt: str,
        choices: list[str],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:

        choice_schema = {"type": "string", "enum": choices}
        return self.generate_json(prompt, choice_schema, model, temperature, max_tokens, **kwargs)

    def _run_with_constraint(
        self,
        prompt: str,
        model: str,
        constraint: Any,
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError(
            "llguidance requires integration with an inference engine. "
            "Use a provider with backend='llguidance' instead."
        )

    def get_logits_processor(self, **kwargs: Any) -> Any:
        import json

        import llguidance

        schema = kwargs.get("schema")
        regex = kwargs.get("regex")
        grammar_str = kwargs.get("grammar")

        if schema is not None:
            return llguidance.JsonCompiler(json.dumps(schema))
        if regex is not None:
            return llguidance.RegexCompiler(regex)
        if grammar_str is not None:
            return llguidance.LarkCompiler(grammar_str)
        raise ValueError("Provide schema, regex, or grammar for logits processor")

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "regex", "grammar", "choice"]
