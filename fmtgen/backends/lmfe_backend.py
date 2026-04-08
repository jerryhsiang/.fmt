from __future__ import annotations

from typing import Any

from fmtgen.backends.base import BaseBackend


class LmfeBackend(BaseBackend):
    name = "lm-format-enforcer"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import lmformatenforcer  # noqa: F401

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

        from lmformatenforcer import JsonSchemaParser

        parser = JsonSchemaParser(json.dumps(schema))
        return self._run_with_parser(prompt, model, parser, temperature, max_tokens)

    def generate_regex(
        self,
        prompt: str,
        pattern: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        from lmformatenforcer import RegexParser

        parser = RegexParser(pattern)
        return self._run_with_parser(prompt, model, parser, temperature, max_tokens)

    def generate_grammar(
        self,
        prompt: str,
        grammar: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError(
            "lm-format-enforcer does not support grammar (CFG) constraints. "
            "Use outlines, xgrammar, or llguidance instead."
        )

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

    def _run_with_parser(
        self,
        prompt: str,
        model: str,
        parser: Any,
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError(
            "lm-format-enforcer requires integration with an inference engine. "
            "Use a provider with backend='lm-format-enforcer' instead."
        )

    def get_logits_processor(self, **kwargs: Any) -> Any:
        import json

        from lmformatenforcer import JsonSchemaParser, RegexParser

        schema = kwargs.get("schema")
        regex = kwargs.get("regex")

        if schema is not None:
            return JsonSchemaParser(json.dumps(schema))
        if regex is not None:
            return RegexParser(regex)
        raise ValueError("lm-format-enforcer supports schema and regex. Provide one of those.")

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "regex", "choice"]
