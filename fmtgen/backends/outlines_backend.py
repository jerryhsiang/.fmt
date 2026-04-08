from __future__ import annotations

from typing import Any

from fmtgen.backends.base import BaseBackend


class OutlinesBackend(BaseBackend):
    name = "outlines"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import outlines  # noqa: F401

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

        import outlines.generate as generate
        import outlines.models as models

        llm = models.transformers(model)
        generator = generate.json(llm, json.dumps(schema))
        result = generator(prompt, max_tokens=max_tokens)
        return json.dumps(result) if not isinstance(result, str) else result

    def generate_regex(
        self,
        prompt: str,
        pattern: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        import outlines.generate as generate
        import outlines.models as models

        llm = models.transformers(model)
        generator = generate.regex(llm, pattern)
        return str(generator(prompt, max_tokens=max_tokens))

    def generate_grammar(
        self,
        prompt: str,
        grammar: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        import outlines.generate as generate
        import outlines.models as models

        llm = models.transformers(model)
        generator = generate.cfg(llm, grammar)
        return str(generator(prompt, max_tokens=max_tokens))

    def generate_choice(
        self,
        prompt: str,
        choices: list[str],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        import outlines.generate as generate
        import outlines.models as models

        llm = models.transformers(model)
        generator = generate.choice(llm, choices)
        return str(generator(prompt))

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "regex", "grammar", "choice"]
