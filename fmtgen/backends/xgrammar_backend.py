from __future__ import annotations

from typing import Any

from fmtgen.backends.base import BaseBackend


class XGrammarBackend(BaseBackend):
    name = "xgrammar"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import xgrammar  # noqa: F401

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

        import xgrammar

        grammar = xgrammar.Grammar.from_json_schema(json.dumps(schema))
        compiler = xgrammar.GrammarCompiler(grammar)
        logits_processor = compiler.get_logits_processor()
        return self._run_with_processor(prompt, model, logits_processor, temperature, max_tokens)

    def generate_regex(
        self,
        prompt: str,
        pattern: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        import xgrammar

        grammar = xgrammar.Grammar.from_regex(pattern)
        compiler = xgrammar.GrammarCompiler(grammar)
        logits_processor = compiler.get_logits_processor()
        return self._run_with_processor(prompt, model, logits_processor, temperature, max_tokens)

    def generate_grammar(
        self,
        prompt: str,
        grammar: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        import xgrammar

        compiled_grammar = xgrammar.Grammar.from_ebnf(grammar)
        compiler = xgrammar.GrammarCompiler(compiled_grammar)
        logits_processor = compiler.get_logits_processor()
        return self._run_with_processor(prompt, model, logits_processor, temperature, max_tokens)

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

    def _run_with_processor(
        self,
        prompt: str,
        model: str,
        logits_processor: Any,
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError(
            "XGrammar requires integration with an inference engine (vLLM, SGLang). "
            "Use the vLLM or SGLang provider with backend='xgrammar' instead."
        )

    def get_logits_processor(self, **kwargs: Any) -> Any:
        import json

        import xgrammar

        schema = kwargs.get("schema")
        regex = kwargs.get("regex")
        grammar_str = kwargs.get("grammar")

        if schema is not None:
            grammar = xgrammar.Grammar.from_json_schema(json.dumps(schema))
        elif regex is not None:
            grammar = xgrammar.Grammar.from_regex(regex)
        elif grammar_str is not None:
            grammar = xgrammar.Grammar.from_ebnf(grammar_str)
        else:
            raise ValueError("Provide schema, regex, or grammar for logits processor")

        compiler = xgrammar.GrammarCompiler(grammar)
        return compiler.get_logits_processor()

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "regex", "grammar", "choice"]
