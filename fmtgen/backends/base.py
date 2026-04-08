from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseBackend(ABC):
    name: str = "base"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str: ...

    @abstractmethod
    def generate_regex(
        self,
        prompt: str,
        pattern: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str: ...

    @abstractmethod
    def generate_grammar(
        self,
        prompt: str,
        grammar: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str: ...

    @abstractmethod
    def generate_choice(
        self,
        prompt: str,
        choices: list[str],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str: ...

    def get_logits_processor(self, **kwargs: Any) -> Any:
        raise NotImplementedError(f"Backend '{self.name}' does not expose a logits processor.")

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "regex", "grammar", "choice"]
