from __future__ import annotations

from abc import ABC, abstractmethod

from fmtgen.types import GenerateRequest, GenerateResult


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, request: GenerateRequest) -> GenerateResult: ...

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "choice"]
