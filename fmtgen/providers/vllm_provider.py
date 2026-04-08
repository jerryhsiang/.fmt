from __future__ import annotations

import json
from typing import Any

from fmtgen.exceptions import ProviderError
from fmtgen.providers.base import BaseProvider
from fmtgen.schema import pydantic_to_json_schema
from fmtgen.types import ConstraintType, GenerateRequest, GenerateResult, Timer


class VllmProvider(BaseProvider):
    name = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        guided_backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._guided_backend = guided_backend
        self._kwargs = kwargs

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError(
                "vllm",
                "openai package is required for vLLM provider. "
                "Install with: pip install fmtgen[openai]",
            )
        return OpenAI(base_url=self._base_url, api_key="EMPTY", **self._kwargs)

    def generate(self, request: GenerateRequest) -> GenerateResult:
        client = self._get_client()
        constraint_type = request.constraint_type

        messages = [{"role": "user", "content": request.prompt}]
        extra_body: dict[str, Any] = {}

        if self._guided_backend:
            extra_body["guided_decoding_backend"] = self._guided_backend

        if constraint_type == ConstraintType.JSON_SCHEMA and request.schema is not None:
            schema = pydantic_to_json_schema(request.schema)
            extra_body["guided_json"] = schema
        elif constraint_type == ConstraintType.REGEX and request.regex is not None:
            extra_body["guided_regex"] = request.regex
        elif constraint_type == ConstraintType.GRAMMAR and request.grammar is not None:
            extra_body["guided_grammar"] = request.grammar
        elif constraint_type == ConstraintType.CHOICE and request.choice is not None:
            extra_body["guided_choice"] = request.choice

        with Timer() as timer:
            try:
                response = client.chat.completions.create(
                    model=request.model_name,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    extra_body=extra_body if extra_body else None,
                )
            except Exception as e:
                raise ProviderError("vllm", str(e))

        raw = response.choices[0].message.content or ""

        parsed: Any = raw
        if request.schema is not None:
            data = json.loads(raw)
            parsed = request.schema.model_validate(data)

        return GenerateResult(
            raw=raw,
            parsed=parsed,
            backend_used=self._guided_backend or "auto",
            provider_used=self.name,
            model=request.model,
            latency_ms=timer.elapsed_ms,
            tokens_generated=response.usage.completion_tokens if response.usage else 0,
            constraint_type=constraint_type.value,
        )

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "regex", "grammar", "choice"]


class SglangProvider(VllmProvider):
    name = "sglang"
