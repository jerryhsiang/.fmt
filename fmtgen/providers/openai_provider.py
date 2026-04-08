from __future__ import annotations

import json
from typing import Any

from fmtgen.exceptions import ProviderError, UnsupportedConstraintError
from fmtgen.providers.base import BaseProvider
from fmtgen.schema import pydantic_to_json_schema
from fmtgen.types import ConstraintType, GenerateRequest, GenerateResult, Timer


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError(
                "openai",
                "openai package is not installed. Install with: pip install fmtgen[openai]",
            ) from e
        return OpenAI(**self._kwargs)

    def generate(self, request: GenerateRequest) -> GenerateResult:
        constraint_type = request.constraint_type

        if constraint_type == ConstraintType.REGEX:
            raise UnsupportedConstraintError("openai", "regex", self.supported_constraints())
        if constraint_type == ConstraintType.GRAMMAR:
            raise UnsupportedConstraintError("openai", "grammar", self.supported_constraints())

        client = self._get_client()

        messages = [{"role": "user", "content": request.prompt}]
        call_kwargs: dict[str, Any] = {
            "model": request.model_name,
            "messages": messages,
            "temperature": request.temperature,
            "max_completion_tokens": request.max_tokens,
        }

        if constraint_type == ConstraintType.JSON_SCHEMA and request.schema is not None:
            schema = pydantic_to_json_schema(request.schema)
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }
        elif constraint_type == ConstraintType.CHOICE and request.choice is not None:
            messages[0]["content"] = (
                f"{request.prompt}\n\n"
                f"You must respond with exactly one of these options: "
                f"{', '.join(request.choice)}\n"
                f"Respond with only the chosen option, nothing else."
            )

        with Timer() as timer:
            try:
                response = client.chat.completions.create(**call_kwargs)
            except Exception as e:
                raise ProviderError("openai", str(e)) from e

        if not response.choices:
            raise ProviderError(
                "openai", "API returned empty choices (content may have been filtered)"
            )
        raw = response.choices[0].message.content or ""

        parsed: Any = raw
        if request.schema is not None:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ProviderError(
                    "openai", f"Failed to parse JSON from model output: {raw[:200]}"
                ) from e
            parsed = request.schema.model_validate(data)

        return GenerateResult(
            raw=raw,
            parsed=parsed,
            backend_used="none",
            provider_used="openai",
            model=request.model,
            latency_ms=timer.elapsed_ms,
            tokens_generated=response.usage.completion_tokens if response.usage else 0,
            constraint_type=constraint_type.value,
        )

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "choice"]
