from __future__ import annotations

import json
from typing import Any

from fmtgen.exceptions import ProviderError, UnsupportedConstraintError
from fmtgen.providers.base import BaseProvider
from fmtgen.schema import pydantic_to_json_schema
from fmtgen.types import ConstraintType, GenerateRequest, GenerateResult, Timer


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def _get_client(self) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ProviderError(
                "anthropic",
                "anthropic package is not installed. Install with: pip install fmtgen[anthropic]",
            ) from e
        return Anthropic(**self._kwargs)

    def generate(self, request: GenerateRequest) -> GenerateResult:
        constraint_type = request.constraint_type

        if constraint_type == ConstraintType.REGEX:
            raise UnsupportedConstraintError("anthropic", "regex", self.supported_constraints())
        if constraint_type == ConstraintType.GRAMMAR:
            raise UnsupportedConstraintError("anthropic", "grammar", self.supported_constraints())

        client = self._get_client()

        if constraint_type == ConstraintType.JSON_SCHEMA and request.schema is not None:
            schema = pydantic_to_json_schema(request.schema)
            tool_name = f"extract_{request.schema.__name__.lower()}"

            with Timer() as timer:
                try:
                    response = client.messages.create(
                        model=request.model_name,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        tools=[
                            {
                                "name": tool_name,
                                "description": f"Extract {request.schema.__name__} from the input",
                                "input_schema": schema,
                            }
                        ],
                        tool_choice={"type": "tool", "name": tool_name},
                        messages=[{"role": "user", "content": request.prompt}],
                    )
                except Exception as e:
                    raise ProviderError("anthropic", str(e)) from e

            tool_result = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_result = block.input
                    break

            if tool_result is None:
                raise ProviderError("anthropic", "No tool_use block in response")

            raw = json.dumps(tool_result)
            parsed = request.schema.model_validate(tool_result)

        elif constraint_type == ConstraintType.CHOICE and request.choice is not None:
            prompt = (
                f"{request.prompt}\n\n"
                f"You must respond with exactly one of these options: "
                f"{', '.join(request.choice)}\n"
                f"Respond with only the chosen option, nothing else."
            )

            with Timer() as timer:
                try:
                    response = client.messages.create(
                        model=request.model_name,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                except Exception as e:
                    raise ProviderError("anthropic", str(e)) from e

            if not response.content:
                raise ProviderError("anthropic", "API returned empty content")
            raw = response.content[0].text
            parsed = raw.strip()
        else:
            raise UnsupportedConstraintError(
                "anthropic", constraint_type.value, self.supported_constraints()
            )

        tokens = 0
        if hasattr(response, "usage") and response.usage:
            tokens = response.usage.output_tokens

        return GenerateResult(
            raw=raw,
            parsed=parsed,
            backend_used="none",
            provider_used="anthropic",
            model=request.model,
            latency_ms=timer.elapsed_ms,
            tokens_generated=tokens,
            constraint_type=constraint_type.value,
        )

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "choice"]
