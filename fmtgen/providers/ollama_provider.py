from __future__ import annotations

import json
from typing import Any

from fmtgen.exceptions import ProviderError, UnsupportedConstraintError
from fmtgen.providers.base import BaseProvider
from fmtgen.schema import pydantic_to_json_schema
from fmtgen.types import ConstraintType, GenerateRequest, GenerateResult, Timer


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", **kwargs: Any) -> None:
        self._base_url = base_url.rstrip("/")
        self._kwargs = kwargs

    def generate(self, request: GenerateRequest) -> GenerateResult:
        import httpx

        constraint_type = request.constraint_type

        if constraint_type == ConstraintType.REGEX:
            raise UnsupportedConstraintError("ollama", "regex", self.supported_constraints())
        if constraint_type == ConstraintType.GRAMMAR:
            raise UnsupportedConstraintError("ollama", "grammar", self.supported_constraints())

        payload: dict[str, Any] = {
            "model": request.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        if constraint_type == ConstraintType.JSON_SCHEMA and request.schema is not None:
            schema = pydantic_to_json_schema(request.schema)
            payload["format"] = schema
        elif constraint_type == ConstraintType.CHOICE and request.choice is not None:
            choice_schema = {"type": "string", "enum": request.choice}
            payload["format"] = choice_schema
            payload["prompt"] = (
                f"{request.prompt}\n\nRespond with exactly one of: {', '.join(request.choice)}"
            )

        with Timer() as timer:
            try:
                resp = httpx.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                    timeout=120.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise ProviderError(
                    "ollama", f"HTTP {e.response.status_code}: {e.response.text}"
                ) from e
            except httpx.ConnectError as e:
                raise ProviderError(
                    "ollama",
                    f"Cannot connect to Ollama at {self._base_url}. "
                    "Is Ollama running? Start with: ollama serve",
                ) from e
            except Exception as e:
                raise ProviderError("ollama", str(e)) from e

        data = resp.json()
        raw = data.get("response", "")

        parsed: Any = raw
        if request.schema is not None:
            try:
                parsed_data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ProviderError(
                    "ollama", f"Failed to parse JSON from model output: {raw[:200]}"
                ) from e
            parsed = request.schema.model_validate(parsed_data)
        elif request.choice is not None:
            parsed = raw.strip().strip('"')

        return GenerateResult(
            raw=raw,
            parsed=parsed,
            backend_used="none",
            provider_used="ollama",
            model=request.model,
            latency_ms=timer.elapsed_ms,
            tokens_generated=data.get("eval_count", 0),
            constraint_type=constraint_type.value,
        )

    async def agenerate(self, request: GenerateRequest) -> GenerateResult:
        import httpx

        constraint_type = request.constraint_type

        if constraint_type == ConstraintType.REGEX:
            raise UnsupportedConstraintError("ollama", "regex", self.supported_constraints())
        if constraint_type == ConstraintType.GRAMMAR:
            raise UnsupportedConstraintError("ollama", "grammar", self.supported_constraints())

        payload: dict[str, Any] = {
            "model": request.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        if constraint_type == ConstraintType.JSON_SCHEMA and request.schema is not None:
            schema = pydantic_to_json_schema(request.schema)
            payload["format"] = schema
        elif constraint_type == ConstraintType.CHOICE and request.choice is not None:
            choice_schema = {"type": "string", "enum": request.choice}
            payload["format"] = choice_schema
            payload["prompt"] = (
                f"{request.prompt}\n\nRespond with exactly one of: {', '.join(request.choice)}"
            )

        with Timer() as timer:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{self._base_url}/api/generate",
                        json=payload,
                        timeout=120.0,
                    )
                    resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise ProviderError(
                    "ollama", f"HTTP {e.response.status_code}: {e.response.text}"
                ) from e
            except httpx.ConnectError as e:
                raise ProviderError(
                    "ollama",
                    f"Cannot connect to Ollama at {self._base_url}. "
                    "Is Ollama running? Start with: ollama serve",
                ) from e
            except Exception as e:
                raise ProviderError("ollama", str(e)) from e

        data = resp.json()
        raw = data.get("response", "")

        parsed: Any = raw
        if request.schema is not None:
            try:
                parsed_data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ProviderError(
                    "ollama", f"Failed to parse JSON from model output: {raw[:200]}"
                ) from e
            parsed = request.schema.model_validate(parsed_data)
        elif request.choice is not None:
            parsed = raw.strip().strip('"')

        return GenerateResult(
            raw=raw,
            parsed=parsed,
            backend_used="none",
            provider_used="ollama",
            model=request.model,
            latency_ms=timer.elapsed_ms,
            tokens_generated=data.get("eval_count", 0),
            constraint_type=constraint_type.value,
        )

    @classmethod
    def supported_constraints(cls) -> list[str]:
        return ["json_schema", "choice"]
