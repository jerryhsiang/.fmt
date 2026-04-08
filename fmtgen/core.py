from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from fmtgen.backends import BackendRegistry
from fmtgen.exceptions import GenerationError
from fmtgen.providers import ProviderRegistry
from fmtgen.types import (
    BenchmarkResult,
    BenchmarkRun,
    GenerateRequest,
    GenerateResult,
    Timer,
)


class Fmt:
    def __init__(
        self,
        backend: str = "auto",
        provider_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._backend = backend
        self._provider_kwargs = provider_kwargs or {}

    def generate(
        self,
        model: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        regex: str | None = None,
        grammar: str | None = None,
        choice: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> BaseModel | str:
        result = self.generate_raw(
            model=model,
            prompt=prompt,
            schema=schema,
            regex=regex,
            grammar=grammar,
            choice=choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed: BaseModel | str = result.parsed
        return parsed

    def generate_raw(
        self,
        model: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        regex: str | None = None,
        grammar: str | None = None,
        choice: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> GenerateResult:
        request = GenerateRequest(
            model=model,
            prompt=prompt,
            schema=schema,
            regex=regex,
            grammar=grammar,
            choice=choice,
            temperature=temperature,
            max_tokens=max_tokens,
            backend=self._backend,
        )
        request.validate_constraints()

        provider_name = request.provider_name

        if provider_name and provider_name in ProviderRegistry.list_providers():
            kwargs = self._provider_kwargs.get(provider_name, {})
            provider = ProviderRegistry.get(provider_name, **kwargs)
            return provider.generate(request)

        raise GenerationError(
            f"Cannot route model '{model}'. "
            f"Use format 'provider/model_name' (e.g., 'openai/gpt-4o', 'ollama/llama3').",
            suggestion=(
                f"Available providers: {', '.join(ProviderRegistry.list_providers())}\n"
                "For local models, use 'ollama/model_name' or 'vllm/model_name'."
            ),
        )

    def status(self) -> None:
        print("fmtgen Backend Status")
        print("=" * 50)
        for info in BackendRegistry.list_all():
            icon = "\u2705" if info["available"] else "\u274c"
            constraints = ", ".join(info["constraints"]) if info["constraints"] else "n/a"
            print(f"  {icon} {info['name']:<25} [{constraints}]")
        print()
        print("Available Providers")
        print("=" * 50)
        for name in ProviderRegistry.list_providers():
            print(f"  \u2022 {name}")

    @property
    def backends(self) -> list[str]:
        return BackendRegistry.list_available()


def benchmark(
    prompt: str,
    model: str,
    schema: type[BaseModel] | None = None,
    regex: str | None = None,
    grammar: str | None = None,
    choice: list[str] | None = None,
    backends: list[str] | None = None,
    iterations: int = 10,
) -> BenchmarkResult:
    if backends is None:
        backends = BackendRegistry.list_available()

    result = BenchmarkResult()

    for backend_name in backends:
        backend = BackendRegistry.get(backend_name)
        for _ in range(iterations):
            with Timer() as timer:
                try:
                    if schema is not None:
                        from fmtgen.schema import pydantic_to_json_schema

                        json_schema = pydantic_to_json_schema(schema)
                        backend.generate_json(
                            prompt=prompt,
                            schema=json_schema,
                            model=model,
                        )
                    elif regex is not None:
                        backend.generate_regex(
                            prompt=prompt,
                            pattern=regex,
                            model=model,
                        )
                    elif grammar is not None:
                        backend.generate_grammar(
                            prompt=prompt,
                            grammar=grammar,
                            model=model,
                        )
                    elif choice is not None:
                        backend.generate_choice(
                            prompt=prompt,
                            choices=choice,
                            model=model,
                        )
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)

            result.add_run(
                backend_name,
                BenchmarkRun(
                    backend=backend_name,
                    latency_ms=timer.elapsed_ms,
                    success=success,
                    error=error,
                ),
            )

    return result
