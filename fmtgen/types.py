from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ConstraintType(str, Enum):
    JSON_SCHEMA = "json_schema"
    REGEX = "regex"
    GRAMMAR = "grammar"
    CHOICE = "choice"


@dataclass
class GenerateRequest:
    model: str
    prompt: str
    schema: type[BaseModel] | None = None
    regex: str | None = None
    grammar: str | None = None
    choice: list[str] | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    backend: str | None = None

    @property
    def constraint_type(self) -> ConstraintType:
        if self.schema is not None:
            return ConstraintType.JSON_SCHEMA
        if self.regex is not None:
            return ConstraintType.REGEX
        if self.grammar is not None:
            return ConstraintType.GRAMMAR
        if self.choice is not None:
            return ConstraintType.CHOICE
        raise ValueError("No constraint specified. Provide one of: schema, regex, grammar, choice")

    def validate_constraints(self) -> None:
        specified = sum(x is not None for x in [self.schema, self.regex, self.grammar, self.choice])
        if specified == 0:
            raise ValueError(
                "No constraint specified. Provide one of: schema, regex, grammar, choice"
            )
        if specified > 1:
            raise ValueError(
                "Multiple constraints specified. Provide exactly one of: "
                "schema, regex, grammar, choice"
            )

    @property
    def provider_name(self) -> str | None:
        if "/" in self.model:
            return self.model.split("/", 1)[0]
        return None

    @property
    def model_name(self) -> str:
        if "/" in self.model:
            return self.model.split("/", 1)[1]
        return self.model


@dataclass
class GenerateResult:
    raw: str
    parsed: Any
    backend_used: str
    provider_used: str
    model: str
    latency_ms: float
    tokens_generated: int
    constraint_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "parsed": (
                self.parsed.model_dump() if isinstance(self.parsed, BaseModel) else self.parsed
            ),
            "backend_used": self.backend_used,
            "provider_used": self.provider_used,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "tokens_generated": self.tokens_generated,
            "constraint_type": self.constraint_type,
        }


@dataclass
class BenchmarkRun:
    backend: str
    latency_ms: float
    success: bool
    error: str | None = None


@dataclass
class BenchmarkResult:
    runs: dict[str, list[BenchmarkRun]] = field(default_factory=dict)

    def add_run(self, backend: str, run: BenchmarkRun) -> None:
        if backend not in self.runs:
            self.runs[backend] = []
        self.runs[backend].append(run)

    def _stats(self, backend: str) -> dict[str, float]:
        latencies = [r.latency_ms for r in self.runs[backend] if r.success]
        if not latencies:
            return {"avg": 0, "p50": 0, "p99": 0, "success_rate": 0}
        latencies.sort()
        n = len(latencies)
        total = len(self.runs[backend])
        return {
            "avg": sum(latencies) / n,
            "p50": latencies[n // 2],
            "p99": latencies[int(n * 0.99)],
            "success_rate": (n / total) * 100 if total > 0 else 0,
        }

    def print_table(self) -> None:
        header = (
            f"{'Backend':<25} {'Avg (ms)':>10} {'p50 (ms)':>10} {'p99 (ms)':>10} {'Success':>10}"
        )
        print(header)
        print("-" * len(header))
        for backend in sorted(self.runs.keys()):
            stats = self._stats(backend)
            print(
                f"{backend:<25} {stats['avg']:>10.1f} {stats['p50']:>10.1f} "
                f"{stats['p99']:>10.1f} {stats['success_rate']:>9.1f}%"
            )


class Timer:
    def __init__(self) -> None:
        self._start: float = 0
        self._end: float = 0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1000
