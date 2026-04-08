from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from fmtgen.exceptions import (
    BackendNotAvailableError,
    GenerationError,
    NoBackendAvailableError,
    SchemaValidationError,
    StructGenError,
    UnsupportedConstraintError,
)
from fmtgen.schema import pydantic_to_json_schema, resolve_refs, validate_json_output
from fmtgen.types import (
    BenchmarkResult,
    BenchmarkRun,
    ConstraintType,
    GenerateRequest,
    GenerateResult,
    Timer,
)

# --- Test Models ---


class SimpleUser(BaseModel):
    name: str
    age: int


class NestedAddress(BaseModel):
    street: str
    city: str
    zip_code: str


class UserWithAddress(BaseModel):
    name: str
    address: NestedAddress


class UserWithOptional(BaseModel):
    name: str
    email: str | None = None
    tags: list[str] = Field(default_factory=list)


# --- Types Tests ---


class TestConstraintType:
    def test_enum_values(self) -> None:
        assert ConstraintType.JSON_SCHEMA == "json_schema"
        assert ConstraintType.REGEX == "regex"
        assert ConstraintType.GRAMMAR == "grammar"
        assert ConstraintType.CHOICE == "choice"


class TestGenerateRequest:
    def test_schema_constraint(self) -> None:
        req = GenerateRequest(model="openai/gpt-4o", prompt="test", schema=SimpleUser)
        assert req.constraint_type == ConstraintType.JSON_SCHEMA

    def test_regex_constraint(self) -> None:
        req = GenerateRequest(model="openai/gpt-4o", prompt="test", regex=r"\d+")
        assert req.constraint_type == ConstraintType.REGEX

    def test_grammar_constraint(self) -> None:
        req = GenerateRequest(model="vllm/model", prompt="test", grammar="start: WORD")
        assert req.constraint_type == ConstraintType.GRAMMAR

    def test_choice_constraint(self) -> None:
        req = GenerateRequest(model="openai/gpt-4o", prompt="test", choice=["a", "b", "c"])
        assert req.constraint_type == ConstraintType.CHOICE

    def test_no_constraint_raises(self) -> None:
        req = GenerateRequest(model="openai/gpt-4o", prompt="test")
        with pytest.raises(ValueError, match="No constraint specified"):
            req.constraint_type

    def test_validate_no_constraint(self) -> None:
        req = GenerateRequest(model="openai/gpt-4o", prompt="test")
        with pytest.raises(ValueError, match="No constraint specified"):
            req.validate_constraints()

    def test_validate_multiple_constraints(self) -> None:
        req = GenerateRequest(model="openai/gpt-4o", prompt="test", schema=SimpleUser, regex=r"\d+")
        with pytest.raises(ValueError, match="Multiple constraints"):
            req.validate_constraints()

    def test_provider_name_with_prefix(self) -> None:
        req = GenerateRequest(model="openai/gpt-4o", prompt="test", schema=SimpleUser)
        assert req.provider_name == "openai"

    def test_provider_name_without_prefix(self) -> None:
        req = GenerateRequest(model="llama3", prompt="test", schema=SimpleUser)
        assert req.provider_name is None

    def test_model_name_with_prefix(self) -> None:
        req = GenerateRequest(model="openai/gpt-4o", prompt="test", schema=SimpleUser)
        assert req.model_name == "gpt-4o"

    def test_model_name_without_prefix(self) -> None:
        req = GenerateRequest(model="llama3", prompt="test", schema=SimpleUser)
        assert req.model_name == "llama3"

    def test_model_name_nested_path(self) -> None:
        req = GenerateRequest(
            model="vllm/meta-llama/Llama-3.1-8B", prompt="test", schema=SimpleUser
        )
        assert req.provider_name == "vllm"
        assert req.model_name == "meta-llama/Llama-3.1-8B"


class TestGenerateResult:
    def test_to_dict_with_model(self) -> None:
        user = SimpleUser(name="Alice", age=30)
        result = GenerateResult(
            raw='{"name": "Alice", "age": 30}',
            parsed=user,
            backend_used="none",
            provider_used="openai",
            model="openai/gpt-4o",
            latency_ms=100.0,
            tokens_generated=20,
            constraint_type="json_schema",
        )
        d = result.to_dict()
        assert d["parsed"] == {"name": "Alice", "age": 30}
        assert d["backend_used"] == "none"
        assert d["latency_ms"] == 100.0

    def test_to_dict_with_string(self) -> None:
        result = GenerateResult(
            raw="positive",
            parsed="positive",
            backend_used="none",
            provider_used="openai",
            model="openai/gpt-4o",
            latency_ms=50.0,
            tokens_generated=1,
            constraint_type="choice",
        )
        d = result.to_dict()
        assert d["parsed"] == "positive"


class TestBenchmarkResult:
    def test_add_run_and_stats(self) -> None:
        result = BenchmarkResult()
        result.add_run("outlines", BenchmarkRun("outlines", 10.0, True))
        result.add_run("outlines", BenchmarkRun("outlines", 20.0, True))
        result.add_run("outlines", BenchmarkRun("outlines", 15.0, True))

        stats = result._stats("outlines")
        assert stats["avg"] == pytest.approx(15.0)
        assert stats["success_rate"] == 100.0

    def test_print_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = BenchmarkResult()
        result.add_run("test", BenchmarkRun("test", 10.0, True))
        result.print_table()
        captured = capsys.readouterr()
        assert "test" in captured.out
        assert "10.0" in captured.out


class TestTimer:
    def test_timer(self) -> None:
        with Timer() as t:
            pass
        assert t.elapsed_ms >= 0


# --- Schema Tests ---


class TestSchema:
    def test_simple_schema(self) -> None:
        schema = pydantic_to_json_schema(SimpleUser)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_nested_schema_resolves_refs(self) -> None:
        schema = pydantic_to_json_schema(UserWithAddress)
        assert "$ref" not in json.dumps(schema)
        assert "street" in json.dumps(schema)

    def test_optional_fields(self) -> None:
        schema = pydantic_to_json_schema(UserWithOptional)
        assert "email" in schema["properties"]
        assert "tags" in schema["properties"]

    def test_validate_json_output_success(self) -> None:
        raw = '{"name": "Bob", "age": 25}'
        result = validate_json_output(raw, SimpleUser)
        assert result.name == "Bob"
        assert result.age == 25

    def test_validate_json_output_invalid(self) -> None:
        raw = '{"name": "Bob"}'
        with pytest.raises(Exception):
            validate_json_output(raw, SimpleUser)

    def test_resolve_refs_no_defs(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        resolved = resolve_refs(schema)
        assert resolved == {"type": "object", "properties": {"x": {"type": "string"}}}


# --- Exception Tests ---


class TestExceptions:
    def test_struct_gen_error_with_suggestion(self) -> None:
        err = StructGenError("Something failed", suggestion="Try this fix")
        assert "Something failed" in str(err)
        assert "Try this fix" in str(err)
        assert err.suggestion == "Try this fix"

    def test_struct_gen_error_without_suggestion(self) -> None:
        err = StructGenError("Something failed")
        assert "Something failed" in str(err)
        assert err.suggestion is None

    def test_backend_not_available(self) -> None:
        err = BackendNotAvailableError("xgrammar", available=["outlines"])
        assert "xgrammar" in str(err)
        assert "pip install" in str(err)
        assert "outlines" in str(err)

    def test_no_backend_available(self) -> None:
        err = NoBackendAvailableError()
        assert "No constrained decoding backends" in str(err)
        assert "pip install" in str(err)

    def test_unsupported_constraint(self) -> None:
        err = UnsupportedConstraintError("openai", "regex", ["json_schema", "choice"])
        assert "regex" in str(err)
        assert "openai" in str(err)

    def test_schema_validation_error(self) -> None:
        err = SchemaValidationError("User", '{"bad": "data"}', "age is required")
        assert "User" in str(err)
        assert "age is required" in str(err)

    def test_generation_error(self) -> None:
        err = GenerationError("Cannot route", suggestion="Use provider/model format")
        assert "Cannot route" in str(err)
        assert "provider/model" in str(err)


# --- Routing Tests ---


class TestRouting:
    def test_generate_unknown_provider_raises(self) -> None:
        from fmtgen import Fmt

        fmt = Fmt()
        with pytest.raises(GenerationError, match="Cannot route"):
            fmt.generate(model="llama3", prompt="test", schema=SimpleUser)

    def test_generate_no_constraint_raises(self) -> None:
        from fmtgen import Fmt

        fmt = Fmt()
        with pytest.raises(ValueError, match="No constraint specified"):
            fmt.generate(model="openai/gpt-4o", prompt="test")

    def test_generate_multiple_constraints_raises(self) -> None:
        from fmtgen import Fmt

        fmt = Fmt()
        with pytest.raises(ValueError, match="Multiple constraints"):
            fmt.generate(
                model="openai/gpt-4o",
                prompt="test",
                schema=SimpleUser,
                regex=r"\d+",
            )

    def test_status_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        from fmtgen import Fmt

        fmt = Fmt()
        fmt.status()
        captured = capsys.readouterr()
        assert "Backend Status" in captured.out
        assert "Available Providers" in captured.out

    def test_backends_property(self) -> None:
        from fmtgen import Fmt

        fmt = Fmt()
        backends = fmt.backends
        assert isinstance(backends, list)
