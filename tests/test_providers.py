from __future__ import annotations

import json
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from fmtgen.exceptions import (
    GenerationError,
    ProviderError,
    UnsupportedConstraintError,
)
from fmtgen.types import GenerateRequest

# ---------------------------------------------------------------------------
# Test Pydantic models
# ---------------------------------------------------------------------------


class SimpleUser(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    street: str
    city: str


class UserWithAddress(BaseModel):
    name: str
    address: Address


class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class UserWithOptionals(BaseModel):
    name: str
    age: int | None = None
    tags: list[str] = []
    color: Color | None = None


class Company(BaseModel):
    name: str
    ceo: UserWithAddress
    employees: list[SimpleUser]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(content: str, completion_tokens: int = 10):
    """Build a mock OpenAI chat completion response."""
    usage = MagicMock()
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_openai_empty_response():
    response = MagicMock()
    response.choices = []
    return response


def _make_anthropic_tool_response(
    tool_input: dict,
    tool_name: str = "extract_simpleuser",
):
    block = MagicMock()
    block.type = "tool_use"
    block.input = tool_input

    usage = MagicMock()
    usage.output_tokens = 15

    response = MagicMock()
    response.content = [block]
    response.usage = usage
    return response


def _make_anthropic_text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.output_tokens = 5

    response = MagicMock()
    response.content = [block]
    response.usage = usage
    return response


# ===================================================================
# OpenAI provider tests
# ===================================================================


class TestOpenAIProvider:
    def _make_provider(self, mock_client: MagicMock):
        from fmtgen.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        provider._get_client = lambda: mock_client  # type: ignore[assignment]
        return provider

    def test_json_schema_generation(self):
        mock_client = MagicMock()
        user_data = {"name": "Alice", "age": 30}
        mock_client.chat.completions.create.return_value = _make_openai_response(
            json.dumps(user_data)
        )

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract user info",
            schema=SimpleUser,
        )
        result = provider.generate(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["name"] == "SimpleUser"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True
        assert call_kwargs["max_completion_tokens"] == 2048
        assert call_kwargs["model"] == "gpt-4o"

        assert isinstance(result.parsed, SimpleUser)
        assert result.parsed.name == "Alice"
        assert result.parsed.age == 30
        assert result.provider_used == "openai"
        assert result.constraint_type == "json_schema"

    def test_choice_generation(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Is the sky blue?",
            choice=["yes", "no"],
        )
        result = provider.generate(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        prompt_sent = call_kwargs["messages"][0]["content"]
        assert "yes, no" in prompt_sent
        assert "exactly one" in prompt_sent
        assert result.raw == "yes"
        assert result.constraint_type == "choice"

    def test_empty_choices_raises_provider_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_empty_response()

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract user",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError, match="empty choices"):
            provider.generate(request)

    def test_malformed_json_raises_provider_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response("not valid json {")

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract user",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError, match="Failed to parse JSON"):
            provider.generate(request)

    def test_api_exception_wrapped_with_chaining(self):
        mock_client = MagicMock()
        original_exc = RuntimeError("rate limit exceeded")
        mock_client.chat.completions.create.side_effect = original_exc

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract user",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError, match="rate limit exceeded") as exc_info:
            provider.generate(request)
        assert exc_info.value.__cause__ is original_exc

    def test_regex_raises_unsupported(self):
        from fmtgen.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract",
            regex=r"\d+",
        )
        with pytest.raises(UnsupportedConstraintError, match="regex"):
            provider.generate(request)

    def test_grammar_raises_unsupported(self):
        from fmtgen.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract",
            grammar="root ::= [a-z]+",
        )
        with pytest.raises(UnsupportedConstraintError, match="grammar"):
            provider.generate(request)


# ===================================================================
# Anthropic provider tests
# ===================================================================


class TestAnthropicProvider:
    def _make_provider(self, mock_client: MagicMock):
        from fmtgen.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        provider._get_client = lambda: mock_client  # type: ignore[assignment]
        return provider

    def test_json_schema_via_tool_use(self):
        mock_client = MagicMock()
        user_data = {"name": "Bob", "age": 25}
        mock_client.messages.create.return_value = _make_anthropic_tool_response(
            user_data, "extract_simpleuser"
        )

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="anthropic/claude-sonnet-4-20250514",
            prompt="Extract user info",
            schema=SimpleUser,
        )
        result = provider.generate(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "extract_simpleuser"
        assert "input_schema" in tools[0]
        assert call_kwargs["tool_choice"]["type"] == "tool"
        assert call_kwargs["tool_choice"]["name"] == "extract_simpleuser"

        assert isinstance(result.parsed, SimpleUser)
        assert result.parsed.name == "Bob"
        assert result.parsed.age == 25
        assert result.provider_used == "anthropic"
        assert result.constraint_type == "json_schema"

    def test_choice_generation(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_text_response("yes")

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="anthropic/claude-sonnet-4-20250514",
            prompt="Is the sky blue?",
            choice=["yes", "no"],
        )
        result = provider.generate(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_sent = call_kwargs["messages"][0]["content"]
        assert "yes, no" in prompt_sent
        assert "exactly one" in prompt_sent
        assert result.parsed == "yes"
        assert result.constraint_type == "choice"

    def test_empty_content_raises_provider_error(self):
        mock_client = MagicMock()
        response = MagicMock()
        response.content = []
        response.usage = MagicMock(output_tokens=0)
        mock_client.messages.create.return_value = response

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="anthropic/claude-sonnet-4-20250514",
            prompt="Is the sky blue?",
            choice=["yes", "no"],
        )
        with pytest.raises(ProviderError, match="empty content"):
            provider.generate(request)

    def test_no_tool_use_block_raises_provider_error(self):
        mock_client = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "here is some text"
        response = MagicMock()
        response.content = [text_block]
        mock_client.messages.create.return_value = response

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="anthropic/claude-sonnet-4-20250514",
            prompt="Extract user",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError, match="No tool_use block"):
            provider.generate(request)

    def test_api_exception_wrapped_with_chaining(self):
        mock_client = MagicMock()
        original_exc = RuntimeError("auth failed")
        mock_client.messages.create.side_effect = original_exc

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="anthropic/claude-sonnet-4-20250514",
            prompt="Extract user",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError, match="auth failed") as exc_info:
            provider.generate(request)
        assert exc_info.value.__cause__ is original_exc

    def test_regex_raises_unsupported(self):
        from fmtgen.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        request = GenerateRequest(
            model="anthropic/claude-sonnet-4-20250514",
            prompt="Extract",
            regex=r"\d+",
        )
        with pytest.raises(UnsupportedConstraintError, match="regex"):
            provider.generate(request)

    def test_grammar_raises_unsupported(self):
        from fmtgen.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        request = GenerateRequest(
            model="anthropic/claude-sonnet-4-20250514",
            prompt="Extract",
            grammar="root ::= [a-z]+",
        )
        with pytest.raises(UnsupportedConstraintError, match="grammar"):
            provider.generate(request)


# ===================================================================
# Ollama provider tests
# ===================================================================


class TestOllamaProvider:
    def _make_provider(self):
        from fmtgen.providers.ollama_provider import OllamaProvider

        return OllamaProvider(base_url="http://localhost:11434")

    @patch("httpx.post")
    def test_json_schema_generation(self, mock_post):
        user_data = {"name": "Charlie", "age": 40}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": json.dumps(user_data),
            "eval_count": 12,
        }
        mock_post.return_value = mock_resp

        provider = self._make_provider()
        request = GenerateRequest(
            model="ollama/llama3",
            prompt="Extract user info",
            schema=SimpleUser,
        )
        result = provider.generate(request)

        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        payload = call_args[1]["json"]
        assert "format" in payload
        assert payload["model"] == "llama3"
        assert payload["stream"] is False

        assert isinstance(result.parsed, SimpleUser)
        assert result.parsed.name == "Charlie"
        assert result.parsed.age == 40
        assert result.provider_used == "ollama"
        assert result.constraint_type == "json_schema"

    @patch("httpx.post")
    def test_choice_generation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": '"yes"',
            "eval_count": 1,
        }
        mock_post.return_value = mock_resp

        provider = self._make_provider()
        request = GenerateRequest(
            model="ollama/llama3",
            prompt="Is the sky blue?",
            choice=["yes", "no"],
        )
        result = provider.generate(request)

        payload = mock_post.call_args[1]["json"]
        assert payload["format"]["type"] == "string"
        assert payload["format"]["enum"] == ["yes", "no"]
        assert "Respond with exactly one of" in payload["prompt"]
        assert result.parsed == "yes"

    @patch("httpx.post")
    def test_connect_error_raises_provider_error(self, mock_post):
        import httpx

        mock_post.side_effect = httpx.ConnectError("Connection refused")

        provider = self._make_provider()
        request = GenerateRequest(
            model="ollama/llama3",
            prompt="Extract",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError, match="Cannot connect"):
            provider.generate(request)

    @patch("httpx.post")
    def test_http_error_raises_provider_error(self, mock_post):
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_request = MagicMock()

        mock_post.side_effect = httpx.HTTPStatusError(
            "Server Error", request=mock_request, response=mock_response
        )

        provider = self._make_provider()
        request = GenerateRequest(
            model="ollama/llama3",
            prompt="Extract",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError, match="HTTP"):
            provider.generate(request)

    @patch("httpx.post")
    def test_malformed_json_raises_provider_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "not json {",
            "eval_count": 0,
        }
        mock_post.return_value = mock_resp

        provider = self._make_provider()
        request = GenerateRequest(
            model="ollama/llama3",
            prompt="Extract user",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError, match="Failed to parse JSON"):
            provider.generate(request)

    def test_regex_raises_unsupported(self):
        provider = self._make_provider()
        request = GenerateRequest(
            model="ollama/llama3",
            prompt="Extract",
            regex=r"\d+",
        )
        with pytest.raises(UnsupportedConstraintError, match="regex"):
            provider.generate(request)

    def test_grammar_raises_unsupported(self):
        provider = self._make_provider()
        request = GenerateRequest(
            model="ollama/llama3",
            prompt="Extract",
            grammar="root ::= [a-z]+",
        )
        with pytest.raises(UnsupportedConstraintError, match="grammar"):
            provider.generate(request)


# ===================================================================
# vLLM provider tests
# ===================================================================


class TestVllmProvider:
    def _make_provider(
        self,
        mock_client: MagicMock,
        guided_backend: str | None = None,
    ):
        from fmtgen.providers.vllm_provider import VllmProvider

        provider = VllmProvider(
            base_url="http://localhost:8000/v1",
            guided_backend=guided_backend,
        )
        provider._get_client = lambda: mock_client  # type: ignore[assignment]
        return provider

    def test_json_schema_guided_json(self):
        mock_client = MagicMock()
        user_data = {"name": "Dana", "age": 35}
        mock_client.chat.completions.create.return_value = _make_openai_response(
            json.dumps(user_data)
        )

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="vllm/meta-llama/Llama-3-8B",
            prompt="Extract user",
            schema=SimpleUser,
        )
        result = provider.generate(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "guided_json" in call_kwargs["extra_body"]
        assert isinstance(result.parsed, SimpleUser)
        assert result.parsed.name == "Dana"
        assert result.provider_used == "vllm"

    def test_regex_guided_regex(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response("12345")

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="vllm/meta-llama/Llama-3-8B",
            prompt="Give a number",
            regex=r"\d+",
        )
        result = provider.generate(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["guided_regex"] == r"\d+"
        assert result.raw == "12345"

    def test_grammar_guided_grammar(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response("hello")

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="vllm/meta-llama/Llama-3-8B",
            prompt="Say something",
            grammar="root ::= [a-z]+",
        )
        result = provider.generate(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["guided_grammar"] == "root ::= [a-z]+"
        assert result.raw == "hello"

    def test_choice_guided_choice(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="vllm/meta-llama/Llama-3-8B",
            prompt="Is it?",
            choice=["yes", "no"],
        )
        result = provider.generate(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["guided_choice"] == ["yes", "no"]
        assert result.raw == "yes"

    def test_guided_backend_passed_through(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response(
            json.dumps({"name": "X", "age": 1})
        )

        provider = self._make_provider(mock_client, guided_backend="outlines")
        request = GenerateRequest(
            model="vllm/model",
            prompt="Extract",
            schema=SimpleUser,
        )
        result = provider.generate(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["guided_decoding_backend"] == "outlines"
        assert result.backend_used == "outlines"

    def test_empty_choices_raises_provider_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_empty_response()

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="vllm/model",
            prompt="Extract",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError, match="empty choices"):
            provider.generate(request)

    def test_api_exception_wrapped(self):
        mock_client = MagicMock()
        original_exc = RuntimeError("connection refused")
        mock_client.chat.completions.create.side_effect = original_exc

        provider = self._make_provider(mock_client)
        request = GenerateRequest(
            model="vllm/model",
            prompt="Extract",
            schema=SimpleUser,
        )
        with pytest.raises(ProviderError) as exc_info:
            provider.generate(request)
        assert exc_info.value.__cause__ is original_exc


# ===================================================================
# Sglang provider tests
# ===================================================================


class TestSglangProvider:
    def test_sglang_name(self):
        from fmtgen.providers.vllm_provider import SglangProvider

        provider = SglangProvider(base_url="http://localhost:8000/v1")
        assert provider.name == "sglang"

    def test_sglang_inherits_vllm_behavior(self):
        from fmtgen.providers.vllm_provider import SglangProvider

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response(
            json.dumps({"name": "Sg", "age": 1})
        )

        provider = SglangProvider(base_url="http://localhost:8000/v1")
        provider._get_client = lambda: mock_client  # type: ignore[assignment]

        request = GenerateRequest(
            model="sglang/model",
            prompt="Extract",
            schema=SimpleUser,
        )
        result = provider.generate(request)
        assert result.provider_used == "sglang"
        assert isinstance(result.parsed, SimpleUser)


# ===================================================================
# Fmt class provider routing tests
# ===================================================================


class TestFmtRouting:
    def test_routes_to_openai(self):
        from fmtgen.core import Fmt

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response(
            json.dumps({"name": "Alice", "age": 30})
        )

        fmt = Fmt()
        with patch("fmtgen.providers.ProviderRegistry.get") as mock_get:
            mock_provider = MagicMock()
            from fmtgen.types import GenerateResult

            mock_provider.generate.return_value = GenerateResult(
                raw=json.dumps({"name": "Alice", "age": 30}),
                parsed=SimpleUser(name="Alice", age=30),
                backend_used="none",
                provider_used="openai",
                model="openai/gpt-4o",
                latency_ms=100.0,
                tokens_generated=10,
                constraint_type="json_schema",
            )
            mock_get.return_value = mock_provider

            result = fmt.generate(
                model="openai/gpt-4o",
                prompt="Extract user",
                schema=SimpleUser,
            )
            assert isinstance(result, SimpleUser)
            assert result.name == "Alice"
            mock_get.assert_called_once_with("openai")

    def test_unknown_provider_raises_generation_error(self):
        from fmtgen.core import Fmt

        fmt = Fmt()
        with pytest.raises(GenerationError, match="Cannot route model"):
            fmt.generate(
                model="unknown_provider/some-model",
                prompt="Extract user",
                schema=SimpleUser,
            )

    def test_routes_to_anthropic(self):
        from fmtgen.core import Fmt

        fmt = Fmt()
        with patch("fmtgen.providers.ProviderRegistry.get") as mock_get:
            mock_provider = MagicMock()
            from fmtgen.types import GenerateResult

            mock_provider.generate.return_value = GenerateResult(
                raw=json.dumps({"name": "Bob", "age": 25}),
                parsed=SimpleUser(name="Bob", age=25),
                backend_used="none",
                provider_used="anthropic",
                model="anthropic/claude-sonnet-4-20250514",
                latency_ms=100.0,
                tokens_generated=10,
                constraint_type="json_schema",
            )
            mock_get.return_value = mock_provider

            result = fmt.generate(
                model="anthropic/claude-sonnet-4-20250514",
                prompt="Extract user",
                schema=SimpleUser,
            )
            assert isinstance(result, SimpleUser)
            assert result.name == "Bob"
            mock_get.assert_called_once_with("anthropic")

    def test_no_constraint_raises_value_error(self):
        from fmtgen.core import Fmt

        fmt = Fmt()
        with pytest.raises(ValueError, match="No constraint specified"):
            fmt.generate(
                model="openai/gpt-4o",
                prompt="Just a prompt with no constraint",
            )


# ===================================================================
# Schema edge cases
# ===================================================================


class TestSchemaEdgeCases:
    def _make_openai_provider(self, mock_client: MagicMock):
        from fmtgen.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test")
        provider._get_client = lambda: mock_client  # type: ignore[assignment]
        return provider

    def test_nested_model_with_refs(self):
        mock_client = MagicMock()
        data = {
            "name": "Eve",
            "address": {"street": "123 Main St", "city": "NYC"},
        }
        mock_client.chat.completions.create.return_value = _make_openai_response(json.dumps(data))

        provider = self._make_openai_provider(mock_client)
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract",
            schema=UserWithAddress,
        )
        result = provider.generate(request)

        assert isinstance(result.parsed, UserWithAddress)
        assert isinstance(result.parsed.address, Address)
        assert result.parsed.address.city == "NYC"

        # Verify the schema sent has no $ref (resolved)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        schema = call_kwargs["response_format"]["json_schema"]["schema"]
        schema_str = json.dumps(schema)
        assert "$ref" not in schema_str

    def test_deeply_nested_model(self):
        mock_client = MagicMock()
        data = {
            "name": "Acme Corp",
            "ceo": {
                "name": "Jane",
                "address": {"street": "1 CEO Lane", "city": "SV"},
            },
            "employees": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
        }
        mock_client.chat.completions.create.return_value = _make_openai_response(json.dumps(data))

        provider = self._make_openai_provider(mock_client)
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract company",
            schema=Company,
        )
        result = provider.generate(request)

        assert isinstance(result.parsed, Company)
        assert result.parsed.ceo.address.city == "SV"
        assert len(result.parsed.employees) == 2

    def test_optional_fields_and_enums(self):
        mock_client = MagicMock()
        data = {
            "name": "Zoe",
            "age": None,
            "tags": ["dev", "lead"],
            "color": "red",
        }
        mock_client.chat.completions.create.return_value = _make_openai_response(json.dumps(data))

        provider = self._make_openai_provider(mock_client)
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract user",
            schema=UserWithOptionals,
        )
        result = provider.generate(request)

        assert isinstance(result.parsed, UserWithOptionals)
        assert result.parsed.age is None
        assert result.parsed.tags == ["dev", "lead"]
        assert result.parsed.color == Color.RED

    def test_optional_fields_defaults(self):
        mock_client = MagicMock()
        data = {"name": "Minimal"}
        mock_client.chat.completions.create.return_value = _make_openai_response(json.dumps(data))

        provider = self._make_openai_provider(mock_client)
        request = GenerateRequest(
            model="openai/gpt-4o",
            prompt="Extract user",
            schema=UserWithOptionals,
        )
        result = provider.generate(request)

        assert isinstance(result.parsed, UserWithOptionals)
        assert result.parsed.name == "Minimal"
        assert result.parsed.age is None
        assert result.parsed.tags == []
        assert result.parsed.color is None
