from __future__ import annotations

from typing import Any

from fmtgen.exceptions import ProviderError
from fmtgen.providers.base import BaseProvider

_PROVIDER_MAP = {
    "openai": "fmtgen.providers.openai_provider:OpenAIProvider",
    "anthropic": "fmtgen.providers.anthropic_provider:AnthropicProvider",
    "ollama": "fmtgen.providers.ollama_provider:OllamaProvider",
    "vllm": "fmtgen.providers.vllm_provider:VllmProvider",
    "sglang": "fmtgen.providers.vllm_provider:SglangProvider",
}


class ProviderRegistry:
    @staticmethod
    def get(name: str, **kwargs: Any) -> BaseProvider:
        if name not in _PROVIDER_MAP:
            raise ProviderError(
                name,
                f"Unknown provider '{name}'. "
                f"Available providers: {', '.join(_PROVIDER_MAP.keys())}",
            )

        module_path, class_name = _PROVIDER_MAP[name].rsplit(":", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        result: BaseProvider = cls(**kwargs)
        return result

    @staticmethod
    def list_providers() -> list[str]:
        return list(_PROVIDER_MAP.keys())
