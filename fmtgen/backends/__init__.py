from __future__ import annotations

from typing import Any

from fmtgen.backends.base import BaseBackend
from fmtgen.exceptions import BackendNotAvailableError, NoBackendAvailableError

_BACKEND_PRIORITY = [
    "xgrammar",
    "llguidance",
    "outlines",
    "lm-format-enforcer",
]


def _get_backend_class(name: str) -> type[BaseBackend]:
    if name == "xgrammar":
        from fmtgen.backends.xgrammar_backend import XGrammarBackend

        return XGrammarBackend
    if name == "llguidance":
        from fmtgen.backends.llguidance_backend import LlguidanceBackend

        return LlguidanceBackend
    if name == "outlines":
        from fmtgen.backends.outlines_backend import OutlinesBackend

        return OutlinesBackend
    if name == "lm-format-enforcer":
        from fmtgen.backends.lmfe_backend import LmfeBackend

        return LmfeBackend
    raise BackendNotAvailableError(name, available=list_available())


class BackendRegistry:
    @staticmethod
    def get(name: str) -> BaseBackend:
        cls = _get_backend_class(name)
        if not cls.is_available():
            raise BackendNotAvailableError(name, available=list_available())
        return cls()

    @staticmethod
    def auto_select(constraint_type: str | None = None) -> BaseBackend:
        for name in _BACKEND_PRIORITY:
            try:
                cls = _get_backend_class(name)
            except BackendNotAvailableError:
                continue
            if not cls.is_available():
                continue
            if constraint_type and constraint_type not in cls.supported_constraints():
                continue
            return cls()
        raise NoBackendAvailableError()

    @staticmethod
    def list_available() -> list[str]:
        return list_available()

    @staticmethod
    def list_all() -> list[dict[str, Any]]:
        result = []
        for name in _BACKEND_PRIORITY:
            try:
                cls = _get_backend_class(name)
                available = cls.is_available()
                constraints = cls.supported_constraints()
            except Exception:
                available = False
                constraints = []
            result.append(
                {
                    "name": name,
                    "available": available,
                    "constraints": constraints,
                }
            )
        return result


def list_available() -> list[str]:
    available = []
    for name in _BACKEND_PRIORITY:
        try:
            cls = _get_backend_class(name)
            if cls.is_available():
                available.append(name)
        except Exception:
            continue
    return available
