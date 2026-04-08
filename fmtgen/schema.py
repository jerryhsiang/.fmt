from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def pydantic_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    return resolve_refs(schema)


def resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})
    if not defs:
        return schema
    resolved: dict[str, Any] = _resolve_node(schema, defs, set())
    resolved.pop("$defs", None)
    return resolved


_MAX_REF_DEPTH = 50


def _resolve_node(node: Any, defs: dict[str, Any], resolving: set[str], depth: int = 0) -> Any:
    if depth > _MAX_REF_DEPTH:
        return node
    if isinstance(node, dict):
        if "$ref" in node:
            ref_path = node["$ref"]
            ref_name = ref_path.rsplit("/", 1)[-1]
            if ref_name in resolving:
                # Self-referential schema — inline without further recursion
                resolved = dict(defs[ref_name])
                resolved.pop("$ref", None)
                return {k: v for k, v in resolved.items() if k != "$ref"}
            if ref_name in defs:
                resolving.add(ref_name)
                resolved = _resolve_node(dict(defs[ref_name]), defs, resolving, depth + 1)
                resolving.discard(ref_name)
                return resolved
            return node
        return {k: _resolve_node(v, defs, resolving, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_node(item, defs, resolving, depth + 1) for item in node]
    return node


def validate_json_output(raw: str, model: type[BaseModel]) -> BaseModel:
    data = json.loads(raw)
    return model.model_validate(data)


def json_schema_to_string(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2)
