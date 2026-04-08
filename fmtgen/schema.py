from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def pydantic_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    return resolve_refs(schema)


def resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.pop("$defs", {})
    if not defs:
        return schema
    resolved: dict[str, Any] = _resolve_node(schema, defs)
    resolved.pop("$defs", None)
    return resolved


def _resolve_node(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            ref_path = node["$ref"]
            ref_name = ref_path.rsplit("/", 1)[-1]
            if ref_name in defs:
                resolved = _resolve_node(dict(defs[ref_name]), defs)
                return resolved
            return node
        return {k: _resolve_node(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_node(item, defs) for item in node]
    return node


def validate_json_output(raw: str, model: type[BaseModel]) -> BaseModel:
    data = json.loads(raw)
    return model.model_validate(data)


def json_schema_to_string(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2)
