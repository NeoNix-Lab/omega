"""Shared helpers for declarative spec files: strict YAML loading and canonical identities."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any

import yaml


class _StrictLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys (PyYAML silently keeps the last one)."""


def _construct_unique_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_yaml_strict(text: str) -> Any:
    """Parse YAML text; raises ``yaml.YAMLError`` (including on duplicate keys)."""
    return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass


def canonical_decimal(value: int | Decimal) -> int | str:
    """Stable text for a number: ``Decimal('0.50')`` and ``Decimal('0.5')`` give the same value."""
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return str(int(value))
        return format(value.normalize(), "f")
    return value


def identity_hash(payload: Any) -> str:
    """SHA-256 hex digest of the canonical JSON form of ``payload``."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
