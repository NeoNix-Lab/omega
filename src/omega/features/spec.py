"""Declarative feature specs (``omega.feature-spec/v1``).

A spec selects a registered kernel, binds its input representation (trades or fixed-duration
candles) and parameters, and has a deterministic *identity*: the SHA-256 of its canonical form
(defaults filled in, durations normalised to nanoseconds), so two spellings of the same feature
share one identity and any semantic change produces a new one.

Example::

    schema: omega.feature-spec/v1
    name: momentum_5m_12
    version: 1
    inputs: {representation: candles, bar: 5m}
    kernel: price_momentum
    params: {window: 12}
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from .. import platform_link as qp
from .kernels import KernelDef, ParamDef, get_kernel
from .kernels import KernelError as KernelError  # re-exported for callers

SCHEMA = "omega.feature-spec/v1"
IDENTITY_DOMAIN = "omega-feature-spec-v1"
KEY_PREFIX = "feature."

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TOP_LEVEL_KEYS = {"schema", "name", "version", "inputs", "kernel", "params", "warmup", "output"}
_REQUIRED_KEYS = {"schema", "name", "version", "inputs", "kernel"}


class FeatureSpecError(ValueError):
    """A feature spec is malformed or semantically invalid."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: int
    representation: str  # "trades" | "candles"
    bar_ns: int | None  # candle duration in ns (candles only)
    kernel: str
    params: tuple[tuple[str, int | Decimal], ...]  # defaults filled, sorted by name
    warmup: int  # units (trades or closed candles) consumed before the first emission
    output: str

    @property
    def key(self) -> str:
        """The ``StrategyInput.key`` this feature is emitted under."""
        return f"{KEY_PREFIX}{self.name}"

    @property
    def params_dict(self) -> dict[str, int | Decimal]:
        return dict(self.params)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "version": self.version,
            "inputs": {
                "representation": self.representation,
                "bar_ns": None if self.bar_ns is None else str(self.bar_ns),
            },
            "kernel": self.kernel,
            "params": {name: _canonical_param(value) for name, value in self.params},
            "warmup": self.warmup,
            "output": self.output,
        }

    @property
    def identity(self) -> str:
        payload = json.dumps(
            self.stable_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
        return f"{IDENTITY_DOMAIN}:sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _canonical_param(value: int | Decimal) -> int | str:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return str(int(value))
        return format(value.normalize(), "f")
    return value


# ---------------------------------------------------------------------------------------------
# YAML loading (strict: duplicate keys are rejected)
# ---------------------------------------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_feature_spec(path: str | Path) -> FeatureSpec:
    """Load and validate a feature spec file."""
    path = Path(path)
    return parse_feature_spec_text(path.read_text(encoding="utf-8"), source=str(path))


def parse_feature_spec_text(text: str, *, source: str = "<spec>") -> FeatureSpec:
    try:
        raw = yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise FeatureSpecError(f"{source}: invalid YAML: {exc}") from exc
    return parse_feature_spec(raw, source=source)


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------


def parse_feature_spec(raw: Any, *, source: str = "<spec>") -> FeatureSpec:
    def fail(message: str) -> FeatureSpecError:
        return FeatureSpecError(f"{source}: {message}")

    if not isinstance(raw, Mapping):
        raise fail("a feature spec must be a mapping")
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS, key=str)
    if unknown:
        raise fail(f"unknown key(s) {unknown}; allowed: {sorted(_TOP_LEVEL_KEYS)}")
    missing = sorted(_REQUIRED_KEYS - set(raw))
    if missing:
        raise fail(f"missing required key(s) {missing}")

    if raw["schema"] != SCHEMA:
        raise fail(f"schema must be {SCHEMA!r}, got {raw['schema']!r}")

    name = raw["name"]
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise fail(f"name must match {_NAME_RE.pattern}, got {name!r}")

    version = raw["version"]
    if type(version) is not int or version < 1:
        raise fail(f"version must be an integer >= 1, got {version!r}")

    kernel_id = raw["kernel"]
    if not isinstance(kernel_id, str):
        raise fail("kernel must be a string")
    try:
        kernel = get_kernel(kernel_id)
    except KernelError as exc:
        raise fail(str(exc)) from exc

    representation, bar_ns = _parse_inputs(raw["inputs"], kernel, fail)
    params = _parse_params(raw.get("params"), kernel, fail)

    minimum_warmup = kernel.min_warmup(dict(params))
    warmup = raw.get("warmup")
    if warmup is None:
        warmup = minimum_warmup
    elif type(warmup) is not int or warmup < minimum_warmup:
        raise fail(
            f"warmup must be an integer >= {minimum_warmup} {kernel.unit} "
            f"(the minimum for kernel {kernel_id!r} with these params), got {warmup!r}"
        )

    output = raw.get("output", kernel.output)
    if output != kernel.output:
        raise fail(f"output must be {kernel.output!r} for kernel {kernel_id!r}, got {output!r}")

    return FeatureSpec(
        name=name,
        version=version,
        representation=representation,
        bar_ns=bar_ns,
        kernel=kernel_id,
        params=tuple(sorted(params.items())),
        warmup=warmup,
        output=output,
    )


def _parse_inputs(raw: Any, kernel: KernelDef, fail: Any) -> tuple[str, int | None]:
    if not isinstance(raw, Mapping):
        raise fail("inputs must be a mapping")
    unknown = sorted(set(raw) - {"representation", "bar"}, key=str)
    if unknown:
        raise fail(f"unknown inputs key(s) {unknown}; allowed: ['bar', 'representation']")
    representation = raw.get("representation")
    if representation not in {"trades", "candles"}:
        raise fail(f"inputs.representation must be 'trades' or 'candles', got {representation!r}")
    if representation != kernel.representation:
        raise fail(
            f"kernel {kernel.kernel_id!r} works on {kernel.representation}, "
            f"but inputs.representation is {representation!r}"
        )
    if representation == "trades":
        if "bar" in raw:
            raise fail("inputs.bar is only valid for candle inputs")
        return representation, None
    if "bar" not in raw:
        raise fail("inputs.bar (candle duration, e.g. 5m) is required for candle inputs")
    bar = raw["bar"]
    if type(bar) not in {int, str}:
        raise fail(f"inputs.bar must be a duration string like '5m' or an integer of ns, got {bar!r}")
    try:
        return representation, int(qp.CandleDefinitionV1.from_duration(bar).duration_ns)
    except qp.InvalidRequest as exc:
        raise fail(f"inputs.bar: {exc}") from exc


def _parse_params(raw: Any, kernel: KernelDef, fail: Any) -> dict[str, int | Decimal]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise fail("params must be a mapping")
    declared = {param.name: param for param in kernel.params}
    unknown = sorted(set(raw) - set(declared), key=str)
    if unknown:
        raise fail(f"unknown param(s) {unknown} for kernel {kernel.kernel_id!r}; allowed: {sorted(declared)}")
    params: dict[str, int | Decimal] = {}
    for name, definition in declared.items():
        params[name] = _coerce_param(raw[name], definition, fail) if name in raw else definition.default
    return params


def _coerce_param(value: Any, definition: ParamDef, fail: Any) -> int | Decimal:
    label = f"param {definition.name!r}"
    if isinstance(value, bool):
        raise fail(f"{label} must be a {definition.kind}, got a boolean")
    if definition.kind == "int":
        if type(value) is not int:
            raise fail(f"{label} must be an integer, got {value!r}")
        number: int | Decimal = value
    else:
        if isinstance(value, float):
            raise fail(f"{label} must be quoted (e.g. '0.5') or an integer: floats are not exact, got {value!r}")
        try:
            number = Decimal(value) if isinstance(value, (str, int)) else None  # type: ignore[assignment]
        except InvalidOperation:
            number = None  # type: ignore[assignment]
        if number is None or not number.is_finite():
            raise fail(f"{label} must be a finite decimal, got {value!r}")
    if definition.minimum is not None and number < definition.minimum:
        raise fail(f"{label} must be >= {definition.minimum}, got {value!r}")
    if definition.maximum is not None and number > definition.maximum:
        raise fail(f"{label} must be <= {definition.maximum}, got {value!r}")
    return number
