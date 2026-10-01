"""Declarative strategy specs (``omega.strategy-spec/v1``).

A strategy spec says *when to enter* (rules over feature values), *when to exit* (take-profit,
stop-loss, maximum holding time, optional rules), and the sizing/risk/cooldown parameters. It is
validated and given a deterministic identity here; :mod:`omega.strategies.compiler` turns it into a
platform ``StrategySpec`` plus a signal provider.

Example::

    schema: omega.strategy-spec/v1
    name: breakout_long
    version: 1
    features: [momentum_1m_5]
    entry:
      direction: long
      rules:
        - {feature: momentum_1m_5, op: ">", value: "0.003"}
    exit:
      take_profit_pct: "2.0"
      stop_loss_pct: "0.8"
    position: {size: "0.01"}
    sizing: {lot_size: "0.001", min_size: "0.001"}

Limits of the platform strategy model that this format inherits (see docs/STRATEGY_SPECS.md):
one entry direction per spec, one open position at a time, market orders only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from .. import platform_link as qp
from ..specio import canonical_decimal, identity_hash, load_yaml_strict

SCHEMA = "omega.strategy-spec/v1"
IDENTITY_DOMAIN = "omega-strategy-spec-v1"

COMPARATORS = (">", "<", ">=", "<=")

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TOP_LEVEL_KEYS = {
    "schema",
    "name",
    "version",
    "features",
    "entry",
    "exit",
    "position",
    "sizing",
    "risk",
    "session",
    "cooldown",
}
_REQUIRED_KEYS = {"schema", "name", "version", "features", "entry", "exit", "position", "sizing"}

# Defaults for the optional sections. They are filled in before the identity is computed, so an
# explicit default and an omitted section are the same strategy. The risk defaults do NOT constrain
# anything (every fraction is 1): set them explicitly to study the effect of risk limits.
RISK_DEFAULTS: dict[str, Decimal | int] = {
    "max_drawdown_fraction": Decimal(1),
    "max_position_notional_fraction": Decimal(1),
    "max_total_exposure_fraction": Decimal(1),
    "risk_per_trade_fraction": Decimal(1),
    "max_evidence_age_seconds": 3600,
}
COOLDOWN_DEFAULTS: dict[str, int] = {
    "post_loss_cooldown_seconds": 0,
    "consecutive_loss_count": 1_000_000,
    "consecutive_loss_cooldown_seconds": 0,
    "max_entries_per_utc_day": 1_000_000,
    "max_trade_history_age_seconds": 3600,
}
SESSIONS = ("always_open",)


class StrategySpecError(ValueError):
    """A strategy spec is malformed or semantically invalid."""


@dataclass(frozen=True)
class Rule:
    feature: str
    op: str
    value: Decimal

    def stable_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "op": self.op, "value": canonical_decimal(self.value)}


@dataclass(frozen=True)
class EntrySpec:
    direction: str  # "long" | "short"
    mode: str  # "all" | "any"
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class ExitSpec:
    take_profit_pct: Decimal | None
    stop_loss_pct: Decimal | None
    max_holding_ns: int | None
    rules: tuple[Rule, ...]  # any rule true -> exit


@dataclass(frozen=True)
class StrategyDef:
    name: str
    version: int
    features: tuple[str, ...]
    entry: EntrySpec
    exit: ExitSpec
    position_size: Decimal
    lot_size: Decimal
    min_size: Decimal
    max_size: Decimal | None
    risk: tuple[tuple[str, Decimal | int], ...]  # defaults filled, sorted by name
    session: str
    cooldown: tuple[tuple[str, int], ...]  # defaults filled, sorted by name

    @property
    def risk_dict(self) -> dict[str, Decimal | int]:
        return dict(self.risk)

    @property
    def cooldown_dict(self) -> dict[str, int]:
        return dict(self.cooldown)

    def stable_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "version": self.version,
            "features": list(self.features),
            "entry": {
                "direction": self.entry.direction,
                "mode": self.entry.mode,
                "rules": [rule.stable_dict() for rule in self.entry.rules],
            },
            "exit": {
                "take_profit_pct": _opt(self.exit.take_profit_pct),
                "stop_loss_pct": _opt(self.exit.stop_loss_pct),
                "max_holding_ns": None if self.exit.max_holding_ns is None else str(self.exit.max_holding_ns),
                "rules": [rule.stable_dict() for rule in self.exit.rules],
            },
            "position": {"size": canonical_decimal(self.position_size)},
            "sizing": {
                "lot_size": canonical_decimal(self.lot_size),
                "min_size": canonical_decimal(self.min_size),
                "max_size": _opt(self.max_size),
            },
            "risk": {key: canonical_decimal(value) for key, value in self.risk},
            "session": self.session,
            "cooldown": dict(self.cooldown),
        }

    @property
    def identity(self) -> str:
        return f"{IDENTITY_DOMAIN}:sha256:{identity_hash(self.stable_dict())}"


def _opt(value: Decimal | None) -> int | str | None:
    return None if value is None else canonical_decimal(value)


# ---------------------------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------------------------


def load_strategy_spec(path: str | Path) -> StrategyDef:
    path = Path(path)
    return parse_strategy_spec_text(path.read_text(encoding="utf-8"), source=str(path))


def parse_strategy_spec_text(text: str, *, source: str = "<spec>") -> StrategyDef:
    try:
        raw = load_yaml_strict(text)
    except yaml.YAMLError as exc:
        raise StrategySpecError(f"{source}: invalid YAML: {exc}") from exc
    return parse_strategy_spec(raw, source=source)


def parse_strategy_spec(raw: Any, *, source: str = "<spec>") -> StrategyDef:
    def fail(message: str) -> StrategySpecError:
        return StrategySpecError(f"{source}: {message}")

    if not isinstance(raw, Mapping):
        raise fail("a strategy spec must be a mapping")
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

    features = _parse_features(raw["features"], fail)
    entry = _parse_entry(raw["entry"], features, fail)
    exit_ = _parse_exit(raw["exit"], features, fail)

    position = _section(raw["position"], "position", {"size"}, fail)
    if "size" not in position:
        raise fail("position.size is required")
    position_size = _decimal(position["size"], "position.size", fail, minimum=Decimal(0), exclusive=True)

    sizing = _section(raw["sizing"], "sizing", {"lot_size", "min_size", "max_size"}, fail)
    if "lot_size" not in sizing:
        raise fail("sizing.lot_size is required")
    lot_size = _decimal(sizing["lot_size"], "sizing.lot_size", fail, minimum=Decimal(0), exclusive=True)
    min_size = _decimal(sizing.get("min_size", 0), "sizing.min_size", fail, minimum=Decimal(0))
    max_size = None
    if sizing.get("max_size") is not None:
        max_size = _decimal(sizing["max_size"], "sizing.max_size", fail, minimum=Decimal(0), exclusive=True)
        if max_size < min_size:
            raise fail("sizing.max_size must be >= sizing.min_size")

    risk = _parse_risk(raw.get("risk"), fail)
    session = raw.get("session", SESSIONS[0])
    if session not in SESSIONS:
        raise fail(f"session must be one of {list(SESSIONS)}, got {session!r}")
    cooldown = _parse_cooldown(raw.get("cooldown"), fail)

    return StrategyDef(
        name=name,
        version=version,
        features=features,
        entry=entry,
        exit=exit_,
        position_size=position_size,
        lot_size=lot_size,
        min_size=min_size,
        max_size=max_size,
        risk=tuple(sorted(risk.items())),
        session=session,
        cooldown=tuple(sorted(cooldown.items())),
    )


def _section(raw: Any, label: str, allowed: set[str], fail: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise fail(f"{label} must be a mapping")
    unknown = sorted(set(raw) - allowed, key=str)
    if unknown:
        raise fail(f"unknown {label} key(s) {unknown}; allowed: {sorted(allowed)}")
    return raw


def _decimal(
    value: Any,
    label: str,
    fail: Any,
    *,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
    exclusive: bool = False,
    exclusive_max: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise fail(f"{label} must be a decimal, got a boolean")
    if isinstance(value, float):
        raise fail(f"{label} must be quoted (e.g. '0.003') or an integer: floats are not exact, got {value!r}")
    try:
        number = Decimal(value) if isinstance(value, (str, int)) else None
    except InvalidOperation:
        number = None
    if number is None or not number.is_finite():
        raise fail(f"{label} must be a finite decimal, got {value!r}")
    if minimum is not None and (number <= minimum if exclusive else number < minimum):
        raise fail(f"{label} must be {'>' if exclusive else '>='} {minimum}, got {value!r}")
    if maximum is not None and (number >= maximum if exclusive_max else number > maximum):
        raise fail(f"{label} must be {'<' if exclusive_max else '<='} {maximum}, got {value!r}")
    return number


def _parse_features(raw: Any, fail: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise fail("features must be a non-empty list of feature spec names")
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not _NAME_RE.fullmatch(item):
            raise fail(f"features entries must be feature names matching {_NAME_RE.pattern}, got {item!r}")
        names.append(item)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise fail(f"duplicate feature(s) {duplicates}")
    return tuple(names)


def _parse_rules(raw: Any, label: str, features: tuple[str, ...], fail: Any) -> tuple[Rule, ...]:
    if not isinstance(raw, list):
        raise fail(f"{label} must be a list of rules")
    rules = []
    for index, item in enumerate(raw):
        where = f"{label}[{index}]"
        rule = _section(item, where, {"feature", "op", "value"}, fail)
        missing = sorted({"feature", "op", "value"} - set(rule))
        if missing:
            raise fail(f"{where} is missing {missing}")
        feature = rule["feature"]
        if feature not in features:
            raise fail(f"{where}.feature {feature!r} is not listed in features {list(features)}")
        op = rule["op"]
        if op not in COMPARATORS:
            raise fail(f"{where}.op must be one of {list(COMPARATORS)}, got {op!r}")
        rules.append(Rule(feature=feature, op=op, value=_decimal(rule["value"], f"{where}.value", fail)))
    return tuple(rules)


def _parse_entry(raw: Any, features: tuple[str, ...], fail: Any) -> EntrySpec:
    entry = _section(raw, "entry", {"direction", "mode", "rules"}, fail)
    direction = entry.get("direction")
    if direction in {"both", "long_short", "any"}:
        raise fail(
            "entry.direction 'both' is not supported: the platform strategy model has one entry direction per "
            "spec. Write two specs (one 'long', one 'short') and replay them separately"
        )
    if direction not in {"long", "short"}:
        raise fail(f"entry.direction must be 'long' or 'short', got {direction!r}")
    mode = entry.get("mode", "all")
    if mode not in {"all", "any"}:
        raise fail(f"entry.mode must be 'all' or 'any', got {mode!r}")
    if "rules" not in entry:
        raise fail("entry.rules is required")
    rules = _parse_rules(entry["rules"], "entry.rules", features, fail)
    if not rules:
        raise fail("entry.rules must contain at least one rule")
    return EntrySpec(direction=direction, mode=mode, rules=rules)


def _parse_exit(raw: Any, features: tuple[str, ...], fail: Any) -> ExitSpec:
    section = _section(raw, "exit", {"take_profit_pct", "stop_loss_pct", "max_holding", "rules"}, fail)
    take_profit = None
    if section.get("take_profit_pct") is not None:
        take_profit = _decimal(
            section["take_profit_pct"], "exit.take_profit_pct", fail, minimum=Decimal(0), exclusive=True
        )
    stop_loss = None
    if section.get("stop_loss_pct") is not None:
        stop_loss = _decimal(
            section["stop_loss_pct"],
            "exit.stop_loss_pct",
            fail,
            minimum=Decimal(0),
            exclusive=True,
            maximum=Decimal(100),
            exclusive_max=True,
        )
    max_holding_ns = None
    if section.get("max_holding") is not None:
        value = section["max_holding"]
        if type(value) not in {int, str}:
            raise fail(f"exit.max_holding must be a duration like '4h' or an integer of ns, got {value!r}")
        try:
            max_holding_ns = int(qp.CandleDefinitionV1.from_duration(value).duration_ns)
        except qp.InvalidRequest as exc:
            raise fail(f"exit.max_holding: {exc}") from exc
    rules = _parse_rules(section.get("rules", []), "exit.rules", features, fail)
    if take_profit is None and stop_loss is None and max_holding_ns is None and not rules:
        raise fail(
            "exit needs at least one of take_profit_pct, stop_loss_pct, max_holding, rules "
            "(a strategy without an exit would hold its position forever)"
        )
    return ExitSpec(take_profit_pct=take_profit, stop_loss_pct=stop_loss, max_holding_ns=max_holding_ns, rules=rules)


def _parse_risk(raw: Any, fail: Any) -> dict[str, Decimal | int]:
    section = _section(raw if raw is not None else {}, "risk", set(RISK_DEFAULTS), fail)
    risk: dict[str, Decimal | int] = {}
    for key, default in RISK_DEFAULTS.items():
        if key not in section:
            risk[key] = default
        elif key == "max_evidence_age_seconds":
            value = section[key]
            if type(value) is not int or value < 0:
                raise fail(f"risk.{key} must be a non-negative integer, got {value!r}")
            risk[key] = value
        else:
            risk[key] = _decimal(
                section[key], f"risk.{key}", fail, minimum=Decimal(0), exclusive=True, maximum=Decimal(1)
            )
    return risk


def _parse_cooldown(raw: Any, fail: Any) -> dict[str, int]:
    section = _section(raw if raw is not None else {}, "cooldown", set(COOLDOWN_DEFAULTS), fail)
    cooldown: dict[str, int] = {}
    for key, default in COOLDOWN_DEFAULTS.items():
        value = section.get(key, default)
        minimum = 1 if key in {"consecutive_loss_count", "max_entries_per_utc_day"} else 0
        if type(value) is not int or value < minimum:
            raise fail(f"cooldown.{key} must be an integer >= {minimum}, got {value!r}")
        cooldown[key] = value
    return cooldown
