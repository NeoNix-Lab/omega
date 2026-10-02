"""Strategy specs: validation, identity, compilation to a platform StrategySpec, decisions, signal layer."""

from __future__ import annotations

import glob
import textwrap
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("quant_platform")

from omega import platform_link as qp  # noqa: E402
from omega.features import load_feature_spec, parse_feature_spec_text  # noqa: E402
from omega.strategies import (  # noqa: E402
    StrategySpecError,
    compile_strategy,
    load_strategy_spec,
    parse_strategy_spec_text,
)
from omega.strategies.signals import SignalProvider  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS = REPO_ROOT / "studies" / "specs"

FEATURE_TEXT = """\
schema: omega.feature-spec/v1
name: mom
version: 1
inputs: {representation: candles, bar: 1m}
kernel: price_momentum
params: {window: 1}
"""

BASE = """\
schema: omega.strategy-spec/v1
name: breakout_long
version: 1
features: [mom]
entry:
  direction: long
  rules:
    - {feature: mom, op: ">", value: "0.003"}
exit:
  take_profit_pct: "2.0"
  stop_loss_pct: "0.8"
position: {size: "0.01"}
sizing: {lot_size: "0.001", min_size: "0.001"}
"""


def _feature(text: str = FEATURE_TEXT) -> Any:
    return parse_feature_spec_text(text)


def _compile(text: str = BASE, features: list[Any] | None = None) -> Any:
    return compile_strategy(parse_strategy_spec_text(text), features or [_feature()])


# ---------------------------------------------------------------------------------------------
# Parsing and identity
# ---------------------------------------------------------------------------------------------


def test_valid_spec_fields() -> None:
    spec = parse_strategy_spec_text(BASE)
    assert spec.name == "breakout_long"
    assert spec.features == ("mom",)
    assert spec.entry.direction == "long" and spec.entry.mode == "all"
    assert spec.entry.rules[0].value == Decimal("0.003")
    assert spec.exit.take_profit_pct == Decimal("2.0") and spec.exit.stop_loss_pct == Decimal("0.8")
    assert spec.position_size == Decimal("0.01")
    assert spec.session == "always_open"
    assert spec.identity.startswith("omega-strategy-spec-v1:sha256:")


def test_identity_is_stable_across_formatting_and_key_order() -> None:
    reordered = textwrap.dedent(
        """
        sizing: {min_size: "0.001", lot_size: "0.001"}
        position:
          size: "0.0100"
        exit:
          stop_loss_pct: "0.80"
          take_profit_pct: "2"
        entry:
          rules: [{value: "0.0030", op: ">", feature: mom}]
          direction: long
        features:
          - mom
        version: 1
        name: breakout_long
        schema: omega.strategy-spec/v1
        """
    )
    a, b = parse_strategy_spec_text(BASE), parse_strategy_spec_text(reordered)
    assert a.identity == b.identity


def test_explicit_defaults_equal_omitted_sections() -> None:
    explicit = BASE + textwrap.dedent(
        """\
        session: always_open
        risk:
          max_drawdown_fraction: "1"
          max_position_notional_fraction: 1
          max_total_exposure_fraction: "1.0"
          risk_per_trade_fraction: "1"
          max_evidence_age_seconds: 3600
        cooldown:
          post_loss_cooldown_seconds: 0
          consecutive_loss_count: 1000000
          consecutive_loss_cooldown_seconds: 0
          max_entries_per_utc_day: 1000000
          max_trade_history_age_seconds: 3600
        """
    )
    assert parse_strategy_spec_text(explicit).identity == parse_strategy_spec_text(BASE).identity


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('value: "0.003"', 'value: "0.004"'),  # one unit in the last decimal place
        ('value: "0.003"', 'value: "0.0031"'),
        ('op: ">"', 'op: ">="'),
        ("direction: long", "direction: short"),
        ('take_profit_pct: "2.0"', 'take_profit_pct: "2.1"'),
        ('stop_loss_pct: "0.8"', 'stop_loss_pct: "0.9"'),
        ('size: "0.01"', 'size: "0.02"'),
        ('lot_size: "0.001"', 'lot_size: "0.01"'),
        ("name: breakout_long", "name: breakout_other"),
        ("version: 1", "version: 2"),
    ],
)
def test_identity_changes_with_any_semantic_change(old: str, new: str) -> None:
    assert parse_strategy_spec_text(BASE).identity != parse_strategy_spec_text(BASE.replace(old, new)).identity


def test_identity_changes_with_risk_cooldown_and_extra_exit_conditions() -> None:
    base = parse_strategy_spec_text(BASE).identity
    assert parse_strategy_spec_text(BASE + "risk: {risk_per_trade_fraction: '0.5'}\n").identity != base
    assert parse_strategy_spec_text(BASE + "cooldown: {post_loss_cooldown_seconds: 60}\n").identity != base
    assert parse_strategy_spec_text(BASE.replace("exit:\n", "exit:\n  max_holding: 4h\n")).identity != base


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (BASE.replace('"0.003"', '"NaN"'), "finite decimal"),
        (BASE.replace('"0.003"', '"Infinity"'), "finite decimal"),
        (BASE.replace('"0.003"', "0.003"), "floats are not exact"),
        (BASE.replace('"0.003"', "true"), "boolean"),
        (BASE.replace('stop_loss_pct: "0.8"', 'stop_loss_pct: "-1"'), "must be >"),
        (BASE.replace('stop_loss_pct: "0.8"', 'stop_loss_pct: "100"'), "must be <"),
        (BASE.replace('stop_loss_pct: "0.8"', 'stop_loss_pct: "150"'), "must be <"),
        (BASE.replace('stop_loss_pct: "0.8"', 'stop_loss_pct: "0"'), "must be >"),
        (BASE.replace('take_profit_pct: "2.0"', 'take_profit_pct: "0"'), "must be >"),
        (BASE.replace('op: ">"', 'op: "="'), "op must be one of"),
        (BASE + "surprise: 1\n", "unknown key"),
        (BASE.replace("direction: long", "direction: long\n  mood: bullish"), "unknown entry key"),
        (BASE.replace("direction: long", "direction: both"), "two specs"),
        (BASE.replace("direction: long", "direction: sideways"), "'long' or 'short'"),
        (BASE.replace("direction: long", "direction: long\n  mode: some"), "'all' or 'any'"),
        (BASE.replace("features: [mom]", "features: [mom, mom]"), "duplicate feature"),
        (BASE.replace("features: [mom]", "features: []"), "non-empty list"),
        (BASE.replace("features: [mom]", "features: [Mom-1]"), "feature names"),
        (BASE.replace("feature: mom,", "feature: other,"), "not listed in features"),
        (BASE.replace("    - {feature", "    - {nothing: 1, feature"), "unknown entry.rules"),
        (BASE.replace('exit:\n  take_profit_pct: "2.0"\n  stop_loss_pct: "0.8"\n', "exit: {}\n"), "at least one of"),
        (BASE.replace('position: {size: "0.01"}', "position: {}"), "position.size is required"),
        (BASE.replace('size: "0.01"', 'size: "0"'), "must be >"),
        (BASE.replace('size: "0.01"', 'size: "-1"'), "must be >"),
        (BASE.replace('lot_size: "0.001", ', ""), "lot_size is required"),
        (BASE.replace('min_size: "0.001"', 'min_size: "0.001", max_size: "0.0001"'), "max_size must be >="),
        (BASE + "risk: {max_drawdown_fraction: '0'}\n", "must be >"),
        (BASE + "risk: {risk_per_trade_fraction: '1.5'}\n", "must be <="),
        (BASE + "risk: {max_evidence_age_seconds: -1}\n", "non-negative integer"),
        (BASE + "session: nyse\n", "session must be one of"),
        (BASE + "cooldown: {max_entries_per_utc_day: 0}\n", ">= 1"),
        (BASE + "cooldown: {post_loss_cooldown_seconds: -5}\n", ">= 0"),
        (BASE.replace("version: 1", "version: 0"), "version must be"),
        (BASE.replace("name: breakout_long", "name: Bad-Name"), "name must match"),
        (BASE.replace("omega.strategy-spec/v1", "other/v2"), "schema must be"),
        (BASE.replace("exit:\n", "exit:\n  max_holding: 5x\n"), "exit.max_holding"),
        ("[1, 2]", "must be a mapping"),
    ],
)
def test_invalid_specs_fail_at_load_with_actionable_messages(text: str, fragment: str) -> None:
    with pytest.raises(StrategySpecError, match=fragment):
        parse_strategy_spec_text(text, source="strategy.yaml")


def test_yaml_duplicate_keys_are_rejected_and_errors_name_the_file() -> None:
    with pytest.raises(StrategySpecError, match=r"strategy\.yaml: invalid YAML.*duplicate key"):
        parse_strategy_spec_text(BASE + "version: 3\n", source="strategy.yaml")


# ---------------------------------------------------------------------------------------------
# Compilation to a platform StrategySpec
# ---------------------------------------------------------------------------------------------


def test_missing_feature_spec_fails_at_compile_time_naming_it() -> None:
    spec = parse_strategy_spec_text(BASE)
    other = _feature(FEATURE_TEXT.replace("name: mom", "name: other"))
    with pytest.raises(StrategySpecError, match=r"needs feature spec\(s\) \['mom'\].*\['other'\]"):
        compile_strategy(spec, [other])


def test_compiled_strategy_is_a_valid_platform_strategy_spec() -> None:
    compiled = _compile()
    platform_spec = compiled.platform_spec
    assert isinstance(platform_spec, qp.StrategySpec)
    assert platform_spec.entry_policy.direction is qp.Direction.LONG
    assert platform_spec.entry_policy.signal_key == "signal.entry"
    assert platform_spec.exit_policy.signal_key == "signal.exit"
    assert platform_spec.position_policy.long_target_position == Decimal("0.01")
    assert platform_spec.strategy_identity.startswith("strategy-spec-v1:sha256:")
    assert compiled.identity == platform_spec.strategy_identity


def test_strategy_identity_is_deterministic() -> None:
    assert _compile().identity == _compile().identity


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('value: "0.003"', 'value: "0.004"'),
        ('take_profit_pct: "2.0"', 'take_profit_pct: "2.5"'),
        ('stop_loss_pct: "0.8"', 'stop_loss_pct: "0.7"'),
        ('size: "0.01"', 'size: "0.02"'),
        ("direction: long", "direction: short"),
        ('op: ">"', 'op: "<"'),
    ],
)
def test_platform_identity_changes_with_the_signal_logic(old: str, new: str) -> None:
    """Thresholds, exits and direction live in the signal layer; they must still change the identity."""
    assert _compile().identity != _compile(BASE.replace(old, new)).identity


def test_platform_identity_changes_when_a_feature_spec_changes() -> None:
    changed = _feature(FEATURE_TEXT.replace("{window: 1}", "{window: 2}"))
    assert _compile().identity != _compile(features=[changed]).identity


def test_unrelated_extra_feature_specs_are_ignored() -> None:
    extra = _feature(FEATURE_TEXT.replace("name: mom", "name: unrelated"))
    assert _compile().identity == _compile(features=[_feature(), extra]).identity


@pytest.mark.parametrize("path", sorted(glob.glob(str(SPECS / "strategies" / "*.yaml"))))
def test_shipped_strategy_specs_compile_against_the_shipped_feature_specs(path: str) -> None:
    features = [load_feature_spec(p) for p in sorted(glob.glob(str(SPECS / "features" / "*.yaml")))]
    definition = load_strategy_spec(path)
    compiled = compile_strategy(definition, features)
    assert Path(path).stem == definition.name
    assert compiled.identity == compile_strategy(definition, features).identity


def test_shipped_strategies_cover_both_directions_and_have_unique_identities() -> None:
    features = [load_feature_spec(p) for p in sorted(glob.glob(str(SPECS / "features" / "*.yaml")))]
    compiled = [
        compile_strategy(load_strategy_spec(p), features) for p in glob.glob(str(SPECS / "strategies" / "*.yaml"))
    ]
    assert len(compiled) >= 4
    assert len({c.identity for c in compiled}) == len(compiled)
    assert {c.definition.entry.direction for c in compiled} == {"long", "short"}


def test_ported_legacy_parameters() -> None:
    breakout = load_strategy_spec(SPECS / "strategies" / "volatility_breakout_long.yaml")
    assert breakout.exit.stop_loss_pct == Decimal("0.8") and breakout.exit.take_profit_pct == Decimal("2.0")
    assert breakout.entry.rules[0].value == Decimal("0.003")
    absorption = load_strategy_spec(SPECS / "strategies" / "orderflow_absorption_short.yaml")
    assert absorption.exit.stop_loss_pct == Decimal("0.6") and absorption.exit.take_profit_pct == Decimal("1.2")
    assert absorption.entry.rules[0].value == Decimal("2")


# ---------------------------------------------------------------------------------------------
# Decisions through the platform's compose_decision (no gateway needed)
# ---------------------------------------------------------------------------------------------

T0 = qp.Instant.parse("2024-01-15T00:10:00Z")


def _signals(entry: bool, exit_: bool) -> list[Any]:
    return [
        qp.StrategyInput(key="signal.entry", value=entry, available_at=T0, provenance="test"),
        qp.StrategyInput(key="signal.exit", value=exit_, available_at=T0, provenance="test"),
    ]


@pytest.mark.parametrize(("direction", "expected"), [("long", qp.Direction.LONG), ("short", qp.Direction.SHORT)])
def test_entry_signal_produces_an_entry_intent_in_the_spec_direction(direction: str, expected: Any) -> None:
    compiled = _compile(BASE.replace("direction: long", f"direction: {direction}"))
    result = qp.compose_decision(compiled.platform_spec, _signals(True, False), decision_time=T0, instrument="BTCUSDT")
    intent = result.decision_intent
    assert intent is not None and intent.direction is expected
    assert abs(intent.target_position) == Decimal("0.01")
    assert intent.strategy_identity == compiled.identity


def test_exit_signal_flattens_and_wins_over_entry() -> None:
    compiled = _compile()
    spec = compiled.platform_spec
    exit_only = qp.compose_decision(spec, _signals(False, True), decision_time=T0, instrument="BTCUSDT")
    both = qp.compose_decision(spec, _signals(True, True), decision_time=T0, instrument="BTCUSDT")
    assert exit_only.decision_intent.direction is qp.Direction.FLAT
    assert both.decision_intent.direction is qp.Direction.FLAT


def test_no_signal_means_no_decision() -> None:
    compiled = _compile()
    result = qp.compose_decision(compiled.platform_spec, _signals(False, False), decision_time=T0, instrument="BTCUSDT")
    assert result.decision_intent is None and result.no_decision is not None


# ---------------------------------------------------------------------------------------------
# Signal layer with a fake ledger (the rules, exits and the no-pyramiding gate)
# ---------------------------------------------------------------------------------------------


@dataclass
class _Side:
    quantity: Decimal = Decimal(0)
    average_basis: Decimal = Decimal(0)
    opened_at: Any = None


@dataclass
class _Position:
    long: _Side
    short: _Side


@dataclass
class _Ledger:
    positions: dict[str, _Position]


@dataclass
class _Context:
    ledger: _Ledger


FLAT = _Context(_Ledger({}))


def _long_position(basis: str = "100", opened_at: str = "2024-01-15T00:00:00Z") -> _Context:
    side = _Side(Decimal(1), Decimal(basis), qp.Instant.parse(opened_at))
    return _Context(_Ledger({"BTCUSDT": _Position(long=side, short=_Side())}))


def _short_position(basis: str = "100", opened_at: str = "2024-01-15T00:00:00Z") -> _Context:
    side = _Side(Decimal(1), Decimal(basis), qp.Instant.parse(opened_at))
    return _Context(_Ledger({"BTCUSDT": _Position(long=_Side(), short=side)}))


def _provider(text: str, value: str | None) -> SignalProvider:
    """A signal provider whose feature `mom` emits `value` at every tick (or nothing when None)."""

    def features(record: Any, context: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        return (
            qp.StrategyInput(
                key="feature.mom",
                value=Decimal(value),
                available_at=record.exchange_ts,
                provenance="stub",
            ),
        )

    return SignalProvider(parse_strategy_spec_text(text), features, provenance="omega-test")


def _tick(price: str, ts: str = "2024-01-15T00:10:00Z") -> Any:
    return qp.TradeRecord("bybit", "BTCUSDT", qp.Instant.parse(ts), price, "1", "buy", trade_id="t", sequence="1")


def _signal_values(provider: SignalProvider, context: Any, price: str = "100", ts: str = "2024-01-15T00:10:00Z"):
    out = {item.key: item for item in provider(_tick(price, ts), context)}
    return out["signal.entry"].value, out["signal.exit"].value


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        (">", "0.0031", True),
        (">", "0.003", False),  # strict
        (">=", "0.003", True),
        (">=", "0.0029", False),
        ("<", "0.0029", True),
        ("<", "0.003", False),  # strict
        ("<=", "0.003", True),
        ("<=", "0.0031", False),
    ],
)
def test_entry_comparators_including_the_boundary(op: str, value: str, expected: bool) -> None:
    text = BASE.replace('op: ">"', f'op: "{op}"')
    assert _signal_values(_provider(text, value), FLAT)[0] is expected


def test_entry_is_false_when_the_feature_is_not_emitted() -> None:
    assert _signal_values(_provider(BASE, None), FLAT) == (False, False)


def test_entry_is_blocked_while_a_position_is_open_no_pyramiding() -> None:
    provider = _provider(BASE, "0.5")  # rule satisfied
    assert _signal_values(provider, FLAT)[0] is True
    assert _signal_values(provider, _long_position())[0] is False
    assert _signal_values(provider, _short_position())[0] is False  # either side blocks a new entry


def test_all_versus_any_for_multiple_rules() -> None:
    two_rules = BASE.replace(
        '    - {feature: mom, op: ">", value: "0.003"}',
        '    - {feature: mom, op: ">", value: "0.003"}\n    - {feature: mom, op: "<", value: "0.001"}',
    )
    assert _signal_values(_provider(two_rules, "0.0005"), FLAT)[0] is False  # all: only the second rule holds
    any_rules = two_rules.replace("direction: long", "direction: long\n  mode: any")
    assert _signal_values(_provider(any_rules, "0.0005"), FLAT)[0] is True  # any: one rule is enough
    assert _signal_values(_provider(any_rules, "0.002"), FLAT)[0] is False  # any: neither holds


@pytest.mark.parametrize(
    ("context", "price", "expected_exit"),
    [
        (_long_position("100"), "102", True),  # +2.0% == TP (>=)
        (_long_position("100"), "101.999", False),
        (_long_position("100"), "99.2", True),  # -0.8% == SL (<=)
        (_long_position("100"), "99.201", False),
        (_short_position("100"), "98", True),  # short +2.0% == TP
        (_short_position("100"), "98.001", False),
        (_short_position("100"), "100.8", True),  # short -0.8% == SL
        (_short_position("100"), "100.799", False),
        (FLAT, "50", False),  # nothing to exit when flat
    ],
)
def test_take_profit_and_stop_loss_boundaries(context: _Context, price: str, expected_exit: bool) -> None:
    direction = "short" if context.ledger.positions and context.ledger.positions["BTCUSDT"].short.quantity else "long"
    text = BASE.replace("direction: long", f"direction: {direction}")
    provider = _provider(text, "0")
    assert _signal_values(provider, context, price)[1] is expected_exit


def test_max_holding_triggers_exactly_at_the_duration() -> None:
    text = BASE.replace('  take_profit_pct: "2.0"\n  stop_loss_pct: "0.8"\n', '  max_holding: 2h\n')
    provider = _provider(text, "0")
    context = _long_position("100", opened_at="2024-01-15T00:00:00Z")
    assert _signal_values(provider, context, "100", "2024-01-15T01:59:59.999999999Z")[1] is False
    assert _signal_values(provider, context, "100", "2024-01-15T02:00:00Z")[1] is True


def test_exit_rules_trigger_when_any_holds_and_missing_feature_is_false() -> None:
    text = BASE.replace(
        '  stop_loss_pct: "0.8"\n',
        '  stop_loss_pct: "0.8"\n  rules:\n    - {feature: mom, op: "<", value: "-0.001"}\n',
    )
    context = _long_position("100")
    assert _signal_values(_provider(text, "-0.002"), context, "100.5")[1] is True
    assert _signal_values(_provider(text, "0.002"), context, "100.5")[1] is False
    assert _signal_values(_provider(text, None), context, "100.5")[1] is False


def test_signals_are_available_at_the_tick_and_carry_the_strategy_provenance() -> None:
    provider = _provider(BASE, "0.5")
    ts = "2024-01-15T00:10:00Z"
    out = provider(_tick("100", ts), FLAT)
    signals = [item for item in out if item.key.startswith("signal.")]
    assert [s.key for s in signals] == ["signal.entry", "signal.exit"]
    assert all(s.available_at == qp.Instant.parse(ts) and s.provenance == "omega-test" for s in signals)
    assert any(item.key == "feature.mom" for item in out)  # features are passed through for traceability
