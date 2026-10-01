"""Causal feature provider: warm-up, available_at, truncation invariance and reference parity."""

from __future__ import annotations

import decimal
import math
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

pytest.importorskip("quant_platform")

from omega import platform_link as qp  # noqa: E402
from omega.features import (  # noqa: E402
    FeatureSpecError,
    compile_feature_provider,
    parse_feature_spec_text,
)
from omega.features.kernels import DECIMAL_CONTEXT, FeatureComputationError  # noqa: E402

BAR_NS = 60_000_000_000  # 1m
START = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)


def _spec_text(name: str, kernel: str, representation: str, params: str) -> str:
    inputs = "{representation: candles, bar: 1m}" if representation == "candles" else "{representation: trades}"
    return (
        f"schema: omega.feature-spec/v1\nname: {name}\nversion: 1\ninputs: {inputs}\n"
        f"kernel: {kernel}\nparams: {params}\n"
    )


SPEC_TEXTS = {
    "mom": _spec_text("mom", "price_momentum", "candles", "{window: 3}"),
    "vol": _spec_text("vol", "realized_vol", "candles", "{window: 4}"),
    "atr": _spec_text("atr", "atr_pct", "candles", "{window: 3}"),
    "imb": _spec_text("imb", "imbalance_ratio", "trades", "{window: 20}"),
    "cvd": _spec_text("cvd", "cvd_zscore", "trades", "{window: 20}"),
}
SPECS = {name: parse_feature_spec_text(text) for name, text in SPEC_TEXTS.items()}


def _stream(n: int, seed: int = 7) -> list[Any]:
    rnd = random.Random(seed)
    price = Decimal("50000.00")
    ts = START
    out = []
    for i in range(n):
        ts = ts + timedelta(milliseconds=rnd.randint(100, 4000))
        price = max(Decimal("1"), price + Decimal(rnd.randint(-250, 250)) / 100)
        out.append(
            qp.TradeRecord(
                venue="bybit",
                instrument="BTCUSDT",
                exchange_ts=qp.Instant.parse(ts),
                price=f"{price:.2f}",
                size=f"{Decimal(rnd.randint(1, 2000)) / 1000:.3f}",
                aggressor_side="buy" if rnd.random() < 0.5 else "sell",
                trade_id=f"t{i}",
                sequence=str(i),
            )
        )
    return out


STREAM = _stream(1200)


def _run(specs: list[Any], trades: list[Any]) -> list[tuple[Any, ...]]:
    provider = compile_feature_provider(specs)
    return [provider(trade, None) for trade in trades]


def _as_tuples(outputs: list[tuple[Any, ...]]) -> list[tuple[tuple[str, Decimal, int], ...]]:
    return [tuple((i.key, i.value, i.available_at.epoch_ns) for i in step) for step in outputs]


def _by_key(step: tuple[Any, ...]) -> dict[str, Any]:
    return {item.key: item for item in step}


@pytest.fixture(scope="module")
def full_run() -> list[tuple[Any, ...]]:
    return _run(list(SPECS.values()), STREAM)


# ---------------------------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------------------------


def test_truncation_invariance_at_60_cut_points(full_run: list[tuple[Any, ...]]) -> None:
    """Removing every trade after t must not change anything emitted up to and including t."""
    full = _as_tuples(full_run)
    cuts = list(range(25, len(STREAM) - 1, 19))
    assert len(cuts) >= 60
    for cut in cuts:
        truncated = _as_tuples(_run(list(SPECS.values()), STREAM[: cut + 1]))
        assert truncated == full[: cut + 1], f"emissions changed when trades after index {cut} were removed"


def test_available_at_never_exceeds_the_triggering_trade(full_run: list[tuple[Any, ...]]) -> None:
    for trade, step in zip(STREAM, full_run, strict=True):
        for item in step:
            assert item.available_at <= trade.exchange_ts


def test_candle_features_are_available_at_a_candle_end_trade_features_at_the_trade(
    full_run: list[tuple[Any, ...]],
) -> None:
    for trade, step in zip(STREAM, full_run, strict=True):
        for key, item in _by_key(step).items():
            if key in {"feature.mom", "feature.vol", "feature.atr"}:
                assert item.available_at.epoch_ns % BAR_NS == 0
                assert item.available_at.epoch_ns <= trade.exchange_ts.epoch_ns
            else:
                assert item.available_at == trade.exchange_ts


def test_provenance_is_the_spec_identity(full_run: list[tuple[Any, ...]]) -> None:
    identities = {spec.key: spec.identity for spec in SPECS.values()}
    for step in full_run:
        for item in step:
            assert item.provenance == identities[item.key]


def test_results_do_not_depend_on_the_process_wide_decimal_context(full_run: list[tuple[Any, ...]]) -> None:
    reference = _as_tuples(full_run)
    with decimal.localcontext() as ctx:
        ctx.prec = 6
        ctx.rounding = decimal.ROUND_DOWN
        again = _as_tuples(_run(list(SPECS.values()), STREAM))
    assert again == reference


def test_two_independent_providers_are_identical(full_run: list[tuple[Any, ...]]) -> None:
    assert _as_tuples(_run(list(SPECS.values()), STREAM)) == _as_tuples(full_run)


# ---------------------------------------------------------------------------------------------
# Warm-up: nothing is emitted (and nothing is filled with 0)
# ---------------------------------------------------------------------------------------------


def _first_emission_index(full_run: list[tuple[Any, ...]], key: str) -> int | None:
    for index, step in enumerate(full_run):
        if key in _by_key(step):
            return index
    return None


@pytest.mark.parametrize("name", ["imb", "cvd"])
def test_trade_kernels_start_exactly_after_their_window(full_run: list[tuple[Any, ...]], name: str) -> None:
    spec = SPECS[name]
    assert _first_emission_index(full_run, spec.key) == spec.warmup - 1  # the 20th trade


@pytest.mark.parametrize("name", ["mom", "vol", "atr"])
def test_candle_kernels_start_at_the_first_trade_after_the_warmup_th_candle_closes(
    full_run: list[tuple[Any, ...]], name: str
) -> None:
    spec = SPECS[name]
    candles = _historical_candles()
    end_of_last_warmup_candle = qp.Instant.parse(candles[spec.warmup - 1].bucket_end)
    expected = next(i for i, t in enumerate(STREAM) if t.exchange_ts >= end_of_last_warmup_candle)
    assert _first_emission_index(full_run, spec.key) == expected


def test_no_emission_value_is_a_filled_zero_during_warmup(full_run: list[tuple[Any, ...]]) -> None:
    early = full_run[:15]  # before any kernel can be warm
    assert all(step == () for step in early)


def test_extra_declared_warmup_delays_the_first_emission() -> None:
    spec = parse_feature_spec_text(
        _spec_text("imb_late", "imbalance_ratio", "trades", "{window: 20}") + "warmup: 100\n"
    )
    outputs = _run([spec], STREAM[:200])
    first = next(i for i, step in enumerate(outputs) if step)
    assert first == 99


def test_zero_dispersion_emits_nothing_instead_of_zero() -> None:
    spec = SPECS["cvd"]
    trades = [
        qp.TradeRecord("bybit", "BTCUSDT", qp.Instant.parse(START + timedelta(seconds=i)), "100", "1", "buy")
        for i in range(60)
    ]
    assert all(step == () for step in _run([spec], trades))  # every delta is +1: std == 0


# ---------------------------------------------------------------------------------------------
# Reference parity (independent computation on the platform's historical D03 candles)
# ---------------------------------------------------------------------------------------------


def _historical_candles() -> list[Any]:
    start = qp.Instant((STREAM[0].exchange_ts.epoch_ns // BAR_NS) * BAR_NS)
    end = qp.Instant((STREAM[-1].exchange_ts.epoch_ns // BAR_NS + 1) * BAR_NS)
    definition = qp.CandleDefinitionV1.from_duration("1m")
    return list(qp.aggregate_historical_candles(STREAM, definition, start, end))


def _closed_before(candles: list[Any], t: Any) -> list[Any]:
    return [c for c in candles if qp.Instant.parse(c.bucket_end) <= t]


def test_candle_features_match_an_independent_computation_on_platform_candles(
    full_run: list[tuple[Any, ...]],
) -> None:
    candles = _historical_candles()
    checked = {"mom": 0, "vol": 0, "atr": 0}
    with decimal.localcontext(DECIMAL_CONTEXT):
        for trade, step in zip(STREAM, full_run, strict=True):
            emitted = _by_key(step)
            closed = _closed_before(candles, trade.exchange_ts)

            if len(closed) >= 4:  # momentum, window 3
                closes = [Decimal(c.close) for c in closed[-4:]]
                assert emitted["feature.mom"].value == closes[-1] / closes[0] - 1
                checked["mom"] += 1
            else:
                assert "feature.mom" not in emitted

            if len(closed) >= 5:  # realized vol, window 4 (float64 cross-check, independent maths)
                closes_f = np.array([float(c.close) for c in closed[-5:]])
                rets = np.diff(np.log(closes_f))
                expected = rets.std(ddof=1) * math.sqrt(365 * 86400 / 60)
                assert float(emitted["feature.vol"].value) == pytest.approx(expected, rel=1e-9)
                checked["vol"] += 1
            else:
                assert "feature.vol" not in emitted

            if len(closed) >= 4:  # atr_pct, window 3
                bars = [(Decimal(c.high), Decimal(c.low), Decimal(c.close)) for c in closed[-4:]]
                trs = [max(h - lo, abs(h - bars[i][2]), abs(lo - bars[i][2])) for i, (h, lo, _) in enumerate(bars[1:])]
                assert emitted["feature.atr"].value == (sum(trs, Decimal(0)) / 3) / bars[-1][2]
                checked["atr"] += 1
            else:
                assert "feature.atr" not in emitted
    assert all(count > 100 for count in checked.values()), checked


def test_trade_features_match_an_independent_computation(full_run: list[tuple[Any, ...]]) -> None:
    window = 20
    for index in range(window - 1, len(STREAM), 7):
        trades = STREAM[index - window + 1 : index + 1]
        signed = [Decimal(t.size) if t.aggressor_side == "buy" else -Decimal(t.size) for t in trades]
        buy = sum((s for s in signed if s > 0), Decimal(0))
        sell = -sum((s for s in signed if s < 0), Decimal(0))
        emitted = _by_key(full_run[index])

        with decimal.localcontext(DECIMAL_CONTEXT):
            expected_imbalance = (buy - sell) / (buy + sell)
        assert emitted["feature.imb"].value == expected_imbalance

        arr = np.array([float(s) for s in signed])
        expected = arr.sum() / arr.std(ddof=1)
        assert float(emitted["feature.cvd"].value) == pytest.approx(expected, rel=1e-9)


def test_imbalance_ratio_is_bounded() -> None:
    values = [i.value for step in _run([SPECS["imb"]], STREAM) for i in step]
    assert values and all(Decimal(-1) <= v <= Decimal(1) for v in values)


# ---------------------------------------------------------------------------------------------
# Provider contract and errors
# ---------------------------------------------------------------------------------------------


def test_candle_features_with_the_same_bar_share_one_platform_builder() -> None:
    provider = compile_feature_provider([SPECS["mom"], SPECS["vol"], SPECS["atr"]])
    assert len(provider._builders) == 1


def test_duplicate_feature_names_and_empty_spec_lists_are_rejected() -> None:
    with pytest.raises(FeatureSpecError, match="duplicate feature name"):
        compile_feature_provider([SPECS["mom"], SPECS["mom"]])
    with pytest.raises(FeatureSpecError, match="at least one"):
        compile_feature_provider([])


def test_non_trade_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="TradeRecord"):
        compile_feature_provider([SPECS["imb"]])({"price": "1"}, None)


def test_unknown_aggressor_side_is_a_clear_error() -> None:
    bad = qp.TradeRecord("bybit", "BTCUSDT", qp.Instant.parse(START), "100", "1", "unknown")
    with pytest.raises(FeatureComputationError, match="aggressor_side"):
        compile_feature_provider([SPECS["imb"]])(bad, None)


def test_out_of_order_trades_are_refused_by_the_platform_candle_runtime() -> None:
    provider = compile_feature_provider([SPECS["mom"]])
    provider(STREAM[10], None)
    with pytest.raises(Exception, match="source order"):
        provider(STREAM[5], None)


def test_emitted_values_are_decimals_accepted_by_the_platform_strategy_input(
    full_run: list[tuple[Any, ...]],
) -> None:
    emitted = [item for step in full_run for item in step]
    assert emitted and all(isinstance(item.value, Decimal) for item in emitted)
    assert all(isinstance(item, qp.StrategyInput) for item in emitted)
