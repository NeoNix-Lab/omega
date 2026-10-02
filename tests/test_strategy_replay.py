"""Compiled strategies inside the platform's real HistoricalReplayRuntime, with hand-computed outcomes.

Each scenario is a handcrafted tape: one price per minute, six trades per minute (every 10 s). Features
use 1-minute candles and `price_momentum` with window 1, so momentum at minute m is
``close(m-2) -> close(m-1)``, known at the first trade of minute m (when candle m-1 closes). Fees are zero
unless a test says otherwise, so equity changes are exact.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

pytest.importorskip("quant_platform")

from omega import platform_link as qp  # noqa: E402
from omega.features import parse_feature_spec_text  # noqa: E402
from omega.strategies import compile_strategy, parse_strategy_spec_text  # noqa: E402

START = datetime(2024, 1, 15, tzinfo=UTC)
DATASET = qp.DatasetIdentity("canonical", "trades", "bybit", "BTCUSDT", "trade-v1")

FEATURE = """\
schema: omega.feature-spec/v1
name: mom
version: 1
inputs: {representation: candles, bar: 1m}
kernel: price_momentum
params: {window: 1}
"""


DEFAULT_EXIT = '  take_profit_pct: "2"\n  stop_loss_pct: "1"\n'


def _strategy(direction: str, op: str, threshold: str, exit_lines: str = DEFAULT_EXIT) -> str:
    return (
        "schema: omega.strategy-spec/v1\n"
        f"name: t_{direction}\nversion: 1\nfeatures: [mom]\n"
        f'entry:\n  direction: {direction}\n  rules:\n    - {{feature: mom, op: "{op}", value: "{threshold}"}}\n'
        f"exit:\n{exit_lines}"
        'position: {size: "1"}\nsizing: {lot_size: "0.001", min_size: "0.001"}\n'
    )


def _tape(prices: list[Any], per_minute: int = 6) -> list[Any]:
    trades = []
    index = 0
    for minute, price in enumerate(prices):
        for k in range(per_minute):
            ts = START + timedelta(minutes=minute, seconds=10 * k)
            trades.append(
                qp.TradeRecord(
                    "bybit",
                    "BTCUSDT",
                    qp.Instant.parse(ts),
                    str(price),
                    "1",
                    "buy" if index % 2 else "sell",
                    trade_id=f"t{index}",
                    sequence=str(index),
                )
            )
            index += 1
    return trades


class _Scan:
    completed_metadata = None

    def __init__(self, batch: tuple[Any, ...]) -> None:
        self._batch = batch

    def __iter__(self):
        yield self._batch


class _Gateway:
    """In-memory stand-in for DataGateway: replay only needs `scan()`."""

    def __init__(self, records: list[Any]) -> None:
        self._records = tuple(records)

    def scan(self, request: Any, *, batch_size: int = 65536) -> _Scan:
        return _Scan(self._records)


def _replay(strategy_text: str, prices: list[Any], taker_fee: str = "0") -> tuple[Any, Any]:
    compiled = compile_strategy(parse_strategy_spec_text(strategy_text), [parse_feature_spec_text(FEATURE)])
    tape = _tape(prices)
    spec = qp.ReplaySpec(
        dataset=DATASET,
        start=START,
        end=qp.Instant(tape[-1].exchange_ts.epoch_ns + 1_000_000_000),
        strategy=compiled.platform_spec,
        initial_capital="10000",
        fee_schedule=qp.FeeSchedule(policy_key="test_fees", maker_fee_rate="0", taker_fee_rate=taker_fee),
        ordering_policy="bybit-trade-key-v1",
        batch_size=1000,
    )
    result = qp.HistoricalReplayRuntime(_Gateway(tape), compiled.make_provider()).run(spec)
    return result, compiled


def _intents(result: Any) -> list[dict[str, Any]]:
    return [d for d in result.decisions if d.get("decision_intent")]


def _final_equity(result: Any) -> Decimal:
    return Decimal(result.equity_curve[-1].equity)


# momentum at minute 4 is 103/101 - 1 = +1.98% (> 1.5%): the long entry fires at the first trade of minute 4
RISING = [100, 100, 101, 103, 103]


def test_long_take_profit_exact_outcome() -> None:
    result, _ = _replay(_strategy("long", ">", "0.015"), [*RISING, Decimal("105.5")])
    entry, exit_ = result.fills
    assert (entry.price, entry.quantity) == (Decimal(103), Decimal(1))
    assert (exit_.price, exit_.quantity) == (Decimal("105.5"), Decimal(1))  # +2.43% >= TP 2% at minute 5
    assert _final_equity(result) == Decimal("10002.5")  # 1 unit * (105.5 - 103), zero fees


def test_long_stop_loss_exact_outcome() -> None:
    result, _ = _replay(_strategy("long", ">", "0.015"), [*RISING, Decimal("101.9")])
    entry, exit_ = result.fills
    assert (entry.price, exit_.price) == (Decimal(103), Decimal("101.9"))  # -1.07% <= SL 1%
    assert _final_equity(result) == Decimal("9998.9")


def test_short_take_profit_exact_outcome() -> None:
    result, _ = _replay(_strategy("short", "<", "-0.015"), [100, 100, 99, 97, 97, 94])
    entry, exit_ = result.fills
    assert (entry.price, exit_.price) == (Decimal(97), Decimal(94))  # +3.09% >= TP 2%
    assert _final_equity(result) == Decimal("10003")


def test_short_stop_loss_exact_outcome() -> None:
    result, _ = _replay(_strategy("short", "<", "-0.015"), [100, 100, 99, 97, 97, Decimal("98.1")])
    entry, exit_ = result.fills
    assert (entry.price, exit_.price) == (Decimal(97), Decimal("98.1"))  # -1.13% <= SL 1%
    assert _final_equity(result) == Decimal("9998.9")


def test_no_pyramiding_while_the_entry_condition_stays_true() -> None:
    """Momentum is +1.98% for all six ticks of minute 4, but only the first one may open a position."""
    result, _ = _replay(_strategy("long", ">", "0.015"), [*RISING, Decimal("105.5")])
    assert len(result.fills) == 2
    assert len(_intents(result)) == 2  # one entry intent + one exit intent, not six entries


def test_threshold_decides_whether_the_strategy_trades() -> None:
    above, _ = _replay(_strategy("long", ">", "0.02"), [*RISING, 103])  # 1.98% < 2% -> never enters
    assert above.fills == ()
    below, _ = _replay(_strategy("long", ">", "0.019"), [*RISING, 103])  # 1.98% > 1.9% -> enters
    assert len(below.fills) == 1


def test_flat_market_never_trades() -> None:
    result, _ = _replay(_strategy("long", ">", "0.001"), [100] * 8)
    assert result.fills == () and _final_equity(result) == Decimal(10000)


def test_max_holding_exit_happens_exactly_at_the_duration() -> None:
    exit_lines = '  max_holding: 2m\n'
    result, _ = _replay(_strategy("long", ">", "0.015", exit_lines), [*RISING, 103, 104])
    entry, exit_ = result.fills
    assert entry.fill_time == qp.Instant.parse(START + timedelta(minutes=4))
    assert exit_.fill_time == qp.Instant.parse(START + timedelta(minutes=6))  # opened_at + 2m, first tick
    assert exit_.price == Decimal(104)
    assert _final_equity(result) == Decimal(10001)


def test_fees_are_applied_to_both_fills() -> None:
    result, _ = _replay(_strategy("long", ">", "0.015"), [*RISING, Decimal("105.5")], taker_fee="0.001")
    entry, exit_ = result.fills
    assert entry.fee == Decimal("0.103") and exit_.fee == Decimal("0.1055")
    assert _final_equity(result) == Decimal("10002.5") - Decimal("0.103") - Decimal("0.1055")


def test_replay_is_deterministic() -> None:
    first, compiled = _replay(_strategy("long", ">", "0.015"), [*RISING, Decimal("105.5")])
    second, _ = _replay(_strategy("long", ">", "0.015"), [*RISING, Decimal("105.5")])
    assert first.identity == second.identity
    assert first.trace_fingerprint == second.trace_fingerprint
    assert first.spec_identity == second.spec_identity
    assert compiled.identity in str(first.decisions)  # decisions are traced to the strategy identity


def test_a_different_threshold_gives_a_different_replay_identity() -> None:
    a, _ = _replay(_strategy("long", ">", "0.015"), [*RISING, 103])
    b, _ = _replay(_strategy("long", ">", "0.016"), [*RISING, 103])
    assert a.spec_identity != b.spec_identity  # the threshold is part of the replay spec identity


# ---------------------------------------------------------------------------------------------
# Shipped strategies, including trade-based kernels, through the real replay (determinism/smoke)
# ---------------------------------------------------------------------------------------------


def _random_tape(n: int, seed: int = 11) -> list[Any]:
    import random

    rnd = random.Random(seed)
    price = Decimal("50000")
    ts = START
    out = []
    for i in range(n):
        ts += timedelta(milliseconds=rnd.randint(200, 3000))
        price = max(Decimal(1), price + Decimal(rnd.randint(-300, 300)) / 10)
        out.append(
            qp.TradeRecord(
                "bybit",
                "BTCUSDT",
                qp.Instant.parse(ts),
                f"{price:.1f}",
                f"{Decimal(rnd.randint(1, 3000)) / 1000:.3f}",
                "buy" if rnd.random() < 0.5 else "sell",
                trade_id=f"t{i}",
                sequence=str(i),
            )
        )
    return out


SHIPPED = [
    "orderflow_absorption_long",
    "orderflow_absorption_short",
    "volatility_breakout_long",
    "volatility_breakout_short",
]


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_strategies_replay_deterministically_on_a_random_tape(name: str) -> None:
    import glob
    from pathlib import Path

    from omega.features import load_feature_spec
    from omega.strategies import load_strategy_spec

    specs = Path(__file__).resolve().parent.parent / "studies" / "specs"
    features = [load_feature_spec(p) for p in sorted(glob.glob(str(specs / "features" / "*.yaml")))]
    tape = _random_tape(1500)

    def run() -> Any:
        compiled = compile_strategy(load_strategy_spec(specs / "strategies" / f"{name}.yaml"), features)
        spec = qp.ReplaySpec(
            dataset=DATASET,
            start=START,
            end=qp.Instant(tape[-1].exchange_ts.epoch_ns + 1_000_000_000),
            strategy=compiled.platform_spec,
            initial_capital="10000",
            fee_schedule=qp.FeeSchedule(policy_key="test_fees"),
            ordering_policy="bybit-trade-key-v1",
            batch_size=1000,
        )
        return qp.HistoricalReplayRuntime(_Gateway(tape), compiled.make_provider()).run(spec)

    first, second = run(), run()
    assert first.identity == second.identity and first.trace_fingerprint == second.trace_fingerprint
    assert len(first.equity_curve) == len(tape)
    if name.startswith("orderflow"):
        # a 500-trade cvd z-score on random order flow often exceeds +-2, so the strategy does trade
        assert first.fills, "expected at least one fill from the trade-based entry rule"
