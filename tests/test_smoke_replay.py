"""scripts/smoke_replay.py on a handcrafted tape (synthetic: a test fixture, not evidence).

The tape and the expected outcome are the ones of the long-take-profit scenario in
test_strategy_replay.py: entry 1 @ 103, exit 1 @ 105.5, zero fees -> equity 10002.5; with a 0.001 taker fee
the fees are 0.103 + 0.1055.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("quant_platform")

from omega import platform_link as qp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_module_spec = importlib.util.spec_from_file_location("smoke_replay", ROOT / "scripts" / "smoke_replay.py")
assert _module_spec is not None and _module_spec.loader is not None
smoke = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(smoke)

START = datetime(2024, 1, 15, tzinfo=UTC)
START_TEXT, END_TEXT = "2024-01-15T00:00:00Z", "2024-01-15T00:10:00Z"

FEATURE = """\
schema: omega.feature-spec/v1
name: mom
version: 1
inputs: {representation: candles, bar: 1m}
kernel: price_momentum
params: {window: 1}
"""

STRATEGY = """\
schema: omega.strategy-spec/v1
name: t_long
version: 1
features: [mom]
entry:
  direction: long
  rules:
    - {feature: mom, op: ">", value: "0.015"}
exit:
  take_profit_pct: "2"
  stop_loss_pct: "1"
position: {size: "1"}
sizing: {lot_size: "0.001", min_size: "0.001"}
"""

PRICES = [100, 100, 101, 103, 103, Decimal("105.5")]


class _Scan:
    completed_metadata = None

    def __init__(self, batch: tuple[Any, ...]) -> None:
        self._batch = batch

    def __iter__(self):
        yield self._batch


class _Gateway:
    def __init__(self, records: list[Any]) -> None:
        self._records = tuple(records)

    def scan(self, request: Any, *, batch_size: int = 65536) -> _Scan:
        return _Scan(self._records)


def _tape(prices: list[Any]) -> list[Any]:
    trades = []
    for minute, price in enumerate(prices):
        for k in range(6):
            index = len(trades)
            trades.append(
                qp.TradeRecord(
                    "bybit",
                    "BTCUSDT",
                    qp.Instant.parse(START + timedelta(minutes=minute, seconds=10 * k)),
                    str(price),
                    "1",
                    "buy" if index % 2 else "sell",
                    trade_id=f"t{index}",
                    sequence=str(index),
                )
            )
    return trades


@pytest.fixture
def specs(tmp_path: Path) -> dict[str, Any]:
    feature, strategy = tmp_path / "mom.yaml", tmp_path / "long.yaml"
    feature.write_text(FEATURE, encoding="utf-8")
    strategy.write_text(STRATEGY, encoding="utf-8")
    return {"features": [feature], "strategy": strategy, "start": START_TEXT, "end": END_TEXT}


def _run(specs: dict[str, Any], prices: list[Any] = PRICES, **overrides: Any) -> dict[str, Any]:
    options = {"maker_fee": "0", "taker_fee": "0", **overrides}
    return smoke.run_smoke(
        _Gateway(_tape(prices)), ordering_policy="bybit-trade-key-v1", **specs, **options
    )


def test_hand_computed_outcome(specs: dict[str, Any]) -> None:
    report = _run(specs)
    assert report["fills"] == 2 and report["orders"] == 2
    assert Decimal(report["final_equity"]) == Decimal("10002.5")
    assert Decimal(report["return_pct"]) == Decimal("0.025")
    assert Decimal(report["fees"]) == 0
    assert report["ticks"] == len(PRICES) * 6


def test_deterministic_identities(specs: dict[str, Any]) -> None:
    first, second = _run(specs), _run(specs)
    for key in ("spec_identity", "result_identity", "trace_fingerprint", "strategy_identity"):
        assert first[key] == second[key]


def test_fees_and_slippage_are_parameters_and_change_the_identity(specs: dict[str, Any]) -> None:
    free = _run(specs)
    paid = _run(specs, taker_fee="0.001")
    assert Decimal(paid["fees"]) == Decimal("0.2085")  # 103 * 0.001 + 105.5 * 0.001
    assert Decimal(paid["final_equity"]) == Decimal("10002.5") - Decimal("0.2085")
    assert paid["spec_identity"] != free["spec_identity"]
    slipped = _run(specs, slippage_bps="5")
    assert slipped["spec_identity"] != free["spec_identity"]


def test_no_records_is_an_error_not_an_empty_report(specs: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="no records"):
        smoke.run_smoke(_Gateway([]), ordering_policy="bybit-trade-key-v1", **specs)


def test_missing_dsn_is_a_one_line_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("OMEGA_CATALOG_DSN", raising=False)
    code = smoke.main(["--start", START_TEXT, "--end", END_TEXT, "--features", "a.yaml", "--strategy", "b.yaml"])
    assert code == 2
    assert "no catalog DSN" in capsys.readouterr().err
