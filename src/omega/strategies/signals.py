"""Signal layer: turns feature values and position state into the platform's boolean signals.

The platform strategy model composes *boolean signals* (``signal.entry`` / ``signal.exit``); it has no
notion of thresholds, stop-loss or take-profit. That logic lives here, in Omega:

* **Entry** is true when the entry rules hold on the features emitted at this tick **and no position is
  open**. The gate matters: the platform opens a new, full-size order on *every* entry intent, so an entry
  signal that stays true while in a position would pyramid at each tick.
* **Exit** is true when a position is open and any of: take-profit, stop-loss, maximum holding time, or an
  exit rule is hit. Take-profit/stop-loss are evaluated on every trade tick against the position's average
  entry price and fill at that tick's price (they are *not* resting stop orders: no gap/queue modelling).
* A rule whose feature is not emitted at this tick (warm-up, undefined value) is **false**; the strategy
  never acts on a missing value.

Both signals are evaluated at the tick, so their ``available_at`` is the tick time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, localcontext
from functools import cache
from typing import Any

from .. import platform_link as qp
from ..features.kernels import DECIMAL_CONTEXT
from ..features.spec import KEY_PREFIX
from ..specio import identity_hash
from .spec import Rule, StrategyDef

ENTRY_SIGNAL_KEY = "signal.entry"
EXIT_SIGNAL_KEY = "signal.exit"
SIGNAL_LAYER_IDENTITY_DOMAIN = "omega-signal-layer-v1"

_COMPARE: dict[str, Callable[[Decimal, Decimal], bool]] = {
    ">": lambda value, threshold: value > threshold,
    "<": lambda value, threshold: value < threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<=": lambda value, threshold: value <= threshold,
}


class SignalLayerPolicy:
    """Identity-backed carrier of the Omega signal logic inside the platform ``StrategySpec``.

    The platform ``StrategySpec`` has no slot for signal rules, thresholds or stop/take-profit, but they
    change what the strategy does, so they must be part of its identity. This object fills the
    ``execution_policy`` slot (a generic identity-backed policy in platform v1) with the full Omega spec
    and the identities of the feature specs it consumes. Tracked as an upstream ask (U11, issue #15).
    """

    def __init__(self, strategy: StrategyDef, feature_identities: Mapping[str, str]) -> None:
        self._payload = {
            "policy_type": "omega-signal-layer-v1",
            "strategy": strategy.stable_dict(),
            "feature_identities": {name: feature_identities[name] for name in sorted(feature_identities)},
        }
        self._identity = f"{SIGNAL_LAYER_IDENTITY_DOMAIN}:sha256:{identity_hash(self._payload)}"

    @property
    def identity(self) -> str:
        return self._identity

    def stable_dict(self) -> dict[str, Any]:
        return dict(self._payload)


@cache
def _always_open_session_class() -> type:
    class AlwaysOpenSessionPolicy(qp.SessionPolicyDefinition):  # type: ignore[misc]
        """24/7 session for continuous crypto markets: every instant is OPEN (no calendar filtering)."""

        def evaluate(self, as_of: Any) -> tuple[Any, ...]:
            instant = qp.Instant.parse(as_of)
            return tuple(
                qp.SessionDecision(
                    policy_identity=self.identity,
                    as_of=instant,
                    reference_market=market,
                    state=qp.SessionState.OPEN,
                    phase="continuous_24_7",
                    trading_date=instant.isoformat()[:10],
                    reason="always_open",
                )
                for market in self.reference_markets
            )

    return AlwaysOpenSessionPolicy


def make_always_open_session(policy_key: str) -> Any:
    return _always_open_session_class()(policy_key=policy_key, reference_markets=(qp.SessionReferenceMarket.NEW_YORK,))


def _evaluate_rules(rules: Sequence[Rule], mode: str, features: Mapping[str, Any]) -> bool:
    results = []
    for rule in rules:
        item = features.get(f"{KEY_PREFIX}{rule.feature}")
        results.append(item is not None and _COMPARE[rule.op](Decimal(item.value), rule.value))
    return all(results) if mode == "all" else any(results)


def _open_side(ledger: Any, instrument: str, direction: str) -> tuple[Any | None, bool]:
    """(the open position side in the strategy's direction or None, whether *any* side is open)."""
    position = ledger.positions.get(instrument)
    if position is None:
        return None, False
    own = position.long if direction == "long" else position.short
    other = position.short if direction == "long" else position.long
    own_open = own.quantity > 0
    return (own if own_open else None), (own_open or other.quantity > 0)


class SignalProvider:
    """Platform ``FeatureProvider`` that emits the features plus ``signal.entry`` / ``signal.exit``.

    Stateless apart from the wrapped feature provider; one instance per replay.
    """

    def __init__(self, strategy: StrategyDef, feature_provider: Callable[[Any, Any], Sequence[Any]], provenance: str):
        self._strategy = strategy
        self._features = feature_provider
        self._provenance = provenance

    def __call__(self, record: Any, context: Any) -> tuple[Any, ...]:
        strategy = self._strategy
        with localcontext(DECIMAL_CONTEXT):
            feature_inputs = tuple(self._features(record, context))
            by_key = {item.key: item for item in feature_inputs}
            now = qp.Instant.parse(record.exchange_ts)
            price = Decimal(record.price)

            side, any_open = _open_side(context.ledger, record.instrument, strategy.entry.direction)
            entry = (not any_open) and _evaluate_rules(strategy.entry.rules, strategy.entry.mode, by_key)
            exit_ = side is not None and self._exit_triggered(side, price, now, by_key)

            signals = tuple(
                qp.StrategyInput(key=key, value=bool(value), available_at=now, provenance=self._provenance)
                for key, value in ((ENTRY_SIGNAL_KEY, entry), (EXIT_SIGNAL_KEY, exit_))
            )
            return (*feature_inputs, *signals)

    def _exit_triggered(self, side: Any, price: Decimal, now: Any, by_key: Mapping[str, Any]) -> bool:
        spec = self._strategy.exit
        basis = side.average_basis
        if self._strategy.entry.direction == "long":
            pnl_pct = (price - basis) / basis * 100
        else:
            pnl_pct = (basis - price) / basis * 100
        if spec.take_profit_pct is not None and pnl_pct >= spec.take_profit_pct:
            return True
        if spec.stop_loss_pct is not None and pnl_pct <= -spec.stop_loss_pct:
            return True
        if spec.max_holding_ns is not None and now.epoch_ns - side.opened_at.epoch_ns >= spec.max_holding_ns:
            return True
        return bool(spec.rules) and _evaluate_rules(spec.rules, "any", by_key)
