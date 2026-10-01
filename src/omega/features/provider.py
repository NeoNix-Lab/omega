"""Compile feature specs into a platform ``FeatureProvider``.

The platform replay calls ``provider(record, context)`` for every trade, in time order, and expects
the ``StrategyInput`` values that are *available* at that instant. :class:`CompiledFeatureProvider`
turns validated :class:`~omega.features.spec.FeatureSpec` objects into such a provider:

* Candle-based features reuse the platform's ``IncrementalCandleBuilder`` (no Omega aggregation).
  A candle is closed with the watermark "the current trade's time": every bucket whose end is
  ``<=`` that time is complete, because trades arrive in non-decreasing order. The feature's
  ``available_at`` is that candle's end (its causal floor), which is ``<=`` the triggering trade.
* Trade-based features update on every trade; ``available_at`` is the trade time.
* While a feature is warming up (or its value is undefined) **nothing is emitted for it**.
* A candle feature keeps emitting its latest value (with its original ``available_at``) on every
  trade until the next candle closes, so strategy rules can be evaluated at any tick.

A provider instance is stateful and must be used for exactly one replay, in time order.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, localcontext
from typing import Any

from .. import platform_link as qp
from .kernels import DECIMAL_CONTEXT, ClosedCandle, KernelState, get_kernel
from .spec import FeatureSpec, FeatureSpecError


class _Feature:
    """Runtime state of one compiled feature."""

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec
        definition = get_kernel(spec.kernel)
        self.state: KernelState = definition.factory(spec.params_dict, spec.bar_ns)
        self.consumed = 0  # units (trades or closed candles) fed to the kernel so far
        self.value: Decimal | None = None
        self.available_at: Any = None

    def feed(self, *args: Any) -> Decimal | None:
        self.consumed += 1
        value = self.state.update(*args)
        return value if self.consumed >= self.spec.warmup else None


def _to_closed_candle(update: Any) -> ClosedCandle:
    record = update.record
    return ClosedCandle(
        start_ns=update.bucket_start.epoch_ns,
        end_ns=update.bucket_end.epoch_ns,
        open=Decimal(record.open),
        high=Decimal(record.high),
        low=Decimal(record.low),
        close=Decimal(record.close),
        volume=Decimal(record.volume),
    )


class CompiledFeatureProvider:
    """Stateful, incremental ``FeatureProvider`` built from feature specs."""

    def __init__(self, specs: Sequence[FeatureSpec]) -> None:
        if not specs:
            raise FeatureSpecError("at least one feature spec is required")
        names = [spec.name for spec in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise FeatureSpecError(f"duplicate feature name(s): {duplicates}")
        self._features = [_Feature(spec) for spec in specs]
        self._trade_features = [f for f in self._features if f.spec.representation == "trades"]
        # One platform incremental builder per distinct bar duration, shared by the features using it.
        self._builders: dict[int, Any] = {}
        self._candle_features: dict[int, list[_Feature]] = {}
        for feature in self._features:
            if feature.spec.representation == "candles":
                bar_ns = int(feature.spec.bar_ns)  # type: ignore[arg-type]
                if bar_ns not in self._builders:
                    self._builders[bar_ns] = qp.IncrementalCandleBuilder(qp.CandleDefinitionV1.from_duration(bar_ns))
                self._candle_features.setdefault(bar_ns, []).append(feature)

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        return tuple(feature.spec for feature in self._features)

    def __call__(self, record: Any, context: Any = None) -> tuple[Any, ...]:
        if not isinstance(record, qp.TradeRecord):
            raise TypeError("feature providers consume platform TradeRecord values")
        with localcontext(DECIMAL_CONTEXT):
            now = qp.Instant.parse(record.exchange_ts)

            for bar_ns, builder in self._builders.items():
                builder.consume(record)
                for update in builder.close_through(now):
                    candle = _to_closed_candle(update)
                    for feature in self._candle_features[bar_ns]:
                        value = feature.feed(candle)
                        feature.value = value
                        feature.available_at = update.bucket_end if value is not None else None

            if self._trade_features:
                price, size = Decimal(record.price), Decimal(record.size)
                for feature in self._trade_features:
                    value = feature.feed(price, size, record.aggressor_side)
                    feature.value = value
                    feature.available_at = now if value is not None else None

            emitted = []
            for feature in self._features:
                if feature.value is None:
                    continue
                emitted.append(
                    qp.StrategyInput(
                        key=feature.spec.key,
                        value=feature.value,
                        available_at=feature.available_at,
                        provenance=feature.spec.identity,
                    )
                )
            return tuple(emitted)


def compile_feature_provider(specs: Sequence[FeatureSpec]) -> CompiledFeatureProvider:
    """Compile validated feature specs into a fresh, single-use platform ``FeatureProvider``."""
    return CompiledFeatureProvider(specs)
