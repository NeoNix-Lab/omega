# Feature specs (`omega.feature-spec/v1`)

A feature spec declares *which* feature to compute and *how*, without writing code. Omega compiles
specs into a platform `FeatureProvider` that the deterministic replay calls on every trade.

```yaml
schema: omega.feature-spec/v1
name: momentum_5m_12          # lower_snake_case; emitted as StrategyInput key "feature.momentum_5m_12"
version: 1                    # bump when you change the meaning of the feature
inputs:
  representation: candles     # trades | candles
  bar: 5m                     # candle duration (candles only): 1m, 5m, 300s, 300000000000 ...
kernel: price_momentum        # one of the registered kernels (below)
params:
  window: 12
# warmup: 50                  # optional: extra warm-up, must be >= the kernel minimum
```

Strategy specs (a later step of the epic) refer to the feature through its key `feature.<name>`.

## Built-in kernels

| kernel | input | what it computes | min warm-up |
|---|---|---|---|
| `price_momentum` | candles | `close_now / close_{window candles ago} - 1` | `window + 1` candles |
| `realized_vol` | candles | annualised sample std of `window` close-to-close log returns (24/7: `days_per_year` * 86400 s / bar) | `window + 1` candles |
| `atr_pct` | candles | mean true range over `window` candles / latest close | `window + 1` candles |
| `imbalance_ratio` | trades | `(buy - sell volume) / (buy + sell volume)` over the last `window` trades, in `[-1, 1]` | `window` trades |
| `cvd_zscore` | trades | rolling signed volume over `window` trades / sample std of the signed volumes | `window` trades |

Ready-made examples live in `studies/specs/features/`.

## Guarantees

* **Causal.** A candle feature is computed only from candles that are *closed* (every bucket whose end is
  `<=` the current trade time). Its `available_at` is the candle end, never later than the triggering
  trade. A trade feature has `available_at` = the trade time. Tested by truncation invariance: removing
  every trade after time *t* never changes anything emitted up to *t*.
* **No fake values.** During warm-up, or when a value is undefined (e.g. zero dispersion), the feature is
  simply **not emitted**. Nothing is ever filled with `0`.
* **Exact and deterministic.** Values are `Decimal` computed under a fixed context (50 digits, round
  half even), independent of the process-wide decimal context. Candle aggregation is the platform's own
  `IncrementalCandleBuilder`, not an Omega reimplementation.
* **Stable identity.** `identity = omega-feature-spec-v1:sha256:<hash of the canonical spec>`. Defaults are
  filled in and durations normalised to nanoseconds, so `5m`, `300s` and `300000000000` are the same
  feature; any semantic change (parameter, bar, version, name) gives a new identity. The identity is the
  `provenance` of every emitted `StrategyInput`.
* **Strict.** Unknown keys, unknown kernels, wrong parameter types or ranges, floats where exact decimals
  are required, and duplicate YAML keys are rejected when the spec is loaded, with the file name in the
  message.

## Things to know

* A candle feature keeps emitting its *latest* value (with its original `available_at`) on every trade
  until the next candle closes, so strategy rules can be evaluated at any tick. After a long quiet period
  the value can therefore be old; its `available_at` says how old.
* Candles with no trades are omitted by the platform, so "N candles back" counts non-empty candles.
* Providers are stateful: use one provider instance per replay.

## Adding a kernel

```python
from decimal import Decimal
from omega.features import KernelDef, register_kernel
from omega.features.kernels import ParamDef

class LastPriceScaled:                      # trade kernel: update(price, size, side) -> Decimal | None
    def __init__(self, scale: int) -> None:
        self._scale = scale
    def update(self, price: Decimal, size: Decimal, side: str) -> Decimal | None:
        return price * self._scale

register_kernel(KernelDef(
    kernel_id="scaled_price",
    representation="trades",                # or "candles": update(ClosedCandle) -> Decimal | None
    description="price * scale",
    params=(ParamDef("scale", "int", 1, 1, 1000),),
    min_warmup=lambda p: 1,
    factory=lambda p, bar_ns: LastPriceScaled(p["scale"]),
))
```

A kernel must only use the data passed to `update` (the past), return `None` instead of a placeholder when
it has no value, and use `Decimal` arithmetic. Add tests: truncation invariance and a warm-up check.
