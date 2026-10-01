# Strategy specs (`omega.strategy-spec/v1`)

A strategy spec declares *when to enter*, *when to exit* and the sizing/risk parameters, on top of
[feature specs](FEATURE_SPECS.md). Omega compiles it into (a) a platform `StrategySpec` and (b) a
**signal provider** that the platform's deterministic replay calls on every trade.

```yaml
schema: omega.strategy-spec/v1
name: breakout_long
version: 1
features: [momentum_1m_5]            # feature spec names this strategy uses
entry:
  direction: long                    # long | short  (one direction per spec, see "Limits")
  mode: all                          # all | any (default all)
  rules:
    - {feature: momentum_1m_5, op: ">", value: "0.003"}    # ops: >  <  >=  <=
exit:                                # at least one of the four conditions is required
  take_profit_pct: "2.0"             # percent of the entry price (2.0 = 2%)
  stop_loss_pct: "0.8"               # 0 < value < 100
  max_holding: 4h                    # optional; duration like 30m, 4h, 1d
  rules: []                          # optional; exit when ANY rule holds
position: {size: "0.01"}             # target position size (instrument units)
sizing: {lot_size: "0.001", min_size: "0.001"}    # max_size optional
risk: {}                             # optional, defaults below
session: always_open                 # the only supported session (24/7 crypto)
cooldown: {}                         # optional, defaults below
```

Numbers must be quoted strings or integers (YAML floats are rejected: they are not exact).

## What the compiled strategy does

* **Entry** fires when the entry rules hold on the features emitted at that tick **and no position is
  open**. This gate is essential: the platform opens a new full-size order on *every* entry intent, so an
  entry signal that stays true would add to the position at each tick.
* **Exit** fires when a position is open and any of take-profit, stop-loss, `max_holding` or an exit rule
  holds. If entry and exit are both true the exit wins (platform composition rule).
* A rule whose feature is **not emitted** at that tick (warm-up, undefined value) is **false**: the
  strategy never acts on a missing value. A feature listed in `features` that has no feature spec is a
  compile-time error (no silent fallback).
* Take-profit and stop-loss are checked on **every trade tick** against the position's average entry price
  and fill at that tick's price (plus slippage, if configured). They are *not* resting stop orders: no gap,
  queue or intra-tick modelling. Orders are market orders (platform v1), filled as taker.

## Defaults (filled in before the identity is computed)

* `risk`: every fraction is `1` (nothing constrained) and `max_evidence_age_seconds: 3600`. Set
  `max_drawdown_fraction`, `max_position_notional_fraction`, `max_total_exposure_fraction`,
  `risk_per_trade_fraction` explicitly to study the effect of risk limits. The order size is
  `min(position.size, risk budget / price)` rounded down to `lot_size`.
* `cooldown`: disabled (`post_loss_cooldown_seconds: 0`, `consecutive_loss_count: 1000000`,
  `consecutive_loss_cooldown_seconds: 0`, `max_entries_per_utc_day: 1000000`,
  `max_trade_history_age_seconds: 3600`).

## Identity

`omega-strategy-spec-v1:sha256:…` identifies the spec (defaults filled, numbers canonical: `2` = `2.0`).
The compiled platform `strategy_identity` additionally covers the **identities of the feature specs** it
uses, so changing a feature window changes the strategy. Thresholds, exits and direction are not part of
the platform's own policy objects, so they are carried in the `execution_policy` slot (an identity-backed
policy in platform v1); changing any of them changes the platform identity and therefore the replay spec
identity.

## Using it

```python
from omega.features import load_feature_spec
from omega.strategies import compile_strategy, load_strategy_spec

features = [load_feature_spec(p) for p in feature_paths]
compiled = compile_strategy(load_strategy_spec("studies/specs/strategies/volatility_breakout_long.yaml"), features)

# ReplaySpec(strategy=compiled.platform_spec, ...)  and one fresh provider per replay:
result = HistoricalReplayRuntime(gateway, compiled.make_provider()).run(replay_spec)
```

(`omega replay` wraps this in a later step of the epic.)

## Limits

* **One entry direction per spec.** The platform strategy model has a single entry direction per
  `StrategySpec`, so `direction: both` is rejected with a message. Write two specs (a long and a short) and
  replay them separately; the shipped examples come in `_long` / `_short` pairs. There is no
  reverse-in-one-step (long → short) either.
* **One position at a time**, market orders, taker fills, fixed-bps slippage only (platform v1).
* The shipped `orderflow_absorption_*` specs are an **analogue**, not an exact port, of the legacy strategy:
  the legacy z-score was over 20 one-minute bars of delta, the `cvd_zscore` kernel works over the last N
  trades. The `volatility_breakout_*` specs port what the legacy code actually did (momentum only; the
  legacy docstring mentioned a volatility gate that the code never applied).
