"""Feature kernels and the kernel registry.

A *kernel* is a small stateful computation that a feature spec selects by id. Kernels are the only
place where Omega owns feature maths; everything else (data access, candle aggregation, replay,
accounting) stays in the platform.

Rules every kernel follows:

* **Causal and incremental:** a kernel only ever sees the past; ``update`` is called once per trade
  (trade kernels) or once per *closed* candle (candle kernels), in time order.
* **Exact arithmetic:** values are ``Decimal`` computed under the fixed context :data:`DECIMAL_CONTEXT`
  (platform ``StrategyInput`` values must be ``Decimal``/``int``/``bool``/``str``, never ``float``).
* **No fake values:** while warming up, or when the value is undefined (e.g. zero variance), ``update``
  returns ``None`` and nothing is emitted. Nothing is ever filled with ``0``.

To add a kernel, define a state class and call :func:`register_kernel` with a :class:`KernelDef`.
"""

from __future__ import annotations

import decimal
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

# Fixed precision/rounding so results never depend on the process-wide decimal context. 50 digits make
# the incremental running sums of realistic prices/sizes exact.
DECIMAL_CONTEXT = decimal.Context(
    prec=50,
    rounding=decimal.ROUND_HALF_EVEN,
    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow],
)

MAX_WINDOW = 100_000
OUTPUT_DECIMAL = "decimal"


class KernelError(ValueError):
    """Invalid kernel definition, registration or lookup."""


class FeatureComputationError(ValueError):
    """A kernel received input it cannot interpret (e.g. an unknown aggressor side)."""


@dataclass(frozen=True)
class ParamDef:
    """Declaration of one kernel parameter (type, default and allowed range)."""

    name: str
    kind: str  # "int" | "decimal"
    default: int | Decimal
    minimum: int | Decimal | None = None
    maximum: int | Decimal | None = None
    description: str = ""


@dataclass(frozen=True)
class ClosedCandle:
    """A CLOSED candle as seen by a candle kernel (exact decimals, ns timestamps)."""

    start_ns: int
    end_ns: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


# A candle kernel state: update(candle) -> value | None. A trade kernel state:
# update(price, size, side) -> value | None.
KernelState = Any
KernelFactory = Callable[[Mapping[str, Any], "int | None"], KernelState]


@dataclass(frozen=True)
class KernelDef:
    kernel_id: str
    representation: str  # "trades" | "candles"
    description: str
    params: tuple[ParamDef, ...]
    min_warmup: Callable[[Mapping[str, Any]], int]  # in units (trades or closed candles)
    factory: KernelFactory
    output: str = OUTPUT_DECIMAL

    @property
    def unit(self) -> str:
        return "trades" if self.representation == "trades" else "candles"


_REGISTRY: dict[str, KernelDef] = {}


def register_kernel(definition: KernelDef) -> KernelDef:
    if definition.representation not in {"trades", "candles"}:
        raise KernelError(f"kernel {definition.kernel_id!r}: representation must be 'trades' or 'candles'")
    if definition.kernel_id in _REGISTRY:
        raise KernelError(f"kernel {definition.kernel_id!r} is already registered")
    _REGISTRY[definition.kernel_id] = definition
    return definition


def get_kernel(kernel_id: str) -> KernelDef:
    try:
        return _REGISTRY[kernel_id]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise KernelError(f"unknown kernel {kernel_id!r}; available kernels: {available}") from None


def list_kernels() -> list[KernelDef]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


# ---------------------------------------------------------------------------------------------
# Trade-based kernels: update(price, size, side) -> Decimal | None
# ---------------------------------------------------------------------------------------------


def _signed(size: Decimal, side: str) -> Decimal:
    if side == "buy":
        return size
    if side == "sell":
        return -size
    raise FeatureComputationError(f"unsupported aggressor_side {side!r} (expected 'buy' or 'sell')")


class ImbalanceRatioState:
    """(buy volume - sell volume) / (buy volume + sell volume) over the last ``window`` trades."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._deltas: deque[Decimal] = deque()
        self._buy = Decimal(0)
        self._sell = Decimal(0)

    def update(self, price: Decimal, size: Decimal, side: str) -> Decimal | None:
        delta = _signed(size, side)
        self._deltas.append(delta)
        if delta > 0:
            self._buy += delta
        else:
            self._sell += -delta
        if len(self._deltas) > self._window:
            old = self._deltas.popleft()
            if old > 0:
                self._buy -= old
            else:
                self._sell -= -old
        if len(self._deltas) < self._window:
            return None
        total = self._buy + self._sell
        if total <= 0:
            return None
        return (self._buy - self._sell) / total


class CvdZscoreState:
    """Rolling signed volume (CVD) over the last ``window`` trades divided by its sample std (ddof=1)."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._deltas: deque[Decimal] = deque()
        self._sum = Decimal(0)
        self._sum_sq = Decimal(0)

    def update(self, price: Decimal, size: Decimal, side: str) -> Decimal | None:
        delta = _signed(size, side)
        self._deltas.append(delta)
        self._sum += delta
        self._sum_sq += delta * delta
        if len(self._deltas) > self._window:
            old = self._deltas.popleft()
            self._sum -= old
            self._sum_sq -= old * old
        n = len(self._deltas)
        if n < self._window:
            return None
        variance = (self._sum_sq - self._sum * self._sum / n) / (n - 1)
        if variance <= 0:
            return None  # zero dispersion: the z-score is undefined, not 0
        return self._sum / variance.sqrt()


# ---------------------------------------------------------------------------------------------
# Candle-based kernels: update(candle) -> Decimal | None (called once per CLOSED candle)
# ---------------------------------------------------------------------------------------------


class PriceMomentumState:
    """close_now / close_{window candles ago} - 1."""

    def __init__(self, window: int) -> None:
        self._closes: deque[Decimal] = deque(maxlen=window + 1)
        self._window = window

    def update(self, candle: ClosedCandle) -> Decimal | None:
        self._closes.append(candle.close)
        if len(self._closes) < self._window + 1:
            return None
        base = self._closes[0]
        if base <= 0:
            return None
        return self._closes[-1] / base - 1


class RealizedVolState:
    """Sample std (ddof=1) of the last ``window`` close-to-close log returns, annualised.

    Annualisation assumes a 24/7 market: ``days_per_year`` * 86400 s / bar duration periods per year.
    """

    def __init__(self, window: int, bar_ns: int, days_per_year: int) -> None:
        self._window = window
        self._closes: deque[Decimal] = deque(maxlen=window + 1)
        self._periods_per_year = Decimal(days_per_year * 86_400 * 1_000_000_000) / Decimal(bar_ns)

    def update(self, candle: ClosedCandle) -> Decimal | None:
        self._closes.append(candle.close)
        if len(self._closes) < self._window + 1:
            return None
        closes = list(self._closes)
        if any(close <= 0 for close in closes):
            return None
        returns = [(closes[i + 1] / closes[i]).ln() for i in range(self._window)]
        mean = sum(returns, Decimal(0)) / self._window
        variance = sum(((r - mean) ** 2 for r in returns), Decimal(0)) / (self._window - 1)
        return variance.sqrt() * self._periods_per_year.sqrt()


class AtrPctState:
    """Mean true range over the last ``window`` candles divided by the latest close."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._bars: deque[tuple[Decimal, Decimal, Decimal]] = deque(maxlen=window + 1)

    def update(self, candle: ClosedCandle) -> Decimal | None:
        self._bars.append((candle.high, candle.low, candle.close))
        if len(self._bars) < self._window + 1:
            return None
        bars = list(self._bars)
        true_ranges = []
        for i in range(1, len(bars)):
            high, low, _ = bars[i]
            prev_close = bars[i - 1][2]
            true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        last_close = bars[-1][2]
        if last_close <= 0:
            return None
        return (sum(true_ranges, Decimal(0)) / self._window) / last_close


# ---------------------------------------------------------------------------------------------
# Registration of the built-in kernels
# ---------------------------------------------------------------------------------------------


def _window(default: int, minimum: int, description: str) -> ParamDef:
    return ParamDef("window", "int", default, minimum, MAX_WINDOW, description)


register_kernel(
    KernelDef(
        kernel_id="imbalance_ratio",
        representation="trades",
        description="(buy - sell volume) / (buy + sell volume) over the last `window` trades; in [-1, 1].",
        params=(_window(500, 1, "number of trades in the rolling window"),),
        min_warmup=lambda p: p["window"],
        factory=lambda p, bar_ns: ImbalanceRatioState(p["window"]),
    )
)
register_kernel(
    KernelDef(
        kernel_id="cvd_zscore",
        representation="trades",
        description="Rolling signed volume (CVD) over `window` trades / sample std of the signed volumes.",
        params=(_window(500, 2, "number of trades in the rolling window"),),
        min_warmup=lambda p: p["window"],
        factory=lambda p, bar_ns: CvdZscoreState(p["window"]),
    )
)
register_kernel(
    KernelDef(
        kernel_id="price_momentum",
        representation="candles",
        description="Close-to-close return over `window` closed candles.",
        params=(_window(12, 1, "number of candles back"),),
        min_warmup=lambda p: p["window"] + 1,
        factory=lambda p, bar_ns: PriceMomentumState(p["window"]),
    )
)
register_kernel(
    KernelDef(
        kernel_id="realized_vol",
        representation="candles",
        description="Annualised sample std of the last `window` close-to-close log returns (24/7 market).",
        params=(
            _window(20, 2, "number of log returns in the window"),
            ParamDef("days_per_year", "int", 365, 1, 366, "days per year used to annualise"),
        ),
        min_warmup=lambda p: p["window"] + 1,
        factory=lambda p, bar_ns: RealizedVolState(p["window"], bar_ns, p["days_per_year"]),
    )
)
register_kernel(
    KernelDef(
        kernel_id="atr_pct",
        representation="candles",
        description="Average true range over `window` candles as a fraction of the latest close.",
        params=(_window(14, 1, "number of true ranges averaged"),),
        min_warmup=lambda p: p["window"] + 1,
        factory=lambda p, bar_ns: AtrPctState(p["window"]),
    )
)
