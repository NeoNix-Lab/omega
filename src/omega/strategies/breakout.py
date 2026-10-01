"""Volatility and Donchian Breakout Strategy."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, SignalDecision
from .registry import register_strategy


@register_strategy(
    name="volatility_breakout",
    description="Captures directional expansion moves when momentum and volatility surge concurrently.",
)
class VolatilityBreakoutStrategy(BaseStrategy):
    """Trend-following breakout strategy using momentum and volatility gates."""

    def __init__(
        self,
        momentum_threshold: float = 0.003,
        stop_loss_pct: float = 0.008,
        take_profit_pct: float = 0.020,
    ) -> None:
        super().__init__(
            momentum_threshold=momentum_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        self.momentum_threshold = momentum_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate_signal(self, row: pd.Series, context: dict[str, Any]) -> SignalDecision:
        current_pos = context.get("current_position", 0)
        entry_price = context.get("entry_price", 0.0)
        close_price = float(row.get("close", 0.0))

        # 1. Manage active position (Exit conditions)
        if current_pos != 0 and entry_price > 0.0 and close_price > 0.0:
            price_change = (close_price - entry_price) / entry_price
            pnl_direction = price_change if current_pos == 1 else -price_change

            if pnl_direction >= self.take_profit_pct:
                return SignalDecision.FLAT(reason="Take profit target reached")
            if pnl_direction <= -self.stop_loss_pct:
                return SignalDecision.FLAT(reason="Trailing/Fixed stop loss breached")

        # 2. Entry signal
        mom = float(row.get("feat_price_momentum", row.get("momentum", 0.0)))

        if mom > self.momentum_threshold and current_pos <= 0:
            return SignalDecision.LONG(
                stop_loss=self.stop_loss_pct,
                take_profit=self.take_profit_pct,
                reason=f"Bullish breakout: momentum {mom:.4f} exceeded threshold",
            )

        if mom < -self.momentum_threshold and current_pos >= 0:
            return SignalDecision.SHORT(
                stop_loss=self.stop_loss_pct,
                take_profit=self.take_profit_pct,
                reason=f"Bearish breakdown: momentum {mom:.4f} breached threshold",
            )

        return SignalDecision.HOLD()
