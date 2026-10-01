"""Order Flow Delta Absorption Reversal Strategy."""

from __future__ import annotations

from typing import Any
import pandas as pd

from .base import BaseStrategy, SignalAction, SignalDecision
from .registry import register_strategy


@register_strategy(
    name="orderflow_absorption",
    description="Detects extreme market delta absorbed by passive limit orders, triggering mean-reversion entries.",
)
class OrderFlowAbsorptionStrategy(BaseStrategy):
    """Mean-reversion strategy based on extreme CVD Z-Score exhaustion."""

    def __init__(
        self,
        delta_threshold: float = 2.0,
        stop_loss_pct: float = 0.006,
        take_profit_pct: float = 0.012,
    ) -> None:
        super().__init__(
            delta_threshold=delta_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        self.delta_threshold = delta_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate_signal(self, row: pd.Series, context: dict[str, Any]) -> SignalDecision:
        current_pos = context.get("current_position", 0)
        entry_price = context.get("entry_price", 0.0)
        close_price = float(row.get("close", 0.0))

        # 1. Manage active position (Exit conditions: Stop-loss or Take-profit)
        if current_pos != 0 and entry_price > 0.0 and close_price > 0.0:
            price_change = (close_price - entry_price) / entry_price
            pnl_direction = price_change if current_pos == 1 else -price_change

            if pnl_direction >= self.take_profit_pct:
                return SignalDecision.FLAT(reason="Take profit target reached")
            if pnl_direction <= -self.stop_loss_pct:
                return SignalDecision.FLAT(reason="Stop loss threshold breached")

        # 2. Check for Absorption Entry
        delta_val = float(row.get("feat_cvd_zscore", row.get("delta", 0.0)))

        # Extreme selling pressure absorbed -> Buy
        if delta_val < -self.delta_threshold and current_pos <= 0:
            return SignalDecision.LONG(
                stop_loss=self.stop_loss_pct,
                take_profit=self.take_profit_pct,
                reason=f"Bullish absorption: extreme sell delta ({delta_val:.2f}) absorbed",
            )

        # Extreme buying pressure absorbed -> Sell
        if delta_val > self.delta_threshold and current_pos >= 0:
            return SignalDecision.SHORT(
                stop_loss=self.stop_loss_pct,
                take_profit=self.take_profit_pct,
                reason=f"Bearish absorption: extreme buy delta ({delta_val:.2f}) absorbed",
            )

        return SignalDecision.HOLD()
