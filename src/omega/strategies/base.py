"""Base Strategy Abstractions and Signals for Omega."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class SignalAction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"  # Close existing position
    HOLD = "HOLD"  # Maintain current state / no action


@dataclass(frozen=True)
class SignalDecision:
    """Strategy decision emitted on a market event or bar."""

    action: SignalAction
    confidence: float = 1.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    size_fraction: float = 1.0
    reason: str = ""

    @classmethod
    def LONG(
        cls,
        confidence: float = 1.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        size: float = 1.0,
        reason: str = "",
    ) -> SignalDecision:
        return cls(
            action=SignalAction.LONG,
            confidence=confidence,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            size_fraction=size,
            reason=reason,
        )

    @classmethod
    def SHORT(
        cls,
        confidence: float = 1.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        size: float = 1.0,
        reason: str = "",
    ) -> SignalDecision:
        return cls(
            action=SignalAction.SHORT,
            confidence=confidence,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            size_fraction=size,
            reason=reason,
        )

    @classmethod
    def FLAT(cls, reason: str = "") -> SignalDecision:
        return cls(action=SignalAction.FLAT, reason=reason)

    @classmethod
    def HOLD(cls) -> SignalDecision:
        return cls(action=SignalAction.HOLD)


class BaseStrategy:
    """Abstract base class for all Omega research strategies."""

    name: str = "base_strategy"
    description: str = "Base strategy interface"

    def __init__(self, **params: Any) -> None:
        self.params = params

    def generate_signal(self, row: pd.Series, context: dict[str, Any]) -> SignalDecision:
        """Evaluate a single row/bar and return a trading decision."""
        raise NotImplementedError("Strategies must implement generate_signal")

    def simulate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run strategy sequentially across the dataset and compute simulated execution."""
        sim_df = df.copy()
        actions = []
        confidences = []
        reasons = []

        context: dict[str, Any] = {
            "current_position": 0,  # 1 for long, -1 for short, 0 for flat
            "entry_price": 0.0,
        }

        for idx, row in sim_df.iterrows():
            decision = self.generate_signal(row, context)
            actions.append(decision.action.value)
            confidences.append(decision.confidence)
            reasons.append(decision.reason)

            # Update context
            if decision.action == SignalAction.LONG:
                context["current_position"] = 1
                context["entry_price"] = row["close"]
            elif decision.action == SignalAction.SHORT:
                context["current_position"] = -1
                context["entry_price"] = row["close"]
            elif decision.action == SignalAction.FLAT:
                context["current_position"] = 0
                context["entry_price"] = 0.0

        sim_df["signal"] = actions
        sim_df["confidence"] = confidences
        sim_df["reason"] = reasons

        # Compute position vector (1, -1, 0)
        pos = np.zeros(len(sim_df))
        curr_pos = 0
        for i, act in enumerate(actions):
            if act == SignalAction.LONG.value:
                curr_pos = 1
            elif act == SignalAction.SHORT.value:
                curr_pos = -1
            elif act == SignalAction.FLAT.value:
                curr_pos = 0
            pos[i] = curr_pos

        sim_df["position"] = pos
        # Returns from holding position to next bar
        pct_change = sim_df["close"].pct_change().shift(-1).fillna(0.0)
        sim_df["strategy_return"] = sim_df["position"] * pct_change
        sim_df["equity_curve"] = (1.0 + sim_df["strategy_return"]).cumprod()

        return sim_df
