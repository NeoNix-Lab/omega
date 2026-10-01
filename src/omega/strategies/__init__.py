"""Canonical Strategy Hub — Directly interfaces with quant_platform.strategy."""

from __future__ import annotations

from typing import Any

from quant_platform.application.golden_replay import minimal_breakout_strategy
from quant_platform.strategy import StrategySpec


def list_canonical_strategies() -> list[dict[str, Any]]:
    """List all canonical strategies and specifications from quant-platform."""
    breakout = minimal_breakout_strategy()
    return [
        {
            "name": "minimal_breakout",
            "strategy_identity": breakout.strategy_identity,
            "sizing_policy": breakout.sizing_policy.identity,
            "risk_policy": breakout.risk_policy.identity,
            "session_policy": breakout.session_policy.identity,
            "description": "Canonical Donchian breakout StrategySpec with deterministic policies.",
        }
    ]


__all__ = ["list_canonical_strategies", "minimal_breakout_strategy", "StrategySpec"]
