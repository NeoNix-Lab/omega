"""Canonical Strategy Hub — Directly interfaces with quant_platform.strategy."""

from __future__ import annotations

from typing import Any

from .. import platform_link as qp
from .compiler import CompiledStrategy, compile_strategy
from .spec import StrategyDef, StrategySpecError, load_strategy_spec, parse_strategy_spec, parse_strategy_spec_text


def list_canonical_strategies() -> list[dict[str, Any]]:
    """List all canonical strategies and specifications from quant-platform."""
    breakout = qp.minimal_breakout_strategy()
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


def __getattr__(name: str) -> Any:
    # Re-exported lazily so that importing omega.strategies does not require the platform.
    if name in {"minimal_breakout_strategy", "StrategySpec"}:
        return getattr(qp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CompiledStrategy",
    "StrategyDef",
    "StrategySpec",
    "StrategySpecError",
    "compile_strategy",
    "list_canonical_strategies",
    "load_strategy_spec",
    "minimal_breakout_strategy",
    "parse_strategy_spec",
    "parse_strategy_spec_text",
]
