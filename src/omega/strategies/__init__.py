"""Omega Strategy Hub — Pluggable alpha strategies and signals."""

# Auto-import standard strategies
from . import breakout as _breakout  # noqa: F401
from . import orderflow_absorption as _absorption  # noqa: F401
from .base import BaseStrategy, SignalAction, SignalDecision
from .registry import StrategyDescriptor, get_strategy, list_strategies, register_strategy

__all__ = [
    "BaseStrategy",
    "SignalAction",
    "SignalDecision",
    "StrategyDescriptor",
    "get_strategy",
    "list_strategies",
    "register_strategy",
]
