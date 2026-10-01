"""Strategy Hub and Registry for Omega.

Allows registering and loading custom research and trading strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Type

from .base import BaseStrategy


@dataclass(frozen=True)
class StrategyDescriptor:
    """Metadata describing a registered strategy."""

    name: str
    description: str
    strategy_cls: Type[BaseStrategy]
    default_params: dict[str, Any]


_STRATEGY_REGISTRY: dict[str, StrategyDescriptor] = {}


def register_strategy(
    name: str,
    description: str = "",
) -> Callable[[Type[BaseStrategy]], Type[BaseStrategy]]:
    """Decorator to register a custom strategy class."""

    def decorator(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
        sig = inspect.signature(cls.__init__)
        defaults = {
            param.name: param.default
            for param in list(sig.parameters.values())[1:]  # skip self
            if param.default is not inspect.Parameter.empty
        }

        desc = StrategyDescriptor(
            name=name,
            description=description or cls.__doc__ or "No description provided.",
            strategy_cls=cls,
            default_params=defaults,
        )
        cls.name = name
        cls.description = desc.description
        _STRATEGY_REGISTRY[name] = desc
        return cls

    return decorator


def get_strategy(name: str, **params: Any) -> BaseStrategy:
    """Instantiate a registered strategy with optional parameter overrides."""
    if name not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        raise KeyError(f"Strategy '{name}' not found. Available strategies: {available}")

    descriptor = _STRATEGY_REGISTRY[name]
    final_params = dict(descriptor.default_params)
    final_params.update(params)
    return descriptor.strategy_cls(**final_params)


def list_strategies() -> list[StrategyDescriptor]:
    """List all registered strategies sorted by name."""
    return [desc for _, desc in sorted(_STRATEGY_REGISTRY.items())]
