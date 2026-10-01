"""Modular Feature Hub and Registry for Omega.

Allows easy creation and plug-and-play registration of derived quantitative features
(order flow, volatility, momentum, microstructural observables).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureDescriptor:
    """Metadata describing one registered derived feature."""

    name: str
    description: str
    required_columns: tuple[str, ...]
    func: Callable[..., pd.Series]
    default_params: dict[str, Any]


_FEATURE_REGISTRY: dict[str, FeatureDescriptor] = {}


def register_feature(
    name: str,
    description: str = "",
    required_columns: tuple[str, ...] = ("close",),
) -> Callable[[Callable[..., pd.Series]], Callable[..., pd.Series]]:
    """Decorator to register a derived feature calculation function.

    The decorated function must accept a `pd.DataFrame` as first argument,
    optionally accept additional parameters, and return a `pd.Series`.
    """

    def decorator(func: Callable[..., pd.Series]) -> Callable[..., pd.Series]:
        sig = inspect.signature(func)
        defaults = {
            param.name: param.default
            for param in list(sig.parameters.values())[1:]  # skip df
            if param.default is not inspect.Parameter.empty
        }

        desc = FeatureDescriptor(
            name=name,
            description=description or func.__doc__ or "No description provided.",
            required_columns=required_columns,
            func=func,
            default_params=defaults,
        )
        _FEATURE_REGISTRY[name] = desc
        return func

    return decorator


def get_feature(name: str) -> FeatureDescriptor:
    """Retrieve a registered feature descriptor by name."""
    if name not in _FEATURE_REGISTRY:
        available = ", ".join(sorted(_FEATURE_REGISTRY.keys()))
        raise KeyError(f"Feature '{name}' not found in registry. Available features: {available}")
    return _FEATURE_REGISTRY[name]


def list_features() -> list[FeatureDescriptor]:
    """Return all registered feature descriptors sorted by name."""
    return [desc for _, desc in sorted(_FEATURE_REGISTRY.items())]


def apply_features(
    df: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...] | str,
    params_override: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute and append specified derived features to the input DataFrame.

    Returns:
        (augmented_df, list_of_computed_column_names)
    """
    if isinstance(feature_names, str):
        feature_names = [f.strip() for f in feature_names.split(",") if f.strip()]

    result_df = df.copy()
    computed_columns: list[str] = []
    params_override = params_override or {}

    for feat_name in feature_names:
        descriptor = get_feature(feat_name)

        # Check required columns
        missing_cols = [col for col in descriptor.required_columns if col not in result_df.columns]
        if missing_cols:
            raise ValueError(
                f"Feature '{feat_name}' requires columns {descriptor.required_columns}, "
                f"but missing: {missing_cols}"
            )

        kwargs = dict(descriptor.default_params)
        if feat_name in params_override:
            kwargs.update(params_override[feat_name])

        col_name = f"feat_{feat_name}"
        series = descriptor.func(result_df, **kwargs)
        result_df[col_name] = series.fillna(0.0)
        computed_columns.append(col_name)

    return result_df, computed_columns
