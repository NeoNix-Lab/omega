"""Omega Feature Hub — Pluggable derived features registry and library."""

from .registry import (
    FeatureDescriptor,
    apply_features,
    get_feature,
    list_features,
    register_feature,
)

# Auto-import standard feature providers to register them
from . import momentum as _momentum  # noqa: F401
from . import orderflow as _orderflow  # noqa: F401
from . import volatility as _volatility  # noqa: F401

__all__ = [
    "FeatureDescriptor",
    "apply_features",
    "get_feature",
    "list_features",
    "register_feature",
]
