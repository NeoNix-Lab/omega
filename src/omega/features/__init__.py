"""Canonical Feature Hub — Directly interfaces with quant_platform.features."""

from __future__ import annotations

from typing import Any

from .. import platform_link as qp


def list_canonical_features() -> list[dict[str, Any]]:
    """List all frozen, canonical features and representations from quant-platform."""
    diag = qp.diagonal_imbalance_definition()
    stacked = qp.stacked_imbalance_definition()

    return [
        {
            "feature_set": qp.H01_IMBALANCE_FEATURE_SET_IDENTITY,
            "feature_key": diag.feature_key,
            "domain": "features",
            "version": diag.semantic_version,
            "description": "Footprint diagonal bid/ask imbalance ratio above threshold.",
        },
        {
            "feature_set": qp.H01_IMBALANCE_FEATURE_SET_IDENTITY,
            "feature_key": stacked.feature_key,
            "domain": "features",
            "version": stacked.semantic_version,
            "description": "Stacked consecutive footprint imbalances at contiguous price levels.",
        },
        {
            "feature_set": "representations.footprint",
            "feature_key": "footprint@1",
            "domain": "representation",
            "version": "1",
            "description": "Exact volume profile aggregated by tick grid and aggressor side (D06).",
        },
        {
            "feature_set": "representations.candle",
            "feature_key": "candle@1",
            "domain": "representation",
            "version": "1",
            "description": "Canonical OHLCV interval representation with availability metadata (D02).",
        },
    ]


def __getattr__(name: str) -> Any:
    # Re-exported lazily so that importing omega.features does not require the platform.
    if name in {"CandleDefinitionV1", "FootprintDefinitionV1"}:
        return getattr(qp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["list_canonical_features", "CandleDefinitionV1", "FootprintDefinitionV1"]
