"""Single import surface between Omega and the ``quant_platform`` core.

Every other Omega module must import platform symbols from here (enforced by
``tests/test_platform_boundary.py``). Symbols are resolved lazily, so ``import omega.platform_link``
works, and ``check_compat()`` reports ``MISSING``, even when the platform is not installed.
"""

from __future__ import annotations

import importlib
from typing import Any

from .compat import (
    DISTRIBUTION,
    PLATFORM_PIN,
    CompatResult,
    CompatStatus,
    PlatformPin,
    check_compat,
)
from .exports import PLATFORM_SYMBOLS

__all__ = [
    "DISTRIBUTION",
    "PLATFORM_PIN",
    "PLATFORM_SYMBOLS",
    "CompatResult",
    "CompatStatus",
    "PlatformPin",
    "check_compat",
    *sorted(PLATFORM_SYMBOLS),
]


def __getattr__(name: str) -> Any:
    module_name = PLATFORM_SYMBOLS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(PLATFORM_SYMBOLS))
