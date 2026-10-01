#!/usr/bin/env python3
"""Convenience CLI entrypoint for Omega."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local src/ is accessible
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure sibling quant-platform is accessible
QUANT_PLATFORM_SRC = Path(__file__).resolve().parent.parent / "quant-platform" / "src"
if QUANT_PLATFORM_SRC.exists() and str(QUANT_PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(QUANT_PLATFORM_SRC))

from omega.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
