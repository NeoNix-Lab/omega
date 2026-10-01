"""Entrypoint for python -m omega."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root src is in path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure quant-platform is in path if available
QUANT_PLATFORM_SRC = SRC_DIR.parent.parent / "quant-platform" / "src"
if QUANT_PLATFORM_SRC.exists() and str(QUANT_PLATFORM_SRC) not in sys.path:
    sys.path.insert(0, str(QUANT_PLATFORM_SRC))

from omega.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
