#!/usr/bin/env python3
"""PyInstaller entry point for the full evaluation sweep."""

from __future__ import annotations

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    ROOT = Path(sys._MEIPASS)
else:
    ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_all_models import main

if __name__ == "__main__":
    main()
