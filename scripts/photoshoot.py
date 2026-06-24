#!/usr/bin/env python3
"""Open the multi-object MuJoCo photoshoot scene."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.scene import load_photoshoot


def main() -> None:
    load_photoshoot()


if __name__ == "__main__":
    main()
