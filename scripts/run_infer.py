#!/usr/bin/env python
"""
scripts/run_infer.py  –  Root forwarder to code/business_entity_resolution/scripts/run_infer.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add code/business_entity_resolution/scripts and src to sys.path
root = Path(__file__).resolve().parents[1]
scripts_dir = root / "code" / "business_entity_resolution" / "scripts"
src_dir = root / "code" / "business_entity_resolution" / "src"

for p in [str(scripts_dir), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import run_infer

if __name__ == "__main__":
    run_infer.main()
