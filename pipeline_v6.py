#!/usr/bin/env python3
"""
Backward-compatibility shim — delegates to the refactored ml.training.train module.

This file exists so that existing automation (scripts, docs, muscle memory)
that run `python pipeline_v6.py` continues to work without changes.

Usage:
    python pipeline_v6.py              # full training run (uses defaults)
    python pipeline_v6.py --dry-run    # data validation only
    python pipeline_v6.py --run-id custom-name

For all new use-cases, prefer invoking directly:
    python -m ml.training.train
"""

import sys
from ml.training.train import main

if __name__ == "__main__":
    sys.exit(main())