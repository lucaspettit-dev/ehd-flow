#!/usr/bin/env python3
"""Backwards-compatible entry point: ``python ehd_flow.py ...``.

Thin shim over the ``ehd_flow`` package in ``src/``. All flags and behavior
are unchanged; ``python -m ehd_flow`` works too (after ``pip install -e .``
or with ``src`` on the import path).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

from ehd_flow.cli import main

if __name__ == "__main__":
    main()
