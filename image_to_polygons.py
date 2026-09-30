#!/usr/bin/env python3
"""image_to_polygons.py -- turn blobs in an image into polygon obstacles.

Thin CLI wrapper: the implementation lives in ``src/ehd_flow/``
(:mod:`ehd_flow.polygons_cli` driving :class:`PolygonExtractor`). Run from
anywhere -- no import path setup needed:

    python image_to_polygons.py input.png obstacles.json --no-window
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

from ehd_flow.polygons_cli import main

if __name__ == "__main__":
    main()
