#!/usr/bin/env python3
"""coulomb_field.py -- Coulomb field lines for image-defined electrodes.

Thin CLI wrapper: the implementation lives in ``src/ehd_flow/``
(:mod:`ehd_flow.coulomb_cli` driving :class:`CoulombField`, which uses the
same :class:`PolygonExtractor` as ``image_to_polygons.py``). Run from
anywhere -- no import path setup needed:

    python coulomb_field.py electrodes.png --no-window
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

from ehd_flow.coulomb_cli import main

if __name__ == "__main__":
    main()
