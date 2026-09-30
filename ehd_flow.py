#!/usr/bin/env python3
"""Backwards-compatible entry point: ``python ehd_flow.py ...``.

The implementation now lives in the ``ehd_flow/`` package
(config / geometry / forces / solver / rendering / cli); this shim keeps
the old command line working unchanged. New code should use
``python -m ehd_flow ...`` instead.
"""

from ehd_flow.cli import main

if __name__ == "__main__":
    main()
