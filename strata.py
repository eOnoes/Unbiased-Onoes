#!/usr/bin/env python3
"""Single local entry point; existing inference services stay isolated."""
from strata_control.server import main

if __name__ == '__main__':
    raise SystemExit(main())
