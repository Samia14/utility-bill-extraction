"""Allow python -m bill_extract to run the command line tool."""

from __future__ import annotations

import sys

from bill_extract.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
