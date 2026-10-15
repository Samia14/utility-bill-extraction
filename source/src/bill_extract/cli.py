"""Command line entry point for bill extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bill_extract.config import load_config
from bill_extract.errors import ConfigurationError
from bill_extract.pipeline import run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read Pakistani electricity bills and write level1.csv."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A JPG or PNG bill image, or a folder of those images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write level1.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return 0 when every file succeeds, 1 when a file fails, and 2 when config is invalid.

    `ask` answers one question about a saved bill and leaves extraction unchanged.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "ask":
        from bill_extract.ask import ask_main

        return ask_main(raw[1:])
    if raw and raw[0] == "answers":
        from bill_extract.answers import answers_main

        return answers_main(raw[1:])
    args = build_parser().parse_args(raw)
    try:
        config = load_config()
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return run_batch(Path(args.input), Path(args.output), config)


if __name__ == "__main__":
    raise SystemExit(main())
