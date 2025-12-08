"""Command-line interface entrypoint for the pygraft package."""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import logging
from pathlib import Path
import sys

from pygraft import (
    create_json_template,
    create_yaml_template,
    generate_kg,
    generate_schema,
)
from pygraft.utils.cli import configure_logging, parse_log_level, print_ascii_header

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------ #
# Init                                                                                             #
# ------------------------------------------------------------------------------------------------ #

SCRIPT_USAGE_DESCRIPTION = "Schema & KG Generator"

# ------------------------------------------------------------------------------------------------ #
# Argument Parsing                                                                                 #
# ------------------------------------------------------------------------------------------------ #


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pygraft CLI.

    Args:
        argv: Optional list of argument strings to parse instead of sys.argv.
            Intended primarily for testing and programmatic use. If None, sys.argv is used implicitly.

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=SCRIPT_USAGE_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    try:
        dist_version = importlib_metadata.version("pygraft")
    except importlib_metadata.PackageNotFoundError:
        dist_version = "unknown"

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"pygraft {dist_version}",
        help="Show the pygraft version and exit.",
    )
    parser.add_argument(
        "--log",
        type=parse_log_level,
        default=20,  # INFO
        help=(
            "Logging level. "
            "Accepts either names or numeric values: debug (10), info (20), "
            "warning/warn (30), error/err (40), critical/crit (50)."
        ),
    )
    parser.add_argument(
        "-t",
        "--template",
        action="store_true",
        default=None,
        help="Create a config template in the current working directory.",
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        default=None,
        help="Config file extension. Options: json | yaml | yml",
    )
    parser.add_argument(
        "-jt",
        "--json_template",
        action="store_true",
        default=None,
        help="Create a json config template in the current working directory.",
    )
    parser.add_argument(
        "-yt",
        "--yaml_template",
        action="store_true",
        default=None,
        help="Create a yml config template in the current working directory.",
    )
    parser.add_argument(
        "-conf",
        "--config",
        type=str,
        default=None,
        help="Load a given config file.",
    )
    parser.add_argument(
        "-g",
        "--gen",
        type=str,
        choices=["generate_schema", "generate_kg", "generate"],
        default=None,
        help=(
            "Which function to call. Options: generate_schema | generate_kg | "
            "generate (runs both schema & KG)."
        ),
    )

    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------ #
# Main Entrypoint                                                                                  #
# ------------------------------------------------------------------------------------------------ #


def main() -> None:
    """Entry point for the pygraft command-line interface."""
    # If called with no CLI arguments, behave like `pygraft --help`.
    if len(sys.argv) == 1:
        parse_arguments(["-h"])
        return

    args = parse_arguments()
    configure_logging(args.log)

    if args.template:
        if args.extension == "json":
            create_json_template()
        else:
            create_yaml_template()

    if args.json_template:
        create_json_template()

    if args.yaml_template:
        create_yaml_template()

    if args.config:
        file_extension = Path(args.config).suffix

        if file_extension.lower() not in (".yml", ".yaml", ".json"):
            print(  # noqa: T201
                "Error: invalid config file extension. Expected .json, .yaml, or .yml.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.gen == "generate_schema":
        print_ascii_header()
        generate_schema(args.config)

    if args.gen == "generate_kg":
        print_ascii_header()
        generate_kg(args.config)

    if args.gen == "generate":
        print_ascii_header()
        generate_schema(args.config)
        generate_kg(args.config)


if __name__ == "__main__":
    # Dev-only fallback: `pygraft --help` is the official entrypoint; `python -m pygraft.cli` is deprecated.
    main()
