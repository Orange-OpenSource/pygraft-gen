#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""PyGraft CLI application.

This module defines the root Typer application, global options (version, log level),
and registers all subcommands.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging

import typer

from pygraft.cli.commands import (
    build,
    explain,
    extract,
    init,
    kg,
    schema,
)

# ------------------------------------------------------------------------------------------------ #
# Logging Configuration                                                                            #
# ------------------------------------------------------------------------------------------------ #

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
    "%(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int = logging.WARNING) -> None:
    """Configure global logging for the PyGraft CLI.

    Args:
        level: Numeric logging level. Valid values:
            - 10 = DEBUG
            - 20 = INFO
            - 30 = WARNING (default)
            - 40 = ERROR
            - 50 = CRITICAL
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
    )


def parse_log_level(value: str) -> int:
    """Parse a log-level string from the --log-level CLI option.

    Accepts:
        - Standard names: debug, info, warning, error, critical
        - Common aliases: warn, err, crit
        - Numeric values: 10, 20, 30, 40, 50

    Args:
        value: Raw value provided by the user.

    Returns:
        The corresponding numeric logging level.

    Raises:
        typer.BadParameter: If value does not match any known level.
    """
    value = value.strip()

    # Numeric form (fast path)
    if value.isdigit():
        numeric = int(value)
        if numeric in (10, 20, 30, 40, 50):
            return numeric
        raise typer.BadParameter(
            f"Invalid numeric log level: {numeric}. Allowed: 10, 20, 30, 40, 50."
        )

    # Symbolic names (case-insensitive)
    name_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "warn": logging.WARNING,
        "error": logging.ERROR,
        "err": logging.ERROR,
        "critical": logging.CRITICAL,
        "crit": logging.CRITICAL,
    }

    level = name_map.get(value.lower())
    if level is not None:
        return level

    raise typer.BadParameter(
        f"Invalid log level: '{value}'. "
        "Use: debug, info, warning, error, critical (or 10, 20, 30, 40, 50)."
    )


# ------------------------------------------------------------------------------------------------ #
# Version Callback                                                                                 #
# ------------------------------------------------------------------------------------------------ #


def version_callback(value: bool) -> None:
    """Print the installed pygraft version and exit."""
    if not value:
        return

    try:
        version = importlib_metadata.version("pygraft")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"

    typer.echo(f"pygraft {version}")
    raise typer.Exit()


# ------------------------------------------------------------------------------------------------ #
# Application Definition                                                                           #
# ------------------------------------------------------------------------------------------------ #

app = typer.Typer(
    name="pygraft",
    help="PyGraft: Configurable generation of Schemas & Knowledge Graphs",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ------------------------------------------------------------------------------------------------ #
# Global Callback                                                                                  #
# ------------------------------------------------------------------------------------------------ #


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show the pygraft version and exit.",
    ),
    log_level: str = typer.Option(
        "warning",
        "--log-level",
        "-l",
        metavar="LEVEL",
        help="Logging verbosity: debug, info, warning, error, critical (or 10-50).",
    ),
) -> None:
    """Global options applied before any subcommand."""
    numeric_level = parse_log_level(log_level)
    configure_logging(numeric_level)


# ------------------------------------------------------------------------------------------------ #
# Command Registration                                                                             #
# ------------------------------------------------------------------------------------------------ #

# Register commands at top level (no subcommand grouping)
app.add_typer(init.app)
app.add_typer(schema.app)
app.add_typer(kg.app)
app.add_typer(build.app)
app.add_typer(extract.app)
app.add_typer(explain.app)
