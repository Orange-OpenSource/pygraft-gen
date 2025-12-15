#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""CLI-facing helpers for PyGraft.

This module provides utilities related to user-facing command-line
presentation, such as the CLI header and shared logging configuration.
Only UI-related helpers should be defined here.
"""

from __future__ import annotations

import argparse
import logging

from rich.console import Console
from rich.rule import Rule

logger = logging.getLogger(__name__)

# ================================================================================================ #
# Logging Configuration                                                                            #
# ================================================================================================ #

LOGURU_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
    "%(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
LOGURU_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int = 20) -> None:
    """Configure global logging for the PyGraft CLI.

    Args:
        level: Numeric logging level (default: 20 / INFO). Valid values:
            - 10 = DEBUG
            - 20 = INFO
            - 30 = WARNING
            - 40 = ERROR
            - 50 = CRITICAL
    """
    logging.basicConfig(
        level=level,
        format=LOGURU_FORMAT,
        datefmt=LOGURU_DATEFMT,
    )


def parse_log_level(value: str) -> int:
    """Parse a log-level string from the --log or -l CLI option.

    This function accepts:
        - Standard names: debug, info, warning, error, critical
        - Common aliases: warn → warning, err → error, crit → critical
        - Numeric values: 10, 20, 30, 40, 50

    Args:
        value: Raw value provided by the user.

    Returns:
        The corresponding numeric logging level.

    Raises:
        argparse.ArgumentTypeError: If value does not match any known level.
    """
    value = value.strip()

    # Numeric form first (fast path)
    if value.isdigit():
        numeric = int(value)
        if numeric in (10, 20, 30, 40, 50):
            return numeric

        msg = (
            f"Invalid numeric log level: {numeric}. "
            "Allowed values: 10, 20, 30, 40, 50."
        )
        raise argparse.ArgumentTypeError(msg)

    # Normalize symbolic names (case-insensitive)
    name = value.lower()

    # Core mappings
    name_map = {
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40,
        "critical": 50,
        # Aliases
        "warn": 30,
        "err": 40,
        "crit": 50,
    }

    if name in name_map:
        return name_map[name]

    msg = (
        f"Invalid log level '{value}'. "
        "Use standard names (debug, info, warning, error, critical), "
        "aliases (warn, err, crit), or numeric values (10, 20, 30, 40, 50)."
    )
    raise argparse.ArgumentTypeError(msg)


# ================================================================================================ #
# CLI Header                                                                                        #
# ================================================================================================ #

console = Console()

def print_ascii_header() -> None:
    """Print the PyGraft CLI header line."""
    console.print()
    console.print(Rule(
        title="[bright_blue]PyGraft-Gen[/bright_blue]",
        style="bright_blue"))
    console.print()
