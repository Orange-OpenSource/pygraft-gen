"""CLI-facing helpers for PyGraft.

This module contains utilities that are strictly related to command-line
presentation, such as the ASCII banner shown at startup, and shared CLI
infrastructure like logging helpers and log-level parsers.

It is intentionally small and focused on user-facing output. Non-UI logic
belongs in other modules.
"""

from __future__ import annotations

from collections.abc import Callable
import argparse
import logging
import random
from typing import cast

from art import text2art as _text2art  # pyright: ignore[reportUnknownVariableType]

logger = logging.getLogger(__name__)

# ================================================================================================ #
# Third-Party Wrappers to please Pyright/BasedPyright                                              #
# ================================================================================================ #

TextToArtCallable = Callable[..., str]
text2art: TextToArtCallable = cast(TextToArtCallable, _text2art)

# ================================================================================================ #
# Logging Configuration                                                                            #
# ================================================================================================ #

LOGURU_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
    "%(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
LOGURU_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int = 20) -> None:
    """
    Configure the global logging system for the Pygraft CLI.

    Parameters
    ----------
    level : int, optional
        The numeric logging level to use (default: 20 / INFO).
        Typical values:
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
    """Parse a user-provided logging level passed via the --log CLI argument.

    This function accepts:
    - Symbolic log level names (case-insensitive):
      debug, info, warning, error, critical
      And the following common aliases:
      warn → warning
      err → error
      crit → critical

    - Numeric log levels:
      10, 20, 30, 40, 50

    Parameters
    ----------
    value : str
        The raw string provided by the user through the --log argument.

    Returns
    -------
    int
        The corresponding numeric Python logging level.

    Raises
    ------
    argparse.ArgumentTypeError
        If the provided value does not match a valid symbolic or numeric log level.
    """
    value = value.strip()

    # Numeric form first (fast path)
    if value.isdigit():
        numeric = int(value)
        if numeric in (10, 20, 30, 40, 50):
            return numeric
        raise argparse.ArgumentTypeError(
            f"Invalid numeric log level: {numeric}. Allowed values: 10, 20, 30, 40, 50."
        )

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

    raise argparse.ArgumentTypeError(
        f"Invalid log level '{value}'. "
        "Use names (debug, info, warning, error, critical) "
        "or aliases (warn, err, crit), "
        "or numeric values (10, 20, 30, 40, 50)."
    )


# ================================================================================================ #
# ASCII Header                                                                                     #
# ================================================================================================ #

_FONT_STYLES = ["dancingfont", "rounded", "varsity", "wetletter", "chunky"]

def print_ascii_header() -> None:
    """Print the PyGraft ASCII header using a randomly selected ASCII-art font."""
    font_name = random.choice(_FONT_STYLES)

    header = text2art("PyGraft", font=font_name)

    # Explicitly user-facing CLI output; prints are intentional here.
    print("\n")  # noqa: T201
    print(header)  # noqa: T201
    print("\n")  # noqa: T201
