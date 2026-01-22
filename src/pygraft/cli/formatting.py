#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""CLI formatting utilities for PyGraft.

This module provides reusable display helpers for consistent CLI output:
path formatting, headers, status messages, and result summaries.
"""

from __future__ import annotations

from pathlib import Path

import typer


# ------------------------------------------------------------------------------------------------ #
# Path Formatting                                                                                  #
# ------------------------------------------------------------------------------------------------ #


def format_path(path: Path) -> str:
    """Format a path for CLI display.

    If the path is inside the current working directory, display it as "./...".
    Otherwise display the absolute path.

    Args:
        path: Path to format.

    Returns:
        Formatted path string.
    """
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return str(resolved)
    return f"./{relative}"


# ------------------------------------------------------------------------------------------------ #
# Header                                                                                           #
# ------------------------------------------------------------------------------------------------ #


def print_header() -> None:
    """Print the PyGraft CLI header."""
    from rich.console import Console
    from rich.rule import Rule

    console = Console()
    console.print()
    console.print(Rule(title="[bright_blue]PyGraft-Gen[/bright_blue]", style="bright_blue"))
    console.print()


# ------------------------------------------------------------------------------------------------ #
# Status Messages                                                                                  #
# ------------------------------------------------------------------------------------------------ #


def print_path_line(label: str, path: Path) -> None:
    """Print a labeled path line.

    Args:
        label: Description label (e.g., "Schema written to:").
        path: Path to display.
    """
    typer.echo(
        typer.style(label, bold=True)
        + " "
        + typer.style(format_path(path), fg=typer.colors.CYAN)
    )


def print_consistency_status(*, is_consistent: bool | None, reasoner: str = "HermiT") -> None:
    """Print a consistency check result.

    Args:
        is_consistent: True if consistent, False if inconsistent, None if skipped.
        reasoner: Name of the reasoner used.
    """
    if is_consistent is True:
        typer.echo(
            typer.style(f"({reasoner}) Consistent", fg=typer.colors.GREEN, bold=True)
        )
    elif is_consistent is False:
        typer.echo(
            typer.style(f"({reasoner}) Inconsistent", fg=typer.colors.RED, bold=True)
        )
    else:
        typer.echo(
            typer.style(f"({reasoner}) Skipped", fg=typer.colors.YELLOW, bold=True)
        )


def print_success(message: str) -> None:
    """Print a success message in green."""
    typer.echo(typer.style(message, fg=typer.colors.GREEN, bold=True))


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    typer.echo(typer.style(message, fg=typer.colors.YELLOW, bold=True))


def print_error(message: str) -> None:
    """Print an error message in red."""
    typer.echo(typer.style(message, fg=typer.colors.RED, bold=True))


def print_info(message: str) -> None:
    """Print an info message in cyan."""
    typer.echo(typer.style(message, fg=typer.colors.CYAN))


def print_section_header(title: str) -> None:
    """Print a section header with horizontal rules.

    Args:
        title: Section title to display.
    """
    typer.echo()
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))
    typer.echo(typer.style(title, fg=typer.colors.YELLOW, bold=True))
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))
    typer.echo()


def print_hint(message: str, command: str | None = None) -> None:
    """Print a hint message with optional command suggestion.

    Args:
        message: Hint text.
        command: Optional command to suggest.
    """
    if command:
        typer.echo(
            typer.style(message, fg=typer.colors.YELLOW)
            + " "
            + typer.style(command, fg=typer.colors.CYAN, bold=True)
        )
    else:
        typer.echo(typer.style(message, fg=typer.colors.YELLOW))
