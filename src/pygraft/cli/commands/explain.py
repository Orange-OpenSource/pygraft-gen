#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""CLI command: pygraft explain

Runs a reasoner to explain inconsistencies in an existing knowledge graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pygraft import explain_kg
from pygraft.cli.formatting import (
    print_error,
    print_header,
    print_hint,
    print_info,
    print_path_line,
    print_success,
    print_warning,
)

# ------------------------------------------------------------------------------------------------ #
# Command App                                                                                      #
# ------------------------------------------------------------------------------------------------ #

app = typer.Typer()


# ------------------------------------------------------------------------------------------------ #
# Command                                                                                          #
# ------------------------------------------------------------------------------------------------ #


@app.command("explain")
def explain_command(
    ctx: typer.Context,
    kg_path: Annotated[
        Path | None,
        typer.Argument(
            metavar="KG_PATH",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the knowledge graph file (kg.ttl, kg.rdf, or kg.nt).",
        ),
    ] = None,
    reasoner: Annotated[
        str,
        typer.Option(
            "--reasoner",
            "-r",
            help=(
                "Which reasoner(s) to use: "
                "hermit (fast, no explanation), "
                "pellet (detailed explanations), "
                "both (hermit first, then pellet if inconsistent)."
            ),
        ),
    ] = "pellet",
) -> None:
    """Run reasoner to explain inconsistencies in an existing KG.

        \b
        This command analyzes an existing KG file and provides detailed explanations
        of any logical inconsistencies found. The schema is automatically detected
        from the same directory as the KG.

        \b
        Examples:
            pygraft explain ./output_pygraft/my-project/kg.ttl
            pygraft explain ./output_pygraft/my-project/kg.ttl --reasoner hermit
            pygraft explain ./output_pygraft/my-project/kg.ttl --reasoner both
    """
    if kg_path is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    kg_path = kg_path.expanduser().resolve()

    print_header()

    # Section header
    typer.echo()
    typer.echo(typer.style("=" * 72, fg=typer.colors.YELLOW))
    typer.echo(typer.style("CONSISTENCY EXPLANATION", fg=typer.colors.YELLOW, bold=True))
    typer.echo(typer.style("=" * 72, fg=typer.colors.YELLOW))
    typer.echo()

    print_path_line("Analyzing KG:", kg_path)
    typer.echo(
        typer.style("Reasoner mode:", bold=True)
        + " "
        + typer.style(reasoner, fg=typer.colors.CYAN)
    )
    typer.echo()

    # Run explanation
    try:
        is_consistent, explanation = explain_kg(str(kg_path), reasoner=reasoner)
    except (FileNotFoundError, ValueError) as e:
        typer.echo()
        print_error(f"Error: {e}")
        raise typer.Exit(1)

    typer.echo()

    # Display results
    if is_consistent:
        _display_consistent_result(reasoner)
    else:
        _display_inconsistent_result(reasoner, explanation)

    typer.echo()
    typer.echo(typer.style("=" * 72, fg=typer.colors.YELLOW))
    typer.echo()


# ------------------------------------------------------------------------------------------------ #
# Display Helpers                                                                                  #
# ------------------------------------------------------------------------------------------------ #


def _display_consistent_result(reasoner: str) -> None:
    """Display output for a consistent KG."""
    if reasoner == "hermit":
        print_success("(HermiT) Knowledge Graph is CONSISTENT")
    elif reasoner == "pellet":
        print_success("(Pellet) Knowledge Graph is CONSISTENT")
    elif reasoner == "both":
        print_success("(HermiT) Knowledge Graph is CONSISTENT")
        print_info("(Pellet check skipped - HermiT confirmed consistency)")

    typer.echo()
    typer.echo("No inconsistencies found.")


def _display_inconsistent_result(reasoner: str, explanation: str | None) -> None:
    """Display output for an inconsistent KG."""
    if reasoner == "hermit":
        print_error("(HermiT) Knowledge Graph is INCONSISTENT")
        typer.echo()
        print_hint("Run with --reasoner pellet for detailed explanation")

    elif reasoner in ("pellet", "both"):
        reasoner_name = "Pellet" if reasoner == "pellet" else "HermiT + Pellet"
        print_error(f"({reasoner_name}) Knowledge Graph is INCONSISTENT")
        typer.echo()

        if explanation:
            typer.echo(typer.style("--- Pellet Explanation ---", fg=typer.colors.YELLOW, bold=True))
            typer.echo()
            typer.echo(explanation)
        else:
            print_warning("Warning: Pellet reasoner did not produce an explanation.")
            typer.echo("This can happen with large or highly complex KGs where Pellet times out.")
