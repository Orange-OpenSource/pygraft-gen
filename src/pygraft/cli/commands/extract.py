#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""CLI command: pygraft extract

Extracts ontology metadata into PyGraft JSON artefacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pygraft import extract_ontology
from pygraft.cli.extract_helper import ensure_and_update_config_for_ontology
from pygraft.cli.formatting import (
    format_path,
    print_header,
    print_section_header,
    print_warning,
)

# ------------------------------------------------------------------------------------------------ #
# Command App                                                                                      #
# ------------------------------------------------------------------------------------------------ #

app = typer.Typer()


# ------------------------------------------------------------------------------------------------ #
# Command                                                                                          #
# ------------------------------------------------------------------------------------------------ #


@app.command("extract")
def extract_command(
    ctx: typer.Context,
    ontology: Annotated[
        Path | None,
        typer.Argument(
            metavar="ONTOLOGY",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the ontology file (.ttl, .rdf, .owl, .xml).",
        ),
    ] = None,
) -> None:
    """Extract ontology metadata into PyGraft JSON artefacts.

    \b
    Analyzes an existing ontology and creates a configuration file
    pre-populated with the extracted statistics.

    \b
    Example:
        pygraft extract ./ontologies/my-ontology.ttl
    """
    if ontology is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    ontology = ontology.expanduser()

    print_header()

    # Status message
    print_warning(">>> Extracting ontology... This may take up to 2 minutes.")
    typer.echo()

    # Run extraction
    namespaces_path, class_info_path, relation_info_path = extract_ontology(ontology)

    # Schema file path
    schema_ext = ".ttl" if ontology.suffix.lower() == ".ttl" else ".rdf"
    schema_path = namespaces_path.parent / f"schema{schema_ext}"

    # --- Output Files ---
    print_section_header("OUTPUT FILES")

    typer.echo(
        typer.style("Namespaces:", bold=True)
        + "   "
        + typer.style(format_path(namespaces_path), fg=typer.colors.CYAN)
    )
    typer.echo(
        typer.style("Classes:", bold=True)
        + "      "
        + typer.style(format_path(class_info_path), fg=typer.colors.CYAN)
    )
    typer.echo(
        typer.style("Relations:", bold=True)
        + "    "
        + typer.style(format_path(relation_info_path), fg=typer.colors.CYAN)
    )
    typer.echo(
        typer.style("Schema:", bold=True)
        + "       "
        + typer.style(format_path(schema_path), fg=typer.colors.CYAN)
    )

    # --- Configuration ---
    print_section_header("CONFIGURATION")

    config_path, _ = ensure_and_update_config_for_ontology(
        ontology_file=ontology,
        class_info_path=class_info_path,
        relation_info_path=relation_info_path,
    )

    typer.echo(
        typer.style("Updated:", bold=True)
        + " "
        + typer.style(format_path(config_path), fg=typer.colors.CYAN)
    )
    typer.echo()
    typer.echo(typer.style("  [kg]", fg=typer.colors.GREEN, bold=True) + "       Edit to control KG generation")
    typer.echo(typer.style("  [general]", fg=typer.colors.YELLOW, bold=True) + "  Auto-configured from ontology")
    typer.echo(typer.style("  [schema]", fg=typer.colors.CYAN, bold=True) + "   Read-only extraction statistics")

    # --- Next ---
    typer.echo()
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))
    typer.echo(
        typer.style("Next:", bold=True)
        + " "
        + typer.style(f"pygraft kg {format_path(config_path)}", fg=typer.colors.CYAN)
    )
    typer.echo()
