#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""CLI command: pygraft build

Generates both schema and knowledge graph in sequence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pygraft import generate_kg, generate_schema
from pygraft.cli.formatting import (
    format_path,
    print_consistency_status,
    print_header,
    print_hint,
    print_info,
    print_path_line,
)
from pygraft.cli.validators import validate_config_extension
from pygraft.utils.config import load_config

# ------------------------------------------------------------------------------------------------ #
# Command App                                                                                      #
# ------------------------------------------------------------------------------------------------ #

app = typer.Typer()


# ------------------------------------------------------------------------------------------------ #
# Command                                                                                          #
# ------------------------------------------------------------------------------------------------ #


@app.command("build")
def build_command(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Argument(
            metavar="CONFIG",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the configuration file.",
        ),
    ] = None,
) -> None:
    """Generate both schema and KG in one step (fully synthetic workflow)."""
    if config is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    config = config.expanduser()
    validate_config_extension(config)

    print_header()

    # --- Schema Generation ---
    schema_file, schema_consistent = generate_schema(str(config))

    typer.echo()
    print_path_line("Schema written to:", schema_file)
    print_consistency_status(is_consistent=schema_consistent, reasoner="HermiT")
    typer.echo()

    # --- KG Generation ---
    config_data = load_config(config)
    project_name = config_data.get("general", {}).get("project_name", "Unknown")

    print_info(f">>> Generating KG: {project_name}")
    typer.echo()

    _, kg_file, kg_consistent = generate_kg(str(config))
    kg_path = Path(kg_file)

    typer.echo()
    print_path_line("KG written to:", kg_path)
    print_consistency_status(is_consistent=kg_consistent, reasoner="HermiT")

    # Hint for inconsistent KG
    if kg_consistent is False:
        typer.echo()
        print_hint(
            "To understand why, run:",
            f"pygraft explain {format_path(kg_path)}",
        )

    typer.echo()
