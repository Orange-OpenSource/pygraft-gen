#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""CLI command: pygraft schema

Generates an ontology schema from a configuration file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pygraft import generate_schema
from pygraft.cli.formatting import (
    print_consistency_status,
    print_header,
    print_path_line,
)
from pygraft.cli.validators import validate_config_extension

# ------------------------------------------------------------------------------------------------ #
# Command App                                                                                      #
# ------------------------------------------------------------------------------------------------ #

app = typer.Typer()


# ------------------------------------------------------------------------------------------------ #
# Command                                                                                          #
# ------------------------------------------------------------------------------------------------ #


@app.command("schema")
def schema_command(
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
    """Generate a synthetic schema from configuration parameters."""
    if config is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    config = config.expanduser()
    validate_config_extension(config)

    print_header()

    schema_file, is_consistent = generate_schema(str(config))

    typer.echo()
    print_path_line("Schema written to:", schema_file)
    print_consistency_status(is_consistent=is_consistent, reasoner="HermiT")
    typer.echo()
