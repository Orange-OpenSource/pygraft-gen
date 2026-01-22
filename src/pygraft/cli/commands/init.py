#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""CLI command: pygraft init

Creates a new PyGraft configuration template file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pygraft import create_config
from pygraft.cli.formatting import format_path, print_path_line

# ------------------------------------------------------------------------------------------------ #
# Command App                                                                                      #
# ------------------------------------------------------------------------------------------------ #

app = typer.Typer()


# ------------------------------------------------------------------------------------------------ #
# Command                                                                                          #
# ------------------------------------------------------------------------------------------------ #


@app.command("init")
def init_command(
    ctx: typer.Context,
    config_format: Annotated[
        str | None,
        typer.Argument(
            metavar="FORMAT",
            help="Config file format to generate (json, yml, yaml).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=False,
            help="Destination directory for the config file. Defaults to current directory.",
        ),
    ] = None,
) -> None:
    """Create a new configuration template."""
    if config_format is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    normalized = config_format.strip().lower()
    if normalized not in {"json", "yml", "yaml"}:
        raise typer.BadParameter("FORMAT must be one of: json, yml, yaml")

    resolved_output_dir = output_dir.expanduser() if output_dir is not None else None

    try:
        created_path = create_config(config_format=normalized, output_dir=resolved_output_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo()
    print_path_line("Configuration file written to:", created_path)
    typer.echo()
