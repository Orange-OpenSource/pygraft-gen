#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Typer-based command-line interface for the pygraft package.

Global options:
    -V / --version     Show the pygraft version and exit.
    -l / --log-level   Configure logging verbosity.

Subcommands:
    init               Create a JSON/YAML config template.
    schema             Generate only the schema from a config file.
    kg                 Generate only the knowledge graph (KG).
    build              Generate both schema and KG in sequence.
"""

# ruff: noqa: B008
# Typer requires using `typer.Argument()` and `typer.Option()` inside function
# defaults to declare CLI parameters. Ruff rule B008 forbids function-call
# defaults, but that rule does not apply to Typer-style CLIs, so we disable it
# for this module explicitly.

# ================================================================================================ #
# Imports & Setup                                                                                  #
# ================================================================================================ #
from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging
from pathlib import Path
from typing import cast
import json

import click
import typer

from pygraft import create_config, generate_kg, generate_schema, extract_ontology
from pygraft.utils.cli import configure_logging, parse_log_level, print_ascii_header
from pygraft.utils.ontology_extraction import ensure_and_update_config_for_ontology

logger = logging.getLogger(__name__)


# ================================================================================================ #
# Typer Application                                                                                #
# ================================================================================================ #

app = typer.Typer(
    help="Schema & KG Generator",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# ================================================================================================ #
# Version flag handler                                                                             #
# ================================================================================================ #


def _version_callback(value: bool) -> None:
    """Print the installed pygraft version when --version / -V is passed."""
    if not value:
        return

    try:
        version = importlib_metadata.version("pygraft")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"

    typer.echo(f"pygraft {version}")
    raise typer.Exit


# ================================================================================================ #
# Validation helpers                                                                               #
# ================================================================================================ #


def _validate_config_path(config: Path) -> None:
    """Ensure the configuration file has a supported extension."""
    ext = config.suffix.lower()
    if ext not in {".json", ".yaml", ".yml"}:
        msg = "Config file must end with .json, .yaml, or .yml."
        raise typer.BadParameter(msg)


def _format_path_for_display(path: Path) -> str:
    """Format a path for CLI display.

    If the path is inside the current working directory, display it as "./...".
    Otherwise display the absolute path.
    """
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return str(resolved)
    return f"./{relative}"


def _run_schema(config_path: Path) -> None:
    """Generate the schema and print a concise summary for the CLI."""
    schema_file, is_consistent = generate_schema(str(config_path))
    display_path = _format_path_for_display(schema_file)

    path_line = (
        typer.style("Schema written to:", bold=True)
        + " "
        + typer.style(display_path, fg=typer.colors.CYAN)
    )

    if is_consistent:
        status_line = typer.style(
            "(HermiT) Consistent schema",
            fg=typer.colors.GREEN,
            bold=True,
        )
    else:
        status_line = typer.style(
            "(HermiT) Inconsistent schema",
            fg=typer.colors.RED,
            bold=True,
        )

    typer.echo()
    typer.echo(path_line)
    typer.echo(status_line)
    typer.echo()


def _run_kg(config_path: Path, *, explain_inconsistency: bool) -> None:
    """Generate the KG and print a concise summary for the CLI."""
    project_name = json.load(open(config_path)).get('general', {}).get('project_name', 'Unknown')
    typer.echo(typer.style(f"\n>>> Generating KG: {project_name}\n", fg=typer.colors.CYAN))

    pellet_explanations: list[str] = []

    _, kg_file, is_consistent = generate_kg(
        str(config_path),
        explain_inconsistency=explain_inconsistency,
        explanation_sink=pellet_explanations,
    )
    display_path = _format_path_for_display(Path(kg_file))

    path_line = (
        typer.style("KG written to:", bold=True)
        + " "
        + typer.style(display_path, fg=typer.colors.CYAN)
    )

    if is_consistent is True:
        status_line = typer.style(
            "(HermiT) Consistent KG",
            fg=typer.colors.GREEN,
            bold=True,
        )
    elif is_consistent is False:
        status_line = typer.style(
            "(HermiT) Inconsistent KG",
            fg=typer.colors.RED,
            bold=True,
        )
    else:
        status_line = typer.style(
            "(HermiT) Skipped KG reasoning",
            fg=typer.colors.YELLOW,
            bold=True,
        )

    typer.echo()
    typer.echo(path_line)
    typer.echo(status_line)

    if pellet_explanations:
        typer.echo()
        typer.echo(
            typer.style(
                "--- Pellet explanation ---",
                fg=typer.colors.YELLOW,
                bold=True,
            )
        )
        typer.echo(pellet_explanations[-1])


# ================================================================================================ #
# Global callback (runs before any command)                                                        #
# ================================================================================================ #


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the pygraft version and exit.",
    ),
    log_level: str = typer.Option(
        "warning",
        "--log-level",
        "-l",
        metavar="INT|TEXT",
        help=(
            "Logging verbosity level.\n\n"
            "Values:\n"
            "  10 → debug\n"
            "  20 → info\n"
            "  30 → warning, warn\n"
            "  40 → error, err\n"
            "  50 → critical, crit\n"
        ),
    ),
) -> None:
    """Global CLI options executed before any subcommand."""
    numeric_level = parse_log_level(log_level)
    configure_logging(numeric_level)


# ================================================================================================ #
# Subcommands                                                                                      #
# ================================================================================================ #


@app.command("help")
def help_command(
    ctx: typer.Context,
    command: str | None = typer.Argument(
        None,
        metavar="COMMAND",
        help="Name of the command to show help for.",
    ),
) -> None:
    """Show help for pygraft or one of its subcommands"""
    root_ctx = cast(typer.Context, ctx.parent or ctx)

    if command is None:
        typer.echo(root_ctx.get_help())
        raise typer.Exit

    cmd_group: click.Group = cast(click.Group, root_ctx.command)
    click_command: click.Command | None = cmd_group.get_command(root_ctx, command)
    if click_command is None:
        msg = f"Unknown command: {command}"
        raise typer.BadParameter(msg)

    with typer.Context(click_command, info_name=command, parent=root_ctx) as cmd_ctx:
        typer.echo(click_command.get_help(cmd_ctx))

    raise typer.Exit


@app.command()
def init(
    ctx: typer.Context,
    config_format: str | None = typer.Argument(
        None,
        metavar="FORMAT",
        help="Config file format to generate (json, yml, yaml).",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=False,
        help="Optional destination directory for the config file. Defaults to the current working directory.",
    ),
) -> None:
    """Create a new configuration template"""
    if config_format is None:
        typer.echo(ctx.get_help())
        raise typer.Exit

    normalized = config_format.strip().lower()
    if normalized not in {"json", "yml", "yaml"}:
        msg = "FORMAT must be one of: json, yml, yaml"
        raise typer.BadParameter(msg)

    output_dir = output_dir.expanduser() if output_dir is not None else None

    try:
        created_path = create_config(config_format=normalized, output_dir=output_dir)
    except ValueError as exc:
        # create_config raises ValueError for invalid format/output_dir issues.
        raise typer.BadParameter(str(exc)) from exc

    display_path = _format_path_for_display(created_path)

    output_line = (
        typer.style("Configuration file written to:", bold=True)
        + " "
        + typer.style(display_path, fg=typer.colors.CYAN)
    )
    typer.echo(output_line)


@app.command()
def schema(
    ctx: typer.Context,
    config: Path | None = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        readable=True,
        dir_okay=False,
        metavar="CONFIG",
        help="Path to the configuration file.",
    ),
) -> None:
    """Generate only the schema"""
    if config is None:
        typer.echo(ctx.get_help())
        raise typer.Exit

    config = config.expanduser()
    _validate_config_path(config)
    print_ascii_header()
    _run_schema(config)


@app.command()
def extract(
    ctx: typer.Context,
    ontology: Path | None = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        readable=True,
        dir_okay=False,
        metavar="ONTOLOGY",
        help="Path to the ontology file (.ttl, .rdf, .owl, .xml).",
    ),
) -> None:
    """Extract ontology metadata into PyGraft JSON artefacts"""
    if ontology is None:
        typer.echo(ctx.get_help())
        raise typer.Exit

    ontology = ontology.expanduser()
    print_ascii_header()

    # Show status message before extraction
    typer.echo()
    typer.echo(
        typer.style(
            ">>> Extracting ontology... This may take up to 2 minutes, please wait.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    )
    typer.echo()

    namespaces_path, class_info_path, relation_info_path = extract_ontology(ontology)

    # ------------------------------------------------------------------
    # EXTRACTION OUTPUT
    # ------------------------------------------------------------------
    typer.echo()
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))
    typer.echo(
        typer.style(
            "EXTRACTION OUTPUT",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    )
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))
    typer.echo()

    typer.echo(
        typer.style("Namespaces information:", bold=True)
        + " "
        + typer.style(_format_path_for_display(namespaces_path), fg=typer.colors.CYAN)
    )
    typer.echo(
        typer.style("Class statistics:", bold=True)
        + " "
        + typer.style(_format_path_for_display(class_info_path), fg=typer.colors.CYAN)
    )
    typer.echo(
        typer.style("Relation statistics:", bold=True)
        + " "
        + typer.style(_format_path_for_display(relation_info_path), fg=typer.colors.CYAN)
    )

    typer.echo()

    # --- Echo copied ontology (schema.*) ---
    schema_ext = ".ttl" if ontology.suffix.lower() == ".ttl" else ".rdf"
    schema_path = namespaces_path.parent / f"schema{schema_ext}"

    typer.echo(
        typer.style("Ontology schema:", bold=True)
        + " "
        + typer.style(_format_path_for_display(schema_path), fg=typer.colors.CYAN)
    )

    typer.echo()
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))

    # ------------------------------------------------------------------
    # CONFIG FILE
    # ------------------------------------------------------------------
    typer.echo()

    config_path, config_created = ensure_and_update_config_for_ontology(
        ontology_file=ontology,
        class_info_path=class_info_path,
        relation_info_path=relation_info_path,
    )

    if config_created:
        typer.echo(
            typer.style(
                "No config file found — a new config file was created and updated "
                "based on the extracted ontology:",
                fg=typer.colors.RED,
                bold=True,
            )
        )
    else:
        typer.echo(
            typer.style(
                "Config file found and updated based on the extracted ontology:",
                fg=typer.colors.GREEN,
                bold=True,
            )
        )

    typer.echo(
        typer.style("Config file location:", bold=True)
        + " "
        + typer.style(_format_path_for_display(config_path), fg=typer.colors.CYAN)
    )

    # ------------------------------------------------------------------
    # CONFIGURATION GUIDANCE
    # ------------------------------------------------------------------
    typer.echo()
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))
    typer.echo(
        typer.style(
            "CONFIGURATION GUIDANCE",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    )
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))
    typer.echo()

    typer.echo(
        typer.style(
            "What you SHOULD edit:",
            fg=typer.colors.GREEN,
            bold=True,
        )
    )
    typer.echo(
        typer.style(
            "  • kg",
            fg=typer.colors.GREEN,
        )
    )
    typer.echo(
        "    Controls synthetic KG generation (size, density, typing, consistency).\n"
    )

    typer.echo()

    typer.echo(
        typer.style(
            "What you should edit with caution:",
            fg=typer.colors.MAGENTA,
            bold=True,
        )
    )
    typer.echo(
        typer.style(
            "  • general",
            fg=typer.colors.MAGENTA,
        )
    )
    typer.echo(
        "    - project_name: Must match the ontology-based folder name for 'kg' command.\n"
        "    - rdf_format: Must match the ontology format {'ttl', 'rdf'}.\n"
        "    - rng_seed: Null by default (stochastic generation), set to integer for deterministic."
    )
    typer.echo()

    typer.echo(
        typer.style(
            "Informational sections:",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )
    typer.echo(
        typer.style(
            "  • classes\n"
            "  • relations",
            fg=typer.colors.CYAN,
        )
    )
    typer.echo(
        "    These contain statistics extracted from the ontology.\n"
        "    They are descriptive only and do NOT influence KG generation."
    )

    typer.echo()

    # ------------------------------------------------------------------
    # NEXT STEPS
    # ------------------------------------------------------------------
    typer.echo()
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))
    typer.echo(
        typer.style(
            "NEXT STEPS",
            fg=typer.colors.YELLOW,
            bold=True,
        )
    )
    typer.echo(typer.style("-" * 72, fg=typer.colors.YELLOW))
    typer.echo()

    typer.echo(
        typer.style(
            "You can now generate synthetic knowledge graphs from this ontology.",
            fg=typer.colors.GREEN,
            bold=True,
        )
    )
    typer.echo()
    typer.echo(
        typer.style("Next command:", bold=True)
        + " "
        + typer.style(
            "pygraft kg " + _format_path_for_display(config_path),
            fg=typer.colors.CYAN,
        )
    )



@app.command()
def kg(
    ctx: typer.Context,
    config: Path | None = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        readable=True,
        dir_okay=False,
        metavar="CONFIG",
        help="Path to the configuration file.",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help=(
            "If the KG is inconsistent and consistency checking is enabled, "
            "run an additional Pellet explain pass to log inconsistency hints."
        ),
    ),
) -> None:
    """Generate only the Knowledge Graph (KG)"""
    if config is None:
        typer.echo(ctx.get_help())
        raise typer.Exit

    config = config.expanduser()
    _validate_config_path(config)
    print_ascii_header()
    _run_kg(config, explain_inconsistency=explain)


@app.command()
def build(
    ctx: typer.Context,
    config: Path | None = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        readable=True,
        dir_okay=False,
        metavar="CONFIG",
        help="Path to the configuration file.",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help=(
            "If the KG is inconsistent and consistency checking is enabled, "
            "run an additional Pellet explain pass to log inconsistency hints."
        ),
    ),
) -> None:
    """Generate both the schema and the Knowledge Graph (KG)"""
    if config is None:
        typer.echo(ctx.get_help())
        raise typer.Exit

    config = config.expanduser()
    _validate_config_path(config)
    print_ascii_header()

    _run_schema(config)
    _run_kg(config, explain_inconsistency=explain)





# ================================================================================================ #
# Local module execution                                                                           #
# ================================================================================================ #

if __name__ == "__main__":
    app()
