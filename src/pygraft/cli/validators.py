#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""CLI validators for PyGraft."""

from __future__ import annotations

from pathlib import Path

import typer


def validate_config_extension(config: Path) -> None:
    """Ensure the configuration file has a supported extension.

    Args:
        config: Path to the configuration file.

    Raises:
        typer.BadParameter: If the extension is not .json, .yaml, or .yml.
    """
    ext = config.suffix.lower()
    if ext not in {".json", ".yaml", ".yml"}:
        raise typer.BadParameter("Config file must end with .json, .yaml, or .yml.")
