#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Configuration template helpers for PyGraft.

This module provides a single function to create a starter PyGraft
configuration file by copying a packaged template.

The configuration file is always named either:
- pygraft.config.json
- pygraft.config.yml
"""

from __future__ import annotations

from importlib import resources
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
# Constants                                                                                        #
# ------------------------------------------------------------------------------------------------ #
PYGRAFT_CONFIG_JSON_FILENAME = "pygraft.config.json"
PYGRAFT_CONFIG_YML_FILENAME = "pygraft.config.yml"

# ------------------------------------------------------------------------------------------------ #
# API                                                                                              #
# ------------------------------------------------------------------------------------------------ #

def create_config(*, config_format: str = "json", output_dir: Path | str | None = None) -> Path:
    """Create a PyGraft configuration file by copying a packaged template.

    Args:
        config_format:
            Configuration format. Must be one of: "json", "yml", "yaml".
            Defaults to "json".
        output_dir:
            Destination directory. If None, the current working directory
            is used. The configuration filename is fixed and cannot be
            overridden.

    Returns:
        Path to the created configuration file.

    Raises:
        ValueError:
            If the format is invalid or output_dir is not a directory.
    """
    normalized = config_format.strip().lower()

    if normalized == "json":
        template_name = PYGRAFT_CONFIG_JSON_FILENAME
    elif normalized in {"yml", "yaml"}:
        template_name = PYGRAFT_CONFIG_YML_FILENAME
    else:
        msg = "config_format must be one of: 'json', 'yml', 'yaml'"
        raise ValueError(msg)

    target_dir = Path.cwd() if output_dir is None else Path(output_dir)

    if target_dir.exists() and not target_dir.is_dir():
        msg = f"output_dir must be a directory: {target_dir}"
        raise ValueError(msg)

    target_dir.mkdir(parents=True, exist_ok=True)

    src = resources.files("pygraft") / "resources" / "templates" / template_name
    dst = target_dir / template_name

    shutil.copy(str(src), str(dst))
    logger.info("Created configuration file at: %s", dst)

    return dst
