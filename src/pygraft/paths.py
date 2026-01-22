#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Public constant: default root directory for all PyGraft output artefacts.
OUTPUT_ROOT: Path = Path("output_pygraft")


# ========================================================================== #
# PUBLIC API                                                                 #
# ========================================================================== #

def resolve_project_folder(
    project_name: str,
    *,
    mode: str,
    output_root: Path | None = None,
) -> str:
    """Resolve and (if needed) create the output folder for this PyGraft run.

    This is the single entry point for deciding where a project's artefacts
    live on disk.

    Args:
        project_name:
            Either "auto" or a user-defined project name string.
        mode:
            Either "schema" (fresh or explicit schema run) or "kg" (reuse an
            existing schema run for KG generation).
        output_root:
            Optional base directory for all runs. When None, the global
            OUTPUT_ROOT under the current working directory is used.

    Returns:
        The final project folder name to use under the chosen output_root.

    Raises:
        ValueError: If mode is invalid, or if project_name is "auto" in KG
            mode and no previous synthetic schema run exists.
    """
    base_root = output_root or OUTPUT_ROOT

    # Defensive normalization
    if project_name != "auto":
        project_name = slugify_project_name(project_name)

    if mode == "schema":
        # "auto" -> generate a new timestamp-based folder
        if project_name == "auto":
            run_name = _generate_timestamp_run_name()
        else:
            run_name = project_name

        _initialize_folder(base_root, run_name)
        return run_name

    if mode == "kg":
        # "auto" -> reuse most recent SYNTHETIC schema run (timestamped only)
        if project_name == "auto":
            latest = _find_latest_synthetic_schema(base_root)
            if latest is None:
                raise ValueError(
                    "project_name='auto' in KG mode requires a synthetic schema, but none found. "
                    f"Checked for timestamped folders under {base_root}.\n"
                    "Either:\n"
                    "  1. Run 'pygraft schema' with project_name='auto' first, or\n"
                    "  2. Specify an explicit project_name (e.g., 'noria', 'bot') for extracted ontologies."
                )
            logger.info("Auto-selected synthetic schema: %s", latest)
            return latest

        # Explicit name -> reuse existing schema folder (extracted ontology)
        return project_name

    raise ValueError(f"Unknown mode {mode!r}. Expected 'schema' or 'kg'.")


# ========================================================================== #
# INTERNAL HELPERS (private)                                                 #
# ========================================================================== #

def slugify_project_name(name: str) -> str:
    """Convert an arbitrary user string into a filesystem-safe project name.

    Rules:
        - Lowercase everything.
        - Remove accents/diacritics.
        - Replace all non-alphanumeric characters with underscores.
        - Collapse multiple underscores.
        - Strip leading/trailing underscores.
    """
    # Normalize Unicode -> remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))

    # Lowercase
    name = name.lower()

    # Replace non-alphanumeric with underscores
    name = re.sub(r"[^a-z0-9]+", "_", name)

    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)

    # Strip leading/trailing underscores
    return name.strip("_")

def _generate_timestamp_run_name() -> str:
    """Return a sortable timestamp run name, e.g., '2025-12-05_13-22-44'."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

def _find_latest_synthetic_schema(output_root: Path) -> str | None:
    """Find the most recently created synthetic (timestamped) schema folder.

    Only looks for folders matching the timestamp pattern: YYYY-MM-DD_HH-MM-SS
    Ignores named folders like 'noria', 'bot', 'foaf', etc.

    Args:
        output_root: Base directory containing schema folders.

    Returns:
        Name of the most recent synthetic schema folder, or None if not found.
    """
    if not output_root.exists():
        return None

    # Pattern to match timestamp folders: YYYY-MM-DD_HH-MM-SS
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')

    # Find all subdirectories that match the timestamp pattern
    synthetic_folders = [
        entry for entry in output_root.iterdir()
        if entry.is_dir() and timestamp_pattern.match(entry.name)
    ]

    if not synthetic_folders:
        return None

    # Return the most recently created one
    most_recent = max(synthetic_folders, key=lambda path: path.stat().st_ctime)
    return most_recent.name

def _initialize_folder(output_root: Path, folder_name: str) -> str:
    """Create or reuse a folder under output_root and return its name.

    Args:
        output_root: Base directory where project folders are created.
        folder_name: Name of the subfolder for this run.

    Returns:
        The folder_name, unchanged, for convenience.
    """
    directory = (output_root / folder_name).resolve()
    existed_before = directory.exists()
    directory.mkdir(parents=True, exist_ok=True)

    if existed_before:
        logger.info("Reused output folder at: %s", directory)
    else:
        logger.info("Created output folder at: %s", directory)

    return folder_name


def _get_most_recent_subfolder(folder_path: Path) -> str | None:
    """Return the name of the most recently created subfolder.

    Args:
        folder_path: Directory to inspect for subfolders.

    Returns:
        The name of the most recently created subfolder, or None if folder_path
        does not exist or contains no subdirectories.
    """
    base_path = folder_path

    if not base_path.exists():
        return None

    subfolders = [entry for entry in base_path.iterdir() if entry.is_dir()]
    if not subfolders:
        return None

    most_recent = max(subfolders, key=lambda path: path.stat().st_ctime)
    return most_recent.name
