from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Public constant: default root directory for all PyGraft output artefacts.
OUTPUT_ROOT: Path = Path("pygraft_output")


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
            Either "auto" or a user-defined project name string. The value
            is assumed to have been validated and slugified by config.py.
        mode:
            Either "schema" (fresh or explicit schema run) or "kg" (reuse an
            existing schema run for KG generation).
        output_root:
            Optional base directory for all runs. When None, the global
            OUTPUT_ROOT ("pygraft_output" under the current working
            directory) is used. This parameter is intended for Python API
            callers; the CLI always relies on OUTPUT_ROOT.

    Returns:
        The final project folder name to use under the chosen output_root.

    Raises:
        ValueError: If mode is not "schema" or "kg", or if project_name is
            "auto" in KG mode and no previous run can be found.
    """
    base_root = output_root or OUTPUT_ROOT

    if mode == "schema":
        # "auto" -> generate a new timestamp-based folder name.
        if project_name == "auto":
            run_name = _generate_timestamp_run_name()
        else:
            run_name = project_name

        _initialize_folder(base_root, run_name)
        return run_name

    if mode == "kg":
        # "auto" -> reuse the most recent run folder under base_root.
        if project_name == "auto":
            latest = _get_most_recent_subfolder(base_root)
            if latest is None:
                raise ValueError(
                    "project_name='auto' but no previous PyGraft output folder exists. "
                    f"Checked under {base_root}. Generate a schema first or specify "
                    "an explicit project_name."
                )
            return latest

        # Explicit project name: do not create anything here; KG reuses an existing schema run.
        return project_name

    raise ValueError(f"Unknown mode {mode!r}. Expected 'schema' or 'kg'.")


# ========================================================================== #
# INTERNAL HELPERS (private)                                                 #
# ========================================================================== #

def _generate_timestamp_run_name() -> str:
    """Return a sortable timestamp run name, e.g., '2025-12-05_13-22-44'."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


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
