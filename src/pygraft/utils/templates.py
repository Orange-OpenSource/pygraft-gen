"""Template file helpers for PyGraft.

This module contains utilities that copy the packaged example templates
into the current working directory. These helpers are thin wrappers
around importlib.resources and shutil and are used by the high-level
PyGraft API functions that create starter configuration files.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)


def copy_json_config_template() -> None:
    """Copy the packaged JSON configuration template into the CWD."""
    src = (
        resources.files("pygraft")
        / "resources"
        / "templates"
        / "pygraft_config.json"
    )
    dst = Path.cwd() / "pygraft_config.json"
    shutil.copy(str(src), dst)
    logger.info("Created configuration file at: %s", dst)


def copy_yaml_config_template() -> None:
    """Copy the packaged YAML configuration template into the CWD."""
    src = (
        resources.files("pygraft")
        / "resources"
        / "templates"
        / "pygraft_config.yml"
    )
    dst = Path.cwd() / "pygraft_config.yml"
    shutil.copy(str(src), dst)
    logger.info("Created configuration file at: %s", dst)
