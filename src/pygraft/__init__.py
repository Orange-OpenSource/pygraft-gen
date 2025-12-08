"""Top-level package for Pygraft."""

from __future__ import annotations

import logging
import importlib.metadata as importlib_metadata

from .pygraft import (
    create_template,
    create_json_template,
    create_yaml_template,
    generate_schema,
    generate_kg,
)


__all__ = [
    "create_template",
    "create_json_template",
    "create_yaml_template",
    "generate_schema",
    "generate_kg",
]

try:
    __version__ = importlib_metadata.version("pygraft")
except importlib_metadata.PackageNotFoundError:
    __version__ = "unknown"


logging.getLogger(__name__).addHandler(logging.NullHandler())
