#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Top-level package for Pygraft."""

from __future__ import annotations

import logging
import importlib.metadata as importlib_metadata

from pygraft.pygraft import (
    create_config,
    generate_schema,
    extract_ontology,
    generate_kg,
)


__all__ = [
    "create_config",
    "generate_schema",
    "extract_ontology",
    "generate_kg",
]

try:
    __version__ = importlib_metadata.version("pygraft")
except importlib_metadata.PackageNotFoundError:
    __version__ = "unknown"


logging.getLogger(__name__).addHandler(logging.NullHandler())
