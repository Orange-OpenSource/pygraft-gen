#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""PyGraft: Configurable generation of Schemas & Knowledge Graphs.

PyGraft provides APIs for generating synthetic RDF knowledge graphs
from OWL ontologies with configurable size, typing, and constraint
enforcement.

**Exported Functions:**

- `create_config`: Create a validated configuration dictionary.
- `generate_schema`: Generate OWL schema (classes + relations).
- `extract_ontology`: Extract metadata from existing ontology.
- `generate_kg`: Generate instance-level KG triples.
- `explain_kg`: Analyze KG for logical inconsistencies.

**Exported Types:**

- `ClassInfoDict`: Type for class_info.json structure.
- `RelationInfoDict`: Type for relation_info.json structure.
- `PyGraftConfigDict`: Type for configuration dictionary.
- `KGInfoDict`: Type for kg_info.json structure.
"""
from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging

# Core API
from pygraft.pygraft import (
    create_config,
    extract_ontology,
    generate_kg,
    generate_schema,
    explain_kg,
)

# Type definitions (for type-safe user code)
from pygraft.types import (
    ClassInfoDict,
    KGInfoDict,
    PyGraftConfigDict,
    RelationInfoDict,
)

__all__ = [
    # Functions
    "create_config",
    "generate_schema",
    "extract_ontology",
    "generate_kg",
    "explain_kg",
    # Types
    "ClassInfoDict",
    "RelationInfoDict",
    "PyGraftConfigDict",
    "KGInfoDict",
]

try:
    __version__ = importlib_metadata.version("pygraft")
except importlib_metadata.PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())
