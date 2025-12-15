#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Query-loading utilities for ontology extraction.

This module provides a small, centralized API for retrieving SPARQL queries
from packaged resources. Queries are stored in organized folders according to
their extraction domain:

    * `pygraft.resources.queries.ontology.classes`
    * `pygraft.resources.queries.ontology.relations`

Each extraction step (e.g., class collection, relation mapping) uses a
dedicated `.rq` file stored in one of these domains. Callers load queries by
providing the exact filename, for example:

    load_class_query("classes.rq")
    load_relation_query("rel2dom.rq")

This module performs no filename normalization or suffix inference; the
provided filename must match the resource exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from typing import Any, Final, cast

# ============================================================================
# Resource roots
# ============================================================================
# Base resource directories where `.rq` files are stored.
_BASE_CLASSES: Final[Any] = files("pygraft.resources.queries.ontology.classes")
_BASE_RELATIONS: Final[Any] = files("pygraft.resources.queries.ontology.relations")

_QUERY_ENCODING: Final[str] = "utf-8"


# ============================================================================
# Public query loaders
# ============================================================================
def load_class_query(filename: str) -> str:
    """Load a `.rq` query from the classes query directory."""
    return _load_query_from_base(_BASE_CLASSES, filename)


def load_relation_query(filename: str) -> str:
    """Load a `.rq` query from the relations query directory."""
    return _load_query_from_base(_BASE_RELATIONS, filename)


# ============================================================================
# Small helper for SPARQL rows
# ============================================================================
def get_row_str(row: Any, key: str) -> str:
    """Extract a string value from an rdflib query result row.

    rdflib's result row type is not precisely typed, so we cast to a mapping
    for static type checking and then coerce the selected value to ``str``.
    """
    row_mapping = cast("Mapping[str, Any]", row)
    return str(row_mapping[key])


# ============================================================================
# Internal generic loader
# ============================================================================
def _load_query_from_base(base_dir: Any, filename: str) -> str:
    """Load a SPARQL query from a specified resource directory.

    Args:
        base_dir:
            The resource directory containing the `.rq` files.
        filename:
            Exact name of the query file, including the `.rq` suffix.

    Returns:
        The text of the SPARQL query.

    Raises:
        FileNotFoundError: If the query resource does not exist in the directory.
    """
    query_path = base_dir / filename

    if not query_path.is_file():
        msg = f"SPARQL query resource not found: {filename!r} in {base_dir!r}"
        raise FileNotFoundError(msg)

    return query_path.read_text(encoding=_QUERY_ENCODING)
