#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""TODO: fill module docstring once finished but not now."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rdflib import Graph

from pygraft.ontology_extraction.classes import build_extracted_class_info
from pygraft.ontology_extraction.namespaces import build_namespaces_dict
from pygraft.ontology_extraction.relations import build_extracted_relation_info

if TYPE_CHECKING:
    from pygraft.ontology_extraction.namespaces import NamespaceInfoDict
    from pygraft.types import ClassInfoDict, RelationInfoDict

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
# Constants                                                                                        #
# ------------------------------------------------------------------------------------------------ #


# ================================================================================================ #
# Extraction Pipeline                                                                              #
# ================================================================================================ #
def ontology_extraction_pipeline(
    ontology_path: Path,
) -> tuple[NamespaceInfoDict, ClassInfoDict, RelationInfoDict]:
    """Extract high-level structural metadata from an ontology.

    This function is the public entry point for ontology extraction.
    It loads the ontology into memory and derives JSON-serializable
    metadata describing its namespaces, classes, and relations.

    The extraction process is intentionally read-only and schema-focused:
    no inferencing, reasoning, or transformation of the ontology is performed.

    Supported ontology formats:
        - Turtle (.ttl)
        - RDF/XML (.rdf, .owl, .xml)

    Args:
        ontology_path:
            Path to the ontology file to extract metadata from.

    Returns:
        A tuple containing:
            - namespaces_info:
                Extracted namespace prefix mappings.
            - class_info:
                Extracted class metadata and statistics.
            - relation_info:
                Extracted relation metadata and statistics.

        All returned objects are JSON-serializable.

    Raises:
        FileNotFoundError:
            If the ontology file does not exist.
        ValueError:
            If the ontology format is not supported.
    """
    # --- Load Graph
    graph: Graph = load_ontology_graph(ontology_path)
    logger.debug("Loaded ontology graph from: %s", ontology_path.resolve())

    # --- Namespaces
    namespaces_info: NamespaceInfoDict = build_namespaces_dict(graph)

    # --- Classes
    class_info: ClassInfoDict = build_extracted_class_info(
        graph=graph,
        namespaces=namespaces_info,
    )
    # --- Relations
    relation_info: RelationInfoDict = build_extracted_relation_info(
        graph=graph,
        namespaces=namespaces_info,
    )

    return namespaces_info, class_info, relation_info


# ------------------------------------------------------------------------------------------------ #
# Ontology Loader                                                                                 #
# ------------------------------------------------------------------------------------------------ #
def load_ontology_graph(ontology_path: Path) -> Graph:
    """Load an ontology file into an in-memory RDF graph.

    The loader is intentionally strict and minimal:
    it supports only Turtle and RDF/XML serializations and performs
    no inferencing, normalization, or post-processing beyond parsing.

    Supported formats:
        - Turtle (.ttl)
        - RDF/XML (.rdf, .owl, .xml)

    Unsupported formats (explicitly rejected):
        - N-Triples (.nt)
        - Any other RDF serialization

    Args:
        ontology_path:
            Path to the ontology file on disk.

    Returns:
        An rdflib.Graph instance containing the parsed ontology.

    Raises:
        FileNotFoundError:
            If the ontology file does not exist.
        ValueError:
            If the ontology file extension is not supported.
        rdflib.exceptions.ParserError:
            If the ontology file cannot be parsed by rdflib.
    """
    ontology_file = Path(ontology_path).resolve()
    if not ontology_file.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_file}")

    extension = ontology_path.suffix.lower()

    # Accepted ontology formats
    if extension == ".ttl":
        rdflib_format = "turtle"
    elif extension in {".rdf", ".owl", ".xml"}:
        rdflib_format = "xml"
    else:
        msg = (
            f"Unsupported ontology format '{extension}'. "
            "Only .ttl and RDF/XML (.rdf, .owl, .xml) are allowed for ontology loading."
        )
        raise ValueError(msg)

    graph: Graph = Graph()
    graph.parse(ontology_path.as_posix(), format=rdflib_format)
    return graph



# ================================================================================================ #
# Main (local testing only)                                                                        #
# ================================================================================================ #


def main() -> None:
    """Local smoke test for the ontology extraction pipeline.

    This function is for local testing only and is not part of the public API.
    Paths are computed relative to the project root, and the extracted
    artefacts are written as JSON files under "pygraft_output/extraction".
    """
    # extractionextraction.py --> ontology_extraction --> pygraft --> src --> ROOT
    project_root: Path = Path(__file__).resolve().parents[3]
    output_dir: Path = project_root / "pygraft_output" / "extraction"

    # Choose the ontology to test.
    ontology_path: Path = project_root / "ontologies" / "noria.ttl"

    # ------------------------------------------------------------------
    # 1) Pipeline
    # ------------------------------------------------------------------
    namespaces_info, class_info, relation_info = ontology_extraction_pipeline(
        ontology_path=ontology_path,
    )

    # Optional: write files here if you still want the local smoke test to emit JSON
    (output_dir / "namespaces_info.json").write_text(json.dumps(namespaces_info, indent=2),
                                                     encoding="utf-8")
    (output_dir / "class_info.json").write_text(json.dumps(class_info, indent=2), encoding="utf-8")
    (output_dir / "relation_info.json").write_text(json.dumps(relation_info, indent=2),
                                                   encoding="utf-8")

if __name__ == "__main__":
    # Configure a basic console logger only when this module is executed
    # directly, and only if no logging has been configured yet (for example
    # by the CLI or another application entry point).
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s %(name)s - %(message)s",
        )

    main()
