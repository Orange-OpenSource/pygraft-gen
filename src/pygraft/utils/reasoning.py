"""Reasoning utilities for PyGraft.

This module provides a thin, typed wrapper around Owlready2's HermiT
integration. It is responsible for:

- Loading an ontology from disk.
- Running the HermiT reasoner via Owlready2.
- Logging consistency results and surfacing inconsistencies as exceptions.

Design
------
- Function-oriented: no classes are introduced here on purpose.
- Internal-only: this module is used by schema and KG orchestration code,
  but is not part of the public `pygraft` API.
- Deterministic given the ontology file; no randomness is involved.
- Uses the shared logging infrastructure; no direct printing.

Performance
-----------
The dominant cost is the underlying HermiT reasoning process, which depends
on ontology size and expressivity. This wrapper adds negligible overhead
beyond:

- Ontology loading via Owlready2.
- A single call to `sync_reasoner_hermit`.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any, cast

import owlready2
from rdflib import Graph as RDFGraph

logger = logging.getLogger(__name__)

# ================================================================================================ #
# Owlready2 / HermiT Type Aliases                                                                  #
# ================================================================================================ #

OntologyHandle = Any

GetOntologyCallable = Callable[[str], OntologyHandle]
get_ontology: GetOntologyCallable = cast(GetOntologyCallable, owlready2.get_ontology)

SyncReasonerCallable = Callable[..., None]
sync_reasoner_hermit: SyncReasonerCallable = cast(
    SyncReasonerCallable,
    owlready2.sync_reasoner_hermit,
)

OwlReadyInconsistentOntologyError = owlready2.OwlReadyInconsistentOntologyError


# ================================================================================================ #
# Public Reasoner Helper                                                                           #
# ================================================================================================ #


def reasoner(
    *,
    schema_file: str | Path,
    kg_file: str | Path | None = None,
    infer_property_values: bool = False,
    debug: bool = False,
    keep_tmp_file: bool = False,
) -> None:
    """Run the HermiT reasoner on a schema or on a schema+KG combination.

    The input files are normalized to RDF/XML only when they are not already
    in RDF/XML format (e.g., `.ttl`, `.nt`). In schema+KG mode, the schema and
    the KG are merged into a temporary RDF/XML ontology used solely for
    reasoning. Temporary files are deleted unless `keep_tmp_file` is True.

    Args:
        schema_file: Path to the schema ontology.
        kg_file: Optional path to a KG generated from the schema.
        infer_property_values: Whether to infer property values during reasoning.
        debug: Enable Owlready2 internal debugging.
        keep_tmp_file: Preserve the temporary RDF/XML file used for reasoning.

    Raises:
        OwlReadyInconsistentOntologyError: If the schema or schema+KG is inconsistent.
    """
    schema_path = Path(schema_file).resolve()
    temp_to_cleanup: Path | None = None

    if kg_file is None:
        # Schema-only: if we know it is RDF/XML (config format="xml"), no need
        # to re-serialize. Otherwise, normalize to a temporary RDF/XML file.
        if schema_path.suffix == ".rdf":
            ontology_path = schema_path
            logger.debug(
                "Selected RDF/XML schema file for reasoning from: %s",
                ontology_path,
            )
        else:
            temp_to_cleanup = _build_temp_schema_graph(schema_path)
            ontology_path = temp_to_cleanup

        resource_label = "schema"
    else:
        # Schema + KG mode
        kg_path = Path(kg_file).resolve()
        temp_to_cleanup = _build_temp_schema_kg_graph(schema_path, kg_path)
        ontology_path = temp_to_cleanup
        resource_label = "KG"

    graph: OntologyHandle = get_ontology(str(ontology_path)).load()
    logger.debug("Loaded ontology for reasoning from: %s", ontology_path)

    try:
        sync_reasoner_hermit(
            graph,
            infer_property_values=infer_property_values,
            debug=debug,
            keep_tmp_file=keep_tmp_file,
        )
        logger.info("(HermiT) Consistent %s", resource_label)
    except OwlReadyInconsistentOntologyError:
        logger.exception("(HermiT) Inconsistent %s", resource_label)
        raise
    finally:
        graph.destroy()

        if temp_to_cleanup is not None and not keep_tmp_file:
            try:
                temp_to_cleanup.unlink(missing_ok=True)
                logger.debug(
                    "Deleted temporary reasoner input file: %s",
                    temp_to_cleanup,
                )
            except OSError:
                logger.warning(
                    "Failed to delete temporary reasoner input file: %s",
                    temp_to_cleanup,
                )

# ================================================================================================ #
# Internal helpers                                                                                 #
# ================================================================================================ #

def _build_temp_schema_graph(schema_file: Path) -> Path:
    """Create a temporary RDF/XML file for schema-only reasoning.

    This helper uses rdflib to parse the original schema file, regardless
    of its on-disk serialization (xml, ttl, nt, ...), and re-serializes it
    as RDF/XML. Owlready2 then loads this temporary file for HermiT.

    Args:
        schema_file: Path to the schema ontology file.

    Returns:
        Path to the temporary RDF/XML file that should be given to Owlready2.
    """
    graph = RDFGraph()
    graph.parse(str(schema_file))

    tmp_path = schema_file.with_name("tmp_schema_reasoner.rdf").resolve()
    graph.serialize(str(tmp_path), format="xml")
    logger.debug(
        "Serialized temporary schema graph for reasoning\n"
        "    from: %s\n"
        "      to: %s",
        schema_file.resolve(),
        tmp_path,
    )

    return tmp_path


def _build_temp_schema_kg_graph(schema_file: Path, kg_file: Path) -> Path:
    """Create a temporary merged schema+KG graph on disk.

    The merged graph is written next to the KG file and is intended for
    short-lived use as a reasoner input. The graph is always serialized
    as RDF/XML so that Owlready2 can reliably load it, regardless of the
    original schema/KG formats.

    Args:
        schema_file: Path to the schema ontology file.
        kg_file: Path to the KG ontology file.

    Returns:
        Path to the temporary RDF/XML file that should be given to Owlready2.
    """
    graph = RDFGraph()
    graph.parse(str(schema_file))
    graph.parse(str(kg_file))

    tmp_path = (kg_file.parent / "tmp_schema_kg.rdf").resolve()
    graph.serialize(str(tmp_path), format="xml")
    logger.debug(
        "Serialized temporary merged schema+KG graph for reasoning\n"
        "    from schema: %s\n"
        "        from KG: %s\n"
        "             to: %s",
        schema_file.resolve(),
        kg_file.resolve(),
        tmp_path,
    )

    return tmp_path
