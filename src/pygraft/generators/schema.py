#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Schema construction utilities for PyGraft.

This module sits at the end of the schema-generation pipeline. It takes the
already computed `ClassInfoDict` and `RelationInfoDict` structures and is responsible
for:

- Persisting them as JSON metadata on disk.
- Building an OWL/RDF schema graph using rdflib.
- Serializing the schema in the user-requested rdf_format.
- Optionally running a reasoner over the resulting schema.

Architecture
------------
The module is intentionally split into three focused components:

- `SchemaBuilderConfig`: Immutable configuration describing where and how the
  schema should be written.
- `SchemaIO`: A small service object that owns folder creation and JSON I/O.
- `SchemaGraphBuilder`: A pure graph builder that translates `ClassInfoDict` and
  `RelationInfoDict` into an rdflib `Graph`.
- `SchemaBuilder`: A high-level orchestrator that wires everything together
  and exposes a single main entry point (`build_schema()`).

External dependencies such as rdflib and the PyGraft `reasoner` helper are
isolated behind these classes so that the rest of the codebase can reason
about schema construction at a higher level.

Public vs internal status
-------------------------
`SchemaBuilderConfig` and `SchemaBuilder` are intended to be part of the
library's internal-but-stable surface and are used by the main `pygraft`
orchestration module. `SchemaIO` and `SchemaGraphBuilder` are internal helpers
and may evolve over time.

Performance characteristics
---------------------------
The dominant cost in this module is I/O:

- JSON writes for `class_info` and `relation_info`.
- rdflib graph construction and serialization.
- Optional reasoner invocation (which may be expensive depending on the
  underlying implementation).

Let:
- n = number of classes
- m = number of relations

Then:

- Graph construction runs in O(n + m) for the number of nodes and edges.
- Memory usage is dominated by the rdflib `Graph`, which is typically O(n + m)
  in the number of triples.

No randomness is introduced here; behavior is deterministic given the input
`ClassInfoDict`, `RelationInfoDict`, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef
from tqdm.auto import tqdm

from pygraft.paths import OUTPUT_ROOT

if TYPE_CHECKING:  # pragma: no cover - import-time type hints only
    from pygraft.types import ClassInfoDict, RelationInfoDict

logger = logging.getLogger(__name__)


# ================================================================================================ #
# Configuration                                                                                    #
# ================================================================================================ #


@dataclass(frozen=True)
class SchemaBuilderConfig:
    """Immutable configuration for the schema builder.

    Attributes:
        folder_name:
            Name of the output folder under the chosen PyGraft output root.
            This is usually the validated general.project_name value.
        rdf_format:
            Serialization format for the RDF graph (e.g. "xml", "ttl", "nt").
        schema_namespace:
            Base IRI for the schema's namespace.
        output_root:
            Optional base directory for schema artefacts. When None, the
            default OUTPUT_ROOT ("pygraft_output" under the current working
            directory) is used.
    """

    folder_name: str
    rdf_format: str
    schema_namespace: str = "http://pygraf.t/"
    output_root: Path | None = None

    def __post_init__(self) -> None:
        """User config is validated in config.py; this checks minimal prerequisites for writing the schema."""
        if not self.folder_name:
            message = "folder_name must be a non-empty string."
            raise ValueError(message)

        if not self.rdf_format:
            message = "rdf_format must be a non-empty string."
            raise ValueError(message)

        if not self.schema_namespace:
            message = "schema_namespace must be a non-empty string."
            raise ValueError(message)


# ================================================================================================ #
# I/O Helper                                                                                       #
# ================================================================================================ #


class SchemaIO:
    """Handle filesystem concerns for schema construction.

    This helper is responsible for:
        - Binding to the output folder under the configured output root.
        - Writing `class_info` and `relation_info` JSON artefacts.
        - Resolving the final schema file path.

    It does not know anything about rdflib or OWL; it only deals with paths
    and simple JSON serialization.
    """

    def __init__(self, *, config: SchemaBuilderConfig) -> None:
        """Initialize the I/O helper with a configuration object.

        Args:
            config: Immutable configuration describing folder and rdf_format.
        """
        self._config: SchemaBuilderConfig = config
        self._directory: Path | None = None
        self._output_root: Path = config.output_root or OUTPUT_ROOT


    # ------------------------------------------------------------------------------------------------ #
    # Public API                                                                                       #
    # ------------------------------------------------------------------------------------------------ #

    @property
    def directory(self) -> Path:
        """Directory where all schema artefacts are written.

        Raises:
            RuntimeError: If the output directory has not been initialized yet.
        """
        if self._directory is None:
            message = (
                "Output directory has not been initialized. Call "
                "initialize_output_folder() before accessing directory."
            )
            raise RuntimeError(message)
        return self._directory

    def initialize_output_folder(self) -> Path:
        """Bind to the output folder for this schema run.

        The folder layout is:

            <output_root>/<folder_name>/

        The folder is expected to have been created by higher-level
        orchestration (via resolve_project_folder). If it does not exist,
        this is treated as a programming error.

        Returns:
            The path to the directory.

        Raises:
            RuntimeError: If the expected directory does not exist.
        """
        if not self._config.folder_name:
            message = "SchemaBuilderConfig.folder_name must be a non-empty string."
            raise ValueError(message)

        directory = self._output_root / self._config.folder_name

        if not directory.exists():
            message = (
                f"Schema output directory {directory} does not exist. "
                "Call resolve_project_folder(..., mode='schema') before "
                "invoking SchemaBuilder.build_schema()."
            )
            raise RuntimeError(message)

        self._directory = directory
        return directory

    def write_relation_info(self, relation_info: RelationInfoDict) -> Path:
        """Write `relation_info` JSON to disk.

        The method ensures that any non-JSON-friendly values (such as sets in
        `rel2patterns`) are converted into lists.

        Args:
            relation_info: Relation metadata produced by RelationGenerator.

        Returns:
            Path to the written relation_info.json file.
        """
        if self._directory is None:
            message = (
                "Output directory has not been initialized. Call "
                "initialize_output_folder() before writing relation_info."
            )
            raise RuntimeError(message)

        output_path = (self._directory / "relation_info.json").resolve()

        serializable_relation_info: dict[str, object] = dict(relation_info)
        rel2patterns = relation_info["rel2patterns"]
        serializable_relation_info["rel2patterns"] = {
            relation_id: sorted(patterns) for relation_id, patterns in rel2patterns.items()
        }

        with output_path.open("w", encoding="utf8") as file:
            json.dump(serializable_relation_info, file, indent=4)

        logger.debug("Wrote relation_info to: %s", output_path)
        return output_path

    def write_class_info(self, class_info: ClassInfoDict) -> Path:
        """Write `class_info` JSON to disk.

        Args:
            class_info: Class metadata produced by ClassGenerator.

        Returns:
            Path to the written class_info.json file.
        """
        if self._directory is None:
            message = (
                "Output directory has not been initialized. Call "
                "initialize_output_folder() before writing class_info."
            )
            raise RuntimeError(message)

        output_path = (self._directory / "class_info.json").resolve()

        with output_path.open("w", encoding="utf8") as file:
            json.dump(class_info, file, indent=4)

        logger.debug("Wrote class_info to: %s", output_path)
        return output_path

    def get_schema_file_path(self) -> Path:
        """Return the path where the RDF schema should be serialized.

        Returns:
            Path to the schema serialization file.

        Raises:
            RuntimeError: If the output directory has not been initialized yet.
        """
        if self._directory is None:
            message = (
                "Output directory has not been initialized. Call "
                "initialize_output_folder() before requesting schema path."
            )
            raise RuntimeError(message)

        extension = "rdf" if self._config.rdf_format == "xml" else self._config.rdf_format
        return (self._directory / f"schema.{extension}").resolve()


# ================================================================================================ #
# Graph Builder                                                                                    #
# ================================================================================================ #


class SchemaGraphBuilder:
    """Build an rdflib graph from class and relation metadata.

    This class is responsible for taking the high-level `ClassInfoDict` and
    `RelationInfoDict` structures and constructing an OWL/RDF graph that encodes:

        - Classes, subclass axioms, and disjointness.
        - Object properties, their logical characteristics, domains, and ranges.
        - Inverse-of and subPropertyOf relations.
        - Ontology and license metadata.

    It is deliberately free of filesystem concerns and does not perform any
    serialization to disk. Callers are expected to serialize the resulting
    `Graph` themselves.

    Performance
    -----------
    Let n = number of classes, m = number of relations.

    - Graph construction touches each class and relation once and adds a small
      constant number of triples per item. Overall complexity is O(n + m).
    - Memory usage is proportional to the number of generated triples.
    """

    def __init__(
        self,
        *,
        class_info: ClassInfoDict,
        relation_info: RelationInfoDict,
        schema_namespace: str,
    ) -> None:
        """Initialize the graph builder.

        Args:
            class_info: Class metadata from the ClassGenerator.
            relation_info: Relation metadata from the RelationGenerator.
            schema_namespace: Base IRI used for schema entities.
        """
        self._class_info: ClassInfoDict = class_info
        self._relation_info: RelationInfoDict = relation_info
        self._schema_namespace: str = schema_namespace

    # ------------------------------------------------------------------------------------------------ #
    # Public API                                                                                       #
    # ------------------------------------------------------------------------------------------------ #

    def build_graph(self) -> Graph:
        """Construct and return the schema graph.

        Returns:
            An rdflib `Graph` containing the OWL schema.
        """
        graph = Graph()

        owl_ns = Namespace("http://www.w3.org/2002/07/owl")
        rdf_ns = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns")
        rdfs_ns = Namespace("http://www.w3.org/2000/01/rdf-schema")
        schema_ns = Namespace(self._schema_namespace)

        graph.bind("owl", owl_ns)
        graph.bind("rdf", rdf_ns)
        graph.bind("rdfs", rdfs_ns)
        graph.bind("sc", schema_ns)

        ontology_uri = URIRef(schema_ns)
        graph.add((ontology_uri, RDF.type, OWL.Ontology))

        cc0_license_uri = URIRef("https://creativecommons.org/publicdomain/zero/1.0/")
        graph.add(
            (
                ontology_uri,
                URIRef("http://purl.org/dc/terms/license"),
                cc0_license_uri,
            ),
        )

        self._add_classes(graph, schema_ns)
        self._add_relations(graph, schema_ns)

        return graph

    # ------------------------------------------------------------------------------------------------ #
    # Internal Helpers                                                                                 #
    # ------------------------------------------------------------------------------------------------ #

    def _add_classes(self, graph: Graph, schema_ns: Namespace) -> None:
        """Populate the graph with class declarations and axioms."""
        classes = self._class_info["classes"]
        class2superclasses = self._class_info["direct_class2superclasses"]
        class2disjoints = self._class_info["class2disjoints"]

        for class_name in tqdm(classes, desc="Writing classes", unit="classes", colour="red"):
            class_uri = schema_ns[class_name]
            graph.add((class_uri, RDF.type, OWL.Class))

            # Subclass relations: emit one triple per direct superclass.
            for superclass_name in class2superclasses.get(class_name, []):
                if superclass_name == "owl:Thing":
                    graph.add((class_uri, RDFS.subClassOf, OWL.Thing))
                else:
                    superclass_uri = schema_ns[superclass_name]
                    graph.add((class_uri, RDFS.subClassOf, superclass_uri))

            # Disjointness axioms
            if class_name in class2disjoints:
                for disjoint_name in class2disjoints[class_name]:
                    disjoint_uri = schema_ns[disjoint_name]
                    graph.add((class_uri, OWL.disjointWith, disjoint_uri))

    def _add_relations(self, graph: Graph, schema_ns: Namespace) -> None:
        """Populate the graph with object properties and their axioms."""
        relations = self._relation_info["relations"]
        rel2patterns = self._relation_info["rel2patterns"]

        # INFO:
        # Domains and ranges are stored as list[str] for OWL compatibility.
        # At this stage of the pipeline, exactly ONE domain and ONE range
        # per relation are expected because we only generate one.
        # This is enforced explicitly here.

        rel2dom_raw = self._relation_info["rel2dom"]  # dict[str, list[str]]
        rel2range_raw = self._relation_info["rel2range"]  # dict[str, list[str]]

        rel2dom: dict[str, str] = {}
        for rel, dom_list in rel2dom_raw.items():
            if not dom_list:
                continue

            if len(dom_list) != 1:
                message = (
                    "Expected exactly one domain for relation "
                    f"{rel!r}, but got {len(dom_list)}: {dom_list!r}"
                )
                raise ValueError(message)

            rel2dom[rel] = dom_list[0]

        rel2range: dict[str, str] = {}
        for rel, range_list in rel2range_raw.items():
            if not range_list:
                continue

            if len(range_list) != 1:
                message = (
                    "Expected exactly one range for relation "
                    f"{rel!r}, but got {len(range_list)}: {range_list!r}"
                )
                raise ValueError(message)

            rel2range[rel] = range_list[0]


        rel2inverse = self._relation_info["rel2inverse"]
        rel2superrel = self._relation_info["rel2superrel"]

        for relation_name in tqdm(
            relations,
            desc="Writing relations",
            unit="relations",
            colour="red",
        ):
            relation_uri = schema_ns[relation_name]
            graph.add((relation_uri, RDF.type, OWL.ObjectProperty))

            patterns = rel2patterns.get(relation_name, set())

            # Logical property types
            for object_property in patterns:
                if object_property == "owl:Symmetric":
                    graph.add((relation_uri, RDF.type, OWL.SymmetricProperty))
                elif object_property == "owl:Asymmetric":
                    graph.add((relation_uri, RDF.type, OWL.AsymmetricProperty))
                elif object_property == "owl:Reflexive":
                    # https://oborel.github.io/obo-relations/reflexivity/:
                    # "Reflexivity is incompatible with domain and range assertions."
                    if relation_name in rel2dom and relation_name in rel2range:
                        # Reflexivity ok only if domain == range
                        if rel2dom[relation_name] == rel2range[relation_name]:
                            graph.add((relation_uri, RDF.type, OWL.ReflexiveProperty))
                    else:
                        graph.add((relation_uri, RDF.type, OWL.ReflexiveProperty))
                elif object_property == "owl:Irreflexive":
                    graph.add((relation_uri, RDF.type, OWL.IrreflexiveProperty))
                elif object_property == "owl:Transitive":
                    graph.add((relation_uri, RDF.type, OWL.TransitiveProperty))
                elif object_property == "owl:Functional":
                    graph.add((relation_uri, RDF.type, OWL.FunctionalProperty))
                elif object_property == "owl:InverseFunctional":
                    graph.add((relation_uri, RDF.type, OWL.InverseFunctionalProperty))

            # Domain / range (skip reflexive)
            if relation_name in rel2dom and "owl:Reflexive" not in patterns:
                graph.add((relation_uri, RDFS.domain, schema_ns[rel2dom[relation_name]]))

            if relation_name in rel2range and "owl:Reflexive" not in patterns:
                graph.add((relation_uri, RDFS.range, schema_ns[rel2range[relation_name]]))

            # inverseOf
            if relation_name in rel2inverse:
                graph.add((relation_uri, OWL.inverseOf, schema_ns[rel2inverse[relation_name]]))

            # subPropertyOf (single direct parent expected; stored as list[str] for OWL compatibility)
            if relation_name in rel2superrel:
                superrels = rel2superrel[relation_name]

                if len(superrels) != 1:
                    message = (
                        "Expected exactly one direct super-property for relation "
                        f"{relation_name!r}, but got {len(superrels)}: {superrels!r}"
                    )
                    raise ValueError(message)

                graph.add((relation_uri, RDFS.subPropertyOf, schema_ns[superrels[0]]))


# ================================================================================================ #
# Orchestrator                                                                                     #
# ================================================================================================ #


class SchemaBuilder:
    """High-level orchestrator for building and validating a schema.

    This class glues together:

    - `SchemaIO`: to handle on-disk artefacts (JSON + final RDF file).
    - `SchemaGraphBuilder`: to construct the in-memory rdflib graph.
    - The optional PyGraft `reasoner` invocation for schema checking.

    Typical usage from higher-level orchestration code is:

        config = SchemaBuilderConfig(folder_name="my_schema", rdf_format="ttl")
        builder = SchemaBuilder(config=config, class_info=class_info, relation_info=relation_info)
        builder.build_schema()

    Notes:
        - This class is not intended for inheritance.
        - Side effects (file I/O) are localized to `build_schema()`.
    """

    def __init__(
        self,
        *,
        config: SchemaBuilderConfig,
        class_info: ClassInfoDict,
        relation_info: RelationInfoDict,
        io_helper: SchemaIO | None = None,
        graph_builder: SchemaGraphBuilder | None = None,
    ) -> None:
        """Initialize the schema builder with configuration and inputs.

        Args:
            config: Immutable configuration parameters for this builder.
            class_info: Class metadata produced by the ClassGenerator.
            relation_info: Relation metadata produced by the RelationGenerator.
            io_helper: Optional preconfigured `SchemaIO` instance. If None, a
                default instance is created from `config`.
            graph_builder: Optional preconfigured `SchemaGraphBuilder` instance.
                If None, a default instance is created from `config`,
                `class_info`, and `relation_info`.
        """
        self._config: SchemaBuilderConfig = config
        self._class_info: ClassInfoDict = class_info
        self._relation_info: RelationInfoDict = relation_info

        self._io: SchemaIO = io_helper if io_helper is not None else SchemaIO(config=config)
        self._graph_builder: SchemaGraphBuilder = (
            graph_builder
            if graph_builder is not None
            else SchemaGraphBuilder(
                class_info=class_info,
                relation_info=relation_info,
                schema_namespace=config.schema_namespace,
            )
        )

    # ------------------------------------------------------------------------------------------------ #
    # Public API                                                                                       #
    # ------------------------------------------------------------------------------------------------ #

    def __repr__(self) -> str:
        """Return a debug-friendly representation for this builder."""
        return (
            "SchemaBuilder("
            f"folder_name={self._config.folder_name!r}, "
            f"rdf_format={self._config.rdf_format!r}, "
            f"schema_namespace={self._config.schema_namespace!r}"
            ")"
        )

    def build_schema(self) -> Path:
        """Run the full schema construction pipeline.

        This method:

        1. Initializes the output directory.
        2. Writes `class_info` and `relation_info` as JSON.
        3. Builds the rdflib graph and serializes it to disk.

        Returns:
            Path to the serialized schema file.

        Raises:
            RuntimeError: If any internal invariant is violated (e.g., output
                directory not initialized) during the build process.
        """
        self._io.initialize_output_folder()

        self._io.write_class_info(self._class_info)
        self._io.write_relation_info(self._relation_info)

        graph = self._graph_builder.build_graph()
        schema_file = self._io.get_schema_file_path()
        graph.serialize(schema_file, format=self._config.rdf_format)

        logger.info("Serialized schema graph to: %s", schema_file)
        return schema_file

