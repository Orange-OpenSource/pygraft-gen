"""Instance Generator Module.

============================
This module implements PyGraft's internal instance generator for synthetic
knowledge graphs (KGs). It takes an already generated OWL-style project_name
(classes + relations) and produces instance triples (entities and links)
that respect:

- class hierarchy and disjointness
- domain / range constraints
- logical relation patterns (functional, inverse-functional, asymmetric,
  irreflexive, inverse-of, subproperty)

High-Level Purpose
------------------
The module's role is to build a KG that is statistically shaped according
to user configuration while remaining logically consistent with the
project_name. It is used for testing, benchmarking, and synthetic KG creation.

Main abstractions:

- InstanceGeneratorConfig:
    Immutable configuration object describing KG size, typing behavior,
    and generation heuristics (balance, multityping, etc.).

- InstanceGenerator:
    Orchestrator that:
      - loads `class_info.json` and `relation_info.json` for a project_name
      - generates entities and their class typing
      - creates triples guided by domain/range and relation patterns
      - optionally oversamples via logical inference (inverse-of,
        symmetry, subproperty)
      - writes the final KG file and a small JSON metadata summary

Randomness and Determinism
--------------------------
All randomness in this module goes through a private NumPy
`Generator` instance owned by `InstanceGenerator`. When a rng_seed is
provided in `InstanceGeneratorConfig`, generation is deterministic and
reproducible for the same configuration and project_name.

This module does not modify NumPy's global random state.

Module Invariants
-----------------
For a successful run:

- Entities receive class typings that are consistent with class
  disjointness axioms.
- Triples respect:
    * asymmetric / irreflexive constraints
    * domain / range compatibility
    * functional / inverse-functional constraints
- Optional inference-based oversampling does not violate the above.

Performance Summary
-------------------
Let:

- n_e = number of entities (`num_entities`)
- n_t = number of triples (`num_triples`)
- n_c = number of classes
- n_r = number of relations

Then:

- Typing:
    Entity typing is driven by hierarchy layers, disjointness, and
    optional multityping. It typically runs in O(n_e · log n_c) with
    vectorised NumPy pieces dominating the cost.

- Triple generation:
    Triples are generated relation-by-relation with rejection checks
    for constraints. In practice, this is roughly O(n_t) with small
    constant factors, but pathological configurations may require more
    attempts due to rejections.

- Oversampling:
    Optional logical oversampling (inverse-of, symmetry, subproperty)
    runs over observed triples for selected relations. This is usually
    O(n_t) and is guarded by iteration caps.

- Memory:
    The main structures are:
      * ent2classes_specific / ent2classes_transitive
      * class2entities / class2unseen
      * the KG itself (`self._kg`)
    Space usage is O(n_e + n_t) with additional O(n_c + n_r) project_name
    metadata.

Intended Use
------------
This module is intended to be called by higher-level orchestration code
after a project_name has been generated and serialized via the
`SchemaBuilder`. Typical usage:

    config = InstanceGeneratorConfig(...)
    generator = InstanceGenerator(config=config)
    kg_info, kg_file = generator.generate_kg()

The module does not parse CLI arguments and performs no logging via
`print`; it relies on the standard logging infrastructure.

This module is internal to PyGraft and is not part of the public, stable API surface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import copy
from dataclasses import dataclass
import itertools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from rdflib import RDF, Graph as RDFGraph, Namespace, URIRef
from tqdm.auto import tqdm

from pygraft.types import (
    ClassInfoDict,
    RelationInfoDict,
    Triple,
    build_kg_info,
    KGUserParameters,
    KGStatistics,

)
from pygraft.utils.kg import (
    generate_random_numbers,
    get_fast_ratio,
    inverse_inference,
    subproperty_inference,
    symmetric_inference,
)
from pygraft.paths import OUTPUT_ROOT

if TYPE_CHECKING:
    from pygraft.types import EntityId, KGInfoDict, RelationId, TripleSet

logger = logging.getLogger(__name__)


# ================================================================================================ #
# Configuration                                                                                    #
# ================================================================================================ #


@dataclass(frozen=True)
class InstanceGeneratorConfig:
    """Immutable configuration for the instance generator.

    Attributes:
        project_name: Name of the schema folder under "output/". The generator
            expects to find `class_info.json`, `relation_info.json`, and
            the schema serialization there.
        rdf_format: Serialization format of the schema (e.g. "xml", "ttl").
        rng_seed: Optional RNG seed for deterministic behavior.

        num_entities: Target number of entities in the KG.
        num_triples: Target number of instance triples in the KG.

        enable_fast_generation: Whether to accelerate instance generation by
            generating a smaller prototype set and scaling it up.
        enable_inference_oversampling: Whether logical inferences (inverse-of,
            symmetry, subproperty) may be used to oversample triples.

        relation_usage_uniformity: Controls how evenly triples are
            distributed across relations. Values near 1.0 enforce balance.
        prop_untyped_entities: Proportion of entities left without any class.
        avg_specific_class_depth: Target average depth of the most-specific
            assigned class.
        multityping: Whether entities may receive multiple most-specific classes.
        avg_types_per_entity: Target average number of most-specific assigned
            classes per entity.
        check_kg_consistency: Whether to run PyGraft’s reasoner to validate
            consistency of the generated KG.
    """
    # General (JSON order)
    project_name: str
    rdf_format: str
    rng_seed: int | None

    # KG (JSON order)
    num_entities: int
    num_triples: int

    enable_fast_generation: bool
    enable_inference_oversampling: bool

    relation_usage_uniformity: float
    prop_untyped_entities: float

    avg_specific_class_depth: float

    multityping: bool
    avg_types_per_entity: float

    check_kg_consistency: bool

    # Output directory
    output_root: Path | None = None

    def __post_init__(self) -> None:
        """Relies on config.py for semantic validation; enforces core KG generator preconditions only."""
        if not self.project_name:
            message = "project_name must be a non-empty string."
            raise ValueError(message)

        if self.num_entities <= 0:
            message = "num_entities must be a positive integer."
            raise ValueError(message)

        if self.num_triples <= 0:
            message = "num_triples must be a positive integer."
            raise ValueError(message)

        if not 0.0 <= self.relation_usage_uniformity <= 1.0:
            message = "relation_usage_uniformity must be between 0.0 and 1.0."
            raise ValueError(message)

        if not 0.0 <= self.prop_untyped_entities <= 1.0:
            message = "prop_untyped_entities must be between 0.0 and 1.0."
            raise ValueError(message)

        if self.avg_specific_class_depth <= 0.0:
            message = "avg_specific_class_depth must be strictly positive."
            raise ValueError(message)

        if self.avg_types_per_entity < 0.0:
            message = "avg_types_per_entity must be non-negative."
            raise ValueError(message)

        if not self.rdf_format:
            message = "rdf_format must be a non-empty string."
            raise ValueError(message)


# ================================================================================================ #
# Instance Generator                                                                               #
# ================================================================================================ #


class InstanceGenerator:
    """Generate a synthetic knowledge graph consistent with a project_name.

    This class loads `class_info.json` and `relation_info.json` for a
    given project_name, assigns classes to entities, and instantiates triples
    respecting all relevant constraints.

    Notes:
        - This class is considered internal to PyGraft and is not part
          of the public, stable API surface.
        - Determinism is controlled by the `rng_seed` field of
          `InstanceGeneratorConfig`. If `rng_seed` is None, each run is
          stochastic.
        - This class is not intended for inheritance.

    Performance:
        - Entity typing and triple generation are both linear in the
          target sizes in typical configurations, with safety caps to
          prevent pathological loops.
    """

    # ------------------------------------------------------------------------------------------------ #
    # Construction                                                                                     #
    # ------------------------------------------------------------------------------------------------ #

    def __init__(self, *, config: InstanceGeneratorConfig) -> None:
        """Initialize the instance generator with a configuration object.

        The constructor is intentionally lightweight: it performs no
        filesystem I/O and no heavy computation. Schema information is
        loaded lazily when `generate_kg()` is called.

        Args:
            config: Immutable configuration parameters for this generator.
        """
        self._config: InstanceGeneratorConfig = config
        self._rng: np.random.Generator = np.random.default_rng(config.rng_seed)

        # Output directory
        base_root = self._config.output_root or OUTPUT_ROOT
        self._output_directory_path: Path = base_root / self._config.project_name
        self._output_directory = str(self._output_directory_path)

        # Fast generation / oversampling parameters.
        # TODO: Rename it to self._entity_replication_factor so its clearer not only here but everywhere else
        self._fast_ratio: int = (
            get_fast_ratio(self._config.num_entities) if self._config.enable_fast_generation else 1
        )
        self._oversample_every: int = (
            int(self._config.num_triples / self._fast_ratio)
            if self._fast_ratio > 0
            else self._config.num_triples
        )

        # Schema metadata (loaded later).
        self._class_info: ClassInfoDict = cast(ClassInfoDict, {})
        self._relation_info: RelationInfoDict = cast(RelationInfoDict, {})

        # Per-run state (populated during the pipeline).
        self._entities: list[EntityId] = []
        self._typed_entities: set[EntityId] = set()
        self._layer2classes: dict[int, list[str]] = {}
        self._class2layer: dict[str, int] = {}
        self._class2disjoints_extended: dict[str, list[str]] = {}
        self._classes: list[str] = []
        self._non_disjoint_classes: set[str] = set()

        self._ent2layer_specific: dict[EntityId, int] = {}
        self._ent2classes_specific: dict[EntityId, list[str]] = {}
        self._ent2classes_transitive: dict[EntityId, list[str]] = {}

        self._current_avg_depth_specific_class: float = 0.0
        self._badly_typed: dict[EntityId, dict[str, Any]] = {}

        self._class2entities: dict[str, list[EntityId]] = {}
        self._class2unseen: dict[str, list[EntityId]] = {}
        self._unseen_entities_pool: list[EntityId] = []

        self._priority_untyped_entities: set[EntityId] = set()
        self._untyped_entities: list[EntityId] = []

        self._rel2dom: dict[RelationId, str] = {}
        self._rel2range: dict[RelationId, str] = {}
        self._rel2patterns: dict[RelationId, list[str]] = {}
        self._num_relations: int = 0
        self._relation_sampling_weights: list[float] = []
        self._triples_per_rel: dict[RelationId, int] = {}

        self._instance_triples: TripleSet = set()
        self._last_oversample: int = 0

        self._graph: RDFGraph | None = None


    # ------------------------------------------------------------------------------------------------ #
    # Internal convenience                                                                            #
    # ------------------------------------------------------------------------------------------------ #

    def _is_multityping_enabled(self) -> bool:
        """Return True if multityping is effectively enabled."""
        return self._config.multityping and self._config.avg_types_per_entity > 0.0

    # ------------------------------------------------------------------------------------------------ #
    # Public API                                                                                       #
    # ------------------------------------------------------------------------------------------------ #

    def __repr__(self) -> str:
        """Return a debug-friendly representation of this generator."""
        return (
            "InstanceGenerator("
            f"project_name={self._config.project_name!r}, "
            f"num_entities={self._config.num_entities}, "
            f"num_triples={self._config.num_triples}, "
            f"relation_usage_uniformity={self._config.relation_usage_uniformity}, "
            f"prop_untyped_entities={self._config.prop_untyped_entities}, "
            f"avg_specific_class_depth={self._config.avg_specific_class_depth}, "
            f"multityping={self._is_multityping_enabled()}, "
            f"avg_types_per_entity={self._config.avg_types_per_entity}, "
            f"rdf_format={self._config.rdf_format!r}, "
            f"enable_fast_generation={self._config.enable_fast_generation}, "
            f"enable_inference_oversampling={self._config.enable_inference_oversampling}, "
            f"check_kg_consistency={self._config.check_kg_consistency}, "
            f"rng_seed={self._config.rng_seed}"
            ")"
        )

    def generate_kg(self) -> tuple[KGInfoDict, str]:
        """Run the full KG generation pipeline.

        Steps:
            1. Load project_name metadata (`class_info.json`, `relation_info.json`).
            2. Run the entity and typing pipeline.
            3. Generate triples subject to logical constraints.
            4. Apply consistency checks and cleanups.
            5. Assemble and write `kg_info.json`.
            6. Serialize the KG as an RDF/RDF-like graph file.
            7. Optionally run the PyGraft reasoner.

        Returns:
            A tuple with:
                - kg_info: Dictionary of KG statistics and user parameters.
                - kg_file: Path to the serialized KG file.
        """
        self._load_schema_info()
        self._run_entity_and_triple_pipeline()
        self._filter_asymmetry_conflicts()
        self._filter_inverse_asymmetry_conflicts()
        self._filter_domain_range_conflicts()
        self._filter_domain_disjoint_conflicts()
        self._filter_inverse_domain_range_disjoint_conflicts()

        kg_info = self._build_kg_info()
        kg_file = self._serialize_kg()

        return kg_info, kg_file

    # ================================================================================================ #
    # Schema Loading & Metadata                                                                        #
    # ================================================================================================ #

    def _load_schema_info(self) -> None:
        """Load project_name metadata from `class_info.json` and `relation_info.json`.

        This method populates internal project_name dictionaries and validates
        configuration against the observed hierarchy depth.

        Raises:
            FileNotFoundError: If one of the JSON files is missing.
            json.JSONDecodeError: If a JSON file is malformed.
            ValueError: If `avg_specific_class_depth` is incompatible
                with the project_name hierarchy depth.
        """
        class_info_path = self._output_directory_path / "class_info.json"
        relation_info_path = self._output_directory_path / "relation_info.json"

        with class_info_path.open("r", encoding="utf8") as file:
            self._class_info = cast(ClassInfoDict, json.load(file))

        with relation_info_path.open("r", encoding="utf8") as file:
            self._relation_info = cast(RelationInfoDict, json.load(file))

        hierarchy_depth = self._class_info["statistics"]["hierarchy_depth"]
        max_allowed = hierarchy_depth + 1

        if self._config.avg_specific_class_depth > max_allowed:
            message = (
                "avg_specific_class_depth is incompatible with the project_name hierarchy depth: "
                f"got {self._config.avg_specific_class_depth:.2f}, but the maximum allowed value "
                f"for project_name {self._config.project_name!r} is {max_allowed:.2f}. "
                "Please lower avg_specific_class_depth or regenerate the project_name with a "
                "deeper hierarchy."
            )
            raise ValueError(message)

    def _build_kg_info(self) -> KGInfoDict:
        """Assemble and persist a compact summary of the generated KG.

        The summary includes both user parameters (from the config) and
        observed statistics such as:

        - number of entities and instantiated relations
        - number of triples
        - realized proportion of untyped entities
        - realized average depth of most-specific classes
        - realized average multityping

        Returns:
            A dictionary describing the KG (`kg_info`).
        """
        observed_entities: set[EntityId] = {
            entity for head, _, tail in self._instance_triples for entity in (head, tail)
        }
        typed_observed: set[EntityId] = {
            entity for entity in observed_entities if entity in self._ent2classes_specific
        }
        observed_relations: set[RelationId] = {
            triple[1] for triple in self._instance_triples
        }

        # --- user parameters: mirror the JSON "kg" block order ---
        user_parameters: KGUserParameters = {
            "num_entities": self._config.num_entities,
            "num_triples": self._config.num_triples,
            "enable_fast_generation": self._config.enable_fast_generation,
            "enable_inference_oversampling": self._config.enable_inference_oversampling,
            "relation_usage_uniformity": self._config.relation_usage_uniformity,
            "prop_untyped_entities": self._config.prop_untyped_entities,
            "avg_specific_class_depth": self._config.avg_specific_class_depth,
            "multityping": self._is_multityping_enabled(),
            "avg_types_per_entity": self._config.avg_types_per_entity,
            "check_kg_consistency": self._config.check_kg_consistency,
        }

        # --- observed statistics ---
        statistics: KGStatistics = {
            "num_entities": len(observed_entities),
            "num_instantiated_relations": len(observed_relations),
            "num_triples": len(self._instance_triples),
            "prop_untyped_entities": round(
                1 - (len(typed_observed) / max(1, len(observed_entities))),
                2,
            ),
            "avg_specific_class_depth": float(self._current_avg_depth_specific_class),
            "avg_types_per_entity": (
                round(self._compute_avg_multityping(), 2)
                if self._typed_entities
                else 0.0
            ),
        }

        kg_info: KGInfoDict = build_kg_info(
            user_parameters=user_parameters,
            statistics=statistics,
        )

        kg_info_path = (self._output_directory_path / "kg_info.json").resolve()
        with kg_info_path.open("w", encoding="utf8") as file:
            json.dump(kg_info, file, indent=4)

        logger.debug("Wrote kg_info to: %s", kg_info_path)
        return kg_info

    def _serialize_kg(self) -> str:
        """Serialize the KG instances (no project_name) to disk.

        Returns:
            Path to the serialized KG file as a string.
        """
        self._graph = RDFGraph()

        kg_ns = Namespace("http://pygraf.t/")
        self._graph.bind("sc", kg_ns)

        for h, r, t in tqdm(
            self._instance_triples,
            desc="Writing instance triples",
            unit="triples",
            colour="red",
        ):
            self._graph.add((URIRef(kg_ns + h), URIRef(kg_ns + r), URIRef(kg_ns + t)))

            if h in self._ent2classes_specific:
                for class_name in self._ent2classes_specific[h]:
                    self._graph.add(
                        (URIRef(kg_ns + h), RDF.type, URIRef(kg_ns + class_name)),
                    )

            if t in self._ent2classes_specific:
                for class_name in self._ent2classes_specific[t]:
                    self._graph.add(
                        (URIRef(kg_ns + t), RDF.type, URIRef(kg_ns + class_name)),
                    )

        if self._config.rdf_format == "xml":
            kg_path = self._output_directory_path / "kg.rdf"
            self._graph.serialize(str(kg_path), format="xml")
        else:
            kg_path = self._output_directory_path / f"kg.{self._config.rdf_format}"
            self._graph.serialize(str(kg_path), format=self._config.rdf_format)

        logger.info("Serialized KG graph to: %s", kg_path.resolve())
        return str(kg_path)

    # ================================================================================================ #
    # Pipeline: Entities, Typing, and Triples                                                         #
    # ================================================================================================ #

    def _run_entity_and_triple_pipeline(self) -> None:
        """Run the end-to-end pipeline for entity typing and triple generation."""
        base_count = (
            int(self._config.num_entities / self._fast_ratio)
            if self._config.enable_fast_generation
            else self._config.num_entities
        )
        self._entities = [f"E{i}" for i in range(1, base_count + 1)]

        entities_copy = list(self._entities)
        self._rng.shuffle(entities_copy)

        threshold = int(len(self._entities) * (1 - self._config.prop_untyped_entities))
        self._typed_entities = set(entities_copy[:threshold])

        self._layer2classes = {int(k): v for k, v in self._class_info["layer2classes"].items()}
        self._class2layer = self._class_info["class2layer"]
        self._class2disjoints_extended = self._class_info["class2disjoints_extended"]
        self._classes = self._class_info["classes"]
        self._non_disjoint_classes = set(self._classes) - set(
            self._class2disjoints_extended.keys(),
        )

        self._assign_most_specific_classes()

        if self._is_multityping_enabled():
            self._add_multitype_classes()

        self._compute_transitive_types()
        self._fix_disjoint_multitypes()

        if self._config.enable_fast_generation:
            ent2classes_spec_values = list(self._ent2classes_specific.values())
            ent2classes_trans_values = list(self._ent2classes_transitive.values())
            last_ent = len(self._entities)

            for _ in range(1, self._fast_ratio):
                entity_batch = [
                    f"E{i}"
                    for i in range(
                        last_ent + 1,
                        last_ent + int(self._config.num_entities / self._fast_ratio) + 1,
                    )
                ]
                self._rng.shuffle(entity_batch)
                threshold_batch = int(
                    len(entity_batch) * (1 - self._config.prop_untyped_entities),
                )
                typed_entities = entity_batch[:threshold_batch]
                self._typed_entities.update(typed_entities)

                ent2classes_specific = {
                    ent: ent2classes_spec_values[idx] for idx, ent in enumerate(typed_entities)
                }
                ent2classes_transitive = {
                    ent: ent2classes_trans_values[idx] for idx, ent in enumerate(typed_entities)
                }

                self._ent2classes_specific.update(ent2classes_specific)
                self._ent2classes_transitive.update(ent2classes_transitive)
                self._entities += entity_batch
                last_ent = len(self._entities)

        self._generate_triples()

    def _assign_most_specific_classes(self) -> None:
        """Assign a most-specific class to each typed entity.

        Uses the class hierarchy depth from `class_info` to sample a
        target layer in [1, hierarchy_depth] for each typed entity and
        then chooses a most-specific class from that layer.
        """
        hierarchy_depth = self._class_info["statistics"]["hierarchy_depth"]

        if not self._typed_entities:
            self._current_avg_depth_specific_class = 0.0
            self._ent2layer_specific = {}
            self._ent2classes_specific = {}
            return

        # If there is only one named layer below owl:Thing, all typed
        # entities must live in that layer.
        if hierarchy_depth <= 1:
            generated_numbers = np.ones(len(self._typed_entities), dtype=int)
        else:
            shape = hierarchy_depth / (hierarchy_depth - 1)
            numbers = self._rng.power(shape, size=len(self._typed_entities))
            scaled_numbers = (
                numbers
                / float(np.mean(numbers))
                * float(self._config.avg_specific_class_depth)
            )

            # Clamp to valid layers [1, hierarchy_depth].
            generated_numbers = np.clip(
                np.floor(scaled_numbers),
                1,
                hierarchy_depth,
            ).astype(int)

        self._current_avg_depth_specific_class = float(np.mean(generated_numbers))
        self._ent2layer_specific = {
            entity: int(layer)
            for entity, layer in zip(self._typed_entities, generated_numbers, strict=True)
        }

        self._ent2classes_specific = {
            entity: [self._rng.choice(self._layer2classes[layer])]
            for entity, layer in self._ent2layer_specific.items()
        }

    def _add_multitype_classes(self) -> None:
        """Optionally add additional most-specific classes for multityping."""
        current_avg_multityping = 1.0
        entity_list = list(self._typed_entities)
        attempt_count = 0

        if not entity_list or self._config.avg_types_per_entity <= 1.0:
            return

        while current_avg_multityping < self._config.avg_types_per_entity and attempt_count < 10:
            ent = self._rng.choice(entity_list)
            most_specific_classes = self._ent2classes_specific[ent]
            specific_layer = self._ent2layer_specific[ent]
            compatible_classes = self._compute_compatible_classes(most_specific_classes)
            specific_compatible_classes = list(
                set(self._layer2classes[specific_layer]).intersection(compatible_classes),
            )
            specific_compatible_classes = [
                cls for cls in specific_compatible_classes if cls not in most_specific_classes
            ]

            if specific_compatible_classes:
                other_specific_class = self._rng.choice(specific_compatible_classes)
                self._ent2classes_specific[ent].append(other_specific_class)
                current_avg_multityping = self._compute_avg_multityping()
                attempt_count = 0
            else:
                attempt_count += 1

    def _fix_disjoint_multitypes(self) -> None:
        """Check multityping assignments for violations of disjointness."""
        self._badly_typed = {}

        for entity, classes in self._ent2classes_transitive.items():
            for cls in classes:
                disjoints = self._class2disjoints_extended.get(cls, [])
                if set(disjoints).intersection(classes):
                    self._badly_typed[entity] = {
                        "all_classes": classes,
                        "problematic_class": cls,
                        "disjointwith": disjoints,
                    }
                    chosen_specific = self._rng.choice(self._ent2classes_specific[entity])
                    self._ent2classes_specific[entity] = [chosen_specific]
                    self._ent2classes_transitive[entity] = self._class_info[
                        "transitive_class2superclasses"
                    ][chosen_specific]
                    break

    def _compute_transitive_types(self) -> None:
        """Extend entities with their transitive superclasses."""
        self._ent2classes_transitive = {
            ent: list(specific_classes)
            for ent, specific_classes in self._ent2classes_specific.items()
        }

        for ent, specific_classes in self._ent2classes_specific.items():
            transitive_set: set[str] = set(self._ent2classes_transitive[ent])

            for specific_class in specific_classes:
                transitive_set.update(
                    self._class_info["transitive_class2superclasses"][specific_class],
                )

            self._ent2classes_transitive[ent] = list(transitive_set)

    def _compute_avg_multityping(self) -> float:
        """Calculate the average multityping over typed entities."""
        if not self._typed_entities:
            return 0.0

        specific_instantiations = sum(
            len(classes) for classes in self._ent2classes_specific.values()
        )
        return float(specific_instantiations / len(self._typed_entities))

    def _compute_compatible_classes(self, class_list: Iterable[str]) -> set[str]:
        """Return the set of classes compatible with a given list."""
        base_classes = list(class_list)

        compatible_classes = [
            candidate
            for candidate in self._class2disjoints_extended
            if all(
                candidate not in self._class2disjoints_extended.get(specific, [])
                for specific in base_classes
            )
        ]

        return (set(compatible_classes) - set(base_classes)) | self._non_disjoint_classes

    def _generate_triples(self) -> None:
        """Generate triples for the KG with consistency checks.

        This method performs rejection sampling of triples that respect
        domain/range and logical constraints. When oversampling is enabled,
        the method behaves as follows:

        - If enable_fast_generation is False (or the fast ratio is 1), it tries to reach
          `num_triples` via pure sampling. If it stops early because of too
          many failed attempts, a post-processing oversampling phase is used
          to fill the remaining gap up to `num_triples`.

        - If enable_fast_generation is True and the fast ratio is greater than 1, it only
          samples a smaller "base" KG of size approximately
          `num_triples / fast_ratio` and then uses oversampling as a
          post-processing step to reach `num_triples`. This acts as a
          speed hack for restrictive configurations: we reuse patterns from
          already accepted triples instead of continuously sampling new ones.
        """
        # Map each class to the entities that are (transitively) typed with it.
        self._class2entities = {}
        for entity_id, classes in self._ent2classes_transitive.items():
            for class_name in classes:
                self._class2entities.setdefault(class_name, []).append(entity_id)

        # Keep track of "unseen" entities per class and a global pool of unseen entities.
        self._class2unseen = copy.deepcopy(self._class2entities)
        self._unseen_entities_pool = list(
            set(itertools.chain.from_iterable(self._class2entities.values())),
        )

        # Untyped entities are treated specially to try to give them at least one incident triple.
        self._priority_untyped_entities = set(self._entities) - set(self._typed_entities)
        self._untyped_entities = list(copy.deepcopy(self._priority_untyped_entities))

        # Cache domain/range/pattern metadata for faster access inside the sampling loop.
        rel2dom_raw = self._relation_info["rel2dom"]  # dict[RelationId, list[str]]
        rel2range_raw = self._relation_info["rel2range"]  # dict[RelationId, list[str]]

        self._rel2dom = {
            relation_id: dom_list[0]
            for relation_id, dom_list in rel2dom_raw.items()
            if dom_list
        }
        self._rel2range = {
            relation_id: range_list[0]
            for relation_id, range_list in rel2range_raw.items()
            if range_list
        }
        self._rel2patterns = self._relation_info["rel2patterns"]

        # Storage for generated triples and relation-level allocation.
        self._instance_triples = set()
        self._compute_triples_per_relation()
        self._last_oversample = 0

        # Decide how many triples we try to obtain via direct sampling before oversampling.
        # - Without enable_inference_oversampling (or with fast_ratio == 1): we sample up to num_triples.
        # - With enable_inference_oversampling and fast_ratio > 1: we intentionally sample only a "base"
        #   subset of size approximately num_triples / fast_ratio, then enable_inference_oversampling.
        generation_target: int = self._config.num_triples
        if self._config.enable_inference_oversampling and self._fast_ratio > 1:
            generation_target = min(self._oversample_every, self._config.num_triples)

        failed_attempts = 0
        while len(self._instance_triples) < generation_target:
            sampled_relation: RelationId = self._rng.choice(
                self._relation_info["relations"],
                p=self._relation_sampling_weights,
            )
            new_triple = self._sample_triple_for_relation(sampled_relation)
            failed_attempts += 1

            is_consistent = (
                self._triple_is_consistent(new_triple)
                if None not in new_triple
                else False
            )
            if is_consistent:
                self._instance_triples.add(cast(Triple, new_triple))
                failed_attempts = 0

            if failed_attempts > 10:
                logger.warning(
                    "Stopping triple generation loop after %d failed attempts; "
                    "current KG size: %d (sampling target: %d, final target: %d).",
                    failed_attempts,
                    len(self._instance_triples),
                    generation_target,
                    self._config.num_triples,
                )
                break

        # Post-processing oversampling in both use cases (Nico's intent):
        # - Fill remaining gap to num_triples if sampling stopped early.
        # - Inflate a smaller "base" KG (enable_fast_generation + restrictive configs).
        if self._config.enable_inference_oversampling and len(self._instance_triples) < self._config.num_triples:
            logger.info(
                "Oversampling via inference from %d to target %d triples.",
                len(self._instance_triples),
                self._config.num_triples,
            )
            self._oversample_triples_via_inference()

    def _compute_triples_per_relation(self) -> None:
        """Distribute triples across relations based on configuration."""
        self._num_relations = len(self._relation_info["relations"])

        if self._config.num_triples < self._num_relations:
            # For very small KGs, use a uniform distribution over relations and
            # assign at most one triple per relation in order.
            uniform_weight = 1.0 / self._num_relations
            self._relation_sampling_weights = [uniform_weight] * self._num_relations
            self._triples_per_rel = {
                rel: 1 if index < self._config.num_triples else 0
                for index, rel in enumerate(self._relation_info["relations"])
            }
            return

        mean = int(self._config.num_triples / len(self._relation_info["relations"]))
        spread_coeff = (1 - self._config.relation_usage_uniformity) * mean
        weights_array = generate_random_numbers(
            mean,
            spread_coeff,
            self._num_relations,
        )

        # Keep attribute type as list[float] for static typing,
        # but still use the ndarray for downstream numeric ops.
        self._relation_sampling_weights = list(weights_array)

        weights_array = weights_array * self._config.num_triples
        self._triples_per_rel = {
            rel: int(np.ceil(tpr))
            for rel, tpr in zip(self._relation_info["relations"], weights_array, strict=True)
        }

    def _sample_triple_for_relation(
        self,
        relation: RelationId,
    ) -> tuple[EntityId | None, RelationId, EntityId | None]:
        """Generate a single triple for a given relation."""
        required_domain_class = self._rel2dom.get(relation)
        required_range_class = self._rel2range.get(relation)
        token_dom = False

        h: EntityId | None = None
        t: EntityId | None = None

        if required_domain_class:
            head_candidates: list[EntityId] = self._class2unseen.get(
                required_domain_class,
                [],
            )
            h = self._rng.choice(head_candidates) if head_candidates else None

            if h is not None and self._is_entity_compatible_with_class(
                h,
                required_domain_class,
            ):
                token_dom = True
            else:
                h = None
                eligible_entities = self._class2entities.get(required_domain_class, [])
                for _ in range(10):
                    if not eligible_entities:
                        break
                    candidate = self._rng.choice(eligible_entities)
                    if self._is_entity_compatible_with_class(candidate, required_domain_class):
                        h = candidate
                        break
        else:
            if self._untyped_entities:
                if self._priority_untyped_entities:
                    h = self._priority_untyped_entities.pop()
                else:
                    h = self._rng.choice(self._untyped_entities)
            else:
                h = (
                    self._rng.choice(self._unseen_entities_pool)
                    if self._unseen_entities_pool
                    else None
                )

        if required_range_class:
            range_candidates: list[EntityId] = self._class2unseen.get(
                required_range_class,
                [],
            )
            t = self._rng.choice(range_candidates) if range_candidates else None

            if (
                t is not None
                and h is not None
                and self._is_entity_compatible_with_class(t, required_range_class)
            ):
                self._class2unseen[required_range_class].remove(t)
                if token_dom and required_domain_class:
                    unseen_domain_entities = self._class2unseen.get(required_domain_class, [])
                    if h in unseen_domain_entities:
                        unseen_domain_entities.remove(h)
            else:
                t = None
                eligible_entities = self._class2entities.get(required_range_class, [])
                for _ in range(10):
                    if not eligible_entities:
                        break
                    candidate = self._rng.choice(eligible_entities)
                    if self._is_entity_compatible_with_class(candidate, required_range_class):
                        t = candidate
                        break
        else:
            if self._untyped_entities:
                if self._priority_untyped_entities:
                    t = self._priority_untyped_entities.pop()
                else:
                    t = self._rng.choice(self._untyped_entities)
            else:
                t = (
                    self._rng.choice(self._unseen_entities_pool)
                    if self._unseen_entities_pool
                    else None
                )

        return (h, relation, t)

    def _triple_is_consistent(
        self,
        triple: tuple[EntityId | None, RelationId, EntityId | None],
    ) -> bool:
        """Check whether a candidate triple is consistent with constraints."""
        h, r, t = triple

        if h is None or t is None:
            return False

        if r in self._relation_info["irreflexive_relations"] and h == t:
            return False

        if r in self._relation_info["asymmetric_relations"] and (
            h == t or (t, r, h) in self._instance_triples
        ):
            return False

        if (
            r in self._relation_info["functional_relations"]
            and any(
                existing_triple[:2] == (h, r)
                for existing_triple in self._instance_triples
            )
        ):
            return False

        if (
            r in self._relation_info["inversefunctional_relations"]
            and any(
                existing_triple[1:] == (r, t)
                for existing_triple in self._instance_triples
            )
        ):
            return False

        return True

    # ================================================================================================ #
    # Consistency Checks & Inference                                                                   #
    # ================================================================================================ #

    def _filter_inverse_asymmetry_conflicts(self) -> None:
        """Check inverse-of / asymmetry interactions in the KG."""
        rel2inverse = self._build_rel_inverse_map()

        for r1, r2 in rel2inverse.items():
            if (
                r1 not in self._relation_info["asymmetric_relations"]
                and r2 not in self._relation_info["asymmetric_relations"]
            ):
                continue

            subset_kg = [trip for trip in self._instance_triples if trip[1] in (r1, r2)]
            counter = Counter(subset_kg)
            duplicates: set[Triple] = {
                trip for trip, count in counter.items() if count > 1
            }

            if duplicates:
                self._instance_triples -= duplicates

    def _filter_domain_range_conflicts(self) -> None:
        """Remove triples whose entities violate domain/range constraints."""
        to_remove: set[Triple] = set()

        for triple in self._instance_triples:
            h, r, t = triple
            r2dom = self._rel2dom.get(r)
            r2range = self._rel2range.get(r)

            if r2dom and h in self._ent2classes_transitive:
                is_valid_dom = self._is_entity_compatible_with_class(h, r2dom)
                if not is_valid_dom:
                    to_remove.add(triple)

            if r2range and t in self._ent2classes_transitive:
                is_valid_range = self._is_entity_compatible_with_class(t, r2range)
                if not is_valid_range:
                    to_remove.add(triple)

        self._instance_triples -= to_remove

    def _build_rel_inverse_map(self) -> dict[RelationId, RelationId]:
        """Return a canonical inverse map (r ↔ inv), independent of JSON order."""
        rel2inv: dict[RelationId, RelationId] = self._relation_info["rel2inverse"]

        # Validate symmetry
        for r, inv in rel2inv.items():
            if rel2inv.get(inv) != r:
                message = f"Malformed rel2inverse: {r!r} <-> {inv!r} not symmetric."
                raise ValueError(message)

        # Canonicalize pairs
        seen: set[frozenset[RelationId]] = set()
        canonical: dict[RelationId, RelationId] = {}

        for r, inv in rel2inv.items():
            pair = frozenset({r, inv})
            if len(pair) != 2:
                message = f"Relation {r!r} is its own inverse."
                raise ValueError(message)
            if pair in seen:
                continue
            seen.add(pair)
            a, b = sorted(pair)
            canonical[a] = b

        return canonical

    def _filter_asymmetry_conflicts(self) -> None:
        """Enforce asymmetry: remove symmetric counterparts for asymmetric relations."""
        for relation in self._relation_info["asymmetric_relations"]:
            subset_kg: list[Triple] = [
                trip for trip in self._instance_triples if trip[1] == relation
            ]

            to_remove: set[Triple] = set()
            seen: set[Triple] = set()

            for triple in subset_kg:
                symmetric_triple = (triple[2], triple[1], triple[0])
                if symmetric_triple in seen:
                    to_remove.add(triple)
                seen.add(triple)

            self._instance_triples -= to_remove

    def _is_entity_compatible_with_class(self, ent: EntityId, expected_class: str) -> bool:
        """Check disjointness between an entity's classes and an expected class."""
        entity_classes = set(self._ent2classes_transitive[ent])
        relation_side_classes = self._class_info["transitive_class2superclasses"][expected_class]

        for cls in relation_side_classes:
            disjoint_classes = set(self._class2disjoints_extended.get(cls, []))
            if disjoint_classes & entity_classes:
                return False

        return True

    def _oversample_triples_via_inference(self) -> None:
        """Oversample triples using logical inference patterns.

        This is a pure post-processing step applied after the main triple
        generation loop. It assumes that `self._instance_triples` already
        contains a (possibly partial) KG generated via rejection sampling
        and then:

        - Infers new triples using inverse-of, symmetric, and subproperty
          patterns defined in `relation_info`.
        - Adds those inferred triples to the KG until the requested
          `num_triples` is reached or a safety limit is hit.

        Conceptually, there are two main use cases:

        - "Completion": for normal configurations, if the sampling loop
          stops before reaching `num_triples`, this method fills the gap
          so that the final KG size is as close as possible to the target.

        - "Augmentation hack": for restrictive schemas (e.g. many disjoint
          classes) and/or when `enable_fast_generation` is enabled, we only generate a
          smaller base KG via sampling and then enable_inference_oversampling it, reusing the
          already valid patterns (similar to data augmentation in computer
          vision).
        """
        used_relations: set[RelationId] = set()
        id2pattern: dict[int, list[RelationId]] = {
            1: self._relation_info["inverseof_relations"],
            2: self._relation_info["symmetric_relations"],
            3: self._relation_info["subrelations"],
        }

        max_attempts = 1000
        attempts = 0

        while len(self._instance_triples) < self._config.num_triples and attempts < max_attempts:
            attempts += 1
            chosen_id = int(self._rng.integers(1, len(id2pattern) + 1))
            candidate_relations = id2pattern[chosen_id]

            if not candidate_relations:
                continue

            self._rng.shuffle(candidate_relations)
            relation = candidate_relations[0]

            if relation in used_relations:
                continue

            used_relations.add(relation)
            attempts = 0

            subset_kg: set[Triple] = {
                trip for trip in self._instance_triples if trip[1] == relation
            }
            if not subset_kg:
                continue

            if chosen_id == 1:
                inverse_relation = self._relation_info["rel2inverse"][relation]
                inferred_triples = inverse_inference(subset_kg, inverse_relation)
            elif chosen_id == 2:
                inferred_triples = symmetric_inference(subset_kg)
            else:
                super_relation = self._relation_info["rel2superrel"][relation]
                inferred_triples = subproperty_inference(subset_kg, super_relation)

            if not inferred_triples:
                continue

            self._instance_triples |= set(inferred_triples)

            if len(self._instance_triples) >= self._config.num_triples:
                return

    def _filter_domain_disjoint_conflicts(self) -> None:
        """Check that domains/ranges are compatible with instantiated triples."""
        for rel, dom in self._rel2dom.items():
            if dom not in self._class2disjoints_extended:
                continue

            subset_kg = {trip for trip in self._instance_triples if trip[1] == rel}
            disjoint_with_dom = self._class2disjoints_extended[dom]
            wrong_heads: set[EntityId] = set()

            for h, _, _ in subset_kg:
                if h in self._ent2classes_transitive:
                    intersection = set(self._ent2classes_transitive[h]).intersection(
                        disjoint_with_dom,
                    )
                    if intersection:
                        wrong_heads.add(h)

            problematic_triples = {
                (head, relation, tail) for head, relation, tail in subset_kg if head in wrong_heads
            }
            self._instance_triples -= problematic_triples

        for rel, rng in self._rel2range.items():
            if rng not in self._class2disjoints_extended:
                continue

            subset_kg = {trip for trip in self._instance_triples if trip[1] == rel}
            disjoint_with_range = self._class2disjoints_extended[rng]
            wrong_tails: set[EntityId] = set()

            for _, _, t in subset_kg:
                if t in self._ent2classes_transitive:
                    intersection = set(self._ent2classes_transitive[t]).intersection(
                        disjoint_with_range,
                    )
                    if intersection:
                        wrong_tails.add(t)

            problematic_triples = {
                (head, relation, tail) for head, relation, tail in subset_kg if tail in wrong_tails
            }
            self._instance_triples -= problematic_triples

    def _filter_inverse_domain_range_disjoint_conflicts(self) -> None:
        """Check inverse relations for compatibility with disjointness."""
        rel2inverse = self._build_rel_inverse_map()

        for r1 in rel2inverse:
            r2 = rel2inverse[r1]
            subset_kg = {trip for trip in self._instance_triples if trip[1] == r1}

            if r2 in self._rel2range and self._rel2range[r2] in self._class2disjoints_extended:
                range_r2 = self._rel2range[r2]
                disjoint_with_range = self._class2disjoints_extended[range_r2]
                wrong_heads: set[EntityId] = set()

                for h, _, _ in subset_kg:
                    if h in self._ent2classes_transitive:
                        intersection = set(self._ent2classes_transitive[h]).intersection(
                            disjoint_with_range,
                        )
                        if intersection:
                            wrong_heads.add(h)

                problematic_triples = {
                    (head, relation, tail)
                    for head, relation, tail in subset_kg
                    if head in wrong_heads
                }
                self._instance_triples -= problematic_triples

            if r2 in self._rel2dom and self._rel2dom[r2] in self._class2disjoints_extended:
                dom_r2 = self._rel2dom[r2]
                disjoint_with_dom = self._class2disjoints_extended[dom_r2]
                wrong_tails: set[EntityId] = set()

                for _, _, t in subset_kg:
                    if t in self._ent2classes_transitive:
                        intersection = set(self._ent2classes_transitive[t]).intersection(
                            disjoint_with_dom,
                        )
                        if intersection:
                            wrong_tails.add(t)

                problematic_triples = {
                    (head, relation, tail)
                    for head, relation, tail in subset_kg
                    if tail in wrong_tails
                }
                self._instance_triples -= problematic_triples
