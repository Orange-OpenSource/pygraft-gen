#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Knowledge Graph Instance Generator.

This module implements PyGraft-gen's synthetic knowledge graph (KG) generator.
It consumes ontology metadata extracted by the PyGraft-gen extraction pipeline
(`class_info.json`, `relation_info.json`, `namespaces_info.json`) and
produces instance-level RDF triples that respect the ontology's declared
structure.

Overview:
    The generator creates entities, assigns class types respecting hierarchy
    and disjointness constraints, then generates relational triples using a
    batch-sampling pipeline with two-phase constraint filtering. The output
    is a valid RDF knowledge graph where every triple satisfies the ontology's
    explicit axioms.

    PyGraft-gen focuses exclusively on **object properties** (relations between
    entities). Datatype properties, literal-valued attributes, and complex OWL
    constructs (blank-node class expressions, value restrictions, compound
    domain/range expressions) are intentionally excluded.

Architecture:
    The InstanceGenerator orchestrates four phases via dedicated modules:

    1. **Schema Loading** (`schema.py`):
       Loads JSON metadata, builds ID mappings, initializes constraint caches,
       validates schema for forbidden OWL combinations, computes disjoint
       envelopes.

    2. **Entity Generation** (`entities.py`):
       Creates entity ID space, assigns class types via power-law sampling,
       adds multityping, computes transitive closures, resolves disjoint
       conflicts.

    3. **Triple Generation** (`generation.py`):
       Batch-sampling pipeline with two-phase constraint filtering:
       - Fast filtering: vectorized elimination (irreflexive, duplicates,
         functional constraints)
       - Deep validation: per-triple checks (typing, disjoint envelopes,
         inverse relations, transitive cycles)

    4. **Output** (`output.py`):
       Serializes KG to RDF file, writes kg_info.json with statistics.

OWL Constraint Enforcement:
    The generator enforces the following OWL 2 property characteristics:

    - **owl:IrreflexiveProperty**: Fast rejection of self-loop candidates.
    - **owl:SymmetricProperty**: Duplicate prevention for reverse edges.
    - **owl:AsymmetricProperty**: Fast rejection where reverse edge exists;
      implies irreflexivity.
    - **owl:TransitiveProperty**: Deep validation to prevent transitive cycles
      with irreflexive closure.
    - **owl:FunctionalProperty**: Fast rejection where subject already has an
      outgoing edge for this relation.
    - **owl:InverseFunctionalProperty**: Fast rejection where object already
      has an incoming edge for this relation.
    - **owl:ReflexiveProperty**: Not explicitly generated; self-loops are
      inferred by DL reasoners from the property declaration.

    Relational constraints enforced:

    - **rdfs:domain / rdfs:range**: Entity pools filtered to satisfy all
      declared domain/range classes (intersection semantics).
    - **owl:inverseOf**: Validates that inverse triples would be valid;
      respects functional constraints on both properties.
    - **rdfs:subPropertyOf**: Subproperties inherit constraints from
      superproperties (domain, range, characteristics, disjointness).
    - **owl:propertyDisjointWith**: Rejects triples where the same entity
      pair exists for a disjoint property.

Schema Validation:
    Before triple generation, the ontology schema undergoes validation to
    detect logical contradictions and modeling errors:

    - **SEVERE errors** (stop generation): Reflexive+Irreflexive,
      Symmetric+Asymmetric, bidirectional inverse mapping violations,
      inheritance conflicts (symmetric sub of asymmetric super).
    - **WARNING errors** (exclude relation): Asymmetric+Functional,
      Transitive+Functional, domain/range with disjoint class pairs,
      symmetric properties with mismatched domain/range.

Main Abstractions:
    InstanceGeneratorConfig:
        Immutable configuration dataclass specifying KG size parameters
        (`num_entities`, `num_triples`), typing behavior
        (`prop_untyped_entities`, `avg_specific_class_depth`,
        `multityping`, `avg_types_per_entity`), and generation heuristics
        (`relation_usage_uniformity`, `enable_fast_generation`).

    InstanceGenerator:
        Thin orchestrator that coordinates the four pipeline phases.
        Holds configuration and RNG, constructs phase-specific collaborators,
        exposes `generate_kg()` as the single public method.

    Supporting dataclasses (in `structures.py`):

    - `SchemaMetadata`: Bidirectional string-to-integer ID mappings.
    - `ConstraintCaches`: Pre-computed constraint data for validation.
    - `EntityTypingState`: Entity class assignments and transitive closures.
    - `TripleGenerationState`: Budgets, weights, and KG storage.

Randomness and Determinism:
    All randomness flows through a private `numpy.random.Generator` instance
    owned by `InstanceGenerator`. When `rng_seed` is provided in the
    configuration, generation is fully deterministic and reproducible for the
    same configuration and ontology.

    This module does not modify NumPy's global random state.

Module Invariants:
    For a successful run:

    - Entities receive class typings consistent with class disjointness axioms.
    - Triples respect all enforced OWL property characteristics.
    - Domain/range constraints are satisfied for every triple.
    - No duplicate triples exist in the output KG.
    - Symmetric relations store only one direction per entity pair.

Performance:
    Complexity analysis where n_e = `num_entities`, n_t = `num_triples`,
    n_c = number of classes, n_r = number of relations:

    - **Entity Typing**: O(n_e * log(n_c)) driven by hierarchy layer sampling
      and disjointness validation, with vectorized NumPy operations.
    - **Triple Generation**: O(n_t * batch_size * validation_cost) where
      validation cost depends on constraint complexity. Fast filtering is
      O(batch_size) vectorized; deep validation is O(1) per triple for most
      constraints.
    - **Memory**: O(n_e + n_t) for entity structures and KG storage, plus
      O(n_c + n_r) for schema metadata and constraint caches.

    Adaptive batch sizing and stall detection prevent pathological loops
    in heavily constrained configurations.

Example:
    Typical usage after ontology extraction::

        config = InstanceGeneratorConfig(
            project_name="my_ontology",
            rdf_format="ttl",
            rng_seed=42,
            num_entities=10000,
            num_triples=50000,
            enable_fast_generation=False,
            relation_usage_uniformity=0.8,
            prop_untyped_entities=0.1,
            avg_specific_class_depth=2.5,
            multityping=True,
            avg_types_per_entity=1.5,
            check_kg_consistency=False,
        )
        generator = InstanceGenerator(config=config)
        kg_info, kg_file = generator.generate_kg()

Module Structure:
    This module (`kg.py`) contains only the thin orchestrator class.
    Implementation details are delegated to:

    - `types.py`: Type aliases (EntityId, ClassId, Triple, etc.)
    - `structures.py`: Data classes (SchemaMetadata, ConstraintCaches, etc.)
    - `config.py`: InstanceGeneratorConfig
    - `schema.py`: SchemaLoader (loading, validation, cache initialization)
    - `entities.py`: EntityGenerator (entity creation and typing)
    - `generation.py`: TripleGenerator (batch sampling, constraint filtering)
    - `output.py`: Serialization functions (RDF output, kg_info.json)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from pygraft.generators.kg.entities import EntityGenerator
from pygraft.generators.kg.generation import TripleGenerator
from pygraft.generators.kg.output import build_kg_info, serialize_kg
from pygraft.generators.kg.schema import SchemaLoader
from pygraft.generators.kg.structures import (
    ConstraintCaches,
    EntityTypingState,
    SchemaMetadata,
    TripleGenerationState,
)
from pygraft.paths import OUTPUT_ROOT
from pygraft.types import ClassInfoDict, RelationInfoDict
from pygraft.utils.kg import get_fast_ratio

if TYPE_CHECKING:
    from pygraft.generators.kg.config import InstanceGeneratorConfig
    from pygraft.types import KGInfoDict


class InstanceGenerator:
    """Generate synthetic knowledge graphs consistent with an extracted ontology.

    Thin orchestrator that coordinates the KG generation pipeline:
    schema loading → entity generation → triple generation → output serialization.

    Attributes:
        _config: Generation configuration.
        _rng: NumPy random generator for deterministic sampling.
        _output_directory_path: Path to project output folder.
        _fast_ratio: Scaling factor for fast generation mode.
        _class_info: Loaded class_info.json contents.
        _relation_info: Loaded relation_info.json contents.
        _schema: String-to-integer ID mappings.
        _constraints: Pre-computed constraint caches.
        _entities: Entity typing state.
        _triples: Triple generation state.
        _ontology_prefix: Ontology namespace prefix.
        _ontology_namespace: Ontology namespace URI.
        _prefix2namespace: Prefix to namespace URI mappings.

    Example:
        generator = InstanceGenerator(config=config)
        kg_info, kg_file = generator.generate_kg()

    Note:
        Internal to PyGraft-gen. Not intended for subclassing.
    """

    # ================================================================================================ #
    # CONSTRUCTION                                                                                     #
    # ================================================================================================ #

    def __init__(self, *, config: InstanceGeneratorConfig) -> None:
        """Initialize the generator with a configuration object.

        Args:
            config: Immutable configuration specifying KG size, typing behavior,
                and generation heuristics.
        """
        self._config = config
        self._rng = np.random.default_rng(config.rng_seed)

        base_root = self._config.output_root or OUTPUT_ROOT
        self._output_directory_path = base_root / self._config.project_name
        self._output_directory = str(self._output_directory_path)

        self._fast_ratio = (
            get_fast_ratio(self._config.num_entities) if self._config.enable_fast_generation else 1
        )

        # Schema information (populated by _load_schema_metadata)
        self._class_info = cast(ClassInfoDict, {})
        self._relation_info = cast(RelationInfoDict, {})

        # Structured state (populated by pipeline phases)
        self._schema = SchemaMetadata()
        self._constraints = ConstraintCaches()
        self._entities = EntityTypingState()
        self._triples = TripleGenerationState()

        # Namespace configuration (populated by _load_schema_metadata)
        self._ontology_prefix = "sc"
        self._ontology_namespace = "http://pygraf.t/"
        self._prefix2namespace: dict[str, str] = {}

    # ================================================================================================ #
    # PUBLIC API                                                                                       #
    # ================================================================================================ #

    def generate_kg(self) -> tuple[KGInfoDict, str]:
        """Run the full KG generation pipeline.

        Executes the four-phase pipeline:
            1. Load schema metadata and build constraint caches
            2. Validate configuration against schema
            3. Generate entities with class type assignments
            4. Generate triples with constraint filtering
            5. Serialize output and write statistics

        Returns:
            Tuple of (kg_info, kg_file) where kg_info is a dictionary of KG
            statistics and user parameters, and kg_file is the path to the
            serialized RDF file.

        Raises:
            ValueError: If configuration is incompatible with schema.
            RuntimeError: If serialization fails.
        """
        # Phase 1: Load schema
        self._load_schema_metadata()

        # Phase 2: Validate config vs schema
        self._validate_configuration()

        # Phase 3: Generate entities
        self._generate_typed_entities()

        # Phase 4: Generate triples
        self._generate_triples()

        # Phase 5: Serialize output
        kg_info = build_kg_info(
            config=self._config,
            entities=self._entities,
            triples=self._triples,
            output_directory_path=self._output_directory_path,
            is_multityping_enabled=self._is_multityping_enabled(),
        )
        kg_file = serialize_kg(
            config=self._config,
            schema=self._schema,
            entities=self._entities,
            triples=self._triples,
            ontology_namespace=self._ontology_namespace,
            ontology_prefix=self._ontology_prefix,
            prefix2namespace=self._prefix2namespace,
            output_directory_path=self._output_directory_path,
        )

        return kg_info, kg_file

    # ================================================================================================ #
    # PIPELINE PHASES                                                                                  #
    # ================================================================================================ #

    def _load_schema_metadata(self) -> None:
        """Phase 1: Load schema data and build constraint caches.

        Uses SchemaLoader to:
            - Read class_info.json, relation_info.json, namespaces_info.json
            - Build bidirectional ID mappings
            - Initialize class and relation constraint caches
            - Validate schema for forbidden OWL combinations
            - Compute disjoint envelopes for fast rejection

        Raises:
            ValueError: If schema contains SEVERE constraint violations.
            FileNotFoundError: If required JSON files are missing.
        """
        schema_loader = SchemaLoader(output_directory_path=self._output_directory_path)
        schema_loader.load()

        # Transfer results to instance
        self._schema = schema_loader.schema
        self._constraints = schema_loader.constraints
        self._class_info = schema_loader.class_info
        self._relation_info = schema_loader.relation_info
        self._ontology_prefix = schema_loader.ontology_prefix
        self._ontology_namespace = schema_loader.ontology_namespace
        self._prefix2namespace = schema_loader.prefix2namespace

    def _validate_configuration(self) -> None:
        """Phase 2: Validate configuration against loaded schema.

        Ensures user-specified parameters are achievable given the ontology's
        structure. This is config-vs-schema compatibility checking, not schema
        validation (which happens in SchemaLoader).

        Raises:
            ValueError: If configuration is incompatible with schema.
        """
        self._validate_avg_specific_class_depth()

    def _generate_typed_entities(self) -> None:
        """Phase 3: Generate entities and assign class types.

        Uses EntityGenerator to:
            - Initialize entity ID space and storage arrays
            - Assign most specific classes via power-law sampling
            - Add multitype classes respecting disjointness
            - Compute transitive superclass closures
            - Resolve any disjoint type conflicts
            - Replicate profiles if fast generation enabled

        Complexity:
            O(n_e * d) where n_e is entity count and d is average hierarchy depth.
        """
        entity_generator = EntityGenerator(
            config=self._config,
            constraints=self._constraints,
            class_info=self._class_info,
            schema=self._schema,
            rng=self._rng,
            fast_ratio=self._fast_ratio,
        )
        self._entities = entity_generator.generate()

    def _generate_triples(self) -> None:
        """Phase 4: Generate KG triples with constraint filtering.

        Uses TripleGenerator to:
            - Build candidate entity pools per relation
            - Distribute triple budget across relations
            - Run batch-sampling loop with adaptive sizing
            - Apply fast filtering (vectorized constraint elimination)
            - Apply deep validation (per-triple constraint checks)
            - Track and record accepted triples

        Complexity:
            O(n_t * batch_size) with vectorized fast filtering; deep validation
            adds O(1) per surviving candidate for most constraints.
        """
        triple_generator = TripleGenerator(
            config=self._config,
            constraints=self._constraints,
            schema=self._schema,
            entities=self._entities,
            triples=self._triples,
            relation_info=self._relation_info,
            rng=self._rng,
        )
        triple_generator.generate()

    # ================================================================================================ #
    # INTERNAL UTILITIES                                                                               #
    # ================================================================================================ #

    def _validate_avg_specific_class_depth(self) -> None:
        """Validate that avg_specific_class_depth is achievable with schema hierarchy.

        Layers are 1-indexed and capped at hierarchy_depth. Layer 0 (owl:Thing)
        is excluded. Configured depth cannot exceed the deepest layer containing
        actual classes.

        Raises:
            ValueError: If avg_specific_class_depth exceeds hierarchy_depth.
        """
        hierarchy_depth = self._class_info["statistics"]["hierarchy_depth"]

        if self._config.avg_specific_class_depth > hierarchy_depth:
            message = (
                f"avg_specific_class_depth ({self._config.avg_specific_class_depth:.2f}) exceeds "
                f"hierarchy depth ({hierarchy_depth}) for schema '{self._config.project_name}'. "
                f"Reduce to <= {hierarchy_depth}."
            )
            raise ValueError(message)

    def _is_multityping_enabled(self) -> bool:
        """Check if multityping is effectively enabled.

        Returns:
            True if both multityping flag is set and avg_types_per_entity > 0.
        """
        return self._config.multityping and self._config.avg_types_per_entity > 0.0
