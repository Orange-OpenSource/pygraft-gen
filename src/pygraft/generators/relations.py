#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""
Relation Generator Module
=========================

This module implements PyGraft's internal synthetic object-property generator. It
creates a full OWL object-property schema — relation identifiers, logical
property patterns, inverse-of relations, subPropertyOf hierarchies, and
domain/range profiles — using controlled randomness and compatibility rules.

High-Level Purpose
------------------
The module's role is to produce coherent, statistically-shaped relation schemas
for testing, benchmarking, and synthetic ontology generation. It is internal to
PyGraft and not part of the public API, so its structure may evolve over time.

The module defines two main abstractions:

- RelationGeneratorConfig:
    Immutable configuration object for relation counts, probability targets,
    profiling depth, and determinism settings.

- RelationGenerator:
    A stochastic generator responsible for:
        - creating relation identifiers (R1…Rn)
        - sampling OWL logical property patterns
        - pairing compatible inverse relations
        - assigning domain/range profiles using hierarchy + disjointness
          constraints
        - constructing subPropertyOf hierarchies when valid
        - assembling a final RelationInfoDict summary

Determinism and Randomness
--------------------------
All randomness comes from a private RNG instance. When the configuration includes
a rng_seed, generation becomes fully deterministic and reproducible. No global random
state is used.

Module Invariants
-----------------
- No incompatible OWL property combinations are assigned.
- Disjointness will always be respected when sampling domains and ranges.
- Reflexive relations are never assigned domain/range profiles.
- Subproperties are added only when domain/range constraints justify them.
- All internal mappings are rebuilt fresh for each schema generation run.

Performance Notes
-----------------
Let n = number of relations and m = number of classes.

- Pattern assignment: ~O(n), occasionally up to O(n log n).
- Inverse-of pairing: worst case O(n²) due to compatibility searching; bounded by
  safety caps.
- Profiling (domain/range): up to O(n · m), typically cheaper due to filtering.
- Memory use: O(n) identifiers with potential O(n²) relationships in dense cases.

Intended Use
------------
This module is used by higher-level orchestration layers for dataset creation,
benchmarking, and testing. External callers should use:

    RelationGenerator.generate_relation_schema()

as the single public entry point. The module performs no CLI work, no external
I/O beyond reading packaged compatibility tables, and exposes no mutable internal
state.
"""


from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
import itertools
import json
import logging
import random
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pygraft.types import build_relation_info

if TYPE_CHECKING:
    from pygraft.types import ClassInfoDict, RelationInfoDict, RelationStatistics

logger = logging.getLogger(__name__)


# ================================================================================================ #
# Configuration                                                                                    #
# ================================================================================================ #


@dataclass(frozen=True)
class RelationGeneratorConfig:
    """Immutable configuration for RelationGenerator.

    Attributes:
        rng_seed: Optional seed for deterministic behavior of the internal RNG.

        num_relations: Total number of relations (object properties) to generate.
        relation_specificity: Target average depth (layer) of domain/range
            classes used in relation profiles. Higher means more specific classes.
        prop_profiled_relations: Target proportion of non-reflexive relations
            that receive a domain and/or range profile.
        profile_side: Whether to profile "both" domain and range or only one side ("partial").

        prop_symmetric_relations: Target proportion of symmetric relations.
        prop_inverse_relations: Target proportion of relations that appear
            in an inverse-of pair.
        prop_transitive_relations: Target proportion of transitive relations.
        prop_asymmetric_relations: Target proportion of asymmetric relations.
        prop_reflexive_relations: Target proportion of reflexive relations.
        prop_irreflexive_relations: Target proportion of irreflexive relations.
        prop_functional_relations: Target proportion of functional relations.
        prop_inverse_functional_relations: Target proportion of inverse-functional relations.
        prop_subproperties: Target proportion of relations participating in
            a subPropertyOf relationship.
    """

    # General
    rng_seed: int | None

    # Relations
    num_relations: int
    relation_specificity: float
    prop_profiled_relations: float
    profile_side: str

    prop_symmetric_relations: float
    prop_inverse_relations: float
    prop_transitive_relations: float
    prop_asymmetric_relations: float
    prop_reflexive_relations: float
    prop_irreflexive_relations: float
    prop_functional_relations: float
    prop_inverse_functional_relations: float
    prop_subproperties: float


    def __post_init__(self) -> None:
        """Assumes config.py already validated user input; checks only local invariants for relation generation."""
        if self.num_relations <= 0:
            message = "num_relations must be a positive integer."
            raise ValueError(message)

        self._validate_ratio(self.prop_profiled_relations, "prop_profiled_relations")
        self._validate_ratio(self.prop_symmetric_relations, "prop_symmetric_relations")
        self._validate_ratio(self.prop_inverse_relations, "prop_inverse_relations")
        self._validate_ratio(self.prop_functional_relations, "prop_functional_relations")
        self._validate_ratio(
            self.prop_inverse_functional_relations,
            "prop_inverse_functional_relations",
        )
        self._validate_ratio(self.prop_transitive_relations, "prop_transitive_relations")
        self._validate_ratio(self.prop_subproperties, "prop_subproperties")
        self._validate_ratio(self.prop_reflexive_relations, "prop_reflexive_relations")
        self._validate_ratio(self.prop_irreflexive_relations, "prop_irreflexive_relations")
        self._validate_ratio(self.prop_asymmetric_relations, "prop_asymmetric_relations")

        if self.relation_specificity < 0.0:
            message = "relation_specificity must be non-negative."
            raise ValueError(message)

        if self.profile_side not in {"both", "partial"}:
            message = "profile_side must be either 'both' or 'partial'."
            raise ValueError(message)

    @staticmethod
    def _validate_ratio(value: float, name: str) -> None:
        """Validate that a proportion is within [0.0, 1.0]."""
        if not 0.0 <= value <= 1.0:
            message = f"{name} must be between 0.0 and 1.0."
            raise ValueError(message)


# ================================================================================================ #
# Relation Generator                                                                               #
# ================================================================================================ #

class RelationGenerator:
    """Generate a synthetic OWL object property schema.

    This class is responsible for:
        - Creating a list of relation identifiers (R1, R2, ..., Rn).
        - Sampling logical property patterns for each relation
          (Reflexive, Irreflexive, Symmetric, Asymmetric, Transitive,
          Functional, InverseFunctional).
        - Pairing compatible relations as inverses.
        - Building subproperty hierarchies when compatible with domains/ranges.
        - Assigning domain/range profiles guided by class information.

    Notes:
        - This class is considered internal to PyGraft and is not part of the
          public, stable API surface.
        - Behavior is deterministic only when a rng_seed is provided in the
          configuration. If `rng_seed` is None, randomness will vary across runs.

    Performance:
        Let n = num_relations and m = number of classes.

        - Property pattern assignment:
            Property patterns are assigned using set lookups and sampling over
            pools of candidate relations. The dominant operations are:
              * Building and updating pattern2rels maps.
              * Sampling subsets of relations for each property.
            In practice this is roughly O(n) to O(n log n).

        - Inverse-of generation:
            Inverse-of pairing searches over observed property patterns and
            compatibility matrices derived from pre-computed JSON/TXT files.
            The worst-case number of attempts is capped; the behavior is
            approximately O(n^2) in the worst case.

        - Profiling (domains/ranges):
            Domain/range sampling interacts with the class hierarchy and
            disjointness maps, performing filtered choices over O(m) classes
            and iterating until the desired proportion of profiled relations
            is reached. Overall complexity is roughly O(n * m) in the worst
            case, with typical runs significantly cheaper.

        - Memory usage:
            The main structures are:
              * rel2patterns, pattern2rels
              * rel2dom, rel2range
              * rel2inverse, prop2superprop
            Each is bounded by O(n) keys and O(n^2) edges in the worst case.
    """

    def __init__(
        self,
        *,
        config: RelationGeneratorConfig,
        class_info: ClassInfoDict,
    ) -> None:
        """Initialize the relation generator with configuration and class info.

        This constructor is intentionally lightweight and does not perform I/O
        or random sampling. All heavy work happens in generate_relation_schema().

        Args:
            config: Immutable configuration parameters for this generator.
            class_info: Pre-computed class schema, including disjointness and
                layer information.
        """
        # User-supplied configuration (immutable).
        self._config: RelationGeneratorConfig = config
        self._class_info: ClassInfoDict = class_info

        # Internal RNG (no global random state).
        self._rng: random.Random = random.Random(config.rng_seed)

        # ------------------------------------------------------------------
        # Cached class-level structures
        # ------------------------------------------------------------------
        self._class2disjoints_extended: dict[str, list[str]] = {}
        self._layer2classes: dict[int, list[str]] = {}
        self._class2layer: dict[str, int] = {}

        # ------------------------------------------------------------------
        # Relation-level state (initialized per run)
        # ------------------------------------------------------------------
        self._relations: list[str] | None = None
        self._rel2dom: dict[str, str] = {}
        self._rel2range: dict[str, str] = {}
        self._rel2inverse: dict[str, str] = {}
        self._rel2patterns: dict[str, set[str]] = {}
        self._pattern2rels: defaultdict[frozenset[str], set[str]] = defaultdict(set)
        self._prop2superprop: dict[str, str] = {}

        self._reflexive_relations: list[str] = []
        self._irreflexive_relations: list[str] = []
        self._functional_relations: list[str] = []
        self._inversefunctional_relations: list[str] = []
        self._symmetric_relations: list[str] = []
        self._asymmetric_relations: list[str] = []
        self._transitive_relations: list[str] = []
        self._inverseof_relations: list[str] = []
        self._subproperties: list[str] = []

        # Profiling state.
        self._unprofiled_relations: dict[str, list[str]] = {}
        self._num_relations_wo_reflexive: int = 0
        self._current_profile_ratio: float = 0.0

        # Compatibility data loaded from resource files.
        self._one_rel_compatibilities: list[list[str]] = []
        self._compat_inverseof: dict[frozenset[str], list[frozenset[str]]] = {}

        self._compat_initialized: bool = False

    # ------------------------------------------------------------------------------------------------ #
    # Public API                                                                                       #
    # ------------------------------------------------------------------------------------------------ #

    def __repr__(self) -> str:
        """Return a debug-friendly representation of this generator."""
        return (
            "RelationGenerator("
            f"num_relations={self._config.num_relations}, "
            f"relation_specificity={self._config.relation_specificity}, "
            f"prop_profiled_relations={self._config.prop_profiled_relations}, "
            f"profile_side={self._config.profile_side!r}, "
            f"prop_symmetric_relations={self._config.prop_symmetric_relations}, "
            f"prop_inverse_relations={self._config.prop_inverse_relations}, "
            f"prop_functional_relations={self._config.prop_functional_relations}, "
            f"prop_inverse_functional_relations={self._config.prop_inverse_functional_relations}, "
            f"prop_transitive_relations={self._config.prop_transitive_relations}, "
            f"prop_subproperties={self._config.prop_subproperties}, "
            f"prop_reflexive_relations={self._config.prop_reflexive_relations}, "
            f"prop_irreflexive_relations={self._config.prop_irreflexive_relations}, "
            f"prop_asymmetric_relations={self._config.prop_asymmetric_relations}, "
            f"rng_seed={self._config.rng_seed}, "
            ")"
        )

    def generate_relation_schema(self) -> RelationInfoDict:
        """Generate the full relation schema and return a structured summary.

        This is the main public entry point. It will:
        - Initialize relations and internal state.
        - Load property compatibility tables from resource files.
        - Assign property patterns, inverses, and subproperty relations.
        - Profile domains and ranges guided by class information.
        - Optionally log a summary table.
        - Return a RelationInfoDict mapping with all relevant metadata.

        Returns:
            A RelationInfoDict structure describing the generated relations and
            their properties.
        """
        self._initialize_from_class_info()
        self._initialize_relations()
        self._ensure_compatibilities_loaded()
        self._generate_relations()

        relation_info: RelationInfoDict = self._assemble_relation_info()
        return relation_info

    # ------------------------------------------------------------------------------------------------ #
    # Internal Helpers: Initialization                                                                 #
    # ------------------------------------------------------------------------------------------------ #

    def _initialize_from_class_info(self) -> None:
        """Cache class-level structures from the provided ClassInfoDict."""
        self._class2disjoints_extended = dict(self._class_info["class2disjoints_extended"])
        self._layer2classes = {
            int(layer): list(classes)
            for layer, classes in self._class_info["layer2classes"].items()
        }
        self._class2layer = dict(self._class_info["class2layer"])

    def _initialize_relations(self) -> None:
        """Initialize relation identifiers and basic mappings."""
        self._relations = [f"R{i}" for i in range(1, self._config.num_relations + 1)]
        self._rel2dom = {}
        self._rel2range = {}
        self._rel2inverse = {}
        self._inverseof_relations = []

        self._rel2patterns = {}
        self._pattern2rels = defaultdict(set)
        self._prop2superprop = {}

        self._reflexive_relations = []
        self._irreflexive_relations = []
        self._functional_relations = []
        self._inversefunctional_relations = []
        self._symmetric_relations = []
        self._asymmetric_relations = []
        self._transitive_relations = []
        self._subproperties = []

        self._unprofiled_relations = {}
        self._num_relations_wo_reflexive = 0
        self._current_profile_ratio = 0.0

    def _ensure_compatibilities_loaded(self) -> None:
        """Load property compatibility tables from resource files if needed."""
        if self._compat_initialized:
            return

        self._load_one_rel_compatibilities()
        self._load_inverseof_compatibilities()
        self._compat_initialized = True

    def _load_one_rel_compatibilities(self) -> None:
        """Load all valid combinations of relation properties from JSON.

        Uses importlib.resources.Traversable objects directly, which is safe for
        both filesystem and zipped/packaged installations.
        """
        # Stubs for importlib.resources/Traversable are not perfectly precise in all
        # environments, so we treat the resource handle as Any at this boundary.
        combinations_resource = cast(
            Any,
            resources.files("pygraft").joinpath("resources/property_checks/combinations.json"),
        )

        try:
            # Runtime behavior is correct; we simply avoid over-constraining the
            # type of the file handle so Pyright doesn't propagate Unknown.
            with combinations_resource.open("r", encoding="utf8") as file:
                raw_data = json.load(file)
        except FileNotFoundError as exc:
            message = (
                "Could not load relation property combinations from packaged resources "
                "(property_checks/combinations.json). Ensure the package is installed "
                "with its data files."
            )
            raise RuntimeError(message) from exc

        data: dict[str, str] = cast(dict[str, str], raw_data)

        compatibilities = [key for key, value in data.items() if value == "True"]
        self._one_rel_compatibilities = [op.split(",") for op in compatibilities]

    def _load_inverseof_compatibilities(self) -> None:
        """Load all valid combinations of inverse relations from text file.

        As with _load_one_rel_compatibilities, this uses Traversable.open to remain
        robust across different packaging formats.
        """
        self._compat_inverseof = {}

        inverseof_resource = cast(
            Any,
            resources.files("pygraft").joinpath(
                "resources/property_checks/compat_p1p2_inverseof.txt",
            ),
        )

        try:
            # Keep the file handle loosely typed at this boundary.
            with inverseof_resource.open("r", encoding="utf8") as file:
                for line in file:
                    stripped = line.strip()
                    if "True" not in stripped:
                        continue
                    tokens = stripped.split(" ")
                    pattern_part = tokens[0][:-1]
                    v1_raw, v2_raw = pattern_part.split("|")
                    v1: frozenset[str] = frozenset(v1_raw.strip().split(","))
                    v2: frozenset[str] = frozenset(v2_raw.strip().split(","))
                    self._compat_inverseof.setdefault(v1, []).append(v2)
        except FileNotFoundError as exc:
            message = (
                "Could not load inverse-of compatibility data from packaged resources "
                "(property_checks/compat_p1p2_inverseof.txt). Ensure the package is "
                "installed with its data files."
            )
            raise RuntimeError(message) from exc

    # ------------------------------------------------------------------------------------------------ #
    # Internal Helpers: Main Generation Pipeline                                                       #
    # ------------------------------------------------------------------------------------------------ #

    def _generate_relations(self) -> None:
        """Generate relations and add various properties to them."""
        if self._relations is None:
            message = "Relations have not been initialized."
            raise RuntimeError(message)

        # Initialize patterns.
        self._rel2patterns = {rel: set() for rel in self._relations}
        self._pattern2rels = defaultdict(set)
        for pattern in self._one_rel_compatibilities:
            self._pattern2rels[frozenset(pattern)] = set()

        # Reflexive / Irreflexive.
        reflexive_sample_size = int(
            len(self._relations) * self._config.prop_reflexive_relations,
        )
        self._reflexive_relations = self._rng.sample(self._relations, k=reflexive_sample_size)
        self._update_rel2patterns("owl:Reflexive")

        irreflexive_pool = list(set(self._relations) - set(self._reflexive_relations))
        irreflexive_sample_size = int(
            len(self._relations) * self._config.prop_irreflexive_relations,
        )
        irreflexive_sample_size = min(irreflexive_sample_size, len(irreflexive_pool))
        self._irreflexive_relations = self._rng.sample(irreflexive_pool, k=irreflexive_sample_size)
        self._update_rel2patterns("owl:Irreflexive")

        # Other properties.
        self._add_property("owl:Symmetric")
        self._add_property("owl:Asymmetric")
        self._add_property("owl:Transitive")
        self._add_property("owl:Functional")
        self._add_property("owl:InverseFunctional")

        # Inverse-of pairs.
        self._add_inverseof()

        # Profiling: domains / ranges (reflexive relations cannot be profiled).
        eligible = [
            rel for rel, patterns in self._rel2patterns.items() if "owl:Reflexive" not in patterns
        ]
        self._unprofiled_relations["both"] = list(eligible)
        self._unprofiled_relations["dom"] = list(eligible)
        self._unprofiled_relations["range"] = list(eligible)
        self._num_relations_wo_reflexive = len(eligible)

        self._current_profile_ratio = 0.0
        while (
            self._current_profile_ratio < self._config.prop_profiled_relations
            and self._num_relations_wo_reflexive > 0
        ):
            self._add_one_relation_profile()
            self._current_profile_ratio = (len(self._rel2dom) + len(self._rel2range)) / (
                2 * self._num_relations_wo_reflexive
            )

        # Subproperty hierarchy.
        self._add_property("rdfs:subPropertyOf")

    # ------------------------------------------------------------------------------------------------ #
    # Schema Assembly Helpers                                                                          #
    # ------------------------------------------------------------------------------------------------ #

    def _assemble_relation_info(self) -> RelationInfoDict:
        """Assemble and return a RelationInfoDict summary of the current state."""
        if self._relations is None:
            message = "Relations have not been generated yet."
            raise RuntimeError(message)

        num_relations = len(self._relations)

        # --- statistics ---
        prop_reflexive = round(len(self._reflexive_relations) / num_relations, 2)
        prop_irreflexive = round(len(self._irreflexive_relations) / num_relations, 2)
        prop_functional = round(len(self._functional_relations) / num_relations, 2)
        prop_inversefunctional = round(
            len(self._inversefunctional_relations) / num_relations,
            2,
        )
        prop_symmetric = round(len(self._symmetric_relations) / num_relations, 2)
        prop_asymmetric = round(len(self._asymmetric_relations) / num_relations, 2)
        prop_transitive = round(len(self._transitive_relations) / num_relations, 2)
        prop_inverseof = round(len(self._inverseof_relations) / num_relations, 2)
        prop_subpropertyof = round(2 * len(self._prop2superprop) / num_relations, 2)
        prop_profiled_relations = round(self._current_profile_ratio, 2)
        relation_specificity = round(self._calculate_relation_specificity(), 2)

        statistics: RelationStatistics = {
            "num_relations": num_relations,
            "prop_reflexive": prop_reflexive,
            "prop_irreflexive": prop_irreflexive,
            "prop_functional": prop_functional,
            "prop_inversefunctional": prop_inversefunctional,
            "prop_symmetric": prop_symmetric,
            "prop_asymmetric": prop_asymmetric,
            "prop_transitive": prop_transitive,
            "prop_inverseof": prop_inverseof,
            "prop_subpropertyof": prop_subpropertyof,
            "prop_profiled_relations": prop_profiled_relations,
            "relation_specificity": relation_specificity,
        }

        # --- relation list ---
        relations: list[str] = list(self._relations)

        # --- OWL pattern mappings ---
        rel2patterns = {rel: set(patterns) for rel, patterns in self._rel2patterns.items()}

        # --- OWL logical characteristics (per property) ---
        reflexive_relations = list(self._reflexive_relations)
        irreflexive_relations = list(self._irreflexive_relations)
        symmetric_relations = list(self._symmetric_relations)
        asymmetric_relations = list(self._asymmetric_relations)
        functional_relations = list(self._functional_relations)
        inversefunctional_relations = list(self._inversefunctional_relations)
        transitive_relations = list(self._transitive_relations)

        # --- OWL inverse-of relationships ---
        inverseof_relations = list(self._inverseof_relations)
        rel2inverse = dict(self._rel2inverse)

        # --- RDFS subPropertyOf hierarchy ---
        subrelations = list(self._subproperties)
        rel2superrel: dict[str, list[str]] = {
            subprop: [superprop] for subprop, superprop in self._prop2superprop.items()
        }

        # --- disjointness mappings (placeholder until implemented) ---
        rel2disjoints: dict[str, list[str]] = {}
        rel2disjoints_symmetric: list[str] = []
        rel2disjoints_extended: dict[str, list[str]] = {}

        # --- RDFS/OWL domain and range ---
        rel2dom_lists = {rel: [dom] for rel, dom in self._rel2dom.items()}
        rel2range_lists = {rel: [rng] for rel, rng in self._rel2range.items()}

        relation_info: RelationInfoDict = build_relation_info(
            statistics=statistics,
            relations=relations,
            rel2patterns=rel2patterns,
            reflexive_relations=reflexive_relations,
            irreflexive_relations=irreflexive_relations,
            symmetric_relations=symmetric_relations,
            asymmetric_relations=asymmetric_relations,
            functional_relations=functional_relations,
            inversefunctional_relations=inversefunctional_relations,
            transitive_relations=transitive_relations,
            inverseof_relations=inverseof_relations,
            rel2inverse=rel2inverse,
            subrelations=subrelations,
            rel2superrel=rel2superrel,
            rel2disjoints=rel2disjoints,
            rel2disjoints_symmetric=rel2disjoints_symmetric,
            rel2disjoints_extended=rel2disjoints_extended,
            rel2dom=rel2dom_lists,
            rel2range=rel2range_lists,
        )

        return relation_info

    # ------------------------------------------------------------------------------------------------ #
    # Profiling Helpers                                                                                #
    # ------------------------------------------------------------------------------------------------ #

    def _calculate_relation_specificity(self) -> float:
        """Calculate the current specificity of profiled relations."""
        if not self._rel2dom and not self._rel2range:
            return 0.0

        domains = list(self._rel2dom.values())
        ranges = list(self._rel2range.values())
        both = domains + ranges
        layers = [self._class2layer[class_name] for class_name in both]
        return float(np.mean(layers))

    def _add_one_relation_profile(self) -> None:
        """Add domain/range profile for a single relation."""
        if self._config.profile_side == "both":
            self._add_complete_relation_profile()
        elif self._config.profile_side == "partial":
            self._add_partial_relation_profile()

    def _add_partial_relation_profile(self) -> None:
        """Generate a partial relation profile by assigning domain or range.

        This picks a relation (domain or range side) that has not been
        profiled yet and assigns a sampled class. It propagates compatible
        profiling to symmetric/transitive and inverse relations when needed.
        """
        current_specificity = self._calculate_relation_specificity()
        sampled_class = self._sample_class(current_specificity)
        domain_or_range = "domain" if self._rng.random() < 0.5 else "range"

        if domain_or_range == "domain":
            relation = self._unprofiled_relations["dom"].pop(0)
            rel_patterns = self._rel2patterns[relation]
            self._rel2dom[relation] = sampled_class

            if "owl:Transitive" in rel_patterns or "owl:Symmetric" in rel_patterns:
                self._rel2range[relation] = sampled_class
                if relation in self._unprofiled_relations["range"]:
                    self._unprofiled_relations["range"].remove(relation)
                if relation in self._rel2inverse:
                    inverse_rel = self._rel2inverse[relation]
                    if inverse_rel in self._unprofiled_relations["dom"]:
                        self._unprofiled_relations["dom"].remove(inverse_rel)
                    self._rel2dom[inverse_rel] = self._rel2range[relation]

            if relation in self._rel2inverse:
                inverse_rel = self._rel2inverse[relation]
                if inverse_rel in self._unprofiled_relations["range"]:
                    self._unprofiled_relations["range"].remove(inverse_rel)
                self._rel2range[inverse_rel] = self._rel2dom[relation]

        else:
            relation = self._unprofiled_relations["range"].pop(0)
            rel_patterns = self._rel2patterns[relation]
            self._rel2range[relation] = sampled_class

            if "owl:Transitive" in rel_patterns or "owl:Symmetric" in rel_patterns:
                self._rel2dom[relation] = sampled_class
                if relation in self._unprofiled_relations["dom"]:
                    self._unprofiled_relations["dom"].remove(relation)
                if relation in self._rel2inverse:
                    inverse_rel = self._rel2inverse[relation]
                    if inverse_rel in self._unprofiled_relations["range"]:
                        self._unprofiled_relations["range"].remove(inverse_rel)
                    self._rel2range[inverse_rel] = self._rel2dom[relation]

            if relation in self._rel2inverse:
                inverse_rel = self._rel2inverse[relation]
                if inverse_rel in self._unprofiled_relations["dom"]:
                    self._unprofiled_relations["dom"].remove(inverse_rel)
                self._rel2dom[inverse_rel] = self._rel2range[relation]

    def _add_complete_relation_profile(self) -> None:
        """Generate a complete relation profile (domain and range)."""
        current_specificity = self._calculate_relation_specificity()
        sampled_domain = self._sample_class(current_specificity)

        relation = self._unprofiled_relations["both"].pop(0)
        rel_patterns = self._rel2patterns[relation]

        if "owl:Reflexive" in rel_patterns:
            return

        self._rel2dom[relation] = sampled_domain
        current_specificity = self._calculate_relation_specificity()

        if "owl:Transitive" in rel_patterns or "owl:Symmetric" in rel_patterns:
            self._rel2range[relation] = sampled_domain
        else:
            sampled_range = self._sample_class_constrained(current_specificity, sampled_domain)
            self._rel2range[relation] = sampled_range

        if relation in self._rel2inverse:
            inverse_rel = self._rel2inverse[relation]
            if inverse_rel in self._unprofiled_relations["both"]:
                self._unprofiled_relations["both"].remove(inverse_rel)

            self._rel2dom[inverse_rel] = self._rel2range[relation]
            self._rel2range[inverse_rel] = self._rel2dom[relation]

    def _sample_class(self, current_specificity: float) -> str:
        """Sample a class so specificity converges toward the target.

        This function is defensive against empty candidate sets: if filtering
        unexpectedly yields no candidates (even after internal fallbacks),
        it raises a clear RuntimeError rather than failing with IndexError.
        """
        potential_classes = self._filter_classes(current_specificity)
        if not potential_classes:
            message = (
                "No classes available for sampling after filtering; this likely "
                "indicates that the class hierarchy or configuration is "
                "inconsistent with relation_specificity."
            )
            raise RuntimeError(message)

        return self._rng.choice(potential_classes)

    def _sample_class_constrained(self, current_specificity: float, other_class: str) -> str:
        """Sample a class compatible with another class (not disjoint).

        This method is defensive against pathological disjointness patterns:
        it tries a bounded number of specificity-guided samples and, if no
        compatible class is found, falls back to any non-disjoint class in
        the hierarchy before ultimately raising a clear RuntimeError.
        """
        incompatible: set[str] = set(self._class2disjoints_extended.get(other_class, []))
        max_attempts: int = 1000

        # First, try with the specificity-guided filter.
        for _ in range(max_attempts):
            potential_classes = self._filter_classes(current_specificity)
            if not potential_classes:
                break

            compatible_classes = [
                class_name for class_name in potential_classes if class_name not in incompatible
            ]
            if compatible_classes:
                return self._rng.choice(compatible_classes)

        # Fallback: ignore specificity and try any non-disjoint class.
        all_classes = list(itertools.chain.from_iterable(self._layer2classes.values()))
        if not all_classes:
            message = (
                "No classes available in layer2classes while attempting constrained "
                "sampling; class_info may be inconsistent."
            )
            raise RuntimeError(message)

        relaxed_compatible = [
            class_name for class_name in all_classes if class_name not in incompatible
        ]
        if relaxed_compatible:
            logger.warning(
                "Falling back to relaxed constrained sampling for %s: using any "
                "non-disjoint class after %d unsuccessful specificity-guided "
                "attempts.",
                other_class,
                max_attempts,
            )
            return self._rng.choice(relaxed_compatible)

        message = (
            f"Could not find a class compatible with {other_class!r} after "
            f"{max_attempts} attempts (including relaxed sampling). This likely "
            "indicates that extended disjointness constraints are too strong for "
            "the current configuration."
        )
        raise RuntimeError(message)

    def _filter_classes(self, current_specificity: float) -> list[str]:
        """Filter classes based on current relation specificity.

        This steers the average depth towards the configured relation_specificity,
        adding a small amount of noise to avoid degenerate behavior.

        The implementation is defensive: if filtering yields no candidates for the
        current specificity band, it falls back to using all known classes rather
        than returning an empty list (which would later cause crashes or loops).
        """
        if not self._layer2classes:
            message = "layer2classes is empty; class_info may be inconsistent."
            raise RuntimeError(message)

        target_layer = int(self._config.relation_specificity)

        if current_specificity < self._config.relation_specificity:
            filtered_by_layer = [
                classes for layer, classes in self._layer2classes.items() if layer > target_layer
            ]
            if self._rng.random() < 0.1:
                filtered_by_layer = [
                    classes
                    for layer, classes in self._layer2classes.items()
                    if layer <= target_layer
                ]
        else:
            filtered_by_layer = [
                classes for layer, classes in self._layer2classes.items() if layer <= target_layer
            ]
            if self._rng.random() < 0.1:
                filtered_by_layer = [
                    classes
                    for layer, classes in self._layer2classes.items()
                    if layer > target_layer
                ]

        flattened_candidates = list(itertools.chain.from_iterable(filtered_by_layer))

        if not flattened_candidates:
            # Fallback: avoid returning an empty list, which would lead to
            # crashes in downstream sampling code. This typically happens only
            # for pathological configurations (e.g. relation_specificity is
            # greater than the maximum class depth).
            all_classes = list(itertools.chain.from_iterable(self._layer2classes.values()))
            if not all_classes:
                message = (
                    "No classes available in layer2classes after filtering; this "
                    "likely indicates an inconsistent class_info or configuration."
                )
                raise RuntimeError(message)

            logger.debug(
                "Class filtering produced no candidates for "
                "current_specificity=%s and relation_specificity=%s; falling back "
                "to all %d available classes.",
                current_specificity,
                self._config.relation_specificity,
                len(all_classes),
            )
            return all_classes

        return flattened_candidates

    # ------------------------------------------------------------------------------------------------ #
    # Property Patterns And Compatibilities                                                            #
    # ------------------------------------------------------------------------------------------------ #

    def _update_rel2patterns(self, property_iri: str) -> None:
        """Update rel2patterns for the given property and refresh pattern2rels."""
        property_mappings: dict[str, list[str]] = {
            "owl:Reflexive": self._reflexive_relations,
            "owl:Irreflexive": self._irreflexive_relations,
            "owl:Symmetric": self._symmetric_relations,
            "owl:Asymmetric": self._asymmetric_relations,
            "owl:Transitive": self._transitive_relations,
        }

        affected_relations = property_mappings.get(property_iri)
        if affected_relations is None:
            # Nothing special to do for properties not tracked in property_mappings;
            # we still keep pattern2rels in sync.
            self._update_pattern2rels()
            return

        for relation in affected_relations:
            # rel2patterns is initialized with all relations, so this should
            # always succeed without needing defensive defaults.
            self._rel2patterns[relation].add(property_iri)

        self._update_pattern2rels()

    def _update_pattern2rels(self) -> None:
        """Rebuild the pattern2rels mapping from rel2patterns."""
        self._pattern2rels = defaultdict(set)
        for rel, pattern_set in self._rel2patterns.items():
            self._pattern2rels[frozenset(pattern_set)].add(rel)

    def _add_property(self, property_iri: str) -> None:
        """Assign a given property to a subset of relations when compatible."""
        combinations_with_property = [
            combi for combi in self._one_rel_compatibilities if property_iri in combi
        ]
        combinations_without_property = [
            [item for item in combi if item != property_iri] for combi in combinations_with_property
        ]

        # Build a concrete set[str] pool without using an untyped set() as base.
        empty_set: set[str] = set()
        relation_pool: set[str] = set()
        for combination in combinations_without_property:
            compatible_rels = self._pattern2rels.get(frozenset(combination), empty_set)
            relation_pool.update(compatible_rels)

        if property_iri == "owl:Functional":
            self._assign_functional_relations()
        elif property_iri == "owl:InverseFunctional":
            self._assign_inversefunctional_relations()
        elif property_iri == "rdfs:subPropertyOf":
            self._assign_subproperty_relations()
        else:
            self._assign_simple_property(property_iri, relation_pool)

        self._update_rel2patterns(property_iri)

    def _assign_functional_relations(self) -> None:
        """Assign the Functional property to a subset of relations."""
        self._functional_relations = []

        potential_relations = [rel for rel, patterns in self._rel2patterns.items() if not patterns]

        target_count = int(self._config.prop_functional_relations * self._config.num_relations)
        target_count = min(target_count, len(potential_relations))

        if target_count <= 0:
            return

        sampled_relations = self._rng.sample(potential_relations, k=target_count)
        self._functional_relations.extend(sampled_relations)

        for relation in sampled_relations:
            self._rel2patterns[relation] = set(self._rel2patterns[relation]) | {"owl:Functional"}

    def _assign_inversefunctional_relations(self) -> None:
        """Assign the InverseFunctional property to a subset of relations."""
        self._inversefunctional_relations = []

        # First pass: relations with no pattern.
        base_potential = [rel for rel, patterns in self._rel2patterns.items() if not patterns]
        random_scaling = self._rng.uniform(0.25, 0.75)
        target_first_phase = int(
            random_scaling
            * self._config.prop_inverse_functional_relations
            * self._config.num_relations,
        )
        first_phase_count = min(target_first_phase, len(base_potential))

        if first_phase_count > 0:
            first_phase_sample = self._rng.sample(base_potential, k=first_phase_count)
            self._inversefunctional_relations.extend(first_phase_sample)
            for relation in first_phase_sample:
                self._rel2patterns[relation] = set(self._rel2patterns[relation]) | {
                    "owl:InverseFunctional"
                }

        # Second pass: relations that are at most Functional.
        potential_relations = [
            rel
            for rel, patterns in self._rel2patterns.items()
            if not patterns - {"owl:Functional"} and rel not in self._inversefunctional_relations
        ]

        final_target = int(self._config.prop_inverse_functional_relations * self._config.num_relations)
        remaining_needed = max(0, final_target - len(self._inversefunctional_relations))
        remaining_needed = min(remaining_needed, len(potential_relations))

        if remaining_needed <= 0:
            return

        second_phase_sample = self._rng.sample(potential_relations, k=remaining_needed)
        self._inversefunctional_relations.extend(second_phase_sample)
        for relation in second_phase_sample:
            self._rel2patterns[relation] = set(self._rel2patterns[relation]) | {
                "owl:InverseFunctional"
            }

    def _assign_subproperty_relations(self) -> None:
        """Assign subPropertyOf relationships between compatible relations."""
        self._prop2superprop = {}
        self._subproperties = []

        if self._relations is None:
            return

        relations_copy = list(self._relations)

        for r1 in relations_copy:
            for r2 in relations_copy:
                if (
                    r1 == r2
                    or r1 in self._prop2superprop
                    or r2 in self._prop2superprop
                    or self._rel2inverse.get(r1) == r2
                    or self._rel2inverse.get(r2) == r1
                    or self._rel2patterns[r1] != self._rel2patterns[r2]
                ):
                    continue

                # Case (0): no domain/range for either relation.
                if (
                    r1 not in self._rel2dom
                    and r1 not in self._rel2range
                    and r2 not in self._rel2dom
                    and r2 not in self._rel2range
                ):
                    self._prop2superprop[r1] = r2
                    self._subproperties.append(r1)
                    break

                # Case (1): both have full domain/range.
                if (
                    r1 in self._rel2dom
                    and r1 in self._rel2range
                    and r2 in self._rel2dom
                    and r2 in self._rel2range
                ):
                    dom1 = self._rel2dom[r1]
                    dom2 = self._rel2dom[r2]
                    range1 = self._rel2range[r1]
                    range2 = self._rel2range[r2]

                    trans_sup = self._class_info["transitive_class2superclasses"]

                    # (1) same domain and same range.
                    if dom1 == dom2 and range1 == range2:
                        self._prop2superprop[r1] = r2
                        self._subproperties.append(r1)
                        break

                    # (2) same domain, range2 is superclass of range1.
                    if dom1 == dom2 and range2 in trans_sup[range1]:
                        self._prop2superprop[r1] = r2
                        self._subproperties.append(r1)
                        break

                    # (3) same range, dom2 is superclass of dom1.
                    if range1 == range2 and dom2 in trans_sup[dom1]:
                        self._prop2superprop[r1] = r2
                        self._subproperties.append(r1)
                        break

                    # (4) dom2 and range2 are superclasses of dom1 and range1.
                    if dom2 in trans_sup[dom1] and range2 in trans_sup[range1]:
                        self._prop2superprop[r1] = r2
                        self._subproperties.append(r1)
                        break

            if (
                2 * len(self._prop2superprop)
                >= self._config.prop_subproperties * self._config.num_relations
            ):
                return

    def _assign_simple_property(self, property_iri: str, relation_pool: set[str]) -> None:
        """Assign a simple pattern (Symmetric, Asymmetric, Transitive).

        The caller is responsible for constructing `relation_pool` as a set of
        relation identifiers that are compatible with the given property.
        """
        if self._relations is None:
            return

        total_relations = len(self._relations)
        pool_list = list(relation_pool)

        if not pool_list:
            return

        if property_iri == "owl:Symmetric":
            sample_size = int(total_relations * self._config.prop_symmetric_relations)
            sample_size = min(sample_size, len(pool_list))
            self._symmetric_relations = self._rng.sample(pool_list, k=sample_size)

        elif property_iri == "owl:Asymmetric":
            sample_size = int(total_relations * self._config.prop_asymmetric_relations)
            sample_size = min(sample_size, len(pool_list))
            self._asymmetric_relations = self._rng.sample(pool_list, k=sample_size)

        elif property_iri == "owl:Transitive":
            sample_size = int(total_relations * self._config.prop_transitive_relations)
            sample_size = min(sample_size, len(pool_list))
            self._transitive_relations = self._rng.sample(pool_list, k=sample_size)

    # ------------------------------------------------------------------------------------------------ #
    # Inverse-of Handling                                                                              #
    # ------------------------------------------------------------------------------------------------ #

    def _add_inverseof(self) -> None:
        """Determine and add inverse relations based on compatibilities.

        The implementation is careful to:

        - Pair relations with no pattern first.
        - Use the compatibility tables for the remaining relations.
        - Avoid blanket exception swallowing: incompatible or uncovered patterns
          are handled explicitly via backoff, while true internal inconsistencies
          are allowed to surface as clear errors.
        """
        if self._relations is None:
            message = "Relations have not been initialized."
            raise RuntimeError(message)

        observed_patterns = [frozenset(patterns) for patterns in self._rel2patterns.values()]
        running_inverse_prop = 0.0
        attempt_counter = 0
        warning_emitted = False

        # First, pair relations with no pattern.
        unpatterned_relations = [rel for rel in self._relations if not self._rel2patterns[rel]]
        if len(unpatterned_relations) % 2 == 1:
            unpatterned_relations = unpatterned_relations[:-1]

        effective_target = self._config.prop_inverse_relations

        while running_inverse_prop < effective_target and len(unpatterned_relations) >= 2:
            first_rel = unpatterned_relations.pop()
            second_rel = unpatterned_relations.pop()
            self._pair_inverseof(first_rel, second_rel)
            running_inverse_prop = self._calculate_inverseof()

        # Then, use compatibility tables.
        while running_inverse_prop < effective_target:
            attempt_counter += 1

            first_pattern = self._rng.choice(observed_patterns)
            compatible_patterns = set(self._compat_inverseof.get(first_pattern, []))

            if not compatible_patterns:
                if attempt_counter > 1000:
                    effective_target = max(0.0, effective_target - 0.005)
                    attempt_counter = 0
                    warning_emitted = True
                continue

            possible_patterns = set(observed_patterns).intersection(compatible_patterns)
            if not possible_patterns:
                if attempt_counter > 1000:
                    effective_target = max(0.0, effective_target - 0.005)
                    attempt_counter = 0
                    warning_emitted = True
                continue

            second_pattern = self._rng.choice(list(possible_patterns))

            # If these lookups fail, we let the resulting KeyError/IndexError propagate,
            # because it indicates an internal inconsistency between the observed
            # patterns and the compatibility tables.
            first_rel = self._rng.choice(list(self._pattern2rels[first_pattern]))
            second_rel = self._rng.choice(list(self._pattern2rels[second_pattern]))

            if (
                first_rel != second_rel
                and first_rel not in self._rel2inverse
                and second_rel not in self._rel2inverse
                and "owl:Reflexive" not in self._rel2patterns[first_rel]
                and "owl:Reflexive" not in self._rel2patterns[second_rel]
                and "owl:Irreflexive" not in self._rel2patterns[first_rel]
                and "owl:Irreflexive" not in self._rel2patterns[second_rel]
                and "owl:Symmetric" not in self._rel2patterns[first_rel]
                and "owl:Symmetric" not in self._rel2patterns[second_rel]
                and not (
                    "owl:Asymmetric" in self._rel2patterns[first_rel]
                    and "owl:Asymmetric" in self._rel2patterns[second_rel]
                )
            ):
                self._pair_inverseof(first_rel, second_rel)
                running_inverse_prop = self._calculate_inverseof()
                attempt_counter = 0
            else:
                if attempt_counter > 1000:
                    effective_target = max(0.0, effective_target - 0.005)
                    attempt_counter = 0
                    warning_emitted = True

        if warning_emitted:
            logger.warning(
                "Proportion of inverse relations reduced due to incompatibilities "
                "with other properties.",
            )

    def _pair_inverseof(self, rel: str, inv_rel: str) -> None:
        """Pair two relations as inverses of each other."""
        self._inverseof_relations.append(rel)
        self._inverseof_relations.append(inv_rel)
        self._rel2inverse[rel] = inv_rel
        self._rel2inverse[inv_rel] = rel

    def _calculate_inverseof(self) -> float:
        """Calculate the proportion of inverse relations."""
        if self._relations is None or not self._relations:
            return 0.0
        return len(self._inverseof_relations) / len(self._relations)
