#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Schema loading, validation, and constraint cache initialization."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from pygraft.generators.kg.structures import ConstraintCaches, SchemaMetadata
from pygraft.types import ClassInfoDict, RelationInfoDict

if TYPE_CHECKING:
    from pathlib import Path

    from pygraft.generators.kg.types import ClassIdFrozenSet, ClassIdSet

logger = logging.getLogger(__name__)


class SchemaLoader:
    """Load and validate ontology schema, build constraint caches.

    Handles the schema loading and preparation phase of KG generation:
        1. Load JSON files (class_info, relation_info, namespaces_info)
        2. Build bidirectional ID mappings
        3. Initialize class hierarchy caches
        4. Initialize relation constraint caches
        5. Validate schema for forbidden OWL combinations
        6. Compute disjoint envelopes for fast rejection

    Attributes:
        schema: String-to-integer ID mappings.
        constraints: Pre-computed constraint caches.
        class_info: Loaded class_info.json contents.
        relation_info: Loaded relation_info.json contents.
        ontology_prefix: Ontology namespace prefix.
        ontology_namespace: Ontology namespace URI.
        prefix2namespace: Prefix to namespace URI mappings.
    """

    def __init__(self, *, output_directory_path: Path) -> None:
        """Initialize the schema loader.

        Args:
            output_directory_path: Path to project output folder containing JSON files.
        """
        self._output_directory_path = output_directory_path

        # Outputs (populated by load())
        self.schema = SchemaMetadata()
        self.constraints = ConstraintCaches()
        self.class_info: ClassInfoDict = cast(ClassInfoDict, {})
        self.relation_info: RelationInfoDict = cast(RelationInfoDict, {})
        self.ontology_prefix = "sc"
        self.ontology_namespace = "http://pygraf.t/"
        self.prefix2namespace: dict[str, str] = {}

    def load(self) -> None:
        """Run the full schema loading and preparation pipeline.

        Raises:
            ValueError: If schema contains SEVERE constraint violations.
            FileNotFoundError: If required JSON files are missing.
        """
        # Step 1: Load JSON files
        self._load_class_info()
        self._load_relation_info()
        self._load_namespaces_info()

        # Step 2: Build internal structures
        self._build_id_mappings()
        self._initialize_class_caches()
        self._initialize_relation_caches()

        # Step 3: Validate and optimize
        self._validate_schema_constraints()
        self._compute_disjoint_envelopes()

    # ================================================================================================ #
    # SCHEMA LOADING                                                                                   #
    # ================================================================================================ #

    def _load_class_info(self) -> None:
        """Load class_info.json into class_info.

        Contains class hierarchy, disjointness constraints, and layer mappings
        extracted from the ontology.
        """
        class_info_path = self._output_directory_path / "class_info.json"
        with class_info_path.open("r", encoding="utf8") as file:
            self.class_info = cast(ClassInfoDict, json.load(file))

    def _load_relation_info(self) -> None:
        """Load relation_info.json into relation_info.

        Contains relation characteristics, domain/range constraints, inverse
        mappings, and subproperty hierarchy extracted from the ontology.
        """
        relation_info_path = self._output_directory_path / "relation_info.json"
        with relation_info_path.open("r", encoding="utf8") as file:
            self.relation_info = cast(RelationInfoDict, json.load(file))

    def _load_namespaces_info(self) -> None:
        """Load namespaces_info.json if present.

        Enables CURIE-aware serialization where "prefix:LocalName" tokens are
        expanded using declared namespace mappings. Falls back to legacy
        serialization if file is missing.

        The "_empty_prefix" sentinel is normalized to "" for Turtle compatibility.
        """
        namespaces_path = self._output_directory_path / "namespaces_info.json"
        if not namespaces_path.exists():
            logger.debug("No namespaces_info.json found at: %s", namespaces_path)
            return

        with namespaces_path.open("r", encoding="utf8") as file:
            namespaces_data = cast(dict[str, Any], json.load(file))

        # Extract ontology configuration
        ontology_info = namespaces_data.get("ontology", {})
        self.ontology_prefix = ontology_info.get("prefix", self.ontology_prefix)
        self.ontology_namespace = ontology_info.get("namespace", self.ontology_namespace)

        # Extract prefix mappings
        prefix_mappings = namespaces_data.get("prefixes", {})
        self.prefix2namespace = dict(prefix_mappings)

        # Normalize sentinel empty-prefix key to real Turtle empty prefix
        empty_prefix_namespace = self.prefix2namespace.get("_empty_prefix")
        if empty_prefix_namespace:
            self.prefix2namespace[""] = empty_prefix_namespace

    # ================================================================================================ #
    # INTERNAL PREPARATION - ID MAPPINGS                                                               #
    # ================================================================================================ #

    def _build_id_mappings(self) -> None:
        """Build bidirectional mappings between schema strings and integer IDs.

        Populates schema with class2id, id2class, rel2id, and id2rel from
        loaded class_info and relation_info.

        Complexity:
            O(n_c + n_r) where n_c is class count and n_r is relation count.
        """
        classes = list(self.class_info["classes"])
        self.schema.class2id = {name: idx for idx, name in enumerate(classes)}
        self.schema.id2class = classes

        relations = list(self.relation_info["relations"])
        self.schema.rel2id = {name: idx for idx, name in enumerate(relations)}
        self.schema.id2rel = relations

        logger.debug(
            "Built ID mappings: %d classes, %d relations",
            len(self.schema.id2class),
            len(self.schema.id2rel),
        )

    # ================================================================================================ #
    # INTERNAL PREPARATION - CLASS CACHES                                                              #
    # ================================================================================================ #

    def _initialize_class_caches(self) -> None:
        """Initialize class hierarchy constraint caches.

        Builds:
            - Layer mappings (depth-based class organization)
            - Disjointness constraints (class2disjoints_extended_ids)
            - Transitive superclass closures (transitive_supers_ids)

        Transitive supers include the class itself per RDFS reflexivity.
        Extended disjointness already includes inherited constraints from
        the extraction pipeline.

        Complexity:
            O(n_c * avg_superclass_depth) for transitive closure construction.
        """
        num_classes = len(self.schema.id2class)

        # Layer mappings (for depth-based sampling)
        self.constraints.layer2classes_ids = {}
        for layer_str, class_names in self.class_info["layer2classes"].items():
            layer_id = int(layer_str)
            self.constraints.layer2classes_ids[layer_id] = [
                self.schema.class2id[name] for name in class_names if name in self.schema.class2id
            ]

        self.constraints.class2layer_id = [0] * num_classes
        for class_name, layer in self.class_info["class2layer"].items():
            class_id = self.schema.class2id.get(class_name)
            if class_id is not None:
                self.constraints.class2layer_id[class_id] = int(layer)

        # Class disjointness constraints (already includes inherited disjointness)
        self.constraints.class2disjoints_extended_ids = [frozenset() for _ in range(num_classes)]
        for class_name, disjoint_names in self.class_info["class2disjoints_extended"].items():
            class_id = self.schema.class2id.get(class_name)
            if class_id is None:
                continue
            disjoint_ids = [
                self.schema.class2id[d] for d in disjoint_names if d in self.schema.class2id
            ]
            self.constraints.class2disjoints_extended_ids[class_id] = frozenset(disjoint_ids)

        # Transitive superclass closures
        # Note: JSON already includes self per RDFS rdfs:subClassOf* semantics
        self.constraints.transitive_supers_ids = [frozenset() for _ in range(num_classes)]
        for class_name, super_names in self.class_info["transitive_class2superclasses"].items():
            class_id = self.schema.class2id.get(class_name)
            if class_id is None:
                continue
            sup_ids = {self.schema.class2id[s] for s in super_names if s in self.schema.class2id}
            self.constraints.transitive_supers_ids[class_id] = frozenset(sup_ids)

        logger.debug("Initialized class hierarchy caches for %d classes", num_classes)

    # ================================================================================================ #
    # INTERNAL PREPARATION - RELATION CACHES                                                           #
    # ================================================================================================ #

    def _initialize_relation_caches(self) -> None:
        """Initialize relation constraint caches.

        Builds:
            - OWL property characteristic sets (functional, symmetric, etc.)
            - Domain/range class ID mappings
            - Inverse relation mappings
            - Subproperty hierarchy mappings
            - Relation disjointness (including inherited via subproperty chains)

        Complexity:
            O(n_r * avg_constraint_count) where n_r is relation count.
        """
        num_relations = len(self.schema.id2rel)

        # Domain/range type constraints
        self.constraints.rel2dom_ids = self._build_relation_class_constraints(
            self.relation_info["rel2dom"], num_relations
        )
        self.constraints.rel2range_ids = self._build_relation_class_constraints(
            self.relation_info["rel2range"], num_relations
        )

        # Relation disjointness constraints
        self.constraints.rel2disjoints_extended_ids = [frozenset() for _ in range(num_relations)]
        for rel_name, disjoint_names in self.relation_info.get(
            "rel2disjoints_extended", {}
        ).items():
            rel_id = self.schema.rel2id.get(rel_name)
            if rel_id is None:
                continue
            disjoint_ids = [
                self.schema.rel2id[r] for r in disjoint_names if r in self.schema.rel2id
            ]
            self.constraints.rel2disjoints_extended_ids[rel_id] = frozenset(disjoint_ids)

        # Relation characteristics (property patterns)

        # Reflexive: forall x: R(x,x) - not enforced (semantically redundant, inferred by reasoners)
        self.constraints.reflexive_ids = {
            self.schema.rel2id[r]
            for r in self.relation_info.get("reflexive_relations", [])
            if r in self.schema.rel2id
        }

        # Irreflexive: forall x: !R(x,x) - forbids self-loops
        self.constraints.irreflex_ids = {
            self.schema.rel2id[r]
            for r in self.relation_info.get("irreflexive_relations", [])
            if r in self.schema.rel2id
        }

        # Asymmetric: R(x,y) -> !R(y,x) - forbids bidirectional edges, implies irreflexive
        self.constraints.asym_ids = {
            self.schema.rel2id[r]
            for r in self.relation_info.get("asymmetric_relations", [])
            if r in self.schema.rel2id
        }
        # Asymmetric implies irreflexive (merge into irreflexive set)
        self.constraints.irreflex_ids |= self.constraints.asym_ids

        # Symmetric: R(x,y) <-> R(y,x) - bidirectional but store only one direction
        self.constraints.symmetric_ids = {
            self.schema.rel2id[r]
            for r in self.relation_info.get("symmetric_relations", [])
            if r in self.schema.rel2id
        }

        # Transitive: R(x,y) & R(y,z) -> R(x,z) - cycle detection for irreflexive/asymmetric combos
        self.constraints.transitive_ids = {
            self.schema.rel2id[r]
            for r in self.relation_info.get("transitive_relations", [])
            if r in self.schema.rel2id
        }

        # Functional: forall x exists-unique y: R(x,y) - each head has at most one tail
        self.constraints.functional_ids = {
            self.schema.rel2id[r]
            for r in self.relation_info.get("functional_relations", [])
            if r in self.schema.rel2id
        }

        # Inverse-functional: forall y exists-unique x: R(x,y) - each tail has at most one head
        self.constraints.invfunctional_ids = {
            self.schema.rel2id[r]
            for r in self.relation_info.get("inversefunctional_relations", [])
            if r in self.schema.rel2id
        }

        # Inverse relation mappings: owl:inverseOf - R1(x,y) <-> R2(y,x)
        self.constraints.rel2inverse_ids = {}
        for rel_name, inv_name in self.relation_info.get("rel2inverse", {}).items():
            rel_id = self.schema.rel2id.get(rel_name)
            inv_id = self.schema.rel2id.get(inv_name)
            if rel_id is not None and inv_id is not None:
                self.constraints.rel2inverse_ids[rel_id] = inv_id

        # Subproperty hierarchies: rdfs:subPropertyOf - inheritance of constraints
        self.constraints.rel2superrel_ids = {}
        for rel_name, super_names in self.relation_info.get("rel2superrel", {}).items():
            rel_id = self.schema.rel2id.get(rel_name)
            if rel_id is None:
                continue
            super_ids = [self.schema.rel2id[s] for s in super_names if s in self.schema.rel2id]
            self.constraints.rel2superrel_ids[rel_id] = super_ids

        logger.debug("Initialized relation constraint caches for %d relations", num_relations)

    def _build_relation_class_constraints(
        self,
        constraint_dict: dict[str, list[str]],
        num_relations: int,
    ) -> list[ClassIdFrozenSet]:
        """Convert relation domain/range constraints from strings to integer ID sets.

        Args:
            constraint_dict: Mapping from relation names to class name lists.
            num_relations: Total number of relations for pre-allocation.

        Returns:
            List indexed by relation ID, containing frozensets of class IDs.
        """
        result: list[frozenset[int]] = [frozenset() for _ in range(num_relations)]
        for rel_name, class_names in constraint_dict.items():
            rel_id = self.schema.rel2id.get(rel_name)
            if rel_id is None:
                continue
            result[rel_id] = frozenset(
                self.schema.class2id[name] for name in class_names if name in self.schema.class2id
            )
        return result

    # ================================================================================================ #
    # SCHEMA VALIDATION                                                                                #
    # ================================================================================================ #

    def _validate_schema_constraints(self) -> None:
        """Validate schema for forbidden OWL characteristic combinations.

        Checks five categories:
            1. Inverse mapping structure (bidirectional consistency, no self-inverses)
            2. Inverse characteristics (symmetric+inverseOf, inverseOf+disjoint+reflexive)
            3. Property characteristics (reflexive+irreflexive, symmetric+asymmetric)
            4. Domain/range constraints (disjoint class pairs, symmetric mismatches)
            5. Subproperty inheritance (symmetric sub of asymmetric super)

        SEVERE errors stop generation. WARNING errors exclude affected relations.

        Raises:
            ValueError: If schema contains SEVERE constraint violations.
        """
        severe_errors: list[str] = []

        # Category 1: Validate inverse mapping structure (SEVERE)
        try:
            self._validate_inverse_symmetry()
        except ValueError as e:
            severe_errors.append(str(e))

        # Category 2: Validate inverse characteristic consistency (SEVERE)
        inverse_errors = self._validate_inverse_characteristics()
        severe_errors.extend(inverse_errors)

        # Category 3: Validate property characteristics (SEVERE for contradictions, WARNING for OWL 2 DL issues)
        contradiction_errors = self._validate_property_characteristics()
        severe_errors.extend(contradiction_errors)

        # Category 4: Validate domain/range constraints (WARNING only)
        self._validate_domain_range_constraints()

        # Category 5: Validate subproperty inheritance conflicts (SEVERE)
        subproperty_errors = self._validate_subproperty_characteristics()
        severe_errors.extend(subproperty_errors)

        # If severe errors found, stop generation
        if severe_errors:
            error_list = "\n".join(f"  - {err}" for err in severe_errors)
            msg = (
                f"Schema validation failed with {len(severe_errors)} severe error(s):\n"
                f"{error_list}\n\n"
                f"These errors indicate logical contradictions or corrupted ontology structure.\n"
                f"Please fix the ontology before attempting generation."
            )
            raise ValueError(msg)

        # Log summary of excluded properties
        if self.constraints.unsatisfiable_relation_ids:
            logger.warning(
                "Excluded %d relation(s) due to validation issues (see warnings above).",
                len(self.constraints.unsatisfiable_relation_ids),
            )

    # --- Category 1: Relational Structure Validation ---
    def _validate_inverse_symmetry(self) -> None:
        """Validate that inverse relation mappings are symmetric.

        Ensures that if P inverseOf Q, then Q inverseOf P. Also prevents
        self-inverse relations (P inverseOf P).

        Raises:
            ValueError: If inverse mappings are malformed (indicates extraction bug).
        """
        for rid, inv_rid in self.constraints.rel2inverse_ids.items():
            rel_name = self.schema.id2rel[rid]
            inv_name = self.schema.id2rel[inv_rid]

            if rid == inv_rid:
                msg = f"Relation '{rel_name}' is declared as its own inverse (owl:inverseOf self)"
                raise ValueError(msg)

            if self.constraints.rel2inverse_ids.get(inv_rid) != rid:
                msg = (
                    f"Asymmetric inverse mapping: '{rel_name}' inverseOf '{inv_name}', "
                    f"but '{inv_name}' does not map back to '{rel_name}'"
                )
                raise ValueError(msg)

    # --- Category 2: Inverse Characteristic Validation ---
    def _validate_inverse_characteristics(self) -> list[str]:
        """Validate characteristic consistency for inverse relations.

        SEVERE checks:
            - Symmetric + InverseOf conflict: A symmetric property can only be
              self-inverse. If P is symmetric and P inverseOf Q where P != Q,
              this forces P=Q, contradicting distinct declarations.
            - InverseOf + Disjoint + Reflexive conflict: If P inverseOf Q and
              P disjoint Q, neither can be reflexive (self-loop in one implies
              self-loop in disjoint relation).

        Returns:
            List of severe error messages (empty if none found).
        """
        severe_errors: list[str] = []

        for rel_id, inv_id in self.constraints.rel2inverse_ids.items():
            rel_name = self.schema.id2rel[rel_id]
            inv_name = self.schema.id2rel[inv_id]

            # Skip if already marked unsatisfiable
            if rel_id in self.constraints.unsatisfiable_relation_ids:
                continue

            # Check 1: Symmetric + InverseOf conflict (when P != Q)
            if rel_id in self.constraints.symmetric_ids and rel_id != inv_id:
                msg = (
                    f"Relation '{rel_name}' is symmetric but has inverse '{inv_name}' (not itself).\n"
                    f"    Logical contradiction: symmetric requires P(x,y) <-> P(y,x), "
                    f"but inverseOf requires P(x,y) <-> Q(y,x).\n"
                    f"    This forces P=Q for all instances, contradicting the declaration of distinct relations.\n"
                    f"    Valid options: (1) Remove symmetric, (2) Remove inverseOf, or (3) Declare P inverseOf P."
                )
                logger.error(msg)
                severe_errors.append(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                self.constraints.unsatisfiable_relation_ids.add(inv_id)
                continue

            # Check 2: InverseOf + Disjoint + Reflexive conflict
            disjoint_rels = self.constraints.rel2disjoints_extended_ids[rel_id]

            if inv_id in disjoint_rels:
                # P and Q are inverse AND disjoint - check for reflexive declarations
                if rel_id in self.constraints.reflexive_ids:
                    msg = (
                        f"Relation '{rel_name}' cannot be reflexive.\n"
                        f"    '{rel_name}' is inverse of '{inv_name}' AND disjoint with '{inv_name}'.\n"
                        f"    If {rel_name}(x,x) exists, then {inv_name}(x,x) must exist (by inverse), "
                        f"but {rel_name} and {inv_name} are disjoint (cannot share instances).\n"
                        f"    This combination logically implies both relations must be irreflexive."
                    )
                    logger.error(msg)
                    severe_errors.append(msg)
                    self.constraints.unsatisfiable_relation_ids.add(rel_id)
                    self.constraints.unsatisfiable_relation_ids.add(inv_id)
                    continue

                if inv_id in self.constraints.reflexive_ids:
                    msg = (
                        f"Relation '{inv_name}' cannot be reflexive.\n"
                        f"    '{inv_name}' is inverse of '{rel_name}' AND disjoint with '{rel_name}'.\n"
                        f"    If {inv_name}(x,x) exists, then {rel_name}(x,x) must exist (by inverse), "
                        f"but {rel_name} and {inv_name} are disjoint (cannot share instances).\n"
                        f"    This combination logically implies both relations must be irreflexive."
                    )
                    logger.error(msg)
                    severe_errors.append(msg)
                    self.constraints.unsatisfiable_relation_ids.add(rel_id)
                    self.constraints.unsatisfiable_relation_ids.add(inv_id)
                    continue

        return severe_errors

    # --- Category 3: Property Characteristics Validation ---
    def _validate_property_characteristics(self) -> list[str]:
        """Validate property characteristic combinations for all relations.

        SEVERE (direct contradictions):
            - Reflexive + Irreflexive
            - Symmetric + Asymmetric
            - Asymmetric + Reflexive

        WARNING (OWL 2 DL inconsistencies, relation excluded):
            - Asymmetric + Functional
            - Asymmetric + InverseFunctional
            - Transitive + Functional
            - Transitive + InverseFunctional

        Returns:
            List of severe error messages (empty if none found).
        """
        num_relations = len(self.schema.id2rel)
        severe_errors: list[str] = []

        for rel_id in range(num_relations):
            rel_name = self.schema.id2rel[rel_id]

            # Skip if already marked unsatisfiable
            if rel_id in self.constraints.unsatisfiable_relation_ids:
                continue

            # ===== SEVERE: Direct Logical Contradictions =====

            # Check 1: Reflexive + Irreflexive
            if rel_id in self.constraints.reflexive_ids and rel_id in self.constraints.irreflex_ids:
                msg = (
                    f"Relation '{rel_name}' has forbidden combination: Reflexive + Irreflexive.\n"
                    f"    Direct mathematical contradiction: reflexive requires R(x,x) for all x, "
                    f"irreflexive forbids all R(x,x)."
                )
                logger.error(msg)
                severe_errors.append(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

            # Check 2: Symmetric + Asymmetric
            if rel_id in self.constraints.symmetric_ids and rel_id in self.constraints.asym_ids:
                msg = (
                    f"Relation '{rel_name}' has forbidden combination: Symmetric + Asymmetric.\n"
                    f"    Direct mathematical contradiction: symmetric requires R(x,y) <-> R(y,x), "
                    f"asymmetric forbids R(y,x) when R(x,y) exists."
                )
                logger.error(msg)
                severe_errors.append(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

            # Check 3: Asymmetric + Reflexive
            if rel_id in self.constraints.asym_ids and rel_id in self.constraints.reflexive_ids:
                msg = (
                    f"Relation '{rel_name}' has forbidden combination: Asymmetric + Reflexive.\n"
                    f"    Asymmetric implies irreflexive (no self-loops), which directly contradicts reflexivity."
                )
                logger.error(msg)
                severe_errors.append(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

            # ===== WARNING: OWL 2 DL Reasoning Inconsistencies =====

            # Check 4: Asymmetric + Functional
            if rel_id in self.constraints.asym_ids and rel_id in self.constraints.functional_ids:
                msg = (
                    f"Relation '{rel_name}' has problematic combination: Asymmetric + Functional.\n"
                    f"    OWL 2 DL inconsistency: combination creates contradictions in property chain reasoning.\n"
                    f"    Property excluded from generation."
                )
                logger.warning(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

            # Check 5: Asymmetric + InverseFunctional
            if rel_id in self.constraints.asym_ids and rel_id in self.constraints.invfunctional_ids:
                msg = (
                    f"Relation '{rel_name}' has problematic combination: Asymmetric + InverseFunctional.\n"
                    f"    OWL 2 DL inconsistency: combination creates contradictions in property chain reasoning.\n"
                    f"    Property excluded from generation."
                )
                logger.warning(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

            # Check 6: Transitive + Functional
            if (
                rel_id in self.constraints.transitive_ids
                and rel_id in self.constraints.functional_ids
            ):
                msg = (
                    f"Relation '{rel_name}' has problematic combination: Transitive + Functional.\n"
                    f"    OWL 2 DL inconsistency: transitivity propagates chains (a->b->c implies a->c), "
                    f"but functionality requires unique targets.\n"
                    f"    Property excluded from generation."
                )
                logger.warning(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

            # Check 7: Transitive + InverseFunctional
            if (
                rel_id in self.constraints.transitive_ids
                and rel_id in self.constraints.invfunctional_ids
            ):
                msg = (
                    f"Relation '{rel_name}' has problematic combination: Transitive + InverseFunctional.\n"
                    f"    OWL 2 DL inconsistency: transitivity propagates chains (a->b->c implies a->c), "
                    f"but inverse-functionality requires unique sources.\n"
                    f"    Property excluded from generation."
                )
                logger.warning(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

        return severe_errors

    # --- Category 4: Domain/Range Constraints Validation ---
    def _validate_domain_range_constraints(self) -> None:
        """Validate domain/range constraints for all relations.

        WARNING checks (relation excluded):
            - Domain contains disjoint class pairs (conjunctive impossible).
            - Range contains disjoint class pairs (conjunctive impossible).
            - Symmetric property with different domain and range (OWL violation).
        """
        num_relations = len(self.schema.id2rel)

        for rel_id in range(num_relations):
            # Skip if already marked unsatisfiable
            if rel_id in self.constraints.unsatisfiable_relation_ids:
                continue

            rel_name = self.schema.id2rel[rel_id]
            dom_classes = self.constraints.rel2dom_ids[rel_id]
            rng_classes = self.constraints.rel2range_ids[rel_id]

            # Check 1: Domain contains disjoint class pairs
            if self._has_disjoint_pair_in_set(dom_classes):
                dom_names = [self.schema.id2class[c] for c in dom_classes]
                msg = (
                    f"Relation '{rel_name}' has disjoint classes in domain {dom_names}.\n"
                    f"    Conjunctive domain requires entities to satisfy ALL classes, "
                    f"but disjoint classes cannot share instances.\n"
                    f"    Property excluded from generation."
                )
                logger.warning(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

            # Check 2: Range contains disjoint class pairs
            if self._has_disjoint_pair_in_set(rng_classes):
                rng_names = [self.schema.id2class[c] for c in rng_classes]
                msg = (
                    f"Relation '{rel_name}' has disjoint classes in range {rng_names}.\n"
                    f"    Conjunctive range requires entities to satisfy ALL classes, "
                    f"but disjoint classes cannot share instances.\n"
                    f"    Property excluded from generation."
                )
                logger.warning(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

            # Check 3: Symmetric properties must have domain = range
            if (
                rel_id in self.constraints.symmetric_ids
                and dom_classes
                and rng_classes
                and dom_classes != rng_classes
            ):
                dom_names = [self.schema.id2class[c] for c in dom_classes]
                rng_names = [self.schema.id2class[c] for c in rng_classes]
                msg = (
                    f"Symmetric relation '{rel_name}' has different domain {dom_names} and range {rng_names}.\n"
                    f"    OWL requires symmetric properties to have identical domain and range.\n"
                    f"    Property excluded from generation."
                )
                logger.warning(msg)
                self.constraints.unsatisfiable_relation_ids.add(rel_id)
                continue

    # --- Category 5: Subproperty Inheritance Validation ---
    def _validate_subproperty_characteristics(self) -> list[str]:
        """Validate that subproperty characteristics don't contradict superproperties.

        Subproperties inherit all constraints from superproperties and cannot be
        more permissive. More restrictive is allowed (asymmetric sub of symmetric
        super is valid).

        SEVERE checks:
            - Symmetric sub of Asymmetric super.
            - Reflexive sub of Irreflexive super.

        Returns:
            List of severe error messages (empty if none found).
        """
        severe_errors: list[str] = []

        for sub_id, super_ids in self.constraints.rel2superrel_ids.items():
            sub_name = self.schema.id2rel[sub_id]

            # Skip if already marked unsatisfiable
            if sub_id in self.constraints.unsatisfiable_relation_ids:
                continue

            # Check each superproperty in the hierarchy
            for super_id in super_ids:
                super_name = self.schema.id2rel[super_id]

                # Check 1: Symmetric sub of Asymmetric super
                if (
                    sub_id in self.constraints.symmetric_ids
                    and super_id in self.constraints.asym_ids
                ):
                    msg = (
                        f"Relation '{sub_name}' is symmetric but is a subproperty of asymmetric '{super_name}'.\n"
                        f"    Contradiction: {sub_name}(x,y) creates {sub_name}(y,x) (symmetric), "
                        f"which implies {super_name}(x,y) and {super_name}(y,x) (subproperty), "
                        f"violating {super_name}'s asymmetry.\n"
                        f"    Subproperties cannot be MORE PERMISSIVE than their superproperties."
                    )
                    logger.error(msg)
                    severe_errors.append(msg)
                    self.constraints.unsatisfiable_relation_ids.add(sub_id)
                    break

                # Check 2: Reflexive sub of Irreflexive super
                if (
                    sub_id in self.constraints.reflexive_ids
                    and super_id in self.constraints.irreflex_ids
                ):
                    msg = (
                        f"Relation '{sub_name}' is reflexive but is a subproperty of irreflexive '{super_name}'.\n"
                        f"    Contradiction: {sub_name}(x,x) must exist for all x (reflexive), "
                        f"which implies {super_name}(x,x) (subproperty), "
                        f"violating {super_name}'s irreflexivity.\n"
                        f"    Subproperties cannot be MORE PERMISSIVE than their superproperties."
                    )
                    logger.error(msg)
                    severe_errors.append(msg)
                    self.constraints.unsatisfiable_relation_ids.add(sub_id)
                    break

        return severe_errors

    # --- Utility: Disjointness Check ---
    def _has_disjoint_pair_in_set(self, class_ids: ClassIdFrozenSet) -> bool:
        """Check if a set of class IDs contains any mutually disjoint pair.

        Used for conjunctive constraints (e.g., domain = {A, B}). If A and B are
        disjoint, no entity can satisfy both.

        Args:
            class_ids: Frozenset of class IDs to check.

        Returns:
            True if any pair in the set are disjoint.
        """
        if len(class_ids) < 2:
            return False  # Single class or empty is always satisfiable

        class_list = list(class_ids)
        for i, class_id in enumerate(class_list):
            # Check if this class is disjoint with any remaining class in the set
            remaining = set(class_list[i + 1 :])
            if self.constraints.class2disjoints_extended_ids[class_id] & remaining:
                return True

        return False

    # ================================================================================================ #
    # DISJOINT ENVELOPES                                                                               #
    # ================================================================================================ #

    def _compute_disjoint_envelopes(self) -> None:
        """Compute disjoint envelopes for fast domain/range rejection.

        For each relation, precomputes the set of classes disjoint with all
        domain classes (dom_disjoint_envelope) and all range classes
        (range_disjoint_envelope). Enables O(1) rejection during deep validation.

        Complexity:
            O(n_r * n_c) worst case, but typically sparse due to limited
            disjointness declarations.
        """
        num_relations = len(self.schema.id2rel)

        self.constraints.dom_disjoint_envelope = []
        self.constraints.range_disjoint_envelope = []

        for rel_id in range(num_relations):
            # Skip expensive computation for unsatisfiable relations
            if rel_id in self.constraints.unsatisfiable_relation_ids:
                self.constraints.dom_disjoint_envelope.append(frozenset())
                self.constraints.range_disjoint_envelope.append(frozenset())
                continue

            dom_classes = self.constraints.rel2dom_ids[rel_id]
            rng_classes = self.constraints.rel2range_ids[rel_id]

            # Union all classes disjoint with any domain class
            dom_env: ClassIdSet = set()
            for cid in dom_classes:
                dom_env.update(self.constraints.class2disjoints_extended_ids[cid])

            # Union all classes disjoint with any range class
            rng_env: ClassIdSet = set()
            for cid in rng_classes:
                rng_env.update(self.constraints.class2disjoints_extended_ids[cid])

            self.constraints.dom_disjoint_envelope.append(frozenset(dom_env))
            self.constraints.range_disjoint_envelope.append(frozenset(rng_env))

        num_computed = num_relations - len(self.constraints.unsatisfiable_relation_ids)
        logger.debug(
            "Computed disjoint envelopes for %d relations (%d skipped as unsatisfiable)",
            num_computed,
            len(self.constraints.unsatisfiable_relation_ids),
        )
