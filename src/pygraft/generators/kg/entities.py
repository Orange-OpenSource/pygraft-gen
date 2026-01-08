#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#


"""Entity creation and class assignment for KG generation."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import TYPE_CHECKING

import numpy as np

from pygraft.generators.kg.structures import EntityTypingState

if TYPE_CHECKING:
    from numpy.random import Generator as NumpyRNG

    from pygraft.generators.kg.config import InstanceGeneratorConfig
    from pygraft.generators.kg.structures import ConstraintCaches, SchemaMetadata
    from pygraft.generators.kg.types import ClassId, ClassIdSet
    from pygraft.types import ClassInfoDict

logger = logging.getLogger(__name__)


class EntityGenerator:
    """Generates entities and assigns class types respecting ontology constraints.

    Handles the entity typing phase of KG generation:
        1. Initialize entity ID space and storage arrays
        2. Assign most-specific classes using power distribution
        3. Add additional classes for multityping
        4. Compute transitive superclass closures
        5. Resolve disjoint class conflicts
        6. Replicate profiles for fast generation mode

    Attributes:
        _config: Generator configuration.
        _constraints: Pre-computed constraint caches.
        _class_info: Loaded class_info.json contents.
        _schema: String-to-integer ID mappings.
        _rng: NumPy random generator.
        _fast_ratio: Scaling factor for fast generation mode.
        _entities: Entity typing state being built.
    """

    def __init__(
        self,
        *,
        config: InstanceGeneratorConfig,
        constraints: ConstraintCaches,
        class_info: ClassInfoDict,
        schema: SchemaMetadata,
        rng: NumpyRNG,
        fast_ratio: int,
    ) -> None:
        """Initialize the entity typer.

        Args:
            config: Generator configuration.
            constraints: Pre-computed constraint caches.
            class_info: Loaded class_info.json contents.
            schema: String-to-integer ID mappings.
            rng: NumPy random generator (mutated during generation).
            fast_ratio: Scaling factor for fast generation mode.
        """
        self._config = config
        self._constraints = constraints
        self._class_info = class_info
        self._schema = schema
        self._rng = rng
        self._fast_ratio = fast_ratio
        self._entities = EntityTypingState()

    def generate(self) -> EntityTypingState:
        """Run the full entity typing pipeline.

        Returns:
            Populated EntityTypingState with all entities and their class assignments.
        """
        # Step 1: Initialize entity space
        self._initialize_entity_space()

        # Step 2: Assign type profiles
        self._assign_most_specific_classes()
        if self._is_multityping_enabled():
            self._add_multitype_classes()
        self._compute_transitive_types()
        self._resolve_disjoint_type_conflicts()

        # Step 3: Replicate profiles if fast generation enabled
        if self._config.enable_fast_generation and self._fast_ratio > 1:
            self._replicate_entity_profiles()

        return self._entities

    # ------------------------------------------------------------------------------------------------ #
    # Entity Generation - Initialization                                                               #
    # ------------------------------------------------------------------------------------------------ #

    def _initialize_entity_space(self) -> None:
        """Create entity ID space and allocate storage arrays.

        Determines seed batch size (smaller for fast generation), randomly selects
        typed entity subset based on prop_untyped_entities, and pre-allocates
        per-entity storage arrays sized to final target space.

        Complexity:
            O(n_e) where n_e is entity count.
        """
        # Determine seed batch size (smaller for fast generation, full otherwise)
        base_count = (
            int(self._config.num_entities / self._fast_ratio)
            if self._config.enable_fast_generation
            else self._config.num_entities
        )

        # Create entity ID list for seed batch
        self._entities.entities = list(range(base_count))

        # Randomly select typed entity subset
        shuffled = self._entities.entities.copy()
        self._rng.shuffle(shuffled)

        threshold = int(len(self._entities.entities) * (1 - self._config.prop_untyped_entities))
        self._entities.typed_entities = set(shuffled[:threshold])

        # Allocate dense per-entity arrays sized to final target space
        num_entities = self._config.num_entities
        self._entities.ent2layer_specific = [-1] * num_entities
        self._entities.ent2classes_specific = [[] for _ in range(num_entities)]
        self._entities.ent2classes_transitive = [[] for _ in range(num_entities)]
        self._entities.ent2classes_transitive_sets = [set() for _ in range(num_entities)]

    # ------------------------------------------------------------------------------------------------ #
    # Entity Generation - Type Assignment                                                              #
    # ------------------------------------------------------------------------------------------------ #

    def _assign_most_specific_classes(self) -> None:
        """Assign most-specific classes to typed entities using power distribution.

        Uses a power distribution scaled to achieve the target avg_specific_class_depth.
        Each typed entity receives exactly one most-specific class from its assigned
        hierarchy layer.

        Complexity:
            O(n_e) where n_e is entity count.
        """
        hierarchy_depth = self._class_info["statistics"]["hierarchy_depth"]

        if not self._entities.typed_entities:
            self._entities.current_avg_depth_specific_class = 0.0
            return

        if hierarchy_depth <= 1:
            generated_numbers = np.ones(len(self._entities.typed_entities), dtype=int)
        else:
            shape = hierarchy_depth / (hierarchy_depth - 1)
            numbers = self._rng.power(shape, size=len(self._entities.typed_entities))
            scaled_numbers = (
                numbers / float(np.mean(numbers)) * float(self._config.avg_specific_class_depth)
            )
            generated_numbers = np.clip(np.floor(scaled_numbers), 1, hierarchy_depth).astype(int)

        self._entities.current_avg_depth_specific_class = float(np.mean(generated_numbers))

        for entity_id, layer in zip(
            sorted(self._entities.typed_entities), generated_numbers, strict=True
        ):
            layer_id = int(layer)
            self._entities.ent2layer_specific[entity_id] = layer_id

            layer_classes = self._constraints.layer2classes_ids.get(layer_id, [])
            if not layer_classes:
                continue

            chosen_class_id = int(self._rng.choice(layer_classes))
            self._entities.ent2classes_specific[entity_id] = [chosen_class_id]

    def _add_multitype_classes(self) -> None:
        """Add additional most-specific classes for multityping.

        Iteratively adds compatible classes from across the hierarchy until
        avg_types_per_entity is reached or no compatible candidates remain.
        A class is eligible if it is not disjoint with any existing class.

        Complexity:
            O(n_typed * avg_types * n_c) worst case for compatibility checks.
        """
        entity_list = sorted(self._entities.typed_entities)  # sorted for rng seed determinism
        attempt_count = 0

        if not entity_list or self._config.avg_types_per_entity <= 1.0:
            return

        def _avg() -> float:
            total = sum(len(self._entities.ent2classes_specific[e]) for e in entity_list)
            return float(total / max(1, len(entity_list)))

        current_avg = _avg()

        while current_avg < self._config.avg_types_per_entity and attempt_count < 10:
            ent = int(self._rng.choice(entity_list))
            most_specific = self._entities.ent2classes_specific[ent]
            layer = self._entities.ent2layer_specific[ent]
            if layer < 0:
                attempt_count += 1
                continue

            compatible = self._compute_compatible_classes(most_specific)
            candidates = sorted(compatible - set(most_specific))  # sorted for rng seed determinism

            if not candidates:
                attempt_count += 1
                continue

            most_specific.append(int(self._rng.choice(candidates)))
            current_avg = _avg()
            attempt_count = 0

    def _compute_transitive_types(self) -> None:
        """Extend entities with their transitive superclasses.

        For each entity, computes the transitive closure of its specific classes
        by adding all superclasses from the ontology hierarchy.

        Complexity:
            O(n_typed * avg_superclass_depth).
        """
        for entity_id in sorted(self._entities.typed_entities):
            specific_ids = self._entities.ent2classes_specific[entity_id]
            if not specific_ids:
                continue

            transitive: ClassIdSet = set(specific_ids)
            for cid in specific_ids:
                transitive.update(self._constraints.transitive_supers_ids[cid])

            trans_list = sorted(transitive)  # sorted for rng seed determinism
            self._entities.ent2classes_transitive[entity_id] = trans_list
            self._entities.ent2classes_transitive_sets[entity_id] = set(trans_list)

    def _resolve_disjoint_type_conflicts(self) -> None:
        """Resolve entities with disjoint class assignments.

        When multityping produces entities with mutually disjoint classes, repairs
        conflicts by surgically removing only the conflicting additional classes
        while preserving the first assigned class (from power distribution).

        This maximizes multityping by keeping as many compatible additional classes
        as possible. Final avg_types_per_entity may fall slightly below target.

        Complexity:
            O(n_typed * avg_types^2) for conflict detection and resolution.
        """
        self._entities.badly_typed = {}

        for entity_id in sorted(self._entities.typed_entities):
            trans = self._entities.ent2classes_transitive_sets[entity_id]
            if not trans:
                continue

            # Quick check: any conflicts exist?
            has_conflict = False
            for cid in trans:
                disj = self._constraints.class2disjoints_extended_ids[cid]
                if disj and (disj & trans):
                    has_conflict = True
                    break

            if not has_conflict:
                continue  # Entity is fine, skip

            # Conflict detected - perform surgical resolution
            specific_classes = self._entities.ent2classes_specific[entity_id]

            if len(specific_classes) == 1:
                # Only one class but still has conflict - this shouldn't happen
                # but log it for debugging
                continue

            # Start with first class (sacred) and its transitive closure
            first_class = specific_classes[0]
            baseline_trans = set(self._constraints.transitive_supers_ids[first_class])
            baseline_trans.add(first_class)

            # Check each additional class for conflicts with baseline
            kept_classes: list[ClassId] = [first_class]
            removed_classes: list[ClassId] = []

            for additional_class in specific_classes[1:]:
                # Get transitive closure of this additional class
                additional_trans = set(self._constraints.transitive_supers_ids[additional_class])
                additional_trans.add(additional_class)

                # Check if this additional class conflicts with current baseline
                conflict_found = False
                for cid in additional_trans:
                    disj = self._constraints.class2disjoints_extended_ids[cid]
                    if disj and (disj & baseline_trans):
                        conflict_found = True
                        break

                if conflict_found:
                    # This additional class conflicts - remove it
                    removed_classes.append(additional_class)
                else:
                    # No conflict - keep it and expand baseline
                    kept_classes.append(additional_class)
                    baseline_trans.update(additional_trans)

            # Record the conflict for diagnostics
            if removed_classes:
                self._entities.badly_typed[entity_id] = {
                    "original_classes": list(specific_classes),
                    "kept_classes": kept_classes,
                    "removed_classes": removed_classes,
                }

            # Update entity with kept classes and rebuilt transitive closure
            self._entities.ent2classes_specific[entity_id] = kept_classes

            final_trans_list = sorted(baseline_trans)  # sorted for rng seed determinism
            self._entities.ent2classes_transitive[entity_id] = final_trans_list
            self._entities.ent2classes_transitive_sets[entity_id] = set(final_trans_list)

    # ------------------------------------------------------------------------------------------------ #
    # Entity Generation - Profile Replication                                                          #
    # ------------------------------------------------------------------------------------------------ #

    def _replicate_entity_profiles(self) -> None:
        """Replicate typed entity profiles for fast generation mode.

        Creates additional entity batches by copying type profiles from the seed
        batch in round-robin fashion, avoiding recomputation of type assignments.

        Complexity:
            O(n_e) where n_e is final entity count.
        """
        # Capture seed batch type profiles
        typed_seed = sorted(self._entities.typed_entities)
        specific_seed = [self._entities.ent2classes_specific[e][:] for e in typed_seed]
        trans_seed = [self._entities.ent2classes_transitive[e][:] for e in typed_seed]
        layer_seed = [self._entities.ent2layer_specific[e] for e in typed_seed]

        # Replicate seed profiles across additional batches
        last_ent = len(self._entities.entities)
        for _ in range(1, self._fast_ratio):
            batch_size = int(self._config.num_entities / self._fast_ratio)
            entity_batch = list(range(last_ent, last_ent + batch_size))
            self._rng.shuffle(entity_batch)

            # Select typed subset for this batch
            threshold_batch = int(len(entity_batch) * (1 - self._config.prop_untyped_entities))
            typed_batch = entity_batch[:threshold_batch]
            self._entities.typed_entities.update(typed_batch)

            # Copy type profiles from seed batch (round-robin)
            for idx, e in enumerate(typed_batch):
                src = idx % max(1, len(typed_seed))
                self._entities.ent2classes_specific[e] = specific_seed[src][:]
                self._entities.ent2classes_transitive[e] = trans_seed[src][:]
                self._entities.ent2classes_transitive_sets[e] = set(trans_seed[src])
                self._entities.ent2layer_specific[e] = layer_seed[src]

            # Extend entity list with new batch
            self._entities.entities += entity_batch
            last_ent = len(self._entities.entities)

    # ------------------------------------------------------------------------------------------------ #
    # Entity Generation - Utilities                                                                    #
    # ------------------------------------------------------------------------------------------------ #

    def _is_multityping_enabled(self) -> bool:
        """Return True if multityping is effectively enabled."""
        return self._config.multityping and self._config.avg_types_per_entity > 0.0

    def _compute_compatible_classes(self, class_list: Iterable[ClassId]) -> set[ClassId]:
        """Return class IDs compatible with a given list of classes.

        A class is compatible if it is not disjoint with any class in the input.

        Args:
            class_list: Class IDs to check compatibility against.

        Returns:
            Set of class IDs that can be safely added without violating disjointness.

        Complexity:
            O(n_c * len(class_list)) where n_c is total class count.
        """
        base = list(class_list)
        base_set = set(base)
        if not base:
            return set(range(len(self._schema.id2class)))

        compatible: set[ClassId] = set()
        for candidate_id in range(len(self._schema.id2class)):
            if candidate_id in base_set:
                continue
            if any(
                candidate_id in self._constraints.class2disjoints_extended_ids[sid] for sid in base
            ):
                continue
            compatible.add(candidate_id)

        # Also allow classes that never appear as keys in disjoint dict (they map to empty sets here anyway)
        return compatible
