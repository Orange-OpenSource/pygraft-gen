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
Class Generator Module
======================

This module implements PyGraft's internal generator for synthetic OWL-style class hierarchies.
It builds a labeled class tree under `owl:Thing` and assigns disjointness constraints between
classes, producing a structured `ClassInfoDict` summary used throughout the schema pipeline.

High-Level Purpose
------------------
The module's role is to create coherent, statistically shaped class hierarchies for testing,
benchmarking, and synthetic ontology generation. It is considered internal to PyGraft and is
not part of the public, stable API surface.

The module defines two main abstractions:

- ClassGeneratorConfig:
    An immutable configuration object that validates all input parameters, including the
    number of classes, maximum hierarchy depth, target average depth, inheritance ratio,
    and average disjointness.

- ClassGenerator:
    A stochastic generator responsible for:
      - creating named classes (C1, C2, ..., Cn) under `owl:Thing`
      - shaping the hierarchy to match the requested depth and inheritance ratio
      - inserting disjointness constraints while respecting ancestry relationships
      - computing aggregate statistics over the generated hierarchy
      - assembling a `ClassInfoDict` mapping as the main output

Randomness and Determinism
--------------------------
All randomness is routed through an internal `random.Random` instance owned by the
ClassGenerator. Behavior is:

- deterministic and reproducible when a rng_seed is provided in ClassGeneratorConfig
- stochastic and non-reproducible when `rng_seed` is `None`

This module never touches the global random state; runs are isolated by construction.

Module Invariants
-----------------
The generator maintains several important invariants:

- The class hierarchy is a rooted tree under `owl:Thing` with no cycles.
- Depth and inheritance statistics are kept close to the configured targets, subject to
  discrete constraints from the finite number of classes.
- Disjointness axioms are never introduced between ancestor and descendant classes.
- Extended disjointness (propagated to subclasses) remains consistent with the direct
  disjointness relations recorded for each class.
- All mutable state is recreated from scratch for each generation run.

Performance Summary
-------------------
Let `n` be the number of classes:

- Hierarchy generation:
    Classes are attached one by one while recomputing global metrics such as average
    depth and inheritance ratio. This yields an effective worst-case time complexity
    of roughly O(n²), dominated by repeated metric calculations on the growing tree.

- Disjointness generation:
    Disjointness pairs are sampled subject to compatibility constraints and then
    propagated to descendants. With the configured safety caps, the algorithm performs
    up to O(n²) pair selections, and each propagation step can touch O(n) classes, for
    an effective worst-case complexity of O(n³) on large or adversarial hierarchies.

- Memory usage:
    Transitive superclass / subclass mappings and disjointness structures store
    relationships between many class pairs. In dense cases, these mappings require
    O(n²) space.

Intended Use
------------
This module is intended to be called by higher-level orchestration components that
coordinate dataset creation and benchmarking. External callers should treat
`generate_class_schema()` as the single entry point for producing a class schema.

The module performs no CLI parsing, no network or filesystem I/O (beyond logging), and
does not expose internal mutable structures directly to callers.
"""



# ------------------------------------------------------------------------------------------------ #
# Imports                                                                                          #
# ------------------------------------------------------------------------------------------------ #

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import random
from typing import TYPE_CHECKING

from pygraft.utils.schema import (
    calculate_average_depth,
    calculate_class_disjointness,
    calculate_inheritance_ratio,
    extend_class_mappings,
    generate_class2layer,
    get_max_depth,
)
from pygraft.types import build_class_info, ClassStatisticsDict

if TYPE_CHECKING:
    from pygraft.types import ClassInfoDict

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------ #
# Configuration Dataclasses                                                                        #
# ------------------------------------------------------------------------------------------------ #


@dataclass(frozen=True)
class ClassGeneratorConfig:
    """Immutable configuration for the ClassGenerator.

    Attributes:
        num_classes: Total number of classes to generate.
        max_hierarchy_depth: Maximum allowed depth of the class hierarchy.
        avg_class_depth: Target average depth for non-root classes.
        avg_children_per_parent: Target inheritance ratio (internal nodes vs. leaves).
        avg_disjointness: Target average disjointness between classes.
        rng_seed: Optional seed for deterministic behavior of the internal RNG.
    """
    # General
    rng_seed: int | None
    # Classes
    num_classes: int
    max_hierarchy_depth: int
    avg_class_depth: float
    avg_children_per_parent: float
    avg_disjointness: float


    def __post_init__(self) -> None:
        """Assumes user config was validated in config.py; enforces generator-specific class hierarchy invariants."""
        if self.num_classes <= 0:
            message = "num_classes must be a positive integer."
            raise ValueError(message)

        if self.max_hierarchy_depth <= 0:
            message = "max_hierarchy_depth must be a positive integer."
            raise ValueError(message)

        if self.num_classes < self.max_hierarchy_depth:
            message = (
                "num_classes must be greater than or equal to max_hierarchy_depth. "
                "The generator builds an initial vertical spine with length "
                "max_hierarchy_depth, so you need at least that many classes."
            )
            raise ValueError(message)

        # This is a ratio (non-trivial children / non-leaf classes), not a probability, so it may be > 1.0.
        if self.avg_children_per_parent <= 0.0:
            message = (
                "avg_children_per_parent must be strictly positive. It represents the target ratio "
                "of non-trivial children to non-leaf classes, as returned by calculate_inheritance_ratio()."
            )
            raise ValueError(message)

        if not (0.0 <= self.avg_disjointness <= 1.0):
            message = "avg_disjointness must be between 0.0 and 1.0."
            raise ValueError(message)

        if not (0.0 < self.avg_class_depth <= float(self.max_hierarchy_depth)):
            message = "avg_class_depth must be > 0.0 and less than or equal to max_hierarchy_depth."
            raise ValueError(message)


class ClassGenerator:
    """Generate a synthetic OWL-style class hierarchy and disjointness schema.

    This class is responsible for:
        - Creating a hierarchy of named classes (C1, C2, ..., Cn) under owl:Thing.
        - Shaping the hierarchy to approximately match the requested depth and inheritance ratio.
        - Injecting disjointness constraints between classes and their descendants.
        - Exposing a single high-level API method: generate_class_schema().

    Notes:
        - This class is considered internal to PyGraft and is not part of the public, stable API surface.
        - Behavior is deterministic only when a rng_seed is explicitly provided in the configuration.
          If `rng_seed` is None (the default), the generator uses nondeterministic randomness and the
          results will vary across runs even with the same configuration.

    Performance:
        Let n = num_classes.

        - Hierarchy generation:
            The algorithm incrementally places each of the n classes while recomputing global
            metrics such as average depth and inheritance ratio after each placement.
            In practice this leads to roughly O(n^2) time in the worst case, dominated by the
            repeated metric calculations over the growing hierarchy.

        - Disjointness generation:
            Disjointness is generated by sampling compatible class pairs and propagating their
            incompatibilities to descendants. With the safety caps in place, the loop performs up
            to O(n^2) iterations, and each iteration can touch O(n) classes when propagating
            constraints and recomputing aggregate disjointness. This yields an effective worst-case
            time of O(n^3) for very large n and adversarial hierarchies, although typical runs are
            closer to O(n^2).

        - Memory usage:
            The transitive superclass/subclass mappings and disjointness structures store
            relationships between many class pairs. In the worst case, these mappings require
            O(n^2) space.
    """

    def __init__(self, *, config: ClassGeneratorConfig) -> None:
        """Initialize the class generator with a configuration object.

        Args:
            config: Immutable configuration parameters for this generator instance.

        Raises:
            ValueError: If configuration parameters are outside their allowed ranges.
        """
        # User-supplied configuration (immutable).
        self._config: ClassGeneratorConfig = config

        # Internal RNG (no global random state).
        self._rng: random.Random = random.Random(config.rng_seed)

        # Internal mutable state (not intended for external mutation).
        self._classes: list[str] | None = None
        self._class2superclass_direct: dict[str, str] = {}
        self._class2superclasses_transitive: dict[str, list[str]] = defaultdict(list)
        self._class2subclasses_direct: dict[str, list[str]] = defaultdict(list)
        self._class2subclasses_transitive: dict[str, list[str]] = defaultdict(list)
        self._layer2classes: dict[int, list[str]] = defaultdict(list)

        self._disjoint_with: dict[str, list[str]] = {}
        self._disjoint_with_extended: dict[str, list[str]] = {}
        self._mutual_disjointness: set[str] = set()

    # ================================================================================================ #
    # Public API                                                                                       #
    # ================================================================================================ #

    def __repr__(self) -> str:
        """Return a debug-friendly representation for this generator."""
        return (
            "ClassGenerator("
            f"num_classes={self._config.num_classes}, "
            f"max_hierarchy_depth={self._config.max_hierarchy_depth}, "
            f"avg_class_depth={self._config.avg_class_depth}, "
            f"avg_children_per_parent={self._config.avg_children_per_parent}, "
            f"avg_disjointness={self._config.avg_disjointness}, "
            f"rng_seed={self._config.rng_seed}, "
            ")"
        )

    def generate_class_schema(self) -> ClassInfoDict:
        """Generate the full class schema and return a structured summary.

        The method is the main public entry point. It will:
        - Generate the hierarchy of classes.
        - Add disjointness axioms according to configuration.
        - Optionally log a summary table.
        - Return a `ClassInfoDict` mapping with all relevant metadata.

        Returns:
            A `ClassInfoDict` structure describing the generated classes and their relations.
        """
        self._generate_class_hierarchy()
        self._generate_class_disjointness()

        class_info: ClassInfoDict = self._assemble_class_info()
        return class_info


    # ================================================================================================ #
    # Internal Helper Methods (Private)                                                                #
    # ================================================================================================ #

    # ------------------------------------------------------------------------------------------------ #
    # Schema Assembly Helpers                                                                          #
    # ------------------------------------------------------------------------------------------------ #

    def _assemble_class_info(self) -> ClassInfoDict:
        """Assemble and return a ClassInfoDict summary of the current schema state."""
        if self._classes is None:
            message = "Classes have not been generated yet. Call generate_class_schema() first."
            raise RuntimeError(message)

        # --- classes ---
        classes_copy: list[str] = list(self._classes)

        # --- direct mappings ---
        direct_class2subclasses: dict[str, list[str]] = {
            class_name: list(children)
            for class_name, children in self._class2subclasses_direct.items()
        }
        # Internal hierarchy is single-parent, so we wrap each parent in a list to match the multi-parent external schema.
        direct_class2superclasses: dict[str, list[str]] = {
            child: [parent]
            for child, parent in self._class2superclass_direct.items()
        }

        # --- transitive mappings ---
        transitive_class2subclasses: dict[str, list[str]] = {
            class_name: list(children)
            for class_name, children in self._class2subclasses_transitive.items()
        }
        transitive_class2superclasses: dict[str, list[str]] = {
            class_name: list(parents)
            for class_name, parents in self._class2superclasses_transitive.items()
        }

        # --- disjointness ---
        class2disjoints: dict[str, list[str]] = {
            class_name: list(disjoints)
            for class_name, disjoints in self._disjoint_with.items()
        }
        class2disjoints_extended: dict[str, list[str]] = {
            class_name: sorted(set(disjoints))
            for class_name, disjoints in self._disjoint_with_extended.items()
        }

        # --- layer mappings ---
        layer2classes: dict[int, list[str]] = {
            int(layer): list(class_list)
            for layer, class_list in self._layer2classes.items()
        }
        class2layer: dict[str, int] = generate_class2layer(self._layer2classes)

        # --- statistics ---
        statistics: ClassStatisticsDict = {
            "num_classes": len(classes_copy),
            "hierarchy_depth": get_max_depth(self._layer2classes) or 0,
            "avg_class_depth": round(calculate_average_depth(self._layer2classes), 2),
            "avg_children_per_parent": round(
                calculate_inheritance_ratio(
                    self._class2superclass_direct,
                    self._class2subclasses_direct,
                ),
                2,
            ),
            "avg_class_disjointness": round(
                calculate_class_disjointness(self._disjoint_with, len(classes_copy)),
                2,
            ),
        }

        # --- build final structure ---
        return build_class_info(
            statistics=statistics,

            classes=classes_copy,

            direct_class2subclasses=direct_class2subclasses,
            direct_class2superclasses=direct_class2superclasses,

            transitive_class2subclasses=transitive_class2subclasses,
            transitive_class2superclasses=transitive_class2superclasses,

            class2disjoints=class2disjoints,
            class2disjoints_symmetric=list(self._mutual_disjointness),
            class2disjoints_extended=class2disjoints_extended,

            layer2classes=layer2classes,
            class2layer=class2layer,
        )

    # ------------------------------------------------------------------------------------------------ #
    # Hierarchy Generation Helpers                                                                     #
    # ------------------------------------------------------------------------------------------------ #

    def _generate_classes(self) -> None:
        """Create the base set of class identifiers (C1, C2, ..., Cn)."""
        self._classes = [f"C{index}" for index in range(1, self._config.num_classes + 1)]

    def _generate_class_hierarchy(self) -> None:
        """Generate the class hierarchy under owl:Thing.

        The algorithm:
        - Builds an initial vertical spine from owl:Thing to `max_hierarchy_depth`.
        - Iteratively attaches remaining classes, adjusting placement to approach:
          * the desired average depth; and
          * the desired inheritance ratio.
        - Adds optional noise to avoid overly skinny trees and create more realistic shapes.
        """
        self._generate_classes()
        if self._classes is None:
            message = "Failed to initialize classes list."
            raise RuntimeError(message)

        unconnected_classes: list[str] = list(self._classes)

        # Seed hierarchy: a chain from owl:Thing down to max_hierarchy_depth.
        current_class: str = unconnected_classes.pop()
        self._link_child2parent(current_class, "owl:Thing", layer=1)

        for layer in range(1, self._config.max_hierarchy_depth):
            next_class: str = unconnected_classes.pop()
            self._link_child2parent(next_class, current_class, layer=layer + 1)
            current_class = next_class

        current_avg_depth = calculate_average_depth(self._layer2classes)
        current_inheritance_ratio = calculate_inheritance_ratio(
            self._class2superclass_direct,
            self._class2subclasses_direct,
        )

        stochastic_noise_until = int(len(unconnected_classes) * 0.5)

        while unconnected_classes:
            class_name = unconnected_classes.pop()

            if (
                self._rng.random() < 0.35
                and len(unconnected_classes) >= stochastic_noise_until
                and self._config.max_hierarchy_depth > 3
            ):
                self._noisy_placing(class_name, current_avg_depth, current_inheritance_ratio)
            else:
                self._smart_placing(class_name, current_avg_depth, current_inheritance_ratio)

            current_avg_depth = calculate_average_depth(self._layer2classes)
            current_inheritance_ratio = calculate_inheritance_ratio(
                self._class2superclass_direct,
                self._class2subclasses_direct,
            )

    def _smart_placing(
        self,
        class_name: str,
        current_avg_depth: float,
        current_inheritance_ratio: float,
    ) -> None:
        """Place a class based on current depth and inheritance ratio.

        Args:
            class_name: Class identifier to place.
            current_avg_depth: Current average depth of the hierarchy.
            current_inheritance_ratio: Current average children per parent.
        """
        depth_ok = current_avg_depth <= self._config.avg_class_depth
        inheritance_ok = current_inheritance_ratio <= self._config.avg_children_per_parent

        if depth_ok and inheritance_ok:
            self._create_deep_leaf_realistic(class_name)
        elif depth_ok and not inheritance_ok:
            self._create_deep_child_realistic(class_name)
        elif not depth_ok and inheritance_ok:
            self._create_shallow_leaf(class_name)
        else:
            # depth not ok and inheritance not ok
            self._create_shallow_leaf_root(class_name)

    def _noisy_placing(
        self,
        class_name: str,
        current_avg_depth: float,
        current_inheritance_ratio: float,
    ) -> None:
        """Add controlled noise to the class placement.

        The algorithm biases placement towards intermediate layers to prevent
        overly tall, skinny trees.

        Args:
            class_name: Class identifier to place.
            current_avg_depth: Current average depth of the hierarchy.
            current_inheritance_ratio: Current inheritance ratio.
        """
        focus_layers = [key - 1 for key, value in self._layer2classes.items() if value]
        focus_layers = [
            layer
            for layer in focus_layers
            if layer not in [0, 1, self._config.max_hierarchy_depth - 1]
        ]

        if focus_layers:
            layer = self._rng.choice(focus_layers)
            parent = self._rng.choice(self._layer2classes[layer])
            self._link_child2parent(class_name, parent, layer=layer + 1)
        else:
            self._smart_placing(class_name, current_avg_depth, current_inheritance_ratio)

    def _create_deep_leaf_realistic(self, class_name: str) -> None:
        """Attach a new leaf to a deep parent in the hierarchy.

        This increases the inheritance ratio and total leaf count.

        Args:
            class_name: Class identifier to place.
        """
        deep_layers = [
            key - 1
            for key, value in self._layer2classes.items()
            if value and key >= self._config.avg_class_depth
        ]

        if not deep_layers:
            message = (
                "Cannot create a deep leaf realistically: no deep layers "
                "available. The hierarchy invariants appear to be broken."
            )
            raise RuntimeError(message)

        layer = self._rng.choice(deep_layers)
        min_layer = 0
        max_steps = self._config.max_hierarchy_depth + 5
        steps = 0

        while layer >= min_layer and steps < max_steps:
            current_parents = [
                candidate
                for candidate in self._layer2classes[layer]
                if candidate in self._class2subclasses_direct
            ]

            if current_parents:
                parent = self._rng.choice(current_parents)
                self._link_child2parent(class_name, parent, layer=layer + 1)
                return

            layer -= 1
            steps += 1

        message = (
            "Failed to find a deep parent with children when creating a deep "
            "leaf. This should not happen if the hierarchy invariants hold."
        )
        raise RuntimeError(message)

    def _create_deep_child_realistic(self, class_name: str) -> None:
        """Attach a new child to a deep leaf, keeping leaf count stable.

        This tends to decrease the inheritance ratio because a former leaf
        becomes an internal node.

        Args:
            class_name: Class identifier to place.
        """
        deep_layers = [
            key - 1
            for key, value in self._layer2classes.items()
            if value and key >= self._config.avg_class_depth
        ]

        if not deep_layers:
            message = (
                "Cannot create a deep child realistically: no deep layers "
                "available. The hierarchy invariants appear to be broken."
            )
            raise RuntimeError(message)

        layer = self._rng.choice(deep_layers)
        min_layer = 0
        max_steps = self._config.max_hierarchy_depth + 5
        steps = 0

        while layer >= min_layer and steps < max_steps:
            current_leaves = [
                candidate
                for candidate in self._layer2classes[layer]
                if candidate not in self._class2subclasses_direct
            ]

            if current_leaves:
                parent = self._rng.choice(current_leaves)
                self._link_child2parent(class_name, parent, layer=layer + 1)
                return

            layer -= 1
            steps += 1

        message = (
            "Failed to find a deep leaf when creating a deep child. "
            "This should not happen if the hierarchy invariants hold."
        )
        raise RuntimeError(message)

    def _create_shallow_leaf(self, class_name: str) -> None:
        """Attach a new leaf near the top of the hierarchy.

        Args:
            class_name: Class identifier to place as a shallow leaf.
        """
        layer = 1
        parent = self._rng.choice(self._layer2classes[layer])
        self._link_child2parent(class_name, parent, layer=layer + 1)

    def _create_shallow_leaf_root(self, class_name: str) -> None:
        """Attach a new leaf directly under owl:Thing.

        Args:
            class_name: Class identifier to place just under the root.
        """
        self._link_child2parent(class_name, "owl:Thing", layer=1)

    # NOTE:
    # These deterministic placement methods existed in the original implementation but were never invoked
    # by the generator logic. They are preserved here for historical continuity and to keep the modernized
    # version behaviorally aligned with the old code. They remain intentionally unused for now.
    # If a future update introduces a use case, they can be integrated or safely removed.
    def _create_deep_leaf_deterministic(self, class_name: str) -> None:
        """Attach a new leaf to the deepest available parent.

        Args:
            class_name: Class identifier to place.
        """
        last_layer = max(
            (layer for layer, nodes in self._layer2classes.items() if nodes),
            default=None,
        )
        if last_layer is None:
            message = "Cannot determine last layer: layer2classes is empty."
            raise RuntimeError(message)

        min_layer = 0
        max_steps = self._config.max_hierarchy_depth + 5
        steps = 0
        layer = last_layer - 1

        while layer >= min_layer and steps < max_steps:
            current_parents = [
                candidate
                for candidate in self._layer2classes[layer]
                if candidate in self._class2subclasses_direct
            ]

            if current_parents:
                parent = self._rng.choice(current_parents)
                self._link_child2parent(class_name, parent, layer=layer + 1)
                return

            layer -= 1
            steps += 1

        message = (
            "Failed to find a deep parent with children when creating a deep leaf "
            "deterministically. This should not happen if the hierarchy invariants hold."
        )
        raise RuntimeError(message)

    def _create_deep_child_deterministic(self, class_name: str) -> None:
        """Attach a new child to the deepest available leaf.

        Args:
            class_name: Class identifier to place.
        """
        last_layer = max(
            (layer for layer, nodes in self._layer2classes.items() if nodes),
            default=None,
        )
        if last_layer is None:
            message = "Cannot determine last layer: layer2classes is empty."
            raise RuntimeError(message)

        min_layer = 0
        max_steps = self._config.max_hierarchy_depth + 5
        steps = 0
        layer = last_layer - 1

        while layer >= min_layer and steps < max_steps:
            current_leaves = [
                candidate
                for candidate in self._layer2classes[layer]
                if candidate not in self._class2subclasses_direct
            ]

            if current_leaves:
                parent = self._rng.choice(current_leaves)
                self._link_child2parent(class_name, parent, layer=layer + 1)
                return

            layer -= 1
            steps += 1

        message = (
            "Failed to find a leaf when creating a deep child deterministically. "
            "The hierarchy appears to have no leaves, which should not be possible."
        )
        raise RuntimeError(message)

    # ------------------------------------------------------------------------------------------------ #
    # Disjointness Generation Helpers                                                                  #
    # ------------------------------------------------------------------------------------------------ #

    def _generate_class_disjointness(self) -> None:
        """Generate disjointness axioms between classes.

        The algorithm:
        - Repeatedly selects random pairs of classes that are not in an ancestor/descendant relation.
        - Marks them as mutually disjoint.
        - Propagates disjointness to transitive subclasses.
        - Stops once the average disjointness threshold is reached or an iteration safety cap is hit.
        """
        if self._classes is None:
            message = "Classes have not been generated yet. Call generate_class_schema() first."
            raise RuntimeError(message)

        current_class_disjointness = 0.0

        (
            self._class2superclasses_transitive,
            self._class2subclasses_transitive,
        ) = extend_class_mappings(self._class2superclass_direct)

        self._disjoint_with = {}
        self._disjoint_with_extended = {}
        self._mutual_disjointness = set()

        # Safety cap: in the worst case there are ~N^2 possible pairs.
        # We allow up to max(N^2, 1000) iterations before giving up.
        max_iterations = max(1000, self._config.num_classes * self._config.num_classes)
        iteration_count = 0

        while (
            current_class_disjointness < self._config.avg_disjointness
            and iteration_count < max_iterations
        ):
            # Pick two compatible classes (no ancestor/descendant relation).
            class_a, class_b = self._pick_compatible_class_pair()

            # Register disjointness.
            self._disjoint_with.setdefault(class_a, []).append(class_b)
            self._disjoint_with.setdefault(class_b, []).append(class_a)
            self._disjoint_with_extended.setdefault(class_a, []).append(class_b)
            self._disjoint_with_extended.setdefault(class_b, []).append(class_a)

            mutual_key = (
                f"{class_a}-{class_b}"
                if int(class_a[1:]) < int(class_b[1:])
                else f"{class_b}-{class_a}"
            )
            self._mutual_disjointness.add(mutual_key)

            # Propagate to descendants.
            self._extend_incompatibilities(class_a, class_b)

            # Update aggregate disjointness statistic.
            current_class_disjointness = calculate_class_disjointness(
                self._disjoint_with,
                self._config.num_classes,
            )

            iteration_count += 1

        if current_class_disjointness < self._config.avg_disjointness:
            logger.warning(
                (
                    "Could not reach target avg disjointness %.3f; "
                    "achieved %.3f after %d iterations."
                ),
                self._config.avg_disjointness,
                current_class_disjointness,
                iteration_count,
            )

    def _pick_compatible_class_pair(self) -> tuple[str, str]:
        """Pick a pair of classes that are not in an ancestor/descendant relation.

        Raises:
            RuntimeError: If no such pair can be found within a reasonable number
                of attempts. This typically means the hierarchy degenerated into
                a near-total order (pure chain) where every pair of classes is
                ancestor/descendant.
        """
        if self._classes is None:
            message = "Classes have not been generated yet. Call generate_class_schema() first."
            raise RuntimeError(message)

        num_classes = len(self._classes)
        if num_classes < 2:
            message = "Need at least two classes to pick a compatible pair."
            raise RuntimeError(message)

        # Quadratic-ish cap: if the hierarchy is degenerate (pure chain),
        # we do not want to spin forever trying to find a non-ancestor pair.
        max_attempts = max(num_classes * num_classes * 2, 100)

        for _ in range(max_attempts):
            class_a = self._rng.choice(self._classes)
            class_b = self._rng.choice(self._classes)

            if class_a == class_b:
                continue

            is_parent = (
                class_a in self._class2superclasses_transitive
                and class_b in self._class2superclasses_transitive[class_a]
            )
            is_child = (
                class_a in self._class2subclasses_transitive
                and class_b in self._class2subclasses_transitive[class_a]
            )

            if not (is_parent or is_child):
                return class_a, class_b

        message = (
            "Failed to find a non-ancestor/descendant class pair after "
            f"{max_attempts} attempts. The generated hierarchy appears to "
            "be effectively a total order (pure chain). Consider increasing "
            "num_classes or decreasing max_hierarchy_depth for this "
            "configuration."
        )
        raise RuntimeError(message)

    def _extend_incompatibilities(
        self,
        class_a: str,
        class_b: str,
    ) -> None:
        """Propagate incompatibilities between two classes to their descendants.

        Args:
            class_a: First class in the disjoint pair.
            class_b: Second class in the disjoint pair.
        """
        children_a = self._class2subclasses_transitive.get(class_a, [])
        children_b = self._class2subclasses_transitive.get(class_b, [])

        def add_extended_pair(left: str, right: str) -> None:
            """Register an extended disjoint pair and its symmetric key."""
            self._disjoint_with_extended.setdefault(left, []).append(right)
            self._disjoint_with_extended.setdefault(right, []).append(left)
            mutual_key = f"{left}-{right}" if int(left[1:]) < int(right[1:]) else f"{right}-{left}"
            self._mutual_disjointness.add(mutual_key)

        # Both have children: propagate to all child pairs.
        for child_a in children_a:
            for child_b in children_b:
                add_extended_pair(child_a, child_b)

        # Only B has children.
        if not children_a:
            for child_b in children_b:
                add_extended_pair(class_a, child_b)

        # Only A has children.
        if not children_b:
            for child_a in children_a:
                add_extended_pair(child_a, class_b)

    # ------------------------------------------------------------------------------------------------ #
    # Low-Level Mutation Helpers                                                                       #
    # ------------------------------------------------------------------------------------------------ #

    def _link_child2parent(
        self,
        child: str,
        parent: str,
        layer: int,
    ) -> None:
        """Link a child to its direct parent and update layer bookkeeping.

        This keeps a proper multi-child mapping:
        - `parent -> [child1, child2, ...]`
        - `child -> direct parent`
        - tracks the child in the given `layer`.

        Args:
            child: Child class identifier.
            parent: Parent class identifier.
            layer: Depth layer (1-based, owl:Thing at layer 0).
        """
        # `_class2subclasses_direct` is a defaultdict(list), so this safely appends.
        self._class2subclasses_direct[parent].append(child)
        self._class2superclass_direct[child] = parent
        self._layer2classes[layer].append(child)
