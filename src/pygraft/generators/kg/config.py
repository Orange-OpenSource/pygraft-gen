#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Configuration for the KG generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class InstanceGeneratorConfig:
    """Immutable configuration for knowledge graph generation.

    Specifies KG size parameters, entity typing behavior, and generation
    heuristics. Validated at construction time.

    Attributes:
        project_name: Schema folder name under output directory.
        rdf_format: Output serialization format ("xml", "ttl", "nt").
        rng_seed: Seed for deterministic generation. None for stochastic.
        num_entities: Target entity count.
        num_triples: Target triple count.
        enable_fast_generation: Generate smaller prototype then scale up.
        relation_usage_uniformity: Triple distribution evenness (0.0-1.0).
        prop_untyped_entities: Proportion of entities without class assignment.
        avg_specific_class_depth: Target average hierarchy depth of assigned classes.
        multityping: Allow multiple most-specific classes per entity.
        avg_types_per_entity: Target average class count per entity.
        check_kg_consistency: Run reasoner validation after generation.
        output_root: Override for output directory. Defaults to OUTPUT_ROOT.
    """

    # General (JSON order)
    project_name: str
    rdf_format: str
    rng_seed: int | None

    # KG (JSON order)
    num_entities: int
    num_triples: int

    enable_fast_generation: bool

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
