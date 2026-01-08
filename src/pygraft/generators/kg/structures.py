#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Internal state structures for the KG generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pygraft.generators.kg.types import (
        ClassId,
        ClassIdFrozenSet,
        ClassIdList,
        ClassIdSet,
        ClassIdToEntityIds,
        EntityId,
        EntityIdList,
        EntityIdSet,
        RelationId,
        RelationIdToHeadTailPairs,
    )


@dataclass
class SchemaMetadata:
    """Bidirectional mappings between schema strings and integer IDs.

    Enables efficient integer-based operations internally while preserving
    string-based identifiers for RDF serialization.

    Attributes:
        class2id: Class name to integer ID.
        id2class: Integer ID to class name.
        rel2id: Relation name to integer ID.
        id2rel: Integer ID to relation name.
    """

    class2id: dict[str, ClassId] = field(default_factory=dict)
    id2class: list[str] = field(default_factory=list)
    rel2id: dict[str, RelationId] = field(default_factory=dict)
    id2rel: list[str] = field(default_factory=list)


@dataclass
class ConstraintCaches:
    """Pre-computed constraint data for efficient validation.

    Built once during initialization and used throughout triple generation.
    All lookups use integer IDs for performance.

    Attributes:
        layer2classes_ids: Hierarchy layer to class IDs in that layer.
        class2layer_id: Class ID to hierarchy layer.
        class2disjoints_extended_ids: Class ID to disjoint class IDs (includes inherited).
        transitive_supers_ids: Class ID to all superclass IDs (includes self).
        rel2dom_ids: Relation ID to required domain class IDs.
        rel2range_ids: Relation ID to required range class IDs.
        rel2disjoints_extended_ids: Relation ID to disjoint relation IDs.
        dom_disjoint_envelope: Relation ID to classes disjoint with domain.
        range_disjoint_envelope: Relation ID to classes disjoint with range.
        reflexive_ids: Reflexive relation IDs.
        irreflex_ids: Irreflexive relation IDs.
        asym_ids: Asymmetric relation IDs.
        symmetric_ids: Symmetric relation IDs.
        transitive_ids: Transitive relation IDs.
        functional_ids: Functional relation IDs.
        invfunctional_ids: Inverse-functional relation IDs.
        rel2inverse_ids: Relation ID to its inverse relation ID.
        rel2superrel_ids: Relation ID to direct super-relation IDs.
        unsatisfiable_relation_ids: Relations excluded due to schema errors.
    """

    # Class hierarchy and disjointness
    layer2classes_ids: dict[int, ClassIdList] = field(default_factory=dict)
    class2layer_id: list[int] = field(default_factory=list)
    class2disjoints_extended_ids: list[ClassIdFrozenSet] = field(default_factory=list)
    transitive_supers_ids: list[ClassIdFrozenSet] = field(default_factory=list)

    # Relation domain/range and disjointness
    rel2dom_ids: list[ClassIdFrozenSet] = field(default_factory=list)
    rel2range_ids: list[ClassIdFrozenSet] = field(default_factory=list)
    rel2disjoints_extended_ids: list[frozenset[RelationId]] = field(default_factory=list)
    dom_disjoint_envelope: list[ClassIdFrozenSet] = field(default_factory=list)
    range_disjoint_envelope: list[ClassIdFrozenSet] = field(default_factory=list)

    # Relation properties
    reflexive_ids: set[RelationId] = field(default_factory=set)
    irreflex_ids: set[RelationId] = field(default_factory=set)
    asym_ids: set[RelationId] = field(default_factory=set)
    symmetric_ids: set[RelationId] = field(default_factory=set)
    transitive_ids: set[RelationId] = field(default_factory=set)
    functional_ids: set[RelationId] = field(default_factory=set)
    invfunctional_ids: set[RelationId] = field(default_factory=set)

    # Relation relationships
    rel2inverse_ids: dict[RelationId, RelationId] = field(default_factory=dict)
    rel2superrel_ids: dict[RelationId, list[RelationId]] = field(default_factory=dict)

    # Relations marked unsatisfiable due to schema-level modeling errors
    unsatisfiable_relation_ids: set[RelationId] = field(default_factory=set)


@dataclass
class EntityTypingState:
    """Mutable state for entity creation and class assignment.

    Tracks entity existence, class memberships, and reverse indices for
    candidate pool construction.

    Attributes:
        entities: All entity IDs.
        typed_entities: Entity IDs with at least one class.
        ent2layer_specific: Entity ID to hierarchy layer of most-specific class.
        ent2classes_specific: Entity ID to most-specific class IDs.
        ent2classes_transitive: Entity ID to all class IDs (includes supers).
        ent2classes_transitive_sets: Set version for O(1) membership checks.
        current_avg_depth_specific_class: Running average depth of specific classes.
        badly_typed: Entities with disjoint class violations (debug info).
        class2entities: Class ID to entity IDs with that class.
        class2unseen: Class ID to entity IDs not yet used in triples.
        unseen_entities_pool: All entity IDs not yet in any triple.
        priority_untyped_entities: Untyped entities prioritized for sampling.
        untyped_entities: Entity IDs with no class assignment.
    """

    entities: EntityIdList = field(default_factory=list)
    typed_entities: EntityIdSet = field(default_factory=set)

    ent2layer_specific: list[int] = field(default_factory=list)
    ent2classes_specific: list[ClassIdList] = field(default_factory=list)
    ent2classes_transitive: list[ClassIdList] = field(default_factory=list)
    ent2classes_transitive_sets: list[ClassIdSet] = field(default_factory=list)

    current_avg_depth_specific_class: float = 0.0
    badly_typed: dict[EntityId, dict[str, Any]] = field(default_factory=dict)

    class2entities: ClassIdToEntityIds = field(default_factory=dict)
    class2unseen: ClassIdToEntityIds = field(default_factory=dict)
    unseen_entities_pool: EntityIdList = field(default_factory=list)

    priority_untyped_entities: EntityIdSet = field(default_factory=set)
    untyped_entities: EntityIdList = field(default_factory=list)


@dataclass
class TripleGenerationState:
    """Mutable state for triple generation progress and storage.

    Tracks sampling weights, budget allocations, and incremental constraint
    state for functional properties.

    Attributes:
        relation_sampling_weights: Probability weights for relation sampling.
        num_relations: Count of active relations.
        triples_per_rel: Target triple count per relation.
        kg_pairs_by_rid: Generated triples as relation ID to (head, tail) pairs.
        functional_heads: Heads already used per functional relation.
        invfunctional_tails: Tails already used per inverse-functional relation.
    """

    relation_sampling_weights: list[float] = field(default_factory=list)
    num_relations: int = 0
    triples_per_rel: dict[RelationId, int] = field(default_factory=dict)
    kg_pairs_by_rid: RelationIdToHeadTailPairs = field(default_factory=dict)
    functional_heads: dict[RelationId, set[EntityId]] = field(default_factory=dict)
    invfunctional_tails: dict[RelationId, set[EntityId]] = field(default_factory=dict)
