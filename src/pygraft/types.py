#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Type definitions for PyGraft data structures.

This module centralizes the static types for configuration files and
derived JSON artefacts (class_info, relation_info, kg_info, etc.).
"""

from __future__ import annotations

from typing import TypeAlias, TypedDict

# ================================================================================================ #
# Pygraft Configuration File : pygraft.config.{json/yml}                                           #
# ================================================================================================ #


class PyGraftConfigDict(TypedDict):
    """Top-level configuration driving schema and KG generation.

    This mirrors the grouped structure of `pygraft_config.{json,yml}`, with
    three main sections:

    - general: Global output, RDF format, and RNG settings.
    - schema:  Schema generation parameters (classes and relations).
    - kg:      Knowledge graph instance generation parameters.
    """

    general: GeneralConfigDict
    schema: SchemaConfigDict
    kg: KGGenConfigDict


# ------------------------------------------------------------------------------------------------ #
# General Section                                                                                  #
# ------------------------------------------------------------------------------------------------ #


class GeneralConfigDict(TypedDict):
    """Global configuration for output naming, RDF syntax, and RNG behavior.

    Attributes:
        project_name: Human-readable name for the generation run. Also used as
            the output folder name under "output/". When null, PyGraft will
            either create a timestamped folder (schema generation) or pick the
            latest existing one (KG generation).
        rdf_format: RDF serialization format used for schema and KG graphs
            (e.g. "xml", "ttl", "nt"). Passed directly to RDFLib.
        rng_seed: Optional global RNG seed used to make class, relation, and KG
            generation deterministic for testing and development. When null,
            generation remains fully stochastic.
    """

    project_name: str
    rdf_format: str
    rng_seed: int | None


# ------------------------------------------------------------------------------------------------ #
# Schema Section                                                                                   #
# ------------------------------------------------------------------------------------------------ #


class SchemaConfigDict(TypedDict):
    """Schema configuration wrapping class and relation generation parameters.

    Attributes:
        classes: Class hierarchy generation parameters.
        relations: Relation (object property) generation parameters.
    """

    classes: ClassGenConfigDict
    relations: RelationGenConfigDict


# --- Classes --- #


class ClassGenConfigDict(TypedDict):
    """Class generation configuration.

    Attributes:
        num_classes: Total number of classes (C1..Cn) to generate.
        max_hierarchy_depth: Maximum depth of the class hierarchy under owl:Thing.
        avg_class_depth: Target average depth for non-root classes; used to
            control how deep classes are placed in the hierarchy.
        avg_children_per_parent: Target average number of direct subclasses per
            non-leaf class; controls how branchy the hierarchy is.
        avg_disjointness: Target average disjointness between classes, expressed
            as a fraction of class pairs that should be disjoint.
    """

    num_classes: int
    max_hierarchy_depth: int
    avg_class_depth: float
    avg_children_per_parent: float
    avg_disjointness: float


# --- Relations --- #


class RelationGenConfigDict(TypedDict):
    """Relation generation configuration.

    Attributes:
        num_relations: Number of object properties (R1..Rn) to generate.
        relation_specificity: Target average depth of the domain and range
            classes assigned to relations. Higher values favor more specific
            (deeper) classes.
        prop_profiled_relations: Fraction of eligible (non-reflexive) relations
            that receive domain and/or range class constraints.
        profile_side: Controls how domain/range constraints are applied:
            "both" assigns both domain and range; "partial" assigns only one
            side (domain or range) per relation.
        prop_symmetric_relations: Fraction of symmetric relations.
        prop_inverse_relations: Fraction of relations paired as inverses.
        prop_transitive_relations: Fraction of transitive relations.
        prop_asymmetric_relations: Fraction of asymmetric relations.
        prop_reflexive_relations: Fraction of reflexive relations. Reflexive
            relations never receive domain/range constraints.
        prop_irreflexive_relations: Fraction of irreflexive relations.
        prop_functional_relations: Fraction of functional relations.
        prop_inverse_functional_relations: Fraction of inverse-functional
            relations.
        prop_subproperties: Fraction of relations that participate as
            subproperties in subPropertyOf hierarchies.
    """

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


# ------------------------------------------------------------------------------------------------ #
# Kg Section                                                                                       #
# ------------------------------------------------------------------------------------------------ #


class KGGenConfigDict(TypedDict):
    """KG instance generation configuration.

    Attributes:
        num_entities: Number of entities (E1..En) to generate.
        num_triples: Target number of triples before augmentation.
        enable_fast_generation: Whether to use the fast seed-and-replicate KG
            generation mode, trading some diversity for speed.
        relation_usage_uniformity: Controls how evenly triples are
            distributed across relations, with higher values producing more
            uniform usage.
        prop_untyped_entities: Proportion of intentionally untyped entities.
        avg_specific_class_depth: Target depth of the most specific class
            assigned to entities.
        multityping: Whether entities may have multiple most-specific classes.
        avg_types_per_entity: Average number of most-specific classes per typed
            entity, subject to the multityping rules.
        check_kg_consistency: Whether to run HermiT-based consistency checking
            after KG generation.
    """

    num_entities: int
    num_triples: int

    enable_fast_generation: bool

    relation_usage_uniformity: float
    prop_untyped_entities: float

    avg_specific_class_depth: float

    multityping: bool
    avg_types_per_entity: float

    check_kg_consistency: bool


# ================================================================================================ #
# Class Info JSON                                                                                  #
# ================================================================================================ #


class ClassInfoDict(TypedDict):
    """Top-level structure of class_info.json.

    Only the fields that are needed by the rest of the library are typed
    precisely. Remaining nested structures may be widened to Dict[str, Any]
    to keep the type surface manageable.
    """

    statistics: ClassStatisticsDict

    # --- class list ---
    classes: list[str]

    # --- direct mappings ----
    direct_class2subclasses: dict[str, list[str]]
    direct_class2superclasses: dict[str, list[str]]

    # --- transitive mappings ---
    transitive_class2subclasses: dict[str, list[str]]
    transitive_class2superclasses: dict[str, list[str]]

    # --- disjointness mappings ---
    class2disjoints: dict[str, list[str]]
    class2disjoints_symmetric: list[str]
    class2disjoints_extended: dict[str, list[str]]

    # --- layer mappings ---
    layer2classes: dict[int, list[str]]
    class2layer: dict[str, int]


class ClassStatisticsDict(TypedDict):
    """Summary statistics for the generated class hierarchy."""

    num_classes: int
    hierarchy_depth: int
    avg_class_depth: float
    avg_children_per_parent: float
    avg_class_disjointness: float


def build_class_info(
    *,
    # --- statistics  ---
    statistics: ClassStatisticsDict,
    # --- class list ---
    classes: list[str],
    # --- direct mappings ---
    direct_class2subclasses: dict[str, list[str]],
    direct_class2superclasses: dict[str, list[str]],
    # --- transitive mappings ---
    transitive_class2subclasses: dict[str, list[str]],
    transitive_class2superclasses: dict[str, list[str]],
    # --- disjointness mappings ---
    class2disjoints: dict[str, list[str]],
    class2disjoints_symmetric: list[str],
    class2disjoints_extended: dict[str, list[str]],
    # --- layer mappings ---
    layer2classes: dict[int, list[str]],
    class2layer: dict[str, int],
) -> ClassInfoDict:
    """Build a JSON-friendly ClassInfoDict instance.

    This function centralizes the structure of class_info.json so that
    generators provide only raw values without re-implementing the layout.
    """
    return {
        "statistics": statistics,
        "classes": list(classes),
        "direct_class2subclasses": dict(direct_class2subclasses),
        "direct_class2superclasses": dict(direct_class2superclasses),
        "transitive_class2subclasses": dict(transitive_class2subclasses),
        "transitive_class2superclasses": dict(transitive_class2superclasses),
        "class2disjoints": dict(class2disjoints),
        "class2disjoints_symmetric": list(class2disjoints_symmetric),
        "class2disjoints_extended": dict(class2disjoints_extended),
        "layer2classes": dict(layer2classes),
        "class2layer": dict(class2layer),
    }


# ================================================================================================ #
# Relation Info JSON                                                                               #
# ================================================================================================ #


class RelationInfoDict(TypedDict):
    """Top-level structure of relation_info.json.

    Note:
        The rel2patterns mapping is stored as lists in JSON, even if the
        in-memory representation uses sets. The builder function is
        responsible for converting sets to lists when needed.
    """

    statistics: RelationStatistics

    # --- relation list ---
    relations: list[str]

    # --- OWL pattern mappings ---
    rel2patterns: dict[str, list[str]]

    # --- OWL logical characteristics (per property) ---
    reflexive_relations: list[str]
    irreflexive_relations: list[str]
    symmetric_relations: list[str]
    asymmetric_relations: list[str]
    functional_relations: list[str]
    inversefunctional_relations: list[str]
    transitive_relations: list[str]

    # --- OWL inverse-of relationships ---
    inverseof_relations: list[str]
    rel2inverse: dict[str, str]

    # --- RDFS subPropertyOf hierarchy ---
    subrelations: list[str]
    rel2superrel: dict[str, list[str]]

    # --- disjointness mappings ---
    rel2disjoints: dict[str, list[str]]
    rel2disjoints_symmetric: list[str]
    rel2disjoints_extended: dict[str, list[str]]

    # --- RDFS/OWL domain and range ---
    rel2dom: dict[str, list[str]]
    rel2range: dict[str, list[str]]


class RelationStatistics(TypedDict):
    """Summary statistics for generated relations."""

    num_relations: int
    prop_reflexive: float
    prop_irreflexive: float
    prop_functional: float
    prop_inversefunctional: float
    prop_symmetric: float
    prop_asymmetric: float
    prop_transitive: float
    prop_inverseof: float
    prop_subpropertyof: float
    prop_profiled_relations: float
    relation_specificity: float


def build_relation_info(
    *,
    # --- statistics ---
    statistics: RelationStatistics,
    # --- relation list ---
    relations: list[str],
    # --- OWL pattern mappings ---
    rel2patterns: dict[str, set[str]] | dict[str, list[str]],
    # --- OWL logical characteristics (per property) ---
    reflexive_relations: list[str],
    irreflexive_relations: list[str],
    symmetric_relations: list[str],
    asymmetric_relations: list[str],
    functional_relations: list[str],
    inversefunctional_relations: list[str],
    transitive_relations: list[str],
    # --- OWL inverse-of relationships ---
    inverseof_relations: list[str],
    rel2inverse: dict[str, str],
    # --- RDFS subPropertyOf hierarchy ---
    subrelations: list[str],
    rel2superrel: dict[str, list[str]],
    # --- disjointness mappings ---
    rel2disjoints: dict[str, list[str]],
    rel2disjoints_symmetric: list[str],
    rel2disjoints_extended: dict[str, list[str]],
    # --- RDFS/OWL domain and range ---
    rel2dom: dict[str, list[str]],
    rel2range: dict[str, list[str]],
) -> RelationInfoDict:
    """Build a JSON-friendly RelationInfoDict instance.

    This function centralizes the structure of relation_info.json so that
    generators provide only raw values without re-implementing the layout.
    """
    rel2patterns_json: dict[str, list[str]] = {}
    for relation_id, patterns in rel2patterns.items():
        if isinstance(patterns, set):
            rel2patterns_json[relation_id] = sorted(patterns)
        else:
            rel2patterns_json[relation_id] = list(patterns)

    return {
        "statistics": statistics,
        "relations": list(relations),
        "rel2patterns": rel2patterns_json,
        "reflexive_relations": list(reflexive_relations),
        "irreflexive_relations": list(irreflexive_relations),
        "symmetric_relations": list(symmetric_relations),
        "asymmetric_relations": list(asymmetric_relations),
        "functional_relations": list(functional_relations),
        "inversefunctional_relations": list(inversefunctional_relations),
        "transitive_relations": list(transitive_relations),
        "inverseof_relations": list(inverseof_relations),
        "rel2inverse": dict(rel2inverse),
        "subrelations": list(subrelations),
        "rel2superrel": dict(rel2superrel),
        "rel2disjoints": dict(rel2disjoints),
        "rel2disjoints_symmetric": list(rel2disjoints_symmetric),
        "rel2disjoints_extended": dict(rel2disjoints_extended),
        "rel2dom": dict(rel2dom),
        "rel2range": dict(rel2range),
    }


# ================================================================================================ #
# KG Info JSON                                                                                     #
# ================================================================================================ #


class KGInfoDict(TypedDict):
    """Top-level structure of kg_info.json."""

    user_parameters: KGUserParameters
    statistics: KGStatistics


class KGUserParameters(TypedDict):
    """User-defined parameters for KG generation."""

    # --- general KG settings (mirrors the JSON "kg" block order) ---
    num_entities: int
    num_triples: int
    enable_fast_generation: bool

    # --- relation usage + typing balance ---
    relation_usage_uniformity: float
    prop_untyped_entities: float

    # --- class assignment depth ---
    avg_specific_class_depth: float

    # --- multityping parameters ---
    multityping: bool
    avg_types_per_entity: float

    # --- consistency / validation ---
    check_kg_consistency: bool


class KGStatistics(TypedDict):
    """Observed statistics for a generated KG."""

    num_entities: int
    num_instantiated_relations: int
    num_triples: int
    prop_untyped_entities: float
    avg_specific_class_depth: float
    avg_types_per_entity: float


def build_kg_info(
    *,
    user_parameters: KGUserParameters,
    statistics: KGStatistics,
) -> KGInfoDict:
    """Build a JSON-friendly KGInfoDict instance.

    Centralizes the kg_info.json structure so that generators do not
    hand-craft the resulting dictionary.
    """
    return {
        "user_parameters": user_parameters,
        "statistics": statistics,
    }


# ---------------------------------------------------------------------------
# Core KG identifiers and triples TODO:Find better header name
# ---------------------------------------------------------------------------


EntityId: TypeAlias = str
RelationId: TypeAlias = str

Triple: TypeAlias = tuple[EntityId, RelationId, EntityId]
TripleSet: TypeAlias = set[Triple]
