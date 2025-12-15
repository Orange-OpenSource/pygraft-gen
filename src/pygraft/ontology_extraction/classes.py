#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Class extraction utilities for ontology graphs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pygraft.ontology_extraction.namespaces import iri_to_prefixed_name
from pygraft.ontology_extraction.queries import get_row_str, load_class_query
from pygraft.types import build_class_info

if TYPE_CHECKING:
    from rdflib import Graph

    from pygraft.ontology_extraction.namespaces import NamespaceInfoDict
    from pygraft.types import ClassInfoDict, ClassStatisticsDict


# ================================================================================================ #
# Top-level Orchestration                                                                          #
# ================================================================================================ #


def build_extracted_class_info(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> ClassInfoDict:
    """Build a ClassInfoDict extracted from an ontology graph.

    Currently this implementation populates the class list and basic statistics.
    All mapping structures (hierarchy, disjointness, layering) are returned
    empty as scaffolding to be filled incrementally as more SPARQL queries are
    added to the extraction pipeline.

    Args:
        graph:
            Parsed ontology graph that contains the TBox axioms.
        namespaces:
            Namespace information computed from the same graph. This is used to
            generate `prefix:LocalName` identifiers from raw IRIs.

    Returns:
        A ClassInfoDict instance with the classes list populated and all other
        mappings initialized but empty.
    """
    # --- class list ---
    classes = _extract_classes(graph=graph, namespaces=namespaces)
    # --- direct mappings ---
    direct_class2subclasses = _extract_direct_class2subclasses(
        graph=graph,
        namespaces=namespaces,
    )
    direct_class2superclasses = _extract_direct_class2superclasses(
        graph=graph,
        namespaces=namespaces,
    )
    # --- transitive mappings ---
    transitive_class2subclasses = _extract_transitive_class2subclasses(
        graph=graph,
        namespaces=namespaces,
    )
    transitive_class2superclasses = _extract_transitive_class2superclasses(
        graph=graph,
        namespaces=namespaces,
    )

    transitive_class2superclasses = _ensure_total_transitive_class2superclasses(
        classes=classes,
        transitive_class2superclasses=transitive_class2superclasses,
    )

    # --- disjointness mappings ---
    class2disjoints = _extract_class2disjoints(graph=graph, namespaces=namespaces)
    class2disjoints_symmetric = _build_class2disjoints_symmetric(
        class2disjoints=class2disjoints,
    )
    class2disjoints_extended = _extract_class2disjoints_extended(
        graph=graph,
        namespaces=namespaces,
    )
    # --- layer mappings ---
    layer2classes, class2layer = _compute_layer_mappings(
        classes=classes,
        direct_class2subclasses=direct_class2subclasses,
        direct_class2superclasses=direct_class2superclasses,
    )

    # --- statistics ---
    statistics = compute_stats(
        classes=classes,
        class2layer=class2layer,
        direct_class2subclasses=direct_class2subclasses,
        class2disjoints=class2disjoints,
    )

    return build_class_info(
        # --- statistics ---
        statistics=statistics,
        # --- class list ---
        classes=classes,
        # --- direct mappings ---
        direct_class2subclasses=direct_class2subclasses,
        direct_class2superclasses=direct_class2superclasses,
        # --- transitive mappings ---
        transitive_class2subclasses=transitive_class2subclasses,
        transitive_class2superclasses=transitive_class2superclasses,
        # --- disjointness mappings ---
        class2disjoints=class2disjoints,
        class2disjoints_symmetric=class2disjoints_symmetric,
        class2disjoints_extended=class2disjoints_extended,
        # --- layer mappings ---
        layer2classes=layer2classes,
        class2layer=class2layer,
    )


# ================================================================================================ #
# SPARQL Query Execution                                                                           #
# ================================================================================================ #

# ------------------------------------------------------------------------------------------------ #
# Class List                                                                                       #
# ------------------------------------------------------------------------------------------------ #


def _extract_classes(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> list[str]:
    """Extract named classes from the ontology and return them as prefix:LocalName identifiers.

    The underlying SPARQL query (classes.rq) binds ?class_uri and ensures
    that only valid named class IRIs are returned. This function performs
    the final IRI -> prefix:LocalName normalisation using the namespace
    metadata extracted from the ontology graph.

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A set of prefixed class identifiers, for example:

            {"bot:Site",
             "foaf:Agent",
             "noria:Application",
             "seas:System",
             ":LocalClass",                  # empty prefix
             "_missing:ExternalClass"}       # no known namespace matched
    """
    query_text = load_class_query("classes.rq")
    results: Iterable[Any] = graph.query(query_text)

    prefixed_classes: set[str] = set()

    for row in results:
        class_iri = get_row_str(row, "class_uri")
        prefixed_class = iri_to_prefixed_name(class_iri, namespaces)
        prefixed_classes.add(prefixed_class)

    return sorted(prefixed_classes)


# ------------------------------------------------------------------------------------------------ #
# Direct Mappings                                                                                  #
# ------------------------------------------------------------------------------------------------ #


def _extract_direct_class2subclasses(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> dict[str, list[str]]:
    """Extract direct subclass mappings as prefix:LocalName identifiers.

    The underlying SPARQL query (direct_class2subclasses.rq) must bind:
        - ?class_uri    : IRI of the parent or superclass
        - ?subClass_uri : IRI of the direct subclass

    Only IRIs are returned by the query. This function normalises the IRIs
    to prefix:LocalName identifiers and aggregates subclasses under each
    parent identifier.

    The returned mapping is fully deterministic:
    - subclass lists are sorted
    - parent keys are sorted

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A mapping from parent class identifier to a sorted list of direct
        subclass identifiers.
    """
    query_text = load_class_query("direct_class2subclasses.rq")
    results: Iterable[Any] = graph.query(query_text)

    parent_to_children: defaultdict[str, set[str]] = defaultdict(set)

    for row in results:
        parent_iri = get_row_str(row, "class_uri")
        child_iri = get_row_str(row, "subClass_uri")

        parent_id = iri_to_prefixed_name(parent_iri, namespaces)
        child_id = iri_to_prefixed_name(child_iri, namespaces)

        parent_to_children[parent_id].add(child_id)

    # Create deterministic, fully sorted mapping
    direct_mapping: dict[str, list[str]] = {
        parent_id: sorted(children_ids)
        for parent_id, children_ids in sorted(parent_to_children.items())
    }

    return direct_mapping


def _extract_direct_class2superclasses(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> dict[str, list[str]]:
    """Extract direct superclass mappings as prefix:LocalName identifiers.

    This function reuses the SPARQL query direct_class2subclasses.rq and
    computes its inverse. The query returns:
        - ?class_uri    : IRI of the parent/superclass
        - ?subClass_uri : IRI of the direct subclass

    By flipping the roles, we derive the inverse mapping:
        subclass -> [direct superclasses]

    Only IRIs are returned by the query. Normalisation to prefix:LocalName
    is applied using the same logic as all other extractors.

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A mapping from subclass identifier to a sorted list of its direct
        superclass identifiers.
    """
    query_text = load_class_query("direct_class2subclasses.rq")
    results: Iterable[Any] = graph.query(query_text)

    subclass_to_superclasses: defaultdict[str, set[str]] = defaultdict(set)

    for row in results:
        parent_iri = get_row_str(row, "class_uri")
        child_iri = get_row_str(row, "subClass_uri")

        parent_id = iri_to_prefixed_name(parent_iri, namespaces)
        child_id = iri_to_prefixed_name(child_iri, namespaces)

        subclass_to_superclasses[child_id].add(parent_id)

    direct_mapping: dict[str, list[str]] = {
        subclass_id: sorted(superclass_ids)
        for subclass_id, superclass_ids in sorted(subclass_to_superclasses.items())
    }

    return direct_mapping


# ------------------------------------------------------------------------------------------------ #
# Transitive Mappings                                                                              #
# ------------------------------------------------------------------------------------------------ #


def _extract_transitive_class2subclasses(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> dict[str, list[str]]:
    """Extract transitive subclass mappings as prefix:LocalName identifiers.

    The underlying SPARQL query (transitive_class2subclasses.rq) must bind:
        - ?class_uri    : IRI of the ancestor or superclass
        - ?subClass_uri : IRI of the (possibly indirect) subclass

    The query follows rdfs:subClassOf+ paths between named classes and filters
    out non-IRI terms. This function normalises both ends to prefix:LocalName
    identifiers and aggregates all reachable subclasses per ancestor.

    The returned mapping is deterministic:
    - descendant lists are sorted
    - ancestor keys are sorted

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A mapping from ancestor class identifier to a sorted list of all
        (direct or indirect) subclass identifiers.
    """
    query_text = load_class_query("transitive_class2subclasses.rq")
    results: Iterable[Any] = graph.query(query_text)

    ancestor_to_descendants: defaultdict[str, set[str]] = defaultdict(set)

    for row in results:
        ancestor_iri = get_row_str(row, "class_uri")
        descendant_iri = get_row_str(row, "subClass_uri")

        ancestor_id = iri_to_prefixed_name(ancestor_iri, namespaces)
        descendant_id = iri_to_prefixed_name(descendant_iri, namespaces)

        ancestor_to_descendants[ancestor_id].add(descendant_id)

    transitive_mapping: dict[str, list[str]] = {
        ancestor_id: sorted(descendant_ids)
        for ancestor_id, descendant_ids in sorted(ancestor_to_descendants.items())
    }

    return transitive_mapping


def _extract_transitive_class2superclasses(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> dict[str, list[str]]:
    """Extract transitive superclass mappings as prefix:LocalName identifiers.

    This function reuses the SPARQL query transitive_class2subclasses.rq and
    computes its inverse.

    The query returns:
        - ?class_uri    : IRI of the ancestor or superclass
        - ?subClass_uri : IRI of the (direct or indirect) subclass

    By flipping the roles, we derive the inverse mapping:
        subclass -> [all reachable superclasses]

    Only IRIs are returned by the query. Normalisation to prefix:LocalName
    is applied using the same logic as all other extractors.

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A mapping from subclass identifier to a sorted list of its transitive
        superclass identifiers.
    """
    query_text = load_class_query("transitive_class2subclasses.rq")
    results: Iterable[Any] = graph.query(query_text)

    subclass_to_ancestors: defaultdict[str, set[str]] = defaultdict(set)

    for row in results:
        ancestor_iri = get_row_str(row, "class_uri")
        descendant_iri = get_row_str(row, "subClass_uri")

        ancestor_id = iri_to_prefixed_name(ancestor_iri, namespaces)
        descendant_id = iri_to_prefixed_name(descendant_iri, namespaces)

        subclass_to_ancestors[descendant_id].add(ancestor_id)

    transitive_mapping: dict[str, list[str]] = {
        subclass_id: sorted(ancestor_ids)
        for subclass_id, ancestor_ids in sorted(subclass_to_ancestors.items())
    }

    return transitive_mapping


# ------------------------------------------------------------------------------------------------ #
# Disjointness Mappings                                                                            #
# ------------------------------------------------------------------------------------------------ #


def _extract_class2disjoints(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> dict[str, list[str]]:
    """Extract explicit class disjointness as prefix:LocalName identifiers.

    The underlying SPARQL query (class2disjoints.rq) binds:
        - ?class_uri    : IRI of the first class
        - ?disjoint_uri : IRI of a class declared disjoint with ?class_uri

    It returns the explicit owl:disjointWith assertions exactly as they appear in
    the ontology graph, in a single direction (?class_uri -> ?disjoint_uri). This
    function does not add the reverse edge and does not perform any inference.

    In terms of the exported structures in class_info:
        - class2disjoints:
            holds exactly these explicit pairs, as returned by this function;
        - class2disjoints_symmetric:
            is built later from class2disjoints by adding the reverse direction
            (if A lists B, then B also lists A);
        - class2disjoints_extended:
            is reserved for future, inferred or propagated disjointness (for
            example, along subclass chains or via more complex OWL patterns).

    Only named IRIs are returned by the SPARQL query. This function normalises
    both IRIs to prefix:LocalName identifiers and groups all disjoint partners
    per class.

    The returned mapping is deterministic:
    - disjoint lists are sorted
    - class keys are sorted

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A mapping from class identifier to a sorted list of class identifiers
        explicitly declared disjoint with it, in the forward direction as
        returned by the SPARQL query.
    """
    query_text = load_class_query("class2disjoints.rq")
    results: Iterable[Any] = graph.query(query_text)

    class_to_disjoints: defaultdict[str, set[str]] = defaultdict(set)

    for row in results:
        class_iri = get_row_str(row, "class_uri")
        disjoint_iri = get_row_str(row, "disjoint_uri")

        class_id = iri_to_prefixed_name(class_iri, namespaces)
        disjoint_id = iri_to_prefixed_name(disjoint_iri, namespaces)

        class_to_disjoints[class_id].add(disjoint_id)

    disjoint_mapping: dict[str, list[str]] = {
        class_id: sorted(disjoint_ids)
        for class_id, disjoint_ids in sorted(class_to_disjoints.items())
    }

    return disjoint_mapping


def _build_class2disjoints_symmetric(
    *,
    class2disjoints: dict[str, list[str]],
) -> list[str]:
    """Build symmetric disjointness pairs from explicit class2disjoints.

    This function takes the explicit, one-directional mapping produced by
    `_extract_class2disjoints` and converts it into a symmetric set of
    unordered class pairs. For example:

        class2disjoints:
            A -> [B, C]
            D -> [E]

        class2disjoints_symmetric (conceptually):
            (A, B), (A, C), (D, E)

    Each unordered pair appears only once in the result, regardless of whether
    the ontology contains A -> B, B -> A, or both. No OWL reasoning is
    performed; explicit declarations are merely normalized.

    Intended semantics inside pygraft:

        - class2disjoints:
            "what the ontology explicitly says"
        - class2disjoints_symmetric:
            "the same information, but as symmetric, de-duplicated pairs"
        - class2disjoints_extended:
            "reserved for future inferred / propagated disjointness"

    Output format:
        To maximize human readability in JSON, each symmetric pair is encoded
        as a single, tuple-shaped string:

            "(ClassA, ClassB)"

        where ClassA and ClassB are lexicographically sorted identifiers.
        This keeps each pair on a single JSON line while preserving its
        pair-wise semantics.

    Args:
        class2disjoints:
            Mapping from a class identifier to the list of explicitly disjoint
            class identifiers.

    Returns:
        A sorted list of string-encoded symmetric disjointness pairs, each in
        the form "(ClassA, ClassB)".
    """
    encoded_pairs: set[str] = set()

    for class_id, disjoint_list in class2disjoints.items():
        for disjoint_id in disjoint_list:
            if class_id == disjoint_id:
                continue

            class_a, class_b = sorted((class_id, disjoint_id))
            encoded_pairs.add(f"({class_a}, {class_b})")

    return sorted(encoded_pairs)


def _extract_class2disjoints_extended(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> dict[str, list[str]]:
    """Extract inherited class disjointness as prefix:LocalName identifiers.

    The underlying SPARQL query (class2disjoints_extended.rq) starts from
    explicit owl:disjointWith axioms between named classes and propagates
    disjointness downward along rdfs:subClassOf* on both sides. It binds:
        - ?class_uri             : IRI of the source class
        - ?disjoint_extended_uri : IRI of a class disjoint with ?class_uri
                                   by inheritance

    The query uses only:
        - explicit owl:disjointWith triples, and
        - explicit rdfs:subClassOf links between named classes.

    It does not interpret owl:AllDisjointClasses, owl:disjointUnionOf,
    owl:complementOf, or any disjointness expressed via blank nodes or
    complex class expressions.

    In terms of the exported structures in class_info:
        - class2disjoints:
            holds only the explicit owl:disjointWith pairs;
        - class2disjoints_symmetric:
            is a symmetric, de-duplicated view of class2disjoints;
        - class2disjoints_extended:
            holds the inheritance-extended disjointness pairs, derived from
            the explicit axioms and the subclass hierarchy.

    Only named IRIs are returned by the SPARQL query. This function normalises
    both IRIs to prefix:LocalName identifiers and groups all extended disjoint
    partners per class.

    The returned mapping is deterministic:
    - extended disjoint lists are sorted
    - class keys are sorted

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A mapping from class identifier to a sorted list of class identifiers
        that are disjoint with it by inheritance, according to the ontology's
        explicit owl:disjointWith and rdfs:subClassOf structure.
    """
    query_text = load_class_query("class2disjoints_extended.rq")
    results: Iterable[Any] = graph.query(query_text)

    class_to_extended: defaultdict[str, set[str]] = defaultdict(set)

    for row in results:
        class_iri = get_row_str(row, "class_uri")
        disjoint_ext_iri = get_row_str(row, "disjoint_extended_uri")

        class_id = iri_to_prefixed_name(class_iri, namespaces)
        disjoint_ext_id = iri_to_prefixed_name(disjoint_ext_iri, namespaces)

        if class_id == disjoint_ext_id:
            continue

        class_to_extended[class_id].add(disjoint_ext_id)

    extended_mapping: dict[str, list[str]] = {
        class_id: sorted(disjoint_ids)
        for class_id, disjoint_ids in sorted(class_to_extended.items())
    }

    return extended_mapping


# ------------------------------------------------------------------------------------------------ #
# Layer Mappings                                                                                   #
# ------------------------------------------------------------------------------------------------ #


def _compute_layer_mappings(
    *,
    classes: list[str],
    direct_class2subclasses: dict[str, list[str]],
    direct_class2superclasses: dict[str, list[str]],
) -> tuple[dict[int, list[str]], dict[str, int]]:
    """Compute ontology-local layer mappings from direct subclass edges.

    This function assigns a positive integer "layer" (depth) to every class in
    `classes`, using only the subclass structure induced by those classes.
    Superclasses that are not present in `classes` are treated as external
    (for example, FOAF, SEAS, ORG) and conceptually live at layer 0
    alongside owl:Thing. They do not increase depth.

    The resulting layers are:

    - ontology-local:
      depth is measured only relative to the `classes` universe;
    - DAG-aware:
      multiple inheritance is allowed and handled correctly; and
    - compatible with the KG generator:
      higher layer indices mean more specific classes.

    Semantics:

    * Layer 1:
      A class is placed in layer 1 if it has no "internal" superclasses, where
      an internal superclass is a parent that also appears in `classes`.
      Classes whose parents are exclusively external behave as if they were
      directly under owl:Thing and become ontology-local roots.

    * Non-root classes:
      A non-root class is assigned a layer only after all of its internal
      parents have already been assigned one. Its layer is then:

          1 + max(layer(parent) for parent in internal_parents)

      This ensures that in the presence of multiple inheritance a class is
      always at least as deep as its deepest internal parent.

    * Fallback layer:
      If any classes remain unassigned after the BFS-style propagation
      (for example, due to cycles or malformed subclass links), they are all
      assigned to a single fallback layer equal to:

          max_assigned_layer + 1

      This guarantees that every class in `classes` receives a layer and keeps
      `class2layer` total, even for pathological graphs.

    Args:
        classes:
            Sorted list of class identifiers that define the internal universe
            of the ontology. Only these are layered; any superclass not in this
            list is treated as external.
        direct_class2subclasses:
            Mapping from a class identifier to the list of its direct subclasses.
            This is the explicit, one-hop rdfs:subClassOf structure.
        direct_class2superclasses:
            Mapping from a class identifier to the list of its direct
            superclasses. This is the inverse one-hop rdfs:subClassOf structure.

    Returns:
        A tuple (layer2classes, class2layer) where:

        layer2classes:
            Mapping from layer index (as an int) to a sorted list of class
            identifiers at that layer. For example: {1: ["A", "B"], 2: ["C"]}.
            When serialized to JSON, these keys naturally become strings.
        class2layer:
            Mapping from class identifier to its integer layer index. Every
            class in `classes` appears exactly once in this mapping.

    Notes:
        The highest layer value in `class2layer` is the ontology's
        `hierarchy_depth`. The KG generator uses these layers as a measure of
        type specificity when sampling class assignments for entities.
    """
    classes_set: set[str] = set(classes)

    # ------------------------------------------------------------
    # 1. Identify layer-1 roots
    # ------------------------------------------------------------
    roots: list[str] = []
    for class_id in classes:
        superclasses = direct_class2superclasses.get(class_id, [])
        internal_superclasses = [parent for parent in superclasses if parent in classes_set]
        if not internal_superclasses:  # only external parents → layer 1
            roots.append(class_id)

    roots_sorted = sorted(roots)

    # ------------------------------------------------------------
    # 2. BFS layering (DAG-aware)
    # ------------------------------------------------------------
    class2layer: dict[str, int] = {}
    current_frontier = roots_sorted
    current_layer_index = 1

    while current_frontier:
        # Assign this layer to all classes in the current frontier
        for class_id in current_frontier:
            existing_layer = class2layer.get(class_id)
            if existing_layer is None or current_layer_index < existing_layer:
                class2layer[class_id] = current_layer_index

        # Determine next layer frontier
        next_frontier: set[str] = set()

        for parent_id in current_frontier:
            for child_id in direct_class2subclasses.get(parent_id, []):
                if child_id not in classes_set:
                    continue  # external → ignore

                # Already at a shallower or equal layer → skip
                if child_id in class2layer and class2layer[child_id] <= current_layer_index:
                    continue

                # Collect child's internal parents
                child_superclasses = direct_class2superclasses.get(child_id, [])
                internal_superclasses = [
                    parent for parent in child_superclasses if parent in classes_set
                ]

                # Only advance child when all internal parents have layers
                if all(parent in class2layer for parent in internal_superclasses):
                    next_frontier.add(child_id)

        current_frontier = sorted(next_frontier)
        current_layer_index += 1

    # ------------------------------------------------------------
    # 3. Fallback for unassigned nodes (cycles, bad data)
    # ------------------------------------------------------------
    if len(class2layer) < len(classes):
        fallback_layer = max(class2layer.values(), default=0) + 1
        for class_id in classes:
            if class_id not in class2layer:
                class2layer[class_id] = fallback_layer

    # ------------------------------------------------------------
    # 4. Build layer2classes (int keys for JSON-friendly mapping)
    # ------------------------------------------------------------
    layer2classes: dict[int, list[str]] = {}
    for class_id, layer_index in class2layer.items():
        layer2classes.setdefault(layer_index, []).append(class_id)

    for layer_index in list(layer2classes.keys()):
        layer2classes[layer_index] = sorted(layer2classes[layer_index])

    return layer2classes, class2layer


# ------------------------------------------------------------------------------------------------ #
# Statistics                                                                                       #
# ------------------------------------------------------------------------------------------------ #


def compute_stats(
    *,
    classes: list[str],
    class2layer: dict[str, int],
    direct_class2subclasses: dict[str, list[str]],
    class2disjoints: dict[str, list[str]],
) -> ClassStatisticsDict:
    """Compute high-level structural statistics for the extracted ontology.

    These statistics summarize the ontology structure based solely on
    explicit classes, subclass links, and disjointness declarations.

    Returned metrics:
        num_classes: total number of extracted classes.
        hierarchy_depth: maximum assigned layer (integer).
        avg_class_depth: mean layer index across all classes (rounded to 2 decimals).
        avg_children_per_parent: average number of direct subclasses per parent (rounded).
        avg_class_disjointness: average number of disjoint declarations per class (rounded).
    """
    # -----------------------------
    # num_classes
    # -----------------------------
    num_classes = len(classes)

    # -----------------------------
    # hierarchy_depth
    # -----------------------------
    hierarchy_depth = max(class2layer.values()) if class2layer else 0

    # -----------------------------
    # avg_class_depth
    # -----------------------------
    avg_class_depth = round(sum(class2layer.values()) / len(class2layer), 2) if class2layer else 0.0

    # -----------------------------
    # avg_children_per_parent
    # -----------------------------
    total_children = sum(len(v) for v in direct_class2subclasses.values())
    parents_with_children = sum(1 for v in direct_class2subclasses.values() if v)

    if parents_with_children > 0:
        avg_children_per_parent = round(total_children / parents_with_children, 2)
    else:
        avg_children_per_parent = 0.0

    # -----------------------------
    # avg_class_disjointness
    # -----------------------------
    if classes:
        total_disjoints = sum(len(v) for v in class2disjoints.values())
        avg_class_disjointness = round(total_disjoints / len(classes), 2)
    else:
        avg_class_disjointness = 0.0

    return {
        "num_classes": num_classes,
        "hierarchy_depth": hierarchy_depth,
        "avg_class_depth": avg_class_depth,
        "avg_children_per_parent": avg_children_per_parent,
        "avg_class_disjointness": avg_class_disjointness,
    }


# ------------------------------------------------------------------------------------------------ #
# KG Generators Helpers                                                                            #
# ------------------------------------------------------------------------------------------------ #

def _ensure_total_transitive_class2superclasses(
    *,
    classes: list[str],
    transitive_class2superclasses: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Ensure transitive superclass mapping is total over extracted classes.

    The KG generator expects `transitive_class2superclasses` to contain a key for
    every class that may be sampled as a "specific" class during type assignment.
    When a class is missing, downstream code crashes with KeyError.

    Semantics:
        - For every class in `classes`, guarantee there is a mapping entry.
        - If a class has no extracted transitive superclasses, fall back to the
          identity mapping: cls -> [cls].
        - Keep only ancestors that are inside `classes` (ontology-local universe).
        - Always include the class itself in its own transitive set.

    Args:
        classes:
            Sorted list of extracted class identifiers (e.g., "bot:Space").
        transitive_class2superclasses:
            Mapping from class identifier to its transitive superclasses, as
            produced by SPARQL extraction.

    Returns:
        A sanitized mapping where every class in `classes` appears as a key, and
        every value list is sorted and de-duplicated.
    """
    classes_set: set[str] = set(classes)
    total_mapping: dict[str, list[str]] = dict(transitive_class2superclasses)

    for class_id in classes:
        raw_ancestors = total_mapping.get(class_id)
        if not raw_ancestors:
            total_mapping[class_id] = [class_id]
            continue

        cleaned_ancestors = [a for a in raw_ancestors if a in classes_set and a]
        cleaned_ancestors.append(class_id)

        total_mapping[class_id] = sorted(set(cleaned_ancestors))

    return total_mapping

