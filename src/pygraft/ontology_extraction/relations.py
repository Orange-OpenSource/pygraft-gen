#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Relation extraction utilities for ontology graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

from pygraft.ontology_extraction.namespaces import iri_to_prefixed_name
from pygraft.ontology_extraction.queries import get_row_str, load_relation_query
from pygraft.types import build_relation_info

if TYPE_CHECKING:
    from rdflib import Graph

    from pygraft.ontology_extraction.namespaces import NamespaceInfoDict
    from pygraft.types import RelationInfoDict, RelationStatistics


# ================================================================================================ #
# Top-level Orchestration                                                                          #
# ================================================================================================ #


def build_extracted_relation_info(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> RelationInfoDict:
    """Build a minimal RelationInfoDict extracted from an ontology.

    For now this is scaffolding only: it returns an empty but structurally
    complete RelationInfoDict so that downstream code can rely on the JSON
    shape while we implement real extraction.

    Args:
        graph:
            Parsed ontology graph that contains the TBox axioms.
        namespaces:
            Namespace information computed from the same graph. Currently
            unused but reserved for later use to generate "prefix:basename"
            identifiers for object properties.

    Returns:
        RelationInfoDict with all mappings empty and statistics set to zero.
    """
    # --- relation list ---
    relations = _extract_relations(graph=graph, namespaces=namespaces)
    # --- OWL pattern mappings ---
    rel2patterns = _extract_rel2patterns(graph=graph, namespaces=namespaces)
    # --- logical characteristics ---
    characteristics = _build_relation_characteristics(rel2patterns=rel2patterns)
    # --- inverse-of relationships ---
    inverseof_relations, rel2inverse = _extract_inverseof_relations(
        graph=graph,
        namespaces=namespaces,
    )
    # --- subPropertyOf hierarchy ---
    subrelations, rel2superrel = _extract_subrelations(graph=graph, namespaces=namespaces)
    # --- disjointness mappings ---
    rel2disjoints = _extract_rel2disjoints(graph=graph, namespaces=namespaces)
    rel2disjoints_symmetric = _build_rel2disjoints_symmetric(rel2disjoints=rel2disjoints)
    rel2disjoints_extended = _extract_rel2disjoints_extended(graph=graph, namespaces=namespaces)
    # --- domain / range ---
    rel2dom, rel2range = _extract_rel2dom_rel2range(graph=graph, namespaces=namespaces)
    # --- statistics ---
    statistics: RelationStatistics = _compute_relation_statistics(
        relations=relations,
        rel2patterns=rel2patterns,
        characteristics=characteristics,
        inverseof_relations=inverseof_relations,
        subrelations=subrelations,
        rel2dom=rel2dom,
        rel2range=rel2range,
    )

    return build_relation_info(
        # --- statistics ---
        statistics=statistics,
        # --- relation list ---
        relations=relations,
        # --- OWL pattern mappings ---
        rel2patterns=rel2patterns,
        # --- logical characteristics ---
        reflexive_relations=characteristics["reflexive_relations"],
        irreflexive_relations=characteristics["irreflexive_relations"],
        symmetric_relations=characteristics["symmetric_relations"],
        asymmetric_relations=characteristics["asymmetric_relations"],
        functional_relations=characteristics["functional_relations"],
        inversefunctional_relations=characteristics["inversefunctional_relations"],
        transitive_relations=characteristics["transitive_relations"],
        # --- inverse-of relationships ---
        inverseof_relations=inverseof_relations,
        rel2inverse=rel2inverse,
        # --- subPropertyOf hierarchy ---
        subrelations=subrelations,
        rel2superrel=rel2superrel,
        # --- disjointness mappings ---
        rel2disjoints=rel2disjoints,
        rel2disjoints_symmetric=rel2disjoints_symmetric,
        rel2disjoints_extended=rel2disjoints_extended,
        # --- domain / range ---
        rel2dom=rel2dom,
        rel2range=rel2range,
    )



# ------------------------------------------------------------------------------------------------ #
# Query Loader Helpers                                                                             #
# ------------------------------------------------------------------------------------------------ #
def _load_relation_query_with_seed(query_filename: str) -> str:
    """Load a relation query and inject the shared relations seed block when requested.

    Injection is safe and deterministic:
      - Only replaces lines whose stripped content is exactly "# @RELATIONS_SEED".
      - Does not expand placeholders mentioned in comments or inline text.
      - Preserves indentation of the marker line (useful for readability).

    Args:
        query_filename:
            The filename of the .rq query to load.

    Returns:
        Query text with the shared seed block injected when requested.

    Raises:
        ValueError:
            If the query mentions the marker token but never uses it as a
            standalone line (likely a formatting mistake).
    """
    seed_marker = "# @RELATIONS_SEED"
    query_text = load_relation_query(query_filename)

    lines = query_text.splitlines(keepends=True)

    marker_line_indexes: list[int] = []
    for i, line in enumerate(lines):
        if line.strip() == seed_marker:
            marker_line_indexes.append(i)

    if not marker_line_indexes:
        if "@RELATIONS_SEED" in query_text:
            msg = (
                f"Query '{query_filename}' mentions '@RELATIONS_SEED' but does not contain "
                f"'{seed_marker}' as a standalone line. Put it on its own line to enable safe injection."
            )
            raise ValueError(msg)
        return query_text

    seed_block = load_relation_query("relations_seed.rq").rstrip() + "\n"

    rendered_lines: list[str] = []
    for line in lines:
        if line.strip() != seed_marker:
            rendered_lines.append(line)
            continue

        # Preserve indentation (spaces/tabs before the placeholder).
        indentation = line[: len(line) - len(line.lstrip())]
        indented_seed_block = "".join(
            f"{indentation}{seed_line}"
            for seed_line in seed_block.splitlines(keepends=True)
        )
        rendered_lines.append(indented_seed_block)

    return "".join(rendered_lines)



# ================================================================================================ #
# SPARQL Query Execution                                                                           #
# ================================================================================================ #


# ------------------------------------------------------------------------------------------------ #
# Relation List                                                                                    #
# ------------------------------------------------------------------------------------------------ #
def _extract_relations(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> list[str]:
    """Extract object properties and return them as prefix:LocalName identifiers.

    The underlying SPARQL query (relations.rq) binds ?prop_uri and ensures
    that only relevant object-property IRIs are returned. It filters out
    datatype properties, annotation properties, and properties whose range
    is a literal or an XSD datatype, while following rdfs:subPropertyOf*
    chains to include inherited properties.

    This function performs the final IRI -> prefix:LocalName normalisation
    using the namespace metadata extracted from the ontology graph.

    Args:
        graph:
            RDF graph containing the ontology TBox axioms to query.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A set of prefixed relation identifiers, for example:

            {
                "noria:hasService",
                "noria:dependsOn",
                "prov:wasGeneratedBy",
                "org:hasUnit",
                ":localRelation",             # empty prefix
                "_missing:externalRelation"   # no known namespace matched
            }
    """
    query_text = _load_relation_query_with_seed("relations.rq")
    results: Iterable[Any] = graph.query(query_text)

    prefixed_relations: set[str] = set()

    for row in results:
        prop_iri = get_row_str(row, "prop_uri")
        prefixed_relation = iri_to_prefixed_name(prop_iri, namespaces)
        prefixed_relations.add(prefixed_relation)

    return sorted(prefixed_relations)


# ------------------------------------------------------------------------------------------------ #
# OWL Pattern Mappings                                                                             #
# ------------------------------------------------------------------------------------------------ #
def _extract_rel2patterns(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> dict[str, list[str]]:
    """Extract OWL characteristic patterns for ontology-local object properties.

    The underlying SPARQL query (rel2patterns.rq) binds:
        - ?prop_uri    : IRI of an ontology-local object-style property
        - ?pattern_uri : IRI of an OWL characteristic type (may be unbound)

    The property universe matches exactly the one used for relation extraction:
    starting from explicitly declared owl:ObjectProperty resources, the query
    expands upward along rdfs:subPropertyOf* and filters out datatype,
    annotation, and literal-valued properties. This ensures that every
    extracted relation is considered for pattern profiling.

    For each extracted property, the query optionally binds one of the
    supported OWL characteristic types when it is explicitly declared in the
    ontology:

        - owl:ReflexiveProperty
        - owl:IrreflexiveProperty
        - owl:SymmetricProperty
        - owl:AsymmetricProperty
        - owl:TransitiveProperty
        - owl:FunctionalProperty
        - owl:InverseFunctionalProperty

    This function normalizes property and pattern IRIs to prefix:LocalName
    identifiers using the ontology namespaces. Pattern identifiers are
    further compacted by stripping the trailing "Property" suffix (for
    example, owl:FunctionalProperty -> owl:Functional).

    The returned mapping is total and deterministic:
        - every extracted relation appears as a key,
        - relations with no declared characteristics map to an empty list,
        - pattern lists are sorted,
        - property keys are sorted.

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A mapping from relation identifier to a sorted list of OWL pattern
        labels, for example:

            {
                "noria:hasService": ["owl:Functional", "owl:Transitive"],
                "noria:dependsOn": [],
                "noria:isPartOf": ["owl:Transitive"]
            }
    """
    query_text = _load_relation_query_with_seed("rel2patterns.rq")
    results: Iterable[Any] = graph.query(query_text)

    rel_to_patterns: dict[str, set[str]] = {}

    for row in results:
        prop_iri = get_row_str(row, "prop_uri")
        prop_id = iri_to_prefixed_name(prop_iri, namespaces)

        # Ensure the property key exists even if no pattern is bound
        patterns_for_prop = rel_to_patterns.setdefault(prop_id, set())

        row_mapping = cast(Mapping[str, Any], row)
        pattern_term = row_mapping.get("pattern_uri")
        if pattern_term is None:
            continue

        pattern_iri = str(pattern_term)
        pattern_prefixed = iri_to_prefixed_name(pattern_iri, namespaces)

        # Drop the trailing "Property" suffix if present, e.g. owl:FunctionalProperty -> owl:Functional
        if pattern_prefixed.endswith("Property"):
            normalized_pattern = pattern_prefixed[: -len("Property")]
        else:
            normalized_pattern = pattern_prefixed

        patterns_for_prop.add(normalized_pattern)

    rel2patterns: dict[str, list[str]] = {
        rel_id: sorted(patterns) for rel_id, patterns in sorted(rel_to_patterns.items())
    }

    return rel2patterns


# ------------------------------------------------------------------------------------------------ #
# Logical Characteristics                                                                          #
# ------------------------------------------------------------------------------------------------ #


def _build_relation_characteristics(
    *,
    rel2patterns: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Group relations by explicitly declared OWL characteristics.

    This function reorganizes `rel2patterns` into characteristic-specific
    buckets. Each bucket contains the relations that explicitly declare the
    corresponding OWL characteristic in the ontology. No inference or
    propagation is performed.

    Relations with no declared characteristics do not appear in any bucket.
    All output lists are sorted for deterministic results.

    Args:
        rel2patterns:
            Mapping from relation identifier to a list of OWL characteristic
            labels.

    Returns:
        A mapping from characteristic name to the sorted list of relations
        declaring it.
    """
    # Initialize empty buckets
    buckets: dict[str, list[str]] = {
        "reflexive_relations": [],
        "irreflexive_relations": [],
        "symmetric_relations": [],
        "asymmetric_relations": [],
        "functional_relations": [],
        "inversefunctional_relations": [],
        "transitive_relations": [],
    }

    # Pattern -> bucket_name mapping
    pattern_to_bucket = {
        "owl:Reflexive": "reflexive_relations",
        "owl:Irreflexive": "irreflexive_relations",
        "owl:Symmetric": "symmetric_relations",
        "owl:Asymmetric": "asymmetric_relations",
        "owl:Functional": "functional_relations",
        "owl:InverseFunctional": "inversefunctional_relations",
        "owl:Transitive": "transitive_relations",
    }

    # Fill buckets based on declared patterns
    for rel_id, patterns in rel2patterns.items():
        for p in patterns:
            bucket = pattern_to_bucket.get(p)
            if bucket:
                buckets[bucket].append(rel_id)

    # Deterministic output (sorted lists)
    for key in buckets:
        buckets[key] = sorted(buckets[key])

    return buckets


# ------------------------------------------------------------------------------------------------ #
# Inverse-of Relationships                                                                         #
# ------------------------------------------------------------------------------------------------ #


def _extract_inverseof_relations(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> tuple[list[str], dict[str, str]]:
    """Extract explicitly declared owl:inverseOf pairs and build a symmetric mapping.

    The underlying SPARQL query (inverseof_relations.rq) binds:
        - ?prop_uri    : IRI of an ontology-local object-style property
        - ?inverse_uri : IRI of its explicitly declared inverse property

    The query is explicit-only: it matches only triples of the form
    (?prop_uri owl:inverseOf ?inverse_uri) present in the ontology graph.
    This function then normalizes IRIs to prefix:LocalName identifiers and
    builds:

        - inverseof_relations: sorted list of all relations involved in at
          least one inverseOf pair (either side).
        - rel2inverse: symmetric mapping where both directions are present
          (A -> B and B -> A) for each declared pair.

    If a relation is declared inverseOf multiple different relations, this
    function raises an error to avoid producing an ambiguous rel2inverse map.

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A tuple (inverseof_relations, rel2inverse) where:
            - inverseof_relations is a sorted list of relation identifiers.
            - rel2inverse maps each relation in a declared inverseOf pair to
              its inverse (symmetric).

    Raises:
        ValueError:
            If conflicting inverseOf declarations are found for the same
            relation (for example, A inverseOf B and A inverseOf C).
    """
    query_text = _load_relation_query_with_seed("inverseof_relations.rq")
    results: Iterable[Any] = graph.query(query_text)

    rel2inverse: dict[str, str] = {}
    inverseof_set: set[str] = set()

    for row in results:
        prop_iri = get_row_str(row, "prop_uri")
        inv_iri = get_row_str(row, "inverse_uri")

        rel_id = iri_to_prefixed_name(prop_iri, namespaces)
        inv_id = iri_to_prefixed_name(inv_iri, namespaces)

        if rel_id == inv_id:
            continue

        inverseof_set.add(rel_id)
        inverseof_set.add(inv_id)

        existing_inverse = rel2inverse.get(rel_id)
        if existing_inverse is not None and existing_inverse != inv_id:
            msg = (
                f"Conflicting owl:inverseOf declarations for '{rel_id}': "
                f"'{existing_inverse}' vs '{inv_id}'."
            )
            raise ValueError(msg)

        existing_inverse_rev = rel2inverse.get(inv_id)
        if existing_inverse_rev is not None and existing_inverse_rev != rel_id:
            msg = (
                f"Conflicting owl:inverseOf declarations for '{inv_id}': "
                f"'{existing_inverse_rev}' vs '{rel_id}'."
            )
            raise ValueError(msg)

        rel2inverse[rel_id] = inv_id
        rel2inverse[inv_id] = rel_id

    inverseof_relations = sorted(inverseof_set)
    rel2inverse = {k: rel2inverse[k] for k in sorted(rel2inverse)}

    return inverseof_relations, rel2inverse


# ------------------------------------------------------------------------------------------------ #
# SubPropertyOf Hierarchy                                                                          #
# ------------------------------------------------------------------------------------------------ #


def _extract_subrelations(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> tuple[list[str], dict[str, list[str]]]:
    """Extract explicit one-hop rdfs:subPropertyOf links between relations.

    The underlying SPARQL query (subrelations.rq) binds:
        - ?subprop_uri   : IRI of the sub-property
        - ?superprop_uri : IRI of the super-property

    This extractor is explicit-only: it captures only asserted one-hop
    rdfs:subPropertyOf edges exactly as they appear in the ontology graph.
    No transitive closure, inference, or hierarchy expansion is performed.

    Multiple direct super-properties per relation are allowed and preserved,
    reflecting the fact that OWL property hierarchies form a DAG rather than
    a strict tree.

    Both sub- and super-properties are normalized to prefix:LocalName
    identifiers using the ontology namespaces.

    The output follows the same structural conventions as class hierarchies:
        - subrelations lists all relations that appear as a subrelation
        - rel2superrel maps each subrelation to its list of direct superrelations

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A tuple (subrelations, rel2superrel) where:
            - subrelations is the sorted list of all relations that appear on
              the left-hand side of an rdfs:subPropertyOf assertion.
            - rel2superrel is a mapping from subrelation identifier to a sorted
              list of its direct superrelation identifiers.

    Notes:
        - Only explicit rdfs:subPropertyOf triples are extracted.
        - Transitive closure should be computed later as a derived structure
          if needed by downstream components.
    """
    query_text = _load_relation_query_with_seed("subrelations.rq")
    results: Iterable[Any] = graph.query(query_text)

    rel2superrel: dict[str, set[str]] = {}
    subrelation_set: set[str] = set()

    for row in results:
        subprop_iri = get_row_str(row, "subprop_uri")
        superprop_iri = get_row_str(row, "superprop_uri")

        subrel_id = iri_to_prefixed_name(subprop_iri, namespaces)
        superrel_id = iri_to_prefixed_name(superprop_iri, namespaces)

        # Ignore trivial self-links if present
        if subrel_id == superrel_id:
            continue

        rel2superrel.setdefault(subrel_id, set()).add(superrel_id)
        subrelation_set.add(subrel_id)

    subrelations = sorted(subrelation_set)

    rel2superrel_final: dict[str, list[str]] = {
        rel_id: sorted(superrels) for rel_id, superrels in sorted(rel2superrel.items())
    }

    return subrelations, rel2superrel_final


# ------------------------------------------------------------------------------------------------ #
# Disjointness Mappings                                                                            #
# ------------------------------------------------------------------------------------------------ #
def _extract_rel2disjoints(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> dict[str, list[str]]:
    """Extract explicit property disjointness as prefix:LocalName identifiers.

    The underlying SPARQL query (rel2disjoints.rq) binds:
        - ?prop_uri     : IRI of the first property
        - ?disjoint_uri : IRI of a property declared disjoint with ?prop_uri

    The query is explicit-only: it returns owl:propertyDisjointWith triples
    exactly as written in the ontology graph (single direction). Any symmetric
    view is derived later in Python.

    In exported relation_info structures:
        - rel2disjoints:
            holds exactly the explicit pairs returned by this function;
        - rel2disjoints_symmetric:
            is built later from rel2disjoints by de-duplicating unordered pairs;
        - rel2disjoints_extended:
            reserved for future derived/inherited disjointness.

    Only IRIs are returned by the SPARQL query, and both sides are normalized
    to prefix:LocalName identifiers. Output is deterministic (sorted keys and
    sorted disjoint lists).

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        Mapping from relation identifier to a sorted list of relation identifiers
        explicitly declared disjoint with it (forward direction only).
    """
    query_text = _load_relation_query_with_seed("rel2disjoints.rq")
    results: Iterable[Any] = graph.query(query_text)

    rel_to_disjoints: dict[str, set[str]] = {}

    for row in results:
        prop_iri = get_row_str(row, "prop_uri")
        disjoint_iri = get_row_str(row, "disjoint_uri")

        rel_id = iri_to_prefixed_name(prop_iri, namespaces)
        disjoint_id = iri_to_prefixed_name(disjoint_iri, namespaces)

        if rel_id == disjoint_id:
            continue

        rel_to_disjoints.setdefault(rel_id, set()).add(disjoint_id)

    rel2disjoints: dict[str, list[str]] = {
        rel_id: sorted(disjoints) for rel_id, disjoints in sorted(rel_to_disjoints.items())
    }

    return rel2disjoints


def _build_rel2disjoints_symmetric(
    *,
    rel2disjoints: dict[str, list[str]],
) -> list[str]:
    """Build symmetric, de-duplicated disjointness pairs from explicit rel2disjoints.

    This function converts the explicit, one-directional mapping produced by
    `_extract_rel2disjoints` into a symmetric set of unordered relation pairs.
    Each unordered pair appears once, even if the ontology asserts both
    directions.

    Output format:
        Each symmetric pair is encoded as:
            "(RelA, RelB)"
        where RelA and RelB are lexicographically sorted identifiers.

    Args:
        rel2disjoints:
            Mapping from relation identifier to explicitly disjoint relation
            identifiers (forward direction as extracted).

    Returns:
        Sorted list of string-encoded symmetric disjointness pairs.
    """
    encoded_pairs: set[str] = set()

    for rel_id, disjoint_list in rel2disjoints.items():
        for disjoint_id in disjoint_list:
            if rel_id == disjoint_id:
                continue

            rel_a, rel_b = sorted((rel_id, disjoint_id))
            encoded_pairs.add(f"({rel_a}, {rel_b})")

    return sorted(encoded_pairs)

def _extract_rel2disjoints_extended(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> dict[str, list[str]]:
    """Extract downward-propagated property disjointness as prefix:LocalName identifiers.

    The underlying SPARQL query (rel2disjoints_extended.rq) binds:
        - ?prop_uri     : IRI of an ontology-local object-style property
        - ?disjoint_uri : IRI of a property disjoint with ?prop_uri after
          downward propagation along rdfs:subPropertyOf*

    This extractor returns the propagated owl:propertyDisjointWith pairs exactly
    as emitted by the query (single direction). Any symmetric / de-duplicated
    view is derived later in Python, the same way as for rel2disjoints.

    Output is deterministic (sorted keys and sorted disjoint lists).

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        Mapping from relation identifier to a sorted list of relation identifiers
        disjoint with it under downward subPropertyOf propagation (forward
        direction only).
    """
    query_text = _load_relation_query_with_seed("rel2disjoints_extended.rq")
    results: Iterable[Any] = graph.query(query_text)

    rel_to_disjoints: dict[str, set[str]] = {}

    for row in results:
        prop_iri = get_row_str(row, "prop_uri")
        disjoint_iri = get_row_str(row, "disjoint_uri")

        rel_id = iri_to_prefixed_name(prop_iri, namespaces)
        disjoint_id = iri_to_prefixed_name(disjoint_iri, namespaces)

        if rel_id == disjoint_id:
            continue

        rel_to_disjoints.setdefault(rel_id, set()).add(disjoint_id)

    rel2disjoints_extended: dict[str, list[str]] = {
        rel_id: sorted(disjoints) for rel_id, disjoints in sorted(rel_to_disjoints.items())
    }

    return rel2disjoints_extended




# ------------------------------------------------------------------------------------------------ #
# Domain / Range                                                                                   #
# ------------------------------------------------------------------------------------------------ #


def _extract_rel2dom_rel2range(
    *,
    graph: Graph,
    namespaces: NamespaceInfoDict,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Extract domain and range mappings for ontology-local object properties.

    The underlying SPARQL query (rel2dom_rel2range.rq) binds:
        - ?prop_uri   : IRI of an ontology-local object-style property
        - ?domain_uri : IRI of a domain class (may be unbound)
        - ?range_uri  : IRI of a range class (may be unbound)

    The property universe matches relations.rq. Domain and range values are
    inherited via rdfs:subPropertyOf* in the query, and redundant superclasses
    are dropped when a stricter subclass is also present for the same property.

    This function normalizes IRIs to prefix:LocalName identifiers and builds:
        - rel2dom: property -> sorted list of domain class identifiers
        - rel2range: property -> sorted list of range class identifiers

    Properties with no bound domain/range are omitted from the corresponding
    mapping (i.e., they do not appear as keys).

    Args:
        graph:
            RDF graph containing the ontology TBox axioms.
        namespaces:
            Namespace metadata used to generate prefix:LocalName identifiers.

    Returns:
        A tuple (rel2dom, rel2range) where:
            - rel2dom maps relation identifier -> sorted list of domain classes.
            - rel2range maps relation identifier -> sorted list of range classes.
    """
    query_text = _load_relation_query_with_seed("rel2dom_rel2range.rq")
    results: Iterable[Any] = graph.query(query_text)

    rel_to_domains: dict[str, set[str]] = {}
    rel_to_ranges: dict[str, set[str]] = {}

    for row in results:
        prop_iri = get_row_str(row, "prop_uri")
        rel_id = iri_to_prefixed_name(prop_iri, namespaces)

        row_mapping = cast(Mapping[str, Any], row)

        domain_term = row_mapping.get("domain_uri")
        if domain_term is not None:
            dom_id = iri_to_prefixed_name(str(domain_term), namespaces)
            rel_to_domains.setdefault(rel_id, set()).add(dom_id)

        range_term = row_mapping.get("range_uri")
        if range_term is not None:
            rng_id = iri_to_prefixed_name(str(range_term), namespaces)
            rel_to_ranges.setdefault(rel_id, set()).add(rng_id)

    rel2dom: dict[str, list[str]] = {
        rel_id: sorted(domains) for rel_id, domains in sorted(rel_to_domains.items())
    }
    rel2range: dict[str, list[str]] = {
        rel_id: sorted(ranges) for rel_id, ranges in sorted(rel_to_ranges.items())
    }

    return rel2dom, rel2range


# ------------------------------------------------------------------------------------------------ #
# Statistics                                                                                       #
# ------------------------------------------------------------------------------------------------ #


def _compute_relation_statistics(
    *,
    relations: list[str],
    rel2patterns: dict[str, list[str]],
    characteristics: dict[str, list[str]],
    inverseof_relations: list[str],
    subrelations: list[str],
    rel2dom: dict[str, list[str]],
    rel2range: dict[str, list[str]],
) -> RelationStatistics:
    """Compute high-level structural statistics for extracted relations.

    These statistics summarize relation structure using only extracted,
    explicit ontology data (no inference).

    Returned metrics:
        - num_relations
        - prop_* ratios: fraction of relations declaring each feature
        - prop_profiled_relations: fraction of relations with any extracted feature
        - relation_specificity: average number of domain/range constraints per relation
    """

    def _ratio(*, count: int, total: int) -> float:
        """Return a stable count/total ratio rounded to 2 decimals."""
        if total <= 0:
            return 0.0
        return round(count / total, 2)

    # -----------------------------
    # num_relations
    # -----------------------------
    num_relations = len(relations)
    relations_set = set(relations)

    # -----------------------------
    # profiled_relations
    # -----------------------------
    profiled_relations: set[str] = set()

    for rel_id, patterns in rel2patterns.items():
        if patterns:
            profiled_relations.add(rel_id)

    profiled_relations.update(inverseof_relations)
    profiled_relations.update(subrelations)
    profiled_relations.update(rel2dom.keys())
    profiled_relations.update(rel2range.keys())

    # Defensive intersection with known relations
    profiled_relations &= relations_set

    # -----------------------------
    # relation_specificity
    # -----------------------------
    total_constraints = 0
    for rel_id in relations:
        total_constraints += len(rel2dom.get(rel_id, []))
        total_constraints += len(rel2range.get(rel_id, []))

    relation_specificity = round(total_constraints / num_relations, 2) if num_relations > 0 else 0.0

    # -----------------------------
    # characteristic ratios
    # -----------------------------
    return {
        "num_relations": num_relations,
        "prop_reflexive": _ratio(
            count=len(characteristics["reflexive_relations"]),
            total=num_relations,
        ),
        "prop_irreflexive": _ratio(
            count=len(characteristics["irreflexive_relations"]),
            total=num_relations,
        ),
        "prop_functional": _ratio(
            count=len(characteristics["functional_relations"]),
            total=num_relations,
        ),
        "prop_inversefunctional": _ratio(
            count=len(characteristics["inversefunctional_relations"]),
            total=num_relations,
        ),
        "prop_symmetric": _ratio(
            count=len(characteristics["symmetric_relations"]),
            total=num_relations,
        ),
        "prop_asymmetric": _ratio(
            count=len(characteristics["asymmetric_relations"]),
            total=num_relations,
        ),
        "prop_transitive": _ratio(
            count=len(characteristics["transitive_relations"]),
            total=num_relations,
        ),
        "prop_inverseof": _ratio(
            count=len(inverseof_relations),
            total=num_relations,
        ),
        "prop_subpropertyof": _ratio(
            count=len(subrelations),
            total=num_relations,
        ),
        "prop_profiled_relations": _ratio(
            count=len(profiled_relations),
            total=num_relations,
        ),
        "relation_specificity": relation_specificity,
    }
