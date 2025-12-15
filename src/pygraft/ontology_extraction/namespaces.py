#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Namespace extraction utilities for RDF ontologies.

Summary:
    This module extracts namespace information and ontology-level metadata
    from RDF ontologies, assuming the ontology is already loaded into an
    rdflib.Graph.

    Design rules:
    * This module NEVER reads files and NEVER parses RDF.
      - Parsing and format policies (for example banning .nt) are enforced
        upstream (for example in pipeline.load_ontology_graph).
    * The empty prefix (":" in Turtle, default xmlns in RDF/XML) is
      normalized to the JSON-safe label "_empty_prefix".
    * Ontology-level prefix and namespace are inferred from:
        - VANN preferredNamespacePrefix / preferredNamespaceUri, or
        - owl:Ontology declarations, or
        - declared prefixes as fallbacks.

    The goal is to keep this module purely graph-based: callers pass a Graph
    and receive a JSON-ready dict with structured metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypedDict

from rdflib import Literal, URIRef
from rdflib.namespace import OWL, RDF, split_uri

if TYPE_CHECKING:
    from rdflib import Graph

# ================================================================================================= #
# Namespace types and result dicts                                                                  #
# ================================================================================================= #


class OntologyMetadataDict(TypedDict):
    """Ontology-level metadata derived from an RDF graph.

    Attributes:
        prefix:
            Logical ontology prefix. Typically the value of the VANN
            preferredNamespacePrefix annotation (for example "faro"), or
            "_empty_prefix" when the ontology relies on the empty prefix.
            May be None when no reasonable candidate can be inferred.
        namespace:
            Ontology namespace IRI. Typically obtained from VANN
            preferredNamespaceUri, an owl:Ontology subject IRI, or the
            empty prefix namespace as a last resort. May be None when no
            such value can be determined reliably.
    """

    prefix: str | None
    namespace: str | None


class PrefixNamespaceMap(dict[str, str]):
    """Mapping from prefix label to namespace IRI.

    Notes:
        * Keys are prefix labels (for example "rdf", "xsd",
          "_empty_prefix").
        * Values are namespace IRIs as plain strings.
    """


class NamespaceInfoDict(TypedDict):
    """Top-level namespace extraction result.

    Attributes:
        ontology:
            Ontology-level metadata, including "prefix" and "namespace"
            fields as defined by OntologyMetadataDict.
        prefixes:
            Mapping from prefix labels to namespace IRIs. Includes the
            special "_empty_prefix" key when the ontology declares an
            unnamed default prefix.
        no_prefixes:
            Sorted list of namespace IRIs that are used in the graph but
            have no declared prefix (neither named nor empty).
    """

    ontology: OntologyMetadataDict
    prefixes: PrefixNamespaceMap
    no_prefixes: list[str]


# ================================================================================================= #
# Constants                                                                                         #
# ================================================================================================= #

#: JSON-safe label for the empty prefix (":" in Turtle, default xmlns in RDF/XML).
_EMPTY_PREFIX_LABEL: Final[str] = "_empty_prefix"

#: Prefix label for unknown namespaces when normalizing IRIs.
_MISSING_PREFIX: Final[str] = "_missing"

#: VANN term IRIs for preferredNamespacePrefix / preferredNamespaceUri.
VANN_PREF_PFX: Final[URIRef] = URIRef("http://purl.org/vocab/vann/preferredNamespacePrefix")
VANN_PREF_URI: Final[URIRef] = URIRef("http://purl.org/vocab/vann/preferredNamespaceUri")


# ================================================================================================= #
# Public API                                                                                        #
# ================================================================================================= #


def build_namespaces_dict(graph: Graph) -> NamespaceInfoDict:
    """Build a JSON-ready namespace description from an ontology graph.

    The input graph must already be parsed (for example by
    pipeline.load_ontology_graph). This function performs no file I/O and
    no RDF parsing.

    Args:
        graph:
            Parsed ontology graph.

    Returns:
        NamespaceInfoDict describing:
            * ontology-level prefix and namespace,
            * declared prefixes,
            * namespaces used without any declared prefix.
    """
    prefixes: PrefixNamespaceMap = _collect_namespaces_from_graph(graph=graph)

    used_namespaces: set[str] = _collect_used_namespaces_from_graph(graph=graph)

    declared_namespace_set: set[str] = set(prefixes.values())

    no_prefixes: list[str] = sorted(
        namespace_iri
        for namespace_iri in used_namespaces
        if namespace_iri not in declared_namespace_set
    )

    ontology_metadata: OntologyMetadataDict = _infer_ontology_metadata(
        graph=graph,
        prefixes=prefixes,
    )

    return NamespaceInfoDict(
        ontology=ontology_metadata,
        prefixes=prefixes,
        no_prefixes=no_prefixes,
    )


def iri_to_prefixed_name(
    iri: str,
    namespaces: NamespaceInfoDict,
) -> str:
    """Convert an IRI into a "prefix:LocalName" identifier.

    This is the shared, canonical IRI normalization utility for ontology
    extraction code. It uses the namespace metadata produced by
    build_namespaces_dict.

    Rules:
        * If split_uri fails (no valid "#" or "/" separator), the entire IRI
          is treated as a local name and returned as "_missing:<iri>".
        * If the IRI namespace matches a declared prefix namespace, return:
            - ":LocalName" when the prefix is "_empty_prefix"
            - "<prefix>:LocalName" otherwise
        * If no namespace match exists, return "_missing:LocalName".

    Args:
        iri:
            Full IRI string for a resource.
        namespaces:
            Namespace extraction result dict (ontology metadata + prefixes).

    Returns:
        A normalized identifier in the form "<prefix>:<local>", ":<local>",
        or "_missing:<local>" when no namespace match can be established.
    """
    try:
        ns, local_name = split_uri(iri)
    except ValueError:
        return f"{_MISSING_PREFIX}:{iri}"

    # Invert prefix->namespace into namespace->prefix for O(1) lookup.
    namespace_to_prefix: dict[str, str] = {
        namespace_iri: prefix for prefix, namespace_iri in namespaces["prefixes"].items()
    }

    ontology_prefix = namespaces["ontology"]["prefix"]
    ontology_ns = namespaces["ontology"]["namespace"]
    if ontology_prefix is not None and ontology_ns is not None:
        namespace_to_prefix.setdefault(ontology_ns, ontology_prefix)

    prefix = namespace_to_prefix.get(ns)
    if prefix is None:
        return f"{_MISSING_PREFIX}:{local_name}"

    if prefix == _EMPTY_PREFIX_LABEL:
        return f":{local_name}"

    return f"{prefix}:{local_name}"


# ================================================================================================= #
# Private helpers: namespace collection and normalization                                           #
# ================================================================================================= #


def _collect_namespaces_from_graph(graph: Graph) -> PrefixNamespaceMap:
    """Collect declared prefixes from a graph.

    This function inspects the namespace manager attached to the graph and
    returns a mapping from normalized prefix label to namespace IRI for all
    declared prefixes. It is independent of the original RDF serialization.

    Turtle (.ttl):
        * Captures prefixes declared with @prefix, including the empty
          prefix ":" which becomes "_empty_prefix".

    RDF/XML (.rdf, .owl):
        * Captures prefixes declared with xmlns:foo="..." as "foo".
        * Captures the default xmlns="..." as "_empty_prefix".

    This function does not compute any undeclared namespaces; those are
    handled in build_namespaces_dict, which has access to the full graph.

    Args:
        graph:
            Parsed rdflib.Graph that contains the ontology.

    Returns:
        PrefixNamespaceMap where normalized prefix labels (for example
        "rdf", "foaf", "_empty_prefix") map to namespace IRI strings.
    """
    declared_prefix_to_namespace: dict[str, str] = {}

    # rdflib already normalizes prefixes from different serializations; we
    # only need to adapt them for JSON consumption.
    for raw_prefix, namespace in graph.namespaces():
        normalized_prefix = _normalize_prefix_for_json(raw_prefix)
        if normalized_prefix is None:
            continue

        declared_prefix_to_namespace[normalized_prefix] = str(namespace)

    result: PrefixNamespaceMap = PrefixNamespaceMap()

    # Sort prefixes to keep output stable and predictable.
    for prefix in sorted(declared_prefix_to_namespace):
        result[prefix] = declared_prefix_to_namespace[prefix]

    return result


def _normalize_prefix_for_json(raw_prefix: str | None) -> str | None:
    """Normalize an rdflib prefix into a JSON-safe label.

    RDF serializations expose namespaces slightly differently, but rdflib
    normalizes them via Graph.namespaces():

    Turtle (.ttl):
        * "@prefix foo: <...> ." becomes prefix "foo".
        * "@prefix : <...> ." (empty prefix) becomes prefix "".
        * "@base <...> ." is used only to resolve relative IRIs and does
          not appear in Graph.namespaces().

    RDF/XML (.rdf, .owl):
        * "xmlns:foo='...'" becomes prefix "foo".
        * "xmlns='...'" (default namespace) becomes prefix "".
        * "xml:base='...'" affects relative IRIs but is not exposed as a
          prefix.

    N-Triples (.nt):
        * The format does not support prefixes or base declarations.
          Graph.namespaces() is usually empty, and there is nothing
          meaningful to expose. Any .nt-specific policy is enforced
          upstream; this function just normalizes what rdflib gives it.

    Normalization rules:
        * None:
            Dropped entirely (no JSON key for None).
        * "":
            Represents the unnamed default prefix. We expose it as
            "_empty_prefix" so downstream code can special-case it.
        * Any other string:
            Kept as-is (for example "foaf", "rdf", "xsd").

    Args:
        raw_prefix:
            Prefix label as returned by Graph.namespaces().

    Returns:
        Normalized prefix string suitable for use as a JSON key, or None
        when the prefix should be ignored.
    """
    if raw_prefix is None:
        return None

    if raw_prefix == "":
        return _EMPTY_PREFIX_LABEL

    return raw_prefix


def _extract_namespace_from_iri(iri: str) -> str:
    """Extract the namespace part from a full IRI string.

    The namespace ends immediately after the last "#" or "/", and the
    local name starts after that character.

    Examples:
        "http://example.org/onto/Person"  -> "http://example.org/onto/"
        "http://example.org/onto#Person" -> "http://example.org/onto#"

    If neither "#" nor "/" is present, the entire IRI is returned as its
    own "namespace".
    """
    hash_index = iri.rfind("#")
    if hash_index != -1:
        return iri[: hash_index + 1]

    slash_index = iri.rfind("/")
    if slash_index != -1:
        return iri[: slash_index + 1]

    return iri


def _collect_used_namespaces_from_graph(graph: Graph) -> set[str]:
    """Collect all namespace IRIs actually used in the ontology graph.

    This function walks over all triples and inspects any URIRef found in
    subject, predicate, or object position. For each IRI, it derives a
    namespace using _extract_namespace_from_iri and accumulates them in a
    set.

    The result represents the full set of namespaces that appear in the
    ontology, regardless of whether they have an associated declared
    prefix.
    """
    used_namespaces: set[str] = set()

    for subject, predicate, obj in graph:
        for node in (subject, predicate, obj):
            if isinstance(node, URIRef):
                iri = str(node)
                namespace = _extract_namespace_from_iri(iri)
                used_namespaces.add(namespace)

    return used_namespaces


# ================================================================================================= #
# Private helpers: ontology metadata inference                                                      #
# ================================================================================================= #


def _infer_ontology_metadata(
    graph: Graph,
    prefixes: PrefixNamespaceMap,
) -> OntologyMetadataDict:
    """Infer ontology-level prefix and namespace from graph and prefixes.

    The inference is intentionally simple and deterministic.

    Namespace resolution (ontology_namespace):
        1. Try VANN preferredNamespaceUri.
        2. Otherwise, use the IRI of the first subject typed as owl:Ontology.
        3. If neither is available, raise a ValueError and ask the user to
           provide VANN metadata or an owl:Ontology declaration.

    Prefix resolution (ontology_prefix):
        1. Try VANN preferredNamespacePrefix.
        2. If the namespace is known but the prefix is not, require an exact
           match in the declared prefixes. If no exact match exists, or if
           multiple prefixes map to the same ontology namespace, raise a
           ValueError and ask the user to declare VANN metadata or a prefix
           for the ontology namespace.

    Args:
        graph:
            Parsed rdflib.Graph for the ontology.
        prefixes:
            PrefixNamespaceMap of declared prefixes.

    Returns:
        OntologyMetadataDict containing inferred "prefix" and "namespace"
        values. The namespace is always non-None on success; the prefix may
        still be None if VANN is missing and no exact prefix match exists.

    Raises:
        ValueError:
            When an ontology namespace cannot be determined, or when a known
            ontology namespace has no unique exact prefix match.
    """
    ontology_namespace: str | None = None
    ontology_prefix: str | None = None

    # Build a clean prefix→namespace map from "prefixes".
    prefix_to_namespace: dict[str, str] = dict(prefixes.items())

    # ---------------------------- #
    # A) Resolve ontology_namespace #
    # ---------------------------- #

    # A1) VANN preferredNamespaceUri.
    for _, _, obj in graph.triples((None, VANN_PREF_URI, None)):
        if isinstance(obj, (Literal, URIRef)):
            ontology_namespace = str(obj)
            break

    # A2) owl:Ontology subject IRI.
    if ontology_namespace is None:
        for subject in graph.subjects(RDF.type, OWL.Ontology):
            if isinstance(subject, URIRef):
                ontology_namespace = str(subject)
                break

    # A3) If still unknown, the ontology is underspecified.
    if ontology_namespace is None:
        msg = (
            "Unable to infer ontology namespace: the graph does not contain any "
            "subject typed as owl:Ontology. This file does not declare itself as an "
            "OWL ontology. Please add an owl:Ontology declaration to the ontology. "
            "VANN metadata (preferredNamespaceUri) is optional but cannot replace "
            "the required owl:Ontology marker."
        )
        raise ValueError(msg)

    # ------------------------- #
    # B) Resolve ontology_prefix #
    # ------------------------- #

    # B1) VANN preferredNamespacePrefix.
    for _, _, obj in graph.triples((None, VANN_PREF_PFX, None)):
        if isinstance(obj, (Literal, URIRef)):
            ontology_prefix = str(obj)
            break

    # B2) Require an exact declared prefix match when namespace is known and
    #     no VANN prefix was found.
    if ontology_prefix is None:
        matching_prefixes: list[str] = [
            prefix
            for prefix, namespace in prefix_to_namespace.items()
            if namespace == ontology_namespace
        ]

        if not matching_prefixes:
            msg = (
                "Unable to infer ontology prefix: no declared prefix maps to the "
                f"ontology namespace '{ontology_namespace}'. Please add a VANN "
                "preferredNamespacePrefix / preferredNamespaceUri pair."
            )
            raise ValueError(msg)

        if len(matching_prefixes) > 1:
            joined = ", ".join(sorted(matching_prefixes))
            msg = (
                "Unable to infer ontology prefix: multiple declared prefixes map to "
                f"the ontology namespace '{ontology_namespace}': {joined}. "
                "Please disambiguate by providing a VANN preferredNamespacePrefix "
                "annotation or simplifying the prefix declarations."
            )
            raise ValueError(msg)

        # Exactly one match: deterministic choice.
        ontology_prefix = matching_prefixes[0]

    return OntologyMetadataDict(
        prefix=ontology_prefix,
        namespace=ontology_namespace,
    )


# ================================================================================================= #
# Local smoke test                                                                                  #
# ================================================================================================= #


if __name__ == "__main__":
    import json
    from pathlib import Path

    # namespaces.py → ontology_extraction → pygraft → src → ROOT
    project_root = Path(__file__).resolve().parents[3]

    # Lazy import to avoid circular imports at module import time.
    from pygraft.ontology_extraction.extraction import load_ontology_graph

    ontology_path = project_root / "ontologies" / "faro.ttl"
    ontology_graph = load_ontology_graph(ontology_path=ontology_path)

    namespace_info = build_namespaces_dict(graph=ontology_graph)
    print(json.dumps(namespace_info, indent=2))  # noqa: T201
