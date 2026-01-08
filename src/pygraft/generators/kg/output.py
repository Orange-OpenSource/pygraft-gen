#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#


"""KG output serialization and metadata generation."""

from __future__ import annotations

from collections.abc import Iterable
import json
import logging
from typing import TYPE_CHECKING

from rdflib import RDF, Graph as RDFGraph, Namespace, URIRef
from tqdm.auto import tqdm

from pygraft.types import build_kg_info as _build_kg_info

if TYPE_CHECKING:
    from pathlib import Path

    from pygraft.generators.kg.config import InstanceGeneratorConfig
    from pygraft.generators.kg.structures import (
        EntityTypingState,
        SchemaMetadata,
        TripleGenerationState,
    )
    from pygraft.generators.kg.types import (
        EntityIdSet,
        RelationId,
        Triple,
    )
    from pygraft.types import (
        KGInfoDict,
        KGStatistics,
        KGUserParameters,
    )

logger = logging.getLogger(__name__)


# ================================================================================================ #
# TRIPLE ITERATION                                                                                 #
# ================================================================================================ #


def iter_triples(triples: TripleGenerationState) -> Iterable[Triple]:
    """Yield all generated triples as (head_id, relation_id, tail_id).

    Lazily iterates without materializing all triples in memory.

    Args:
        triples: Triple generation state containing kg_pairs_by_rid.

    Yields:
        Triple tuples of integer IDs.

    Complexity:
        O(1) per triple yielded.
    """
    for relation_id, pairs in triples.kg_pairs_by_rid.items():
        for head_id, tail_id in pairs:
            yield head_id, relation_id, tail_id


# ================================================================================================ #
# URI CONVERSION                                                                                   #
# ================================================================================================ #


def to_uri(
    identifier: str,
    ontology_namespace: str,
    prefix2namespace: dict[str, str],
) -> URIRef:
    """Convert an internal identifier to a URIRef.

    Resolution rules:
        - "E123" -> ontology namespace + identifier
        - "prefix:Local" -> expanded via namespaces_info.json
        - ":Local" -> empty prefix namespace if configured
        - "prefix_Local" -> legacy underscore form
        - Unknown -> ontology namespace fallback

    Args:
        identifier: Internal string identifier.
        ontology_namespace: Base namespace for entities.
        prefix2namespace: Prefix to namespace URI mappings.

    Returns:
        Expanded URIRef.
    """
    if identifier.startswith("E"):
        return URIRef(ontology_namespace + identifier)

    if ":" in identifier:
        prefix, local = identifier.split(":", 1)
        base = prefix2namespace.get(prefix)
        if base:
            return URIRef(base + local)
        return URIRef(ontology_namespace + identifier)

    if "_" in identifier:
        prefix, local = identifier.split("_", 1)
        base = prefix2namespace.get(prefix)
        if base:
            return URIRef(base + local)

    return URIRef(ontology_namespace + identifier)


# ================================================================================================ #
# KG SERIALIZATION                                                                                 #
# ================================================================================================ #


def serialize_kg(
    *,
    config: InstanceGeneratorConfig,
    schema: SchemaMetadata,
    entities: EntityTypingState,
    triples: TripleGenerationState,
    ontology_namespace: str,
    ontology_prefix: str,
    prefix2namespace: dict[str, str],
    output_directory_path: Path,
) -> str:
    """Serialize the generated KG to disk as RDF.

    Writes instance triples and rdf:type assertions for specific classes.
    Binds namespace prefixes from namespaces_info.json. Drops the in-memory
    RDFLib graph after serialization to free memory.

    Args:
        config: Generator configuration.
        schema: Schema metadata with ID mappings.
        entities: Entity typing state.
        triples: Triple generation state.
        ontology_namespace: Base namespace URI.
        ontology_prefix: Ontology prefix string.
        prefix2namespace: Prefix to namespace mappings.
        output_directory_path: Path to output directory.

    Returns:
        Path to the serialized KG file.

    Raises:
        FileNotFoundError: If output directory doesn't exist.
        RuntimeError: If serialization fails.

    Complexity:
        O(n_t + n_e) where n_t is triple count and n_e is entity count.
    """
    graph: RDFGraph | None = RDFGraph()

    ontology_ns = Namespace(ontology_namespace)
    graph.bind("sc", ontology_ns)

    if ontology_prefix and ontology_prefix != "_empty_prefix":
        graph.bind(ontology_prefix, ontology_ns)

    empty_prefix_namespace = prefix2namespace.get("")
    if empty_prefix_namespace:
        graph.bind("", Namespace(empty_prefix_namespace))

    for prefix, ns in sorted(prefix2namespace.items()):
        if prefix in {"_empty_prefix", ""}:
            continue
        graph.bind(prefix, Namespace(ns))

    id2entity = [f"E{i + 1}" for i in range(len(entities.entities))]

    total_triples = sum(len(pairs) for pairs in triples.kg_pairs_by_rid.values())

    try:
        for head_id, relation_id, tail_id in tqdm(
            iter_triples(triples),
            total=total_triples,
            desc="Serializing instance triples",
            unit="triples",
            colour="red",
        ):
            h = id2entity[head_id]
            r = schema.id2rel[relation_id]
            t = id2entity[tail_id]
            graph.add(
                (
                    to_uri(h, ontology_namespace, prefix2namespace),
                    to_uri(r, ontology_namespace, prefix2namespace),
                    to_uri(t, ontology_namespace, prefix2namespace),
                )
            )

        # rdf:type assertions (specific types only)
        for entity_id in range(len(entities.entities)):
            specific = entities.ent2classes_specific[entity_id]
            if not specific:
                continue

            ent_uri = to_uri(id2entity[entity_id], ontology_namespace, prefix2namespace)
            for class_id in specific:
                graph.add(
                    (
                        ent_uri,
                        RDF.type,
                        to_uri(schema.id2class[class_id], ontology_namespace, prefix2namespace),
                    )
                )

        if config.rdf_format == "xml":
            kg_path = output_directory_path / "kg.rdf"
            graph.serialize(str(kg_path), format="xml")
        else:
            kg_path = output_directory_path / f"kg.{config.rdf_format}"
            graph.serialize(str(kg_path), format=config.rdf_format)

        logger.info("Serialized KG graph to: %s", kg_path.resolve())
        return str(kg_path)

    except FileNotFoundError:
        logger.exception("Failed to write KG to '%s'", output_directory_path)
        raise
    except Exception as exc:
        logger.exception("Unexpected error while serializing KG")
        msg = "Failed to serialize KG."
        raise RuntimeError(msg) from exc


# ================================================================================================ #
# KG INFO                                                                                          #
# ================================================================================================ #


def compute_avg_multityping(entities: EntityTypingState) -> float:
    """Calculate the average number of specific classes per typed entity.

    Args:
        entities: Entity typing state.

    Returns:
        Average number of specific classes per typed entity.
    """
    if not entities.typed_entities:
        return 0.0

    total_specific = 0
    for entity_id in entities.typed_entities:
        total_specific += len(entities.ent2classes_specific[entity_id])

    return float(total_specific / len(entities.typed_entities))


def build_kg_info(
    *,
    config: InstanceGeneratorConfig,
    entities: EntityTypingState,
    triples: TripleGenerationState,
    output_directory_path: Path,
    is_multityping_enabled: bool,
) -> KGInfoDict:
    """Assemble and persist a summary of the generated KG.

    Computes actual statistics (entity count, relation count, triple count,
    typing proportions) and writes kg_info.json to the output directory.

    Args:
        config: Generator configuration.
        entities: Entity typing state.
        triples: Triple generation state.
        output_directory_path: Path to output directory.
        is_multityping_enabled: Whether multityping was enabled.

    Returns:
        KGInfoDict containing user parameters and computed statistics.

    Complexity:
        O(n_t) to iterate all triples for statistics.
    """
    observed_entity_ids: EntityIdSet = set()
    observed_relation_ids: set[RelationId] = set()
    num_triples = 0

    for head_id, relation_id, tail_id in iter_triples(triples):
        num_triples += 1
        observed_entity_ids.add(head_id)
        observed_entity_ids.add(tail_id)
        observed_relation_ids.add(relation_id)

    typed_observed_ids: EntityIdSet = set()
    for entity_id in observed_entity_ids:
        if (
            entity_id < len(entities.ent2classes_specific)
            and entities.ent2classes_specific[entity_id]
        ):
            typed_observed_ids.add(entity_id)

    # --- user parameters ---
    user_parameters: KGUserParameters = {
        "num_entities": config.num_entities,
        "num_triples": config.num_triples,
        "enable_fast_generation": config.enable_fast_generation,
        "relation_usage_uniformity": config.relation_usage_uniformity,
        "prop_untyped_entities": config.prop_untyped_entities,
        "avg_specific_class_depth": config.avg_specific_class_depth,
        "multityping": is_multityping_enabled,
        "avg_types_per_entity": config.avg_types_per_entity,
        "check_kg_consistency": config.check_kg_consistency,
    }

    avg_multityping = compute_avg_multityping(entities)

    statistics: KGStatistics = {
        "num_entities": len(observed_entity_ids),
        "num_instantiated_relations": len(observed_relation_ids),
        "num_triples": num_triples,
        "prop_untyped_entities": round(
            1 - (len(typed_observed_ids) / max(1, len(observed_entity_ids))),
            2,
        ),
        "avg_specific_class_depth": float(entities.current_avg_depth_specific_class),
        "avg_types_per_entity": (round(avg_multityping, 2) if entities.typed_entities else 0.0),
    }

    kg_info: KGInfoDict = _build_kg_info(user_parameters=user_parameters, statistics=statistics)

    kg_info_path = (output_directory_path / "kg_info.json").resolve()
    with kg_info_path.open("w", encoding="utf8") as file:
        json.dump(kg_info, file, indent=4)

    logger.debug("Wrote kg_info to: %s", kg_info_path)
    return kg_info
