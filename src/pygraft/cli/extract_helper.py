#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""CLI helper for ontology extraction configuration.

This module handles the creation and updating of PyGraft configuration files
based on extracted ontology statistics. It is used exclusively by the CLI
extract command.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from pygraft.utils.templates import create_config as _create_config

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------ #
# Constants                                                                                        #
# ------------------------------------------------------------------------------------------------ #

PYGRAFT_CONFIG_JSON_FILENAME = "pygraft.config.json"
PYGRAFT_CONFIG_YML_FILENAME = "pygraft.config.yml"


# ------------------------------------------------------------------------------------------------ #
# API                                                                                              #
# ------------------------------------------------------------------------------------------------ #


def ensure_and_update_config_for_ontology(
    *,
    ontology_file: Path,
    class_info_path: Path,
    relation_info_path: Path,
) -> tuple[Path, bool]:
    """Ensure a PyGraft config file exists and update it using extracted ontology statistics.

    This function rebuilds the config with a clean structure containing only necessary keys.
    Extra keys added by users are removed, and missing essential keys are restored with
    defaults or extracted values. This applies to ALL sections: general, schema, and kg.

    Updates:
        - general.project_name (from ontology filename)
        - general.rdf_format (from ontology file extension)
        - general.rng_seed (preserved from existing config, or null)
        - schema.classes.* (from class_info["statistics"] or defaults)
        - schema.relations.* (from relation_info["statistics"] or defaults)
        - kg.* (values preserved from existing config, or defaults if missing)

    Args:
        ontology_file: Path to the source ontology file.
        class_info_path: Path to the extracted class_info.json.
        relation_info_path: Path to the extracted relation_info.json.

    Returns:
        Tuple of:
            - Path to the config file
            - True if the config was created, False if it already existed
    """
    ontology_file = ontology_file.resolve()

    cwd = Path.cwd()
    json_path = (cwd / PYGRAFT_CONFIG_JSON_FILENAME).resolve()
    yml_path = (cwd / PYGRAFT_CONFIG_YML_FILENAME).resolve()

    if json_path.exists():
        config_path = json_path
        config_format = "json"
        created = False
        logger.info("Reused configuration file at %s", config_path)
    elif yml_path.exists():
        config_path = yml_path
        config_format = "yaml"
        created = False
        logger.info("Reused configuration file at %s", config_path)
    else:
        config_path = _create_config(config_format="json", output_dir=cwd).resolve()
        config_format = "json"
        created = True
        logger.info("Created configuration file at %s", config_path)

    if config_format == "json":
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # --- Load extracted statistics (JSON artefacts) ---
    class_info = json.loads(Path(class_info_path).read_text(encoding="utf-8"))
    relation_info = json.loads(Path(relation_info_path).read_text(encoding="utf-8"))

    class_stats = class_info.get("statistics", {})
    rel_stats = relation_info.get("statistics", {})

    # --- Preserve the existing KG section (if any) ---
    kg_section = config_data.get("kg", {})

    # --- Build a clean config structure with only necessary keys ---
    # This ensures extra keys are removed and missing keys are added with defaults

    # General section (always rebuild)
    new_general = {
        "project_name": ontology_file.stem,
        "rdf_format": "ttl" if ontology_file.suffix.lower() == ".ttl" else "rdf",
        "rng_seed": config_data.get("general", {}).get("rng_seed", None),
    }

    # Classes section (rebuild with extracted stats or defaults)
    new_classes = {
        "num_classes": class_stats.get("num_classes", 50),
        "max_hierarchy_depth": class_stats.get("hierarchy_depth", 4),
        "avg_class_depth": class_stats.get("avg_class_depth", 2.5),
        "avg_children_per_parent": class_stats.get("avg_children_per_parent", 2.0),
        "avg_disjointness": class_stats.get("avg_class_disjointness", 0.3),
    }

    # Relations section (rebuild with extracted stats or defaults)
    # Preserve profile_side from existing config or use default
    existing_profile_side = (
        config_data.get("schema", {}).get("relations", {}).get("profile_side", "both")
    )

    new_relations = {
        "num_relations": rel_stats.get("num_relations", 50),
        "relation_specificity": rel_stats.get("relation_specificity", 2.5),
        "prop_profiled_relations": rel_stats.get("prop_profiled_relations", 0.9),
        "profile_side": existing_profile_side,
        "prop_symmetric_relations": rel_stats.get("prop_symmetric", 0.3),
        "prop_inverse_relations": rel_stats.get("prop_inverseof", 0.3),
        "prop_transitive_relations": rel_stats.get("prop_transitive", 0.1),
        "prop_asymmetric_relations": rel_stats.get("prop_asymmetric", 0.0),
        "prop_reflexive_relations": rel_stats.get("prop_reflexive", 0.3),
        "prop_irreflexive_relations": rel_stats.get("prop_irreflexive", 0.0),
        "prop_functional_relations": rel_stats.get("prop_functional", 0.0),
        "prop_inverse_functional_relations": rel_stats.get("prop_inversefunctional", 0.0),
        "prop_subproperties": rel_stats.get("prop_subpropertyof", 0.3),
    }

    # Rebuild KG section with only necessary keys
    # Preserve values from existing config if they exist, otherwise use defaults
    new_kg = {
        "num_entities": kg_section.get("num_entities", 3000),
        "num_triples": kg_section.get("num_triples", 30000),
        "enable_fast_generation": kg_section.get("enable_fast_generation", False),
        "relation_usage_uniformity": kg_section.get("relation_usage_uniformity", 0.9),
        "prop_untyped_entities": kg_section.get("prop_untyped_entities", 0.0),
        "avg_specific_class_depth": kg_section.get("avg_specific_class_depth", 2.0),
        "multityping": kg_section.get("multityping", False),
        "avg_types_per_entity": kg_section.get("avg_types_per_entity", 1.0),
        "check_kg_consistency": kg_section.get("check_kg_consistency", True),
    }

    # Build the clean config with only necessary sections and keys
    config_data = {
        "general": new_general,
        "schema": {
            "classes": new_classes,
            "relations": new_relations,
        },
        "kg": new_kg,
    }

    if config_format == "json":
        config_path.write_text(json.dumps(config_data, indent=4), encoding="utf-8")
    else:
        config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    logger.info(
        "Updated config file from ontology statistics\n"
        "  ontology: %s\n"
        "  config:   %s",
        ontology_file,
        config_path,
    )

    return config_path, created
