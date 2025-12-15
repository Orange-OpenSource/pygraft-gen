#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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

    Updates:
        - general.project_name
        - general.rdf_format
        - classes.* from class_info["statistics"]
        - relations.* from relation_info["statistics"]

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

    # --- Update general ---
    general = config_data.setdefault("general", {})
    general["project_name"] = ontology_file.stem
    general["rdf_format"] = "ttl" if ontology_file.suffix.lower() == ".ttl" else "rdf"

    # --- Update classes ---
    classes_cfg = config_data.setdefault("classes", {})
    if "num_classes" in class_stats:
        classes_cfg["num_classes"] = class_stats["num_classes"]
    if "hierarchy_depth" in class_stats:
        classes_cfg["max_hierarchy_depth"] = class_stats["hierarchy_depth"]
    if "avg_class_depth" in class_stats:
        classes_cfg["avg_class_depth"] = class_stats["avg_class_depth"]
    if "avg_children_per_parent" in class_stats:
        classes_cfg["avg_children_per_parent"] = class_stats["avg_children_per_parent"]
    if "avg_class_disjointness" in class_stats:
        classes_cfg["avg_disjointness"] = class_stats["avg_class_disjointness"]

    # --- Update relations ---
    relations_cfg = config_data.setdefault("relations", {})
    if "num_relations" in rel_stats:
        relations_cfg["num_relations"] = rel_stats["num_relations"]
    if "relation_specificity" in rel_stats:
        relations_cfg["relation_specificity"] = rel_stats["relation_specificity"]
    if "prop_profiled_relations" in rel_stats:
        relations_cfg["prop_profiled_relations"] = rel_stats["prop_profiled_relations"]

    if "prop_symmetric" in rel_stats:
        relations_cfg["prop_symmetric_relations"] = rel_stats["prop_symmetric"]
    if "prop_inverseof" in rel_stats:
        relations_cfg["prop_inverse_relations"] = rel_stats["prop_inverseof"]
    if "prop_transitive" in rel_stats:
        relations_cfg["prop_transitive_relations"] = rel_stats["prop_transitive"]
    if "prop_asymmetric" in rel_stats:
        relations_cfg["prop_asymmetric_relations"] = rel_stats["prop_asymmetric"]
    if "prop_reflexive" in rel_stats:
        relations_cfg["prop_reflexive_relations"] = rel_stats["prop_reflexive"]
    if "prop_irreflexive" in rel_stats:
        relations_cfg["prop_irreflexive_relations"] = rel_stats["prop_irreflexive"]
    if "prop_functional" in rel_stats:
        relations_cfg["prop_functional_relations"] = rel_stats["prop_functional"]
    if "prop_inversefunctional" in rel_stats:
        relations_cfg["prop_inverse_functional_relations"] = rel_stats["prop_inversefunctional"]
    if "prop_subpropertyof" in rel_stats:
        relations_cfg["prop_subproperties"] = rel_stats["prop_subpropertyof"]

    # Note: profile_side is not derivable from extraction statistics; leave user default.

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
