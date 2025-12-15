#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""PyGraft: A Python library for generating synthetic knowledge graphs and schemas.

This module provides the main API functions for creating templates, generating schemas,
and generating knowledge graphs based on user configuration files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
import logging
from pathlib import Path
import shutil

from pygraft.generators.classes import ClassGenerator, ClassGeneratorConfig
from pygraft.generators.kg import InstanceGenerator, InstanceGeneratorConfig
from pygraft.generators.relations import RelationGenerator, RelationGeneratorConfig
from pygraft.generators.schema import SchemaBuilder, SchemaBuilderConfig
from pygraft.paths import resolve_project_folder, slugify_project_name, OUTPUT_ROOT
from pygraft.utils.templates import create_config as _create_config
from pygraft.utils.config import load_config, validate_user_config
from pygraft.utils.reasoning import reasoner
from pygraft.ontology_extraction.extraction import ontology_extraction_pipeline

if TYPE_CHECKING:
    from pygraft.types import ClassInfoDict, RelationInfoDict, PyGraftConfigDict, KGInfoDict

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
# Load Config                                                                                      #
# ------------------------------------------------------------------------------------------------ #

def create_config(
    *,
    config_format: str = "json",
    output_dir: str | Path | None = None,
) -> Path:
    """Create a PyGraft configuration file.

    This function exposes configuration creation through the main PyGraft API.
    It delegates all logic to pygraft.utils.templates.create_config.

    Args:
        config_format:
            Configuration format. Must be one of: "json", "yml", "yaml".
            Defaults to "json".
        output_dir:
            Optional destination directory. If None, the current working
            directory is used.

    Returns:
        Path to the created configuration file.
    """
    output_dir_path = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    return _create_config(
        config_format=config_format,
        output_dir=output_dir_path,
    )


# ------------------------------------------------------------------------------------------------ #
# Generate Schema                                                                                  #
# ------------------------------------------------------------------------------------------------ #

def generate_schema(config_path: str, *, output_root: str | Path | None = None) -> tuple[Path, bool]:
    """Generate a schema based on the user's configuration file.

    This function is the high-level entry point for schema generation, used by
    both the CLI and Python callers.

    Args:
        config_path:
            Path to the user's configuration file (JSON or YAML).
        output_root:
            Optional base directory where PyGraft outputs will be written. When
            None, the default "./OUTPUT_ROOT" under the current working
            directory is used. This parameter is intended for Python API usage;
            the CLI always relies on the default.

    Returns:
        A tuple containing:
            - Path to the serialized schema file on disk.
            - True if the schema is consistent, False otherwise.
    """
    config_file = Path(config_path).expanduser().resolve()
    config: PyGraftConfigDict = load_config(str(config_file))

    validate_user_config(config, target="schema")
    logger.info("[Schema Generation] started")

    general_cfg = config["general"]
    classes_cfg = config["classes"]
    relations_cfg = config["relations"]

    base_root = Path(output_root).expanduser().resolve() if output_root is not None else None

    # Resolve and initialize the output folder for this schema run.
    general_cfg["project_name"] = resolve_project_folder(
        general_cfg["project_name"],
        mode="schema",
        output_root=base_root,
    )

    class_config = ClassGeneratorConfig(
        # General
        rng_seed=general_cfg["rng_seed"],

        # Classes
        num_classes=classes_cfg["num_classes"],
        max_hierarchy_depth=classes_cfg["max_hierarchy_depth"],
        avg_class_depth=classes_cfg["avg_class_depth"],
        avg_children_per_parent=classes_cfg["avg_children_per_parent"],
        avg_disjointness=classes_cfg["avg_disjointness"],
    )
    class_generator = ClassGenerator(config=class_config)
    class_info: ClassInfoDict = class_generator.generate_class_schema()

    relation_config = RelationGeneratorConfig(
        # General
        rng_seed=general_cfg["rng_seed"],

        # Relations
        num_relations=relations_cfg["num_relations"],
        relation_specificity=relations_cfg["relation_specificity"],
        prop_profiled_relations=relations_cfg["prop_profiled_relations"],
        profile_side=relations_cfg["profile_side"],

        prop_symmetric_relations=relations_cfg["prop_symmetric_relations"],
        prop_inverse_relations=relations_cfg["prop_inverse_relations"],
        prop_transitive_relations=relations_cfg["prop_transitive_relations"],
        prop_asymmetric_relations=relations_cfg["prop_asymmetric_relations"],
        prop_reflexive_relations=relations_cfg["prop_reflexive_relations"],
        prop_irreflexive_relations=relations_cfg["prop_irreflexive_relations"],
        prop_functional_relations=relations_cfg["prop_functional_relations"],
        prop_inverse_functional_relations=relations_cfg["prop_inverse_functional_relations"],
        prop_subproperties=relations_cfg["prop_subproperties"],
    )

    relation_generator = RelationGenerator(
        config=relation_config,
        class_info=class_info,
    )
    relation_info: RelationInfoDict = relation_generator.generate_relation_schema()

    schema_builder_config = SchemaBuilderConfig(
        folder_name=general_cfg["project_name"],
        rdf_format=general_cfg["rdf_format"],
        output_root=base_root,
    )

    schema_builder = SchemaBuilder(
        config=schema_builder_config,
        class_info=class_info,
        relation_info=relation_info,
    )
    schema_file = schema_builder.build_schema()

    # --- HermiT reasoning for the schema ---
    is_consistent = reasoner(schema_file=schema_file)

    logger.info("[Schema Generation] finished")
    return schema_file, is_consistent


# ------------------------------------------------------------------------------------------------ #
# Ontology Extraction                                                                              #
# ------------------------------------------------------------------------------------------------ #
def extract_ontology(
    ontology_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Extract ontology metadata and write PyGraft JSON artefacts."""
    ontology_file = Path(ontology_path).expanduser().resolve()

    base_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (Path.cwd() / OUTPUT_ROOT).resolve()
    )

    # Write extraction artefacts into the same project folder layout used by schema/KG runs.
    output_dir = base_root / slugify_project_name(ontology_file.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[Ontology Extraction] started")
    logger.info("using ontology file at %s", ontology_file)

    namespaces_info, class_info, relation_info = ontology_extraction_pipeline(
        ontology_path=ontology_file,
    )

    namespaces_path = output_dir / "namespaces_info.json"
    class_info_path = output_dir / "class_info.json"
    relation_info_path = output_dir / "relation_info.json"

    namespaces_path.write_text(json.dumps(namespaces_info, indent=4), encoding="utf-8")
    class_info_path.write_text(json.dumps(class_info, indent=4), encoding="utf-8")
    relation_info_path.write_text(json.dumps(relation_info, indent=4), encoding="utf-8")

    logger.info("serialized namespaces_info.json to %s", namespaces_path)
    logger.info("serialized class_info.json to %s", class_info_path)
    logger.info("serialized relation_info.json to %s", relation_info_path)

    # --- Copy ontology file as schema.{format} ---
    source_ext = ontology_file.suffix.lower()
    schema_ext = ".ttl" if source_ext == ".ttl" else ".rdf"
    schema_path = output_dir / f"schema{schema_ext}"

    shutil.copyfile(ontology_file, schema_path)

    logger.info(
        "copied ontology file\n"
        "  from %s\n"
        "  to   %s",
        ontology_file,
        schema_path,
    )

    logger.info("[Ontology Extraction] finished")

    return namespaces_path, class_info_path, relation_info_path


# ------------------------------------------------------------------------------------------------ #
# KG Generation                                                                                    #
# ------------------------------------------------------------------------------------------------ #

def generate_kg(
    config_path: str,
    *,
    output_root: str | Path | None = None,
    explain_inconsistency: bool = False,
    explanation_sink: list[str] | None = None,
) -> tuple[KGInfoDict, str, bool | None]:
    """Generate a knowledge graph based on the user's configuration file.

    This function is the high-level entry point for KG generation. It assumes
    that a compatible schema has already been generated for the same project.

    Args:
        config_path:
            Path to the user's configuration file (JSON or YAML).
        output_root:
            Optional base directory where PyGraft outputs will be written. When
            None, the default "./pygraft_output" under the current working
            directory is used. This parameter is intended for Python API usage;
            the CLI always relies on the default.
        explain_inconsistency:
            When True and KG consistency checking is enabled, run an additional
            Pellet explain pass if the KG is found to be inconsistent.
        explanation_sink:
            Optional list used to collect a human-readable Pellet explanation when
            inconsistency is detected (used by the CLI for clean output).

    Returns:
        A tuple containing:
            - kg_info: Dictionary with KG statistics and parameters.
            - kg_file: Path to the serialized KG file.
            - is_consistent: True if HermiT reports the KG as consistent,
              False if inconsistent, and None if KG consistency checking
              was disabled in the configuration.
    """
    config_file = Path(config_path).expanduser().resolve()
    config: PyGraftConfigDict = load_config(str(config_file))

    validate_user_config(config, target="kg")
    logger.info("[KG Generation] started")

    general_cfg = config["general"]
    kg_cfg = config["kg"]

    base_root = Path(output_root).expanduser().resolve() if output_root is not None else None

    # Resolve the project folder for this KG run (reusing an existing schema).
    general_cfg["project_name"] = resolve_project_folder(
        general_cfg["project_name"],
        mode="kg",
        output_root=base_root,
    )

    instance_config = InstanceGeneratorConfig(
        # General
        project_name=general_cfg["project_name"],
        rdf_format=general_cfg["rdf_format"],
        rng_seed=general_cfg["rng_seed"],

        # KG
        num_entities=kg_cfg["num_entities"],
        num_triples=kg_cfg["num_triples"],

        enable_fast_generation=kg_cfg["enable_fast_generation"],
        enable_inference_oversampling=kg_cfg["enable_inference_oversampling"],

        relation_usage_uniformity=kg_cfg["relation_usage_uniformity"],
        prop_untyped_entities=kg_cfg["prop_untyped_entities"],

        avg_specific_class_depth=kg_cfg["avg_specific_class_depth"],

        multityping=kg_cfg["multityping"],
        avg_types_per_entity=kg_cfg["avg_types_per_entity"],

        check_kg_consistency=kg_cfg["check_kg_consistency"],
        # Output directory
        output_root=base_root,
    )

    instance_generator = InstanceGenerator(config=instance_config)
    kg_info, kg_file = instance_generator.generate_kg()

    kg_consistent: bool | None = None

    # --- Optional HermiT reasoning for the KG ---
    if kg_cfg["check_kg_consistency"]:
        kg_path = Path(kg_file)

        # Infer schema config_path from the KG location + format.
        if kg_path.suffix == ".rdf":
            schema_file = kg_path.with_name("schema.rdf")
        else:
            schema_file = kg_path.with_name(f"schema{kg_path.suffix}")

        kg_consistent = reasoner(
            schema_file=schema_file,
            kg_file=kg_file,
            explain_inconsistency=explain_inconsistency,
            explanation_sink=explanation_sink,
        )
    else:
        logger.info("(HermiT) Skipped KG reasoning step")

    logger.info("[KG Generation] finished")
    return kg_info, kg_file, kg_consistent
