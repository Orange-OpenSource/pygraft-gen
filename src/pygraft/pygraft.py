"""PyGraft: A Python library for generating synthetic knowledge graphs and schemas.

This module provides the main API functions for creating templates, generating schemas,
and generating knowledge graphs based on user configuration files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging
from pathlib import Path

from pygraft.generators.classes import ClassGenerator, ClassGeneratorConfig
from pygraft.generators.kg import InstanceGenerator, InstanceGeneratorConfig
from pygraft.generators.relations import RelationGenerator, RelationGeneratorConfig
from pygraft.generators.schema import SchemaBuilder, SchemaBuilderConfig
from pygraft.paths import resolve_project_folder
from pygraft.utils.templates import copy_json_config_template, copy_yaml_config_template
from pygraft.utils.config import load_config, validate_user_config
from pygraft.utils.reasoning import reasoner

if TYPE_CHECKING:
    from pygraft.types import ClassInfoDict, RelationInfoDict, PyGraftConfigDict, KGInfoDict

logger = logging.getLogger(__name__)


class InvalidExtensionError(ValueError):
    """Raised when an invalid file extension is provided."""

    def __init__(self, extension: str) -> None:
        """Initialize the error with the invalid extension.

        Args:
            extension: The invalid extension that was provided.
        """
        super().__init__(
            f"Unknown extension file format: {extension}. "
            "Please enter one of the following: json, yaml, yml"
        )


# ------------------------------------------------------------------------------------------------ #
# Load Config                                                                                      #
# ------------------------------------------------------------------------------------------------ #

def create_template(extension: str = "yml") -> None:
    """Create a template file for the user to fill in.

    Args:
        extension: File extension of the template file. Defaults to "yml".

    Raises:
        InvalidExtensionError: If the extension is not one of: json, yaml, yml.
    """
    if extension == "json":
        copy_json_config_template()
    elif extension in {"yaml", "yml"}:
        copy_yaml_config_template()
    else:
        raise InvalidExtensionError(extension)


def create_json_template() -> None:
    """Create a json template file for the user to fill in."""
    copy_json_config_template()


def create_yaml_template() -> None:
    """Create a yaml template file for the user to fill in."""
    copy_yaml_config_template()


# ------------------------------------------------------------------------------------------------ #
# Generate Schema                                                                                  #
# ------------------------------------------------------------------------------------------------ #

def generate_schema(path: str, *, output_root: str | Path | None = None) -> Path:
    """Generate a schema based on the user's configuration file.

    This function is the high-level entry point for schema generation, used by
    both the CLI and Python callers.

    Args:
        path:
            Path to the user's configuration file (JSON or YAML).
        output_root:
            Optional base directory where PyGraft outputs will be written. When
            None, the default "./pygraft_output" under the current working
            directory is used. This parameter is intended for Python API usage;
            the CLI always relies on the default.

    Returns:
        Path to the serialized schema file on disk.
    """
    config: PyGraftConfigDict = load_config(path)

    validate_user_config(config, target="schema")
    logger.info("[Schema Generation] started")

    general_cfg = config["general"]
    classes_cfg = config["classes"]
    relations_cfg = config["relations"]

    base_root = Path(output_root) if output_root is not None else None

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
    reasoner(schema_file=schema_file)

    logger.info("[Schema Generation] finished")
    return schema_file


# ------------------------------------------------------------------------------------------------ #
# KG Generation                                                                                    #
# ------------------------------------------------------------------------------------------------ #

def generate_kg(path: str, *, output_root: str | Path | None = None) -> tuple[KGInfoDict, str]:
    """Generate a knowledge graph based on the user's configuration file.

    This function is the high-level entry point for KG generation. It assumes
    that a compatible schema has already been generated for the same project.

    Args:
        path:
            Path to the user's configuration file (JSON or YAML).
        output_root:
            Optional base directory where PyGraft outputs will be written. When
            None, the default "./pygraft_output" under the current working
            directory is used. This parameter is intended for Python API usage;
            the CLI always relies on the default.

    Returns:
        A tuple containing:
            - kg_info: Dictionary with KG statistics and parameters.
            - kg_file: Path to the serialized KG file.
    """
    config: PyGraftConfigDict = load_config(path)

    validate_user_config(config, target="kg")
    logger.info("[KG Generation] started")

    general_cfg = config["general"]
    kg_cfg = config["kg"]

    base_root = Path(output_root) if output_root is not None else None

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

    # --- Optional HermiT reasoning for the KG ---
    if kg_cfg["check_kg_consistency"]:
        kg_path = Path(kg_file)

        # Infer schema path from the KG location + format.
        if kg_path.suffix == ".rdf":
            schema_file = kg_path.with_name("schema.rdf")
        else:
            schema_file = kg_path.with_name(f"schema{kg_path.suffix}")

        reasoner(schema_file=schema_file, kg_file=kg_file)
    else:
        logger.info("(HermiT) Skipped KG reasoning step")

    logger.info("[KG Generation] finished")
    return kg_info, kg_file
