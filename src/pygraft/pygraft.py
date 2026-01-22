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

This module provides the main API functions for creating configuration templates,
generating synthetic schemas, extracting ontology metadata, and generating
knowledge graphs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from pygraft.generators.classes import ClassGenerator, ClassGeneratorConfig
from pygraft.generators.kg.config import InstanceGeneratorConfig
from pygraft.generators.kg.kg import InstanceGenerator
from pygraft.generators.relations import RelationGenerator, RelationGeneratorConfig
from pygraft.generators.schema import SchemaBuilder, SchemaBuilderConfig
from pygraft.ontology_extraction.extraction import ontology_extraction_pipeline
from pygraft.paths import OUTPUT_ROOT, resolve_project_folder, slugify_project_name
from pygraft.utils.config import load_config, validate_user_config
from pygraft.utils.reasoning import reasoner_hermit, reasoner_pellet
from pygraft.utils.templates import create_config as _create_config

if TYPE_CHECKING:
    from pygraft.types import ClassInfoDict, KGInfoDict, PyGraftConfigDict, RelationInfoDict

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
# Load Config                                                                                      #
# ------------------------------------------------------------------------------------------------ #


def create_config(
    *,
    config_format: str = "json",
    output_dir: str | Path | None = None,
) -> Path:
    """Create a new PyGraft configuration file with default values.

    Generates a template configuration file that can be customized for
    schema and knowledge graph generation.

    Args:
        config_format: Output format for the configuration file.
            Must be one of `"json"`, `"yml"`, or `"yaml"`.
        output_dir: Directory where the configuration file will be written.
            If `None`, uses the current working directory.

    Returns:
        Path to the created configuration file.

    Raises:
        ValueError: If `config_format` is not a supported format.

    Example:
        ```python
        from pygraft import create_config

        # Create JSON config in current directory
        config_path = create_config()

        # Create YAML config in specific directory
        config_path = create_config(config_format="yaml", output_dir="./configs")
        ```
    """
    output_dir_path = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    return _create_config(
        config_format=config_format,
        output_dir=output_dir_path,
    )


# ------------------------------------------------------------------------------------------------ #
# Generate Schema                                                                                  #
# ------------------------------------------------------------------------------------------------ #


def generate_schema(
    config_path: str,
    *,
    output_root: str | Path | None = None,
) -> tuple[Path, bool]:
    """Generate a synthetic OWL schema from configuration parameters.

    Creates an ontology with classes, relations, and OWL constraints based on
    the statistical parameters defined in the configuration file. The generated
    schema is validated for consistency using the HermiT reasoner.

    This is typically the first step in the fully synthetic workflow,
    followed by `generate_kg()`.

    Args:
        config_path: Path to a PyGraft configuration file (JSON or YAML).
        output_root: Base directory for output files. If `None`, uses
            `./output_pygraft/` in the current working directory.

    Returns:
        A tuple containing:

        - **schema_path**: Path to the generated schema file (`.ttl`, `.rdf`, or `.nt`).
        - **is_consistent**: `True` if HermiT confirms the schema is logically consistent.

    Raises:
        ValueError: If the configuration file is invalid or contains
            incompatible parameter combinations.
        FileNotFoundError: If the configuration file does not exist.

    Example:
        ```python
        from pygraft import generate_schema

        schema_path, is_consistent = generate_schema("pygraft.config.json")

        if is_consistent:
            print(f"Schema generated at: {schema_path}")
        else:
            print("Warning: Schema has logical inconsistencies")
        ```
    """
    config_file = Path(config_path).expanduser().resolve()
    config: PyGraftConfigDict = load_config(str(config_file))

    validate_user_config(config, target="schema")
    logger.info("[Schema Generation] started")

    general_cfg = config["general"]
    classes_cfg = config["schema"]["classes"]
    relations_cfg = config["schema"]["relations"]

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
        # Class Hierarchy
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
        # Relation Core
        num_relations=relations_cfg["num_relations"],
        relation_specificity=relations_cfg["relation_specificity"],
        prop_profiled_relations=relations_cfg["prop_profiled_relations"],
        profile_side=relations_cfg["profile_side"],
        # OWL Characteristics
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
    is_consistent = reasoner_hermit(schema_file=schema_file)

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
    """Extract metadata from an existing ontology into PyGraft JSON artefacts.

    Analyzes the structure of an OWL ontology and extracts statistics about
    its classes and relations. The extracted metadata is written as JSON files
    that can be used to configure KG generation.

    This is the first step in the ontology-based workflow, followed by
    `generate_kg()`.

    Args:
        ontology_path: Path to the ontology file (`.ttl`, `.rdf`, `.owl`, or `.xml`).
        output_root: Base directory for output files. If `None`, uses
            `./output_pygraft/` in the current working directory.

    Returns:
        A tuple of paths to the generated JSON artefacts:

        - **namespaces_path**: Namespace prefix mappings (`namespaces_info.json`).
        - **class_info_path**: Class hierarchy statistics (`class_info.json`).
        - **relation_info_path**: Relation statistics (`relation_info.json`).

    Raises:
        FileNotFoundError: If the ontology file does not exist.

    Note:
        The original ontology is also copied to the output directory as
        `schema.ttl` or `schema.rdf` for use with `generate_kg()`.

    Example:
        ```python
        from pygraft import extract_ontology

        namespaces, classes, relations = extract_ontology("./ontologies/pizza.ttl")

        print(f"Extracted class info to: {classes}")
        # Now run: pygraft kg pygraft.config.json
        ```
    """
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
        "copied ontology file\n  from %s\n  to   %s",
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
) -> tuple[KGInfoDict, str, bool | None]:
    """Generate a Knowledge Graph from an existing schema.

    Creates entity instances and relation triples based on the configuration
    parameters. Requires a schema to already exist in the project folder,
    either from `generate_schema()` (synthetic workflow) or `extract_ontology()`
    (ontology-based workflow).

    Args:
        config_path: Path to a PyGraft configuration file (JSON or YAML).
        output_root: Base directory for output files. If `None`, uses
            `./output_pygraft/` in the current working directory.

    Returns:
        A tuple containing:

        - **kg_info**: Dictionary with generation statistics (entity counts,
            triple counts, type distributions).
        - **kg_path**: Path to the generated KG file (`.ttl`, `.rdf`, or `.nt`).
        - **is_consistent**: Consistency check result:
            - `True` if HermiT confirms the KG is logically consistent.
            - `False` if HermiT detects inconsistencies.
            - `None` if consistency checking was disabled in the configuration.

    Raises:
        ValueError: If the configuration is invalid.
        FileNotFoundError: If the configuration file or required schema does not exist.

    Example:
        ```python
        from pygraft import generate_kg

        kg_info, kg_path, is_consistent = generate_kg("pygraft.config.json")

        print(f"Generated {kg_info['num_triples']} triples")
        print(f"KG written to: {kg_path}")

        if is_consistent is False:
            print("Warning: KG has inconsistencies. Run 'pygraft explain' for details.")
        ```
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
        # KG Size
        num_entities=kg_cfg["num_entities"],
        num_triples=kg_cfg["num_triples"],
        # Generation Mode
        enable_fast_generation=kg_cfg["enable_fast_generation"],
        # Entity & Relation Distribution
        relation_usage_uniformity=kg_cfg["relation_usage_uniformity"],
        prop_untyped_entities=kg_cfg["prop_untyped_entities"],
        avg_specific_class_depth=kg_cfg["avg_specific_class_depth"],
        # Multityping
        multityping=kg_cfg["multityping"],
        avg_types_per_entity=kg_cfg["avg_types_per_entity"],
        # Consistency Checking
        check_kg_consistency=kg_cfg["check_kg_consistency"],
        # Output
        output_root=base_root,
    )

    instance_generator = InstanceGenerator(config=instance_config)
    kg_info, kg_file = instance_generator.generate_kg()

    kg_consistent: bool | None = None

    # --- Optional HermiT reasoning for the KG ---
    if kg_cfg["check_kg_consistency"]:
        kg_path = Path(kg_file)

        # Infer schema path from the KG location + format.
        if kg_path.suffix == ".rdf":
            schema_file = kg_path.with_name("schema.rdf")
        else:
            schema_file = kg_path.with_name(f"schema{kg_path.suffix}")

        kg_consistent = reasoner_hermit(
            schema_file=schema_file,
            kg_file=kg_file,
        )
    else:
        logger.info("(HermiT) Skipped KG reasoning step")

    logger.info("[KG Generation] finished")
    return kg_info, kg_file, kg_consistent


# ------------------------------------------------------------------------------------------------ #
# Explain KG Inconsistencies                                                                       #
# ------------------------------------------------------------------------------------------------ #


def explain_kg(
    kg_path: str | Path,
    *,
    reasoner: str = "pellet",
) -> tuple[bool, str | None]:
    """Analyze a Knowledge Graph for logical inconsistencies.

    Runs OWL reasoners to check consistency and provide detailed explanations
    when inconsistencies are found. The schema is automatically inferred from
    the same directory as the KG file.

    Args:
        kg_path: Path to the knowledge graph file (`.ttl`, `.rdf`, or `.nt`).
        reasoner: Which reasoner(s) to use:

            - `"hermit"`: Fast consistency check only (no explanation).
            - `"pellet"`: Detailed explanations (default).
            - `"both"`: HermiT first, then Pellet if inconsistent.

    Returns:
        A tuple containing:

        - **is_consistent**: `True` if consistent, `False` if inconsistent.
        - **explanation**: Human-readable explanation of inconsistencies,
            or `None` if consistent or using HermiT-only mode.

    Raises:
        FileNotFoundError: If the KG file or inferred schema file does not exist.
        ValueError: If the KG file has an unsupported extension or
            `reasoner` is not one of the valid options.

    Example:
        ```python
        from pygraft import explain_kg

        is_consistent, explanation = explain_kg("./output_pygraft/my-project/kg.ttl")

        if not is_consistent:
            print("Inconsistency detected:")
            print(explanation)
        ```
    """
    # Validate reasoner parameter
    valid_reasoners = {"hermit", "pellet", "both"}
    if reasoner not in valid_reasoners:
        msg = f"Invalid reasoner: {reasoner}. Must be one of: {valid_reasoners}"
        raise ValueError(msg)

    kg_file = Path(kg_path).expanduser().resolve()

    # Validate KG file exists
    if not kg_file.exists():
        msg = f"KG file not found: {kg_file}"
        raise FileNotFoundError(msg)

    # Validate KG file extension
    if kg_file.suffix.lower() not in {".ttl", ".rdf", ".nt"}:
        msg = f"KG file must have extension .ttl, .rdf, or .nt, got: {kg_file.suffix}"
        raise ValueError(msg)

    # Infer schema path from KG location
    if kg_file.suffix == ".rdf":
        schema_file = kg_file.with_name("schema.rdf")
    else:
        schema_file = kg_file.with_name(f"schema{kg_file.suffix}")

    # Validate schema file exists
    if not schema_file.exists():
        msg = f"Schema file not found at expected location: {schema_file}"
        raise FileNotFoundError(msg)

    logger.info("[KG Explanation] started")
    logger.info("Analyzing KG file at %s", kg_file)
    logger.info("Using schema file at %s", schema_file)
    logger.info("Reasoner mode: %s", reasoner)

    is_consistent: bool
    explanation: str | None = None

    if reasoner == "hermit":
        # HermiT only - fast consistency check, no explanation
        is_consistent = reasoner_hermit(
            schema_file=schema_file,
            kg_file=kg_file,
        )
        explanation = None

    elif reasoner == "pellet":
        # Pellet only - consistency + explanation
        is_consistent, explanation = reasoner_pellet(
            schema_file=schema_file,
            kg_file=kg_file,
        )

    elif reasoner == "both":
        # HermiT first, then Pellet if inconsistent
        logger.info("Running HermiT first for fast consistency check...")
        is_consistent = reasoner_hermit(
            schema_file=schema_file,
            kg_file=kg_file,
        )

        if is_consistent:
            # Consistent - skip Pellet (saves time)
            logger.info("HermiT reports consistent - skipping Pellet")
            explanation = None
        else:
            # Inconsistent - run Pellet for detailed explanation
            logger.info("HermiT reports inconsistent - running Pellet for explanation...")
            _, explanation = reasoner_pellet(
                schema_file=schema_file,
                kg_file=kg_file,
            )

    logger.info("[KG Explanation] finished")
    return is_consistent, explanation
