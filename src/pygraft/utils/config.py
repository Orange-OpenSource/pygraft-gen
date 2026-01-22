#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Configuration loading and validation for PyGraft.

This module is responsible for:

- Loading a PyGraft configuration from JSON or YAML files.
- Validating that the configuration:
  - Uses the expected keys and no extras.
  - Chooses a supported RDF serialization format.
  - Respects schema-level constraints (class + relation generation).
  - Respects KG-level constraints (instance generation and multityping).

Public API
----------
External callers are expected to use:

    - load_config
    - validate_user_config

All other helpers are internal and may change without notice.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

from pygraft.paths import slugify_project_name
from pygraft.types import (
    ClassGenConfigDict,
    GeneralConfigDict,
    KGGenConfigDict,
    PyGraftConfigDict as RuntimePyGraftConfigDict,
    RelationGenConfigDict,
    SchemaConfigDict,
)

if TYPE_CHECKING:
    from pygraft.types import PyGraftConfigDict

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------ #
# Constants                                                                                        #
# ------------------------------------------------------------------------------------------------ #

# Top-level keys: "general", "schema", "kg".
REQUIRED_CONFIG_KEYS: frozenset[str] = frozenset(RuntimePyGraftConfigDict.__annotations__.keys())

# Section-level required keys, derived from the TypedDicts.
# --- general
REQUIRED_GENERAL_KEYS: frozenset[str] = frozenset(GeneralConfigDict.__annotations__.keys())
# --- schema
REQUIRED_SCHEMA_KEYS: frozenset[str] = frozenset(SchemaConfigDict.__annotations__.keys())
REQUIRED_CLASS_KEYS: frozenset[str] = frozenset(ClassGenConfigDict.__annotations__.keys())
REQUIRED_RELATION_KEYS: frozenset[str] = frozenset(RelationGenConfigDict.__annotations__.keys())
# --- kg
REQUIRED_KG_KEYS: frozenset[str] = frozenset(KGGenConfigDict.__annotations__.keys())

# Target of validation: which parts of the pipeline will consume this config.
ValidationTarget = Literal["schema", "kg", "both"]

# ------------------------------------------------------------------------------------------------ #
# Public API                                                                                       #
# ------------------------------------------------------------------------------------------------ #


def load_config(path: str | Path) -> PyGraftConfigDict:
    """Load a configuration file in JSON or YAML format.

    This helper is the single entry point for loading user configuration files.
    It abstracts away the underlying serialization format and always returns a
    mapping conforming to PyGraftConfigDict.

    Args:
        path: Location of the configuration file. Must have suffix ".json",
            ".yaml", or ".yml".

    Returns:
        Parsed configuration as a PyGraftConfigDict mapping.

    Raises:
        ValueError: If the file does not have a supported extension or cannot
            be parsed as JSON/YAML.
    """
    config_path = Path(path).resolve()

    if config_path.suffix == ".json":
        with config_path.open(encoding="utf8") as file:
            data = json.load(file)
    elif config_path.suffix in {".yaml", ".yml"}:
        with config_path.open(encoding="utf8") as file:
            data = yaml.safe_load(file)
    else:
        message = f"Unknown config format {config_path.suffix}. Use .json, .yaml, or .yml."
        raise ValueError(message)

    logger.info("Loaded configuration file from: %s", config_path)
    # Parsed JSON/YAML is dynamically typed; the structural shape is enforced later by
    # validate_user_config against PyGraftConfigDict.
    return data


def validate_user_config(
    config: PyGraftConfigDict,
    target: ValidationTarget = "both",
) -> None:
    """Validate a user configuration for a given usage target.

    This is the main validation entry point and the only function that external
    callers should use. It enforces:

    - Structural validity:
        - All required top-level keys ("general", "schema", "kg") are present.
        - The "schema" section contains "classes" and "relations" subsections.
        - All required keys inside each section are present.
        - No unexpected keys are included at any level.
    - Serialization validity:
        - The "rdf_format" field under "general" is one of: xml, ttl, nt.
        - The "rng_seed" field, if provided, is either null or an integer.
    - Schema-level constraints (when target is "schema" or "both"):
        - Class generation constraints (depth, branching, disjointness).
        - Relation generation constraints (ratios, domain/range depth, mode).
    - KG-level constraints (when target is "kg" or "both"):
        - KG size constraints (num_entities, num_triples).
        - Ratio constraints for KG distributions.
        - Multityping semantics.

    Args:
        config: Parsed configuration mapping.
        target: Which part of the pipeline the config is being validated for.
            - "schema": validate keys, RDF format, and schema-related parameters.
            - "kg": validate keys, RDF format, and KG-related parameters.
            - "both": run both schema and KG validations.

    Raises:
        ValueError: If the configuration is structurally invalid or any
            parameter combination is logically inconsistent for the chosen
            target.
    """
    # Top-level section names.
    _validate_config_keys(config)
    # Nested keys inside each section, driven by the TypedDicts.
    _validate_section_keys(config)
    # Scalar type validation for all known fields.
    _validate_section_types(config)

    # General
    general_cfg = config["general"]
    # Schema
    schema_cfg = config["schema"]
    classes_cfg = schema_cfg["classes"]
    relations_cfg = schema_cfg["relations"]
    # KG
    kg_cfg = config["kg"]

    _validate_general_section(general_cfg)

    if target in {"schema", "both"}:
        _validate_schema_config(classes_cfg, relations_cfg)

    if target in {"kg", "both"}:
        _validate_kg_config(kg_cfg, classes_cfg)


# --------------------------------------------------------------------------------------------------
# Structural validation (internal)
# --------------------------------------------------------------------------------------------------


def _validate_config_keys(config: PyGraftConfigDict) -> None:
    """Validate that the configuration uses the expected top-level keys.

    This check ensures that:

    - All keys defined in PyGraftConfigDict are present.
    - No additional, unknown top-level keys are included.

    It protects the rest of the pipeline from typo-driven bugs and stale
    configuration parameters.
    """
    expected = REQUIRED_CONFIG_KEYS
    provided = set(config.keys())

    missing = sorted(expected - provided)
    extra = sorted(provided - expected)

    if not missing and not extra:
        return

    parts: list[str] = []
    if missing:
        parts.append(f"missing required keys: {', '.join(missing)}")
    if extra:
        parts.append(f"unexpected keys: {', '.join(extra)}")

    message = "Invalid configuration: " + "; ".join(parts) + "."
    raise ValueError(message)


def _validate_section_keys(config: PyGraftConfigDict) -> None:
    """Validate that each section uses the expected nested keys.

    This leverages the section-level TypedDicts to enforce that "general",
    "schema", and "kg" have exactly the fields we expect and nothing else.
    The "schema" section is validated at two levels: first as a container
    with "classes" and "relations" keys, then each subsection is validated
    for its own expected keys.
    """
    sections_to_validate: list[tuple[str, Mapping[str, object], frozenset[str]]] = [
        # General
        ("general", config["general"], REQUIRED_GENERAL_KEYS),
        # Schema
        ("schema", config["schema"], REQUIRED_SCHEMA_KEYS),
        ("schema.classes", config["schema"]["classes"], REQUIRED_CLASS_KEYS),
        ("schema.relations", config["schema"]["relations"], REQUIRED_RELATION_KEYS),
        # KG
        ("kg", config["kg"], REQUIRED_KG_KEYS),
    ]

    for section_name, section_mapping, expected_keys in sections_to_validate:
        _validate_mapping_keys(section_name, section_mapping, expected_keys)


def _validate_mapping_keys(
    section_name: str,
    mapping: Mapping[str, object],
    expected_keys: frozenset[str],
) -> None:
    """Validate a single section's keys against the expected set."""
    if not isinstance(mapping, dict):
        message = (
            f"Invalid configuration in section {section_name!r}: "
            f"expected a mapping, got {type(mapping)!r}."
        )
        raise TypeError(message)

    provided = set(mapping.keys())
    missing = sorted(expected_keys - provided)
    extra = sorted(provided - expected_keys)

    if not missing and not extra:
        return

    parts: list[str] = []
    if missing:
        parts.append(f"missing required keys: {', '.join(missing)}")
    if extra:
        parts.append(f"unexpected keys: {', '.join(extra)}")

    message = f"Invalid configuration in section {section_name!r}: " + "; ".join(parts) + "."
    raise ValueError(message)


# --------------------------------------------------------------------------------------------------
# Type validation (internal)
# --------------------------------------------------------------------------------------------------


def _validate_section_types(config: PyGraftConfigDict) -> None:
    """Validate scalar types for each configuration section.

    This pass runs after structural validation so that downstream validators
    can assume all values have the correct primitive types (str, int, float,
    bool, or None where explicitly allowed).
    """
    # General
    _validate_general_types(config["general"])
    # Schema
    _validate_class_types(config["schema"]["classes"])
    _validate_relation_types(config["schema"]["relations"])
    # KG
    _validate_kg_types(config["kg"])


def _validate_general_types(general_cfg: GeneralConfigDict) -> None:
    """Validate and normalize scalar types in the 'general' section."""
    general_cfg["project_name"] = _ensure_string_type(
        "general",
        "project_name",
        general_cfg["project_name"],
    )
    general_cfg["rdf_format"] = _ensure_string_type(
        "general",
        "rdf_format",
        general_cfg["rdf_format"],
    )

    rng_seed_value = general_cfg["rng_seed"]
    if rng_seed_value is not None:
        general_cfg["rng_seed"] = _ensure_int_type(
            "general",
            "rng_seed",
            rng_seed_value,
        )


def _validate_class_types(classes_cfg: ClassGenConfigDict) -> None:
    """Validate and normalize scalar types in the 'classes' section."""
    classes_cfg["num_classes"] = _ensure_int_type(
        "classes",
        "num_classes",
        classes_cfg["num_classes"],
    )
    classes_cfg["max_hierarchy_depth"] = _ensure_int_type(
        "classes",
        "max_hierarchy_depth",
        classes_cfg["max_hierarchy_depth"],
    )
    classes_cfg["avg_class_depth"] = _ensure_numeric_type(
        "classes",
        "avg_class_depth",
        classes_cfg["avg_class_depth"],
    )
    classes_cfg["avg_children_per_parent"] = _ensure_numeric_type(
        "classes",
        "avg_children_per_parent",
        classes_cfg["avg_children_per_parent"],
    )
    classes_cfg["avg_disjointness"] = _ensure_numeric_type(
        "classes",
        "avg_disjointness",
        classes_cfg["avg_disjointness"],
    )


def _validate_relation_types(relations_cfg: RelationGenConfigDict) -> None:
    """Validate and normalize scalar types in the 'relations' section."""
    relations_cfg["num_relations"] = _ensure_int_type(
        "relations",
        "num_relations",
        relations_cfg["num_relations"],
    )
    relations_cfg["relation_specificity"] = _ensure_numeric_type(
        "relations",
        "relation_specificity",
        relations_cfg["relation_specificity"],
    )
    relations_cfg["prop_profiled_relations"] = _ensure_numeric_type(
        "relations",
        "prop_profiled_relations",
        relations_cfg["prop_profiled_relations"],
    )
    relations_cfg["profile_side"] = _ensure_string_type(
        "relations",
        "profile_side",
        relations_cfg["profile_side"],
    )

    relations_cfg["prop_symmetric_relations"] = _ensure_numeric_type(
        "relations",
        "prop_symmetric_relations",
        relations_cfg["prop_symmetric_relations"],
    )
    relations_cfg["prop_inverse_relations"] = _ensure_numeric_type(
        "relations",
        "prop_inverse_relations",
        relations_cfg["prop_inverse_relations"],
    )
    relations_cfg["prop_transitive_relations"] = _ensure_numeric_type(
        "relations",
        "prop_transitive_relations",
        relations_cfg["prop_transitive_relations"],
    )
    relations_cfg["prop_asymmetric_relations"] = _ensure_numeric_type(
        "relations",
        "prop_asymmetric_relations",
        relations_cfg["prop_asymmetric_relations"],
    )
    relations_cfg["prop_reflexive_relations"] = _ensure_numeric_type(
        "relations",
        "prop_reflexive_relations",
        relations_cfg["prop_reflexive_relations"],
    )
    relations_cfg["prop_irreflexive_relations"] = _ensure_numeric_type(
        "relations",
        "prop_irreflexive_relations",
        relations_cfg["prop_irreflexive_relations"],
    )
    relations_cfg["prop_functional_relations"] = _ensure_numeric_type(
        "relations",
        "prop_functional_relations",
        relations_cfg["prop_functional_relations"],
    )
    relations_cfg["prop_inverse_functional_relations"] = _ensure_numeric_type(
        "relations",
        "prop_inverse_functional_relations",
        relations_cfg["prop_inverse_functional_relations"],
    )
    relations_cfg["prop_subproperties"] = _ensure_numeric_type(
        "relations",
        "prop_subproperties",
        relations_cfg["prop_subproperties"],
    )


def _validate_kg_types(kg_cfg: KGGenConfigDict) -> None:
    """Validate and normalize scalar types in the 'kg' section."""
    kg_cfg["num_entities"] = _ensure_int_type(
        "kg",
        "num_entities",
        kg_cfg["num_entities"],
    )
    kg_cfg["num_triples"] = _ensure_int_type(
        "kg",
        "num_triples",
        kg_cfg["num_triples"],
    )

    kg_cfg["enable_fast_generation"] = _ensure_bool_type(
        "kg",
        "enable_fast_generation",
        kg_cfg["enable_fast_generation"],
    )

    kg_cfg["relation_usage_uniformity"] = _ensure_numeric_type(
        "kg",
        "relation_usage_uniformity",
        kg_cfg["relation_usage_uniformity"],
    )
    kg_cfg["prop_untyped_entities"] = _ensure_numeric_type(
        "kg",
        "prop_untyped_entities",
        kg_cfg["prop_untyped_entities"],
    )

    kg_cfg["avg_specific_class_depth"] = _ensure_numeric_type(
        "kg",
        "avg_specific_class_depth",
        kg_cfg["avg_specific_class_depth"],
    )

    kg_cfg["multityping"] = _ensure_bool_type(
        "kg",
        "multityping",
        kg_cfg["multityping"],
    )
    kg_cfg["avg_types_per_entity"] = _ensure_numeric_type(
        "kg",
        "avg_types_per_entity",
        kg_cfg["avg_types_per_entity"],
    )

    kg_cfg["check_kg_consistency"] = _ensure_bool_type(
        "kg",
        "check_kg_consistency",
        kg_cfg["check_kg_consistency"],
    )


# ------------------------------------------------------------------------------------------------ #
# General Section Validation                                                                       #
# ------------------------------------------------------------------------------------------------ #


def _validate_general_section(general_cfg: GeneralConfigDict) -> None:
    """Validate RDF format, RNG seed, and project_name."""
    # At this point, types have already been validated by _validate_section_types.

    # --- rdf_format ---
    if general_cfg["rdf_format"] not in {"xml", "ttl", "nt"}:
        message = f"Invalid rdf_format {general_cfg['rdf_format']!r}. Allowed: 'xml', 'ttl', 'nt'."
        raise ValueError(message)

    # --- rng_seed ---
    # Type is already enforced as Optional[int] by _validate_general_types.
    # No additional semantic constraints beyond that for now.

    # --- project_name ---
    name = general_cfg["project_name"]

    name = name.strip()
    if not name:
        msg = "project_name cannot be empty. Use 'auto' or a string."
        raise ValueError(msg)

    if name != "auto":
        # Sanitize automatically
        sanitized = slugify_project_name(name)

        if not sanitized:
            message = f"project_name {name!r} has no usable characters after sanitization."
            raise ValueError(message)

        general_cfg["project_name"] = sanitized
    else:
        # Leave "auto" untouched
        general_cfg["project_name"] = "auto"


# --------------------------------------------------------------------------------------------------
# Schema validation (internal)
# --------------------------------------------------------------------------------------------------


def _validate_schema_config(
    classes_cfg: ClassGenConfigDict,
    relations_cfg: RelationGenConfigDict,
) -> None:
    """Validate schema-related configuration parameters.

    This covers all parameters that influence schema construction:

    - Class generation:
        - num_classes > 0
        - max_hierarchy_depth ≥ 1
        - num_classes ≥ max_hierarchy_depth
        - 0 < avg_class_depth < max_hierarchy_depth
        - avg_children_per_parent > 0
        - avg_disjointness ∈ [0, 1]
    - Relation generation:
        - num_relations > 0
        - relation_specificity ∈ [0, max_hierarchy_depth]
        - profile_side ∈ {"both", "partial"}
        - All relation proportions ∈ [0, 1]
        - Symmetric + asymmetric ≤ 1
        - Reflexive + irreflexive ≤ 1
        - Functional / inverse-functional vs subproperties combination is valid.
    """
    # Types for classes_cfg and relations_cfg have already been validated.

    # --- Class generation ---------------------------------------------------- #
    if classes_cfg["num_classes"] <= 0:
        message = "num_classes must be a positive integer."
        raise ValueError(message)

    if classes_cfg["max_hierarchy_depth"] <= 0:
        message = "max_hierarchy_depth must be a positive integer."
        raise ValueError(message)

    if classes_cfg["num_classes"] < classes_cfg["max_hierarchy_depth"]:
        message = (
            "Invalid configuration: num_classes must be >= max_hierarchy_depth. "
            "The hierarchy cannot be deeper than the number of classes."
        )
        raise ValueError(message)

    if classes_cfg["avg_class_depth"] <= 0.0:
        message = "avg_class_depth must be > 0.0."
        raise ValueError(message)

    if classes_cfg["avg_class_depth"] >= classes_cfg["max_hierarchy_depth"]:
        message = "avg_class_depth cannot be >= max_hierarchy_depth."
        raise ValueError(message)

    if classes_cfg["avg_children_per_parent"] <= 0.0:
        message = "avg_children_per_parent must be > 0.0."
        raise ValueError(message)

    _validate_ratio(classes_cfg["avg_disjointness"], "avg_disjointness")

    # --- Relation generation: ranges and ratios ----------------------------- #
    if relations_cfg["num_relations"] <= 0:
        message = "num_relations must be a positive integer."
        raise ValueError(message)

    if relations_cfg["relation_specificity"] < 0.0:
        message = "relation_specificity must be >= 0.0."
        raise ValueError(message)

    if relations_cfg["relation_specificity"] > classes_cfg["max_hierarchy_depth"]:
        message = "relation_specificity cannot be > max_hierarchy_depth."
        raise ValueError(message)

    _validate_profile_side(relations_cfg["profile_side"])

    _validate_ratio(
        relations_cfg["prop_profiled_relations"],
        "prop_profiled_relations",
    )
    _validate_ratio(
        relations_cfg["prop_symmetric_relations"],
        "prop_symmetric_relations",
    )
    _validate_ratio(
        relations_cfg["prop_inverse_relations"],
        "prop_inverse_relations",
    )
    _validate_ratio(
        relations_cfg["prop_transitive_relations"],
        "prop_transitive_relations",
    )
    _validate_ratio(
        relations_cfg["prop_asymmetric_relations"],
        "prop_asymmetric_relations",
    )
    _validate_ratio(
        relations_cfg["prop_reflexive_relations"],
        "prop_reflexive_relations",
    )
    _validate_ratio(
        relations_cfg["prop_irreflexive_relations"],
        "prop_irreflexive_relations",
    )
    _validate_ratio(
        relations_cfg["prop_functional_relations"],
        "prop_functional_relations",
    )
    _validate_ratio(
        relations_cfg["prop_inverse_functional_relations"],
        "prop_inverse_functional_relations",
    )
    _validate_ratio(
        relations_cfg["prop_subproperties"],
        "prop_subproperties",
    )

    # --- Relation generation: cross-parameter constraints ------------------- #
    if relations_cfg["prop_symmetric_relations"] + relations_cfg["prop_asymmetric_relations"] > 1.0:
        message = "Proportions of owl:Asymmetric and owl:Symmetric relations cannot exceed 1."
        raise ValueError(message)

    if (
        relations_cfg["prop_reflexive_relations"] + relations_cfg["prop_irreflexive_relations"]
        > 1.0
    ):
        message = "Proportions of owl:Reflexive and owl:Irreflexive relations cannot exceed 1."
        raise ValueError(message)

    # Subproperty vs functional / inverse-functional:
    # The generator currently supports either:
    #   - No subproperties with any functional/inverse-functional ratio, or
    #   - Subproperties with at most one of functional/inverse-functional enabled.
    valid_combo = (
        relations_cfg["prop_subproperties"] == 0.0
        and (
            relations_cfg["prop_functional_relations"] >= 0.0
            or relations_cfg["prop_inverse_functional_relations"] >= 0.0
        )
    ) or (
        relations_cfg["prop_subproperties"] >= 0.0
        and (
            relations_cfg["prop_functional_relations"] == 0.0
            or relations_cfg["prop_inverse_functional_relations"] == 0.0
        )
    )

    if not valid_combo:
        message = "Invalid combination of subproperties, functional, and inverse-functional values."
        raise ValueError(message)

    logger.debug("Validated Schema configuration")


def _validate_profile_side(profile_side: str) -> None:
    """Validate that 'profile_side' is one of the supported values.

    The relation generator currently supports two strategies:

    - "both":    assign both domain and range constraints when possible.
    - "partial": assign either a domain or a range constraint (but not both).

    Any other value would lead to undefined behavior when generating relation
    schemas, so we fail fast here.
    """
    allowed_values = {"both", "partial"}
    if profile_side not in allowed_values:
        allowed_str = ", ".join(sorted(allowed_values))
        message = f"Invalid profile_side: {profile_side!r}. Allowed values are: {allowed_str}."
        raise ValueError(message)


# --------------------------------------------------------------------------------------------------
# KG validation (internal)
# --------------------------------------------------------------------------------------------------


def _validate_kg_config(
    kg_cfg: KGGenConfigDict,
    classes_cfg: ClassGenConfigDict,
) -> None:
    """Validate KG-related configuration parameters.

    Checks structural validity (entity/triple counts), ratio bounds,
    multityping rules, and compatibility between avg_specific_class_depth
    and the maximum possible class hierarchy depth implied by the schema configuration.
    """
    # Types for kg_cfg have already been validated.

    # --- Basic numeric constraints ----------------------------------------- #
    if kg_cfg["num_entities"] <= 0:
        message = "num_entities must be a positive integer."
        raise ValueError(message)

    if kg_cfg["num_triples"] <= 0:
        message = "num_triples must be a positive integer."
        raise ValueError(message)

    _validate_ratio(
        kg_cfg["relation_usage_uniformity"],
        "relation_usage_uniformity",
    )
    _validate_ratio(
        kg_cfg["prop_untyped_entities"],
        "prop_untyped_entities",
    )

    # --- Depth constraints -------------------------------------------------- #
    if kg_cfg["avg_specific_class_depth"] <= 0.0:
        message = "avg_specific_class_depth must be > 0.0."
        raise ValueError(message)

    # Cross-check against the schema's theoretical maximum depth.
    # Classes span depths 0..max_hierarchy_depth
    # Entity typing conceptually spans 1..(max_hierarchy_depth + 1)
    max_depth = classes_cfg["max_hierarchy_depth"]
    max_allowed = max_depth + 1
    avg_depth = kg_cfg["avg_specific_class_depth"]
    if avg_depth > max_allowed:
        message = (
            "avg_specific_class_depth is incompatible with max_hierarchy_depth: "
            f"got {avg_depth}, but with max_hierarchy_depth={max_depth} the "
            f"maximum allowed value is {max_allowed}. Lower "
            "avg_specific_class_depth or increase max_hierarchy_depth."
        )
        raise ValueError(message)

    # --- Multityping logic -------------------------------------------------- #
    _enforce_multityping_rules(kg_cfg)

    logger.debug("Validated KG configuration ")


def _enforce_multityping_rules(kg_cfg: KGGenConfigDict) -> None:
    """Enforce that avg_types_per_entity aligns with multityping semantics.

    Semantics:
        - Each typed entity must have at least one most-specific class.
        - When multityping is disabled, every typed entity has exactly one class.
        - When multityping is enabled, entities may have one or more classes.

    This implies:
        - If multityping is False:
            avg_types_per_entity must be exactly 1.0.
        - If multityping is True:
            avg_types_per_entity must be >= 1.0.
            Values below 1.0 are not only invalid but logically impossible.
    """
    if not kg_cfg["multityping"] and kg_cfg["avg_types_per_entity"] != 1.0:
        message = (
            "Invalid avg_types_per_entity: when multityping is false, each entity must "
            "have exactly one type. avg_types_per_entity must therefore be 1.0."
        )
        raise ValueError(message)

    if kg_cfg["multityping"] and kg_cfg["avg_types_per_entity"] < 1.0:
        message = (
            "Invalid avg_types_per_entity: when multityping is true, entities may have "
            "one or more types. avg_types_per_entity must therefore be >= 1.0."
        )
        raise ValueError(message)


# ================================================================================================ #
# General Helpers                                                                                  #
# ================================================================================================ #


def _validate_ratio(value: float, name: str) -> None:
    """Validate that a proportion is within [0.0, 1.0].

    This helper is used for all configuration fields that represent a ratio or
    probability, such as prop_profiled_relations or prop_untyped_entities.
    """
    if not 0.0 <= value <= 1.0:
        message = f"{name} must be between 0.0 and 1.0."
        raise ValueError(message)


def _ensure_string_type(section_name: str, key: str, value: object) -> str:
    """Ensure that a configuration value is a string.

    Args:
        section_name: Name of the configuration section (e.g. "general", "kg").
        key: Name of the key within the section.
        value: Raw value loaded from JSON/YAML.

    Returns:
        The same value, typed as a string.

    Raises:
        TypeError: If the value is not a string.
    """
    if not isinstance(value, str):
        message = (
            f"Invalid configuration in section {section_name!r}: "
            f"{key} must be a string, got {type(value)!r}."
        )
        raise TypeError(message)

    return value


def _ensure_bool_type(section_name: str, key: str, value: object) -> bool:
    """Ensure that a configuration value is a boolean.

    Args:
        section_name: Name of the configuration section.
        key: Name of the key within the section.
        value: Raw value loaded from JSON/YAML.

    Returns:
        The same value, typed as a boolean.

    Raises:
        TypeError: If the value is not a boolean.
    """
    if not isinstance(value, bool):
        message = (
            f"Invalid configuration in section {section_name!r}: "
            f"{key} must be a boolean (true/false), got {type(value)!r}."
        )
        raise TypeError(message)

    return value


def _ensure_int_type(section_name: str, key: str, value: object) -> int:
    """Ensure that a configuration value is an integer and not a boolean.

    Args:
        section_name: Name of the configuration section.
        key: Name of the key within the section.
        value: Raw value loaded from JSON/YAML.

    Returns:
        The same value, typed as an integer.

    Raises:
        TypeError: If the value is not an integer or is a boolean.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        message = (
            f"Invalid configuration in section {section_name!r}: "
            f"{key} must be an integer, got {type(value)!r}."
        )
        raise TypeError(message)

    return value


def _ensure_numeric_type(section_name: str, key: str, value: object) -> float | int:
    """Ensure that a configuration value is numeric (int or float, but not bool).

    This is used for fields that conceptually behave like floats but accept integer
    literals from JSON/YAML.

    Args:
        section_name: Name of the configuration section.
        key: Name of the key within the section.
        value: Raw value loaded from JSON/YAML.

    Returns:
        The same value, typed as a float or integer.

    Raises:
        TypeError: If the value is not numeric.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = (
            f"Invalid configuration in section {section_name!r}: "
            f"{key} must be a number, got {type(value)!r}."
        )
        raise TypeError(message)

    return value
