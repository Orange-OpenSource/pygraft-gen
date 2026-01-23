---
title: Configuration Reference
description: Complete reference for pygraft.config.json — general settings, schema parameters, and KG generation options.
---

# Configuration File

This page documents the structure of `pygraft.config.{json/yml}`, the main configuration file for PyGraft-gen.

This file controls **all aspects of schema and Knowledge Graph generation**, including class hierarchies, relation characteristics, instance populations, and output formats.

**On this page:**

- [Structure](#structure) - The three main configuration sections
- [Formats](#formats) - JSON vs YAML
- [Configuration for Extracted Ontologies](#configuration-for-extracted-ontologies) - Extraction workflow specifics
- [FAQ](#faq) - Common questions and edge cases


!!! warning "Configuration Scope"
    - **`general`**: Used by all commands
    - **`schema`** (contains `classes` and `relations`): Only used during **schema generation**
    - **`kg`**: Only used during **KG generation**
    
    **Critical:** If you run `pygraft kg`, the `schema` section is completely ignored

---

## Structure

The configuration file is organized into three main sections, each controlling different aspects of generation.

### :fontawesome-solid-gear: `general`

Project-wide settings that apply to all generation tasks.

| Parameter      | Description                                                                                                                                               | Allowed / Typical Values        |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| `project_name` | Output folder name. Use `"auto"` for automatic timestamped folders (schema) or to reuse existing schema folders (KG).                                     | Any string or `"auto"`          |
| `rdf_format`   | RDF serialization format for schema and KG output.                                                                                                        | `"xml"`, `"ttl"`, `"nt"`        |
| `rng_seed`     | Random seed for reproducibility. When `null`, generation is stochastic; when an integer, all outputs become deterministic.                                | `null` (default) or any integer |


### :fontawesome-solid-sitemap: `schema.classes`

Controls synthetic class hierarchy generation.

| Parameter                 | Description                                                                                                        | Allowed / Typical Values        |
|---------------------------|--------------------------------------------------------------------------------------------------------------------|---------------------------------|
| `num_classes`             | Total number of classes to generate.                                                                               | Positive integer                |
| `max_hierarchy_depth`     | Maximum depth of the class hierarchy under `owl:Thing`.                                                            | >= 1                            |
| `avg_class_depth`         | Target average depth for classes in the hierarchy.                                                                 | > 0 and < `max_hierarchy_depth` |
| `avg_children_per_parent` | Average number of direct subclasses per parent (controls branching/tree shape).                                    | > 0.0                           |
| `avg_disjointness`        | Target proportion of class pairs marked as disjoint. Higher values create more logically separated class clusters. | 0.0-1.0                         |


### :fontawesome-solid-link: `schema.relations`

Controls object property generation and OWL/RDFS characteristics.

| Parameter                           | Description                                                                                                      | Allowed / Typical Values  |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------|---------------------------|
| `num_relations`                     | Number of object properties to generate.                                                                         | Positive integer          |
| `relation_specificity`              | Target average depth of domain/range class assignments. Higher values encourage more specific class constraints. | 0.0-`max_hierarchy_depth` |
| `prop_profiled_relations`           | Proportion of relations that receive `rdfs:domain` and/or `rdfs:range` constraints.                              | 0.0-1.0                   |
| `profile_side`                      | Whether profiled relations have both domain and range or at least one.                                           | `"both"`, `"partial"`     |
| `prop_symmetric_relations`          | Proportion marked as `owl:SymmetricProperty`.                                                                    | 0.0-1.0                   |
| `prop_inverse_relations`            | Proportion that participate in `owl:inverseOf` pairs.                                                            | 0.0-1.0                   |
| `prop_transitive_relations`         | Proportion declared as `owl:TransitiveProperty`.                                                                 | 0.0-1.0                   |
| `prop_asymmetric_relations`         | Proportion declared as `owl:AsymmetricProperty`.                                                                 | 0.0-1.0                   |
| `prop_reflexive_relations`          | Proportion declared as `owl:ReflexiveProperty`. **Never receive domain/range constraints.**                      | 0.0-1.0                   |
| `prop_irreflexive_relations`        | Proportion declared as `owl:IrreflexiveProperty`.                                                                | 0.0-1.0                   |
| `prop_functional_relations`         | Proportion declared as `owl:FunctionalProperty` (each subject has at most one object).                           | 0.0-1.0                   |
| `prop_inverse_functional_relations` | Proportion declared as `owl:InverseFunctionalProperty` (each object has at most one subject).                    | 0.0-1.0                   |
| `prop_subproperties`                | Proportion assigned as subproperties in `rdfs:subPropertyOf` hierarchies.                                        | 0.0-1.0                   |


### :fontawesome-solid-database: `kg`

Controls Knowledge Graph instance generation.

| Parameter                   | Description                                                                                                               | Allowed / Typical Values           |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------|------------------------------------|
| `num_entities`              | Number of entity instances to generate.                                                                                   | > 0                                |
| `num_triples`               | Target number of triples before inference-driven expansion.                                                               | > 0                                |
| `enable_fast_generation`    | Speed optimization: creates smaller base KG then replicates it. Trades diversity for faster generation on large graphs.   | `true`, `false`                    |
| `relation_usage_uniformity` | Controls distribution of relations across triples. Higher values produce more balanced usage.                             | 0.0-1.0                            |
| `prop_untyped_entities`     | Proportion of entities that remain untyped (no `rdf:type` assertion).                                                     | 0.0-1.0                            |
| `avg_specific_class_depth`  | Average depth of the most specific class assigned to each typed entity.                                                   | > 0.0 and <= `max_hierarchy_depth` |
| `multityping`               | Whether entities may receive multiple most-specific types.                                                                | `true`, `false`                    |
| `avg_types_per_entity`      | Average number of most-specific classes per typed entity. Must be **1.0** when `multityping=false`; **>= 1.0** otherwise. | >= 1.0                             |
| `check_kg_consistency`      | Whether to run post-generation HermiT reasoning to verify schema+KG consistency.                                          | `true`, `false`                    |

---

## Formats

PyGraft-gen supports both JSON and YAML configuration formats.
```bash
pygraft init --format json  # Creates pygraft.config.json
pygraft init --format yaml  # Creates pygraft.config.yml
```

Both formats are functionally equivalent - use whichever you prefer.

### Example Configuration
```json
{
  "general": {
    "project_name": "auto",
    "rdf_format": "ttl",
    "rng_seed": null
  },

  "schema": {
    "classes": {
      "num_classes": 50,
      "max_hierarchy_depth": 4,
      "avg_class_depth": 2.5,
      "avg_children_per_parent": 2.0,
      "avg_disjointness": 0.3
    },

    "relations": {
      "num_relations": 50,
      "relation_specificity": 2.5,
      "prop_profiled_relations": 0.9,
      "profile_side": "both",

      "prop_symmetric_relations": 0.3,
      "prop_inverse_relations": 0.3,
      "prop_transitive_relations": 0.1,
      "prop_asymmetric_relations": 0.0,
      "prop_reflexive_relations": 0.3,
      "prop_irreflexive_relations": 0.0,
      "prop_functional_relations": 0.0,
      "prop_inverse_functional_relations": 0.0,
      "prop_subproperties": 0.3
    }
  },

  "kg": {
    "num_entities": 3000,
    "num_triples": 30000,

    "enable_fast_generation": true,

    "relation_usage_uniformity": 0.9,
    "prop_untyped_entities": 0.0,

    "avg_specific_class_depth": 2.0,

    "multityping": false,
    "avg_types_per_entity": 1.0,

    "check_kg_consistency": true
  }
}
```

---

## Configuration for Extracted Ontologies

When using the **ontology extraction workflow** (`pygraft extract`), the configuration file is partially auto-generated.

**What gets auto-generated:**

1. Run extraction: `pygraft extract ontology.ttl`
2. PyGraft creates or updates a config file with:
   - **`general.project_name`**: Set to extraction output folder name
   - **`general.rdf_format`**: Matches ontology format
   - **`schema` section**: Auto-filled with extraction statistics (informational only)
   - **`kg` section**: Left untouched if file exists, otherwise populated with default template values

**What you configure:**

Edit the generated config to set KG generation parameters in the `kg` section, then run:
```bash
pygraft kg pygraft.config.json
```

!!! warning "Schema Section is Read-Only"
    After extraction, the `schema` section shows what was found but **does not control generation**. Only the `kg` section matters.

!!! tip "Learn More"
    See [Ontology Extraction](../../concepts/ontology-extraction.md) for details on this workflow.

---

## FAQ

??? question "How does `project_name` work differently across commands?"
    The `project_name` parameter behaves differently depending on the command:

    **Schema generation** (`pygraft schema` or `pygraft build`):

    - `"auto"`: Creates timestamped folder (e.g., `2025-12-05_13-22-44`)
    - Custom name: Creates/reuses named folder (normalized and slugified)

    **KG generation** (`pygraft kg`):

    - `"auto"`: Reuses most recent synthetic (timestamped) schema folder
    - Custom name: Reuses existing folder (required for extracted ontologies like "noria", "foaf")

??? question "What are `prop_*` parameters?"
    All `prop_*` parameters are proportions between 0.0 and 1.0, controlling what percentage of relations receive specific characteristics.

??? question "How does `rng_seed` affect generation?"
    The `rng_seed` parameter affects all random decisions throughout generation. Set it to an integer for fully reproducible results, or leave it as `null` for stochastic generation.

??? question "What happens when `multityping=false`?"
    When `multityping=false`, `avg_types_per_entity` must be exactly 1.0 (automatically enforced if not specified).

??? question "Why don't reflexive relations get domain/range constraints?"
    Reflexive relations never receive domain/range constraints because they must apply to all entities in their domain by definition.
