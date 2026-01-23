---
title: KG Info
description: Reference for kg_info.json — generation statistics comparing requested parameters vs actual results.
---

# KG Info

This file provides summary statistics comparing your requested generation parameters against the actual measured results after KG generation.

**On this page:**

- [Structure](#structure) - Complete field reference
- [Example](#example) - Sample JSON output



## Structure

| Field                                           | Description                                                                |
|-------------------------------------------------|----------------------------------------------------------------------------|
| **:fontawesome-solid-sliders: User Parameters** |                                                                            |
| `num_entities`                                  | Requested number of entity instances                                       |
| `num_triples`                                   | Target number of triples                                                   |
| `enable_fast_generation`                        | Whether fast generation mode was enabled                                   |
| `relation_usage_uniformity`                     | Requested distribution uniformity of relations (0.0-1.0)                   |
| `prop_untyped_entities`                         | Requested proportion of untyped entities (0.0-1.0)                         |
| `avg_specific_class_depth`                      | Target average depth of most specific classes assigned to entities         |
| `multityping`                                   | Whether entities could receive multiple most-specific types                |
| `avg_types_per_entity`                          | Target average number of types per entity                                  |
| `check_kg_consistency`                          | Whether HermiT consistency checking was performed                          |
| **:fontawesome-solid-chart-bar: Statistics**    |                                                                            |
| `num_entities`                                  | Actual number of entities appearing in triples (may differ from requested) |
| `num_instantiated_relations`                    | Number of distinct relations actually used in the KG                       |
| `num_triples`                                   | Actual triple count after generation                                       |
| `prop_untyped_entities`                         | Measured proportion of entities without `rdf:type` assertions              |
| `avg_specific_class_depth`                      | Measured average depth of most specific classes assigned                   |
| `avg_types_per_entity`                          | Measured average number of most-specific types per typed entity            |

!!! info "Key Details"
    - **User Parameters** mirrors the `kg` section from `pygraft_config.json`
    - **Statistics** shows actual measured values after generation
    - `num_entities` in statistics may be lower than requested if some entities remain isolated
    - `num_triples` should match requested count unless generation was interrupted
    - `num_instantiated_relations` shows how many available relations were actually used



## Example

```json
{
  "user_parameters": {
    "num_entities": 1000,
    "num_triples": 10000,
    "enable_fast_generation": false,
    "relation_usage_uniformity": 0.9,
    "prop_untyped_entities": 0.0,
    "avg_specific_class_depth": 2.0,
    "multityping": false,
    "avg_types_per_entity": 1.0,
    "check_kg_consistency": false
  },
  "statistics": {
    "num_entities": 950,
    "num_instantiated_relations": 32,
    "num_triples": 10000,
    "prop_untyped_entities": 0.0,
    "avg_specific_class_depth": 1.71,
    "avg_types_per_entity": 1.0
  }
}
```
