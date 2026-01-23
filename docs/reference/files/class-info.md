---
title: Class Info
description: Reference for class_info.json — class hierarchy, transitive closures, layer mappings, and disjointness constraints.
---

# Class Info

This file captures the class hierarchy and constraints extracted from your ontology, providing the structural metadata needed for KG generation to enforce class relationships and disjointness rules.

**On this page:**

- [Structure](#structure) - Complete field reference
- [Example](#example) - Sample JSON output


## Structure

The file is organized into four main groups: statistics, class lists, hierarchy mappings, and disjointness constraints.

| Field                                              | Description                                                                         |
|----------------------------------------------------|-------------------------------------------------------------------------------------|
| **:fontawesome-solid-chart-bar: Statistics**       |                                                                                     |
| `num_classes`                                      | Total number of extracted classes                                                   |
| `hierarchy_depth`                                  | Maximum layer depth in the ontology                                                 |
| `avg_class_depth`                                  | Mean layer index across all classes                                                 |
| `avg_children_per_parent`                          | Average number of direct subclasses per parent class                                |
| `avg_class_disjointness`                           | Average number of disjoint declarations per class                                   |
| **:fontawesome-solid-list: Class List**            |                                                                                     |
| `classes`                                          | Sorted list of all extracted class identifiers in `prefix:LocalName` format         |
| **:fontawesome-solid-sitemap: Hierarchy Mappings** |                                                                                     |
| **Direct Hierarchy**                               |                                                                                     |
| `direct_class2subclasses`                          | Maps each class to its immediate subclasses                                         |
| `direct_class2superclasses`                        | Maps each class to its immediate superclasses                                       |
| **Transitive Hierarchy**                           |                                                                                     |
| `transitive_class2subclasses`                      | Maps each class to all direct and indirect subclasses                               |
| `transitive_class2superclasses`                    | Maps each class to all direct and indirect superclasses (includes the class itself) |
| **Layering**                                       |                                                                                     |
| `layer2classes`                                    | Groups classes by layer index (keys become strings in JSON)                         |
| `class2layer`                                      | Maps each class to its integer layer depth                                          |
| **:fontawesome-solid-ban: Disjointness Mappings**  |                                                                                     |
| `class2disjoints`                                  | Maps each class to explicitly declared disjoint classes (asserted only)             |
| `class2disjoints_symmetric`                        | List of symmetric disjoint pairs encoded as `"ClassA-ClassB"`                       |
| `class2disjoints_extended`                         | Maps each class to all classes disjoint by inheritance propagation                  |

!!! info "Key Details"
    - All class identifiers use `prefix:LocalName` format from `namespaces_info.json`
    - Classes with no matching prefix: `_missing:LocalName`
    - `transitive_class2superclasses` includes each class in its own ancestor set
    - Integer keys in `layer2classes` are serialized as JSON strings



## Example

```json
{
  "statistics": {
    "num_classes": 3,
    "hierarchy_depth": 2,
    "avg_class_depth": 1.67,
    "avg_children_per_parent": 1.5,
    "avg_class_disjointness": 0.67
  },
  "classes": [
    "C1",
    "C2",
    "C3"
  ],
  "direct_class2subclasses": {
    "owl:Thing": ["C1", "C3"],
    "C1": ["C2"]
  },
  "direct_class2superclasses": {
    "C1": ["owl:Thing"],
    "C2": ["C1"],
    "C3": ["owl:Thing"]
  },
  "transitive_class2subclasses": {
    "owl:Thing": ["C1", "C2", "C3"],
    "C1": ["C2"]
  },
  "transitive_class2superclasses": {
    "C1": ["C1", "owl:Thing"],
    "C2": ["C2", "C1", "owl:Thing"],
    "C3": ["C3", "owl:Thing"]
  },
  "class2disjoints": {
    "C1": ["C3"]
  },
  "class2disjoints_symmetric": [
    "C1-C3"
  ],
  "class2disjoints_extended": {
    "C1": ["C3"],
    "C2": ["C3"],
    "C3": ["C1", "C2"]
  },
  "layer2classes": {
    "1": ["C1", "C3"],
    "2": ["C2"]
  },
  "class2layer": {
    "C1": 1,
    "C2": 2,
    "C3": 1
  }
}
```
