---
title: Relation Info
description: Reference for relation_info.json — OWL property characteristics, domain/range, inverses, and subproperties.
---

# Relation Info

This file captures object property constraints and characteristics extracted from your ontology, providing the metadata needed for KG generation to enforce relation semantics and OWL rules.

**On this page:**

- [Structure](#structure) - Complete field reference
- [Example](#example) - Sample JSON output


## Structure

The file is organized into six main groups: statistics, relation lists, property characteristics, and constraint mappings.

| Field                                                       | Description                                                                    |
|-------------------------------------------------------------|--------------------------------------------------------------------------------|
| **:fontawesome-solid-chart-bar: Statistics**                |                                                                                |
| `num_relations`                                             | Total number of extracted relations                                            |
| `prop_reflexive`                                            | Proportion of relations declared reflexive                                     |
| `prop_irreflexive`                                          | Proportion of relations declared irreflexive                                   |
| `prop_functional`                                           | Proportion of relations declared functional                                    |
| `prop_inversefunctional`                                    | Proportion of relations declared inverse-functional                            |
| `prop_symmetric`                                            | Proportion of relations declared symmetric                                     |
| `prop_asymmetric`                                           | Proportion of relations declared asymmetric                                    |
| `prop_transitive`                                           | Proportion of relations declared transitive                                    |
| `prop_inverseof`                                            | Proportion of relations with explicit inverse declarations                     |
| `prop_subpropertyof`                                        | Proportion of relations participating in subproperty hierarchy                 |
| `prop_profiled_relations`                                   | Proportion of relations with any extracted feature                             |
| `relation_specificity`                                      | Average number of domain/range constraints per relation                        |
| **:fontawesome-solid-list: Relation List**                  |                                                                                |
| `relations`                                                 | Sorted list of all extracted relation identifiers in `prefix:LocalName` format |
| **:fontawesome-solid-link: OWL Property Characteristics**   |                                                                                |
| `rel2patterns`                                              | Maps each relation to its list of declared OWL characteristics                 |
| `reflexive_relations`                                       | Relations declared as `owl:ReflexiveProperty`                                  |
| `irreflexive_relations`                                     | Relations declared as `owl:IrreflexiveProperty`                                |
| `symmetric_relations`                                       | Relations declared as `owl:SymmetricProperty`                                  |
| `asymmetric_relations`                                      | Relations declared as `owl:AsymmetricProperty`                                 |
| `functional_relations`                                      | Relations declared as `owl:FunctionalProperty`                                 |
| `inversefunctional_relations`                               | Relations declared as `owl:InverseFunctionalProperty`                          |
| `transitive_relations`                                      | Relations declared as `owl:TransitiveProperty`                                 |
| **:fontawesome-solid-arrows-rotate: Inverse Relationships** |                                                                                |
| `inverseof_relations`                                       | List of relations participating in `owl:inverseOf` declarations                |
| `rel2inverse`                                               | Symmetric mapping from each relation to its declared inverse                   |
| **:fontawesome-solid-sitemap: Subproperty Hierarchy**       |                                                                                |
| `subrelations`                                              | List of relations that are subproperties of other relations                    |
| `rel2superrel`                                              | Maps each subrelation to its list of direct superrelations                     |
| **:fontawesome-solid-ban: Disjointness Mappings**           |                                                                                |
| `rel2disjoints`                                             | Maps each relation to explicitly declared disjoint relations (asserted only)   |
| `rel2disjoints_symmetric`                                   | List of symmetric disjoint pairs encoded as `"RelA-RelB"`                      |
| `rel2disjoints_extended`                                    | Maps each relation to all relations disjoint by inheritance propagation        |
| **:fontawesome-solid-tags: Domain and Range**               |                                                                                |
| `rel2dom`                                                   | Maps each relation to its list of domain class constraints                     |
| `rel2range`                                                 | Maps each relation to its list of range class constraints                      |


!!! info "Key Details"
    - All relation identifiers use `prefix:LocalName` format from `namespaces_info.json`
    - Relations with no matching prefix: `_missing:LocalName`
    - Only **object properties** are extracted (datatype and annotation properties excluded)
    - `rel2patterns` includes every relation, even with no characteristics (empty list)
    - Relations without domain/range constraints are omitted from `rel2dom`/`rel2range`
    - Conflicting inverse declarations cause extraction to fail



## Example

```json
{
  "statistics": {
    "num_relations": 4,
    "prop_reflexive": 0.0,
    "prop_irreflexive": 0.0,
    "prop_functional": 0.25,
    "prop_inversefunctional": 0.0,
    "prop_symmetric": 0.25,
    "prop_asymmetric": 0.0,
    "prop_transitive": 0.25,
    "prop_inverseof": 0.5,
    "prop_subpropertyof": 0.25,
    "prop_profiled_relations": 0.75,
    "relation_specificity": 1.5
  },
  "relations": [
    "R1",
    "R2",
    "R3",
    "R4"
  ],
  "rel2patterns": {
    "R1": ["owl:SymmetricProperty"],
    "R2": ["owl:TransitiveProperty"],
    "R3": [],
    "R4": ["owl:FunctionalProperty"]
  },
  "reflexive_relations": [],
  "irreflexive_relations": [],
  "symmetric_relations": ["R1"],
  "asymmetric_relations": [],
  "functional_relations": ["R4"],
  "inversefunctional_relations": [],
  "transitive_relations": ["R2"],
  "inverseof_relations": ["R1", "R3"],
  "rel2inverse": {
    "R1": "R3",
    "R3": "R1"
  },
  "subrelations": ["R2"],
  "rel2superrel": {
    "R2": ["R1"]
  },
  "rel2disjoints": {},
  "rel2disjoints_symmetric": [],
  "rel2disjoints_extended": {},
  "rel2dom": {
    "R2": ["C1"],
    "R4": ["C2"]
  },
  "rel2range": {
    "R2": ["C3"],
    "R4": ["C1"]
  }
}
```
