---
title: Namespaces Info
description: Reference for namespaces_info.json — prefix mappings, ontology namespace, and IRI normalization.
---

# Namespaces Info

This file provides namespace prefix mappings for normalizing IRIs to human-readable `prefix:LocalName` format across all extraction outputs.

**On this page:**

- [Structure](#structure) - Complete field reference
- [Example](#example) - Sample JSON output


## Structure

| Field         | Description                                                                                                |
|---------------|------------------------------------------------------------------------------------------------------------|
| `ontology`    | The ontology's logical prefix and namespace, inferred from VANN annotations or `owl:Ontology` declarations |
| `prefixes`    | All namespace prefixes declared in the ontology graph, normalized for JSON serialization                   |
| `no_prefixes` | Namespaces used in the ontology but lacking declared prefixes (remain as full IRIs)                        |

!!! info "Key Details"
    - `_empty_prefix` represents the default namespace declared as `@prefix :` in the ontology
    - `_missing` is a synthetic fallback prefix for IRIs with no matching namespace
    - `_missing` never appears in this file but may appear in `class_info.json` and `relation_info.json` as `_missing:LocalName`
    - All identifiers are normalized to `prefix:LocalName` format for human readability


## Example

```json
{
  "ontology": {
    "prefix": "sc",
    "namespace": "http://pygraf.t/"
  },
  "prefixes": {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "sc": "http://pygraf.t/",
    "_empty_prefix": "http://pygraf.t/"
  },
  "no_prefixes": [
    "http://external.example.org/vocab/",
    "http://another.external/schema#"
  ]
}
```
