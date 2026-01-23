---
title: Quickstart
description: Generate your first Knowledge Graph in under 5 minutes. Extract from existing ontologies or generate everything from scratch.
---

# Quickstart

Generate your first Knowledge Graph in **under 5 minutes**.

!!! info "Before You Start"
    Make sure you've [installed PyGraft-gen](installation.md)

## Choose Your Approach

[:material-file-document-check: Ontology Extraction + KG Generation](#ontology-extraction-kg-generation){ .md-button .md-button--primary }
[:fontawesome-solid-wand-magic-sparkles: Full Synthetic Generation](#full-synthetic){ .md-button }


---

### :material-file-document-check: Ontology Extraction + KG Generation

You have an existing ontology and want to create synthetic instance data that follows its schema.

**1. Extract the ontology structure**
```bash
pygraft extract path/to/your-ontology.ttl
```
This creates `pygraft.config.json` in your current working directory with extracted metadata.

**2. Configure KG generation**

Open `pygraft.config.json` and edit the `kg` section:
```json
{
  "kg": {
    "num_entities": 3000,               // Number of entity instances to generate
    "num_triples": 30000,               // Target number of triples

    "enable_fast_generation": false,    // Speed optimization: creates smaller base KG then replicates it

    "relation_usage_uniformity": 0.9,   // Controls distribution of relations across triples
    "prop_untyped_entities": 0.0,       // Proportion of entities that remain untyped

    "avg_specific_class_depth": 2.0,    // Average depth of the most specific class assigned to each entity

    "multityping": false,               // Whether entities may receive multiple most-specific types
    "avg_types_per_entity": 1.0,        // Average number of most-specific classes per typed entity

    "check_kg_consistency": true        // Run post-generation HermiT reasoning to verify consistency
  }
}
```

**3. Generate the Knowledge Graph**
```bash
pygraft kg pygraft.config.json
```

**4. Verify the output**

You should see:

```
output_pygraft/your-ontology-name/
├── schema.ttl              # Copy of your ontology
├── kg.ttl                  # Generated Knowledge Graph ✨
├── class_info.json         # Extracted class metadata
├── relation_info.json      # Extracted relation metadata
├── namespaces_info.json    # Namespace mappings
└── kg_info.json            # Generation statistics
```

**5. Debug inconsistencies (if needed)**

If HermiT reports your KG as inconsistent, use the `explain` command to identify problematic axioms:

```bash
pygraft explain output_pygraft/your-ontology-name/kg.ttl
```

This runs the Pellet reasoner to provide detailed explanations of which specific triples or constraints are causing logical contradictions.

!!! tip "Learn More"
    See [Consistency Checking](../concepts/consistency-checking.md).

!!! success "You're Done!"
    Your Knowledge Graph is in `kg.ttl`. Open it to see the generated triples, or use `kg_info.json` to review statistics.


### :fontawesome-solid-wand-magic-sparkles: Full Synthetic

Generate both the ontology schema and instance data from statistical parameters.

**1. Create a configuration template**
```bash
pygraft init json
```
This creates `pygraft.config.json` in your current working directory.

**2. Configure generation parameters**

Open `pygraft.config.json` and edit the essential parameters:

```json
{
  "general": {
    "project_name": "auto",           // Auto-generate timestamped folder name
    "rdf_format": "ttl"                // Output format (ttl, rdf, or nt)
  },

  "schema": {
    "classes": {
      "num_classes": 50,               // Number of classes in ontology
      "max_hierarchy_depth": 4,        // How deep the class tree goes
      ...
    },
    "relations": {
      "num_relations": 50,             // Number of object properties
      ...
    }
  },

  "kg": {
    "num_entities": 3000,              // Number of entity instances
    "num_triples": 30000,              // Target number of triples
    "check_kg_consistency": true       // Run HermiT reasoning to verify consistency
  }
}
```

!!! tip "Learn More"
    See [Configuration Reference](../reference/files/config.md) for all available parameters.


**3. Generate schema and KG**

=== "Single Command"
    ```bash
    pygraft build pygraft.config.json
    ```

=== "Separate Steps"
    ```bash
    pygraft schema pygraft.config.json
    pygraft kg pygraft.config.json
    ```

**4. Verify the output**

You should see:
```
output_pygraft/2026-01-16_14-30-22/
├── schema.ttl              # Generated ontology ✨
├── kg.ttl                  # Generated Knowledge Graph ✨
├── class_info.json         # Class hierarchy metadata
├── relation_info.json      # Relation constraints
└── kg_info.json            # Generation statistics
```

!!! success "You're Done!"
    Both your synthetic schema and KG are ready. Review `schema.ttl` to see the ontology structure and `kg.ttl` for the instance data.

---

## What's Supported

!!! info "Current Focus: Object Properties"
    PyGraft-gen currently focuses on **object properties** (entity-to-entity relations). We're working to complete full object property support before moving to datatype properties. Future versions will add:
    
    - **Blank-node class expressions** – `owl:Restriction`, `owl:unionOf`, `owl:intersectionOf`, `owl:complementOf`, etc.
    - **Value restrictions** – `owl:someValuesFrom`, `owl:allValuesFrom`, `owl:hasValue`, etc.
    - **Compound domain/range** – Complex class expressions in property constraints
    - **Higher-level disjointness** – `owl:AllDisjointClasses`, `owl:disjointUnionOf`, etc.
    
    Once object properties are complete, we'll add **datatype properties** (literal-valued attributes like strings, integers, dates, etc.).
    
    These additions require defining how they should be modeled from the extracted ontology, enforced during generation, and integrated with existing constraints. These are design questions we're actively exploring.



## FAQ

??? question "Consistency checking takes forever?"
    Consistency checking uses the **HermiT reasoner**. Runtime depends on:
    
    - Schema complexity (constraints, disjointness, property characteristics)
    - KG size (entities and triples)
    
    **Typical runtime:**
    
    - Small KGs (10K entities): Seconds to minutes
    - Medium KGs (100K entities): Minutes to tens of minutes  
    - Large KGs (1M+ entities): Hours or may not complete
    
    **For large KGs:**
    
    1. Test on small KG (1K-100K entities) with checking enabled first
    2. Once validated, disable for production: `"check_kg_consistency": false`
    
    &rarr; [Learn more about consistency checking](../concepts/consistency-checking.md)

---

## What's Next?

**Based on your goal:**

<div class="grid" markdown>

- :fontawesome-solid-brain: **[Core Concepts](../concepts/index.md)** – Learn how PyGraft-gen works
- :fontawesome-solid-sliders: **[Configuration Reference](../reference/files/config.md)** – Tune generation parameters
- :fontawesome-solid-shield-halved: **[OWL Constraints](../concepts/owl-constraints.md)** – Understand constraints
- :fontawesome-solid-check-circle: **[Consistency Checking](../concepts/consistency-checking.md)** – Understand consistency validation

</div>

**Technical Reference:**

<div class="grid" markdown>

- :fontawesome-solid-terminal: **[CLI Commands](../reference/cli.md)** – Command-line interface
- :fontawesome-brands-python: **[Python API](../reference/api.md)** – Programmatic usage
- :fontawesome-solid-file-lines: **[Files & Outputs](../reference/files/index.md)** – Configuration and generated files

</div>
