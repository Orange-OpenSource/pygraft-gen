---
title: Consistency Checking
description: Validate generated schemas and Knowledge Graphs with HermiT and Pellet OWL 2 reasoners.
---

# Consistency Checking

Consistency checking validates that generated schemas and knowledge graphs are logically coherent according to [OWL](https://www.w3.org/TR/owl2-overview/){target="_blank" rel="noopener"} semantics.

**On this page:**

- [How It Works](#how-it-works) - Understanding HermiT and Pellet reasoners
- [Checking Consistency](#checking-consistency) - Automatic vs manual validation
- [Performance Considerations](#performance-considerations) - Runtime expectations and optimization
- [Configuration](#configuration) - Enabling/disabling consistency checks
- [Java Requirements](#java-requirements) - Memory configuration for reasoners


!!! warning "Consistency ≠ Correctness"
    **A consistent result only means no ontology rules were violated. It does NOT mean your KG is correct.**

??? danger "Critical Limitation #1: Consistent ≠ Correct"
    A KG can pass consistency checking but still contain errors or nonsensical data. "Consistent" only means no constraints were broken.
    
    **Example:** An empty ontology with no constraints will validate any KG as "consistent" since there are no rules to violate, regardless of whether the data is actually correct.
    
    The KG generator is designed to produce correct data, but errors are possible.

??? danger "Critical Limitation #2: Generation vs Validation Gap"
    **This applies to the ontology extraction workflow:**
    
    - **KG generation uses**: `class_info.json`, `relation_info.json`, `namespaces_info.json`
    - **Consistency checking uses**: Full ontology + KG
    
    If your ontology contains constraints not captured in the info files (unsupported OWL constructs), those constraints are:
    
    - **NOT enforced** during generation
    - **BUT validated** during consistency checking
    
    **Result:** A KG can be generated "correctly" from the info files yet still fail consistency checking against the full ontology.
    
    As PyGraft-gen evolves to support additional OWL constructs (see [What's Supported](../getting-started/quickstart.md/#whats-supported)), this gap will narrow.

---

## How It Works

PyGraft-gen uses two [OWL](https://www.w3.org/TR/owl2-overview/){target="_blank" rel="noopener"} reasoners via [Owlready2](https://owlready2.readthedocs.io/){target="_blank" rel="noopener"} to validate your schemas and knowledge graphs. Each reasoner serves a different purpose: HermiT provides fast yes/no answers, while Pellet explains what went wrong.

- **[HermiT](http://www.hermit-reasoner.com/){target="_blank" rel="noopener"}** - Fast consistency validation (yes/no answer)
- **[Pellet](https://github.com/stardog-union/pellet){target="_blank" rel="noopener"}** - Detailed inconsistency explanations (identifies problematic axioms)

??? info ":fontawesome-solid-gears: Technical Process"
    1. HermiT and Pellet only accept [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/){target="_blank" rel="noopener"} format
    2. If your schema/KG uses [Turtle](https://www.w3.org/TR/turtle/){target="_blank" rel="noopener"} (`.ttl`) or [N-Triples](https://www.w3.org/TR/n-triples/){target="_blank" rel="noopener"} (`.nt`), PyGraft-gen automatically creates a temporary RDF/XML file
    3. For KG validation, schema and KG are merged into a single temporary RDF/XML file
    4. Reasoner runs on the temporary file
    5. Temporary files are cleaned up automatically after reasoning
    
    This conversion is transparent - you can use any RDF format and PyGraft-gen handles the rest.

### :fontawesome-solid-check-circle: HermiT Reasoner

HermiT performs fast consistency checking and reports whether an ontology is consistent or inconsistent.

**:fontawesome-solid-book: What it validates:**

See the [OWL 2 specification](https://www.w3.org/TR/owl2-overview/){target="_blank" rel="noopener"} or [HermiT documentation](http://www.hermit-reasoner.com/){target="_blank" rel="noopener"} for details on OWL reasoning.

**:fontawesome-solid-database: Behavior with datatypes:**

HermiT is configured with `ignore_unsupported_datatypes=True`, meaning it will skip over datatypes it doesn't recognize rather than failing. This allows reasoning to proceed even with custom or uncommon datatype definitions.

**:fontawesome-solid-triangle-exclamation: Limitations:**

- **Cannot provide details** - HermiT only reports consistent/inconsistent, not which axioms are problematic
- **Validates constraints only** - If your schema has few constraints, most KGs will pass validation regardless of data quality

### :fontawesome-solid-magnifying-glass: Pellet Reasoner

Pellet provides detailed explanations when inconsistencies are detected, identifying which specific axioms cause contradictions.

**When to use:**

- KG reported as inconsistent by HermiT
- Need to identify which specific axioms are problematic
- Debugging schema or generation issues

!!! warning "Pellet Performance"
    Pellet is **significantly slower** than HermiT. On large KGs, Pellet can take hours or may not complete. Only use on small to medium KGs for debugging.

---

## Checking Consistency

Now that you understand the reasoners, let's look at how to use them. Consistency checking happens in two contexts: automatically during generation, or manually as a standalone process.

### :fontawesome-solid-bolt: During Generation (Automatic)

When `check_kg_consistency: true` in your config, HermiT runs automatically after KG generation:
```json
{
  "kg": {
    "check_kg_consistency": true
  }
}
```

=== "CLI"
    ```bash
    pygraft kg pygraft.config.json
    # HermiT runs automatically if check_kg_consistency: true
    ```

=== "Python API"
    ```python
    from pygraft import generate_kg

    kg_info, kg_file, is_consistent = generate_kg("pygraft.config.json")
    
    if is_consistent:
        print("KG is consistent!")
    else:
        print("KG is inconsistent - use pygraft explain to debug")
    ```

**:fontawesome-solid-circle-info: Result:** You get a boolean answer (consistent/inconsistent) but no details about what's wrong.

### :fontawesome-solid-hand-pointer: Standalone Checking (Manual)

You can check consistency of any KG file at any time using the `pygraft explain` command.

**:fontawesome-solid-list-check: When to use:**

- Debugging inconsistent KGs without re-running generation
- Testing existing KG files
- Choosing which reasoner to use (HermiT, Pellet, or both)

=== "CLI"
    ```bash
    # Default: uses Pellet (slow but detailed)
    pygraft explain path/to/kg.ttl
    
    # Check with HermiT only (fast, yes/no answer)
    pygraft explain path/to/kg.ttl --reasoner hermit
    
    # Explicit Pellet (same as default)
    pygraft explain path/to/kg.ttl --reasoner pellet
    
    # Run both reasoners (HermiT first, then Pellet if inconsistent)
    pygraft explain path/to/kg.ttl --reasoner both
    ```

=== "Python API"
    ```python
    from pygraft import explain_kg

    # Default: uses Pellet
    is_consistent = explain_kg("path/to/kg.ttl")
    
    # Check with HermiT only
    is_consistent = explain_kg("path/to/kg.ttl", reasoner="hermit")
    
    # Explicit Pellet (same as default)
    is_consistent = explain_kg("path/to/kg.ttl", reasoner="pellet")
    
    # Run both
    is_consistent = explain_kg("path/to/kg.ttl", reasoner="both")
    ```

**:fontawesome-solid-gears: Reasoner options:**

| Option             | Behavior                                                      |
|--------------------|---------------------------------------------------------------|
| `pellet` (default) | Run Pellet only (slow, detailed explanation)                  |
| `hermit`           | Run HermiT only (fast, yes/no answer)                         |
| `both`             | Run HermiT first; if inconsistent, run Pellet for explanation |

!!! tip "Recommended Workflow"
    1. Generate KG with `check_kg_consistency: true` to get quick HermiT validation
    2. If inconsistent, run `pygraft explain kg.ttl` (uses Pellet by default) to debug
    3. This avoids redundant HermiT execution and lets you explain any KG without regenerating

---

## Performance Considerations

Consistency checking performance varies dramatically based on your graph size and schema complexity. Understanding these trade-offs helps you choose the right validation strategy.

### :fontawesome-solid-clock: Typical Runtime

| Graph Size                     | HermiT                     | Pellet                   |
|--------------------------------|----------------------------|--------------------------|
| **Small schemas**              | Seconds                    | Seconds                  |
| **Small KGs** (10K entities)   | Seconds to minutes         | Minutes                  |
| **Medium KGs** (100K entities) | Minutes to tens of minutes | Tens of minutes to hours |
| **Large KGs** (1M+ entities)   | Hours or may not complete  | Likely will not complete |

### :fontawesome-solid-gauge: Factors Affecting Performance

- Number of entities and triples
- Schema complexity (disjointness, property characteristics)
- Number of constraints to validate

!!! danger "Large KG Workflow"
    For large KGs (1M+ entities):
    
    1. Test configuration on small KG (1K-10K entities) with checking enabled
    2. Verify consistency
    3. Test on medium KG (100K entities) with checking enabled
    4. Once validated, disable checking for large production KGs
    5. Generate large KG with `"check_kg_consistency": false`
    
    This validates generation logic without waiting hours on massive graphs.

### Understanding Results

Once validation completes, you'll get one of two outcomes:

!!! success "Consistent"
    ```
    (HermiT) Consistent schema
    ```
    
    No logical contradictions detected. Your KG respects all ontology constraints.

!!! danger "Inconsistent"
    ```
    (HermiT) Inconsistent schema
    ```
    
    Logical contradictions found. Use `pygraft explain` with Pellet to identify specific issues:
    ```bash
    pygraft explain kg.ttl --reasoner pellet
    ```
    
    **Common causes:**
    
    - Entities with disjoint types
    - Property domain/range violations
    - Functional property with multiple values
    - Conflicting property characteristics
    - Transitive property contradictions

---

## Configuration

Consistency checking behavior is controlled through your [configuration file](../reference/files/config.md/#kg), with different rules for schemas and KGs.

!!! abstract "Schema Consistency Checking"
    Always runs and cannot be disabled. Every generated schema is validated for logical coherence.

!!! abstract "KG Consistency Checking"
    Controlled in your config file:
    ```json
    {
      "kg": {
        "check_kg_consistency": true
      }
    }
    ```
    
    | Setting | When to Use |
    |---------|-------------|
    | `true`  | :fontawesome-solid-flask: Development and validation - automatic HermiT check after generation |
    | `false` | :fontawesome-solid-rocket: Production after validation - skip automatic checking |
    
    !!! tip "Flexibility"
        Even with `check_kg_consistency: false`, you can still manually check any KG later using `pygraft explain`.


## Java Requirements

Both reasoners run in a Java Virtual Machine, which requires proper memory configuration to avoid crashes on medium to large ontologies.

### :fontawesome-solid-memory: Automatic Heap Configuration

PyGraft-gen automatically configures the JVM heap to **85% of system RAM** by default. This prevents `OutOfMemoryError` issues that occur with Java's default (often very low) heap size.

**How it works:**

1. :fontawesome-solid-magnifying-glass: PyGraft-gen checks if you've already set `-Xmx` via environment variables
2. :fontawesome-solid-microchip: If not configured, it detects system RAM and sets heap to 85%
3. :fontawesome-solid-shield-halved: Minimum heap: 1GB
4. :fontawesome-solid-bolt: Configuration happens transparently before the first reasoner run

### :fontawesome-solid-sliders: Manual Override

If you need to set a specific heap size:
```bash
export JAVA_TOOL_OPTIONS="-Xmx8g"  # 8GB heap
pygraft explain kg.ttl --reasoner pellet
```

This will override PyGraft-gen's automatic configuration.

!!! info "Why This Matters"
    Without sufficient heap, reasoners will crash with `OutOfMemoryError: Java heap space`, especially on medium to large ontologies. PyGraft-gen's automatic configuration handles this for you.

!!! tip "To install Java"
    See [Java Installation](../getting-started/installation.md/#java-optional) 

---

## What's Next

- :fontawesome-solid-brain: **[Schema Generation](schema-generation.md)** - How ontologies are built
- :fontawesome-solid-database: **[KG Generation](kg-generation.md)** - How instances are created
- :fontawesome-solid-shield-halved: **[OWL Constraints](owl-constraints.md)** - Understanding the constraints being validated
