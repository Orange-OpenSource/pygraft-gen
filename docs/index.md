---
title: PyGraft-gen Documentation
description: Generate synthetic RDFS/OWL ontologies and RDF Knowledge Graphs at scale.
---

# PyGraft-gen

<div class="hero" markdown>

**Generate synthetic RDFS/OWL ontologies and RDF Knowledge Graphs at scale.**

PyGraft-gen creates synthetic Knowledge Graphs with realistic structure and constraint-aware generation, making it ideal for testing AI systems, benchmarking graph algorithms, and advancing research in scenarios where real data is sensitive or unavailable.

[:octicons-rocket-24: Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }
[:octicons-mark-github-16: View on GitHub](https://github.com/Orange-OpenSource/pygraft-gen){target="_blank" rel="noopener" .md-button }

---

## Installation

=== "pip"
    ```bash
    # From PyPI (recommended)
    pip install pygraft-gen

    # From GitHub (latest from main branch)
    pip install git+https://github.com/Orange-OpenSource/pygraft-gen.git
    ```

=== "uv"
    ```bash
    # From PyPI (recommended)
    uv add pygraft-gen

    # From GitHub (latest from main branch)
    uv pip install git+https://github.com/Orange-OpenSource/pygraft-gen.git
    ```

=== "poetry"
    ```bash
    # From PyPI (recommended)
    poetry add pygraft-gen

    # From GitHub (latest from main branch)
    poetry add git+https://github.com/Orange-OpenSource/pygraft-gen.git
    ```

**Requirements:** Python 3.10+, Java (optional)

!!! tip "Learn more"
    See **[Installation](getting-started/installation.md)** for detailed setup instructions and Java configuration.


</div>

---

<div class="grid cards" markdown>

-   :fontawesome-solid-book-open: **New to Knowledge Graphs?**

    ---

    Learn Ontologies, RDF, OWL, and the Semantic Web standards that power PyGraft-gen.
    
    *[Start with the basics &rarr;](getting-started/fundamentals.md)*

-   :fontawesome-solid-arrows-split-up-and-left: **Two Flexible Workflows**

    ---

    Generate from scratch using statistical parameters or extract structure from real ontologies to create synthetic instances.
    
    *[See both workflows &rarr;](getting-started/quickstart.md)*

-   :fontawesome-solid-shield-halved: **Constraint-Aware Generation**

    ---

    Enforces OWL constraints during generation and validates results with HermiT and Pellet reasoners.
    
    *[Learn about constraints &rarr;](concepts/owl-constraints.md)*

-   :fontawesome-solid-gauge-high: **Production-Scale Performance**

    ---

    Built to handle millions of entities and tens of millions of triples with optimized generation architecture and fast sampling mode.
    
    *[Explore generation details &rarr;](concepts/kg-generation.md)*

-   :fontawesome-solid-dice: **Stochastic by Design**

    ---

    Generates diverse, randomized graphs by default. Optionally set a random seed for reproducible results in testing and research.
    
    *[Configure generation &rarr;](reference/files/config.md)*

</div>


---

## Research & Current Focus

!!! success ":trophy: Built on Award-Winning Research"
    PyGraft-gen is built on **[PyGraft](https://github.com/nicolas-hbt/pygraft){target="_blank" rel="noopener"}**, which received the [**Best Resource Paper Award at ESWC 2024**](https://2024.eswc-conferences.org/awards/){target="_blank" rel="noopener"}.
    
    **[Read the paper &rarr;](about/publications.md)**

!!! info "Current Focus: Object Properties"
    PyGraft-gen currently focuses on **object properties** (entity-to-entity relations). We're working to complete full object property support before moving to datatype properties. Future versions will add:
    
    - **Blank-node class expressions** – `owl:Restriction`, `owl:unionOf`, `owl:intersectionOf`, `owl:complementOf`, etc.
    - **Value restrictions** – `owl:someValuesFrom`, `owl:allValuesFrom`, `owl:hasValue`, etc.
    - **Compound domain/range** – Complex class expressions in property constraints
    - **Higher-level disjointness** – `owl:AllDisjointClasses`, `owl:disjointUnionOf`, etc.
    
    Once object properties are complete, we'll add **datatype properties** (literal-valued attributes like strings, integers, dates, etc.).
    
    These additions require defining how they should be modeled from the extracted ontology, enforced during generation, and integrated with existing constraints. These are design questions we're actively exploring.


## Community & Support

<div class="grid" markdown>

- :fontawesome-brands-github: **[Discussions](https://github.com/Orange-OpenSource/pygraft-gen/discussions){target="_blank" rel="noopener"}** &mdash; Questions, ideas, and support
- :fontawesome-brands-github: **[Report Issues](https://github.com/Orange-OpenSource/pygraft-gen/issues){target="_blank" rel="noopener"}** &mdash; Found a bug?
- :fontawesome-solid-book: **[Publications](about/publications.md)** &mdash; Read the research
- :fontawesome-solid-code-pull-request: **[Contributing](about/contributing.md)** &mdash; Join development

</div>
