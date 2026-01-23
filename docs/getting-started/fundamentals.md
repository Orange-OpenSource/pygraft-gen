---
title: Fundamentals
description: Learn the core concepts behind PyGraft-gen — Knowledge Graphs, ontologies, RDF, OWL, and synthetic data generation.
---

# Fundamentals 

This guide introduces the core concepts behind PyGraft-gen: ontologies, Knowledge Graphs, and why you'd want to generate synthetic versions of them.

**On this page, you will find:**

- [What is a Knowledge Graph?](#what-is-a-knowledge-graph)
- [Industry Impact](#industry-impact)
- [What is an Ontology?](#what-is-an-ontology)
- [RDF, OWL & Semantic Web Standards](#rdf-owl-semantic-web-standards)
- [What Does "Synthetic" Mean?](#what-does-synthetic-mean)
- [How PyGraft-gen Fits In](#how-pygraft-gen-fits-in)
- [Ready to Generate?](#ready-to-generate)

---

## :fontawesome-solid-diagram-project: What is a Knowledge Graph?

A **Knowledge Graph** stores information as a network of connected entities and their relationships. Each piece of information is represented as a **triple**: `(subject, predicate, object)`

The triple is the **atomic unit** of Knowledge Graphs – it's the smallest, indivisible piece of structured information. You can't break it down further while preserving meaning.

**Example:**
```turtle
(Neo, knows, Morpheus)
(Neo, escapes, Matrix)
(Matrix, createdBy, Machines)
```

This says: "Neo knows Morpheus. Neo escapes the Matrix. The Matrix was created by Machines."

Knowledge graphs represent facts as interconnected data, making it easy to traverse relationships and discover patterns.


## :fontawesome-solid-chart-line: Industry Impact

Beyond the technical definition, Knowledge Graphs are powering critical systems across industries. They were recognized as a **top enabler technology** by [Gartner's Emerging Tech Impact Radar (2024)](https://www.linkedin.com/posts/gartner-for-high-tech_gartnerht-emergingtech-technology-activity-7209614996492709890-r_7M){target="_blank" rel="noopener"}, driving adoption across:

??? example ":fontawesome-solid-magnifying-glass: Search &mdash; Powering Google's search results (2012)"
    
    Powering [Google's search results](https://en.wikipedia.org/wiki/Knowledge_Graph_(Google)){target="_blank" rel="noopener"} – grew from 570 million entities in 2012 to 500 billion facts on 5 billion entities by 2020, answering roughly one-third of Google's 100 billion monthly searches

??? example ":fontawesome-solid-car: Automotive &mdash; Renault uses KGs to validate car configurations (2012)"
    
    [Renault](https://www.renaultgroup.com/en/){target="_blank" rel="noopener"} uses KGs to [validate car configurations](https://link.springer.com/chapter/10.1007/978-3-642-30284-8_47){target="_blank" rel="noopener"} – encoding constraints between features to automatically filter $10^{20}$ valid configurations from $10^{25}$ possible combinations (only 1 in 100,000 random combinations is valid)

??? example ":fontawesome-solid-industry: Manufacturing &mdash; Volvo Cars developed the Insight Lab (2019)"
    
    [Volvo Cars](https://www.volvocars.com/){target="_blank" rel="noopener"} developed the [Insight Lab](https://linkurious.com/blog/volvo-cars-graph-platform/){target="_blank" rel="noopener"}, an integrated graph service using [Neo4j](https://neo4j.com/videos/graph-tour-2019-volvo-cars/){target="_blank" rel="noopener"} to manage increasingly complex vehicle configurations, customizations, and dependencies between features and functions – transforming complex manufacturing data into actionable insights for cross-team collaboration

??? example ":fontawesome-solid-heart-pulse: Healthcare &mdash; Enabling drug discovery and precision medicine (2020)"
    
    Enabling [drug discovery and precision medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC7327409/){target="_blank" rel="noopener"} through biomedical KGs – integrating genomic, pharmaceutical, and clinical data to identify drug-target interactions and predict disease treatments

??? example ":fontawesome-solid-tower-cell: Telecom &mdash; Nokia leverages KGs for network automation (2023)"
    
    [Nokia](https://www.nokia.com/){target="_blank" rel="noopener"} leverages KGs for [network automation](https://www.nokia.com/blog/learn-how-ai-and-graph-databases-transform-telecom-inventory-solutions/){target="_blank" rel="noopener"} – identifying [$9 billion in potential energy savings](https://watch.knowledgegraph.tech/videos/a-billion-dollar-opportunity-superscaling-knowledge-at-nokia){target="_blank" rel="noopener"} and preventing 280,000 acres of deforestation across mobile network infrastructures

??? example ":fontawesome-solid-mobile-screen: Consumer Electronics &mdash; Samsung acquired Oxford Semantic Technologies (2024)"
    
    [Samsung](https://www.samsung.com/){target="_blank" rel="noopener"} has [acquired Oxford Semantic Technologies](https://news.samsung.com/global/samsung-electronics-announces-acquisition-of-oxford-semantic-technologies-uk-based-knowledge-graph-startup){target="_blank" rel="noopener"} (following [collaboration since 2018](https://techcrunch.com/2024/07/18/samsung-to-acquire-uk-based-knowledge-graph-startup-oxford-semantic-technologies/){target="_blank" rel="noopener"}) – integrating personal Knowledge Graphs with on-device AI to provide hyper-personalized experiences across mobile devices, TVs, and home appliances

!!! info "Source Attribution"
    Industry examples adapted from course materials provided by [Inria Academy](https://www.inria-academy.fr/){target="_blank" rel="noopener"}, authored by [Fabien Gandon](http://fabien.info){target="_blank" rel="noopener"} (Université Côte d'Azur, Inria, CNRS, I3S).

## :fontawesome-solid-sitemap: What is an Ontology?

To build Knowledge Graphs at scale, you need structure. That's where **ontologies** come in – they're the schemas that define what can exist in your graph and how things can relate to each other.

An **ontology** specifies:

- **Classes** – Types of entities (`Person, Company, Location`)
- **Properties** – Relationships between entities (`worksFor, hasRole, locatedIn`)
- **Constraints** – Rules that must be followed (domain, range, disjointness)

**Example ontology:**
```turtle
# Classes
:Person a owl:Class .
:Company a owl:Class .
:Location a owl:Class .

# Properties
:worksFor a owl:ObjectProperty ;
    rdfs:domain :Person ;
    rdfs:range :Company .

:locatedIn a owl:ObjectProperty ;
    rdfs:domain :Company ;
    rdfs:range :Location .

:hasRole a owl:ObjectProperty ;
    rdfs:domain :Person .
```

**This ontology says:**

- People work for Companies (via `worksFor`)
- Companies are located in Locations (via `locatedIn`)
- People have roles (via `hasRole`)

**Domain/Range constraints:**

- `worksFor` can only have a Person as its subject (domain)
- `worksFor` can only have a Company as its object (range)

!!! note "Schema vs Data"
    **Ontology** = Schema (structure and rules)  
    **Knowledge Graph** = Data (actual instances following those rules)


## :fontawesome-solid-globe: RDF, OWL & Semantic Web Standards

Knowledge graphs and ontologies aren't just informal concepts – they're built on rigorous [semantic web standards](https://www.w3.org/standards/semanticweb/){target="_blank" rel="noopener"} developed by the [World Wide Web Consortium (W3C)](https://www.w3.org/){target="_blank" rel="noopener"}:

!!! info "[RDF (Resource Description Framework)](https://www.w3.org/RDF/){target="_blank" rel="noopener"}"
    
    The foundation. Everything is expressed as triples: `(subject, predicate, object)`

!!! info "[RDFS (RDF Schema)](https://www.w3.org/TR/rdf-schema/){target="_blank" rel="noopener"}"
    
    Adds basic vocabulary for defining classes and properties:
    
    - `rdfs:subClassOf` – Class hierarchies
    - `rdfs:domain` – What can be a subject
    - `rdfs:range` – What can be an object

!!! info "[OWL (Web Ontology Language)](https://www.w3.org/TR/owl2-overview/){target="_blank" rel="noopener"}"
    
    Adds rich constraints and logical rules:
    
    - `owl:disjointWith` – Things that can't overlap
    - `owl:FunctionalProperty` – Properties with one value max
    - `owl:SymmetricProperty` – Bidirectional relationships
    - `owl:TransitiveProperty` – Inherited relationships

!!! info "[Turtle syntax](https://www.w3.org/TR/turtle/){target="_blank" rel="noopener"}"
    
    A human-friendly syntax for writing RDF/OWL. All examples in this documentation use Turtle syntax.  
    Other formats exist ([RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/){target="_blank" rel="noopener"}, [N-Triples](https://www.w3.org/TR/n-triples/){target="_blank" rel="noopener"}), but Turtle is the most readable.


## :fontawesome-solid-wand-magic-sparkles: What Does "Synthetic" Mean?

Now that you understand Knowledge Graphs and ontologies, let's talk about why you'd want to generate fake versions of them.

**Synthetic data** is artificially generated data that mimics real data's structure and characteristics without containing actual sensitive information.

=== ":fontawesome-solid-database: Real Knowledge Graph"
    ```turtle
    :JohnDoe a :Patient ;
        :hasCondition :Diabetes ;
        :treatedBy :DrSmith .

    :DrSmith a :Doctor ;
        :worksAt :CityHospital .
    ```

=== ":fontawesome-solid-wand-magic-sparkles: Synthetic Knowledge Graph"
    ```turtle
    :E1 a :Patient ;
        :hasCondition :Diabetes ;
        :treatedBy :E2 .

    :E2 a :Doctor ;
        :worksAt :E3 .
    ```

The synthetic version follows the same ontology (Patient, Doctor, hasCondition, treatedBy, worksAt) but uses generated identifiers (E1, E2, E3) instead of real people and places. The structure, relationships, and constraints are preserved, but the actual data is fake.


## :fontawesome-solid-puzzle-piece: How PyGraft-gen Fits In

PyGraft-gen brings all these concepts together: it generates **synthetic Knowledge Graphs** that follow **ontology constraints** using **W3C semantic web standards**.

You can either:

1. [**Start from an existing ontology**](./quickstart.md/#ontology-extraction-kg-generation) – Extract its structure and generate synthetic instance data that respects all constraints
2. [**Generate everything from scratch**](./quickstart.md/#full-synthetic) – Create both the ontology schema and instance data using statistical parameters

Both approaches produce valid RDF/OWL outputs with constraint-aware generation, ensuring every triple is logically consistent.

---

## :fontawesome-solid-rocket: Ready to Generate?

You now understand the fundamentals of Knowledge Graphs, ontologies, and synthetic data generation.

**Next steps:**

<div class="grid" markdown>

- :fontawesome-solid-download: **[Install PyGraft-gen](installation.md)** – Set up in minutes
- :fontawesome-solid-rocket: **[Quickstart](quickstart.md)** – Generate your first KG
- :fontawesome-solid-brain: **[Core Concepts](../concepts/index.md)** – Learn how the generation algorithms work

</div>
