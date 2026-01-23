---
title: OWL Constraints
description: How PyGraft-gen enforces OWL 2 constraints.
---

# OWL Constraints

PyGraft-gen enforces [OWL 2](https://www.w3.org/TR/owl2-overview/){target="_blank" rel="noopener"} constraints during generation to ensure your Knowledge Graphs are logically consistent. This page explains what these constraints mean and how they shape what gets generated.

**On this page:**

- [Three Types of Constraints](#three-types-of-constraints) - Overview of constraint categories
    - [Property Characteristics](#property-characteristics) - Functional, symmetric, transitive, etc.
    - [Relational Constraints](#relational-constraints) - Domain/range, inverses, subproperties
    - [Class Constraints](#class-constraints) - Disjointness and hierarchy
- [Forbidden Characteristic Combinations](#forbidden-characteristic-combinations) - Invalid combinations

!!! tip "Learn More"
    For implementation details, see [KG Generation](kg-generation.md).

## Three Types of Constraints

OWL constraints fall into three categories, each governing different aspects of your Knowledge Graph:

- [**Property Characteristics**](#property-characteristics) &mdash; How individual relations behave (functional, symmetric, transitive, etc.)

- [**Relational Constraints**](#relational-constraints) &mdash; How properties relate to each other and to classes (domain/range, inverses, subproperties)

- [**Class Constraints**](#class-constraints) &mdash; Rules about entity types (disjointness, hierarchy)

---

### Property Characteristics

Property characteristics define intrinsic behaviors of relations. These are the most fundamental constraints because they affect every triple involving that property.

| Characteristic                                                                                                                    | Property                        | What It Means                                | How It's Enforced                                                         |
|-----------------------------------------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------------------|---------------------------------------------------------------------------|
| [**Functional**](https://www.w3.org/TR/owl2-syntax/#Functional_Object_Properties){target="_blank" rel="noopener"}                 | `owl:FunctionalProperty`        | At most one value per subject                | Rejects triples where head already has an outgoing edge for this relation |
| [**Inverse-Functional**](https://www.w3.org/TR/owl2-syntax/#Inverse-Functional_Object_Properties){target="_blank" rel="noopener"} | `owl:InverseFunctionalProperty` | At most one value per object                 | Rejects triples where tail already has an incoming edge for this relation |
| [**Symmetric**](https://www.w3.org/TR/owl2-syntax/#Symmetric_Object_Properties){target="_blank" rel="noopener"}                   | `owl:SymmetricProperty`         | If A &rarr; B then B &rarr; A                | Generates only one direction, prevents duplicates                         |
| [**Asymmetric**](https://www.w3.org/TR/owl2-syntax/#Asymmetric_Object_Properties){target="_blank" rel="noopener"}                 | `owl:AsymmetricProperty`        | If A &rarr; B then not B &rarr; A            | Rejects triples where reverse edge exists                                 |
| [**Transitive**](https://www.w3.org/TR/owl2-syntax/#Transitive_Object_Properties){target="_blank" rel="noopener"}                 | `owl:TransitiveProperty`        | If A &rarr; B and B &rarr; C then A &rarr; C | Prevents cycles with irreflexive properties                               |
| [**Reflexive**](https://www.w3.org/TR/owl2-syntax/#Reflexive_Object_Properties){target="_blank" rel="noopener"}                   | `owl:ReflexiveProperty`         | Every entity has self-loop                   | **Not materialized** (reasoners infer)                                    |
| [**Irreflexive**](https://www.w3.org/TR/owl2-syntax/#Irreflexive_Object_Properties){target="_blank" rel="noopener"}               | `owl:IrreflexiveProperty`       | No self-loops allowed                        | Rejects triples where head equals tail                                    |



??? example "Functional Properties (`owl:FunctionalProperty`)"
    At most one value for any subject.
    ```turtle
    :hasMotherBiological rdf:type owl:FunctionalProperty .
    ```

    Once E1 has a `:hasMotherBiological` triple, no additional ones are allowed with E1 as the head.

    **Example:**

    - Valid: `E1 hasMotherBiological E2`
    - Invalid: `E1 hasMotherBiological E3` (E1 already has a mother)


??? example "Inverse-Functional Properties (`owl:InverseFunctionalProperty`)"
    At most one value for any object.
    ```turtle
    :hasSocialSecurityNumber rdf:type owl:InverseFunctionalProperty .
    ```

    Once E2 appears as the tail of a `:hasSocialSecurityNumber` triple, no additional ones are allowed with E2 as the tail.

    **Example:**

    - Valid: `E1 hasSocialSecurityNumber E2`
    - Invalid: `E3 hasSocialSecurityNumber E2` (E2 is already someone's SSN)


??? example "Symmetric Properties (`owl:SymmetricProperty`)"
    If it holds from A to B, it automatically holds from B to A.
    ```turtle
    :marriedTo rdf:type owl:SymmetricProperty .
    ```

    PyGraft-gen only creates one direction (e.g., `E1 marriedTo E2`) and relies on reasoners to infer the reverse.

    **Example:**

    - Generated: `E1 marriedTo E2`
    - Inferred: `E2 marriedTo E1` (not explicitly generated)
    - Prevented: `E2 marriedTo E1` (would be a duplicate)


??? example "Asymmetric Properties (`owl:AsymmetricProperty`)"
    If it holds from A to B, it cannot hold from B to A.
    ```turtle
    :isChildOf rdf:type owl:AsymmetricProperty .
    ```

    Before generating `E1 isChildOf E2`, the generator checks that `E2 isChildOf E1` doesn't exist.

    Asymmetric properties are automatically irreflexive (no self-loops).

    **Example:**

    - Valid: `E1 isChildOf E2`
    - Invalid: `E2 isChildOf E1` (violates asymmetry)
    - Invalid: `E1 isChildOf E1` (self-loops forbidden)


??? example "Transitive Properties (`owl:TransitiveProperty`)"
    If it holds from A to B and from B to C, it automatically holds from A to C.
    ```turtle
    :ancestorOf rdf:type owl:TransitiveProperty .
    ```

    The generator prevents transitive cycles with irreflexive properties. If `:ancestorOf` is both transitive and irreflexive, cycles like `E1 &rarr; E2 &rarr; E3 &rarr; E1` are blocked.

    **Example:**

    - Generated: `E1 ancestorOf E2`, `E2 ancestorOf E3`
    - Inferred: `E1 ancestorOf E3` (not explicitly generated)
    - Prevented: `E3 ancestorOf E1` (would create reflexive cycle)


??? example "Reflexive Properties (`owl:ReflexiveProperty`)"
    Must hold for every entity in its domain.
    ```turtle
    :hasIdentity rdf:type owl:ReflexiveProperty ;
                 rdfs:domain :Person .
    ```

    !!! warning "Not Materialized"
        **PyGraft-gen does not generate explicit self-loop triples for reflexive properties.**
        
        OWL 2 DL reasoners automatically infer reflexive closures from the property declaration. Materializing self-loops would add one triple per entity in the domain class with no information gain.
        
        The generated KG is fully compliant. Validation tools recognize reflexive properties and infer self-loops during reasoning.
        
        **What this means:**  
        Schema contains: `:hasIdentity rdf:type owl:ReflexiveProperty`  
        KG does NOT contain: `E1 hasIdentity E1`, `E2 hasIdentity E2`, etc.  
        Reasoners infer: All Person entities have `:hasIdentity` self-loops


??? example "Irreflexive Properties (`owl:IrreflexiveProperty`)"
    No self-loops allowed.
    ```turtle
    :parentOf rdf:type owl:IrreflexiveProperty .
    ```

    The generator immediately rejects any candidate triple where head equals tail.

    **Example:**

    - Valid: `E1 parentOf E2`
    - Invalid: `E1 parentOf E1` (self-loop rejected)


### Relational Constraints

While property characteristics define how individual relations behave, relational constraints define how properties interact with each other and with classes. These constraints establish the structural rules that connect your ontology together.

| Constraint                                                                                                                 | Property                     | What It Means                        | How It's Enforced                                              |
|----------------------------------------------------------------------------------------------------------------------------|------------------------------|--------------------------------------|----------------------------------------------------------------|
| [**Domain/Range**](https://www.w3.org/TR/rdf-schema/#ch_domain){target="_blank" rel="noopener"}                            | `rdfs:domain` / `rdfs:range` | Classes that can be subjects/objects | Pre-filters entity pools to satisfying classes                 |
| [**Inverse Relationships**](https://www.w3.org/TR/owl2-syntax/#Inverse_Object_Properties){target="_blank" rel="noopener"}  | `owl:inverseOf`              | Two properties reverse each other    | Validates inverse triple would be valid                        |
| [**Subproperty Hierarchies**](https://www.w3.org/TR/rdf-schema/#ch_subpropertyof){target="_blank" rel="noopener"}          | `rdfs:subPropertyOf`         | Subproperties inherit constraints    | Validates constraints from all superproperties                 |
| [**Property Disjointness**](https://www.w3.org/TR/owl2-syntax/#Disjoint_Object_Properties){target="_blank" rel="noopener"} | `owl:propertyDisjointWith`   | Properties cannot share instances    | Rejects triples where entity pair exists for disjoint property |


??? example "Domain and Range Constraints (`rdfs:domain` / `rdfs:range`)"
    Domain specifies valid subjects, range specifies valid objects.
    ```turtle
    :worksFor rdfs:domain :Person ;
              rdfs:range :Organization .
    ```

    The generator pre-filters entity pools. For `:worksFor`, only entities typed as `:Person` can be heads, and only entities typed as `:Organization` can be tails.

    **Multiple constraints:** If a property has multiple domains (or ranges), entities must satisfy ALL of them.
    ```turtle
    :teachesAt rdfs:domain :Professor ;
               rdfs:domain :Employee ;
               rdfs:range :University .
    ```

    For `:teachesAt`, heads must be both `:Professor` AND `:Employee`.

    **Example:**

    - Valid: `E1 worksFor E2` (E1 is Person, E2 is Organization)
    - Invalid: `E1 worksFor E3` (E3 is Person, not Organization)


??? example "Inverse Relationships (`owl:inverseOf`)"
    Two properties that reverse each other.
    ```turtle
    :hasChild owl:inverseOf :hasParent .
    ```

    When generating `E1 hasChild E2`, the generator validates that the inverse triple `E2 hasParent E1` would also be valid (checking functional constraints, domain/range, disjointness on both properties).

    The generator does NOT automatically create inverse triples. Only one direction is materialized. Reasoners infer the reverse.

    **Example:**

    - Generated: `E1 hasChild E2`
    - Inferred: `E2 hasParent E1`
    - Validation: Checks that E2 satisfies `:hasParent` domain and E1 satisfies `:hasParent` range


??? example "Subproperty Hierarchies (`rdfs:subPropertyOf`)"
    Subproperties inherit all constraints from their superproperties.
    ```turtle
    :knows rdfs:subPropertyOf :acquaintedWith .
    :acquaintedWith rdfs:domain :Person ;
                    rdfs:range :Person .
    ```

    When generating `:knows` triples, the generator validates constraints from both `:knows` and `:acquaintedWith`.

    **Inherited constraints:**

    - Domain and range restrictions
    - Property characteristics (transitive, symmetric, etc.)
    - Disjointness declarations
    - Functional/inverse-functional constraints

    **Example:**

    If `:acquaintedWith` has domain `:Person` and `:knows` is a subproperty, then `:knows` also requires domain `:Person`.


??? example "Property Disjointness (`owl:propertyDisjointWith`)"
    Two properties cannot share any instance.
    ```turtle
    :hasSpouse owl:propertyDisjointWith :hasSibling .
    ```

    Before generating `E1 hasSpouse E2`, the generator checks that `E1 hasSibling E2` doesn't exist (and vice versa).

    Disjointness propagates through subproperty hierarchies. If P disjoint Q and R subPropertyOf P, then R is also disjoint with Q.

    **Example:**

    - Valid: `E1 hasSpouse E2`
    - Valid: `E1 hasSibling E3`
    - Invalid: `E1 hasSpouse E3` (E3 is already a sibling)


### Class Constraints

Class constraints complete the picture by defining rules about entity types themselves. These constraints determine what combinations of types are valid and how classes relate to each other hierarchically.

??? example "Class Typing (`rdf:type`)"
    Entities are connected to their classes via `rdf:type` assertions.
    ```turtle
    :E1 rdf:type :Student .
    :E2 rdf:type :Professor .
    ```

    When an entity is assigned a class, PyGraft-gen automatically computes all transitive superclasses but only serializes the most-specific types. Reasoners infer the rest.

    **Example:**

    - Entity typed as: `E1 rdf:type :Student`
    - Student subclass of: `:Person` subclass of `owl:Thing`
    - KG contains: `E1 rdf:type :Student` (most-specific only)
    - Reasoners infer: `E1 rdf:type :Person`, `E1 rdf:type owl:Thing`

    [RDF Reference](https://www.w3.org/TR/rdf-schema/#ch_type){target="_blank" rel="noopener"}

??? example "Class Hierarchy (`rdfs:subClassOf`)"
    Classes form a tree structure under `owl:Thing`, the universal root class of all OWL ontologies.
    ```turtle
    owl:Thing a owl:Class .
    
    :Student rdfs:subClassOf :Person .
    :Person rdfs:subClassOf owl:Thing .
    ```

    Every class is implicitly a subclass of `owl:Thing`. The generator automatically adds all transitive superclasses when assigning types.

    **Example:**

    - Assigned: `E1 rdf:type :Student`
    - Computed internally: `E1 rdf:type :Student, :Person, owl:Thing`
    - Serialized to KG: `E1 rdf:type :Student` (most-specific only)

    [RDFS Reference](https://www.w3.org/TR/rdf-schema/#ch_subclassof){target="_blank" rel="noopener"}

??? example "Class Disjointness (`owl:disjointWith`)"
    Entities cannot be instances of disjoint classes simultaneously.
    ```turtle
    :Person owl:disjointWith :Organization .
    ```

    During entity typing, the generator ensures no entity receives types that are mutually disjoint.

    Disjointness extends to subclasses. If `:Person` disjoint `:Organization`, then `:Student` (subclass of `:Person`) is also disjoint with `:Organization`.

    **Example:**

    - Valid: `E1 rdf:type :Person`
    - Valid: `E2 rdf:type :Organization`
    - Invalid: `E1 rdf:type :Person, :Organization` (disjoint types)

    [OWL 2 Reference](https://www.w3.org/TR/owl2-syntax/#Disjoint_Classes){target="_blank" rel="noopener"}

??? info "Multiple Domain/Range Constraints (Intersection Semantics)"
    When a property has multiple domain or range constraints, entities must satisfy **ALL** of them. This is OWL's intersection semantics.
    ```turtle
    :teachesAt rdfs:domain :Professor ;
               rdfs:domain :Employee ;
               rdfs:range :University .
    ```
    
    For `:teachesAt`:
    
    - Subjects must be both `:Professor` **AND** `:Employee`
    - Objects must be `:University`
    
    The generator computes the intersection of all domain classes and all range classes, then pre-filters entity pools accordingly. If the intersection is empty (no entities satisfy all constraints), the relation is excluded from generation.
    
    **Example:**
    
    - Valid: `E1 teachesAt E2` (E1 is Professor AND Employee, E2 is University)
    - Invalid: `E3 teachesAt E2` (E3 is Professor but NOT Employee)
    - Invalid: `E1 teachesAt E4` (E4 is not University)

---

## Forbidden Characteristic Combinations

Not all constraint combinations are valid. Some create logical contradictions that [OWL 2](https://www.w3.org/TR/owl2-overview/){target="_blank" rel="noopener"} forbids. PyGraft-gen detects these during schema loading and either stops generation or excludes problematic relations.

**:fontawesome-solid-circle-xmark: Direct Contradictions (Generation Stops):**

| Combination                                                                                                                                                                                                                           | Why Forbidden                                           |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| [**Reflexive**](https://www.w3.org/TR/owl2-syntax/#Reflexive_Object_Properties){target="_blank" rel="noopener"} + [**Irreflexive**](https://www.w3.org/TR/owl2-syntax/#Irreflexive_Object_Properties){target="_blank" rel="noopener"} | Reflexive requires self-loops; Irreflexive forbids them |
| [**Symmetric**](https://www.w3.org/TR/owl2-syntax/#Symmetric_Object_Properties){target="_blank" rel="noopener"} + [**Asymmetric**](https://www.w3.org/TR/owl2-syntax/#Asymmetric_Object_Properties){target="_blank" rel="noopener"}   | Symmetric requires bidirectional; Asymmetric forbids it |

**:fontawesome-solid-triangle-exclamation: Problematic Combinations (Warnings, Relation Excluded):**

| Combination                                                                                                                                                                                                                                          | Issue                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [**Asymmetric**](https://www.w3.org/TR/owl2-syntax/#Asymmetric_Object_Properties){target="_blank" rel="noopener"} + [**Functional**](https://www.w3.org/TR/owl2-syntax/#Functional_Object_Properties){target="_blank" rel="noopener"}                | Creates inconsistency in [OWL 2](https://www.w3.org/TR/owl2-overview/){target="_blank" rel="noopener"} reasoning |
| [**Asymmetric**](https://www.w3.org/TR/owl2-syntax/#Asymmetric_Object_Properties){target="_blank" rel="noopener"} + [**InverseFunctional**](https://www.w3.org/TR/owl2-syntax/#Inverse-Functional_Object_Properties){target="_blank" rel="noopener"} | Creates inconsistency in [OWL 2](https://www.w3.org/TR/owl2-overview/){target="_blank" rel="noopener"} reasoning |
| [**Transitive**](https://www.w3.org/TR/owl2-syntax/#Transitive_Object_Properties){target="_blank" rel="noopener"} + [**Functional**](https://www.w3.org/TR/owl2-syntax/#Functional_Object_Properties){target="_blank" rel="noopener"}                | Can lead to unintended inference chains and explosions                                                           |

---

## What's Next

- :fontawesome-solid-brain: **[Schema Generation](schema-generation.md)** - How synthetic ontologies are created
- :fontawesome-solid-database: **[KG Generation](kg-generation.md)** - How these constraints are enforced during generation
- :fontawesome-solid-check-circle: **[Consistency Checking](consistency-checking.md)** - Validating generated KGs against constraints
