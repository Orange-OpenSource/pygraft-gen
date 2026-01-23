---
title: Contributing
description: How to contribute to PyGraft-gen — bug reports, feature requests, code contributions, and desirable features.
---

# Contributing to PyGraft-gen

We would love for you to contribute to **PyGraft-gen** and help make it even better than it is today! Please contribute to this repository if any of the following is true:  

- You have expertise in *Knowledge Graphs*, *synthetic datasets*, or *graph generation*,  
- You have expertise in *stochastic generation*, *rule-based generation*, or *subgraph matching techniques*,  
- You want to *challenge AI pipelines* with domain-related Knowledge Graphs and domain-specific graph patterns. 

!!! info "For Developers"
    See the [Development Guide](development.md) for setup instructions, code standards, and tooling.

## How to Contribute

We welcome community contributions, whether documentation, refactoring, tests, or new features.

You can contribute to PyGraft-gen in the following ways:

- Ask questions or share ideas in [Discussions](https://github.com/Orange-OpenSource/pygraft-gen/discussions){target="_blank" rel="noopener"}
- Help other users by answering questions in [Discussions](https://github.com/Orange-OpenSource/pygraft-gen/discussions){target="_blank" rel="noopener"} or commenting on [open issues](https://github.com/Orange-OpenSource/pygraft-gen/issues){target="_blank" rel="noopener"}
- File a [bug report](https://github.com/Orange-OpenSource/pygraft-gen/issues/new?template=bug_report.md){target="_blank" rel="noopener"}
- File a [feature request](https://github.com/Orange-OpenSource/pygraft-gen/issues/new?template=feature_request.md){target="_blank" rel="noopener"}
- Help implementing unit tests
- Help refactoring code and ensuring best practices are respected
- Generate clean docstrings for the existing code base
- Implement a new feature (see [Desirable Features](#desirable-features))

## Desirable Features

Want to contribute but not sure where to start? Here are features we're working toward, organized by priority.

!!! info "Current Focus"
    After ontology extraction ([v0.0.8](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.8){target="_blank" rel="noopener"}) and KG optimization ([v0.0.10](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.10){target="_blank" rel="noopener"}), we're prioritizing infrastructure (testing, CI/CD, code quality) before expanding object property support.

### High Priority

- [x] **Support for any input ontology** *(inherited from PyGraft, [v0.0.8](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.8){target="_blank" rel="noopener"})* - Generate KGs from real-world ontologies, not just PyGraft-generated schemas
- [x] **Large-scale KG generation** *([v0.0.10](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.10){target="_blank" rel="noopener"})* - Optimized KG generator architecture enabling millions of entities and tens of millions of triples
- [ ] **Unit test suite** - Comprehensive tests for core generation modules (schema, entities, triples)
- [ ] **Pre-commit hooks** - Automated code quality checks
- [ ] **CI/CD pipeline** - GitHub Actions for testing and deployment
- [ ] **Docstring standardization** - Clean, consistent Google-style docstrings

### Medium Priority

- [ ] **Conflict resolution** *(inherited from PyGraft)* - Fix conflicts between `rdfs:subPropertyOf`, `owl:FunctionalProperty`, and `owl:InverseFunctionalProperty`
- [ ] **Inconsistency explanations** *(inherited from PyGraft)* - Parse HermiT/Pellet output to identify and remove problematic triples without regenerating
- [ ] **Blank-node class expressions** - Support `owl:Restriction`, `owl:unionOf`, `owl:intersectionOf`, `owl:complementOf`
- [ ] **Value restrictions** - Support `owl:someValuesFrom`, `owl:allValuesFrom`, `owl:hasValue`
- [ ] **Compound domain/range** - Complex class expressions in property constraints
- [ ] **Higher-level disjointness** - Support `owl:AllDisjointClasses`, `owl:disjointUnionOf`

### Low Priority

- [ ] **JSON Schema validation** - Validate user configurations against formal schema

!!! info "Not Currently Prioritized"
    Datatype properties (literal-valued attributes like strings, integers, dates) will be addressed after object property support is complete.

Interested in tackling one of these? Start a [Discussion](https://github.com/Orange-OpenSource/pygraft-gen/discussions){target="_blank" rel="noopener"} to discuss your approach!

---

## Communication

GitHub is our primary communication platform. Use [Discussions](https://github.com/Orange-OpenSource/pygraft-gen/discussions){target="_blank" rel="noopener"} for questions, ideas, and general support. Reserve [Issues](https://github.com/Orange-OpenSource/pygraft-gen/issues){target="_blank" rel="noopener"} for bug reports and feature requests, and [Pull Requests](https://github.com/Orange-OpenSource/pygraft-gen/pulls){target="_blank" rel="noopener"} for code contributions. 

You may also contact the maintainers by email for more specific purposes and questions.

We value respectful and constructive communication. Keep discussions focused on practical problems and solutions. All interactions follow [GitHub's Code of Conduct](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct){target="_blank" rel="noopener"}.
