# PyGraft-gen

The **PyGraft-gen** framework is a tool for generating RDFS/OWL ontologies and knowledge graphs.
It helps to test AI pipelines by creating synthetic data with reliable knowledge structure and characteristic graph patterns.
It also aims to advance the topic of generating realistic RDF Knowledge Graphs through parametric generation.

Fundamentally, *PyGraft-gen* is a major evolution of the *[PyGraft](https://github.com/nicolas-hbt/pygraft)* project initially developed by Nicolas HUBERT.
At the core of *PyGraft-gen*, a stochastic generation approach is used to produce ontologies and knowledge graphs.

Typical workflows with *PyGraft-gen* are:
- Produce a synthetic RDFS/OWL ontology,
- Produce an RDF knowledge graph from a synthetic ontology,
- Produce an RDF knowledge graph from a user-provided ontology.

![pygraft-gen_framework](docs/diagrams/pygraft-gen_framework.png)

## Usage

To install the *PyGraft-gen* framework:
- Git clone this repository,
- Create and activate a [Python virtual environment](https://www.w3schools.com/python/python_virtualenv.asp),
- Install requirements: `pip3 install -e .`

To run the *PyGraft-gen* tool:
- Call the tool from a terminal and check for the available CLI options: `pygraft`

> [!NOTE]
> PyGraft-gen features: we are currently cleaning the PyGraft-gen code to provide a nice user experience.
> Please hold on a few days for the upcoming releases:
> - v0.0.7 CLI Modernization
> - v0.0.8 Ontology Extraction feature
> - etc.

See also the *Repository Structure* for navigating into this repository:
```
pygraft-gen
├───docs <code and usage directions>
├───evaluation <subgraph matching patterns and tools>
└───src <the PyGraft-gen implementation>
```

If you would like to contribute to the *PyGraft-gen* project, please check the [CONTRIBUTING](CONTRIBUTING.md) document.

## Copyright

Copyright (c) 2024-2025, Orange and Nicolas HUBERT. All rights reserved.

## License

[MIT-License](LICENSE.txt).

## Maintainer

* [Ovidiu PASCAL](mailto:ovidiu.pascal@orange.com)
* [Nicolas HUBERT](mailto:nicotab540@gmail.com)
* [Lionel TAILHARDAT](mailto:lionel.tailhardat@orange.com)
