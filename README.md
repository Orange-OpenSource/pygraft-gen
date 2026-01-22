# PyGraft-gen

Generate synthetic RDFS/OWL ontologies and RDF Knowledge Graphs at scale.

**PyGraft-gen** creates synthetic Knowledge Graphs with reliable structure and constraint-aware generation, making it ideal for testing AI pipelines, benchmarking graph algorithms, and advancing research in scenarios where real data is sensitive or unavailable. 

It also aims to advance the topic of generating realistic RDF Knowledge Graphs through parametric generation.

A major evolution of [PyGraft](https://github.com/nicolas-hbt/pygraft), originally developed by Nicolas Hubert and awarded Best Resource Paper at ESWC 2024. **PyGraft-gen** uses stochastic generation to produce ontologies and Knowledge Graphs while respecting OWL constraints.

**Typical workflows are:**

- Generate a synthetic RDFS/OWL ontology from statistical parameters
- Generate an RDF Knowledge Graph from a synthetic ontology
- Generate an RDF Knowledge Graph from a user-provided ontology

<!-- Using raw GitHub URL so image renders on PyPI -->
![pygraft-gen_framework](https://raw.githubusercontent.com/Orange-OpenSource/pygraft-gen/master/docs/assets/images/pygraft-gen_framework.png)

## Installation 

```bash
pip install pygraft-gen

uv add pygraft-gen

poetry add pygraft-gen
```

**Requirements:** Python 3.10+, Java (optional, for reasoning)

See the [installation documentation](https://orange-opensource.github.io/pygraft-gen/getting-started/installation/) for more details. 

## Usage

```bash
pygraft --help
```

See the [quickstart documentation](https://orange-opensource.github.io/pygraft-gen/getting-started/quickstart/) for complete examples.

## Documentation

Full documentation at **[orange-opensource.github.io/pygraft-gen](https://orange-opensource.github.io/pygraft-gen/)**.

## Repository Structure

```
pygraft-gen/
|-- src/          # PyGraft-gen library
|-- docs/         # Documentation source
+-- evaluation/   # Subgraph matching research (experimental)
```

The `evaluation/` directory contains ongoing research on subgraph matching patterns and is separate from the main library.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) to get started, or the [contributing documentation](https://orange-opensource.github.io/pygraft-gen/about/contributing/) for more details.

## Copyright

Copyright (c) 2024-2025, Orange and Nicolas HUBERT. All rights reserved.

## License

[MIT-License](LICENSE.txt).

## Maintainer

* [Ovidiu PASCAL](mailto:ovidiu.pascal@orange.com)
* [Nicolas HUBERT](mailto:nicotab540@gmail.com)
* [Lionel TAILHARDAT](mailto:lionel.tailhardat@orange.com)
