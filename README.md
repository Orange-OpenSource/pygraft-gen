# PyGraft-gen

**Generate synthetic RDFS/OWL ontologies and RDF Knowledge Graphs at scale.**

**PyGraft-gen** uses stochastic generation to produce ontologies and Knowledge Graphs with reliable structure while respecting OWL constraints, making it ideal for testing AI pipelines, benchmarking graph algorithms, and research scenarios where real data is sensitive or unavailable.

**It also aims to advance the topic of generating realistic RDF Knowledge Graphs through parametric generation.**

**PyGraft-Gen** is a major evolution of [PyGraft](https://github.com/nicolas-hbt/pygraft), originally developed by **Nicolas Hubert** and **awarded Best Resource Paper at ESWC 2024**.

**Typical workflows are:**

- Generate a synthetic RDFS/OWL ontology from statistical parameters
- Generate an RDF Knowledge Graph from a synthetic ontology
- Generate an RDF Knowledge Graph from a user-provided ontology

<!-- Using raw GitHub URL so image renders on PyPI -->
![pygraft-gen_framework](https://raw.githubusercontent.com/Orange-OpenSource/pygraft-gen/master/docs/assets/images/pygraft-gen_framework.png)

**Repository Structure:**

```text 
.
├── evaluation/   # Subgraph matching research (experimental)
├── docs/         # Documentation source
└── src/          # PyGraft-gen library
```

The `evaluation/` directory contains ongoing research on subgraph matching patterns and is separate from the main library.

## Installation

**Requirements:** Python 3.10+, Java (optional, for reasoning)

**pip:**
```bash
pip install pygraft-gen
```

**uv:**
```bash
uv add pygraft-gen
```

**poetry:**
```bash
poetry add pygraft-gen
```

Verify the installation:
```bash
pygraft --help
```

See the [installation documentation](https://orange-opensource.github.io/pygraft-gen/getting-started/installation/) for setup details and the [quickstart](https://orange-opensource.github.io/pygraft-gen/getting-started/quickstart/) for complete examples.

## Documentation

See the **[official documentation](https://orange-opensource.github.io/pygraft-gen/)** for guides, API reference, and examples.


## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Copyright

Copyright (c) 2024-2025, Orange and Nicolas HUBERT. All rights reserved.

## License

[MIT-License](LICENSE.txt).

## Maintainer

- [Ovidiu PASCAL](mailto:ovidiu.pascal@orange.com)
- [Lionel TAILHARDAT](mailto:lionel.tailhardat@orange.com)
- [Nicolas HUBERT](mailto:nicotab540@gmail.com)
