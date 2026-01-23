---
title: CLI Reference
description: Command-line interface for PyGraft-gen. 
---

# `pygraft`

PyGraft: Configurable generation of Schemas &amp; Knowledge Graphs

**Usage**:

```console
$ pygraft [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `-V, --version`: Show the pygraft version and exit.
* `-l, --log-level LEVEL`: Logging verbosity: debug, info, warning, error, critical (or 10-50).  [default: warning]
* `--help`: Show this message and exit.

**Commands**:

* `init`: Create a new configuration template.
* `schema`: Generate a synthetic schema from...
* `kg`: Generate a Knowledge Graph from an...
* `build`: Generate both schema and KG in one step...
* `extract`: Extract ontology metadata into PyGraft...
* `explain`: Run reasoner to explain inconsistencies in...

## `pygraft init`

Create a new configuration template.

**Usage**:

```console
$ pygraft init [OPTIONS] FORMAT
```

**Arguments**:

* `FORMAT`: Config file format to generate (json, yml, yaml).

**Options**:

* `-o, --output-dir DIRECTORY`: Destination directory for the config file. Defaults to current directory.
* `--help`: Show this message and exit.

## `pygraft schema`

Generate a synthetic schema from configuration parameters.

**Usage**:

```console
$ pygraft schema [OPTIONS] CONFIG
```

**Arguments**:

* `CONFIG`: Path to the configuration file.

**Options**:

* `--help`: Show this message and exit.

## `pygraft kg`

Generate a Knowledge Graph from an existing schema.


Works with both workflows:
  - After &#x27;pygraft extract&#x27; (ontology-based)
  - After &#x27;pygraft schema&#x27; (fully synthetic)


Example:
    pygraft kg pygraft.config.json

**Usage**:

```console
$ pygraft kg [OPTIONS] CONFIG
```

**Arguments**:

* `CONFIG`: Path to the configuration file.

**Options**:

* `--help`: Show this message and exit.

## `pygraft build`

Generate both schema and KG in one step (fully synthetic workflow).

**Usage**:

```console
$ pygraft build [OPTIONS] CONFIG
```

**Arguments**:

* `CONFIG`: Path to the configuration file.

**Options**:

* `--help`: Show this message and exit.

## `pygraft extract`

Extract ontology metadata into PyGraft JSON artefacts.


Analyzes an existing ontology and creates a configuration file
pre-populated with the extracted statistics.


Example:
    pygraft extract ./ontologies/my-ontology.ttl

**Usage**:

```console
$ pygraft extract [OPTIONS] ONTOLOGY
```

**Arguments**:

* `ONTOLOGY`: Path to the ontology file (.ttl, .rdf, .owl, .xml).

**Options**:

* `--help`: Show this message and exit.

## `pygraft explain`

Run reasoner to explain inconsistencies in an existing KG.


This command analyzes an existing KG file and provides detailed explanations
of any logical inconsistencies found. The schema is automatically detected
from the same directory as the KG.


Examples:
    pygraft explain ./output_pygraft/my-project/kg.ttl
    pygraft explain ./output_pygraft/my-project/kg.ttl --reasoner hermit
    pygraft explain ./output_pygraft/my-project/kg.ttl --reasoner both

**Usage**:

```console
$ pygraft explain [OPTIONS] KG_PATH
```

**Arguments**:

* `KG_PATH`: Path to the knowledge graph file (kg.ttl, kg.rdf, or kg.nt).

**Options**:

* `-r, --reasoner TEXT`: Which reasoner(s) to use: hermit (fast, no explanation), pellet (detailed explanations), both (hermit first, then pellet if inconsistent).  [default: pellet]
* `--help`: Show this message and exit.
