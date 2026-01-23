# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project tries to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

*Changes committed but not yet released will appear here.*

> [!NOTE]  
> **Roadmap**
>
> Planned features are tracked in [CONTRIBUTING](CONTRIBUTING.md/#desirable-features), not here. Unreleased is for completed work awaiting release.

---

## v0.0.11 (2026-01-23)

[:octocat: GitHub release](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.11)

**Documentation Overhaul & CLI Restructuring**

*Migrated documentation from Sphinx/Furo to MkDocs Material, restructured CLI into modular architecture, and introduced standalone consistency explanation*

> [!WARNING]  
> **Breaking Changes**
>
> - Configuration files must now nest `classes` and `relations` under a new `schema` key
> - Removed `explain_inconsistency` parameter from `generate_kg()`
> - Removed `--explain` flag from `kg` and `build` commands
> - Removed `reasoner()` orchestrator from `pygraft.utils.reasoning`

### Added

- **Standalone explain command**: New `pygraft explain` CLI command and `explain_kg()` API for analyzing KG inconsistencies with flexible reasoner selection (`--reasoner hermit|pellet|both`)
- **Separated reasoner functions**: Split monolithic `reasoner()` into `reasoner_hermit()` and `reasoner_pellet()` for clear separation of concerns

### Changed

- **Migrated documentation to MkDocs**: Replaced Sphinx/Furo with MkDocs Material theme, rewriting all documentation from scratch
- **Restructured CLI into modular subpackage**: One file per command (`init`, `schema`, `kg`, `build`, `extract`, `explain`) with shared utilities in `formatting.py`, `validators.py`, and `extract_helper.py`
- **Nested schema configuration**: Grouped `classes` and `relations` under new `schema` section to better reflect logical separation from KG generation parameters
- **Modern CLI type hints**: Migrated all CLI parameters to `Annotated` style with improved docstrings

### Fixed

- **YAML config support in KG generation**: Replaced direct `json.load()` with `load_config()` to properly handle both JSON and YAML configuration files
- **Auto project_name for KG mode**: Now only selects synthetic schemas (timestamped folders) when using `auto`, preventing accidental KG generation against extracted ontologies
- **Config cleanup on extraction**: `pygraft extract` now produces canonical config structure, removing extra keys and restoring missing defaults while preserving KG section values


## v0.0.10 (2026-01-08)

[:octocat: GitHub release](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.10)

**KG Generation Optimization**

*Major performance overhaul enabling practical large-scale generation of Knowledge Graphs with millions of entities and tens of millions of triples*

### Changed

- **Restructured KG generator into specialized modules** with clear responsibilities: types, structures, config, schema loading, entity creation, batch generation, and output serialization
- **Switched to integer-based internal processing**: All operations use lightweight integer IDs instead of strings, dramatically reducing memory overhead. String identifiers only used during final RDF serialization
- **Moved from post-generation cleanup to inline validation**: Triples validated as generated using pre-computed constraint data, eliminating expensive multi-pass graph scans
- **Implemented batch sampling with vectorized operations**: Generate and validate thousands of triples per iteration using NumPy arrays with pre-computed constraint caches
- **Introduced two-phase constraint filtering**: Fast vectorized phase eliminates invalid triples, deep semantic phase only processes remaining candidates
- **Simplified state management**: Generation state organized as structured attributes rather than nested function parameters

### Fixed

- **Eliminated symmetric relation bottleneck**: Replaced expensive data structure rebuilds with incremental constant-time tracking
- **Resolved generation stalls and infinite loops**: Added stall detection (drops unproductive relations), timeout protection, and adaptive oversampling for constrained properties
- **Improved memory efficiency**: Compact indexed structures, sparse arrays for entity pools, explicit cleanup after serialization, single-pass domain/range computation
- **Optimized functional property validation**: Constant-time set lookups replace full triple scans

### Added

- **Intelligent generation heuristics**: Weighted relation sampling, entity freshness bias, and fast generation mode (seed + replication for very large targets)
- **Comprehensive constraint caching**: All schema constraints analyzed once at startup for instant lookup during generation
- **Structured data containers**: Clear separation of schema metadata, constraint caches, entity state, and generation progress

### Performance Impact

| Aspect              | Before                   | After              | Improvement           |
|---------------------|--------------------------|--------------------|-----------------------|
| Memory usage        | Heavy string duplication | Integer IDs        | ~60% reduction        |
| Domain/range lookup | Per-sample recompute     | Pre-cached         | ~1000x faster         |
| Functional checks   | Scan all triples         | Set lookup         | ~10000x faster        |
| Validation          | Multiple post-gen passes | Inline             | 5x fewer scans        |
| Reliability         | Could hang/stall         | Robust termination | No infinite loops     |
| Scale               | Often impractical        | Reliable           | Million+ entities now |

*Actual speedup varies by ontology complexity. Highly constrained schemas (many functional properties, extensive disjointness) see different gains than simpler ontologies, but all scenarios are substantially faster and complete reliably.*

## v0.0.9 (2026-01-08)

[:octocat: GitHub release](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.9)

> [!TIP]  
> **Performance Restored**
>
> This release fixes the critical performance regression introduced in [v0.0.8](#v008-2025-12-15)

**Ontology Extraction (Performance Fix)**

*Fixed single-source-of-truth refactor that caused 20-30x performance regression in ontology extraction*

### Changed

- Introduced `relations_seed.rq` as canonical source of truth for object property universe
- Implemented `@RELATIONS_SEED` marker injection mechanism across all relation SPARQL queries
- Removed working-graph materialization approach that caused slowdown

### Fixed

- Restored extraction performance from ~10 minutes back to ~20-40 seconds
- Ensured all relation extractors (patterns, inverses, hierarchy, disjointness, domain/range) use consistent property universe
- Completed implementation with missing `relations.py` module updates for marker injection

**Technical context:** The earlier refactor attempted to centralize object properties via materialized working graph (membership triples + graph copy), which proved impractical. The new approach uses query-time injection for true single-source semantics without performance penalty.

## v0.0.8 (2025-12-15)

[:octocat: GitHub release](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.8)

> [!CAUTION]  
> **Critical Performance Issue - Do Not Use**
>
> This release contains a severe performance regression (20-30x slowdown) in ontology extraction that makes it impractical for use. **Please use [v0.0.9](#v009-2026-01-08) or later instead.** This entry is preserved for historical reference only.

**Ontology Extraction**

*Introduced full ontology extraction pipeline, enabling PyGraft-gen to generate KGs from real-world ontologies*

### Breaking Changes

- Removed `create_json_template()` and `create_yaml_template()` in favor of unified `create_config()` API
- Configuration template filenames now fixed as `pygraft_config.json` or `pygraft_config.yml` (no longer customizable)

### Added

- Complete ontology extraction pipeline with dedicated modules:
    - `namespaces.py` for prefix and base IRI extraction
    - `classes.py` for class hierarchy extraction (generates `class_info.json`)
    - `relations.py` for property extraction (generates `relation_info.json`)
    - `extraction.py` as the main pipeline entry point
    - `queries.py` for centralized SPARQL query loading
- SPARQL query resources for extraction (class disjointness, hierarchy, relation patterns, inverses, subproperties, domain/range, property disjointness)

### Changed

- Updated CLI `init` command to use new `create_config()` API
- **class_info.json**: `direct_class2superclasses` now uses list structure for OWL multi-inheritance support
- **relation_info.json**: `rel2superrel` now uses list structure for multi-parent property hierarchies

### Fixed

- Corrected KG serialization where CURIEs (e.g., `bot:Site`) produced broken IRIs; now properly expands identifiers via `namespaces_info.json`
- Fixed near-zero triple generation for ontologies with conjunctive domain/range constraints; now correctly samples entities satisfying all required classes
- Eliminated excessive rejection sampling by detecting and disabling relations with empty candidate pools after entity typing
- Corrected inverse domain/range disjointness filtering to work with list-based constraint structure
- Fixed inference oversampling to support multi-parent subproperty hierarchies

## v0.0.7 (2025-12-09)

**CLI Modernization**

*Migrated from argparse to [Typer](https://typer.tiangolo.com/) for improved ergonomics and maintainability*

### Added

- Typer-based CLI with structured subcommands: `help`, `init`, `schema`, `kg`, `build`
- User-facing output messages independent of logging levels (logging now optional via `-l/--log-level`)

### Changed

- Template creation functions (`create_json_template`, `create_yaml_template`) now return file paths for easier API composition
- `reasoner()` returns explicit boolean consistency flag instead of exception-based control flow
- `generate_schema()` returns `(schema_path, is_consistent)` tuple for direct access to both outputs
- Replaced text2art banner with clean Rich-based rule in CLI header
- Improved `-l/--log-level` flag with clearer help text

### Removed

- Legacy argparse CLI implementation

## v0.0.6 (2025-12-08)

**Subgraph matching patterns and tools**

## v0.0.5 (2025-12-08)

**Legacy Code Modernization**

*Major architectural refactor improving maintainability, reproducibility, and standards compliance*

### Added

- Unified RNG strategy across all generators, enabling deterministic reproduction when seeded while keeping default runs stochastic
- Type system via `types.py` providing centralized TypedDict definitions for all configuration files and JSON artifacts
- CLI enhancements: `-V/--version` flag and `--log-level` option for controlling output verbosity
- Comprehensive configuration validation pipeline with structural checks, strict type validation, and semantic constraints
- Centralized builder functions for `class_info`, `relation_info`, and `kg_info` JSON outputs

### Changed

- Migrated from flat layout to modern `src/` directory structure with organized packages (`generators/`, `utils/`, `resources/`)
- Renamed `template.{json/yml}` to `pygraft_config.{json/yml}` for clearer purpose
- Reorganized configuration format into explicit `general`, `classes`, `relations`, and `kg` sections
- Refactored core generators with explicit configuration dataclasses, improved invariants, and well-defined entry points
- Improved CLI implementation with clearer help text and more robust validation
- Separated schema/KG generation from HermiT reasoning; KG files now contain only instance triples
- Standardized output handling with `pygraft_output/` as default root and optional custom paths
- Enhanced logging with consistent INFO milestones, DEBUG internals, and absolute paths
- Improved internal naming across constraint validation and triple generation helpers

### Fixed

- Corrected inverse range-disjointness filtering that previously applied head validation but removed triples based on tail
- Fixed phantom-layer sampling in class assignment that could sample beyond actual hierarchy depth
- Replaced order-dependent inverse mapping with canonical symmetric reconstruction
- Restored and corrected oversampling logic for inference-based triple augmentation
- Fixed functional and inverse-functional constraint checks with proper tuple indexing
- Unified domain/range disjointness validation to consistently use transitive superclass expansion
- Ensured HermiT reasoning works across all RDF formats via automatic RDF/XML conversion

### Removed

- Split `utils.py` into focused modules: `reasoning.py`, `cli.py`, `templates.py`, `paths.py`, `config.py`
- Removed redundant `generate()` API; combined workflow now handled explicitly via CLI

## v0.0.4 (2025-11-27)

**PEP 621 Migration & Tooling Update**

### Added

- Modern development tooling stack:
    - [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
    - [Pyright](https://github.com/microsoft/pyright)/[Basedpyright](https://docs.basedpyright.com/latest/) for static type checking
    - [Codespell](https://github.com/codespell-project/codespull) for spell-checking
    - Project-wide configuration via [EditorConfig](https://editorconfig.org/) and `.python-version`
- Initial `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/) format
- Updated `CONTRIBUTING.md` with clearer development workflow

### Changed

- Migrated to PEP 621 build system using Hatchling
- Switched to dynamic versioning via git tags using `hatch-vcs` (e.g., `v0.0.4` &rarr; `0.0.4`, dev installs show `0.0.5.dev0+...`)
- Renamed `pygraft/main.py` to `pygraft/cli.py` and updated console entrypoint accordingly
- Raised minimum Python version to 3.10 (Python 3.8 reached EOL, 3.9 approaching EOL)

### Removed

- Legacy `setup.py` and `setup.cfg` build configuration files

## v0.0.3 (2023-09-08)

*Derived from [PyGraft v0.0.3 PyPI](https://pypi.org/project/pygraft/0.0.3/) release*

### Added

- Public PyPI release as `pygraft==0.0.3`
- Core generation pipeline with three execution modes: schema-only, KG-only, or combined schema + KG
- Extended RDFS and OWL construct support for standards-compliant modeling with fine-grained control
- Consistency checking of generated schemas and KGs via HermiT DL reasoner
- YAML-based configuration with `create_yaml_template()` function to generate template config files
- High-level Python API with `generate_schema()`, `generate_kg()`, and `generate()` functions (exposed via `__all__`)
- CLI support for running generation pipeline from command line
- Sphinx-based documentation with Read the Docs integration covering installation, parameters, and quickstart

### Changed

- Improved README and documentation with better feature descriptions and usage examples

## v0.0.2 (2023-09-07)

*Derived from [PyGraft v0.0.2 PyPI](https://pypi.org/project/pygraft/0.0.2/) release*

### Fixed

- Packaging metadata and README formatting issues from v0.0.1

## v0.0.1 (2023-09-07)

*Derived from [PyGraft v0.0.1 PyPI](https://pypi.org/project/pygraft/0.0.1/) release*

### Added

- Initial PyPI release of PyGraft
- Configurable schema and KG generator with schema-only, KG-only, and combined pipeline modes with consistency checking
- Initial documentation and README with project goals and basic usage
