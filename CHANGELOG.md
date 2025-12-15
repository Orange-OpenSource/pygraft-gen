# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project tries to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added & Not committed yet

- Installed dev dependency [prek](https://github.com/j178/prek) & used in conjunction with local but uncommitted `.pre-commit-config.yaml`. The goal is to enhance the robustness of my changes while not troubling the 2nd dev of my team if they want to do stuff. **THIS SHALL BE ADDED ONLY AT THE END OF THE DEV PHASE ONCE EVRY THING IS CLEARED & OK**



## [0.0.8] Ontology Extraction Feature

### Breaking Changes

- Removed `create_json_template` and `create_yaml_template` in favor of a single public `create_config()` API.
- Configuration template filenames are now fixed to `pygraft_config.json` or `pygraft_config.yml` and can no longer be customized.

### Added

- Introduced `namespaces.py`, providing full namespace extraction for RDF ontologies (prefixes, base IRI, ontology metadata, undeclared namespaces).
- Added `extraction.py`, implementing the namespace extraction runner and serving as the entry point for the full ontology extraction pipeline
- Added `queries.py`, providing a centralized loader for SPARQL query resources used during ontology extraction.
- Added `classes.py`, implementing full class extraction and generating `class_info.json`.
- Added `relations.py`, implementing full relations extraction and generating `relation_info.json`.
- Added SPARQL query resources for ontology extraction:
  - **Classes**:
    - `class2disjoints.rq`
    - `class2disjoints_extended.rq`
    - `classes.rq`
    - `direct_class2subclasses.rq`
    - `transitive_class2subclasses.rq`
  - **Relations**:
    - `relations.rq`
    - `rel2patterns.rq`
    - `inverseof_relations.rq`
    - `subrelations.rq`
    - `rel2dom_rel2range.rq`
    - `rel2disjoints.rq`
    - `rel2disjoints_extended.rq`

### Changed

- Updated internal callers and the Typer CLI `init` command to rely exclusively on the new `create_config()` API.
- **class_info.json** (`direct_class2superclasses`): Switched to a list-based structure to support OWL multi-inheritance.
- **relation_info.json** (`rel2superrel`): Switched to a list-based structure to prepare relation hierarchies for OWL multi-parent models while preserving current output behavior.


### Fixed

- Fixed KG serialization issues where classes or relations written as CURIEs (e.g. `bot:Site`) produced broken IRIs and unstable `nsX` prefixes; this happened because all identifiers were blindly placed under the internal base namespace, and is now resolved by expanding CURIEs via `namespaces_info.json` when available while keeping the legacy behavior as a fallback. (REF: 7e97bf41)

- Fixed cases where KG generation would almost stop (sometimes producing only a few triples) for ontologies with multiple domain and/or range classes on relations; this was caused by the legacy PyGraft behavior treating domain/range as single values instead of conjunctive constraints, and is now resolved by sampling entities that satisfy all required classes, allowing large KGs to be generated reliably again. (REF: c1e71845)

- Fixed excessive rejection sampling caused by relations that could not be instantiated under conjunctive domain/range constraints in the current generation run; relations with empty head or tail candidate pools are now detected after entity typing and explicitly disabled (sampling weight = 0), preventing wasted attempts and premature termination when `relation_usage_uniformity` is high. (REF: d35920b1)

- Fixed inverse domain/range disjointness filtering in KG generation after the schema upgrade to list-based `rel2dom` / `rel2range`; the inverse-conflict cleanup step was still reading legacy single-value mappings, effectively becoming a no-op and allowing inconsistent inverse triples to survive. The filter now correctly enforces disjointness against all required domain and range classes. (REF: f6442411)

- Fixed inference-based oversampling to fully support list-based `rel2superrel` by allowing relations to have multiple direct super-relations instead of incorrectly assuming a single parent, ensuring subproperty inference remains compatible with OWL multi-parent hierarchies. (REF: f6442411)


## [0.0.7] CLI Modernization

Switched from the argparse CLI to an upgraded [Typer](https://typer.tiangolo.com/) CLI.

### Added

- Added a new Typer-based CLI implementation (`cli.py`) providing a modern, more ergonomic alternative to the legacy argparse CLI.
- Introduced a structured subcommand-based CLI: `help`, `init`, `schema`, `kg`, `build`
- Added user-facing outputs for the subcommands, we do not rely on log-level information as default but as optional.

### Changed

- `create_json_template` and `create_yaml_template` now return the created file path for easier API use & CLI echo.
- Replaced old `--log` flag with modern `-l` / `--log-level` option, including improved help text.
- `reasoner()` now returns a boolean consistency flag instead of relying on an exception path, making HermiT results explicit in the API while preserving existing logging behavior.
- `generate_schema()` now returns a `(schema_path, is_consistent)` tuple so callers (including the CLI) can access both the output file location and the HermiT consistency result.
- Updated the CLI header. Replaced text2art banner with a clean Rich-based rule.

### Removed

- Removed old argparse `cli.py`

## [0.0.6] - Subgraph matching patterns and tools

TODO: To be added

## [0.0.5] - Legacy Code Modernization

"Small overview of the global thing here but only when the full changelog is done, do not write here for now"

### Improvements

- Introduced a unified RNG strategy across all generators (classes, relations, KG).  
  Each generator now uses its own deterministic RNG when a seed is provided, ensuring reproducibility in tests and experiments while keeping default runs fully stochastic.

- Standardized class and relation metadata: direct mappings (e.g., immediate subclasses) and transitive mappings (full hierarchy) are now computed and validated separately.  
  This prevents mixed structures and ensures consistent reasoning and KG generation.

- Made domain, range, and inverse-of compatibility logic consistent across all generators.  
  The new approach eliminates order-dependent behavior, reduces hidden edge cases, and ensures that constraints are applied uniformly.

- Reworked the triple generation pipeline around explicit, well-defined phases.  
  Sampling, validation, inference-based augmentation, and constraint filtering now run through structured helpers, improving readability, debuggability, and long-term maintainability.

### Changed

- Migrated the project from a flat layout to a modern `src/` directory structure
- Modules were reorganized into dedicated packages based on their roles (e.g. `generators/, utils/, resources/`)
- The former `template.{json/yml}` configuration files were renamed to `pygraft_config.{json/yml}` to provide clearer meaning and better reflect their role as the main PyGraft configuration sources.
- Improved the CLI implementation while keeping the existing behavior and flags intact. The argument parsing was cleaned up, help text made clearer, and config file validation made more robust.

- Major refactor of the core generator modules (classes, relations, KG instances) to adopt a clearer, more modular architecture across the entire generation pipeline.  
  The new design introduces explicit configuration dataclasses, deterministic RNG handling when needed (testing/dev purposes only), improved invariants, and well-defined public entry points.

- Renamed several internal KG generator helpers to make their purpose clearer, especially previously vague names.
  For example, `_pipeline` is now `_run_entity_and_triple_pipeline`, `_check_dom_range` is now `_filter_domain_range_conflicts`, `_procedure_1` / `_procedure_2` became `_filter_domain_disjoint_conflicts` / `_filter_inverse_domain_range_disjoint_conflicts`, and `_generate_rel2inverse` is now `_build_rel_inverse_map`.
  This makes the constraint and triple-generation flow easier to follow while keeping the external behavior and outputs unchanged.

- Reworked the user configuration format to explicitly group parameters into `general`, `classes`, `relations`, and `kg` sections.
  This improves clarity around which settings affect each generation stage, reduces ambiguity between global and local parameters, and makes the configuration surface more intuitive and discoverable.

- Separated schema/KG generation from HermiT reasoning for clearer responsibilities, and updated KG serialization so instance files contain only triples (no embedded ontology).

- Centralized all output handling through a new `resolve_project_folder()` API and migrated every generator from `output/` to the explicit `pygraft_output/` root, with optional Python-only `output_root` overrides for fully custom storage paths.

- Standardized logging throughout PyGraft with clearer INFO milestones, DEBUG-level internal logs, and consistent absolute paths for all file operations.

### Fixed

- Corrected inverse range–disjointness filtering, which previously evaluated disjointness on the head of a triple but removed triples based on the tail.  
  This mismatch caused both false removals and missed violations. The updated `_filter_inverse_domain_range_disjoint_conflicts()` now applies head-based validation consistently, matching the intended semantics.

- Fixed phantom-layer sampling in `_assign_most_specific_classes()`, where the generator sometimes sampled class depths beyond the actual hierarchy.  
  The helper now restricts sampling strictly to valid depth ranges, preventing invalid type assignments and crashes in shallow ontologies.

- Replaced order-dependent inverse mapping with a canonical, symmetric reconstruction of inverse pairs.  
  `_build_rel_inverse_map()` now derives inverse relationships deterministically, independent of JSON ordering or declaration order.

- Restored and corrected oversampling logic, which previously existed but did not always execute due to inconsistent trigger conditions.  
  `_oversample_triples_via_inference()` now runs reliably in both completion and augmentation modes, and applies inference rules as intended.

- Corrected functional and inverse-functional consistency checks by fixing tuple slicing and index alignment issues.  
  `_triple_is_consistent()` now compares the correct subject/object components and enforces OWL functional constraints predictably.

- Fixed domain and range disjointness checks, which previously mixed two incompatible definitions of “disjoint” (direct extended disjoint sets vs. disjointness inherited through superclasses).  
  All checks now consistently rely on `_is_entity_compatible_with_class()`, which expands the expected domain/range through all transitive superclasses before applying disjointness rules.  
  This produces accurate and predictable conflict detection across both shallow and deeply nested hierarchies.

- Ensured HermiT reasoning works reliably across all supported output formats by normalizing non-XML schema/KG inputs into temporary RDF/XML files.  
  `_run_hermit_reasoner()` now handles RDF/XML conversion automatically and skips unnecessary conversions when inputs are already XML, ensuring robust and efficient reasoning.

### Added

- Introduced a new `types.py` module providing centralized type definitions for configuration files and all JSON artefacts used by PyGraft (e.g. `pygraft_config.{json/yml}`, `class_info`, `relation_info`, `kg_info`).
- Added a `-V/--version` CLI flag to display the installed `pygraft` version.
- Added a `--log` CLI option to control logging level by name or numeric value (default: `INFO`).
- Added list-based serialization for `rel2dom` and `rel2range` in `RelationInfo` to enable future multi-valued domains/ranges while preserving backward-compatible normalization.
- Added a unified configuration-validation pipeline built around three stages: (1) structural key checks, (2) strict scalar type validation through `_validate_section_types`, (3) semantic validation for schema, relations, and KG. This ensures all fields use the correct primitive types, prevents common input mistakes (like quoted booleans or numbers), and replaces scattered type checks with a single, consistent mechanism.

- Introduced centralized builder functions in `types.py` for `class_info`, `relation_info`, and `kg_info`, and integrated them across all generator modules. This removes duplicated dictionary assembly, enforces a single canonical JSON structure, and greatly simplifies future schema evolution.

### Removed

- The former `utils.py` module was broken down into more focused components all organized under the `utils/` package : `reasoning.py` (HermiT integration), `cli.py`, `templates.py`, `paths.py`, and `config.py`
- Removed the redundant `generate()` public API, which duplicated the logic of `generate_schema()` and `generate_kg()`.  
  The CLI now handles the combined workflow (`--gen generate`) explicitly, and the ASCII header output was moved fully into the CLI to avoid duplicate prints.

## [0.0.4] - PEP 621 Migration & Tooling Update (2025-11-27)

### Added

- Introduced modern development tooling:
    - [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
    - [Pyright](https://github.com/microsoft/pyright)/[Basedpyright](https://docs.basedpyright.com/latest/) for static type checking
    - [Codespell](https://github.com/codespell-project/codespell) for spell-checking
    - [EditorConfig](https://editorconfig.org/), `.python-version`, and a consistent project-level configuration setup
    - Added initial `CHANGELOG.md` and updated `CONTRIBUTING.md` for a clearer workflow.

### Changed

- Migrated the project to a modern PEP 621 build system using Hatchling.
    - Switched to dynamic versioning via git tags using `hatch-vcs` (e.g. `v0.0.4` → `0.0.4`, dev installs show `0.0.5.dev0+...`).
- Renamed `pygraft/main.py` to `pygraft/cli.py` and updated the console entrypoint i.g. `pygraft --help` is now the official way to invoke the tool.
- Raised minimum Python version to 3.10 (3.8 EOL, 3.9 close behind).

### Removed

- Deleted legacy `setup.py` and `setup.cfg` files.

## [0.0.3] - 2023-09-08

### Added

- Public PyPI release `pygraft==0.0.3`.
- Core pipeline to generate:
    - a schema only,
    - a KG only,
    - or both schema and KG in a single pipeline.
- Support for an extended set of RDFS and OWL constructs in generated schemas and KGs, enabling fine grained, standards compliant modeling.
- Consistency checking of generated schemas and KGs via the HermiT DL reasoner.
- YAML based configuration with `create_yaml_template()` to generate a template config file with all tunable parameters.
- High level Python API:
    - `generate_schema(path)`
    - `generate_kg(path)`
    - `generate(path)` for the full pipeline
    - public functions exposed through `pygraft.__all__`.
- Basic CLI support for running the generation pipeline from the command line.
- Sphinx based documentation and Read the Docs configuration, covering installation, overview, parameters, and first steps.

### Changed

- Iterated on README and online documentation to better describe upcoming features and usage examples.

## [0.0.2] - 2023-09-07

### Fixed

- Follow up release to fix minor packaging and documentation issues discovered after `0.0.1` (metadata and README tweaks).

## [0.0.1] - 2023-09-07

### Added

- First public release of PyGraft as a Python package on PyPI.
- Initial implementation of the configurable schema and KG generator, including:
    - schema generation,
    - KG generation,
    - and combined pipeline with consistency checks.
- Initial documentation and README describing goals, features, and basic usage.
