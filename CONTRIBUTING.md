# Contributing to PyGraft-gen

We would love for you to contribute to **PyGraft-gen** and help make it even better than it is today! Please contribute to this repository if any of the following is true:

- You have expertise in *Knowledge Graphs*, *synthetic datasets*, or *graph generation*
- You have expertise in *stochastic generation*, *rule-based generation*, or *subgraph matching techniques*
- You want to *challenge AI pipelines* with domain-related Knowledge Graphs and domain-specific graph patterns

> [!NOTE]  
> **For Full Documentation**
>
> Visit our [documentation site](https://orange-opensource.github.io/pygraft-gen/) for comprehensive guides, examples, and API reference.

## How to Contribute

We welcome community contributions, whether documentation, refactoring, tests, or new features.

You can contribute to PyGraft-gen in the following ways:

- Ask questions or share ideas in [Discussions](https://github.com/Orange-OpenSource/pygraft-gen/discussions)
- Help other users by answering questions in [Discussions](https://github.com/Orange-OpenSource/pygraft-gen/discussions) or commenting on [open issues](https://github.com/Orange-OpenSource/pygraft-gen/issues)
- File a [bug report](https://github.com/Orange-OpenSource/pygraft-gen/issues/new?template=bug_report.md)
- File a [feature request](https://github.com/Orange-OpenSource/pygraft-gen/issues/new?template=feature_request.md)
- Help implementing unit tests
- Help refactoring code and ensuring best practices are respected
- Generate clean docstrings for the existing code base
- Implement a new feature (see [Desirable Features](#desirable-features))

## Desirable Features

Want to contribute but not sure where to start? Here are features we're working toward, organized by priority.

> [!NOTE]  
> **Current Focus**
>
> After ontology extraction ([v0.0.8](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.8)) and KG optimization ([v0.0.10](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.10)), we're prioritizing infrastructure (testing, CI/CD, code quality) before expanding object property support.

### High Priority

- [x] **Support for any input ontology** *(inherited from PyGraft, [v0.0.8](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.8))* - Generate KGs from real-world ontologies, not just PyGraft-generated schemas
- [x] **Large-scale KG generation** *([v0.0.10](https://github.com/Orange-OpenSource/pygraft-gen/releases/tag/v0.0.10))* - Optimized KG generator architecture enabling millions of entities and tens of millions of triples
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

> [!NOTE]  
> **Not Currently Prioritized**
>
> Datatype properties (literal-valued attributes like strings, integers, dates) will be addressed after object property support is complete.

Interested in tackling one of these? Start a [Discussion](https://github.com/Orange-OpenSource/pygraft-gen/discussions) to discuss your approach!

---

## Development Setup

Clone the repository and install in editable mode with development dependencies:

```bash
# Clone the project
git clone https://github.com/Orange-OpenSource/pygraft-gen.git
cd pygraft-gen

# ---------- Using uv (recommended) ----------
uv sync --group dev --group docs
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# ---------- Using pip ----------
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"

# ---------- Using poetry ----------
poetry install --with dev --with docs
poetry run pygraft --version
```

> [!TIP]  
> **Editable Install**
>
> Poetry and uv install in editable mode by default. With pip, use the `-e` flag.    
> Editable mode means source code changes are immediately reflected without reinstalling.

---

## Commit Convention

Commits follow this format: `<emoji> <type>(<scope>): <subject>`

- `<emoji>` is optional

### Template

```text
# ------------------------------------------- TEMPLATE ------------------------------------------- #
<emoji> <type>(<scope>): <short imperative summary>
        ^----^  ^----^   ^------------------------^
        |       |        |
        |       |        +-> Subject: Summary in present tense (WHAT & WHY, not HOW)
        |       +----------> Scope: optional, specific module or area changed
        +------------------> Type: feat, fix, docs, style, refactor, perf, test, chore, etc.

Signed-off-by: git.user.name <git.user.email>

# ------------------------------------------- EXAMPLE ------------------------------------------- #
:sparkles: feat(core): add configuration file support

Implement support for reading settings from a config file.
- Enables environment-specific configuration
- Simplifies deployment and local setup

Signed-off-by: Mad Max <mad.max@example.com>
```

### Commit Types

| Type     | Emoji              | Description             |
|----------|--------------------|-------------------------|
| feat     | :sparkles:         | New feature             |
| fix      | :bug:              | Bug fix                 |
| docs     | :memo:             | Documentation only      |
| style    | :art:              | Formatting/linting      |
| refactor | :recycle:          | Code restructuring      |
| perf     | :zap:              | Performance improvement |
| test     | :white_check_mark: | Add/update tests        |
| chore    | :wrench:           | Maintenance/tooling     |
| build    | :package:          | Build system changes    |
| ci       | :gear:             | CI/CD configuration     |
| revert   | :back:             | Revert previous commit  |

**References:**

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Commit Messages](https://seesparkbox.com/foundry/semantic_commit_messages)

---

## Code Standards

Our code quality is enforced by:

- **[Ruff](https://github.com/astral-sh/ruff)** - Formatting and linting
- **[Pyright](https://github.com/microsoft/pyright)** / **[Basedpyright](https://docs.basedpyright.com/)** - Type checking
- **[Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)** - Documentation

### Example

```python
def top_k(items: list[str], k: int) -> list[str]:
    """Return the first k items.
  
    Args:
        items: Input list.
        k: Number of items to return. Must be >= 0.
  
    Returns:
        The first k items, or all if len(items) < k.
  
    Raises:
        ValueError: If k is negative.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    return items[:k]
```

---

## Documentation

Our documentation uses [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/).

### Start local development server

```bash
mkdocs serve --livereload
```

Open `http://127.0.0.1:8000` in your browser. Changes auto-reload (refresh manually if needed).

> [!NOTE]  
> **Deployment**
>
> Documentation is automatically built and deployed via GitHub Actions. No manual build needed!

### CLI Documentation

> [!WARNING]  
> **After modifying `pygraft/cli.py`**
>
> Regenerate the CLI reference:
> ```bash
> typer pygraft.cli utils docs --name pygraft --output docs/reference/cli.md
> ```
> Don't forget to commit the updated `cli.md` file!

### Writing Style

**Characters:** Use ASCII or HTML entities only
```markdown
&rarr;  Correct
→       Wrong
```

**Math:** Use LaTeX notation
```markdown
Inline: $O(n)$
Block:  $$O(n)$$
```

**External links:** Always open in new tab
```markdown
[Text](https://example.com){target="_blank" rel="noopener"}
```

**Callouts:** [Supported types from Material theme](https://squidfunk.github.io/mkdocs-material/reference/admonitions/#supported-types)
```markdown
!!! note "Optional Title"
    Your content here
```

---

## Communication

GitHub is our primary communication platform. Use [Discussions](https://github.com/Orange-OpenSource/pygraft-gen/discussions) for questions, ideas, and general support. Reserve [Issues](https://github.com/Orange-OpenSource/pygraft-gen/issues) for bug reports and feature requests, and [Pull Requests](https://github.com/Orange-OpenSource/pygraft-gen/pulls) for code contributions.

You may also contact the maintainers by email for more specific purposes and questions.

We value respectful and constructive communication. Keep discussions focused on practical problems and solutions. All interactions follow [GitHub's Code of Conduct](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct).
