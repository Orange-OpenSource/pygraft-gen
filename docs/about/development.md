---
title: Development Guide
description: Developer setup for PyGraft-gen — installation, commit conventions, code standards, and documentation.
---

# Development Guide

Technical guide for contributing code to PyGraft-gen.

**On this page:**

- [Development Setup](#development-setup) - Clone and install
- [Commit Convention](#commit-convention) - How to write commit messages
- [Code Standards](#code-standards) - Style guides and conventions
- [Documentation](#documentation) - Building and maintaining docs


---

## :fontawesome-solid-download: Development Setup

Clone the repository and install in editable mode with development dependencies:

=== "uv (recommended)"
    ```bash
    # Clone the project 
    git clone https://github.com/Orange-OpenSource/pygraft-gen.git
    cd pygraft-gen
    
    # Create .venv & activate project
    uv sync --group dev --group docs
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

=== "pip"
    ```bash
    # Clone the project 
    git clone https://github.com/Orange-OpenSource/pygraft-gen.git
    cd pygraft-gen
    
    # Create .venv & activate project
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -e ".[dev,docs]"
    ```

=== "poetry"
    ```bash
    # Clone the project 
    git clone https://github.com/Orange-OpenSource/pygraft-gen.git
    cd pygraft-gen
    
    # Create .venv & activate project
    poetry install --with dev --with docs
    poetry run pygraft --version
    ```

!!! tip "Editable Install"
    Poetry and uv install in editable mode by default. With pip, use the `-e` flag. Editable mode means source code changes are immediately reflected without reinstalling.

---

## :fontawesome-solid-code-commit: Commit Convention

Commits follow this format: `<emoji> <type>(<scope>): <subject>`

- `<emoji>` is optional

=== "Template"
    ```text
    <emoji> <type>(<scope>): <short imperative summary>
            ^----^  ^----^   ^------------------------^
            |       |        |
            |       |        +-> Subject: Summary in present tense (WHAT & WHY, not HOW)
            |       +----------> Scope: optional, specific module or area changed
            +------------------> Type: feat, fix, docs, style, refactor, perf, test, chore, etc.
    
    Signed-off-by: git.user.name <git.user.email>
    ```

=== "Example"
    ```text
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

!!! tip "See more"
    - [Conventional Commits](https://www.conventionalcommits.org/){target="_blank" rel="noopener"}
    - [Semantic Commit Messages](https://seesparkbox.com/foundry/semantic_commit_messages){target="_blank" rel="noopener"}


## :fontawesome-solid-code: Code Standards

Our code quality is enforced by:

- **[Ruff](https://github.com/astral-sh/ruff){target="_blank" rel="noopener"}** - Formatting and linting
- **[Pyright](https://github.com/microsoft/pyright){target="_blank" rel="noopener"}** / **[Basedpyright](https://docs.basedpyright.com/){target="_blank" rel="noopener"}** - Type checking
- **[Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings){target="_blank" rel="noopener"}** - Documentation

**Example:**
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

## :fontawesome-solid-book: Documentation

Our documentation uses [MkDocs](https://www.mkdocs.org/){target="_blank" rel="noopener"} with the [Material theme](https://squidfunk.github.io/mkdocs-material/){target="_blank" rel="noopener"}.

**Start local development server:**
```bash
mkdocs serve --livereload
```

Open `http://127.0.0.1:8000` in your browser. Changes auto-reload (refresh manually if needed).

!!! info "Deployment"
    Documentation is automatically built and deployed via GitHub Actions. No manual build needed!

!!! warning "CLI Documentation"
    After modifying `pygraft/cli.py`, regenerate the CLI reference:
    ```bash
    typer pygraft.cli utils docs --name pygraft --output docs/reference/cli.md
    ```
    **Don't forget to commit the updated `cli.md` file!**

!!! tip "Writing Style"
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
    
    **Callouts:** [Supported types from Material theme](https://squidfunk.github.io/mkdocs-material/reference/admonitions/#supported-types){target="_blank" rel="noopener"}
    ```markdown
    !!! note "Optional Title"
        Your content here
    ```

---

## What's Next

- :fontawesome-solid-code-pull-request: **[Contributing](contributing.md)** &mdash; How to contribute
