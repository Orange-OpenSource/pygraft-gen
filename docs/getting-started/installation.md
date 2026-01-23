---
title: Installation
description: Install PyGraft-gen with pip, uv, or poetry.
---

# Installation

Install PyGraft-gen using pip, [uv](https://docs.astral.sh/uv/){target="_blank" rel="noopener"} or [poetry](https://python-poetry.org/){target="_blank" rel="noopener"}.

!!! info "Prerequisites"
    - Python 3.10 or higher
    - Java (optional)

## Installation

This installs PyGraft-gen as a regular package in your current environment.


=== "pip"
    ```bash
    # From PyPI (recommended)
    pip install pygraft-gen

    # From GitHub (latest from main branch)
    pip install git+https://github.com/Orange-OpenSource/pygraft-gen.git

    # Verify installation
    pygraft --version
    ```

=== "uv"
    ```bash
    # From PyPI (recommended)
    uv add pygraft-gen

    # From GitHub (latest from main branch)
    uv pip install git+https://github.com/Orange-OpenSource/pygraft-gen.git

    # Verify installation
    pygraft --version
    ```

=== "poetry"
    ```bash
    # From PyPI (recommended)
    poetry add pygraft-gen

    # From GitHub (latest from main branch)
    poetry add git+https://github.com/Orange-OpenSource/pygraft-gen.git

    # Verify installation
    pygraft --version
    ```

## Java (Optional)

Java is required only for **consistency checking**, which uses the [Owlready2](https://pypi.org/project/owlready2/){target="_blank" rel="noopener"} library to run the [HermiT](http://www.hermit-reasoner.com/){target="_blank" rel="noopener"} and [Pellet](https://github.com/stardog-union/pellet){target="_blank" rel="noopener"} reasoners. Enable this in your config with `check_kg_consistency: true`

!!! tip "Learn more"
    See [Configuration Reference](../reference/files/config.md/#kg)

**Install Java:**

- **[Eclipse Temurin](https://adoptium.net/){target="_blank" rel="noopener"}** - Free and open-source OpenJDK (recommended)
- **[Oracle JDK](https://www.oracle.com/java/technologies/downloads/){target="_blank" rel="noopener"}** - Free for development/personal use (commercial licensing applies for production)

!!! info "Operating System"
    - **Linux/macOS**: Owlready2 automatically detects Java
    - **Windows**: You may need to manually configure the Java path and add it to your system `PATH`

**Verify Java:**
```bash
java --version
```

---

## Next Steps

<div class="grid" markdown>

- :fontawesome-solid-graduation-cap: **[Background](fundamentals.md)** – Understand KGs and ontologies
- :fontawesome-solid-rocket: **[Quickstart](quickstart.md)** – Generate your first KG in 5 minutes

</div>
