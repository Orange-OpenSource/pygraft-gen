#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Reasoning utilities for PyGraft.

This module provides a thin, typed wrapper around Owlready2's HermiT
integration. It is responsible for:

- Loading an ontology from disk.
- Running the HermiT reasoner via Owlready2.
- Logging consistency results and surfacing inconsistencies as exceptions.

Design
------
- Function-oriented: no classes are introduced here on purpose.
- Internal-only: this module is used by schema and KG orchestration code,
  but is not part of the public `pygraft` API.
- Deterministic given the ontology file; no randomness is involved.
- Uses the shared logging infrastructure; no direct printing.

Performance
-----------
The dominant cost is the underlying HermiT reasoning process, which depends
on ontology size and expressivity. This wrapper adds negligible overhead
beyond:

- Ontology loading via Owlready2.
- A single call to `sync_reasoner_hermit`.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sys
from typing import Any, cast

import owlready2
from owlready2 import default_world
from rdflib import Graph as RDFGraph

logger = logging.getLogger(__name__)

# ================================================================================================ #
# Constants                                                                                        #
# ================================================================================================ #

_JAVA_HEAP_FRACTION_DEFAULT = 0.85
_JAVA_HEAP_MIN_MB_DEFAULT = 1024
_JAVA_HEAP_MAX_MB_DEFAULT: int | None = None

_JAVA_HEAP_ENVVAR_NAME = "JAVA_TOOL_OPTIONS"
_JAVA_HEAP_OOM_MARKER = "OutOfMemoryError: Java heap space"

_configured_java_heap_mb: int | None = None
_java_heap_config_source: str | None = None

# ================================================================================================ #
# Owlready2 / HermiT Type Aliases                                                                  #
# ================================================================================================ #

OntologyHandle = Any

GetOntologyCallable = Callable[[str], OntologyHandle]
get_ontology: GetOntologyCallable = cast(GetOntologyCallable, owlready2.get_ontology)

SyncReasonerCallable = Callable[..., None]
sync_reasoner_hermit: SyncReasonerCallable = cast(
    SyncReasonerCallable,
    owlready2.sync_reasoner_hermit,
)

sync_reasoner_pellet: SyncReasonerCallable = cast(
    SyncReasonerCallable,
    owlready2.sync_reasoner_pellet,
)

OwlReadyInconsistentOntologyError = owlready2.OwlReadyInconsistentOntologyError
OwlReadyJavaError = cast(type[BaseException], getattr(owlready2, "OwlReadyJavaError", RuntimeError))

# ================================================================================================ #
# Public Reasoner Functions                                                                        #
# ================================================================================================ #


def reasoner_hermit(
    *,
    schema_file: str | Path,
    kg_file: str | Path | None = None,
    infer_property_values: bool = False,
    debug: bool = False,
    keep_tmp_file: bool = False,
) -> bool:
    """Run HermiT reasoner for fast consistency checking.

    The input files are normalized to RDF/XML only when they are not already
    in RDF/XML format (e.g., ".ttl", ".nt"). In schema+KG mode, the schema and
    the KG are merged into a temporary RDF/XML ontology used solely for
    reasoning. Temporary files are deleted unless keep_tmp_file is True.

    Args:
        schema_file: Path to the schema ontology.
        kg_file: Optional path to a KG generated from the schema.
        infer_property_values: Whether to infer property values during reasoning.
        debug: Enable Owlready2 internal debugging.
        keep_tmp_file: Preserve the temporary RDF/XML file used for reasoning.

    Returns:
        True if the schema or schema+KG is consistent, False if it is inconsistent.
    """
    _configure_java_heap_best_effort()

    schema_path = Path(schema_file).resolve()
    temp_to_cleanup: Path | None = None

    if kg_file is None:
        # Schema-only: if we know it is RDF/XML (config format="xml"), no need
        # to re-serialize. Otherwise, normalize to a temporary RDF/XML file.
        if schema_path.suffix == ".rdf":
            ontology_path = schema_path
            logger.debug("Selected RDF/XML schema file for reasoning from: %s", ontology_path)
        else:
            temp_to_cleanup = _build_temp_schema_graph(schema_path)
            ontology_path = temp_to_cleanup

        resource_label = "schema"
    else:
        # Schema + KG mode
        kg_path = Path(kg_file).resolve()
        temp_to_cleanup = _build_temp_schema_kg_graph(schema_path, kg_path)
        ontology_path = temp_to_cleanup
        resource_label = "KG"

    graph: OntologyHandle = get_ontology(str(ontology_path)).load()
    logger.debug("Loaded ontology for reasoning from: %s", ontology_path)

    is_consistent = True

    try:
        sync_reasoner_hermit(
            graph,
            infer_property_values=infer_property_values,
            debug=debug,
            keep_tmp_file=keep_tmp_file,
            ignore_unsupported_datatypes=True,
        )
        logger.info("(HermiT) Consistent %s", resource_label)
    except OwlReadyInconsistentOntologyError:
        logger.info("(HermiT) Inconsistent %s", resource_label)
        is_consistent = False
    finally:
        graph.destroy()

        if temp_to_cleanup is not None and not keep_tmp_file:
            try:
                temp_to_cleanup.unlink(missing_ok=True)
                logger.debug("Deleted temporary reasoner input file: %s", temp_to_cleanup)
            except OSError:
                logger.warning("Failed to delete temporary reasoner input file: %s", temp_to_cleanup)

    return is_consistent


def reasoner_pellet(
    *,
    schema_file: str | Path,
    kg_file: str | Path | None = None,
    infer_property_values: bool = False,
    debug: bool = False,
    keep_tmp_file: bool = False,
) -> tuple[bool, str | None]:
    """Run Pellet reasoner for consistency checking with detailed explanations.

    The input files are normalized to RDF/XML only when they are not already
    in RDF/XML format (e.g., ".ttl", ".nt"). In schema+KG mode, the schema and
    the KG are merged into a temporary RDF/XML ontology used solely for
    reasoning. Temporary files are deleted unless keep_tmp_file is True.

    When the ontology is inconsistent, Pellet attempts to provide a detailed
    explanation of which axioms are causing the inconsistency.

    Args:
        schema_file: Path to the schema ontology.
        kg_file: Optional path to a KG generated from the schema.
        infer_property_values: Whether to infer property values during reasoning.
        debug: Enable Owlready2 internal debugging.
        keep_tmp_file: Preserve the temporary RDF/XML file used for reasoning.

    Returns:
        Tuple of (is_consistent, explanation):
            - is_consistent: True if consistent, False if inconsistent.
            - explanation: Human-readable explanation string if inconsistent,
              None if consistent or if explanation could not be generated.
    """
    _configure_java_heap_best_effort()

    schema_path = Path(schema_file).resolve()
    temp_to_cleanup: Path | None = None

    if kg_file is None:
        # Schema-only: if we know it is RDF/XML (config format="xml"), no need
        # to re-serialize. Otherwise, normalize to a temporary RDF/XML file.
        if schema_path.suffix == ".rdf":
            ontology_path = schema_path
            logger.debug("Selected RDF/XML schema file for reasoning from: %s", ontology_path)
        else:
            temp_to_cleanup = _build_temp_schema_graph(schema_path)
            ontology_path = temp_to_cleanup

        resource_label = "schema"
    else:
        # Schema + KG mode
        kg_path = Path(kg_file).resolve()
        temp_to_cleanup = _build_temp_schema_kg_graph(schema_path, kg_path)
        ontology_path = temp_to_cleanup
        resource_label = "KG"

    graph: OntologyHandle = get_ontology(str(ontology_path)).load()
    logger.debug("Loaded ontology for reasoning from: %s", ontology_path)

    is_consistent = True
    explanation: str | None = None

    try:
        # Run Pellet with explain mode enabled
        # Note: Pellet will raise OwlReadyInconsistentOntologyError if inconsistent
        with _suppress_owlready2_output():
            sync_reasoner_pellet(
                graph,
                infer_property_values=infer_property_values,
                debug=debug,
                keep_tmp_file=keep_tmp_file,
            )
        logger.info("(Pellet) Consistent %s", resource_label)
    except OwlReadyInconsistentOntologyError:
        logger.info("(Pellet) Inconsistent %s", resource_label)
        is_consistent = False

        # Try to extract explanation from Pellet
        explanation = _try_log_pellet_explanation(
            graph=graph,
            resource_label=resource_label,
        )
    finally:
        graph.destroy()

        if temp_to_cleanup is not None and not keep_tmp_file:
            try:
                temp_to_cleanup.unlink(missing_ok=True)
                logger.debug("Deleted temporary reasoner input file: %s", temp_to_cleanup)
            except OSError:
                logger.warning("Failed to delete temporary reasoner input file: %s", temp_to_cleanup)

    return is_consistent, explanation


# ================================================================================================ #
# Internal helpers                                                                                 #
# ================================================================================================ #


@contextmanager
def _suppress_owlready2_output() -> None:
    """Temporarily suppress stdout/stderr used by Owlready2 (Java subprocess)."""
    with Path(os.devnull).open("w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _try_log_pellet_explanation(*, graph: OntologyHandle, resource_label: str) -> str | None:
    """Run a best-effort Pellet pass and return inconsistency explanation text.

    Returns:
        A human-readable explanation (when available), otherwise None.

    Note:
        Owlready2/Pellet typically raises an exception when the ontology is
        inconsistent. In that case, the exception message often contains the
        output of "pellet explain", which we extract, log, and return.
    """
    try:
        with _suppress_owlready2_output():
            sync_reasoner_pellet(
                graph,
                infer_property_values=False,
                infer_data_property_values=False,
                debug=2,
            )
    except (OwlReadyInconsistentOntologyError, OwlReadyJavaError, RuntimeError) as exc:
        exc_text = str(exc)

        if _JAVA_HEAP_OOM_MARKER in exc_text:
            heap_msg = (
                f"-Xmx{_configured_java_heap_mb}m"
                if _configured_java_heap_mb is not None
                else "unknown (-Xmx not set by PyGraft)"
            )
            logger.warning(
                "(Pellet) Explain pass ran out of JVM heap for %s (heap=%s, source=%s). "
                "Override with %s (e.g. set %s='-Xmx16g') or raise _JAVA_HEAP_FRACTION_DEFAULT.",
                resource_label,
                heap_msg,
                _java_heap_config_source,
                _JAVA_HEAP_ENVVAR_NAME,
                _JAVA_HEAP_ENVVAR_NAME,
            )
            return None

        if (
            "Ontology is inconsistent" in exc_text
            or "pellet explain" in exc_text
            or "This is the output of `pellet explain`" in exc_text
        ):
            marker = "This is the output of `pellet explain`:"
            explanation = exc_text.split(marker, 1)[1].strip() if marker in exc_text else exc_text

            logger.info("(Pellet) Inconsistency explanation for %s:\n%s", resource_label, explanation)
            return explanation

        logger.warning("(Pellet) Explain pass failed for %s: %s", resource_label, exc)
        return None

    # Fallback if Pellet does not throw but reports inconsistent classes
    inconsistent_classes = list(default_world.inconsistent_classes())
    if inconsistent_classes:
        lines: list[str] = ["Inconsistent classes:"]
        lines.extend([f"- {cls.iri or cls.name}" for cls in inconsistent_classes])

        explanation = "\n".join(lines)
        logger.info("(Pellet) Inconsistency explanation for %s:\n%s", resource_label, explanation)
        return explanation

    return None


def _build_temp_schema_graph(schema_file: Path) -> Path:
    """Create a temporary RDF/XML file for schema-only reasoning.

    This helper uses rdflib to parse the original schema file, regardless
    of its on-disk serialization (xml, ttl, nt, ...), and re-serializes it
    as RDF/XML. Owlready2 then loads this temporary file for HermiT.

    Args:
        schema_file: Path to the schema ontology file.

    Returns:
        Path to the temporary RDF/XML file that should be given to Owlready2.
    """
    graph = RDFGraph()
    graph.parse(str(schema_file))

    tmp_path = schema_file.with_name("tmp_schema_reasoner.rdf").resolve()
    graph.serialize(str(tmp_path), format="xml")
    logger.debug(
        "Serialized temporary schema graph for reasoning\n"
        "    from: %s\n"
        "      to: %s",
        schema_file.resolve(),
        tmp_path,
    )

    return tmp_path


def _build_temp_schema_kg_graph(schema_file: Path, kg_file: Path) -> Path:
    """Create a temporary merged schema+KG graph on disk.

    The merged graph is written next to the KG file and is intended for
    short-lived use as a reasoner input. The graph is always serialized
    as RDF/XML so that Owlready2 can reliably load it, regardless of the
    original schema/KG formats.

    Args:
        schema_file: Path to the schema ontology file.
        kg_file: Path to the KG ontology file.

    Returns:
        Path to the temporary RDF/XML file that should be given to Owlready2.
    """
    graph = RDFGraph()
    graph.parse(str(schema_file))
    graph.parse(str(kg_file))

    tmp_path = (kg_file.parent / "tmp_schema_kg.rdf").resolve()
    graph.serialize(str(tmp_path), format="xml")
    logger.debug(
        "Serialized temporary merged schema+KG graph for reasoning\n"
        "    from schema: %s\n"
        "        from KG: %s\n"
        "             to: %s",
        schema_file.resolve(),
        kg_file.resolve(),
        tmp_path,
    )

    return tmp_path


def _bytes_to_mb(byte_count: int) -> int:
    """Convert bytes to whole MB (base-2)."""
    return max(0, byte_count // (1024 * 1024))


# ================================================================================================ #
# JVM heap sizing (best-effort)                                                                    #
# ================================================================================================ #


def _configure_java_heap_best_effort(
    *,
    heap_fraction: float = _JAVA_HEAP_FRACTION_DEFAULT,
    min_heap_mb: int = _JAVA_HEAP_MIN_MB_DEFAULT,
    max_heap_mb: int | None = _JAVA_HEAP_MAX_MB_DEFAULT,
) -> None:
    """Best-effort JVM heap auto-configuration for Owlready2 reasoners.

    Owlready2 launches Java (HermiT/Pellet). If the JVM heap is too small,
    Pellet (especially "explain") may crash with "OutOfMemoryError: Java heap space".

    Policy:
        - If the user already provided a heap size (-Xmx) via environment variables,
          do nothing.
        - Otherwise, set a conservative heap to a fraction of total RAM.

    Args:
        heap_fraction: Fraction of total RAM to allocate to the JVM heap.
        min_heap_mb: Minimum heap size in MB.
        max_heap_mb: Optional maximum heap size in MB.

    Note:
        This must run before the JVM starts. Calling it early is safe; it
        only mutates environment variables when needed.
    """
    global _configured_java_heap_mb
    global _java_heap_config_source

    if not (0.10 <= heap_fraction <= 0.95):
        logger.debug(
            "Skipping JVM heap auto-config: heap_fraction out of expected range: %s",
            heap_fraction,
        )
        return

    existing_opts = _read_java_option_envelope()
    if "-Xmx" in existing_opts:
        _configured_java_heap_mb = None
        _java_heap_config_source = "preconfigured"
        logger.debug(
            "JVM heap already configured by user environment (found -Xmx in JAVA_* options): %s",
            existing_opts,
        )
        return

    total_ram_bytes = _get_total_system_ram_bytes_best_effort()
    if total_ram_bytes is None or total_ram_bytes <= 0:
        logger.debug("Could not determine system RAM; skipping JVM heap auto-config.")
        return

    total_ram_mb = _bytes_to_mb(total_ram_bytes)
    heap_target_bytes = int(total_ram_bytes * heap_fraction)
    heap_target_mb = max(min_heap_mb, _bytes_to_mb(heap_target_bytes))
    if max_heap_mb is not None:
        heap_target_mb = min(heap_target_mb, max_heap_mb)

    if heap_target_mb <= 0:
        logger.debug("Computed non-positive JVM heap target; skipping JVM heap auto-config.")
        return

    _append_java_tool_options(f"-Xmx{heap_target_mb}m")
    _configured_java_heap_mb = heap_target_mb
    _java_heap_config_source = "auto"

    logger.debug(
        "Detected system RAM: %sm; configured JVM heap: -Xmx%sm via %s",
        total_ram_mb,
        heap_target_mb,
        _JAVA_HEAP_ENVVAR_NAME,
    )
    logger.info(
        "Configured JVM heap for reasoners to -Xmx%sm (%.0f%% of detected RAM)",
        heap_target_mb,
        heap_fraction * 100.0,
    )


def _read_java_option_envelope() -> str:
    """Read the common Java option environment variables into a single string."""
    return " ".join(
        [
            os.environ.get("JAVA_TOOL_OPTIONS", ""),
            os.environ.get("_JAVA_OPTIONS", ""),
            os.environ.get("JAVA_OPTS", ""),
        ],
    ).strip()


def _append_java_tool_options(option: str) -> None:
    """Append a JVM option to JAVA_TOOL_OPTIONS (preferred for subprocess launches)."""
    current = os.environ.get(_JAVA_HEAP_ENVVAR_NAME, "").strip()
    if not current:
        os.environ[_JAVA_HEAP_ENVVAR_NAME] = option
        return
    if option in current:
        return
    os.environ[_JAVA_HEAP_ENVVAR_NAME] = f"{current} {option}"


def _get_total_system_ram_bytes_best_effort() -> int | None:
    """Return total system RAM in bytes, best-effort.

    Tries psutil first (if installed), then falls back to POSIX sysconf.
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.debug(
            "psutil not available; falling back to os.sysconf for RAM detection.",
            exc_info=exc,
        )
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        try:
            return int(psutil.virtual_memory().total)
        except (AttributeError, OSError, ValueError) as exc:
            logger.debug("psutil RAM detection failed; falling back to os.sysconf.", exc_info=exc)

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            return page_size * phys_pages
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("os.sysconf RAM detection failed.", exc_info=exc)
            return None

    return None
