#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Utility helpers for schema inspection and traversal.

These functions are used by the schema/class generators to analyze inheritance
relationships, superclasses, and depth/layer structure of the ontology.
They intentionally avoid any side effects and operate purely on the
precomputed dictionaries provided by the schema constructors.

All functions preserve the legacy behavior for backwards compatibility.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def non_trivial_children(class2superclass_direct: Mapping[str, str]) -> list[str]:
    """Returns a list of classes that have at least one non-trivial parent.

    Args:
        class2superclass_direct (dict): A dictionary mapping classes to their direct superclasses.

    Returns:
        list: A list of classes that have at least one non-trivial parent.
    """
    return [c for c, parent in class2superclass_direct.items() if parent != "owl:Thing"]


def get_subclassof_count(class2layer: Mapping[str, int]) -> int:
    """Returns the number of classes that have at least one subclass.

    Args:
        class2layer (dict): A dictionary mapping classes to their layers.

    Returns:
        int: The number of classes that have at least one non-trivial parent.
    """
    return len([key for key, value in class2layer.items() if value > 1])


def get_leaves(
    class2superclass_direct: Mapping[str, str],
    class2subclasses_direct: Mapping[str, Sequence[str]],
) -> set[str]:
    """Returns a list of classes that have no subclasses, i.e. leaves.

    Args:
        class2superclass_direct (dict): A dictionary mapping classes to their direct superclasses.
        class2subclasses_direct (dict): A dictionary mapping classes to their direct subclasses.

    Returns:
        set: A set of classes that have no subclasses.
    """
    return set(class2superclass_direct.keys()) - set(class2subclasses_direct.keys())


def get_max_depth(layer2classes: Mapping[int, Sequence[str]]) -> int | None:
    """Returns the maximum depth of the schema.

    Args:
        layer2classes (dict): A dictionary mapping layers to classes.

    Returns:
        int | None: The maximum depth of the schema, or None if empty.
    """
    return max(
        (key for key, value in layer2classes.items() if value),
        default=None,
    )


def calculate_inheritance_ratio(
    class2superclass_direct: Mapping[str, str],
    class2subclasses_direct: Mapping[str, Sequence[str]],
) -> float:
    """Calculates the inheritance ratio of the schema.

    Args:
        class2superclass_direct (dict): A dictionary mapping classes to their direct superclasses.
        class2subclasses_direct (dict): A dictionary mapping classes to their direct subclasses.

    Returns:
        float: The inheritance ratio of the schema.
    """
    n_classes: int = len(class2superclass_direct.keys())
    n_leaves: int = len(
        get_leaves(class2superclass_direct, class2subclasses_direct),
    )
    n_non_trivial_children: int = len(
        non_trivial_children(class2superclass_direct),
    )

    return n_non_trivial_children / (n_classes - n_leaves)


def calculate_average_depth(layer2classes: Mapping[int, Sequence[str]]) -> float:
    """Calculates the average depth of the schema.

    Args:
        layer2classes (dict): A dictionary mapping layers to classes.

    Returns:
        float: The average depth of the schema.
    """
    denominator: int = sum(len(classes) for classes in layer2classes.values())
    numerator: float = 0.0

    for layer, classes in layer2classes.items():
        numerator += layer * len(classes)

    return numerator / denominator


def calculate_class_disjointness(
    class2disjoint: Mapping[str, Sequence[str]],
    num_classes: int,
) -> float:
    """Calculates the class disjointness of the schema.

    Args:
        class2disjoint (dict): A dictionary mapping classes to their disjoint classes.
        num_classes (int): The number of classes.

    Returns:
        float: The class disjointness of the schema.
    """
    return len(class2disjoint) / (2 * num_classes)


def get_all_superclasses(
    class_name: str,
    direct_class2superclass: Mapping[str, str],
) -> list[str]:
    """Returns a list of all superclasses of a given class.

    Args:
        class_name (str): The name of the class.
        direct_class2superclass (dict): A dictionary mapping classes to their direct superclasses.

    Returns:
        list: A list of all superclasses of the given class.
    """
    superclasses: list[str] = []

    if class_name in direct_class2superclass:
        superclass: str = direct_class2superclass[class_name]
        superclasses.append(superclass)
        superclasses.extend(
            get_all_superclasses(superclass, direct_class2superclass),
        )

    return superclasses


def get_all_subclasses(
    transitive_class2superclass: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Returns a dictionary mapping classes to their transitive subclasses.

    Args:
        transitive_class2superclass (dict): A dictionary mapping classes to their transitive superclasses.

    Returns:
        dict: A dictionary mapping classes to their subclasses.
    """
    class2subclasses: defaultdict[str, list[str]] = defaultdict(list)

    for subclass, superclasses in transitive_class2superclass.items():
        for superclass in superclasses:
            class2subclasses[superclass].append(subclass)

    return dict(class2subclasses)


def extend_class_mappings(
    direct_class2superclass: Mapping[str, str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Extends the class mappings to include transitive superclasses and subclasses.

    Args:
        direct_class2superclass (dict): A dictionary mapping classes to their direct superclasses.

    Returns:
        tuple: A tuple containing the extended class mappings:
            (transitive_class2superclass, transitive_class2subclasses).
    """
    transitive_class2superclass: dict[str, list[str]] = {}

    for class_name in direct_class2superclass:
        transitive_superclasses: list[str] = get_all_superclasses(
            class_name,
            direct_class2superclass,
        )
        transitive_class2superclass[class_name] = transitive_superclasses

    transitive_class2subclasses: dict[str, list[str]] = get_all_subclasses(
        transitive_class2superclass,
    )

    return transitive_class2superclass, transitive_class2subclasses


def generate_class2layer(layer2classes: Mapping[int, Sequence[str]]) -> dict[str, int]:
    """Generates a dictionary mapping classes to their layers.

    Args:
        layer2classes (dict): A dictionary mapping layers to classes.

    Returns:
        dict: A dictionary mapping classes to their layers.
    """
    class2layer: dict[str, int] = {}

    for layer, classes in layer2classes.items():
        for c in classes:
            class2layer[c] = layer

    return class2layer
