#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Helper utilities for the KG instance generator (kg.py).

This module contains small, generator-scoped helpers used during KG
initialization and stochastic parameter sampling.

Design constraints:
- No global NumPy randomness: all randomness must flow through an explicit
  np.random.Generator provided by the caller.
- Integer-ID world: entities and relations are represented as ints.
- This module is not a general-purpose KG utilities library and should not
  be reused outside the generator context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ================================================================================================ #
# Used in __init__                                                                                 #
# ================================================================================================ #


def get_fast_ratio(num_entities: int) -> int:
    """Return the replication ratio used by fast generation.

    Args:
        num_entities: Number of entities requested for the KG.

    Returns:
        An integer replication factor (>= 1).
    """
    if num_entities >= 1_000_000:
        return 15
    if num_entities >= 500_000:
        return 10
    if num_entities >= 50_000:
        return 5
    if num_entities >= 30_000:
        return 3
    return 1


# ================================================================================================ #
# Used in relation weight sampling                                                                  #
# ================================================================================================ #


def generate_random_numbers(
    rng: np.random.Generator,
    mean: float,
    std_dev: float,
    size: int,
) -> NDArray[np.float64]:
    """Generate a strictly-positive weight vector summing to 1.0.

    Notes:
        This replaces the old global-numpy implementation. Callers must pass `rng`
        to guarantee determinism with a fixed seed.

    Args:
        rng: NumPy Generator seeded by the caller.
        mean: Mean for the normal distribution (pre-normalization).
        std_dev: Std dev for the normal distribution (pre-normalization).
        size: Number of weights to generate.

    Returns:
        Array of length `size`, dtype float64, strictly positive, summing to 1.0.

    Raises:
        ValueError: If size <= 0 or std_dev < 0.
    """
    if size <= 0:
        msg = "size must be a positive integer."
        raise ValueError(msg)
    if std_dev < 0.0:
        msg = "std_dev must be non-negative."
        raise ValueError(msg)

    samples = rng.normal(mean, std_dev, size).astype(np.float64)

    total = float(np.sum(samples))
    if total == 0.0:
        return np.ones(size, dtype=np.float64) / float(size)

    normalized = samples / total
    clipped = np.clip(normalized, np.finfo(np.float64).eps, 1.0)
    return clipped / float(np.sum(clipped))

