#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Type aliases for the KG generator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


# ------------------------------------------------------------------------------------------------ #
# Core ID Types                                                                                    #
# ------------------------------------------------------------------------------------------------ #
EntityId: TypeAlias = int
ClassId: TypeAlias = int
RelationId: TypeAlias = int

# ------------------------------------------------------------------------------------------------ #
# Composite Types                                                                                  #
# ------------------------------------------------------------------------------------------------ #
HeadTailPair: TypeAlias = tuple[EntityId, EntityId]  # (head, tail)
Triple: TypeAlias = tuple[EntityId, RelationId, EntityId]

# ------------------------------------------------------------------------------------------------ #
# Collection Types                                                                                 #
# ------------------------------------------------------------------------------------------------ #
EntityIdList: TypeAlias = list[EntityId]
EntityIdSet: TypeAlias = set[EntityId]

ClassIdFrozenSet: TypeAlias = frozenset[ClassId]
ClassIdSet: TypeAlias = set[ClassId]
ClassIdList: TypeAlias = list[ClassId]

RelationIdToHeadTailPairs: TypeAlias = dict[RelationId, set[HeadTailPair]]
ClassIdToEntityIds: TypeAlias = dict[ClassId, EntityIdList]

# ------------------------------------------------------------------------------------------------ #
# NumPy Array Types                                                                                #
# ------------------------------------------------------------------------------------------------ #
EntityIdArray: TypeAlias = NDArray[np.int64]
BooleanMask: TypeAlias = NDArray[np.bool_]
WeightArray: TypeAlias = NDArray[np.float64]

# ------------------------------------------------------------------------------------------------ #
# Pool Dictionaries                                                                                #
# ------------------------------------------------------------------------------------------------ #
RelationIdToEntityPool: TypeAlias = dict[RelationId, EntityIdArray]
RelationIdToSeenPairs: TypeAlias = dict[RelationId, set[HeadTailPair]]

# ------------------------------------------------------------------------------------------------ #
# Progress Bar                                                                                     #
# ------------------------------------------------------------------------------------------------ #
TqdmProgressBar: TypeAlias = Any
