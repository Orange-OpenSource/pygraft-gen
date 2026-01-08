#  Software Name: PyGraft-gen
#  SPDX-FileCopyrightText: Copyright (c) Orange SA
#  SPDX-License-Identifier: MIT
#
#  This software is distributed under the MIT license, the text of which is available at https://opensource.org/license/MIT/ or see the "LICENSE" file for more details.
#
#  Authors: See CONTRIBUTORS.txt
#  Software description: A RDF Knowledge Graph stochastic generation solution.
#

"""Triple generation with batch sampling and two-phase constraint filtering."""

from __future__ import annotations

from collections import Counter, deque
import copy
import itertools
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from pygraft.utils.kg import generate_random_numbers

if TYPE_CHECKING:
    from numpy.random import Generator as NumpyRNG

    from pygraft.generators.kg.config import InstanceGeneratorConfig
    from pygraft.generators.kg.structures import (
        ConstraintCaches,
        EntityTypingState,
        SchemaMetadata,
        TripleGenerationState,
    )
    from pygraft.generators.kg.types import (
        BooleanMask,
        ClassId,
        ClassIdFrozenSet,
        EntityId,
        EntityIdArray,
        EntityIdList,
        RelationId,
        RelationIdToEntityPool,
        RelationIdToSeenPairs,
        TqdmProgressBar,
        Triple,
    )
    from pygraft.types import RelationInfoDict

logger = logging.getLogger(__name__)


class TripleGenerator:
    """Generate KG triples using batch sampling with two-phase constraint filtering.

    Handles the triple generation phase of KG generation:
        1. Setup - Build candidate pools, distribute budget, initialize tracking
        2. Generation loop - Sample, filter, validate, accept

    Uses adaptive batch sizing and stall detection to handle heavily
    constrained configurations.

    Attributes:
        _config: Generator configuration.
        _constraints: Pre-computed constraint caches.
        _schema: String-to-integer ID mappings.
        _entities: Entity typing state (mutated during generation).
        _triples: Triple generation state (mutated during generation).
        _relation_info: Loaded relation_info.json contents.
        _rng: NumPy random generator.
        _dom_pools: Domain candidate pools per relation (set during setup).
        _rng_pools: Range candidate pools per relation (set during setup).
        _active_relations: Relations eligible for sampling (mutated during loop).
        _seen_pairs: Duplicate tracking per relation (mutated during loop).
        _weights: Sampling weights for active relations (mutated during loop).
        _stall_counts: Consecutive empty batch counts per relation (mutated during loop).
    """

    # ================================================================================================ #
    # GENERATION CONSTANTS                                                                             #
    # ================================================================================================ #

    # Tail phase thresholds (generation progress percentages)
    TAIL_PHASE_START = 0.90  # After which % value to start tail phase timer and stall detection

    # Tail phase protections
    TAIL_TIMEOUT_SECONDS = 1200  # 20 minutes maximum in tail phase
    MAX_CONSECUTIVE_EMPTY_BATCHES = 20  # Stall threshold for dropping relations

    # ================================================================================================ #
    # CONSTRUCTION                                                                                     #
    # ================================================================================================ #

    def __init__(
        self,
        *,
        config: InstanceGeneratorConfig,
        constraints: ConstraintCaches,
        schema: SchemaMetadata,
        entities: EntityTypingState,
        triples: TripleGenerationState,
        relation_info: RelationInfoDict,
        rng: NumpyRNG,
    ) -> None:
        """Initialize the triple generator.

        Args:
            config: Generator configuration.
            constraints: Pre-computed constraint caches.
            schema: String-to-integer ID mappings.
            entities: Entity typing state (mutated during generation).
            triples: Triple generation state (mutated during generation).
            relation_info: Loaded relation_info.json contents.
            rng: NumPy random generator (mutated during generation).
        """
        self._config = config
        self._constraints = constraints
        self._schema = schema
        self._entities = entities
        self._triples = triples
        self._relation_info = relation_info
        self._rng = rng

        # Incremental tracking for symmetric relation participants
        self._symmetric_functional_tails: dict[RelationId, set[EntityId]] = {}
        self._symmetric_invfunctional_heads: dict[RelationId, set[EntityId]] = {}

        # Adjacency list for transitive cycle detection
        # Only maintained for transitive + (irreflexive OR asymmetric) relations
        self._transitive_adjacency: dict[RelationId, dict[EntityId, set[EntityId]]] = {}

        # Loop state (populated by _setup_triple_generation)
        self._dom_pools: RelationIdToEntityPool = {}
        self._rng_pools: RelationIdToEntityPool = {}
        self._active_relations: list[RelationId] = []
        self._seen_pairs: RelationIdToSeenPairs = {}
        self._weights: list[float] = []
        self._stall_counts: dict[RelationId, int] = {}

    # ================================================================================================ #
    # PUBLIC API                                                                                       #
    # ================================================================================================ #

    def generate(self) -> int:
        """Run the full triple generation pipeline.

        Two-phase triple generation:
            1. Setup - Build candidate pools, distribute budget, initialize tracking
            2. Generation loop - Sample, filter, validate, accept

        Returns:
            Number of triples successfully generated.

        Complexity:
            Setup: O(n_e * n_c + n_r * pool_size)
            Loop: O(n_t * batch_size) with vectorized fast filtering; deep
            validation adds O(1) per surviving candidate for most constraints.
        """
        # Phase 1: Setup (one-time preparation)
        self._setup_triple_generation()

        # Phase 2: Generation loop (performance-critical section)
        accepted = self._run_generation_loop()

        logger.info("Triple generation complete: %d triples generated", accepted)
        return accepted

    # ================================================================================================ #
    # RUNTIME SATISFIABILITY                                                                           #
    # ================================================================================================ #

    def _relation_is_satisfiable(self, relation_id: RelationId) -> bool:
        """Check if a relation can generate triples.

        Validates both schema-level (not marked unsatisfiable) and runtime
        (non-empty candidate pools) satisfiability.

        Args:
            relation_id: Relation to check.

        Returns:
            True if relation can generate valid triples.
        """
        # Check schema-level unsatisfiability first (fast)
        if relation_id in self._constraints.unsatisfiable_relation_ids:
            return False

        # Check runtime satisfiability (entity availability)
        dom_classes = self._constraints.rel2dom_ids[relation_id]
        rng_classes = self._constraints.rel2range_ids[relation_id]

        def _has_candidates(required: ClassIdFrozenSet) -> bool:
            if not required:
                return bool(
                    self._entities.priority_untyped_entities
                    or self._entities.untyped_entities
                    or self._entities.unseen_entities_pool
                )

            buckets: list[EntityIdList] = []
            for cid in required:
                ents = self._entities.class2entities.get(cid)
                if not ents:
                    return False
                buckets.append(ents)

            smallest = min(buckets, key=len)
            cand = set(smallest)
            for b in buckets:
                cand.intersection_update(b)
                if not cand:
                    return False
            return True

        return _has_candidates(dom_classes) and _has_candidates(rng_classes)

    # ================================================================================================ #
    # SETUP PHASE                                                                                      #
    # ================================================================================================ #

    def _setup_triple_generation(self) -> None:
        """Prepare all data structures for triple generation.

        Seven-step setup:
            1. Build class-to-entities index for entity sampling
            2. Initialize freshness tracking for sampling bias
            3. Identify satisfiable relations
            4. Build candidate entity pools per relation
            5. Initialize KG storage and tracking structures
            6. Compute triple budgets per relation
            7. Finalize active relations and duplicate tracking

        Stores results on instance attributes for use in generation loop.

        Raises:
            ValueError: If no satisfiable relations found.

        Complexity:
            O(n_e * n_c + n_r * pool_size) dominated by pool construction.
        """
        # Step 1: Build entity sampling indices
        self._build_class_entity_indices()

        # Step 2: Initialize freshness tracking
        self._initialize_freshness_tracking()

        # Step 3: Identify satisfiable relations
        satisfiable_relations = self._identify_satisfiable_relations()

        # Step 4: Build candidate pools (expensive operation)
        self._dom_pools, self._rng_pools = self._build_and_log_candidate_pools(satisfiable_relations)

        # Step 5: Initialize KG storage and tracking structures
        self._initialize_kg_storage()

        # Step 6: Compute triple budgets per relation
        self._compute_triples_per_relation(allowed_relations=satisfiable_relations)

        # Step 7: Finalize active relations and tracking
        self._active_relations, self._seen_pairs = self._finalize_active_relations(
            satisfiable_relations, self._dom_pools, self._rng_pools
        )

        # Initialize loop state
        self._weights = self._recalculate_relation_weights(
            self._active_relations, uniform_fallback=True
        )
        self._stall_counts = {}

    # --- Step 1: Build Class-Entity Indices ---
    def _build_class_entity_indices(self) -> None:
        """Build class-to-entities reverse index for entity sampling.

        Maps class IDs to entity IDs for fast lookup during candidate sampling.
        Only includes typed entities.

        Complexity:
            O(n_e * avg_classes_per_entity).
        """
        self._entities.class2entities = {}
        for entity_id, class_ids in enumerate(self._entities.ent2classes_transitive):
            if not class_ids:
                continue
            for cid in class_ids:
                self._entities.class2entities.setdefault(cid, []).append(entity_id)

    # --- Step 2: Initialize Freshness Tracking ---
    def _initialize_freshness_tracking(self) -> None:
        """Initialize unseen entity tracking for freshness bias.

        Freshness bias prioritizes sampling entities not yet in any triple,
        improving diversity in early generation.

        Complexity:
            O(n_e * avg_classes_per_entity).
        """
        # Copy class2entities for tracking which entities are still unseen per class
        self._entities.class2unseen = copy.deepcopy(self._entities.class2entities)

        # Create flat pool of all unique unseen entities (across all classes)
        # sorted for rng seed determinism
        self._entities.unseen_entities_pool = sorted(
            set(itertools.chain.from_iterable(self._entities.class2entities.values())),
        )

        # Identify untyped entities (entities with no class assignments)
        self._entities.priority_untyped_entities = set(self._entities.entities) - set(
            self._entities.typed_entities
        )
        # sorted for rng seed determinism
        self._entities.untyped_entities = sorted(self._entities.priority_untyped_entities)

    # --- Step 3: Identify Satisfiable Relations ---
    def _identify_satisfiable_relations(self) -> list[RelationId]:
        """Identify relations that can generate triples.

        Filters out relations marked unsatisfiable at schema level or lacking
        entity candidates at runtime.

        Returns:
            List of relation IDs that can generate valid triples.

        Raises:
            ValueError: If no satisfiable relations found.

        Complexity:
            O(n_r * pool_size).
        """
        all_relation_ids = list(range(len(self._schema.id2rel)))
        satisfiable_relations = [
            rid for rid in all_relation_ids if self._relation_is_satisfiable(rid)
        ]

        if not satisfiable_relations:
            msg = "No satisfiable relations found."
            raise ValueError(msg)

        return satisfiable_relations

    # --- Step 4: Build Candidate Pools ---
    def _build_and_log_candidate_pools(
        self, satisfiable_relations: list[RelationId]
    ) -> tuple[RelationIdToEntityPool, RelationIdToEntityPool]:
        """Build candidate pools with progress logging.

        Wraps pool construction with logging for visibility into this expensive
        operation.

        Args:
            satisfiable_relations: Relations to build pools for.

        Returns:
            Tuple of (dom_pools, rng_pools) mapping relation ID to entity arrays.

        Complexity:
            O(n_r * n_e * constraint_complexity).
        """
        logger.info("Building candidate pools for %d relations...", len(satisfiable_relations))

        dom_pools, rng_pools = self._build_relation_candidate_pools()

        logger.info("Candidate pools built: dom=%d, rng=%d", len(dom_pools), len(rng_pools))

        return dom_pools, rng_pools

    def _build_relation_candidate_pools(
        self,
    ) -> tuple[RelationIdToEntityPool, RelationIdToEntityPool]:
        """Precompute domain and range entity pools for each satisfiable relation.

        For each relation, computes candidate sets by intersecting entities that
        satisfy all domain/range classes (conjunctive semantics), then filters
        by disjoint envelopes. Inverse relation constraints are also applied.

        Returns:
            Tuple of (dom_pool, rng_pool) mapping relation ID to NumPy arrays.

        Complexity:
            O(n_r * n_e) for set intersections and filtering.
        """
        dom_pool: RelationIdToEntityPool = {}
        rng_pool: RelationIdToEntityPool = {}

        for rel_id in range(len(self._schema.id2rel)):
            if not self._relation_is_satisfiable(rel_id):
                continue

            dom_class_ids = self._constraints.rel2dom_ids[rel_id]
            rng_class_ids = self._constraints.rel2range_ids[rel_id]

            # Collect domain candidates (conjunctive - must satisfy ALL classes)
            dom_candidates: set[EntityId] = set()
            if dom_class_ids:
                candidate_sets: list[set[EntityId]] = [
                    set(self._entities.class2entities.get(cid, [])) for cid in rng_class_ids
                ]
                if candidate_sets:
                    dom_candidates = candidate_sets[0].intersection(*candidate_sets[1:])
            else:
                dom_candidates = set(self._entities.entities)

            # Filter domain candidates by disjoint envelope
            dom_forbidden = self._constraints.dom_disjoint_envelope[rel_id]
            if dom_forbidden:
                dom_candidates = {
                    e
                    for e in dom_candidates
                    if e not in self._entities.typed_entities
                    or not (self._entities.ent2classes_transitive_sets[e] & dom_forbidden)
                }

            # Collect range candidates (conjunctive)
            rng_candidates: set[EntityId] = set()
            if rng_class_ids:
                candidate_sets = [
                    set(self._entities.class2entities.get(cid, [])) for cid in rng_class_ids
                ]
                if candidate_sets:
                    rng_candidates = candidate_sets[0].intersection(*candidate_sets[1:])
            else:
                rng_candidates = set(self._entities.entities)

            # Filter range candidates by disjoint envelope
            rng_forbidden = self._constraints.range_disjoint_envelope[rel_id]
            if rng_forbidden:
                rng_candidates = {
                    e
                    for e in rng_candidates
                    if e not in self._entities.typed_entities
                    or not (self._entities.ent2classes_transitive_sets[e] & rng_forbidden)
                }

            # Filter by inverse relation's disjoint envelopes (if inverse exists)
            inv_rel_id = self._constraints.rel2inverse_ids.get(rel_id)
            if inv_rel_id is not None:
                # Domain candidates become range in inverse triple (t, inv, h)
                # Must not violate inverse's range disjoint envelope
                inv_rng_forbidden = self._constraints.range_disjoint_envelope[inv_rel_id]
                if inv_rng_forbidden:
                    dom_candidates = {
                        e
                        for e in dom_candidates
                        if e not in self._entities.typed_entities
                        or not (self._entities.ent2classes_transitive_sets[e] & inv_rng_forbidden)
                    }

                # Range candidates become domain in inverse triple (t, inv, h)
                # Must not violate inverse's domain disjoint envelope
                inv_dom_forbidden = self._constraints.dom_disjoint_envelope[inv_rel_id]
                if inv_dom_forbidden:
                    rng_candidates = {
                        e
                        for e in rng_candidates
                        if e not in self._entities.typed_entities
                        or not (self._entities.ent2classes_transitive_sets[e] & inv_dom_forbidden)
                    }

            # Convert to NumPy arrays
            if dom_candidates:
                dom_pool[rel_id] = np.asarray(sorted(dom_candidates), dtype=np.int64)
            if rng_candidates:
                rng_pool[rel_id] = np.asarray(sorted(rng_candidates), dtype=np.int64)

        return dom_pool, rng_pool

    # --- Step 5: Initialize KG Storage ---
    def _initialize_kg_storage(self) -> None:
        """Initialize KG storage and constraint tracking structures.

        Creates empty sets for triple storage and incremental tracking for
        functional/inverse-functional properties. Incremental tracking enables
        O(1) constraint checks instead of O(m) where m = existing triples.

        Complexity:
            O(n_r).
        """
        # Initialize KG storage for all relations
        all_relation_ids = list(range(len(self._schema.id2rel)))
        self._triples.kg_pairs_by_rid = {rel_id: set() for rel_id in all_relation_ids}

        # Initialize incremental constraint tracking for functional properties
        self._triples.functional_heads = {
            rel_id: set() for rel_id in self._constraints.functional_ids
        }
        self._triples.invfunctional_tails = {
            rel_id: set() for rel_id in self._constraints.invfunctional_ids
        }

        # Initialize incremental tracking for symmetric relation participants
        # For symmetric+functional: tails act as heads (need separate tracking)
        # For symmetric+inverse-functional: heads act as tails (need separate tracking)
        symmetric_functional = self._constraints.symmetric_ids & self._constraints.functional_ids
        symmetric_invfunctional = self._constraints.symmetric_ids & self._constraints.invfunctional_ids

        self._symmetric_functional_tails = {rel_id: set() for rel_id in symmetric_functional}
        self._symmetric_invfunctional_heads = {rel_id: set() for rel_id in symmetric_invfunctional}

        # Initialize adjacency lists for transitive cycle detection
        # Only for transitive + (irreflexive OR asymmetric) relations
        problematic_transitive = {
            rel_id for rel_id in self._constraints.transitive_ids
            if rel_id in self._constraints.irreflex_ids or rel_id in self._constraints.asym_ids
        }
        self._transitive_adjacency = {rel_id: {} for rel_id in problematic_transitive}

    # --- Step 6: Compute Triple Budgets ---
    def _compute_triples_per_relation(self, *, allowed_relations: list[RelationId]) -> None:
        """Compute sampling weights and target triple counts per relation.

        Distributes triple budget across relations based on relation_usage_uniformity.
        Higher uniformity produces more even distribution.

        Args:
            allowed_relations: Relations eligible for sampling.

        Raises:
            ValueError: If allowed_relations is empty.

        Complexity:
            O(n_r).
        """
        if not allowed_relations:
            msg = "allowed_relations must be non-empty."
            raise ValueError(msg)

        self._triples.num_relations = len(allowed_relations)

        # Special case: fewer triples than relations.
        if self._config.num_triples < self._triples.num_relations:
            # For very small KGs, use a uniform distribution over relations and
            # assign at most one triple per relation in order.
            uniform_weight = 1.0 / self._triples.num_relations
            self._triples.relation_sampling_weights = [uniform_weight] * self._triples.num_relations
            self._triples.triples_per_rel = {
                rel: 1 if idx < self._config.num_triples else 0
                for idx, rel in enumerate(allowed_relations)
            }
            return

        mean = int(self._config.num_triples / self._triples.num_relations)
        spread = (1.0 - self._config.relation_usage_uniformity) * mean

        weights = generate_random_numbers(self._rng, mean, spread, self._triples.num_relations)
        normalized = weights / float(np.sum(weights))

        self._triples.relation_sampling_weights = list(normalized)

        scaled = normalized * float(self._config.num_triples)
        self._triples.triples_per_rel = {
            rel: int(np.ceil(tpr)) for rel, tpr in zip(allowed_relations, scaled, strict=True)
        }

    # --- Step 7: Finalize Active Relations ---
    def _finalize_active_relations(
        self,
        satisfiable_relations: list[RelationId],
        dom_pools: RelationIdToEntityPool,
        rng_pools: RelationIdToEntityPool,
    ) -> tuple[list[RelationId], dict[RelationId, set[tuple[EntityId, EntityId]]]]:
        """Finalize active relations and initialize duplicate tracking.

        Filters satisfiable relations to those with actual pools and initializes
        empty seen_pairs tracking for duplicate detection.

        Args:
            satisfiable_relations: Relations that passed satisfiability check.
            dom_pools: Domain pools per relation.
            rng_pools: Range pools per relation.

        Returns:
            Tuple of (active_relations, seen_pairs).

        Complexity:
            O(n_r).
        """
        # Initialize seen pairs tracking for fast duplicate detection
        all_relation_ids = list(range(len(self._schema.id2rel)))
        seen_pairs: dict[RelationId, set[tuple[EntityId, EntityId]]] = {
            rid: set() for rid in all_relation_ids
        }

        # Filter to active relations (those with pools and budgets)
        active_relations = [
            rid for rid in satisfiable_relations if rid in dom_pools and rid in rng_pools
        ]

        logger.info(
            "Setup complete: %d active relations ready for generation", len(active_relations)
        )

        return active_relations, seen_pairs

    # ================================================================================================ #
    # GENERATION LOOP                                                                                  #
    # ================================================================================================ #

    def _run_generation_loop(self) -> int:
        """Execute main triple generation loop with adaptive batch sizing.

        Generates triples until target count is reached or all relations exhausted.
        Uses weighted sampling to respect relation_usage_uniformity with per-relation
        quotas.

        Returns:
            Number of triples successfully generated.

        Complexity:
            O(n_t * batch_size) where n_t is target triple count. Dominated by
            sampling and validation costs per iteration.
        """
        # Initialize loop state
        target = self._config.num_triples
        accepted = 0
        base_batch_size = self._get_base_batch_size(target)
        tail_start_time = None
        tail_message_shown = False

        with tqdm(
            total=target,
            desc="Generating instance triples",
            unit="triples",
            colour="red",
            dynamic_ncols=True,
        ) as pbar:
            while accepted < target and self._active_relations:
                progress = accepted / target

                # Handle tail phase entry
                if not tail_message_shown:
                    tail_start_time, tail_message_shown = self._handle_tail_phase_entry(
                        progress, tail_start_time, tail_message_shown, pbar
                    )

                # Check tail timeout
                if self._check_tail_timeout(tail_start_time, accepted, pbar):
                    break

                # Calculate adaptive batch size for this iteration
                iteration_size = self._calculate_iteration_batch_size(
                    progress, base_batch_size, accepted
                )

                # Sample relations for this iteration
                relation_targets = self._sample_weighted_relation_batch(iteration_size)

                # Generate triples for sampled relations
                accepted += self._generate_triples_for_sampled_relations(
                    relation_targets=relation_targets,
                    accepted=accepted,
                    pbar=pbar,
                )

        return accepted

    # --- Loop Management: Tail Phase ---
    def _handle_tail_phase_entry(
        self,
        progress: float,
        tail_start_time: float | None,
        tail_message_shown: bool,
        pbar: TqdmProgressBar,
    ) -> tuple[float | None, bool]:
        """Handle tail phase entry message and timer initialization.

        Tail phase begins at TAIL_PHASE_START (90%) completion, enabling stall
        detection and timeout protection.

        Args:
            progress: Current generation progress (0.0 to 1.0).
            tail_start_time: Timer start time or None if not started.
            tail_message_shown: Whether entry message was shown.
            pbar: Progress bar for writing messages.

        Returns:
            Tuple of (tail_start_time, tail_message_shown) updated values.
        """
        if progress >= self.TAIL_PHASE_START and not tail_message_shown:
            tail_start_time = time.monotonic()
            tail_message_shown = True
            pbar.write(
                f"Entered tail phase at {self.TAIL_PHASE_START * 100:.1f}% completion. "
                f"Generation will stop after {self.TAIL_TIMEOUT_SECONDS / 60:.0f} minutes if target not reached."
            )
        return tail_start_time, tail_message_shown

    def _check_tail_timeout(
        self,
        tail_start_time: float | None,
        accepted: int,
        pbar: TqdmProgressBar,
    ) -> bool:
        """Check if tail phase timeout has expired.

        Args:
            tail_start_time: Timer start time or None if not in tail phase.
            accepted: Number of triples accepted so far.
            pbar: Progress bar for writing messages.

        Returns:
            True if timeout expired and generation should stop.
        """
        if tail_start_time is None:
            return False

        target = self._config.num_triples
        elapsed = time.monotonic() - tail_start_time
        if elapsed >= self.TAIL_TIMEOUT_SECONDS:
            progress = accepted / target
            pbar.write(
                f"Tail timer expired after {elapsed / 60:.1f} minutes. "
                f"Generated {accepted}/{target} triples ({progress * 100:.1f}%)."
            )
            return True

        return False

    # --- Loop Management: Batch Sizing ---
    def _calculate_iteration_batch_size(
        self,
        progress: float,
        base_batch_size: int,
        accepted: int,
    ) -> int:
        """Calculate adaptive batch size for current iteration.

        Scales batch size based on progress to maintain throughput as valid pairs
        become scarcer in tail phase.

        Args:
            progress: Current generation progress (0.0 to 1.0).
            base_batch_size: Base batch size for graph magnitude.
            accepted: Number of triples accepted so far.

        Returns:
            Batch size for this iteration, capped at remaining triples needed.
        """
        progress_factor = self._get_progress_factor(progress)
        iteration_size = int(base_batch_size * progress_factor)

        # Don't exceed remaining triples needed
        remaining = self._config.num_triples - accepted
        return min(iteration_size, remaining)

    @staticmethod
    def _get_base_batch_size(num_triples: int) -> int:
        """Determine base batch size scaled to graph magnitude.

        Larger graphs benefit from larger batches to amortize NumPy overhead
        and provide more candidate diversity per iteration.

        Args:
            num_triples: Target number of triples.

        Returns:
            Base batch size (1K to 100K depending on scale).
        """
        if num_triples < 10_000:
            return 1_000
        if num_triples < 100_000:
            return 5_000
        if num_triples < 1_000_000:
            return 10_000
        return 100_000

    @staticmethod
    def _get_progress_factor(progress: float) -> float:
        """Scale batch size based on generation progress.

        Returns:
            Multiplier for base batch size: 1.0x (0-85%), 1.5x (85-95%), 2.0x (95%+).
        """
        if progress < 0.85:
            return 1.0
        if progress < 0.95:
            return 1.5
        return 2.0

    # --- Loop Management: Relation Sampling ---
    def _sample_weighted_relation_batch(
        self,
        iteration_size: int,
    ) -> dict[RelationId, int]:
        """Sample relations according to weights for this iteration.

        Args:
            iteration_size: Number of triples to target this iteration.

        Returns:
            Dict mapping relation_id to target count for this iteration.

        Complexity:
            O(iteration_size) for weighted sampling.
        """
        sampled_relations = self._rng.choice(
            self._active_relations,
            size=iteration_size,
            replace=True,
            p=self._weights,
        )
        return Counter(sampled_relations)

    def _recalculate_relation_weights(
        self,
        active_relations: list[RelationId],
        *,
        uniform_fallback: bool = False,
    ) -> list[float]:
        """Recalculate normalized sampling weights for active relations.

        Called when relations are added or removed during generation to maintain
        a proper probability distribution.

        Args:
            active_relations: Relation IDs currently eligible for sampling.
            uniform_fallback: If True, return uniform distribution when total is zero.

        Returns:
            List of normalized weights in same order as active_relations.

        Complexity:
            O(n_active) where n_active is count of active relations.
        """
        if not active_relations:
            return []

        # Map relation IDs to their positions in the active list
        rel_id_to_index = {rel_id: idx for idx, rel_id in enumerate(active_relations)}

        # Extract weights for active relations
        weights = [
            self._triples.relation_sampling_weights[rel_id_to_index[rel_id]]
            for rel_id in active_relations
        ]

        # Normalize weights to sum to 1.0
        total = sum(weights)
        if total > 0:
            return [w / total for w in weights]
        if uniform_fallback:
            # Fallback to uniform distribution if all weights are zero
            return [1.0 / len(active_relations)] * len(active_relations)
        return weights

    # --- ORCHESTRATOR Level 3: Relation Iterator ---
    def _generate_triples_for_sampled_relations(
        self,
        relation_targets: dict[RelationId, int],
        accepted: int,
        pbar: TqdmProgressBar,
    ) -> int:
        """Generate triples for all sampled relations in this iteration.

        Iterates over sampled relations, generates candidate batches, validates,
        and accepts valid triples. Updates stall tracking and drops stalled
        relations in tail phase.

        Args:
            relation_targets: Dict mapping relation_id to target count.
            accepted: Current number of accepted triples.
            pbar: Progress bar for updates.

        Returns:
            Number of triples accepted this iteration.

        Complexity:
            O(sum(targets) * validation_cost) where targets are per-relation counts.
        """
        total_accepted = 0
        target = self._config.num_triples

        for rel_id, target_count in relation_targets.items():
            if accepted + total_accepted >= target:
                break

            # Get pools for this relation
            dom_pool = self._dom_pools.get(rel_id)
            rng_pool = self._rng_pools.get(rel_id)

            # Safety check: remove relation if pools disappeared
            if dom_pool is None or rng_pool is None:
                self._active_relations.remove(rel_id)
                self._weights[:] = self._recalculate_relation_weights(
                    self._active_relations, uniform_fallback=True
                )
                continue

            # Generate triples for this relation
            accepted_this_relation = self._sample_validate_and_accept_batch(
                rel_id=rel_id,
                target_count=target_count,
                accepted_so_far=accepted + total_accepted,
                pbar=pbar,
            )

            total_accepted += accepted_this_relation

            # Update stall tracking and drop stalled relations
            progress = (accepted + total_accepted) / target
            self._update_relation_stall_tracking(
                rel_id=rel_id,
                accepted_this_relation=accepted_this_relation,
                progress=progress,
                pbar=pbar,
            )

        return total_accepted

    # --- Stall Tracking ---
    def _update_relation_stall_tracking(
        self,
        rel_id: RelationId,
        accepted_this_relation: int,
        progress: float,
        pbar: TqdmProgressBar,
    ) -> None:
        """Update stall tracking and drop stalled relations in tail phase.

        Tracks consecutive empty batches per relation. Drops relations exceeding
        MAX_CONSECUTIVE_EMPTY_BATCHES threshold during tail phase to prevent
        infinite loops on unsatisfiable constraints.

        Args:
            rel_id: Relation ID to track.
            accepted_this_relation: Number accepted this iteration.
            progress: Current generation progress.
            pbar: Progress bar for messages.
        """
        # Update consecutive empty batch counter
        if accepted_this_relation == 0:
            self._stall_counts[rel_id] = self._stall_counts.get(rel_id, 0) + 1
        else:
            self._stall_counts[rel_id] = 0

        # Drop stalled relations in tail phase
        if (
            progress >= self.TAIL_PHASE_START
            and self._stall_counts.get(rel_id, 0) >= self.MAX_CONSECUTIVE_EMPTY_BATCHES
        ):
            self._active_relations.remove(rel_id)
            pbar.write(
                f"Dropped stalled relation: {self._schema.id2rel[rel_id]} "
                f"({self.MAX_CONSECUTIVE_EMPTY_BATCHES} consecutive empty batches in tail phase)"
            )

            # Recompute weights after relation removal
            if self._active_relations:
                self._weights[:] = self._recalculate_relation_weights(
                    self._active_relations,
                    uniform_fallback=True,
                )

    # ================================================================================================ #
    # BATCH PIPELINE                                                                                   #
    # ================================================================================================ #

    def _sample_validate_and_accept_batch(
        self,
        rel_id: RelationId,
        target_count: int,
        accepted_so_far: int,
        pbar: TqdmProgressBar,
    ) -> int:
        """Process one candidate batch through sampling, validation, and acceptance.

        Three-step pipeline:
            1. Sample candidate pairs from domain/range pools
            2. Fast filtering with vectorized constraints
            3. Deep validation and acceptance of survivors

        Args:
            rel_id: Relation ID.
            target_count: Target number of triples for this relation.
            accepted_so_far: Total triples accepted so far.
            pbar: Progress bar for updates.

        Returns:
            Number of triples accepted for this relation.

        Complexity:
            O(batch_size) for sampling and fast filtering, plus O(survivors) for
            deep validation.
        """
        dom_pool = self._dom_pools[rel_id]
        rng_pool = self._rng_pools[rel_id]

        # Step 1: Sample candidate batch
        candidate_heads, candidate_tails = self._sample_candidate_batch(
            rel_id=rel_id,
            dom_pool=dom_pool,
            rng_pool=rng_pool,
            batch_size=target_count,
        )

        # Step 2: Fast filtering (vectorized/O(1) checks eliminate most invalids)
        survivor_heads, survivor_tails = self._filter_with_fast_constraints(
            rel_id=rel_id,
            candidate_heads=candidate_heads,
            candidate_tails=candidate_tails,
        )

        # Step 3: Deep validation and acceptance (thorough per-triple checks)
        return self._validate_and_accept_survivors(
            rel_id=rel_id,
            survivor_heads=survivor_heads,
            survivor_tails=survivor_tails,
            target_count=target_count,
            accepted_so_far=accepted_so_far,
            pbar=pbar,
        )

    # --- Batch Step 1: Sample Candidates ---
    def _sample_candidate_batch(
        self,
        rel_id: RelationId,
        dom_pool: EntityIdArray,
        rng_pool: EntityIdArray,
        batch_size: int,
    ) -> tuple[EntityIdArray, EntityIdArray]:
        """Sample candidate entity pairs from domain and range pools.

        Pure sampling with freshness bias. No validation or filtering.

        Args:
            rel_id: Relation ID.
            dom_pool: Domain candidate pool.
            rng_pool: Range candidate pool.
            batch_size: Number of candidate pairs to sample.

        Returns:
            Tuple of (candidate_heads, candidate_tails) as NumPy arrays.

        Complexity:
            O(batch_size) for random sampling.
        """
        dom_class_ids = self._constraints.rel2dom_ids[rel_id]
        rng_class_ids = self._constraints.rel2range_ids[rel_id]

        candidate_heads = self._sample_side_entities_batch(
            pool=dom_pool,
            side_class_ids=dom_class_ids,
            batch_size=batch_size,
        )
        candidate_tails = self._sample_side_entities_batch(
            pool=rng_pool,
            side_class_ids=rng_class_ids,
            batch_size=batch_size,
        )

        return candidate_heads, candidate_tails

    # --- Batch Step 2: Fast Filter ---
    def _filter_with_fast_constraints(
        self,
        rel_id: RelationId,
        candidate_heads: EntityIdArray,
        candidate_tails: EntityIdArray,
    ) -> tuple[EntityIdArray, EntityIdArray]:
        """Filter candidates using fast batch-level constraints.

        Applies vectorized and O(1) checks: irreflexive, duplicates, symmetric
        duplicates, asymmetric, functional, inverse-functional.

        Args:
            rel_id: Relation ID.
            candidate_heads: Candidate head entities.
            candidate_tails: Candidate tail entities.

        Returns:
            Tuple of (survivor_heads, survivor_tails) after fast masking.

        Complexity:
            O(batch_size + m) where m is existing pairs for this relation.
        """
        # Build fast constraint mask (vectorized/O(1) checks)
        fast_mask = self._build_fast_local_constraint_mask_for_relation(
            relation_id=rel_id,
            candidate_heads=candidate_heads,
            candidate_tails=candidate_tails,
        )

        # Filter to survivors only
        return candidate_heads[fast_mask], candidate_tails[fast_mask]

    # --- Batch Step 3: Deep Validate and Accept ---
    def _validate_and_accept_survivors(
        self,
        rel_id: RelationId,
        survivor_heads: EntityIdArray,
        survivor_tails: EntityIdArray,
        target_count: int,
        accepted_so_far: int,
        pbar: TqdmProgressBar,
    ) -> int:
        """Validate survivors with deep constraint checks and accept valid triples.

        Performs intra-batch conflict checks (state may have changed since fast
        mask) followed by deep validation: domain/range typing, disjoint envelopes,
        inverse constraints, relation patterns, subproperty inheritance.

        Args:
            rel_id: Relation ID.
            survivor_heads: Survivors from fast filtering.
            survivor_tails: Survivors from fast filtering.
            target_count: Target for this relation.
            accepted_so_far: Total triples accepted so far.
            pbar: Progress bar for updates.

        Returns:
            Number of triples accepted for this relation.

        Complexity:
            O(survivors * validation_cost) where validation_cost is O(1) for most
            constraints but O(path_length) for transitive cycle detection.
        """
        accepted_this_relation = 0
        dom_class_ids = self._constraints.rel2dom_ids[rel_id]
        rng_class_ids = self._constraints.rel2range_ids[rel_id]
        target = self._config.num_triples

        for head_id, tail_id in zip(survivor_heads, survivor_tails, strict=True):
            # Stop if quotas hit
            if accepted_this_relation >= target_count or accepted_so_far >= target:
                break

            h = int(head_id)
            t = int(tail_id)

            # Intra-batch conflict check: duplicate
            # (State may have changed since fast mask was built)
            if (h, t) in self._triples.kg_pairs_by_rid[rel_id]:
                continue

            # Intra-batch conflict check: symmetric duplicate
            if (
                rel_id in self._constraints.symmetric_ids
                and (t, h) in self._triples.kg_pairs_by_rid[rel_id]
            ):
                continue

            # Intra-batch conflict check: functional constraint
            if rel_id in self._constraints.functional_ids:
                if h in self._triples.functional_heads.get(rel_id, set()):
                    continue
                # For symmetric relations, also check if h appears as tail
                # Uses incremental tracking instead of O(n) scan
                if h in self._symmetric_functional_tails.get(rel_id, set()):
                    continue

            # Intra-batch conflict check: inverse-functional constraint
            if rel_id in self._constraints.invfunctional_ids:
                if t in self._triples.invfunctional_tails.get(rel_id, set()):
                    continue
                # For symmetric relations, also check if t appears as head
                # Uses incremental tracking instead of O(n) scan
                if t in self._symmetric_invfunctional_heads.get(rel_id, set()):
                    continue

            # Deep constraint validation
            # Checks domain/range typing, disjoint envelopes, relation patterns, etc.
            if not self._is_triple_valid((h, rel_id, t)):
                continue

            # Accept valid triple
            self._accept_and_record_triple(
                rel_id=rel_id,
                head_id=h,
                tail_id=t,
                dom_class_ids=dom_class_ids,
                rng_class_ids=rng_class_ids,
                pbar=pbar,
            )

            accepted_this_relation += 1
            accepted_so_far += 1

        return accepted_this_relation

    # ================================================================================================ #
    # CANDIDATE SAMPLING                                                                               #
    # ================================================================================================ #

    def _sample_side_entities_batch(
        self,
        pool: EntityIdArray,
        side_class_ids: frozenset[ClassId],
        batch_size: int,
    ) -> EntityIdArray:
        """Sample entities in batch with freshness bias.

        Prioritizes unseen entities to maximize coverage. Exhausts unseen pool
        first, then falls back to full pool.

        Args:
            pool: Full candidate entity pool for this relation side.
            side_class_ids: Required classes for unseen pool lookup.
            batch_size: Number of entities to sample.

        Returns:
            Array of sampled entity IDs.

        Complexity:
            O(batch_size) for sampling.
        """
        if len(pool) == 0:
            return np.array([], dtype=np.int64)

        # Collect unseen entities from relevant classes
        # sorted for rng seed determinism
        unseen: list[EntityId] = []
        for cid in sorted(side_class_ids):
            unseen.extend(self._entities.class2unseen.get(cid, []))

        if not unseen:
            # No unseen entities available - sample from full pool
            return self._rng.choice(pool, size=batch_size, replace=True)

        unseen_arr = np.asarray(unseen, dtype=np.int64)

        if len(unseen_arr) >= batch_size:
            # Enough unseen to fill entire batch - use exclusively
            return self._rng.choice(unseen_arr, size=batch_size, replace=False)

        # Not enough unseen - use ALL unseen + fill remainder from pool
        chosen_unseen = self._rng.choice(unseen_arr, size=len(unseen_arr), replace=False)
        n_remaining = batch_size - len(unseen_arr)
        chosen_from_pool = self._rng.choice(pool, size=n_remaining, replace=True)
        return np.concatenate([chosen_unseen, chosen_from_pool])

    # ================================================================================================ #
    # FAST CONSTRAINT FILTERING                                                                        #
    # ================================================================================================ #

    def _build_fast_local_constraint_mask_for_relation(
        self,
        relation_id: RelationId,
        candidate_heads: EntityIdArray,
        candidate_tails: EntityIdArray,
    ) -> BooleanMask:
        """Build boolean mask for fast batch filtering of constraint violations.

        Phase 1 of two-phase validation. Applies constraints in sequence:
            1. Irreflexive (vectorized)
            2. Duplicate detection
            3. Symmetric duplicate detection
            4. Asymmetric constraint
            5. Functional property
            6. Inverse-functional property

        Args:
            relation_id: Relation being sampled.
            candidate_heads: Candidate head entity IDs.
            candidate_tails: Candidate tail entity IDs.

        Returns:
            Boolean mask where True means candidate survives all fast checks.

        Complexity:
            O(batch_size + m) where m is existing pairs for this relation.
        """
        n = len(candidate_heads)
        if n == 0:
            return np.zeros(0, dtype=bool)

        # Initialize mask: all candidates start as valid
        mask = np.ones(n, dtype=bool)

        # Apply constraint filters in sequence (each modifies mask in-place)
        self._apply_irreflexive_mask(mask, relation_id, candidate_heads, candidate_tails)
        self._apply_duplicates_mask(mask, relation_id, candidate_heads, candidate_tails)
        self._apply_symmetric_duplicates_mask(mask, relation_id, candidate_heads, candidate_tails)
        self._apply_asymmetric_mask(mask, relation_id, candidate_heads, candidate_tails)
        self._apply_functional_mask(mask, relation_id, candidate_heads)
        self._apply_inverse_functional_mask(mask, relation_id, candidate_tails)

        return mask

    def _apply_irreflexive_mask(
        self,
        mask: BooleanMask,
        relation_id: RelationId,
        candidate_heads: EntityIdArray,
        candidate_tails: EntityIdArray,
    ) -> None:
        """Apply irreflexive constraint to mask.

        Filters out self-loops (e, r, e) for irreflexive relations. Vectorized
        O(n) operation using NumPy array comparison.

        Args:
            mask: Boolean mask to modify in-place.
            relation_id: Relation being validated.
            candidate_heads: Candidate head entity IDs.
            candidate_tails: Candidate tail entity IDs.

        Complexity:
            O(batch_size) vectorized.
        """
        if relation_id in self._constraints.irreflex_ids:
            mask &= candidate_heads != candidate_tails

    def _apply_duplicates_mask(
        self,
        mask: BooleanMask,
        relation_id: RelationId,
        candidate_heads: EntityIdArray,
        candidate_tails: EntityIdArray,
    ) -> None:
        """Apply duplicate detection to mask.

        Filters out pairs already accepted in current generation session.

        Args:
            mask: Boolean mask to modify in-place.
            relation_id: Relation being validated.
            candidate_heads: Candidate head entity IDs.
            candidate_tails: Candidate tail entity IDs.

        Complexity:
            O(batch_size) with O(1) set lookups.
        """
        existing = self._seen_pairs.get(relation_id, set())
        if not existing:
            return

        n = len(candidate_heads)
        for i in range(n):
            if not mask[i]:
                continue
            if (int(candidate_heads[i]), int(candidate_tails[i])) in existing:
                mask[i] = False

    def _apply_symmetric_duplicates_mask(
        self,
        mask: BooleanMask,
        relation_id: RelationId,
        candidate_heads: EntityIdArray,
        candidate_tails: EntityIdArray,
    ) -> None:
        """Apply symmetric duplicate detection to mask.

        For symmetric properties, (h, r, t) and (t, r, h) represent the same fact.
        Rejects candidates where the reverse triple already exists.

        Args:
            mask: Boolean mask to modify in-place.
            relation_id: Relation being validated.
            candidate_heads: Candidate head entity IDs.
            candidate_tails: Candidate tail entity IDs.

        Complexity:
            O(batch_size) with O(1) set lookups.
        """
        if relation_id not in self._constraints.symmetric_ids:
            return

        existing_pairs = self._triples.kg_pairs_by_rid[relation_id]
        if not existing_pairs:
            return

        n = len(candidate_heads)
        for i in range(n):
            if not mask[i]:
                continue
            # Check if reverse triple exists
            if (int(candidate_tails[i]), int(candidate_heads[i])) in existing_pairs:
                mask[i] = False

    def _apply_asymmetric_mask(
        self,
        mask: BooleanMask,
        relation_id: RelationId,
        candidate_heads: EntityIdArray,
        candidate_tails: EntityIdArray,
    ) -> None:
        """Apply asymmetric constraint to mask.

        For asymmetric properties, if (h, r, t) exists then (t, r, h) cannot exist.
        Rejects candidates where reverse triple exists.

        Args:
            mask: Boolean mask to modify in-place.
            relation_id: Relation being validated.
            candidate_heads: Candidate head entity IDs.
            candidate_tails: Candidate tail entity IDs.

        Complexity:
            O(batch_size) with O(1) set lookups.
        """
        if relation_id not in self._constraints.asym_ids:
            return

        existing_pairs = self._triples.kg_pairs_by_rid[relation_id]
        if not existing_pairs:
            return

        n = len(candidate_heads)
        for i in range(n):
            if not mask[i]:
                continue
            # Check if reverse triple exists (would violate asymmetry)
            if (int(candidate_tails[i]), int(candidate_heads[i])) in existing_pairs:
                mask[i] = False

    def _apply_functional_mask(
        self,
        mask: BooleanMask,
        relation_id: RelationId,
        candidate_heads: EntityIdArray,
    ) -> None:
        """Apply functional property constraint to mask.

        Filters out candidates where head entity already used in this relation.
        Functional properties allow at most one triple per head. For symmetric
        relations, also checks if head appears as tail.

        Uses incremental tracking for O(1) lookups.

        Args:
            mask: Boolean mask to modify in-place.
            relation_id: Relation being validated.
            candidate_heads: Candidate head entity IDs.

        Complexity:
            O(batch_size) with O(1) set lookups.
        """
        if relation_id not in self._constraints.functional_ids:
            return

        # Use pre-maintained incremental set for O(1) lookups
        existing_heads = self._triples.functional_heads.get(relation_id, set())

        # Check if candidate heads are already used as heads
        n = len(candidate_heads)
        for i in range(n):
            if not mask[i]:
                continue
            if int(candidate_heads[i]) in existing_heads:
                mask[i] = False

        # For symmetric relations, also check if candidate heads appear as tails
        # Uses incremental tracking instead of O(n) set reconstruction
        symmetric_tails = self._symmetric_functional_tails.get(relation_id)
        if symmetric_tails:
            for i in range(n):
                if not mask[i]:
                    continue
                if int(candidate_heads[i]) in symmetric_tails:
                    mask[i] = False

    def _apply_inverse_functional_mask(
        self,
        mask: BooleanMask,
        relation_id: RelationId,
        candidate_tails: EntityIdArray,
    ) -> None:
        """Apply inverse-functional property constraint to mask.

        Filters out candidates where tail entity already used in this relation.
        Inverse-functional properties allow at most one triple per tail. For
        symmetric relations, also checks if tail appears as head.

        Uses incremental tracking for O(1) lookups.

        Args:
            mask: Boolean mask to modify in-place.
            relation_id: Relation being validated.
            candidate_tails: Candidate tail entity IDs.

        Complexity:
            O(batch_size) with O(1) set lookups.
        """
        if relation_id not in self._constraints.invfunctional_ids:
            return

        # Use pre-maintained incremental set for O(1) lookups
        existing_tails = self._triples.invfunctional_tails.get(relation_id, set())

        # Check if candidate tails are already used as tails
        n = len(candidate_tails)
        for i in range(n):
            if not mask[i]:
                continue
            if int(candidate_tails[i]) in existing_tails:
                mask[i] = False

        # For symmetric relations, also check if candidate tails appear as heads
        # Uses incremental tracking instead of O(n) set reconstruction
        symmetric_heads = self._symmetric_invfunctional_heads.get(relation_id)
        if symmetric_heads:
            for i in range(n):
                if not mask[i]:
                    continue
                if int(candidate_tails[i]) in symmetric_heads:
                    mask[i] = False

    # ================================================================================================ #
    # DEEP CONSTRAINT VALIDATION                                                                       #
    # ================================================================================================ #

    def _is_triple_valid(self, triple: Triple) -> bool:
        """Validate triple against all ontology constraints.

        Phase 2 of two-phase validation, called on survivors from fast filtering.
        Checks constraints in order of increasing cost with early returns on failure:
            1. Domain/range typing
            2. Domain/range disjoint envelopes
            3. Inverse domain/range disjoint
            4. Inverse/asymmetry interaction
            5. Relation pattern constraints
            6. Transitive cycle detection
            7. Subproperty inherited constraints

        Args:
            triple: (head_id, relation_id, tail_id) as integers.

        Returns:
            True if triple satisfies all constraints.

        Complexity:
            O(1) best case if early constraint fails. O(V + E) worst case for
            transitive cycle detection via BFS.
        """
        # 1. Domain/range typing constraints
        if not self._check_domain_range_typing(triple):
            return False

        # 2. Domain/range disjoint envelope constraints
        if not self._check_domain_range_disjoint_envelopes(triple):
            return False

        # 3. Inverse-side domain/range disjoint constraints
        if not self._check_inverse_domain_range_disjoint(triple):
            return False

        # 4. Inverse/asymmetry interaction check
        if not self._check_inverse_asymmetry_interaction(triple):
            return False

        # 5. Relation pattern constraints
        # Re-validates against current KG state (may have changed since fast mask)
        if not self._check_relation_pattern_constraints(triple):
            return False

        # 6. Transitive cycle detection (only for transitive + irreflexive/asymmetric)
        if not self._check_transitive_cycle_constraint(triple):
            return False

        # 7. Subproperty inherited constraints (from super-properties)
        return self._check_subproperty_inherited_constraints(triple)

    def _check_domain_range_typing(self, triple: Triple) -> bool:
        """Check if triple satisfies domain and range typing constraints.

        Validates that head entity has all required domain classes and tail entity
        has all required range classes (conjunctive semantics). Untyped entities
        bypass validation.

        Args:
            triple: (head_id, relation_id, tail_id) as integers.

        Returns:
            True if typing constraints satisfied.

        Complexity:
            O(1) set subset check.
        """
        head_id, relation_id, tail_id = triple

        dom_required = self._constraints.rel2dom_ids[relation_id]
        rng_required = self._constraints.rel2range_ids[relation_id]

        # Check domain constraint - skip validation for untyped entities
        if dom_required:
            if head_id not in self._entities.typed_entities:
                # Untyped entity - no types to validate, skip check
                pass
            else:
                # Typed entity - must satisfy all required domain classes
                head_classes = self._entities.ent2classes_transitive_sets[head_id]
                if not dom_required.issubset(head_classes):
                    return False

        # Check range constraint - skip validation for untyped entities
        if rng_required:
            if tail_id not in self._entities.typed_entities:
                # Untyped entity - no types to validate, skip check
                pass
            else:
                # Typed entity - must satisfy all required range classes
                tail_classes = self._entities.ent2classes_transitive_sets[tail_id]
                if not rng_required.issubset(tail_classes):
                    return False

        return True

    def _check_domain_range_disjoint_envelopes(self, triple: Triple) -> bool:
        """Check if triple violates domain or range disjoint envelope constraints.

        Validates that entity classes don't conflict with classes disjoint from
        the relation's domain or range. Untyped entities bypass validation.

        Args:
            triple: (head_id, relation_id, tail_id) as integers.

        Returns:
            True if no disjoint envelope violations.

        Complexity:
            O(1) set intersection check.
        """
        head_id, relation_id, tail_id = triple

        dom_forbidden = self._constraints.dom_disjoint_envelope[relation_id]
        rng_forbidden = self._constraints.range_disjoint_envelope[relation_id]

        # Check domain disjoint envelope - skip for untyped entities
        if dom_forbidden:
            if head_id not in self._entities.typed_entities:
                # Untyped entity - no types to check against forbidden classes
                pass
            else:
                head_classes = self._entities.ent2classes_transitive_sets[head_id]
                if head_classes & dom_forbidden:
                    return False

        # Check range disjoint envelope - skip for untyped entities
        if rng_forbidden:
            if tail_id not in self._entities.typed_entities:
                # Untyped entity - no types to check against forbidden classes
                pass
            else:
                tail_classes = self._entities.ent2classes_transitive_sets[tail_id]
                if tail_classes & rng_forbidden:
                    return False

        return True

    def _check_inverse_domain_range_disjoint(self, triple: Triple) -> bool:
        """Check if triple violates inverse relation's domain/range disjoint constraints.

        For relations with inverses, validates that the implied inverse triple
        (t, inv, h) would not violate disjoint envelopes. Untyped entities bypass
        validation.

        Args:
            triple: (head_id, relation_id, tail_id) as integers.

        Returns:
            True if no inverse disjoint violations.

        Complexity:
            O(1) set intersection check.
        """
        head_id, relation_id, tail_id = triple

        # Only check if relation has an inverse
        inv_rel_id = self._constraints.rel2inverse_ids.get(relation_id)
        if inv_rel_id is None:
            return True

        # In inverse triple (t, inv, h), tail becomes domain
        inv_dom_forbidden = self._constraints.dom_disjoint_envelope[inv_rel_id]
        if inv_dom_forbidden:
            if tail_id not in self._entities.typed_entities:
                # Untyped entity - skip validation
                pass
            else:
                tail_classes = self._entities.ent2classes_transitive_sets[tail_id]
                if tail_classes & inv_dom_forbidden:
                    return False

        # In inverse triple (t, inv, h), head becomes range
        inv_rng_forbidden = self._constraints.range_disjoint_envelope[inv_rel_id]
        if inv_rng_forbidden:
            if head_id not in self._entities.typed_entities:
                # Untyped entity - skip validation
                pass
            else:
                head_classes = self._entities.ent2classes_transitive_sets[head_id]
                if head_classes & inv_rng_forbidden:
                    return False

        return True

    def _check_inverse_asymmetry_interaction(self, triple: Triple) -> bool:
        """Check if triple conflicts with inverse relation due to asymmetry.

        When relation or its inverse is asymmetric, ensures adding (h, r, t) won't
        conflict with existing (t, inv(r), h).

        Args:
            triple: (head_id, relation_id, tail_id) as integers.

        Returns:
            True if no inverse/asymmetry conflict.

        Complexity:
            O(1) set membership check.
        """
        head_id, relation_id, tail_id = triple

        inv_id = self._constraints.rel2inverse_ids.get(relation_id)
        if inv_id is None:
            return True

        # Only check if either relation or its inverse is asymmetric
        if relation_id in self._constraints.asym_ids or inv_id in self._constraints.asym_ids:
            inv_pairs = self._triples.kg_pairs_by_rid.get(inv_id, set())
            if (tail_id, head_id) in inv_pairs:
                return False

        return True

    def _check_relation_pattern_constraints(self, triple: Triple) -> bool:
        """Check if triple satisfies relation pattern constraints.

        Validates OWL property characteristics against current KG state:
            - Irreflexive: no self-loops
            - Asymmetric: no bidirectional edges
            - Functional: one outgoing edge per head
            - Inverse-functional: one incoming edge per tail
            - Disjoint relations: no overlap with disjoint relation instances

        Args:
            triple: (head_id, relation_id, tail_id) as integers.

        Returns:
            True if all pattern constraints satisfied.

        Complexity:
            O(1) for most checks, O(disjoint_count) for disjoint relations.
        """
        head_id, relation_id, tail_id = triple

        # Irreflexive check (should already be caught by fast mask, but belt-and-suspenders)
        if relation_id in self._constraints.irreflex_ids and head_id == tail_id:
            return False

        # Asymmetric check (re-validate against current KG state)
        if relation_id in self._constraints.asym_ids:
            existing_pairs = self._triples.kg_pairs_by_rid.get(relation_id, set())
            if (tail_id, head_id) in existing_pairs:
                return False

        # Functional check (re-validate against current KG state)
        if relation_id in self._constraints.functional_ids:
            # Use incremental set for O(1) lookup
            if head_id in self._triples.functional_heads.get(relation_id, set()):
                return False
            # For symmetric relations, also check if head appears as tail
            # Uses incremental tracking instead of O(n) scan
            if head_id in self._symmetric_functional_tails.get(relation_id, set()):
                return False

        # Inverse-functional check
        if relation_id in self._constraints.invfunctional_ids:
            # Use incremental set for O(1) lookup
            if tail_id in self._triples.invfunctional_tails.get(relation_id, set()):
                return False
            # For symmetric relations, also check if tail appears as head
            # Uses incremental tracking instead of O(n) scan
            if tail_id in self._symmetric_invfunctional_heads.get(relation_id, set()):
                return False

        # Disjoint relations check
        disjoint_rels = self._constraints.rel2disjoints_extended_ids[relation_id]
        if disjoint_rels:
            is_sym = relation_id in self._constraints.symmetric_ids
            for drel in disjoint_rels:
                dpairs = self._triples.kg_pairs_by_rid.get(drel, set())
                if (head_id, tail_id) in dpairs:
                    return False
                d_is_sym = drel in self._constraints.symmetric_ids
                if (is_sym or d_is_sym) and (tail_id, head_id) in dpairs:
                    return False

        return True

    def _check_subproperty_inherited_constraints(self, triple: Triple) -> bool:
        """Check if triple respects inherited constraints from super-properties.

        By RDFS/OWL semantics, (x, P, y) implies (x, Q, y) for all super-properties Q.
        Validates functional, inverse-functional, asymmetric, irreflexive, and
        disjoint constraints from the entire super-property chain.

        Domain/range are NOT re-checked since domain(P) is a subset of domain(Q).

        Args:
            triple: (head_id, relation_id, tail_id) as integers.

        Returns:
            True if all super-property constraints satisfied.

        Complexity:
            O(k) where k is super-property chain length. All per-property checks
            are O(1) using incremental tracking.
        """
        head_id, relation_id, tail_id = triple

        # Get all super-properties in hierarchy
        super_ids = self._constraints.rel2superrel_ids.get(relation_id, [])
        if not super_ids:
            return True  # No super-properties, nothing to check

        # Validate against each super-property's constraints
        for super_id in super_ids:
            # 1. Functional constraint (inherited)
            # Uses O(1) incremental tracking instead of O(n) scan
            if super_id in self._constraints.functional_ids:
                # Check if head already used in super-property (includes inferred triples)
                if head_id in self._triples.functional_heads.get(super_id, set()):
                    return False

            # 2. Inverse-functional constraint (inherited)
            # Uses O(1) incremental tracking instead of O(n) scan
            if super_id in self._constraints.invfunctional_ids:
                # Check if tail already used in super-property (includes inferred triples)
                if tail_id in self._triples.invfunctional_tails.get(super_id, set()):
                    return False

            # 3. Asymmetric constraint (inherited)
            if super_id in self._constraints.asym_ids:
                # Self-loop violates asymmetry
                if head_id == tail_id:
                    return False
                # Check reverse doesn't exist in super-property
                super_pairs = self._triples.kg_pairs_by_rid.get(super_id, set())
                if (tail_id, head_id) in super_pairs:
                    return False
                # Check reverse doesn't exist in current relation (would be inferred to super)
                current_pairs = self._triples.kg_pairs_by_rid.get(relation_id, set())
                if (tail_id, head_id) in current_pairs:
                    return False

            # 4. Irreflexive constraint (inherited)
            if super_id in self._constraints.irreflex_ids and head_id == tail_id:
                return False

            # 5. Disjoint properties (inherited)
            # If super-property is disjoint with R, then sub-property is also disjoint with R
            super_disjoints = self._constraints.rel2disjoints_extended_ids[super_id]
            if super_disjoints:
                is_super_sym = super_id in self._constraints.symmetric_ids
                for drel in super_disjoints:
                    dpairs = self._triples.kg_pairs_by_rid.get(drel, set())
                    if (head_id, tail_id) in dpairs:
                        return False
                    d_is_sym = drel in self._constraints.symmetric_ids
                    if (is_super_sym or d_is_sym) and (tail_id, head_id) in dpairs:
                        return False

        return True

    def _check_transitive_cycle_constraint(self, triple: Triple) -> bool:
        """Check if adding triple would create a cycle in transitive relation.

        Only applies to transitive relations that are also irreflexive or asymmetric:
            - Transitive + Irreflexive: cycle implies self-loop via transitive closure
            - Transitive + Asymmetric: cycle implies bidirectional path

        Uses BFS with adjacency list for O(V+E) cycle detection.

        Args:
            triple: (head_id, relation_id, tail_id) as integers.

        Returns:
            True if no cycle detected or check not applicable.

        Complexity:
            O(1) early exit for non-transitive or safe combinations.
            O(V + E) BFS for transitive + irreflexive/asymmetric combinations.
        """
        head_id, relation_id, tail_id = triple

        # Early exit: only check relations with adjacency tracking
        # (transitive + irreflexive/asymmetric)
        adjacency = self._transitive_adjacency.get(relation_id)
        if adjacency is None:
            return True

        # BFS from tail to see if we can reach head
        visited: set[EntityId] = set()
        queue = deque([tail_id])

        while queue:
            current = queue.popleft()

            # Found a path from tail to head -> adding (head, r, tail) creates cycle!
            if current == head_id:
                return False

            if current in visited:
                continue
            visited.add(current)

            # Follow outgoing edges using adjacency list - O(degree) not O(E)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        # No path from tail to head -> no cycle -> valid triple
        return True

    # ================================================================================================ #
    # BOOKKEEPING                                                                                      #
    # ================================================================================================ #

    def _accept_and_record_triple(
        self,
        rel_id: RelationId,
        head_id: EntityId,
        tail_id: EntityId,
        dom_class_ids: frozenset[ClassId],
        rng_class_ids: frozenset[ClassId],
        pbar: TqdmProgressBar,
    ) -> None:
        """Accept a triple and update all tracking state.

        Updates:
            - KG storage (kg_pairs_by_rid)
            - Duplicate tracking (seen_pairs, symmetric reverse)
            - Functional/inverse-functional head/tail tracking (including super-properties)
            - Symmetric participant tracking
            - Transitive adjacency list (for cycle detection)
            - Freshness pools (drains unseen entities)
            - Progress bar

        Args:
            rel_id: Relation ID.
            head_id: Head entity ID.
            tail_id: Tail entity ID.
            dom_class_ids: Domain class IDs for freshness tracking.
            rng_class_ids: Range class IDs for freshness tracking.
            pbar: Progress bar to update.

        Complexity:
            O(d + r + s) where d and r are domain/range class counts and s is
            super-property count.
        """
        # Add to KG (only one direction, even for symmetric properties)
        self._triples.kg_pairs_by_rid[rel_id].add((head_id, tail_id))

        # Update duplicate tracking
        self._seen_pairs[rel_id].add((head_id, tail_id))

        # For symmetric relations, also track reverse direction to prevent duplicates
        # (but don't add reverse to the actual KG)
        if rel_id in self._constraints.symmetric_ids:
            self._seen_pairs[rel_id].add((tail_id, head_id))

        # Update functional tracking
        if rel_id in self._constraints.functional_ids:
            self._triples.functional_heads[rel_id].add(head_id)
            # For symmetric: tail also acts as head
            if rel_id in self._symmetric_functional_tails:
                self._symmetric_functional_tails[rel_id].add(tail_id)

        # Update inverse-functional tracking
        if rel_id in self._constraints.invfunctional_ids:
            self._triples.invfunctional_tails[rel_id].add(tail_id)
            # For symmetric: head also acts as tail
            if rel_id in self._symmetric_invfunctional_heads:
                self._symmetric_invfunctional_heads[rel_id].add(head_id)

        # Update functional/inverse-functional tracking for super-properties
        # By RDFS semantics, (h, rel_id, t) implies (h, super_id, t) for all super-properties
        for super_id in self._constraints.rel2superrel_ids.get(rel_id, []):
            if super_id in self._constraints.functional_ids:
                self._triples.functional_heads[super_id].add(head_id)
            if super_id in self._constraints.invfunctional_ids:
                self._triples.invfunctional_tails[super_id].add(tail_id)

        # Update transitive adjacency list for O(V+E) cycle detection
        if rel_id in self._transitive_adjacency:
            if head_id not in self._transitive_adjacency[rel_id]:
                self._transitive_adjacency[rel_id][head_id] = set()
            self._transitive_adjacency[rel_id][head_id].add(tail_id)

        # Update progress bar
        pbar.update(1)

        # Mark entities as seen (drain unseen pools for freshness bias)
        for cid in dom_class_ids:
            unseen = self._entities.class2unseen.get(cid)
            if unseen and head_id in unseen:
                unseen.remove(head_id)
        for cid in rng_class_ids:
            unseen = self._entities.class2unseen.get(cid)
            if unseen and tail_id in unseen:
                unseen.remove(tail_id)
