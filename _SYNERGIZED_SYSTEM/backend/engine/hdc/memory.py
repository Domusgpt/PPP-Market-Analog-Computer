"""
Associative Memory - HDC Pattern Storage and Retrieval
======================================================

Implements associative memory for storing and retrieving
hypervector patterns. Key use cases:
- SLAM loop closure detection ("have I been here before?")
- Pattern classification (one-shot learning)
- Anomaly detection (is this similar to known patterns?)
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .operations import similarity, bundle, DEFAULT_DIM


class MatchType(Enum):
    """Type of memory match."""
    EXACT = "exact"           # High confidence match
    SIMILAR = "similar"       # Moderate similarity
    NOVEL = "novel"           # No good match (new pattern)


@dataclass
class MemoryResult:
    """Result of a memory query."""
    match_type: MatchType
    best_match_idx: int          # Index of best match (-1 if novel)
    best_match_label: Any        # Label of best match (None if novel)
    similarity: float            # Similarity score
    all_similarities: np.ndarray # Similarities to all stored patterns
    is_loop_closure: bool        # True if high-confidence match to past


@dataclass
class MemoryEntry:
    """Single entry in associative memory."""
    hypervector: np.ndarray
    label: Any
    timestamp: int
    metadata: Dict


class AssociativeMemory:
    """
    Associative memory for hypervector storage and retrieval.

    Supports:
    - One-shot learning (add pattern, immediately retrievable)
    - Soft matching (find most similar, even if imperfect)
    - Prototype learning (bundle similar patterns into class prototype)
    - Loop closure detection (high-confidence "seen before" check)

    Parameters
    ----------
    dim : int
        Hypervector dimensionality
    exact_threshold : float
        Similarity threshold for "exact" match (default: 0.7)
    similar_threshold : float
        Similarity threshold for "similar" match (default: 0.4)
    max_entries : int
        Maximum memory entries (0 = unlimited)

    Example
    -------
    >>> memory = AssociativeMemory()
    >>> # Store patterns
    >>> memory.store(hv1, label="location_A")
    >>> memory.store(hv2, label="location_B")
    >>> # Query
    >>> result = memory.query(query_hv)
    >>> if result.is_loop_closure:
    ...     print(f"Loop closure! Back at {result.best_match_label}")
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        exact_threshold: float = 0.7,
        similar_threshold: float = 0.4,
        max_entries: int = 0
    ):
        self.dim = dim
        self.exact_threshold = exact_threshold
        self.similar_threshold = similar_threshold
        self.max_entries = max_entries

        # Storage
        self._entries: List[MemoryEntry] = []
        self._prototypes: Dict[Any, np.ndarray] = {}  # Label -> prototype HV
        self._timestamp = 0

    def store(
        self,
        hypervector: np.ndarray,
        label: Any = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Store a hypervector in memory.

        Parameters
        ----------
        hypervector : np.ndarray
            Pattern to store
        label : Any
            Optional label/identifier
        metadata : Dict
            Optional metadata

        Returns
        -------
        int
            Index of stored entry
        """
        # Check capacity
        if self.max_entries > 0 and len(self._entries) >= self.max_entries:
            # Remove oldest entry
            self._entries.pop(0)

        entry = MemoryEntry(
            hypervector=hypervector.copy(),
            label=label,
            timestamp=self._timestamp,
            metadata=metadata or {}
        )

        self._entries.append(entry)
        self._timestamp += 1

        # Update prototype if labeled
        if label is not None:
            self._update_prototype(label, hypervector)

        return len(self._entries) - 1

    def _update_prototype(self, label: Any, hypervector: np.ndarray):
        """Update class prototype with new example."""
        if label in self._prototypes:
            # Bundle with existing prototype
            self._prototypes[label] = bundle(
                [self._prototypes[label], hypervector],
                weights=[0.9, 0.1]  # Favor existing prototype
            )
        else:
            self._prototypes[label] = hypervector.copy()

    def query(
        self,
        hypervector: np.ndarray,
        min_temporal_gap: int = 0
    ) -> MemoryResult:
        """
        Query memory for similar patterns.

        Parameters
        ----------
        hypervector : np.ndarray
            Query pattern
        min_temporal_gap : int
            Minimum timestamp gap for loop closure
            (prevents matching very recent entries)

        Returns
        -------
        MemoryResult
            Query result with match information
        """
        if not self._entries:
            return MemoryResult(
                match_type=MatchType.NOVEL,
                best_match_idx=-1,
                best_match_label=None,
                similarity=0.0,
                all_similarities=np.array([]),
                is_loop_closure=False
            )

        # Compute similarities
        similarities = np.array([
            similarity(hypervector, entry.hypervector, method='cosine')
            for entry in self._entries
        ])

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_sim = similarities[best_idx]
        best_entry = self._entries[best_idx]

        # Check temporal gap for loop closure
        temporal_gap = self._timestamp - best_entry.timestamp
        is_loop_closure = (
            best_sim >= self.exact_threshold and
            temporal_gap >= min_temporal_gap
        )

        # Determine match type
        if best_sim >= self.exact_threshold:
            match_type = MatchType.EXACT
        elif best_sim >= self.similar_threshold:
            match_type = MatchType.SIMILAR
        else:
            match_type = MatchType.NOVEL

        return MemoryResult(
            match_type=match_type,
            best_match_idx=best_idx,
            best_match_label=best_entry.label,
            similarity=float(best_sim),
            all_similarities=similarities,
            is_loop_closure=is_loop_closure
        )

    def query_prototype(self, hypervector: np.ndarray) -> Tuple[Any, float]:
        """
        Query against class prototypes.

        Useful for classification when you have labeled examples.

        Parameters
        ----------
        hypervector : np.ndarray
            Query pattern

        Returns
        -------
        Tuple[Any, float]
            (best_label, similarity) or (None, 0.0)
        """
        if not self._prototypes:
            return None, 0.0

        best_label = None
        best_sim = -1.0

        for label, prototype in self._prototypes.items():
            sim = similarity(hypervector, prototype, method='cosine')
            if sim > best_sim:
                best_sim = sim
                best_label = label

        return best_label, float(best_sim)

    def find_k_nearest(
        self,
        hypervector: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float, Any]]:
        """
        Find k nearest neighbors.

        Parameters
        ----------
        hypervector : np.ndarray
            Query pattern
        k : int
            Number of neighbors

        Returns
        -------
        List[Tuple[int, float, Any]]
            List of (index, similarity, label) tuples
        """
        if not self._entries:
            return []

        similarities = np.array([
            similarity(hypervector, entry.hypervector, method='cosine')
            for entry in self._entries
        ])

        # Get top-k indices
        k = min(k, len(self._entries))
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            entry = self._entries[idx]
            results.append((
                int(idx),
                float(similarities[idx]),
                entry.label
            ))

        return results

    def detect_anomaly(
        self,
        hypervector: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Check if pattern is anomalous (dissimilar to all stored).

        Parameters
        ----------
        hypervector : np.ndarray
            Pattern to check
        threshold : float
            Anomaly threshold (default: similar_threshold)

        Returns
        -------
        Tuple[bool, float]
            (is_anomaly, max_similarity)
        """
        if threshold is None:
            threshold = self.similar_threshold

        result = self.query(hypervector)

        is_anomaly = result.similarity < threshold
        return is_anomaly, result.similarity

    def get_entry(self, index: int) -> Optional[MemoryEntry]:
        """Get entry by index."""
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    def get_all_labels(self) -> List[Any]:
        """Get all unique labels."""
        labels = set()
        for entry in self._entries:
            if entry.label is not None:
                labels.add(entry.label)
        return list(labels)

    def get_entries_by_label(self, label: Any) -> List[int]:
        """Get indices of all entries with given label."""
        return [
            i for i, entry in enumerate(self._entries)
            if entry.label == label
        ]

    def clear(self):
        """Clear all memory."""
        self._entries.clear()
        self._prototypes.clear()
        self._timestamp = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        return {
            'num_entries': len(self._entries),
            'num_prototypes': len(self._prototypes),
            'labels': self.get_all_labels(),
            'current_timestamp': self._timestamp
        }


class SpatialMemory(AssociativeMemory):
    """
    Specialized memory for spatial/SLAM applications.

    Adds spatial coordinates to entries and supports
    spatial-aware queries.
    """

    def store_with_position(
        self,
        hypervector: np.ndarray,
        position: Tuple[float, float],
        label: Any = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Store pattern with spatial position.

        Parameters
        ----------
        hypervector : np.ndarray
            Pattern to store
        position : Tuple[float, float]
            (x, y) position in world coordinates
        label : Any
            Optional label
        metadata : Dict
            Optional metadata

        Returns
        -------
        int
            Entry index
        """
        meta = metadata or {}
        meta['position'] = position

        return self.store(hypervector, label, meta)

    def query_spatial(
        self,
        hypervector: np.ndarray,
        current_position: Tuple[float, float],
        min_distance: float = 5.0
    ) -> MemoryResult:
        """
        Query with spatial constraint.

        Only considers matches that are at least min_distance away
        from current position (prevents trivial matches).

        Parameters
        ----------
        hypervector : np.ndarray
            Query pattern
        current_position : Tuple[float, float]
            Current (x, y) position
        min_distance : float
            Minimum spatial distance for valid match

        Returns
        -------
        MemoryResult
            Query result
        """
        if not self._entries:
            return MemoryResult(
                match_type=MatchType.NOVEL,
                best_match_idx=-1,
                best_match_label=None,
                similarity=0.0,
                all_similarities=np.array([]),
                is_loop_closure=False
            )

        # Compute similarities with spatial filter
        similarities = []
        for entry in self._entries:
            pos = entry.metadata.get('position')
            if pos is not None:
                dist = np.sqrt(
                    (pos[0] - current_position[0])**2 +
                    (pos[1] - current_position[1])**2
                )
                if dist < min_distance:
                    similarities.append(-1.0)  # Exclude nearby
                    continue

            sim = similarity(hypervector, entry.hypervector, method='cosine')
            similarities.append(sim)

        similarities = np.array(similarities)

        # Find best valid match
        valid_mask = similarities >= 0
        if not np.any(valid_mask):
            return MemoryResult(
                match_type=MatchType.NOVEL,
                best_match_idx=-1,
                best_match_label=None,
                similarity=0.0,
                all_similarities=similarities,
                is_loop_closure=False
            )

        valid_similarities = np.where(valid_mask, similarities, -np.inf)
        best_idx = int(np.argmax(valid_similarities))
        best_sim = similarities[best_idx]
        best_entry = self._entries[best_idx]

        # Determine match type
        if best_sim >= self.exact_threshold:
            match_type = MatchType.EXACT
            is_loop_closure = True
        elif best_sim >= self.similar_threshold:
            match_type = MatchType.SIMILAR
            is_loop_closure = False
        else:
            match_type = MatchType.NOVEL
            is_loop_closure = False

        return MemoryResult(
            match_type=match_type,
            best_match_idx=best_idx,
            best_match_label=best_entry.label,
            similarity=float(best_sim),
            all_similarities=similarities,
            is_loop_closure=is_loop_closure
        )
