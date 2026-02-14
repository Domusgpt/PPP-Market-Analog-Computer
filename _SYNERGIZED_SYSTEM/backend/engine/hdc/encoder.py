"""
HDC Encoder - Convert Moiré Patterns to Hypervectors
====================================================

Transforms 2D moiré patterns into high-dimensional binary vectors
that preserve spatial and temporal structure.

Encoding strategy:
1. Spatial: Grid positions bound with quantized intensity values
2. Temporal: Sequence positions encoded via permutation
3. Evolution: Cascade dynamics encoded as temporal derivative
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from .operations import (
    random_hypervector,
    bind,
    bundle,
    permute,
    similarity,
    threshold,
    ItemMemory,
    DEFAULT_DIM
)


@dataclass
class HDCConfig:
    """Configuration for HDC encoder."""
    dim: int = 10000              # Hypervector dimensionality
    num_levels: int = 32          # Quantization levels for intensity
    spatial_sample: int = 16      # Sample grid size (16x16 = 256 positions)
    sequence_length: int = 10     # Max sequence length to encode
    use_local_features: bool = True   # Include local texture features
    normalize_input: bool = True      # Normalize input patterns


@dataclass
class HDCResult:
    """Result of HDC encoding."""
    hypervector: np.ndarray       # The encoded hypervector
    spatial_hv: np.ndarray        # Spatial component
    temporal_hv: Optional[np.ndarray]  # Temporal component (if sequence)
    metadata: Dict                # Encoding metadata


class HDCEncoder:
    """
    Hyperdimensional encoder for moiré patterns.

    Converts 2D spatial patterns and temporal sequences into
    high-dimensional binary vectors that support:
    - Fast similarity matching
    - Temporal sequence encoding
    - Associative memory operations

    Parameters
    ----------
    config : HDCConfig
        Encoder configuration

    Example
    -------
    >>> encoder = HDCEncoder()
    >>> # Encode single pattern
    >>> hv = encoder.encode_pattern(moire_pattern)
    >>> # Encode sequence
    >>> seq_hv = encoder.encode_sequence([pattern1, pattern2, pattern3])
    >>> # Check similarity
    >>> sim = encoder.similarity(hv1, hv2)
    """

    def __init__(self, config: Optional[HDCConfig] = None):
        self.config = config or HDCConfig()
        self.dim = self.config.dim

        # Item memories for consistent encoding
        self._position_memory = ItemMemory(self.dim)
        self._level_memory = ItemMemory(self.dim)

        # Pre-compute position hypervectors for spatial grid
        self._init_spatial_basis()

        # Sequence history for temporal encoding
        self._sequence_history: List[np.ndarray] = []

    def _init_spatial_basis(self):
        """Initialize basis hypervectors for spatial positions."""
        n = self.config.spatial_sample
        self._position_hvs = {}

        for y in range(n):
            for x in range(n):
                # Create unique hypervector for each position
                pos_key = f"pos_{y}_{x}"
                self._position_hvs[(y, x)] = self._position_memory.get(pos_key)

        # Level hypervectors (quantized intensity)
        self._level_hvs = []
        for level in range(self.config.num_levels):
            self._level_hvs.append(
                self._level_memory.get_level(level, self.config.num_levels)
            )

    def encode_pattern(self, pattern: np.ndarray) -> HDCResult:
        """
        Encode a single 2D moiré pattern into a hypervector.

        Encoding strategy:
        1. Downsample pattern to spatial_sample x spatial_sample
        2. Quantize intensities to num_levels
        3. Bind each position with its intensity level
        4. Bundle all position-value pairs

        Parameters
        ----------
        pattern : np.ndarray
            2D pattern array (any size, will be resampled)

        Returns
        -------
        HDCResult
            Encoded hypervector with metadata
        """
        # Normalize input
        if self.config.normalize_input:
            p_min, p_max = pattern.min(), pattern.max()
            if p_max > p_min:
                pattern = (pattern - p_min) / (p_max - p_min)

        # Downsample to grid
        n = self.config.spatial_sample
        if pattern.shape != (n, n):
            from scipy.ndimage import zoom
            factors = (n / pattern.shape[0], n / pattern.shape[1])
            pattern = zoom(pattern, factors, order=1)

        # Quantize intensities
        levels = (pattern * (self.config.num_levels - 1)).astype(int)
        levels = np.clip(levels, 0, self.config.num_levels - 1)

        # Encode each position-value pair
        position_value_hvs = []

        for y in range(n):
            for x in range(n):
                level = levels[y, x]
                pos_hv = self._position_hvs[(y, x)]
                level_hv = self._level_hvs[level]

                # Bind position with value
                pv_hv = bind(pos_hv, level_hv)
                position_value_hvs.append(pv_hv)

        # Bundle all position-value pairs
        spatial_hv = bundle(position_value_hvs)

        # Optional: Add local features (gradients, edges)
        if self.config.use_local_features:
            feature_hv = self._encode_local_features(pattern)
            combined_hv = bundle([spatial_hv, feature_hv], weights=[0.7, 0.3])
        else:
            combined_hv = spatial_hv

        return HDCResult(
            hypervector=combined_hv,
            spatial_hv=spatial_hv,
            temporal_hv=None,
            metadata={
                'encoding_type': 'spatial',
                'grid_size': n,
                'num_levels': self.config.num_levels,
                'pattern_mean': float(np.mean(pattern)),
                'pattern_std': float(np.std(pattern))
            }
        )

    def _encode_local_features(self, pattern: np.ndarray) -> np.ndarray:
        """Encode local texture features (gradients, edges)."""
        # Compute gradients
        gy, gx = np.gradient(pattern)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        gradient_dir = np.arctan2(gy, gx)

        # Quantize gradient features
        n = self.config.spatial_sample
        feature_hvs = []

        # Gradient magnitude hypervector
        mag_levels = (gradient_mag / (gradient_mag.max() + 1e-8) *
                     (self.config.num_levels - 1)).astype(int)
        mag_levels = np.clip(mag_levels, 0, self.config.num_levels - 1)

        for y in range(n):
            for x in range(n):
                level = mag_levels[y, x]
                pos_hv = self._position_hvs[(y, x)]
                level_hv = self._level_hvs[level]
                feature_hvs.append(bind(pos_hv, level_hv))

        return bundle(feature_hvs)

    def encode_sequence(
        self,
        patterns: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> HDCResult:
        """
        Encode a temporal sequence of patterns.

        Uses permutation to encode temporal position:
        sequence_hv = bundle([
            pattern[0],
            permute(pattern[1], 1),
            permute(pattern[2], 2),
            ...
        ])

        This ensures different orderings produce different hypervectors.

        Parameters
        ----------
        patterns : List[np.ndarray]
            Sequence of 2D patterns (temporal order matters)
        weights : List[float], optional
            Temporal weights (default: recent = higher weight)

        Returns
        -------
        HDCResult
            Sequence hypervector
        """
        if len(patterns) == 0:
            raise ValueError("Cannot encode empty sequence")

        # Encode each pattern
        pattern_hvs = []
        for pattern in patterns:
            result = self.encode_pattern(pattern)
            pattern_hvs.append(result.hypervector)

        # Apply temporal permutation
        temporal_hvs = []
        for t, hv in enumerate(pattern_hvs):
            # Permute by temporal position
            temporal_hv = permute(hv, t)
            temporal_hvs.append(temporal_hv)

        # Default weights: exponential decay (recent = more important)
        if weights is None:
            decay = 0.8
            weights = [decay ** (len(patterns) - 1 - t) for t in range(len(patterns))]

        # Bundle with temporal weighting
        sequence_hv = bundle(temporal_hvs, weights)

        # Also keep pure spatial component (last frame)
        spatial_hv = pattern_hvs[-1] if pattern_hvs else None

        return HDCResult(
            hypervector=sequence_hv,
            spatial_hv=spatial_hv,
            temporal_hv=sequence_hv,
            metadata={
                'encoding_type': 'temporal_sequence',
                'sequence_length': len(patterns),
                'weights': weights
            }
        )

    def encode_streaming(self, pattern: np.ndarray) -> HDCResult:
        """
        Encode pattern in streaming mode with history.

        Maintains internal sequence buffer and encodes
        the recent temporal context.

        Parameters
        ----------
        pattern : np.ndarray
            Current frame

        Returns
        -------
        HDCResult
            Hypervector with temporal context
        """
        # Add to history
        self._sequence_history.append(pattern.copy())

        # Trim to max length
        if len(self._sequence_history) > self.config.sequence_length:
            self._sequence_history.pop(0)

        # Encode full sequence
        return self.encode_sequence(self._sequence_history)

    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """
        Compute similarity between two hypervectors.

        Parameters
        ----------
        hv1, hv2 : np.ndarray
            Hypervectors to compare

        Returns
        -------
        float
            Cosine similarity [-1, 1]
        """
        return similarity(hv1, hv2, method='cosine')

    def batch_similarity(
        self,
        query: np.ndarray,
        candidates: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute similarity of query against multiple candidates.

        Parameters
        ----------
        query : np.ndarray
            Query hypervector
        candidates : List[np.ndarray]
            List of candidate hypervectors

        Returns
        -------
        np.ndarray
            Similarity scores for each candidate
        """
        return np.array([self.similarity(query, c) for c in candidates])

    def find_best_match(
        self,
        query: np.ndarray,
        candidates: List[np.ndarray],
        threshold: float = 0.3
    ) -> Tuple[int, float]:
        """
        Find best matching candidate above threshold.

        Parameters
        ----------
        query : np.ndarray
            Query hypervector
        candidates : List[np.ndarray]
            Candidate hypervectors
        threshold : float
            Minimum similarity threshold

        Returns
        -------
        Tuple[int, float]
            (best_index, similarity) or (-1, 0.0) if no match
        """
        if not candidates:
            return -1, 0.0

        similarities = self.batch_similarity(query, candidates)
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        if best_sim >= threshold:
            return int(best_idx), float(best_sim)
        return -1, 0.0

    def reset_streaming(self):
        """Clear streaming history."""
        self._sequence_history.clear()

    def get_history_length(self) -> int:
        """Get current streaming history length."""
        return len(self._sequence_history)
