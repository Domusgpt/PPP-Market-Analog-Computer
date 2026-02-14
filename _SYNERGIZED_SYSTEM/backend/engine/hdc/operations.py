"""
HDC Operations - Core Hyperdimensional Computing Primitives
===========================================================

Implements the fundamental operations of hyperdimensional computing:
- Random hypervector generation (basis vectors)
- Binding (XOR): Creates associations, reversible
- Bundling (majority): Creates superposition, additive
- Permutation (shift): Encodes sequence/position
- Similarity (cosine/hamming): Measures vector closeness
"""

import numpy as np
from typing import List, Union, Optional
from functools import lru_cache


# Default dimensionality - high enough for good separation
DEFAULT_DIM = 10000


def random_hypervector(dim: int = DEFAULT_DIM, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random binary hypervector.

    In HDC, random vectors are quasi-orthogonal in high dimensions,
    meaning they have ~50% overlap (similarity ≈ 0).

    Parameters
    ----------
    dim : int
        Dimensionality of the hypervector
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Binary hypervector of shape (dim,) with dtype uint8
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    return rng.randint(0, 2, size=dim, dtype=np.uint8)


def bind(hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
    """
    Bind two hypervectors using XOR.

    Binding creates an association between two concepts.
    Key properties:
    - Reversible: bind(bind(A, B), B) = A
    - Distributive over bundling
    - Result is dissimilar to both inputs

    Use cases:
    - Associate position with value: bind(position_hv, value_hv)
    - Create key-value pairs
    - Encode spatial relationships

    Parameters
    ----------
    hv1, hv2 : np.ndarray
        Binary hypervectors to bind

    Returns
    -------
    np.ndarray
        Bound hypervector
    """
    return np.bitwise_xor(hv1, hv2)


def bundle(hypervectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Bundle multiple hypervectors using weighted majority.

    Bundling creates a superposition - the result is similar to all inputs.
    This is how we combine multiple features/elements into one representation.

    Key properties:
    - Commutative: order doesn't matter
    - Result is similar to all inputs
    - More inputs = more robust representation

    Parameters
    ----------
    hypervectors : List[np.ndarray]
        List of binary hypervectors to bundle
    weights : List[float], optional
        Weights for each hypervector (default: equal weights)

    Returns
    -------
    np.ndarray
        Bundled hypervector (binary)
    """
    if len(hypervectors) == 0:
        raise ValueError("Cannot bundle empty list")

    if len(hypervectors) == 1:
        return hypervectors[0].copy()

    dim = len(hypervectors[0])

    if weights is None:
        weights = [1.0] * len(hypervectors)

    # Weighted sum
    weighted_sum = np.zeros(dim, dtype=np.float32)
    total_weight = sum(weights)

    for hv, w in zip(hypervectors, weights):
        weighted_sum += w * hv.astype(np.float32)

    # Majority vote with random tiebreaker
    threshold_val = total_weight / 2
    result = (weighted_sum > threshold_val).astype(np.uint8)

    # Random tiebreaker for exactly 50%
    ties = np.abs(weighted_sum - threshold_val) < 1e-6
    if np.any(ties):
        result[ties] = np.random.randint(0, 2, size=np.sum(ties), dtype=np.uint8)

    return result


def permute(hv: np.ndarray, shifts: int = 1) -> np.ndarray:
    """
    Permute a hypervector by circular shift.

    Permutation encodes position/sequence. Different shift amounts
    produce quasi-orthogonal vectors.

    Key properties:
    - Invertible: permute(permute(A, n), -n) = A
    - Creates new quasi-orthogonal vector
    - Used to encode temporal order

    Sequence encoding:
        sequence(A, B, C) = bundle([A, permute(B, 1), permute(C, 2)])

    Parameters
    ----------
    hv : np.ndarray
        Hypervector to permute
    shifts : int
        Number of positions to shift (can be negative)

    Returns
    -------
    np.ndarray
        Permuted hypervector
    """
    return np.roll(hv, shifts)


def similarity(hv1: np.ndarray, hv2: np.ndarray, method: str = 'cosine') -> float:
    """
    Compute similarity between two hypervectors.

    Parameters
    ----------
    hv1, hv2 : np.ndarray
        Hypervectors to compare
    method : str
        'cosine': Cosine similarity [-1, 1] (default)
        'hamming': Normalized Hamming similarity [0, 1]

    Returns
    -------
    float
        Similarity score
    """
    if method == 'hamming':
        # Hamming similarity = 1 - (hamming_distance / dim)
        matches = np.sum(hv1 == hv2)
        return matches / len(hv1)

    elif method == 'cosine':
        # Convert binary to bipolar (-1, +1) for cosine
        bp1 = 2.0 * hv1.astype(np.float32) - 1.0
        bp2 = 2.0 * hv2.astype(np.float32) - 1.0

        dot = np.dot(bp1, bp2)
        norm1 = np.linalg.norm(bp1)
        norm2 = np.linalg.norm(bp2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def threshold(hv: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    Threshold a real-valued vector to binary.

    Parameters
    ----------
    hv : np.ndarray
        Real-valued vector
    thresh : float
        Threshold value

    Returns
    -------
    np.ndarray
        Binary hypervector
    """
    return (hv > thresh).astype(np.uint8)


def normalize(hv: np.ndarray) -> np.ndarray:
    """
    Normalize a real-valued hypervector to [0, 1].

    Parameters
    ----------
    hv : np.ndarray
        Input vector

    Returns
    -------
    np.ndarray
        Normalized vector in [0, 1]
    """
    hv_min, hv_max = hv.min(), hv.max()
    if hv_max > hv_min:
        return (hv - hv_min) / (hv_max - hv_min)
    return np.zeros_like(hv)


class ItemMemory:
    """
    Item memory for storing basis hypervectors.

    Maps discrete items (symbols, positions, features) to
    random hypervectors with caching for consistency.
    """

    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim
        self._cache = {}
        self._seed_counter = 0

    def get(self, key: str) -> np.ndarray:
        """Get or create hypervector for key."""
        if key not in self._cache:
            self._cache[key] = random_hypervector(self.dim, seed=hash(key) % (2**31))
        return self._cache[key]

    def get_position(self, position: int) -> np.ndarray:
        """Get hypervector for a position (uses permutation of base)."""
        base = self.get("__position_base__")
        return permute(base, position)

    def get_level(self, level: int, num_levels: int = 256) -> np.ndarray:
        """
        Get hypervector for a quantization level.

        Creates interpolated hypervectors for continuous values.
        """
        # Create endpoint hypervectors
        hv_min = self.get("__level_min__")
        hv_max = self.get("__level_max__")

        # Linear interpolation in hypervector space
        t = level / (num_levels - 1) if num_levels > 1 else 0.5

        # Flip random bits proportional to t
        flip_count = int(t * self.dim * 0.5)
        flip_indices = np.random.RandomState(level).choice(
            self.dim, size=flip_count, replace=False
        )

        result = hv_min.copy()
        result[flip_indices] = 1 - result[flip_indices]

        return result

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
