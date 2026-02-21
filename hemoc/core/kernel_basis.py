"""
Kernel (Null Space) of the Phillips Matrix
============================================

The Phillips matrix has rank 4 in 8D, so its kernel is 4-dimensional.
Among these 4 kernel directions:

  - 1 direction (the "collision direction" d = (0,1,0,1,0,1,0,1)/2)
    maps 28 of the 240 E8 roots onto 14 collision pairs.

  - 3 directions are "clean" -- they do not cause any collisions among
    the E8 roots.

The 3 clean kernel directions are invisible to the projection and can
freely carry error-correction checksums (the "phason channel").

This is the HEMOC analog of coding theory: the kernel IS the code space,
and we get 3 free redundancy dimensions from the algebraic structure.

References
----------
  Elser & Sloane, "A highly symmetric four-dimensional quasicrystal" (1987).
  Phillips, "The Totalistic Geometry of E8" (2026).
"""

import numpy as np
from typing import Tuple

from hemoc.core.phillips_matrix import PHILLIPS_MATRIX, PHILLIPS_U_L

# =============================================================================
# The collision direction (discovered, not designed)
# =============================================================================

COLLISION_DIRECTION = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]) / 2.0


def compute_kernel_basis() -> np.ndarray:
    """
    Compute an orthonormal basis for the 4D kernel of the Phillips matrix.

    Returns
    -------
    np.ndarray of shape (4, 8)
        Four orthonormal basis vectors spanning null(U).
    """
    _, s, Vt = np.linalg.svd(PHILLIPS_MATRIX, full_matrices=True)
    rank = np.sum(s > 1e-10)
    kernel = Vt[rank:]
    return kernel


def compute_clean_kernel_directions() -> np.ndarray:
    """
    Compute the 3 kernel directions that do NOT produce collisions.

    These are obtained by projecting out the collision direction from
    the full 4D kernel.  The result is a 3D subspace of R^8 that is
    invisible to the projection AND does not distinguish between
    collision partners.

    Returns
    -------
    np.ndarray of shape (k, 8), where k <= 3
        Orthonormal clean kernel directions.
    """
    kernel = compute_kernel_basis()
    if kernel.shape[0] == 0:
        return np.empty((0, 8))

    # Normalize the collision direction
    d = COLLISION_DIRECTION.copy()
    d /= np.linalg.norm(d)

    # Gram-Schmidt: remove collision component from each kernel vector
    clean = []
    for i in range(kernel.shape[0]):
        v = kernel[i].copy()
        # Remove collision direction component
        v -= np.dot(v, d) * d
        # Remove components of previously found clean directions
        for c in clean:
            v -= np.dot(v, c) * c
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            clean.append(v / norm)

    return np.array(clean) if clean else np.empty((0, 8))


def verify_collision_direction_in_kernel() -> dict:
    """
    Verify that the collision direction lies in the Phillips kernel.

    This is a key structural property: d is ALWAYS in null(U) for the
    Phillips sign pattern, regardless of entry values.  This is the basis
    for Conjecture 3 (Collision Universality).

    Returns
    -------
    dict with verification results.
    """
    d = COLLISION_DIRECTION
    image = PHILLIPS_MATRIX @ d

    return {
        "collision_direction": d.tolist(),
        "image_under_U": image.tolist(),
        "image_norm": float(np.linalg.norm(image)),
        "in_kernel": bool(np.allclose(image, 0.0, atol=1e-12)),
    }


def count_collisions(projection_matrix: np.ndarray = None) -> Tuple[int, list]:
    """
    Count collision pairs among the 240 E8 roots under a given projection.

    Two roots "collide" when they project to the same 4D point (within
    numerical tolerance).

    Parameters
    ----------
    projection_matrix : np.ndarray, optional
        (4, 8) projection matrix.  Defaults to PHILLIPS_U_L.

    Returns
    -------
    (n_collision_pairs, collision_pair_list)
    """
    if projection_matrix is None:
        projection_matrix = PHILLIPS_U_L

    from hemoc.core.e8_roots import generate_e8_roots
    roots = generate_e8_roots()
    coords = np.array([r.coordinates for r in roots])
    projected = coords @ projection_matrix.T   # (240, 4)

    from collections import defaultdict
    groups = defaultdict(list)
    for i, p in enumerate(projected):
        key = tuple(np.round(p, 8))
        groups[key].append(i)

    pairs = []
    for key, indices in groups.items():
        if len(indices) > 1:
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    pairs.append((indices[a], indices[b]))

    return len(pairs), pairs
