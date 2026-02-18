"""
H4_L + H4_R Decomposition via the Phillips Matrix
====================================================

The Phillips 8x8 matrix decomposes E8 root vectors into two coupled
H4 copies:

    x  in  R^8   -->   (h_L, h_R)  in  R^4 x R^4

where  h_L = U_L @ x   (contracted block, entries {+-a, +-b})
and    h_R = U_R @ x   (expanded block,  entries {+-a, +-c}).

The two copies are coupled by the golden ratio:
    ||h_R|| = phi * ||h_L||    for any x.

This module provides projection functions and batch processing.
"""

from typing import List, Tuple
from dataclasses import dataclass

import numpy as np

from hemoc.core.phillips_matrix import PHILLIPS_MATRIX, PHILLIPS_U_L, PHILLIPS_U_R
from hemoc.core.e8_roots import E8Root, generate_e8_roots


@dataclass
class PhillipsProjection:
    """
    Result of projecting a single E8 root via the Phillips matrix.

    Attributes
    ----------
    h4_left : np.ndarray
        4D coordinates from the contracted (U_L) block.
    h4_right : np.ndarray
        4D coordinates from the expanded (U_R) block.
    full_8d : np.ndarray
        Full 8D output vector (U @ root).
    source_coordinates : np.ndarray
        Original 8D root coordinates.
    left_radius : float
        ||h4_left||.
    right_radius : float
        ||h4_right||.
    """
    h4_left: np.ndarray
    h4_right: np.ndarray
    full_8d: np.ndarray
    source_coordinates: np.ndarray
    left_radius: float
    right_radius: float


@dataclass
class PhillipsBatchProjection:
    """
    Batch projection of all 240 E8 roots.

    Attributes
    ----------
    left_vertices : np.ndarray
        (240, 4) array of H4_L projections.
    right_vertices : np.ndarray
        (240, 4) array of H4_R projections.
    left_radii : np.ndarray
        Unique shell radii from the left block.
    right_radii : np.ndarray
        Unique shell radii from the right block.
    """
    left_vertices: np.ndarray
    right_vertices: np.ndarray
    left_radii: np.ndarray
    right_radii: np.ndarray


def project_to_h4_left(v8: np.ndarray) -> np.ndarray:
    """Project an 8D vector to H4_L (contracted block)."""
    return PHILLIPS_U_L @ np.asarray(v8, dtype=np.float64)


def project_to_h4_right(v8: np.ndarray) -> np.ndarray:
    """Project an 8D vector to H4_R (expanded block)."""
    return PHILLIPS_U_R @ np.asarray(v8, dtype=np.float64)


def project_to_h4_full(v8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project an 8D vector through both blocks simultaneously.

    Returns
    -------
    (h4_left, h4_right) : Tuple of length-4 numpy arrays.
    """
    full = PHILLIPS_MATRIX @ np.asarray(v8, dtype=np.float64)
    return full[:4], full[4:]


def project_single(root: E8Root) -> PhillipsProjection:
    """Project a single E8 root and return full metadata."""
    full = PHILLIPS_MATRIX @ root.coordinates
    h4_l = full[:4]
    h4_r = full[4:]
    return PhillipsProjection(
        h4_left=h4_l,
        h4_right=h4_r,
        full_8d=full,
        source_coordinates=root.coordinates,
        left_radius=float(np.linalg.norm(h4_l)),
        right_radius=float(np.linalg.norm(h4_r)),
    )


def project_e8_phillips() -> PhillipsBatchProjection:
    """
    Project all 240 E8 roots through the Phillips matrix.

    Returns
    -------
    PhillipsBatchProjection
        Contains (240,4) left and right vertex arrays and unique shell radii.
    """
    roots = generate_e8_roots()
    coords = np.array([r.coordinates for r in roots])

    left = coords @ PHILLIPS_U_L.T     # (240, 4)
    right = coords @ PHILLIPS_U_R.T    # (240, 4)

    left_r = np.linalg.norm(left, axis=1)
    right_r = np.linalg.norm(right, axis=1)

    return PhillipsBatchProjection(
        left_vertices=left,
        right_vertices=right,
        left_radii=np.unique(np.round(left_r, 8)),
        right_radii=np.unique(np.round(right_r, 8)),
    )


def round_trip(v8: np.ndarray) -> np.ndarray:
    """
    Compute the round-trip projection:  U^T @ U @ v.

    Since the Phillips matrix has rank 4 in 8D, this projects v onto
    the row space of U and annihilates the kernel component.
    """
    return PHILLIPS_MATRIX.T @ (PHILLIPS_MATRIX @ np.asarray(v8, dtype=np.float64))
