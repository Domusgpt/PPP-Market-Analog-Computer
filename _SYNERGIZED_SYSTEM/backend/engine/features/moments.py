"""
Shape Moments - Rotation/Scale Invariant Features
================================================

Hu moments and Zernike moments for shape description.
"""

import numpy as np
from typing import Tuple


def compute_raw_moments(image: np.ndarray, max_order: int = 3) -> np.ndarray:
    """
    Compute raw image moments.

    M_pq = sum_x sum_y x^p * y^q * I(x,y)
    """
    ny, nx = image.shape
    y, x = np.mgrid[0:ny, 0:nx]

    moments = np.zeros((max_order + 1, max_order + 1))

    for p in range(max_order + 1):
        for q in range(max_order + 1):
            if p + q <= max_order:
                moments[p, q] = np.sum((x ** p) * (y ** q) * image)

    return moments


def compute_central_moments(image: np.ndarray, max_order: int = 3) -> np.ndarray:
    """
    Compute central (translation-invariant) moments.

    mu_pq = sum_x sum_y (x - x_bar)^p * (y - y_bar)^q * I(x,y)
    """
    ny, nx = image.shape
    y, x = np.mgrid[0:ny, 0:nx]

    # Centroid
    m00 = np.sum(image)
    if m00 == 0:
        return np.zeros((max_order + 1, max_order + 1))

    x_bar = np.sum(x * image) / m00
    y_bar = np.sum(y * image) / m00

    # Central moments
    moments = np.zeros((max_order + 1, max_order + 1))

    for p in range(max_order + 1):
        for q in range(max_order + 1):
            if p + q <= max_order:
                moments[p, q] = np.sum(
                    ((x - x_bar) ** p) * ((y - y_bar) ** q) * image
                )

    return moments


def compute_normalized_moments(image: np.ndarray, max_order: int = 3) -> np.ndarray:
    """
    Compute normalized (scale-invariant) central moments.

    eta_pq = mu_pq / mu_00^((p+q)/2 + 1)
    """
    mu = compute_central_moments(image, max_order)

    mu00 = mu[0, 0]
    if mu00 == 0:
        return np.zeros((max_order + 1, max_order + 1))

    eta = np.zeros((max_order + 1, max_order + 1))

    for p in range(max_order + 1):
        for q in range(max_order + 1):
            if p + q >= 2:
                gamma = (p + q) / 2 + 1
                eta[p, q] = mu[p, q] / (mu00 ** gamma)

    return eta


def compute_hu_moments(image: np.ndarray) -> np.ndarray:
    """
    Compute 7 Hu moment invariants.

    These moments are invariant to translation, scale, and rotation.

    Parameters
    ----------
    image : np.ndarray
        2D input image

    Returns
    -------
    np.ndarray
        7 Hu moments
    """
    eta = compute_normalized_moments(image, 3)

    # Hu's 7 invariant moments
    hu = np.zeros(7)

    # First invariant
    hu[0] = eta[2, 0] + eta[0, 2]

    # Second invariant
    hu[1] = (eta[2, 0] - eta[0, 2])**2 + 4 * eta[1, 1]**2

    # Third invariant
    hu[2] = (eta[3, 0] - 3*eta[1, 2])**2 + (3*eta[2, 1] - eta[0, 3])**2

    # Fourth invariant
    hu[3] = (eta[3, 0] + eta[1, 2])**2 + (eta[2, 1] + eta[0, 3])**2

    # Fifth invariant
    hu[4] = ((eta[3, 0] - 3*eta[1, 2]) * (eta[3, 0] + eta[1, 2]) *
             ((eta[3, 0] + eta[1, 2])**2 - 3*(eta[2, 1] + eta[0, 3])**2) +
             (3*eta[2, 1] - eta[0, 3]) * (eta[2, 1] + eta[0, 3]) *
             (3*(eta[3, 0] + eta[1, 2])**2 - (eta[2, 1] + eta[0, 3])**2))

    # Sixth invariant
    hu[5] = ((eta[2, 0] - eta[0, 2]) *
             ((eta[3, 0] + eta[1, 2])**2 - (eta[2, 1] + eta[0, 3])**2) +
             4 * eta[1, 1] * (eta[3, 0] + eta[1, 2]) * (eta[2, 1] + eta[0, 3]))

    # Seventh invariant (skew)
    hu[6] = ((3*eta[2, 1] - eta[0, 3]) * (eta[3, 0] + eta[1, 2]) *
             ((eta[3, 0] + eta[1, 2])**2 - 3*(eta[2, 1] + eta[0, 3])**2) -
             (eta[3, 0] - 3*eta[1, 2]) * (eta[2, 1] + eta[0, 3]) *
             (3*(eta[3, 0] + eta[1, 2])**2 - (eta[2, 1] + eta[0, 3])**2))

    # Log transform for numerical stability
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return hu.astype(np.float32)


def compute_zernike_moments(
    image: np.ndarray,
    radius: int = None,
    max_order: int = 8
) -> np.ndarray:
    """
    Compute Zernike moment magnitudes.

    Zernike moments are orthogonal and rotation-invariant
    when using only the magnitude.

    Parameters
    ----------
    image : np.ndarray
        2D input image
    radius : int, optional
        Radius for unit circle mapping (default: half image size)
    max_order : int
        Maximum polynomial order

    Returns
    -------
    np.ndarray
        Zernike moment magnitudes
    """
    ny, nx = image.shape

    if radius is None:
        radius = min(ny, nx) // 2

    # Create coordinate grid
    y, x = np.mgrid[0:ny, 0:nx]
    y = (y - ny / 2) / radius
    x = (x - nx / 2) / radius

    # Polar coordinates
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Mask for unit circle
    mask = rho <= 1.0

    moments = []

    def zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """Compute Zernike radial polynomial."""
        R = np.zeros_like(rho)
        for k in range((n - abs(m)) // 2 + 1):
            num = ((-1) ** k) * np.math.factorial(n - k)
            den = (np.math.factorial(k) *
                   np.math.factorial((n + abs(m)) // 2 - k) *
                   np.math.factorial((n - abs(m)) // 2 - k))
            R += (num / den) * (rho ** (n - 2*k))
        return R

    # Compute moments for each order
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            if (n - abs(m)) % 2 == 0:
                # Zernike polynomial
                R = zernike_radial(n, m, rho)
                V = R * np.exp(1j * m * theta)

                # Moment
                Z = np.sum(image[mask] * np.conj(V[mask])) * (n + 1) / np.pi

                # Store magnitude (rotation invariant)
                moments.append(np.abs(Z))

    return np.array(moments, dtype=np.float32)


def compute_all_moments(image: np.ndarray) -> np.ndarray:
    """
    Compute all moment-based features.

    Returns
    -------
    np.ndarray
        Combined Hu + Zernike moments
    """
    hu = compute_hu_moments(image)
    zernike = compute_zernike_moments(image, max_order=6)

    return np.concatenate([hu, zernike])
