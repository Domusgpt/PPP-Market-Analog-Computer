"""
Wavelet Decomposition - Multi-Resolution Features
=================================================

Wavelet-based multi-resolution analysis for moiré patterns.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class WaveletResult:
    """Result from wavelet decomposition."""
    coefficients: Dict[str, np.ndarray]  # 'cA', 'cH', 'cV', 'cD' per level
    features: np.ndarray                  # Feature vector
    energies: Dict[str, float]            # Energy per subband


class WaveletDecomposer:
    """
    2D Wavelet decomposition for feature extraction.

    Performs multi-level wavelet decomposition and extracts
    statistical features from each subband.

    Parameters
    ----------
    wavelet : str
        Wavelet type ('haar', 'db2', 'sym2')
    n_levels : int
        Number of decomposition levels

    Example
    -------
    >>> decomposer = WaveletDecomposer(wavelet='haar', n_levels=3)
    >>> result = decomposer.decompose(pattern)
    >>> features = result.features
    """

    def __init__(self, wavelet: str = 'haar', n_levels: int = 3):
        self.wavelet = wavelet
        self.n_levels = n_levels

        # Define wavelet filters (simple Haar implementation)
        self._setup_filters()

    def _setup_filters(self):
        """Setup wavelet filters."""
        if self.wavelet == 'haar':
            self.lo = np.array([1, 1]) / np.sqrt(2)
            self.hi = np.array([1, -1]) / np.sqrt(2)
        elif self.wavelet == 'db2':
            # Daubechies-2
            self.lo = np.array([0.4829, 0.8365, 0.2241, -0.1294])
            self.hi = np.array([-0.1294, -0.2241, 0.8365, -0.4829])
        else:
            # Default to Haar
            self.lo = np.array([1, 1]) / np.sqrt(2)
            self.hi = np.array([1, -1]) / np.sqrt(2)

    def _convolve_downsample(self, signal: np.ndarray, filt: np.ndarray) -> np.ndarray:
        """Convolve and downsample by 2."""
        # Extend signal for circular convolution
        padded = np.pad(signal, len(filt) // 2, mode='wrap')
        # Convolve
        result = np.convolve(padded, filt, mode='valid')
        # Trim to original length before downsampling to keep exact half resolution
        result = result[:len(signal)]
        return result[::2]

    def _decompose_1d(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single level 1D decomposition."""
        # Low-pass (approximation)
        cA = self._convolve_downsample(signal, self.lo)
        # High-pass (detail)
        cD = self._convolve_downsample(signal, self.hi)
        return cA, cD

    def _decompose_2d_level(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Single level 2D decomposition."""
        ny, nx = image.shape

        # Row-wise decomposition
        L = np.zeros((ny, nx // 2))
        H = np.zeros((ny, nx // 2))
        for i in range(ny):
            L[i], H[i] = self._decompose_1d(image[i])

        # Column-wise decomposition of L
        cA = np.zeros((ny // 2, nx // 2))
        cH = np.zeros((ny // 2, nx // 2))
        for j in range(nx // 2):
            cA[:, j], cH[:, j] = self._decompose_1d(L[:, j])

        # Column-wise decomposition of H
        cV = np.zeros((ny // 2, nx // 2))
        cD = np.zeros((ny // 2, nx // 2))
        for j in range(nx // 2):
            cV[:, j], cD[:, j] = self._decompose_1d(H[:, j])

        return {
            'cA': cA,  # Approximation
            'cH': cH,  # Horizontal detail
            'cV': cV,  # Vertical detail
            'cD': cD   # Diagonal detail
        }

    def decompose(self, image: np.ndarray) -> WaveletResult:
        """
        Perform multi-level wavelet decomposition.

        Parameters
        ----------
        image : np.ndarray
            2D input image (must be power of 2 size for best results)

        Returns
        -------
        WaveletResult
            Decomposition coefficients and features
        """
        coefficients = {}
        energies = {}
        features = []

        current = image.astype(np.float64)

        for level in range(self.n_levels):
            if current.shape[0] < 4 or current.shape[1] < 4:
                break

            # Decompose
            level_coeffs = self._decompose_2d_level(current)

            # Store coefficients
            for key, coeff in level_coeffs.items():
                coefficients[f'{key}_{level}'] = coeff

                # Compute energy
                energy = np.sum(coeff ** 2)
                energies[f'{key}_{level}'] = energy

                # Extract features
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.abs(coeff).max(),
                    energy / coeff.size  # Normalized energy
                ])

            # Use approximation for next level
            current = level_coeffs['cA']

        # Add final approximation features
        features.extend([
            np.mean(current),
            np.std(current)
        ])

        return WaveletResult(
            coefficients=coefficients,
            features=np.array(features, dtype=np.float32),
            energies=energies
        )

    @property
    def n_features(self) -> int:
        """Number of features per decomposition."""
        # 4 subbands × 4 features × n_levels + 2 final
        return 4 * 4 * self.n_levels + 2


def extract_wavelet_features(
    image: np.ndarray,
    wavelet: str = 'haar',
    n_levels: int = 3
) -> np.ndarray:
    """
    Quick function to extract wavelet features.

    Parameters
    ----------
    image : np.ndarray
        2D input image
    wavelet : str
        Wavelet type
    n_levels : int
        Decomposition levels

    Returns
    -------
    np.ndarray
        Feature vector
    """
    decomposer = WaveletDecomposer(wavelet, n_levels)
    result = decomposer.decompose(image)
    return result.features
