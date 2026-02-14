"""
Gabor Filter Bank - Texture Feature Extraction
==============================================

Multi-scale, multi-orientation Gabor filters for
extracting texture features from moiré patterns.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GaborResponse:
    """Response from Gabor filter bank."""
    magnitudes: np.ndarray    # (n_filters, ny, nx)
    phases: np.ndarray        # (n_filters, ny, nx)
    energies: np.ndarray      # (n_filters,) - mean magnitude per filter
    features: np.ndarray      # Flattened feature vector


class GaborFilterBank:
    """
    Gabor filter bank for texture analysis.

    Creates a bank of Gabor filters at multiple scales (frequencies)
    and orientations for comprehensive texture feature extraction.

    Parameters
    ----------
    n_orientations : int
        Number of orientation bands (default: 8)
    n_scales : int
        Number of frequency scales (default: 4)
    frequency_range : Tuple[float, float]
        Min and max spatial frequencies
    sigma : float
        Gaussian envelope standard deviation

    Example
    -------
    >>> bank = GaborFilterBank(n_orientations=8, n_scales=4)
    >>> response = bank.apply(pattern)
    >>> features = response.features
    """

    def __init__(
        self,
        n_orientations: int = 8,
        n_scales: int = 4,
        frequency_range: Tuple[float, float] = (0.05, 0.4),
        sigma: float = 2.0
    ):
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.freq_range = frequency_range
        self.sigma = sigma

        # Generate filter parameters
        self.orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)
        self.frequencies = np.logspace(
            np.log10(frequency_range[0]),
            np.log10(frequency_range[1]),
            n_scales
        )

        # Pre-compute filters
        self._filters: Optional[np.ndarray] = None
        self._filter_size: Optional[Tuple[int, int]] = None

    def _create_gabor_kernel(
        self,
        theta: float,
        frequency: float,
        size: int
    ) -> np.ndarray:
        """Create a single Gabor kernel."""
        # Coordinate grid
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        X, Y = np.meshgrid(x, y)

        # Rotate coordinates
        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

        # Gaussian envelope
        sigma_x = self.sigma
        sigma_y = self.sigma / 0.5  # Elongated
        gaussian = np.exp(-0.5 * (X_rot**2/sigma_x**2 + Y_rot**2/sigma_y**2))

        # Sinusoidal carrier
        sinusoid = np.exp(2j * np.pi * frequency * X_rot)

        # Gabor = Gaussian × Sinusoid
        kernel = gaussian * sinusoid

        # Normalize
        kernel = kernel / np.sqrt(np.sum(np.abs(kernel)**2))

        return kernel

    def _ensure_filters(self, size: Tuple[int, int]):
        """Create filter bank if needed."""
        if self._filters is not None and self._filter_size == size:
            return

        kernel_size = min(size[0], size[1], 31)  # Max kernel size
        if kernel_size % 2 == 0:
            kernel_size -= 1

        n_filters = self.n_orientations * self.n_scales
        self._filters = np.zeros((n_filters, kernel_size, kernel_size), dtype=np.complex128)

        idx = 0
        for freq in self.frequencies:
            for theta in self.orientations:
                self._filters[idx] = self._create_gabor_kernel(theta, freq, kernel_size)
                idx += 1

        self._filter_size = size

    def apply(self, image: np.ndarray) -> GaborResponse:
        """
        Apply filter bank to image.

        Parameters
        ----------
        image : np.ndarray
            2D input image

        Returns
        -------
        GaborResponse
            Filter responses and extracted features
        """
        from scipy.signal import convolve2d

        self._ensure_filters(image.shape)

        n_filters = len(self._filters)
        magnitudes = np.zeros((n_filters, *image.shape))
        phases = np.zeros((n_filters, *image.shape))

        for i, kernel in enumerate(self._filters):
            # Convolve with Gabor filter
            response = convolve2d(image, kernel, mode='same', boundary='wrap')
            magnitudes[i] = np.abs(response)
            phases[i] = np.angle(response)

        # Energy per filter
        energies = np.mean(magnitudes, axis=(1, 2))

        # Build feature vector
        features = []

        # Mean and std of magnitude per filter
        for i in range(n_filters):
            features.extend([
                np.mean(magnitudes[i]),
                np.std(magnitudes[i])
            ])

        # Cross-scale correlations
        for s in range(self.n_scales - 1):
            for o in range(self.n_orientations):
                idx1 = s * self.n_orientations + o
                idx2 = (s + 1) * self.n_orientations + o
                corr = np.corrcoef(
                    magnitudes[idx1].flatten(),
                    magnitudes[idx2].flatten()
                )[0, 1]
                features.append(corr if not np.isnan(corr) else 0.0)

        return GaborResponse(
            magnitudes=magnitudes,
            phases=phases,
            energies=energies,
            features=np.array(features, dtype=np.float32)
        )

    @property
    def n_features(self) -> int:
        """Number of features extracted."""
        n_filters = self.n_orientations * self.n_scales
        # Mean + std per filter + cross-scale correlations
        return n_filters * 2 + (self.n_scales - 1) * self.n_orientations


def extract_gabor_features(
    image: np.ndarray,
    n_orientations: int = 8,
    n_scales: int = 4
) -> np.ndarray:
    """
    Quick function to extract Gabor features.

    Parameters
    ----------
    image : np.ndarray
        2D input image
    n_orientations : int
        Number of orientations
    n_scales : int
        Number of scales

    Returns
    -------
    np.ndarray
        Feature vector
    """
    bank = GaborFilterBank(n_orientations, n_scales)
    response = bank.apply(image)
    return response.features
