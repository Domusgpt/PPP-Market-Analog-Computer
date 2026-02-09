"""
Spectral Analysis - Frequency Domain Features
=============================================

FFT-based frequency analysis for moiré patterns.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SpectralResult:
    """Result from spectral analysis."""
    magnitude_spectrum: np.ndarray
    phase_spectrum: np.ndarray
    power_spectrum: np.ndarray
    features: np.ndarray
    dominant_frequencies: List[Tuple[float, float, float]]  # (fx, fy, power)


class SpectralAnalyzer:
    """
    Frequency domain analyzer for moiré patterns.

    Extracts features from the 2D Fourier transform including
    radial power profile, dominant frequencies, and orientation.

    Parameters
    ----------
    n_radial_bins : int
        Number of radial frequency bins
    n_angular_bins : int
        Number of angular bins for orientation analysis

    Example
    -------
    >>> analyzer = SpectralAnalyzer(n_radial_bins=16)
    >>> result = analyzer.analyze(pattern)
    >>> features = result.features
    """

    def __init__(
        self,
        n_radial_bins: int = 16,
        n_angular_bins: int = 8
    ):
        self.n_radial_bins = n_radial_bins
        self.n_angular_bins = n_angular_bins

    def analyze(self, image: np.ndarray) -> SpectralResult:
        """
        Perform spectral analysis.

        Parameters
        ----------
        image : np.ndarray
            2D input image

        Returns
        -------
        SpectralResult
            Spectral features and data
        """
        # Compute 2D FFT
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)

        # Magnitude and phase
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        power = magnitude ** 2

        # Create frequency coordinate grids
        ny, nx = image.shape
        cy, cx = ny // 2, nx // 2

        y = np.arange(ny) - cy
        x = np.arange(nx) - cx
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Polar coordinates
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        # Extract features
        features = []

        # 1. Radial power profile
        max_r = min(cy, cx)
        radial_bins = np.linspace(0, max_r, self.n_radial_bins + 1)

        for i in range(self.n_radial_bins):
            mask = (R >= radial_bins[i]) & (R < radial_bins[i + 1])
            if np.any(mask):
                features.append(np.mean(power[mask]))
            else:
                features.append(0.0)

        # 2. Angular power distribution
        angular_bins = np.linspace(-np.pi, np.pi, self.n_angular_bins + 1)
        mid_r = max_r // 2

        for i in range(self.n_angular_bins):
            mask = ((Theta >= angular_bins[i]) &
                    (Theta < angular_bins[i + 1]) &
                    (R > mid_r * 0.5) & (R < mid_r * 1.5))
            if np.any(mask):
                features.append(np.mean(power[mask]))
            else:
                features.append(0.0)

        # 3. Global statistics
        features.extend([
            np.mean(power),
            np.std(power),
            np.max(power),
            np.sum(power)
        ])

        # 4. Spectral centroid
        total_power = np.sum(power)
        if total_power > 0:
            centroid_x = np.sum(X * power) / total_power
            centroid_y = np.sum(Y * power) / total_power
            centroid_r = np.sqrt(centroid_x**2 + centroid_y**2)
        else:
            centroid_x = centroid_y = centroid_r = 0

        features.extend([centroid_x, centroid_y, centroid_r])

        # 5. Spectral spread
        if total_power > 0:
            spread = np.sqrt(np.sum(((R - centroid_r)**2) * power) / total_power)
        else:
            spread = 0
        features.append(spread)

        # 6. Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(power + 1e-10)))
        arithmetic_mean = np.mean(power) + 1e-10
        flatness = geometric_mean / arithmetic_mean
        features.append(flatness)

        # Find dominant frequencies
        dominant = self._find_dominant_frequencies(power, X, Y)

        return SpectralResult(
            magnitude_spectrum=magnitude,
            phase_spectrum=phase,
            power_spectrum=power,
            features=np.array(features, dtype=np.float32),
            dominant_frequencies=dominant
        )

    def _find_dominant_frequencies(
        self,
        power: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        n_peaks: int = 5
    ) -> List[Tuple[float, float, float]]:
        """Find dominant frequency components."""
        # Exclude DC component
        ny, nx = power.shape
        cy, cx = ny // 2, nx // 2
        power_no_dc = power.copy()
        power_no_dc[cy-2:cy+3, cx-2:cx+3] = 0

        # Find peaks
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(power_no_dc, size=5)
        peaks = (power_no_dc == local_max) & (power_no_dc > np.mean(power_no_dc))

        peak_indices = np.argwhere(peaks)
        peak_powers = power_no_dc[peaks]

        # Sort by power
        sorted_idx = np.argsort(peak_powers)[::-1]

        dominant = []
        for i in range(min(n_peaks, len(sorted_idx))):
            idx = sorted_idx[i]
            py, px = peak_indices[idx]
            dominant.append((
                float(X[py, px]),
                float(Y[py, px]),
                float(peak_powers[idx])
            ))

        return dominant

    @property
    def n_features(self) -> int:
        """Number of spectral features."""
        # radial + angular + global(4) + centroid(3) + spread(1) + flatness(1)
        return self.n_radial_bins + self.n_angular_bins + 9


def extract_spectral_features(
    image: np.ndarray,
    n_radial_bins: int = 16,
    n_angular_bins: int = 8
) -> np.ndarray:
    """
    Quick function to extract spectral features.

    Parameters
    ----------
    image : np.ndarray
        2D input image
    n_radial_bins : int
        Number of radial bins
    n_angular_bins : int
        Number of angular bins

    Returns
    -------
    np.ndarray
        Feature vector
    """
    analyzer = SpectralAnalyzer(n_radial_bins, n_angular_bins)
    result = analyzer.analyze(image)
    return result.features
