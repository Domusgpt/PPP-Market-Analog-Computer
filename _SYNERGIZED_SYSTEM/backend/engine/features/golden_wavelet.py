"""Golden-ratio (φ-adic) wavelet adapter for unified feature extraction."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..geometry.quasicrystal_architecture import GoldenMRA


@dataclass
class GoldenWaveletResult:
    """Result container aligned with wavelet feature extraction contracts."""

    coefficients: Dict[str, np.ndarray]
    features: np.ndarray
    energies: Dict[str, float]


class GoldenMRAAdapter:
    """Adapter that exposes GoldenMRA through a WaveletDecomposer-like API."""

    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels
        self._mra = GoldenMRA(n_levels=n_levels)

    def decompose(self, image: np.ndarray) -> GoldenWaveletResult:
        """Decompose an image and produce a deterministic feature vector."""
        signal = image.astype(np.float64).flatten()
        coeffs = self._mra.decompose(signal)

        coefficients: Dict[str, np.ndarray] = {}
        energies: Dict[str, float] = {}
        features = []

        approximations = coeffs.get("approximation", [])
        details = coeffs.get("details", [])

        for level, approx in enumerate(approximations):
            coefficients[f"cA_{level}"] = approx
            approx_energy = float(np.sum(approx ** 2))
            energies[f"cA_{level}"] = approx_energy
            features.extend([
                np.mean(approx),
                np.std(approx),
                np.max(np.abs(approx)),
                approx_energy / max(len(approx), 1),
            ])

            if level < len(details):
                for channel, detail in enumerate(details[level]):
                    key = f"cG{channel}_{level}"
                    coefficients[key] = detail
                    detail_energy = float(np.sum(detail ** 2))
                    energies[key] = detail_energy
                    features.extend([
                        np.mean(detail),
                        np.std(detail),
                        np.max(np.abs(detail)),
                        detail_energy / max(len(detail), 1),
                    ])

        final_approx = approximations[-1] if approximations else signal
        features.extend([np.mean(final_approx), np.std(final_approx)])

        return GoldenWaveletResult(
            coefficients=coefficients,
            features=np.array(features, dtype=np.float32),
            energies=energies,
        )

    @property
    def n_features(self) -> int:
        """Upper bound on extracted features for requested levels."""
        return 4 * 4 * self.n_levels + 2

