"""
Unified Feature Extractor
=========================

Combines all feature extraction methods into a single API.
"""

import numpy as np
from typing import Dict, List, Literal, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto

from .gabor import GaborFilterBank
from .hog import HOGDescriptor
from .wavelet import WaveletDecomposer
from .golden_wavelet import GoldenMRAAdapter
from .moments import compute_hu_moments, compute_zernike_moments
from .spectral import SpectralAnalyzer


class FeatureType(Enum):
    """Available feature types."""
    STATISTICS = auto()   # Basic statistics
    GABOR = auto()        # Gabor texture features
    HOG = auto()          # Histogram of Oriented Gradients
    WAVELET = auto()      # Wavelet decomposition
    MOMENTS = auto()      # Hu and Zernike moments
    SPECTRAL = auto()     # Frequency domain features
    ALL = auto()          # All features combined


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Which features to extract
    features: Set[FeatureType] = field(default_factory=lambda: {FeatureType.ALL})

    # Gabor parameters
    gabor_orientations: int = 8
    gabor_scales: int = 4

    # HOG parameters
    hog_cell_size: tuple = (8, 8)
    hog_n_bins: int = 9

    # Wavelet parameters
    wavelet_family: Literal['dyadic', 'golden'] = 'dyadic'
    wavelet_type: str = 'haar'
    wavelet_levels: int = 3

    # Moments parameters
    zernike_order: int = 6

    # Spectral parameters
    spectral_radial_bins: int = 16
    spectral_angular_bins: int = 8

    # Normalization
    normalize: bool = True


@dataclass
class ExtractedFeatures:
    """Container for extracted features."""
    combined: np.ndarray              # All features concatenated
    by_type: Dict[str, np.ndarray]    # Features grouped by type
    feature_names: List[str]          # Names for each feature
    n_features: int                   # Total feature count


class FeatureExtractor:
    """
    Unified feature extraction API.

    Combines multiple feature extraction methods and provides
    a single interface for extracting comprehensive feature vectors.

    Parameters
    ----------
    config : FeatureConfig
        Configuration for feature extraction

    Example
    -------
    >>> config = FeatureConfig(features={FeatureType.GABOR, FeatureType.HOG})
    >>> extractor = FeatureExtractor(config)
    >>> features = extractor.extract(pattern)
    >>> print(f"Extracted {features.n_features} features")
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

        # Initialize extractors lazily
        self._gabor: Optional[GaborFilterBank] = None
        self._hog: Optional[HOGDescriptor] = None
        self._wavelet: Optional[Union[WaveletDecomposer, GoldenMRAAdapter]] = None
        self._spectral: Optional[SpectralAnalyzer] = None

    def _get_gabor(self) -> GaborFilterBank:
        if self._gabor is None:
            self._gabor = GaborFilterBank(
                n_orientations=self.config.gabor_orientations,
                n_scales=self.config.gabor_scales
            )
        return self._gabor

    def _get_hog(self) -> HOGDescriptor:
        if self._hog is None:
            self._hog = HOGDescriptor(
                cell_size=self.config.hog_cell_size,
                n_bins=self.config.hog_n_bins
            )
        return self._hog

    def _get_wavelet(self) -> Union[WaveletDecomposer, GoldenMRAAdapter]:
        if self._wavelet is None:
            if self.config.wavelet_family == "golden":
                self._wavelet = GoldenMRAAdapter(
                    n_levels=self.config.wavelet_levels
                )
            else:
                self._wavelet = WaveletDecomposer(
                    wavelet=self.config.wavelet_type,
                    n_levels=self.config.wavelet_levels
                )
        return self._wavelet

    def _get_spectral(self) -> SpectralAnalyzer:
        if self._spectral is None:
            self._spectral = SpectralAnalyzer(
                n_radial_bins=self.config.spectral_radial_bins,
                n_angular_bins=self.config.spectral_angular_bins
            )
        return self._spectral

    def _should_extract(self, feature_type: FeatureType) -> bool:
        """Check if a feature type should be extracted."""
        return (FeatureType.ALL in self.config.features or
                feature_type in self.config.features)

    def extract(self, image: np.ndarray) -> ExtractedFeatures:
        """
        Extract features from image.

        Parameters
        ----------
        image : np.ndarray
            2D input image

        Returns
        -------
        ExtractedFeatures
            Extracted features with metadata
        """
        features_by_type = {}
        feature_names = []

        # Basic statistics
        if self._should_extract(FeatureType.STATISTICS):
            stats = self._extract_statistics(image)
            features_by_type['statistics'] = stats
            feature_names.extend([
                'mean', 'std', 'min', 'max', 'median',
                'skewness', 'kurtosis', 'energy', 'entropy'
            ])

        # Gabor features
        if self._should_extract(FeatureType.GABOR):
            gabor = self._get_gabor()
            result = gabor.apply(image)
            features_by_type['gabor'] = result.features
            n_gabor = len(result.features)
            feature_names.extend([f'gabor_{i}' for i in range(n_gabor)])

        # HOG features
        if self._should_extract(FeatureType.HOG):
            hog = self._get_hog()
            result = hog.compute(image)
            features_by_type['hog'] = result.features
            n_hog = len(result.features)
            feature_names.extend([f'hog_{i}' for i in range(n_hog)])

        # Wavelet features
        if self._should_extract(FeatureType.WAVELET):
            wavelet = self._get_wavelet()
            result = wavelet.decompose(image)
            features_by_type['wavelet'] = result.features
            n_wavelet = len(result.features)
            feature_names.extend([f'wavelet_{self.config.wavelet_family}_{i}' for i in range(n_wavelet)])

        # Moment features
        if self._should_extract(FeatureType.MOMENTS):
            hu = compute_hu_moments(image)
            zernike = compute_zernike_moments(image, max_order=self.config.zernike_order)
            moments = np.concatenate([hu, zernike])
            features_by_type['moments'] = moments
            feature_names.extend([f'hu_{i}' for i in range(len(hu))])
            feature_names.extend([f'zernike_{i}' for i in range(len(zernike))])

        # Spectral features
        if self._should_extract(FeatureType.SPECTRAL):
            spectral = self._get_spectral()
            result = spectral.analyze(image)
            features_by_type['spectral'] = result.features
            n_spectral = len(result.features)
            feature_names.extend([f'spectral_{i}' for i in range(n_spectral)])

        # Combine all features
        all_features = []
        for key in ['statistics', 'gabor', 'hog', 'wavelet', 'moments', 'spectral']:
            if key in features_by_type:
                all_features.append(features_by_type[key])

        combined = np.concatenate(all_features) if all_features else np.array([])

        # Normalize if requested
        if self.config.normalize and len(combined) > 0:
            combined = self._normalize(combined)

        return ExtractedFeatures(
            combined=combined,
            by_type=features_by_type,
            feature_names=feature_names,
            n_features=len(combined)
        )

    def _extract_statistics(self, image: np.ndarray) -> np.ndarray:
        """Extract basic statistical features."""
        flat = image.flatten()

        mean = np.mean(flat)
        std = np.std(flat)
        min_val = np.min(flat)
        max_val = np.max(flat)
        median = np.median(flat)

        # Higher-order statistics
        if std > 0:
            skewness = np.mean(((flat - mean) / std) ** 3)
            kurtosis = np.mean(((flat - mean) / std) ** 4) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Energy
        energy = np.sum(flat ** 2)

        # Entropy
        hist, _ = np.histogram(flat, bins=256, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return np.array([
            mean, std, min_val, max_val, median,
            skewness, kurtosis, energy, entropy
        ], dtype=np.float32)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """L2 normalize feature vector."""
        norm = np.linalg.norm(features)
        if norm > 0:
            return features / norm
        return features

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from batch of images.

        Parameters
        ----------
        images : List[np.ndarray]
            List of 2D images

        Returns
        -------
        np.ndarray
            Feature matrix (n_images, n_features)
        """
        features = [self.extract(img).combined for img in images]
        return np.stack(features, axis=0)

    @property
    def n_features(self) -> int:
        """Estimate total number of features."""
        n = 0
        if self._should_extract(FeatureType.STATISTICS):
            n += 9
        if self._should_extract(FeatureType.GABOR):
            n += self._get_gabor().n_features
        if self._should_extract(FeatureType.HOG):
            # Depends on image size, estimate for 64x64
            n += 324  # Approximate
        if self._should_extract(FeatureType.WAVELET):
            n += self._get_wavelet().n_features
        if self._should_extract(FeatureType.MOMENTS):
            n += 7 + 25  # Hu + Zernike
        if self._should_extract(FeatureType.SPECTRAL):
            n += self._get_spectral().n_features
        return n
