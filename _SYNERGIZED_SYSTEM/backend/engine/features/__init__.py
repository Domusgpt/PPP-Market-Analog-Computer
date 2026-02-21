"""
Feature Extraction Module
=========================

Advanced feature extraction for moiré patterns.

Modules:
- gabor: Gabor filter bank for texture features
- hog: Histogram of Oriented Gradients
- wavelet: Multi-resolution wavelet decomposition
- moments: Shape moments (Hu, Zernike)
- spectral: Frequency domain features
- extractor: Unified feature extraction API
- lpq_decoder: Local Phase Quantization for moiré feedback
"""

from .gabor import GaborFilterBank, extract_gabor_features
from .hog import HOGDescriptor, extract_hog_features
from .wavelet import WaveletDecomposer, extract_wavelet_features
from .golden_wavelet import GoldenMRAAdapter
from .moments import compute_hu_moments, compute_zernike_moments
from .spectral import SpectralAnalyzer, extract_spectral_features
from .extractor import FeatureExtractor, FeatureConfig
from .lpq_decoder import (
    LPQDecoder,
    LPQConfig,
    LPQDescriptor,
    MoireLPQAnalyzer,
    PhasonStrainDetector,
    PhaseMode,
    QuantizationMethod,
)

__all__ = [
    "GaborFilterBank",
    "extract_gabor_features",
    "HOGDescriptor",
    "extract_hog_features",
    "WaveletDecomposer",
    "extract_wavelet_features",
    "GoldenMRAAdapter",
    "compute_hu_moments",
    "compute_zernike_moments",
    "SpectralAnalyzer",
    "extract_spectral_features",
    "FeatureExtractor",
    "FeatureConfig",
    "LPQDecoder",
    "LPQConfig",
    "LPQDescriptor",
    "MoireLPQAnalyzer",
    "PhasonStrainDetector",
    "PhaseMode",
    "QuantizationMethod",
]
