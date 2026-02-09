"""Tests for feature extraction modules."""

import pytest
import numpy as np
from engine.features.gabor import GaborFilterBank
from engine.features.spectral import SpectralAnalyzer
from engine.features.hog import HOGDescriptor


@pytest.fixture
def test_pattern():
    """A 32x32 test pattern with structure."""
    x = np.linspace(0, 4 * np.pi, 32)
    y = np.linspace(0, 4 * np.pi, 32)
    X, Y = np.meshgrid(x, y)
    return np.sin(X) * np.cos(Y)


class TestGaborFilterBank:
    @pytest.fixture
    def gabor(self):
        return GaborFilterBank(n_orientations=4, n_scales=2)

    def test_creation(self, gabor):
        assert gabor is not None

    def test_apply_returns_gabor_response(self, gabor, test_pattern):
        """apply() returns GaborResponse with .features attribute."""
        response = gabor.apply(test_pattern)
        assert hasattr(response, 'features')
        assert hasattr(response, 'magnitudes')
        assert hasattr(response, 'phases')
        assert hasattr(response, 'energies')

    def test_features_are_finite(self, gabor, test_pattern):
        response = gabor.apply(test_pattern)
        assert np.all(np.isfinite(response.features))

    def test_different_patterns_give_different_features(self, gabor):
        p1 = np.sin(np.linspace(0, 4*np.pi, 32).reshape(1, -1).repeat(32, axis=0))
        p2 = np.random.default_rng(42).random((32, 32))
        f1 = gabor.apply(p1).features
        f2 = gabor.apply(p2).features
        assert not np.allclose(f1, f2, atol=0.01)


class TestSpectralAnalyzer:
    @pytest.fixture
    def spectral(self):
        return SpectralAnalyzer()

    def test_creation(self, spectral):
        assert spectral is not None

    def test_analyze_returns_spectral_result(self, spectral, test_pattern):
        """analyze() returns SpectralResult with .features attribute."""
        result = spectral.analyze(test_pattern)
        assert hasattr(result, 'features')
        assert hasattr(result, 'magnitude_spectrum')
        assert hasattr(result, 'phase_spectrum')
        assert hasattr(result, 'power_spectrum')

    def test_features_are_finite(self, spectral, test_pattern):
        result = spectral.analyze(test_pattern)
        assert np.all(np.isfinite(result.features))

    def test_dominant_frequencies(self, spectral, test_pattern):
        result = spectral.analyze(test_pattern)
        assert hasattr(result, 'dominant_frequencies')


class TestHOGDescriptor:
    @pytest.fixture
    def hog(self):
        return HOGDescriptor()

    def test_creation(self, hog):
        assert hog is not None

    def test_compute_returns_hog_result(self, hog, test_pattern):
        """compute() returns HOGResult with .features attribute."""
        result = hog.compute(test_pattern)
        assert hasattr(result, 'features')
        assert hasattr(result, 'cell_histograms')
        assert hasattr(result, 'gradient_magnitude')
        assert hasattr(result, 'gradient_orientation')

    def test_features_are_finite(self, hog, test_pattern):
        result = hog.compute(test_pattern)
        assert np.all(np.isfinite(result.features))
