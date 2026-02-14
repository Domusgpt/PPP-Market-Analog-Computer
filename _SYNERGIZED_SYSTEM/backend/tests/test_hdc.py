"""Tests for hyperdimensional computing."""

import pytest
import numpy as np
from engine.hdc.operations import (
    random_hypervector,
    bind,
    bundle,
    permute,
    similarity,
)
from engine.hdc.encoder import HDCEncoder, HDCConfig


class TestHDCOperations:
    @pytest.fixture
    def dim(self):
        return 1000

    def test_random_hypervector_shape(self, dim):
        hv = random_hypervector(dim)
        assert len(hv) == dim

    def test_random_hypervector_is_binary(self, dim):
        hv = random_hypervector(dim)
        unique = set(np.unique(hv))
        assert unique.issubset({0, 1}) or unique.issubset({-1, 1}) or unique.issubset({0.0, 1.0}) or unique.issubset({-1.0, 1.0})

    def test_random_hypervector_is_balanced(self, dim):
        """Should have roughly equal 0s and 1s (or -1s and 1s)."""
        hv = random_hypervector(dim)
        mean = np.mean(hv)
        assert abs(mean - 0.5) < 0.1 or abs(mean) < 0.1

    def test_bind_is_self_inverse(self, dim):
        """bind(a, bind(a, b)) ≈ b"""
        a = random_hypervector(dim)
        b = random_hypervector(dim)
        bound = bind(a, b)
        recovered = bind(a, bound)
        sim = similarity(recovered, b)
        assert sim > 0.8

    def test_bundle_preserves_components(self, dim):
        """bundle(a, b) should be similar to both a and b."""
        a = random_hypervector(dim)
        b = random_hypervector(dim)
        bundled = bundle([a, b])
        sim_a = similarity(bundled, a)
        sim_b = similarity(bundled, b)
        assert sim_a > 0.3
        assert sim_b > 0.3

    def test_permute_changes_vector(self, dim):
        a = random_hypervector(dim)
        permuted = permute(a)
        sim = similarity(a, permuted)
        # Permuted should be dissimilar from original
        assert sim < 0.3

    def test_orthogonal_random_vectors(self, dim):
        """Two random hypervectors should be approximately orthogonal."""
        a = random_hypervector(dim)
        b = random_hypervector(dim)
        sim = similarity(a, b)
        assert abs(sim) < 0.15


class TestHDCEncoder:
    @pytest.fixture
    def encoder(self):
        config = HDCConfig(dim=1000)
        return HDCEncoder(config=config)

    def test_creation(self, encoder):
        assert encoder is not None

    def test_encode_pattern(self, encoder):
        """encode_pattern returns HDCResult with .hypervector."""
        pattern = np.random.default_rng(42).random((8, 8))
        result = encoder.encode_pattern(pattern)
        assert hasattr(result, 'hypervector')
        assert len(result.hypervector) == 1000

    def test_similar_patterns_give_similar_encodings(self, encoder):
        p1 = np.random.default_rng(42).random((8, 8))
        p2 = p1 + np.random.default_rng(43).random((8, 8)) * 0.1
        r1 = encoder.encode_pattern(p1)
        r2 = encoder.encode_pattern(p2)
        sim = similarity(r1.hypervector, r2.hypervector)
        assert sim > 0.3

    def test_encode_sequence(self, encoder):
        patterns = [np.random.default_rng(i).random((8, 8)) for i in range(5)]
        result = encoder.encode_sequence(patterns)
        assert hasattr(result, 'hypervector')
        assert len(result.hypervector) == 1000
