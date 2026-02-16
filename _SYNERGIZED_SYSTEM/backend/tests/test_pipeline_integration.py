"""
Integration tests for the full encode-compute-readout pipeline.

Tests the OpticalKirigamiMoire pipeline end-to-end:
- Encoder creation with various configs
- Single-frame encoding
- Sequence encoding
- Feature vector extraction
- Determinism with fixed seeds
"""

import pytest
import numpy as np

from engine.pipeline import (
    OpticalKirigamiMoire,
    PipelineConfig,
    ComputationMode,
    EncodingResult,
    create_encoder,
    encode_image,
)


class TestPipelineCreation:
    """Test encoder construction with various configurations."""

    def test_default_encoder_creates(self):
        encoder = create_encoder()
        assert encoder is not None
        assert isinstance(encoder, OpticalKirigamiMoire)

    def test_custom_grid_size(self):
        encoder = create_encoder(grid_size=(32, 32))
        assert encoder is not None

    def test_all_computation_modes(self):
        for mode in ComputationMode:
            encoder = create_encoder(mode=mode)
            assert encoder is not None

    def test_pipeline_config_defaults(self):
        config = PipelineConfig()
        assert config.grid_size == (64, 64)


class TestSingleFrameEncoding:
    """Test encoding a single input through the pipeline."""

    @pytest.fixture
    def encoder(self):
        return create_encoder(grid_size=(16, 16))

    @pytest.fixture
    def test_input(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((16, 16))

    def test_encode_returns_result(self, encoder, test_input):
        result = encoder.encode(test_input)
        assert isinstance(result, EncodingResult)

    def test_encode_produces_features(self, encoder, test_input):
        result = encoder.encode(test_input)
        assert result.features is not None
        assert len(result.features) > 0
        # All features should be finite
        assert all(np.isfinite(f) for f in result.features)

    def test_encode_result_has_moire_pattern(self, encoder, test_input):
        result = encoder.encode(test_input)
        assert result.moire_pattern is not None
        # Moire pattern should be 2D
        assert len(result.moire_pattern.shape) >= 2

    def test_different_inputs_produce_different_outputs(self, encoder):
        rng = np.random.default_rng(42)
        input_a = rng.standard_normal((16, 16))
        input_b = rng.standard_normal((16, 16)) * 5  # Different scale

        result_a = encoder.encode(input_a)
        result_b = encoder.encode(input_b)

        # Feature vectors should differ
        assert not np.allclose(
            result_a.features, result_b.features
        ), "Different inputs should produce different feature vectors"


class TestDeterminism:
    """Test that encoding is deterministic with fixed seeds."""

    def test_same_input_same_output(self):
        """Encoding the same input twice should produce identical results."""
        rng = np.random.default_rng(123)
        data = rng.standard_normal((16, 16))

        encoder1 = create_encoder(grid_size=(16, 16))
        result1 = encoder1.encode(data)

        encoder2 = create_encoder(grid_size=(16, 16))
        result2 = encoder2.encode(data)

        np.testing.assert_array_almost_equal(
            result1.features,
            result2.features,
            decimal=10,
            err_msg="Encoding should be deterministic"
        )


class TestQuickEncode:
    """Test the convenience encode_image function."""

    def test_quick_encode(self):
        rng = np.random.default_rng(42)
        image = rng.standard_normal((16, 16))
        result = encode_image(image)
        assert isinstance(result, EncodingResult)
        assert result.features is not None


class TestSequenceEncoding:
    """Test encoding sequences of inputs."""

    def test_sequence_encoding(self):
        encoder = create_encoder(grid_size=(16, 16))
        rng = np.random.default_rng(42)
        sequence = [rng.standard_normal((16, 16)) for _ in range(5)]

        results = encoder.encode_sequence(sequence)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, EncodingResult)
            assert r.features is not None
