"""Integration tests for architecture_mode wiring in encoder and stream paths."""

import numpy as np

from engine.main import OpticalKirigamiEncoder, EncoderConfig
from engine.streaming.stream_encoder import StreamEncoder, StreamConfig


def test_optical_encoder_legacy_mode_schema():
    encoder = OpticalKirigamiEncoder(
        EncoderConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="legacy")
    )
    data = np.random.default_rng(42).random((16, 16))
    result = encoder.encode_data(data)

    assert result["architecture_mode"] == "legacy"
    assert "quasicrystal" not in result
    assert result["moire_intensity"].shape == (16, 16)


def test_optical_encoder_quasicrystal_mode_schema():
    encoder = OpticalKirigamiEncoder(
        EncoderConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="quasicrystal")
    )
    data = np.random.default_rng(7).random((16, 16))
    result = encoder.encode_data(data)

    assert result["architecture_mode"] == "quasicrystal"
    assert "quasicrystal" in result
    qc = result["quasicrystal"]
    assert qc["galois_valid"] in (True, False)
    assert qc["padovan_steps"] > 0
    assert qc["reservoir_spectral_radius"] > 0
    assert result["moire_intensity"].shape == (16, 16)


def test_stream_encoder_legacy_mode_process():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="legacy"))
    stream.start()
    frame = stream.process(np.random.default_rng(1).random((16, 16)))
    stream.stop()

    assert frame.pattern.shape == (16, 16)
    assert len(frame.features) == 11


def test_stream_encoder_quasicrystal_mode_process():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="quasicrystal"))
    stream.start()
    frame = stream.process(np.random.default_rng(2).random((16, 16)))
    stream.stop()

    assert frame.pattern.shape == (16, 16)
    assert len(frame.features) == 11

