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
    assert qc["galois_ratio_valid"] in (True, False)
    assert qc["galois_product_valid"] in (True, False)
    assert abs(qc["galois_ratio"] - 1.618033988749895) < 1e-6
    assert qc["galois_expected_product"] >= 0.0
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
    assert frame.metadata is not None
    assert frame.metadata["architecture_mode"] == "legacy"
    assert "quasicrystal" not in frame.metadata


def test_stream_encoder_quasicrystal_mode_process():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="quasicrystal"))
    stream.start()
    frame = stream.process(np.random.default_rng(2).random((16, 16)))
    stream.stop()

    assert frame.pattern.shape == (16, 16)
    assert len(frame.features) == 11
    assert frame.metadata is not None
    assert frame.metadata["architecture_mode"] == "quasicrystal"
    assert "quasicrystal" in frame.metadata
    qc = frame.metadata["quasicrystal"]
    assert qc["galois_ratio_valid"] in (True, False)
    assert qc["galois_product_valid"] in (True, False)
    assert qc["padovan_steps"] > 0
    assert qc["reservoir_spectral_radius"] > 0



def test_optical_encoder_quasicrystal_mode_batch_galois_payload():
    encoder = OpticalKirigamiEncoder(
        EncoderConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="quasicrystal")
    )
    data = np.random.default_rng(9).random((16, 16))
    result = encoder.encode_data(data)
    qc = result["quasicrystal"]

    assert "galois_product" in qc
    assert "galois_product_deviation" in qc
    assert qc["galois_product"] >= 0.0
    assert qc["galois_product_deviation"] >= 0.0


def test_stream_encoder_quasicrystal_statistics_health():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="quasicrystal"))
    stream.start()
    _ = stream.process(np.random.default_rng(21).random((16, 16)))
    stats = stream.get_statistics()
    health = stream.get_quasicrystal_health()
    stream.stop()

    assert stats["schema_version"] == "1.0"
    assert stats["generated_at"] > 0
    assert stats["architecture_mode"] == "quasicrystal"
    assert "quasicrystal_health" in stats
    assert stats["quasicrystal_health"]["enabled"] is True
    assert stats["last_frame_metadata"]["available"] is True
    assert stats["last_frame_metadata"]["architecture_mode"] == "quasicrystal"
    assert health["enabled"] is True
    assert health["verifier_total_checks"] >= 1
    assert health["reservoir_spectral_radius"] > 0


def test_stream_encoder_legacy_statistics_health_disabled():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="legacy"))
    stream.start()
    _ = stream.process(np.random.default_rng(22).random((16, 16)))
    stats = stream.get_statistics()
    health = stream.get_quasicrystal_health()
    stream.stop()

    assert stats["schema_version"] == "1.0"
    assert stats["generated_at"] > 0
    assert stats["architecture_mode"] == "legacy"
    assert "quasicrystal_health" not in stats
    assert stats["last_frame_metadata"]["available"] is True
    assert stats["last_frame_metadata"]["architecture_mode"] == "legacy"
    assert health["enabled"] is False


def test_stream_encoder_last_frame_metadata_resets():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="quasicrystal"))

    before = stream.get_last_frame_metadata()
    assert before["available"] is False

    stream.start()
    _ = stream.process(np.random.default_rng(23).random((16, 16)))
    during = stream.get_last_frame_metadata()
    assert during["available"] is True

    stream.reset()
    after = stream.get_last_frame_metadata()
    stream.stop()

    assert after["available"] is False
    assert after["architecture_mode"] == "quasicrystal"


def test_stream_encoder_metadata_snapshot_is_immutable_copy():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="quasicrystal"))
    stream.start()
    frame = stream.process(np.random.default_rng(24).random((16, 16)))

    assert frame.metadata is not None
    frame.metadata["architecture_mode"] = "tampered"
    if "quasicrystal" in frame.metadata:
        frame.metadata["quasicrystal"]["galois_valid"] = False

    snapshot = stream.get_last_frame_metadata()
    stream.stop()

    assert snapshot["available"] is True
    assert snapshot["architecture_mode"] == "quasicrystal"


def test_stream_encoder_last_metadata_getter_returns_copy():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="legacy"))
    stream.start()
    _ = stream.process(np.random.default_rng(25).random((16, 16)))

    first = stream.get_last_frame_metadata()
    first["architecture_mode"] = "mutated"
    second = stream.get_last_frame_metadata()
    stream.stop()

    assert first["available"] is True
    assert second["available"] is True
    assert second["architecture_mode"] == "legacy"


def test_stream_observability_snapshot_quasicrystal():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="quasicrystal"))
    stream.start()
    _ = stream.process(np.random.default_rng(26).random((16, 16)))

    snapshot = stream.get_observability_snapshot()
    stream.stop()

    assert snapshot["snapshot_kind"] == "stream_observability"
    assert snapshot["schema_version"] == "1.0"
    assert snapshot["generated_at"] > 0
    assert snapshot["architecture_mode"] == "quasicrystal"
    assert snapshot["last_frame_metadata"]["available"] is True
    assert "quasicrystal_health" in snapshot
    assert snapshot["quasicrystal_health"]["enabled"] is True


def test_stream_observability_snapshot_is_copy():
    stream = StreamEncoder(StreamConfig(grid_size=(16, 16), cascade_steps=6, architecture_mode="legacy"))
    stream.start()
    _ = stream.process(np.random.default_rng(27).random((16, 16)))

    first = stream.get_observability_snapshot()
    first["architecture_mode"] = "mutated"
    first["last_frame_metadata"]["architecture_mode"] = "mutated"

    second = stream.get_observability_snapshot()
    stream.stop()

    assert second["snapshot_kind"] == "stream_observability"
    assert second["schema_version"] == "1.0"
    assert second["architecture_mode"] == "legacy"
    assert second["last_frame_metadata"]["architecture_mode"] == "legacy"
