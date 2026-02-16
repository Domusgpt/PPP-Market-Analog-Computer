"""
Integration tests for the WebSocket telemetry server.

Tests SyntheticEngine and TelemetryFrame without requiring
real physics modules or WebSocket connections.
"""

import math
import pytest

from engine.websocket_server import SyntheticEngine, TelemetryFrame


class TestSyntheticEngine:
    """Test the SyntheticEngine telemetry generator."""

    @pytest.fixture
    def engine(self):
        return SyntheticEngine()

    def test_engine_creates(self, engine):
        assert engine is not None
        assert engine.frame_count == 0
        assert engine.paused is False

    def test_step_produces_frame(self, engine):
        frame = engine.step()
        assert isinstance(frame, TelemetryFrame)
        assert frame.frame_id == 1

    def test_frame_id_increments(self, engine):
        f1 = engine.step()
        f2 = engine.step()
        f3 = engine.step()
        assert f1.frame_id == 1
        assert f2.frame_id == 2
        assert f3.frame_id == 3

    def test_frame_has_all_subsystems(self, engine):
        frame = engine.step()
        assert "period" in frame.moire
        assert "contrast" in frame.moire
        assert "dominant_frequency" in frame.moire
        assert "petal_rotations" in frame.kirigami
        assert "lattice_stress" in frame.kirigami
        assert "cell_distribution" in frame.kirigami
        assert "entropy" in frame.reservoir
        assert "lyapunov" in frame.reservoir
        assert "memory_capacity" in frame.reservoir
        assert "gap_mode" in frame.talbot
        assert "gap_distance" in frame.talbot

    def test_contrast_in_valid_range(self, engine):
        for _ in range(100):
            frame = engine.step()
            assert 0 <= frame.moire["contrast"] <= 1, (
                f"Contrast {frame.moire['contrast']} out of [0,1] at frame {frame.frame_id}"
            )

    def test_cell_distribution_valid(self, engine):
        for _ in range(50):
            frame = engine.step()
            dist = frame.kirigami["cell_distribution"]
            assert len(dist) == 3
            for d in dist:
                assert 0 <= d <= 1, f"Cell dist value {d} out of [0,1]"

    def test_feature_vector_exists(self, engine):
        frame = engine.step()
        assert frame.feature_vector is not None
        assert len(frame.feature_vector) == 4
        for v in frame.feature_vector:
            assert math.isfinite(v), f"Feature value {v} is not finite"

    def test_petal_rotations_evolve(self, engine):
        f1 = engine.step()
        for _ in range(10):
            engine.step()
        f12 = engine.step()
        # Petals should have rotated
        assert f1.kirigami["petal_rotations"] != f12.kirigami["petal_rotations"]

    def test_actuators_present(self, engine):
        frame = engine.step()
        assert len(frame.actuators) == 2
        for a in frame.actuators:
            assert "tip" in a
            assert "tilt" in a
            assert "piston" in a


class TestEngineCommands:
    """Test engine command handling."""

    @pytest.fixture
    def engine(self):
        return SyntheticEngine()

    def test_set_mode_changes_angle(self, engine):
        engine.handle_command({"command": "set_mode", "angle": 45.0})
        assert engine.angle == 45.0

    def test_set_mode_changes_gap(self, engine):
        engine.handle_command({"command": "set_mode", "gap": 50.0})
        assert engine.gap_distance == 50.0

    def test_pause_and_resume(self, engine):
        engine.handle_command({"command": "pause"})
        assert engine.paused is True
        engine.handle_command({"command": "resume"})
        assert engine.paused is False

    def test_reset_restores_defaults(self, engine):
        engine.handle_command({"command": "set_mode", "angle": 99.0})
        engine.step()
        engine.step()
        engine.handle_command({"command": "reset"})
        assert engine.frame_count == 0
        assert engine.angle == 0.0


class TestTelemetryFrame:
    """Test TelemetryFrame data structure."""

    def test_frame_fields(self):
        frame = TelemetryFrame(
            frame_id=1,
            timestamp=1000.0,
            moire={"period": 10.0, "contrast": 0.5},
            kirigami={"petal_rotations": [0, 0, 0]},
            reservoir={"entropy": 4.0},
            talbot={"gap_mode": "integer"},
            actuators=[],
            feature_vector=[0.1, 0.2],
        )
        assert frame.frame_id == 1
        assert frame.timestamp == 1000.0
        assert frame.feature_vector == [0.1, 0.2]
