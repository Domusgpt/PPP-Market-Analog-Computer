"""Tests for reservoir computing (ESN + criticality)."""

import pytest
import numpy as np
from engine.reservoir.esn import EchoStateReservoir


class TestEchoStateReservoir:
    @pytest.fixture
    def esn(self):
        return EchoStateReservoir(
            n_inputs=16,
            n_reservoir=64,
            spectral_radius=0.95,
            leak_rate=0.3,
        )

    def test_creation(self, esn):
        assert esn is not None

    def test_reservoir_size(self, esn):
        assert esn.n_reservoir == 64

    def test_spectral_radius_near_one(self, esn):
        """Spectral radius should be set close to target for edge-of-chaos."""
        if hasattr(esn, 'W'):
            eigenvalues = np.linalg.eigvals(esn.W)
            actual_sr = np.max(np.abs(eigenvalues))
            assert actual_sr <= 1.05  # Allow small numerical tolerance

    def test_run_produces_output(self, esn):
        """run() takes 2D array (T, n_inputs), returns 2D (T, n_reservoir)."""
        T = 10
        input_data = np.random.default_rng(42).random((T, 16))
        states = esn.run(input_data)
        assert states is not None
        assert states.ndim == 2
        assert states.shape[0] == T
        assert states.shape[1] == 64  # n_reservoir

    def test_run_single_step(self, esn):
        """Single timestep input."""
        input_data = np.random.default_rng(42).random((1, 16))
        states = esn.run(input_data)
        assert states.shape == (1, 64)

    def test_process_2d(self, esn):
        """process_2d takes a 2D pattern and returns ESNResult."""
        pattern = np.random.default_rng(42).random((8, 8))
        result = esn.process_2d(pattern)
        assert hasattr(result, 'states')
        assert hasattr(result, 'output')
        assert hasattr(result, 'final_state')


class TestCriticalityAnalyzer:
    def test_creation(self):
        from engine.reservoir.criticality import CriticalityAnalyzer
        analyzer = CriticalityAnalyzer()
        assert analyzer is not None

    def test_analyze(self):
        from engine.reservoir.criticality import CriticalityAnalyzer
        analyzer = CriticalityAnalyzer(grid_size=(8, 8))
        metrics = analyzer.analyze(coupling=0.5, damping=0.1, n_trials=3, n_steps=20)
        assert hasattr(metrics, 'lyapunov_estimate')
        assert hasattr(metrics, 'is_critical')
        assert hasattr(metrics, 'regime')
