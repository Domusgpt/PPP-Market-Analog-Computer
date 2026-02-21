"""
Stream Encoder - Real-Time Encoding Pipeline
============================================

Core streaming encoder for real-time data processing.
"""

import numpy as np
from typing import Optional, Callable, Dict, Generator, Literal, Any
from dataclasses import dataclass
from threading import Thread, Event
from queue import Queue
import time
from copy import deepcopy

from ..core.fast_cascade import FastCascadeSimulator
from ..core.fast_moire import FastMoireComputer
from ..geometry.quasicrystal_architecture import (
    NumberFieldHierarchy,
    QuasicrystallineReservoir,
    PadovanCascade,
    GaloisVerifier,
)
from .buffer import RingBuffer, FrameBuffer, FeatureBuffer


@dataclass
class StreamConfig:
    """Configuration for streaming encoder."""
    grid_size: tuple = (64, 64)
    cascade_steps: int = 20
    buffer_size: int = 100
    sample_rate: int = 22050  # For audio
    frame_rate: float = 30.0  # For video
    feature_rate: float = 10.0  # Features per second
    architecture_mode: Literal["legacy", "quasicrystal"] = "legacy"


@dataclass
class StreamFrame:
    """Single frame from stream encoder."""
    timestamp: float
    input_data: np.ndarray
    pattern: np.ndarray
    features: np.ndarray
    frame_index: int
    metadata: Optional[Dict[str, Any]] = None


class StreamEncoder:
    """
    Real-time streaming encoder.

    Processes continuous data streams with minimal latency,
    maintaining reservoir state across frames.

    Parameters
    ----------
    config : StreamConfig
        Stream configuration

    Example
    -------
    >>> encoder = StreamEncoder()
    >>> encoder.start()
    >>> for data in data_stream:
    ...     frame = encoder.process(data)
    ...     display(frame.pattern)
    >>> encoder.stop()
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.architecture_mode = self.config.architecture_mode

        # Core components
        self.simulator = FastCascadeSimulator(
            self.config.grid_size,
            coupling_strength=0.3
        )
        self.moire = FastMoireComputer()

        self._qc_hierarchy: Optional[NumberFieldHierarchy] = None
        self._qc_reservoir: Optional[QuasicrystallineReservoir] = None
        self._qc_cascade: Optional[PadovanCascade] = None
        self._qc_verifier: Optional[GaloisVerifier] = None
        if self.architecture_mode == "quasicrystal":
            qc_grid = min(self.config.grid_size)
            self._qc_hierarchy = NumberFieldHierarchy(base_size=8)
            self._qc_reservoir = QuasicrystallineReservoir(n_reservoir=64, input_dim=8)
            self._qc_cascade = PadovanCascade(max_steps=max(20, self.config.cascade_steps * 4), grid_size=qc_grid, coupling=0.3)
            self._qc_verifier = GaloisVerifier(tolerance=1e-6)

        # Buffers
        self.input_buffer = RingBuffer(self.config.buffer_size * 1024)
        self.pattern_buffer = FrameBuffer(
            self.config.buffer_size,
            self.config.grid_size
        )
        self.feature_buffer = FeatureBuffer(
            self.config.buffer_size,
            n_features=11
        )

        # State
        self.frame_index = 0
        self.start_time = 0.0
        self._running = False
        self._angle_idx = 2  # Default angle
        self._last_frame_metadata: Optional[Dict[str, Any]] = None

        # Callbacks
        self._on_frame: Optional[Callable[[StreamFrame], None]] = None

    def _to_qc_vector(self, data: np.ndarray) -> np.ndarray:
        splits = np.array_split(data.flatten(), 8)
        return np.array([float(np.mean(s)) if len(s) else 0.0 for s in splits], dtype=np.float64)

    def process(self, data: np.ndarray) -> StreamFrame:
        """
        Process single input and return encoded frame.

        Parameters
        ----------
        data : np.ndarray
            Input data (1D or 2D)

        Returns
        -------
        StreamFrame
            Encoded frame with pattern and features
        """
        timestamp = time.time() - self.start_time

        # Reshape if needed
        if data.ndim == 1:
            side = int(np.ceil(np.sqrt(len(data))))
            padded = np.zeros(side * side)
            padded[:len(data)] = data
            data = padded.reshape(side, side)

        # Resize to grid
        if data.shape != self.config.grid_size:
            from scipy.ndimage import zoom
            factors = (self.config.grid_size[0] / data.shape[0],
                      self.config.grid_size[1] / data.shape[1])
            data = zoom(data, factors, order=1)

        # Normalize
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)

        frame_metadata: Dict[str, Any] = {"architecture_mode": self.architecture_mode}
        if self.architecture_mode == "quasicrystal" and all(
            x is not None for x in (self._qc_hierarchy, self._qc_reservoir, self._qc_cascade)
        ):
            qvec = self._to_qc_vector(data)
            hierarchy_states = self._qc_hierarchy.step(qvec)  # type: ignore[union-attr]
            reservoir_state = self._qc_reservoir.step(hierarchy_states[1])  # type: ignore[union-attr]

            side = self._qc_cascade.grid_size  # type: ignore[union-attr]
            cascade_in = np.resize(reservoir_state, side * side).reshape(side, side)
            self._qc_cascade.inject(cascade_in)  # type: ignore[union-attr]
            qc_result = self._qc_cascade.run(n_epochs=1)  # type: ignore[union-attr]
            state = qc_result['final_state']

            if state.shape != self.config.grid_size:
                from scipy.ndimage import zoom
                factors = (self.config.grid_size[0] / state.shape[0], self.config.grid_size[1] / state.shape[1])
                state = zoom(state, factors, order=1)

            if self._qc_verifier is not None:
                galois = self._qc_verifier.verify(qvec)
                frame_metadata["quasicrystal"] = {
                    "galois_valid": bool(galois["valid"]),
                    "galois_ratio": float(galois["ratio"]),
                    "galois_ratio_valid": bool(galois["ratio_valid"]),
                    "galois_ratio_deviation": float(galois["ratio_deviation"]),
                    "galois_product": float(galois["product"]),
                    "galois_expected_product": float(galois["expected_product"]),
                    "galois_product_valid": bool(galois["product_valid"]),
                    "galois_product_deviation": float(galois["product_deviation"]),
                    "padovan_steps": int(qc_result["n_padovan_steps"]),
                    "reservoir_spectral_radius": float(self._qc_reservoir.spectral_radius),
                }
        else:
            # Incremental cascade (maintain state)
            # Apply decay for fading memory
            self.simulator.values *= 0.95
            self.simulator.inject_input(data, scale=0.5)

            # Run short cascade
            result = self.simulator.run(n_steps=self.config.cascade_steps)
            state = result.final_state

        # Compute moiré pattern
        angle = self.moire.COMMENSURATE_ANGLES[self._angle_idx]
        moire_result = self.moire.compute(
            twist_angle=angle,
            grid_size=self.config.grid_size,
            layer1_state=state
        )

        # Extract features
        features = self._extract_features(moire_result.intensity, state)

        # Create frame
        frame = StreamFrame(
            timestamp=timestamp,
            input_data=data,
            pattern=moire_result.intensity,
            features=features,
            frame_index=self.frame_index,
            metadata=deepcopy(frame_metadata),
        )

        self._last_frame_metadata = deepcopy(frame_metadata)

        # Update buffers
        self.pattern_buffer.append(moire_result.intensity)
        self.feature_buffer.append(features, timestamp)
        self.frame_index += 1

        # Callback
        if self._on_frame:
            self._on_frame(frame)

        return frame

    def _extract_features(self, pattern: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Extract basic features."""
        features = [
            np.mean(pattern),
            np.std(pattern),
            np.min(pattern),
            np.max(pattern),
            (np.max(pattern) - np.min(pattern)) / (np.max(pattern) + np.min(pattern) + 1e-8),
        ]

        # FFT features
        fft = np.abs(np.fft.fftshift(np.fft.fft2(pattern)))
        features.extend([
            np.mean(fft),
            np.max(fft)
        ])

        # State features
        features.extend([
            np.mean(state),
            np.std(state),
            np.sum(state > 0.5) / state.size,
            np.sum(state > 0.75) / state.size
        ])

        return np.array(features, dtype=np.float32)

    def process_stream(
        self,
        data_generator: Generator[np.ndarray, None, None]
    ) -> Generator[StreamFrame, None, None]:
        """
        Process stream of data.

        Parameters
        ----------
        data_generator : Generator
            Yields input arrays

        Yields
        ------
        StreamFrame
            Encoded frames
        """
        self.start()

        for data in data_generator:
            yield self.process(data)

        self.stop()

    def start(self):
        """Start stream processing."""
        self._running = True
        self.start_time = time.time()
        self.frame_index = 0
        self._last_frame_metadata = None
        self.simulator.reset()
        if self._qc_hierarchy is not None:
            self._qc_hierarchy.reset()
        if self._qc_reservoir is not None:
            self._qc_reservoir.reset()
        if self._qc_cascade is not None:
            self._qc_cascade.reset()

    def stop(self):
        """Stop stream processing."""
        self._running = False

    def reset(self):
        """Reset encoder state."""
        self.simulator.reset()
        if self._qc_hierarchy is not None:
            self._qc_hierarchy.reset()
        if self._qc_reservoir is not None:
            self._qc_reservoir.reset()
        if self._qc_cascade is not None:
            self._qc_cascade.reset()
        self.pattern_buffer.clear()
        self.feature_buffer.clear()
        self.frame_index = 0
        self._last_frame_metadata = None

    def set_angle(self, angle_idx: int):
        """Set operating angle."""
        self._angle_idx = angle_idx % len(self.moire.COMMENSURATE_ANGLES)

    def on_frame(self, callback: Callable[[StreamFrame], None]):
        """Register frame callback."""
        self._on_frame = callback

    def get_recent_patterns(self, n: int = 10) -> np.ndarray:
        """Get recent pattern frames."""
        return self.pattern_buffer.get_last(n)

    def get_recent_features(self, n: int = 10) -> np.ndarray:
        """Get recent feature vectors."""
        features, _ = self.feature_buffer.get_last(n)
        return features

    def get_last_frame_metadata(self) -> Dict[str, Any]:
        """Return metadata emitted by the latest processed frame."""
        if self._last_frame_metadata is None:
            return {
                "architecture_mode": self.architecture_mode,
                "available": False,
            }
        return {"available": True, **deepcopy(self._last_frame_metadata)}

    def get_quasicrystal_health(self) -> Dict[str, Any]:
        """Return quasicrystal verifier/cascade health metrics for live QA."""
        if self.architecture_mode != "quasicrystal":
            return {
                "architecture_mode": self.architecture_mode,
                "enabled": False,
            }

        verifier = self._qc_verifier
        reservoir = self._qc_reservoir
        cascade = self._qc_cascade

        return {
            "architecture_mode": self.architecture_mode,
            "enabled": True,
            "verifier_error_rate": float(verifier.error_rate) if verifier is not None else 0.0,
            "verifier_total_checks": int(verifier.total_checks) if verifier is not None else 0,
            "verifier_max_deviation": float(verifier.max_deviation) if verifier is not None else 0.0,
            "reservoir_spectral_radius": float(reservoir.spectral_radius) if reservoir is not None else 0.0,
            "cascade_padovan_length": int(len(cascade.padovan_steps)) if cascade is not None else 0,
        }

    def get_observability_snapshot(self) -> Dict[str, Any]:
        """Return a single, immutable snapshot for monitoring integrations."""
        snapshot = deepcopy(self.get_statistics())
        snapshot["last_frame_metadata"] = deepcopy(self.get_last_frame_metadata())
        if self.architecture_mode == "quasicrystal":
            snapshot["quasicrystal_health"] = deepcopy(self.get_quasicrystal_health())
        return snapshot

    def get_statistics(self) -> Dict:
        """Get stream statistics."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        stats = {
            'frames_processed': self.frame_index,
            'elapsed_time': elapsed,
            'fps': self.frame_index / elapsed if elapsed > 0 else 0,
            'buffer_fill': self.pattern_buffer.count / self.config.buffer_size,
            'running': self._running,
            'architecture_mode': self.architecture_mode,
            'last_frame_metadata': self.get_last_frame_metadata(),
        }
        if self.architecture_mode == "quasicrystal":
            stats['quasicrystal_health'] = self.get_quasicrystal_health()
        return stats
