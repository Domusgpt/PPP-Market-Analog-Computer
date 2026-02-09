"""
Temporal Stream Encoder - Proper Sequence Discrimination
========================================================

Implements temporal dynamics encoding per the specification:
- Fading memory with proper ORDER preservation
- Position-based input injection (not just velocity)
- Cascade evolution tracking (not just final state)
- Integrated Talbot gap control for logic switching
- Spectral logic gates (AND, XOR, NAND)

Key insight: The moiré pattern encodes temporal EVOLUTION, not static state.
"""

import numpy as np
from typing import Optional, Callable, Dict, Generator, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from ..core.fast_cascade import FastCascadeSimulator, CascadeResult
from ..core.fast_moire import FastMoireComputer
from ..control.tripole_actuator import TripoleActuator
from .buffer import RingBuffer, FrameBuffer, FeatureBuffer


class LogicMode(Enum):
    """Talbot-controlled logic modes."""
    XOR = "xor"      # Half-integer gap: edge detection
    AND = "and"      # Integer gap: coincidence detection
    NAND = "nand"    # Inverse AND


@dataclass
class TemporalConfig:
    """Configuration for temporal encoder."""
    grid_size: tuple = (64, 64)
    cascade_steps: int = 20
    buffer_size: int = 100

    # Temporal dynamics parameters
    decay_rate: float = 0.85          # Memory decay per frame (lower = more history)
    position_blend: float = 0.6       # How much new input blends into positions
    velocity_scale: float = 0.3       # Input contribution to velocity
    evolution_weight: float = 0.4     # Weight of evolution vs final state

    # Talbot control
    base_gap: float = 10.0            # Base Talbot gap (micrometers)
    wavelength: float = 550.0         # Light wavelength (nm)
    lattice_constant: float = 1.0     # Lattice constant (micrometers)

    # Logic control
    auto_logic_switch: bool = True    # Automatically switch logic based on input
    logic_threshold: float = 0.5      # Threshold for logic mode switching


@dataclass
class TemporalFrame:
    """Single frame from temporal encoder."""
    timestamp: float
    frame_index: int
    input_data: np.ndarray
    pattern: np.ndarray
    evolution_pattern: np.ndarray     # Pattern capturing cascade evolution
    features: np.ndarray
    logic_mode: LogicMode
    spectral_output: np.ndarray       # RGB bichromatic output


class TemporalStreamEncoder:
    """
    Temporal dynamics encoder with proper sequence discrimination.

    This encoder properly preserves temporal ORDER by:
    1. Position-based input injection (blending into values directly)
    2. Tracking cascade EVOLUTION (not just final state)
    3. Using sequence-dependent angle/gap modulation
    4. Maintaining proper fading memory with decay AFTER injection

    The key difference from the basic StreamEncoder:
    - Basic: velocity += input; then decay (loses order)
    - Temporal: values blend with input; track evolution; then decay (preserves order)

    Parameters
    ----------
    config : TemporalConfig
        Encoder configuration

    Example
    -------
    >>> encoder = TemporalStreamEncoder()
    >>> encoder.start()
    >>> for data in data_stream:
    ...     frame = encoder.process(data)
    ...     # frame.pattern encodes temporal context
    >>> encoder.stop()
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()

        # Core components
        self.simulator = FastCascadeSimulator(
            self.config.grid_size,
            coupling_strength=0.3,
            damping=0.1
        )
        self.moire = FastMoireComputer(
            lattice_constant=self.config.lattice_constant
        )

        # Tripole actuator for Talbot/angle control
        self.actuator = TripoleActuator()
        self.actuator.set_gap(self.config.base_gap)

        # Buffers
        self.pattern_buffer = FrameBuffer(
            self.config.buffer_size,
            self.config.grid_size
        )
        self.feature_buffer = FeatureBuffer(
            self.config.buffer_size,
            n_features=14  # Extended features
        )

        # Temporal state - key for sequence discrimination
        self._evolution_accumulator = np.zeros(self.config.grid_size, dtype=np.float64)
        self._sequence_position = np.zeros(self.config.grid_size, dtype=np.float64)
        self._input_history: List[np.ndarray] = []
        self._max_history = 10

        # State
        self.frame_index = 0
        self.start_time = 0.0
        self._running = False
        self._angle_idx = 2  # Default to 13.17 degrees
        self._current_logic = LogicMode.XOR

        # Callbacks
        self._on_frame: Optional[Callable[[TemporalFrame], None]] = None

    def _prepare_input(self, data: np.ndarray) -> np.ndarray:
        """Prepare and normalize input data."""
        # Reshape 1D to 2D
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

        # Normalize to [0, 1]
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)

        return data.astype(np.float64)

    def _inject_temporal_input(self, data: np.ndarray):
        """
        Inject input with proper temporal encoding.

        Key difference: We blend directly into POSITIONS (values),
        not just velocities. This preserves temporal order.
        """
        # 1. Position-based injection (KEY for temporal order)
        #    New input blends into current state
        blend = self.config.position_blend
        self.simulator.values = (1 - blend) * self.simulator.values + blend * data

        # 2. Also add velocity component for cascade dynamics
        self.simulator.velocities += data * self.config.velocity_scale

        # 3. Track sequence position (encodes WHERE in sequence we are)
        #    This creates a gradient that encodes temporal position
        self._sequence_position = 0.9 * self._sequence_position + 0.1 * data

        # 4. Store in history for context
        self._input_history.append(data.copy())
        if len(self._input_history) > self._max_history:
            self._input_history.pop(0)

    def _run_tracked_cascade(self) -> Tuple[CascadeResult, np.ndarray]:
        """
        Run cascade while tracking evolution.

        Returns both the final result AND an evolution pattern
        that encodes the CASCADE DYNAMICS (not just final state).
        """
        n_steps = self.config.cascade_steps
        evolution = np.zeros(self.config.grid_size, dtype=np.float64)

        # Run cascade, accumulating evolution
        for step in range(n_steps):
            old_values = self.simulator.values.copy()
            energy = self.simulator.step(dt=0.02)

            # Track changes weighted by step (later steps weight more)
            step_weight = (step + 1) / n_steps
            changes = np.abs(self.simulator.values - old_values)
            evolution += changes * step_weight

            # Early stop if converged
            if energy < 1e-6:
                break

        # Normalize evolution
        if evolution.max() > 0:
            evolution = evolution / evolution.max()

        # Get final result
        result = CascadeResult(
            final_state=self.simulator.values.copy(),
            discrete_state=self._quantize_state(self.simulator.values),
            history=None,
            steps_taken=step + 1,
            total_energy=energy
        )

        return result, evolution

    def _quantize_state(self, values: np.ndarray) -> np.ndarray:
        """Quantize to tristable states."""
        discrete = np.zeros_like(values)
        discrete[values < 0.25] = 0.0
        discrete[(values >= 0.25) & (values < 0.75)] = 0.5
        discrete[values >= 0.75] = 1.0
        return discrete

    def _compute_talbot_gap(self, input_data: np.ndarray) -> float:
        """
        Compute Talbot gap based on input characteristics.

        Z_T = (3/2) * a^2 / lambda

        Integer Z_T -> AND logic
        Half-integer Z_T -> XOR logic
        """
        a = self.config.lattice_constant * 1000  # Convert to nm
        lambda_nm = self.config.wavelength

        # Base Talbot distance
        z_t = (3/2) * (a ** 2) / lambda_nm

        # Modulate based on input variance (high variance -> XOR, low -> AND)
        input_variance = np.var(input_data)

        if self.config.auto_logic_switch:
            if input_variance > self.config.logic_threshold:
                # High variance: use half-integer (XOR for edge detection)
                gap = z_t * 0.5
                self._current_logic = LogicMode.XOR
            else:
                # Low variance: use integer (AND for coincidence)
                gap = z_t
                self._current_logic = LogicMode.AND
        else:
            gap = self.config.base_gap

        return gap

    def _select_angle(self) -> float:
        """Select twist angle based on temporal context."""
        angle = self.moire.COMMENSURATE_ANGLES[self._angle_idx]

        # Optionally modulate based on sequence history
        if len(self._input_history) >= 2:
            # Compute temporal gradient
            recent = self._input_history[-1]
            previous = self._input_history[-2]
            temporal_change = np.mean(np.abs(recent - previous))

            # Slight angle modulation based on change rate
            # More change -> use larger angle for more sensitivity
            if temporal_change > 0.3:
                angle_idx = min(self._angle_idx + 1, len(self.moire.COMMENSURATE_ANGLES) - 1)
            elif temporal_change < 0.1:
                angle_idx = max(self._angle_idx - 1, 1)
            else:
                angle_idx = self._angle_idx

            angle = self.moire.COMMENSURATE_ANGLES[angle_idx]

        return angle

    def process(self, data: np.ndarray) -> TemporalFrame:
        """
        Process single input and return temporally-encoded frame.

        The output pattern encodes:
        1. Current input information
        2. Temporal position in sequence (fading memory)
        3. Cascade evolution dynamics
        4. Spectral logic output
        """
        timestamp = time.time() - self.start_time

        # Prepare input
        data = self._prepare_input(data)

        # Apply fading memory decay BEFORE injection (maintains history)
        self.simulator.values *= self.config.decay_rate
        self._evolution_accumulator *= self.config.decay_rate

        # Inject input with temporal encoding
        self._inject_temporal_input(data)

        # Compute Talbot gap and update actuator
        gap = self._compute_talbot_gap(data)
        self.actuator.set_gap(gap)

        # Select angle based on temporal context
        angle = self._select_angle()
        self.actuator.command_rotate(angle, snap_to_commensurate=True)

        # Run cascade with evolution tracking
        result, evolution = self._run_tracked_cascade()

        # Accumulate evolution (KEY for sequence discrimination)
        self._evolution_accumulator += evolution * self.config.evolution_weight

        # Compute moiré pattern with both state and evolution
        # Use evolution as layer1 modulation, final state as layer2
        moire_result = self.moire.compute(
            twist_angle=angle,
            grid_size=self.config.grid_size,
            layer1_state=self._evolution_accumulator,  # Encodes temporal history
            layer2_state=result.final_state             # Current state
        )

        # Combine final pattern: weighted mix of intensity and evolution
        combined_pattern = (
            (1 - self.config.evolution_weight) * moire_result.intensity +
            self.config.evolution_weight * self._evolution_accumulator
        )

        # Normalize combined pattern
        p_min, p_max = combined_pattern.min(), combined_pattern.max()
        if p_max > p_min:
            combined_pattern = (combined_pattern - p_min) / (p_max - p_min)

        # Extract features (includes temporal features)
        features = self._extract_temporal_features(
            combined_pattern, result.final_state, evolution, data
        )

        # Create frame
        frame = TemporalFrame(
            timestamp=timestamp,
            frame_index=self.frame_index,
            input_data=data,
            pattern=combined_pattern.astype(np.float32),
            evolution_pattern=self._evolution_accumulator.astype(np.float32),
            features=features,
            logic_mode=self._current_logic,
            spectral_output=moire_result.spectral
        )

        # Update buffers
        self.pattern_buffer.append(combined_pattern)
        self.feature_buffer.append(features, timestamp)
        self.frame_index += 1

        # Callback
        if self._on_frame:
            self._on_frame(frame)

        return frame

    def _extract_temporal_features(
        self,
        pattern: np.ndarray,
        state: np.ndarray,
        evolution: np.ndarray,
        input_data: np.ndarray
    ) -> np.ndarray:
        """Extract features including temporal information."""
        features = []

        # Pattern features
        features.extend([
            np.mean(pattern),
            np.std(pattern),
            np.min(pattern),
            np.max(pattern),
        ])

        # Evolution features (KEY for temporal encoding)
        features.extend([
            np.mean(evolution),
            np.std(evolution),
            np.sum(evolution > 0.5) / evolution.size,
        ])

        # Sequence position features
        features.extend([
            np.mean(self._sequence_position),
            np.std(self._sequence_position),
        ])

        # Temporal gradient (change from last input)
        if len(self._input_history) >= 2:
            temporal_diff = np.mean(np.abs(
                self._input_history[-1] - self._input_history[-2]
            ))
            features.append(temporal_diff)
        else:
            features.append(0.0)

        # FFT features (spatial frequency content)
        fft = np.abs(np.fft.fftshift(np.fft.fft2(pattern)))
        features.extend([
            np.mean(fft),
            np.max(fft)
        ])

        # State features
        features.extend([
            np.mean(state),
            np.std(state),
        ])

        return np.array(features, dtype=np.float32)

    def process_stream(
        self,
        data_generator: Generator[np.ndarray, None, None]
    ) -> Generator[TemporalFrame, None, None]:
        """Process stream of data."""
        self.start()

        for data in data_generator:
            yield self.process(data)

        self.stop()

    def start(self):
        """Start stream processing."""
        self._running = True
        self.start_time = time.time()
        self.frame_index = 0
        self.reset()

    def stop(self):
        """Stop stream processing."""
        self._running = False

    def reset(self):
        """Reset encoder state."""
        self.simulator.reset()
        self._evolution_accumulator.fill(0)
        self._sequence_position.fill(0)
        self._input_history.clear()
        self.pattern_buffer.clear()
        self.feature_buffer.clear()
        self.frame_index = 0

    def set_angle_index(self, angle_idx: int):
        """Set operating angle index."""
        self._angle_idx = angle_idx % len(self.moire.COMMENSURATE_ANGLES)

    def set_logic_mode(self, mode: LogicMode):
        """Manually set logic mode."""
        self._current_logic = mode
        self.config.auto_logic_switch = False

    def on_frame(self, callback: Callable[[TemporalFrame], None]):
        """Register frame callback."""
        self._on_frame = callback

    def get_recent_patterns(self, n: int = 10) -> np.ndarray:
        """Get recent pattern frames."""
        return self.pattern_buffer.get_last(n)

    def get_recent_features(self, n: int = 10) -> np.ndarray:
        """Get recent feature vectors."""
        features, _ = self.feature_buffer.get_last(n)
        return features

    def get_statistics(self) -> Dict:
        """Get stream statistics."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        return {
            'frames_processed': self.frame_index,
            'elapsed_time': elapsed,
            'fps': self.frame_index / elapsed if elapsed > 0 else 0,
            'buffer_fill': self.pattern_buffer.count / self.config.buffer_size,
            'running': self._running,
            'current_angle': self.moire.COMMENSURATE_ANGLES[self._angle_idx],
            'current_logic': self._current_logic.value,
            'evolution_mean': float(np.mean(self._evolution_accumulator)),
        }
