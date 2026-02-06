"""
Optical Kirigami Moiré - Unified Computational Pipeline
======================================================

This is the main entry point for the Emergent Optical Cognition system.

The pipeline implements:
1. Input encoding via kirigami reservoir (physical dynamics)
2. Moiré pattern generation (optical interference)
3. Feature extraction for Vision LLM consumption
4. Linear readout for classification tasks

The system relocates computational burden from digital to optical domain:
- Input → Kirigami perturbation → Cascading dynamics → Moiré pattern → Output

All operations respect the three rule sets:
- Rule 1: Angular Commensurability (Pythagorean)
- Rule 2: Trilatic Tilt Symmetry (Orthogonality)
- Rule 3: Talbot Distance (Integer Gap)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable
from enum import Enum
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing modules (relative imports for package)
from .rules.enforcer import RuleEnforcer, CommensurateLock, TalbotLock, TiltLock, LogicPolarity
from .reservoir.readout import ReservoirReadout, ReadoutConfig, MoireFeatureExtractor


class ComputationMode(Enum):
    """Operating modes for the optical computer."""
    TRANSPARENT = "transparent"     # 0° twist, all-pass
    EDGE_DETECT = "edge_detect"     # 7.34° twist, fine edges
    TEXTURE = "texture"             # 9.43° twist, texture analysis
    INTERMEDIATE = "intermediate"   # 13.17° twist, mid-band filtering
    COARSE = "coarse"               # 21.79° twist, coarse features
    MAGIC = "magic"                 # ~1.1° twist, flat band (threshold)


@dataclass
class PipelineConfig:
    """Configuration for the optical kirigami moiré pipeline."""
    # Grid dimensions
    grid_size: Tuple[int, int] = (64, 64)

    # Physical parameters
    lattice_constant: float = 1.0     # micrometers
    wavelength: float = 550.0         # nm (green light)

    # Reservoir parameters
    base_stiffness: float = 1.0
    coupling_strength: float = 0.5
    damping: float = 0.1
    cascade_steps: int = 50

    # Tripole actuator
    tripole_radius: float = 25.0      # mm
    max_extension: float = 100.0      # micrometers

    # Rule enforcement
    strict_rules: bool = False        # Raise on rule violations

    # Readout
    n_outputs: int = 10               # Classification classes


@dataclass
class EncodingResult:
    """Result of encoding input data."""
    input_shape: Tuple[int, ...]
    reservoir_state: np.ndarray       # Kirigami cell states
    moire_pattern: np.ndarray         # Intensity field
    spectral_pattern: Optional[np.ndarray]  # RGB if bichromatic
    cascade_steps: int                # Steps to convergence
    angle_lock: CommensurateLock
    gap_lock: TalbotLock
    features: Optional[np.ndarray] = None


@dataclass
class SystemState:
    """Current state of the optical system."""
    # Geometric state
    twist_angle: float = 0.0
    gap: float = 1.0
    tip: float = 0.0
    tilt: float = 0.0

    # Locks
    angle_lock: Optional[CommensurateLock] = None
    gap_lock: Optional[TalbotLock] = None
    tilt_lock: Optional[TiltLock] = None

    # Computed values
    moire_period: float = float('inf')
    logic_polarity: LogicPolarity = LogicPolarity.POSITIVE


class OpticalKirigamiMoire:
    """
    Unified pipeline for optical kirigami moiré computation.

    This class orchestrates:
    - Kirigami reservoir (tristable cell dynamics)
    - Moiré interference (bichromatic optical logic)
    - Tripole actuation (angle/gap/tilt control)
    - Rule enforcement (three rule sets)
    - Linear readout (classification)

    Example
    -------
    >>> okm = OpticalKirigamiMoire()
    >>> result = okm.encode(input_image)
    >>> features = okm.extract_features(result.moire_pattern)
    >>> prediction = okm.classify(result.moire_pattern)

    Parameters
    ----------
    config : PipelineConfig, optional
        Configuration parameters
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize rule enforcer
        self.rules = RuleEnforcer(
            lattice_constant=self.config.lattice_constant,
            wavelength=self.config.wavelength,
            strict_mode=self.config.strict_rules
        )

        # Initialize state
        self._state = SystemState()

        # Initialize components (lazy loading)
        self._layer1: Optional[Any] = None  # Kirigami sheet (Cyan/Hole)
        self._layer2: Optional[Any] = None  # Kirigami sheet (Red/Dot)
        self._tripole: Optional[Any] = None
        self._moire: Optional[Any] = None
        self._talbot: Optional[Any] = None

        # Readout layer
        self._readout = ReservoirReadout(ReadoutConfig(
            n_features=256,
            n_outputs=self.config.n_outputs
        ))

        # Feature extractor
        self._feature_extractor = MoireFeatureExtractor(self.config.grid_size)

        # History for temporal analysis
        self._history: List[EncodingResult] = []

    def _ensure_initialized(self):
        """Lazy initialization of components."""
        if self._layer1 is not None:
            return

        # Import here to avoid circular imports
        try:
            from hemoc_stain_glass_src.kirigami.kirigami_sheet import (
                KirigamiSheet, SheetConfig, CutPattern
            )
            from hemoc_stain_glass_src.physics.moire_interference import MoireInterference
            from hemoc_stain_glass_src.physics.talbot_resonator import TalbotResonator
            from hemoc_stain_glass_src.control.tripole_actuator import TripoleActuator
        except ImportError:
            # Fallback to local implementations
            from .kirigami import KirigamiSheet, SheetConfig, CutPattern
            from .physics import MoireInterference, TalbotResonator
            from .control import TripoleActuator

        nx, ny = self.config.grid_size

        # Create kirigami sheets
        sheet_config = SheetConfig(
            n_cells_x=nx,
            n_cells_y=ny,
            lattice_constant=self.config.lattice_constant,
            base_stiffness=self.config.base_stiffness,
            coupling_strength=self.config.coupling_strength,
            damping=self.config.damping,
            cut_pattern=CutPattern.RADIAL_SOFT
        )

        self._layer1 = KirigamiSheet(sheet_config, layer_type="hole_array")
        self._layer2 = KirigamiSheet(sheet_config, layer_type="dot_array")

        # Create moiré engine
        self._moire = MoireInterference(
            lattice_constant=self.config.lattice_constant,
            wavelength_red=650.0,
            wavelength_cyan=500.0
        )

        # Create Talbot resonator
        self._talbot = TalbotResonator(
            lattice_constant=self.config.lattice_constant,
            wavelength=self.config.wavelength
        )

        # Create tripole actuator
        self._tripole = TripoleActuator(
            radius=self.config.tripole_radius,
            max_extension=self.config.max_extension
        )

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_mode(self, mode: ComputationMode):
        """
        Set computation mode (selects twist angle).

        Parameters
        ----------
        mode : ComputationMode
            Desired operating mode
        """
        angle_map = {
            ComputationMode.TRANSPARENT: 0.0,
            ComputationMode.EDGE_DETECT: 7.34,
            ComputationMode.TEXTURE: 9.43,
            ComputationMode.INTERMEDIATE: 13.17,
            ComputationMode.COARSE: 21.79,
            ComputationMode.MAGIC: 1.1,
        }

        target_angle = angle_map[mode]
        self.set_twist_angle(target_angle)

    def set_twist_angle(self, angle: float):
        """
        Set twist angle (enforces Rule 1).

        Parameters
        ----------
        angle : float
            Target twist angle in degrees
        """
        lock = self.rules.enforce_angle(angle)
        self._state.twist_angle = lock.angle
        self._state.angle_lock = lock
        self._state.moire_period = self.rules.get_moiré_period(lock.angle)

    def set_gap(self, gap: float):
        """
        Set layer gap (enforces Rule 3).

        Parameters
        ----------
        gap : float
            Target gap in micrometers
        """
        lock = self.rules.enforce_gap(gap)
        self._state.gap = lock.gap
        self._state.gap_lock = lock
        self._state.logic_polarity = lock.polarity

    def set_tilt(self, angle: float, axis: int = 0):
        """
        Set tilt angle (enforces Rule 2).

        Parameters
        ----------
        angle : float
            Tilt magnitude in degrees
        axis : int
            Tilt axis index (0-5 for k × 60°)
        """
        lock = self.rules.enforce_tilt(angle, axis)
        self._state.tilt = lock.tilt_angle
        self._state.tilt_lock = lock

    def set_logic_polarity(self, polarity: LogicPolarity):
        """
        Set logic polarity by adjusting gap to appropriate Talbot mode.

        POSITIVE: Integer Talbot → AND/OR logic
        NEGATIVE: Half-integer Talbot → NAND/XOR logic
        """
        positive_gap, negative_gap = self.rules.get_logic_gaps(order=1)

        if polarity == LogicPolarity.POSITIVE:
            self.set_gap(positive_gap)
        else:
            self.set_gap(negative_gap)

    @property
    def state(self) -> SystemState:
        """Get current system state."""
        return self._state

    # =========================================================================
    # ENCODING
    # =========================================================================

    def encode(
        self,
        data: np.ndarray,
        normalize: bool = True,
        run_cascade: bool = True
    ) -> EncodingResult:
        """
        Encode input data into moiré pattern.

        This is the main computation function:
        1. Inject data into kirigami reservoir
        2. Run cascade dynamics (fading memory)
        3. Generate bichromatic moiré pattern

        Parameters
        ----------
        data : np.ndarray
            Input data (1D, 2D, or 3D array)
        normalize : bool
            Normalize input to [0, 1]
        run_cascade : bool
            Run reservoir cascade dynamics

        Returns
        -------
        EncodingResult
            Encoded moiré pattern with metadata
        """
        self._ensure_initialized()

        # Prepare input
        input_shape = data.shape
        if data.ndim == 1:
            # 1D: reshape to square-ish 2D
            side = int(np.ceil(np.sqrt(len(data))))
            data_2d = np.zeros((side, side))
            data_2d.flat[:len(data)] = data
        elif data.ndim == 2:
            data_2d = data
        elif data.ndim == 3:
            # RGB: convert to grayscale
            data_2d = np.mean(data, axis=2)
        else:
            raise ValueError(f"Unsupported input dimensions: {data.ndim}")

        # Normalize
        if normalize:
            data_min, data_max = data_2d.min(), data_2d.max()
            if data_max > data_min:
                data_2d = (data_2d - data_min) / (data_max - data_min)

        # Inject into kirigami layers
        self._layer1.inject_input(data_2d, input_scale=0.8)

        # Complementary input for layer 2 (opposite spectrum)
        self._layer2.inject_input(1 - data_2d, input_scale=0.8)

        # Run cascade dynamics
        if run_cascade:
            steps1 = self._layer1.run_cascade(
                n_steps=self.config.cascade_steps,
                convergence_threshold=1e-4
            )
            steps2 = self._layer2.run_cascade(
                n_steps=self.config.cascade_steps,
                convergence_threshold=1e-4
            )
            cascade_steps = max(steps1, steps2)
        else:
            cascade_steps = 0

        # Get reservoir states
        layer1_state = self._layer1.get_state_field()
        layer2_state = self._layer2.get_state_field()

        # Ensure angle and gap are set
        if self._state.angle_lock is None:
            self.set_twist_angle(9.43)  # Default: texture mode
        if self._state.gap_lock is None:
            self.set_gap(self.rules.talbot_length)  # Default: first Talbot

        # Generate moiré pattern
        pattern = self._moire.compute_bichromatic_moire(
            twist_angle=self._state.twist_angle,
            grid_size=self.config.grid_size,
            field_size=(50.0, 50.0),
            layer1_state=layer1_state,
            layer2_state=layer2_state
        )

        # Extract features
        features = self._feature_extractor.extract(pattern.intensity_field)

        result = EncodingResult(
            input_shape=input_shape,
            reservoir_state=(layer1_state + layer2_state) / 2,
            moire_pattern=pattern.intensity_field,
            spectral_pattern=pattern.spectral_field,
            cascade_steps=cascade_steps,
            angle_lock=self._state.angle_lock,
            gap_lock=self._state.gap_lock,
            features=features
        )

        # Record history
        self._history.append(result)

        return result

    def encode_sequence(
        self,
        sequence: List[np.ndarray],
        reset_between: bool = False
    ) -> List[EncodingResult]:
        """
        Encode a sequence of inputs (temporal processing).

        The reservoir maintains fading memory between inputs
        unless reset_between is True.

        Parameters
        ----------
        sequence : List[np.ndarray]
            Sequence of input arrays
        reset_between : bool
            Reset reservoir between inputs

        Returns
        -------
        List[EncodingResult]
            List of encoded patterns
        """
        results = []

        for data in sequence:
            if reset_between:
                self.reset_reservoir()

            result = self.encode(data)
            results.append(result)

        return results

    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================

    def extract_features(self, pattern: np.ndarray) -> np.ndarray:
        """
        Extract features from moiré pattern for Vision LLM.

        Parameters
        ----------
        pattern : np.ndarray
            Moiré intensity pattern

        Returns
        -------
        np.ndarray
            Feature vector
        """
        return self._feature_extractor.extract(pattern)

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    def train_readout(
        self,
        patterns: List[np.ndarray],
        labels: np.ndarray,
        verbose: bool = False
    ) -> float:
        """
        Train readout layer for classification.

        Parameters
        ----------
        patterns : List[np.ndarray]
            Training moiré patterns
        labels : np.ndarray
            Class labels

        Returns
        -------
        float
            Training accuracy
        """
        return self._readout.train(patterns, labels, verbose)

    def classify(self, pattern: np.ndarray) -> Tuple[int, float]:
        """
        Classify a moiré pattern.

        Parameters
        ----------
        pattern : np.ndarray
            Moiré pattern to classify

        Returns
        -------
        Tuple[int, float]
            (predicted_class, confidence)
        """
        result = self._readout.predict(pattern)
        return result.prediction, result.confidence

    # =========================================================================
    # CONTROL
    # =========================================================================

    def reset_reservoir(self):
        """Reset kirigami sheets to initial state."""
        self._ensure_initialized()
        self._layer1.reset()
        self._layer2.reset()

    def reset(self):
        """Full system reset."""
        self._state = SystemState()
        self._history.clear()

        if self._layer1 is not None:
            self._layer1.reset()
            self._layer2.reset()

        if self._tripole is not None:
            self._tripole.reset()

    def set_attention_weights(self, weights: np.ndarray):
        """
        Set spatial attention weights (stiffness map).

        From Section 3.2: The spatial variation in stiffness
        acts as the Weight Matrix of the neural network.

        Parameters
        ----------
        weights : np.ndarray
            2D array of attention weights
        """
        self._ensure_initialized()
        self._layer1.set_stiffness_map(weights)
        self._layer2.set_stiffness_map(weights)

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def analyze_pattern(self, pattern: np.ndarray) -> Dict[str, Any]:
        """
        Analyze moiré pattern for debugging/visualization.

        Returns
        -------
        Dict
            Analysis including contrast, frequency, forks, etc.
        """
        features = self.extract_features(pattern)

        # Compute contrast
        contrast = (pattern.max() - pattern.min()) / (pattern.max() + pattern.min() + 1e-8)

        # Compute dominant frequency (from FFT)
        fft = np.abs(np.fft.fft2(pattern))
        fft_shifted = np.fft.fftshift(fft)
        cy, cx = fft_shifted.shape[0] // 2, fft_shifted.shape[1] // 2

        # Find peak (excluding DC)
        fft_shifted[cy-2:cy+3, cx-2:cx+3] = 0
        peak_idx = np.unravel_index(np.argmax(fft_shifted), fft_shifted.shape)
        dominant_freq = np.sqrt((peak_idx[0] - cy)**2 + (peak_idx[1] - cx)**2)

        return {
            "mean_intensity": float(np.mean(pattern)),
            "std_intensity": float(np.std(pattern)),
            "contrast": float(contrast),
            "dominant_frequency": float(dominant_freq),
            "moire_period": self._state.moire_period,
            "twist_angle": self._state.twist_angle,
            "gap": self._state.gap,
            "logic_polarity": self._state.logic_polarity.value,
            "features": features,
        }

    def get_history(self) -> List[EncodingResult]:
        """Get encoding history."""
        return self._history.copy()

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_commensurate_angles(self) -> Dict[Tuple[int, int], float]:
        """Get all available commensurate angles."""
        return {k: v["angle"] for k, v in self.rules.COMMENSURATE_TABLE.items()}

    def get_talbot_ladder(self, max_gap: float = 10.0) -> List[Dict]:
        """Get all valid Talbot gaps up to max_gap."""
        self._ensure_initialized()
        states = self._talbot.generate_resonance_ladder(max_gap)
        return [
            {"gap": s.gap, "order": s.order, "mode": s.mode.value, "polarity": s.logic_polarity}
            for s in states
        ]

    def __repr__(self) -> str:
        return (f"OpticalKirigamiMoire(grid={self.config.grid_size}, "
                f"angle={self._state.twist_angle:.2f}°, "
                f"gap={self._state.gap:.3f}μm, "
                f"polarity={self._state.logic_polarity.value})")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_encoder(
    grid_size: Tuple[int, int] = (64, 64),
    mode: ComputationMode = ComputationMode.TEXTURE
) -> OpticalKirigamiMoire:
    """
    Create a configured optical kirigami moiré encoder.

    Parameters
    ----------
    grid_size : Tuple[int, int]
        Processing grid size
    mode : ComputationMode
        Operating mode

    Returns
    -------
    OpticalKirigamiMoire
        Configured encoder instance
    """
    config = PipelineConfig(grid_size=grid_size)
    encoder = OpticalKirigamiMoire(config)
    encoder.set_mode(mode)
    return encoder


def encode_image(
    image: np.ndarray,
    mode: ComputationMode = ComputationMode.TEXTURE
) -> EncodingResult:
    """
    Quick encoding of a single image.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB)
    mode : ComputationMode
        Operating mode

    Returns
    -------
    EncodingResult
        Encoded moiré pattern
    """
    encoder = create_encoder(mode=mode)
    return encoder.encode(image)
