#!/usr/bin/env python3
"""
Optical Kirigami Moiré - Main Simulation Engine
================================================

This is the primary simulation loop that ties together:
- Physics kernel (moiré interference, Talbot resonance)
- Kirigami reservoir (tristable cells, cascading dynamics)
- Tripole actuator control

The system functions as a data encoder, transforming arbitrary input
(audio, images, sensor data) into optical moiré patterns suitable for
visual pattern recognition and machine learning training.

Usage:
    python main.py --mode demo
    python main.py --input audio.wav --output encoded_patterns/
    python main.py --input image.png --mode encode
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union, Literal
from dataclasses import dataclass
import json

# Local imports
from .physics.trilatic_lattice import TrilaticLattice
from .physics.moire_interference import MoireInterference
from .physics.talbot_resonator import TalbotResonator
from .kirigami.tristable_cell import TristableCell, CellState
from .kirigami.kirigami_sheet import KirigamiSheet, SheetConfig, CutPattern
from .control.tripole_actuator import TripoleActuator
from .geometry.quasicrystal_architecture import (
    QuasicrystallineReservoir,
    NumberFieldHierarchy,
    GaloisVerifier,
    PadovanCascade,
    FiveFoldAllocator,
)


@dataclass
class EncoderConfig:
    """Configuration for the optical moiré encoder."""
    grid_size: Tuple[int, int] = (64, 64)       # Resolution of kirigami sheets
    lattice_constant: float = 1.0               # Base lattice spacing (μm)
    wavelength: float = 550.0                   # Operating wavelength (nm)
    cascade_steps: int = 50                     # Steps for reservoir dynamics
    n_angle_states: int = 5                     # Commensurate angle states
    n_gap_states: int = 3                       # Talbot gap states
    input_channels: int = 1                     # Number of input channels
    architecture_mode: Literal["legacy", "quasicrystal"] = "legacy"


class OpticalKirigamiEncoder:
    """
    Optical Kirigami Moiré Encoder for data-to-pattern transformation.

    This encoder converts arbitrary input data (audio, images, sensor
    readings) into structured moiré interference patterns. The patterns
    can be used for:
    - Training visual pattern recognition models
    - Data compression via optical computing
    - Analog feature extraction

    The encoding process:
    1. Input normalization and spatial mapping
    2. Kirigami reservoir injection
    3. Cascading dynamics (fading memory processing)
    4. Moiré pattern generation via layer interference
    5. Output as structured visual patterns

    Parameters
    ----------
    config : EncoderConfig
        Encoder configuration parameters
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        self.architecture_mode = self.config.architecture_mode

        # Initialize physics components
        self.lattice = TrilaticLattice(self.config.lattice_constant)
        self.moire = MoireInterference(
            lattice_constant=self.config.lattice_constant,
            wavelength_red=650.0,
            wavelength_cyan=500.0
        )
        self.talbot = TalbotResonator(
            lattice_constant=self.config.lattice_constant,
            wavelength=self.config.wavelength
        )

        # Initialize kirigami sheets (Layer 1: Cyan/Hole, Layer 2: Red/Dot)
        sheet_config = SheetConfig(
            n_cells_x=self.config.grid_size[0],
            n_cells_y=self.config.grid_size[1],
            lattice_constant=self.config.lattice_constant,
            cut_pattern=CutPattern.RADIAL_SOFT
        )

        self.layer1 = KirigamiSheet(sheet_config, layer_type="hole_array")
        self.layer2 = KirigamiSheet(sheet_config, layer_type="dot_array")

        # Initialize actuators for each layer
        self.actuator1 = TripoleActuator()
        self.actuator2 = TripoleActuator()

        # Precompute commensurate angles and Talbot gaps
        self._setup_state_spaces()

        # Encoding state
        self.current_angle_idx = 0
        self.current_gap_idx = 1  # Start at first integer Talbot

        # Optional quasicrystal architecture (Phase A integration spine)
        self._qc: Optional[Dict[str, object]] = None
        if self.architecture_mode == "quasicrystal":
            qc_grid = min(self.config.grid_size)
            self._qc = {
                "hierarchy": NumberFieldHierarchy(base_size=8),
                "reservoir": QuasicrystallineReservoir(n_reservoir=64, input_dim=8),
                "verifier": GaloisVerifier(tolerance=1e-6),
                "cascade": PadovanCascade(
                    max_steps=max(20, self.config.cascade_steps * 4),
                    grid_size=qc_grid,
                    coupling=0.3,
                ),
                "allocator": FiveFoldAllocator(total_budget=1.0),
            }

    def _project_to_qc_vector(self, data: np.ndarray) -> np.ndarray:
        """Project a 2D field to an 8D summary vector for quasicrystal dynamics."""
        splits = np.array_split(data.flatten(), 8)
        return np.array([float(np.mean(s)) if len(s) else 0.0 for s in splits], dtype=np.float64)

    def _reshape_reservoir_to_grid(self, reservoir_state: np.ndarray, grid_size: int) -> np.ndarray:
        """Map a 1D reservoir state into a 2D grid for cascade forcing."""
        side = grid_size
        needed = side * side
        tiled = np.resize(reservoir_state, needed)
        return tiled.reshape(side, side)

    def _setup_state_spaces(self):
        """Setup discrete state spaces for angle and gap control."""
        # Commensurate angles from Rule Set 1
        self.commensurate_angles = [
            0.0,      # Aligned
            7.34,     # Edge detection
            9.43,     # Fine detail
            13.17,    # Intermediate
            21.79     # Coarse filtering
        ]

        # Talbot gaps from Rule Set 3
        talbot_ladder = self.talbot.generate_resonance_ladder(
            max_gap=50.0,
            include_half_integer=True
        )
        self.talbot_states = talbot_ladder[:6]  # First 6 states

    def encode_data(
        self,
        data: np.ndarray,
        mode: str = "spatial"
    ) -> Dict[str, np.ndarray]:
        """
        Encode arbitrary data into moiré patterns.

        Parameters
        ----------
        data : np.ndarray
            Input data array. Can be:
            - 1D: Time series (audio, sensor)
            - 2D: Image or spectrogram
            - 3D: Multi-channel data
        mode : str
            Encoding mode:
            - "spatial": Map to 2D kirigami excitation
            - "temporal": Process as time series with reservoir
            - "spectral": Frequency-domain encoding

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - "moire_intensity": Moiré intensity pattern
            - "moire_spectral": RGB spectral pattern
            - "layer1_state": Layer 1 cell states
            - "layer2_state": Layer 2 cell states
            - "features": Extracted feature vector
        """
        # Normalize input to [0, 1]
        data = self._normalize_input(data)

        # Reshape data to 2D if needed
        if data.ndim == 1:
            data = self._reshape_1d_to_2d(data)
        elif data.ndim == 3:
            # Use first channel or average
            data = np.mean(data, axis=-1)

        # Resize to grid size
        data = self._resize_to_grid(data)

        qc_stats = None
        if self.architecture_mode == "quasicrystal" and self._qc is not None:
            hierarchy: NumberFieldHierarchy = self._qc["hierarchy"]  # type: ignore[assignment]
            reservoir: QuasicrystallineReservoir = self._qc["reservoir"]  # type: ignore[assignment]
            verifier: GaloisVerifier = self._qc["verifier"]  # type: ignore[assignment]
            cascade: PadovanCascade = self._qc["cascade"]  # type: ignore[assignment]
            allocator: FiveFoldAllocator = self._qc["allocator"]  # type: ignore[assignment]

            qvec = self._project_to_qc_vector(data)
            hierarchy_states = hierarchy.step(qvec)
            reservoir_state = reservoir.step(hierarchy_states[1])

            cascade_grid = self._reshape_reservoir_to_grid(reservoir_state, cascade.grid_size)
            cascade.inject(cascade_grid)
            cascade_result = cascade.run(n_epochs=1)

            layer1_state = cascade_result["final_state"]
            layer2_state = 1.0 - layer1_state

            # Upsample to configured moiré grid when cascade grid is smaller
            if layer1_state.shape != self.config.grid_size:
                layer1_state = self._resize_to_grid(layer1_state)
                layer2_state = self._resize_to_grid(layer2_state)

            galois = verifier.verify(qvec)
            qc_stats = {
                "galois_valid": bool(galois["valid"]),
                "galois_deviation": float(galois["deviation"]),
                "galois_ratio": float(galois["ratio"]),
                "galois_ratio_valid": bool(galois["ratio_valid"]),
                "galois_ratio_deviation": float(galois["ratio_deviation"]),
                "galois_product": float(galois["product"]),
                "galois_expected_product": float(galois["expected_product"]),
                "galois_product_valid": bool(galois["product_valid"]),
                "galois_product_deviation": float(galois["product_deviation"]),
                "reservoir_spectral_radius": float(reservoir.spectral_radius),
                "padovan_steps": int(cascade_result["n_padovan_steps"]),
                "allocator_per_node": float(allocator.per_node_budget),
            }
        else:
            # Inject into Layer 1 (primary encoding)
            self.layer1.reset()
            self.layer1.inject_input(data, input_scale=0.8)

            # Run reservoir dynamics
            self.layer1.run_cascade(
                n_steps=self.config.cascade_steps,
                dt=0.01
            )

            # Complementary encoding for Layer 2
            # Use inverted/phase-shifted data for bichromatic logic
            complement_data = 1.0 - data
            self.layer2.reset()
            self.layer2.inject_input(complement_data, input_scale=0.6)
            self.layer2.run_cascade(n_steps=self.config.cascade_steps // 2)

            # Get layer states
            layer1_state = self.layer1.get_state_field()
            layer2_state = self.layer2.get_state_field()

        # Get current configuration
        twist_angle = self.commensurate_angles[self.current_angle_idx]
        gap_state = self.talbot_states[self.current_gap_idx]

        # Generate moiré pattern
        pattern = self.moire.compute_bichromatic_moire(
            twist_angle=twist_angle,
            grid_size=self.config.grid_size,
            field_size=(50.0, 50.0),
            layer1_state=layer1_state,
            layer2_state=layer2_state
        )

        # Extract features from pattern
        features = self._extract_features(pattern, layer1_state, layer2_state)

        result = {
            "moire_intensity": pattern.intensity_field,
            "moire_spectral": pattern.spectral_field,
            "layer1_state": layer1_state,
            "layer2_state": layer2_state,
            "features": features,
            "twist_angle": twist_angle,
            "gap": gap_state.gap,
            "logic_mode": gap_state.logic_polarity,
            "architecture_mode": self.architecture_mode,
        }
        if qc_stats is not None:
            result["quasicrystal"] = qc_stats
        return result

    def encode_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 22050,
        frame_size: int = 1024,
        hop_size: int = 512
    ) -> List[Dict[str, np.ndarray]]:
        """
        Encode audio data as sequence of moiré patterns.

        Converts audio to spectrogram, then encodes each frame
        as a moiré pattern for visual pattern recognition.

        Parameters
        ----------
        audio_data : np.ndarray
            Raw audio samples (mono)
        sample_rate : int
            Audio sample rate
        frame_size : int
            FFT frame size
        hop_size : int
            Hop between frames

        Returns
        -------
        List[Dict]
            List of encoded patterns, one per frame
        """
        # Compute spectrogram
        spectrogram = self._compute_spectrogram(
            audio_data, frame_size, hop_size
        )

        # Encode each frame
        encoded_frames = []
        n_frames = spectrogram.shape[1]

        for i in range(n_frames):
            frame = spectrogram[:, i]

            # Reshape frame to 2D (frequency as spatial dimension)
            frame_2d = self._reshape_1d_to_2d(frame)

            # Cycle through angle states for temporal encoding
            self.current_angle_idx = i % len(self.commensurate_angles)

            # Encode frame
            encoded = self.encode_data(frame_2d, mode="spectral")
            encoded["frame_index"] = i

            encoded_frames.append(encoded)

        return encoded_frames

    def encode_sequence(
        self,
        sequence: np.ndarray,
        use_reservoir_memory: bool = True
    ) -> List[Dict[str, np.ndarray]]:
        """
        Encode time series using reservoir memory.

        The kirigami reservoir maintains state between inputs,
        providing "fading memory" for temporal processing.

        Parameters
        ----------
        sequence : np.ndarray
            Input sequence, shape (time, features) or (time,)
        use_reservoir_memory : bool
            If True, don't reset between frames

        Returns
        -------
        List[Dict]
            Encoded patterns with temporal features
        """
        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, 1)

        encoded_sequence = []

        if not use_reservoir_memory:
            if self.architecture_mode == "quasicrystal" and self._qc is not None:
                self._qc["hierarchy"].reset()  # type: ignore[index]
                self._qc["reservoir"].reset()  # type: ignore[index]
                self._qc["cascade"].reset()  # type: ignore[index]
            else:
                self.layer1.reset()
                self.layer2.reset()

        for t, frame in enumerate(sequence):
            # Partial reset for fading memory
            if use_reservoir_memory and self.architecture_mode != "quasicrystal":
                # Apply decay to existing state
                current_state = self.layer1.get_state_field()
                decayed = current_state * 0.9  # Exponential decay

                for (i, j), cell in self.layer1.cells.items():
                    cell.set_value(decayed[j, i])

            # Reshape and encode
            frame_2d = self._reshape_1d_to_2d(frame.flatten())
            encoded = self.encode_data(frame_2d)
            encoded["time_index"] = t

            encoded_sequence.append(encoded)

        return encoded_sequence

    def set_operating_mode(
        self,
        angle_index: Optional[int] = None,
        gap_index: Optional[int] = None,
        attention_direction: Optional[Tuple[float, float]] = None
    ):
        """
        Set encoder operating mode.

        Parameters
        ----------
        angle_index : int, optional
            Index into commensurate angles (0-4)
        gap_index : int, optional
            Index into Talbot gaps
        attention_direction : Tuple[float, float], optional
            (x, y) direction for attention gradient
        """
        if angle_index is not None:
            self.current_angle_idx = angle_index % len(self.commensurate_angles)
            self.actuator1.command_rotate(
                self.commensurate_angles[self.current_angle_idx]
            )

        if gap_index is not None:
            self.current_gap_idx = gap_index % len(self.talbot_states)
            gap = self.talbot_states[self.current_gap_idx].gap
            self.actuator1.set_gap(gap)
            self.actuator2.set_gap(gap)

        if attention_direction is not None:
            ax, ay = attention_direction
            # Normalize and convert to tip/tilt
            mag = np.sqrt(ax**2 + ay**2)
            if mag > 0:
                tip = ax / mag * 5.0   # Max 5 degrees
                tilt = ay / mag * 5.0
                self.actuator1.command_tip(tip)
                self.actuator1.command_tilt(tilt)

    def _normalize_input(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range."""
        data = np.asarray(data, dtype=np.float64)
        data_min = np.min(data)
        data_max = np.max(data)

        if data_max - data_min > 0:
            return (data - data_min) / (data_max - data_min)
        return np.zeros_like(data) + 0.5

    def _reshape_1d_to_2d(self, data: np.ndarray) -> np.ndarray:
        """Reshape 1D data to square 2D array."""
        n = len(data)
        side = int(np.ceil(np.sqrt(n)))

        # Pad to square
        padded = np.zeros(side * side)
        padded[:n] = data

        return padded.reshape(side, side)

    def _resize_to_grid(self, data: np.ndarray) -> np.ndarray:
        """Resize 2D data to match grid size."""
        from scipy.ndimage import zoom

        target_y, target_x = self.config.grid_size[1], self.config.grid_size[0]
        current_y, current_x = data.shape

        if (current_y, current_x) == (target_y, target_x):
            return data

        zoom_y = target_y / current_y
        zoom_x = target_x / current_x

        return zoom(data, (zoom_y, zoom_x), order=1)

    def _compute_spectrogram(
        self,
        audio: np.ndarray,
        frame_size: int,
        hop_size: int
    ) -> np.ndarray:
        """Compute magnitude spectrogram."""
        # Simple STFT implementation
        n_frames = 1 + (len(audio) - frame_size) // hop_size
        n_freq = frame_size // 2 + 1

        spectrogram = np.zeros((n_freq, n_frames))

        window = np.hanning(frame_size)

        for i in range(n_frames):
            start = i * hop_size
            frame = audio[start:start + frame_size] * window
            spectrum = np.abs(np.fft.rfft(frame))
            spectrogram[:, i] = spectrum

        # Log scale for better dynamic range
        spectrogram = np.log1p(spectrogram)

        return spectrogram

    def _extract_features(
        self,
        pattern,
        layer1_state: np.ndarray,
        layer2_state: np.ndarray
    ) -> np.ndarray:
        """Extract feature vector from moiré pattern."""
        features = []

        # Intensity statistics
        I = pattern.intensity_field
        features.extend([
            np.mean(I),
            np.std(I),
            np.min(I),
            np.max(I)
        ])

        # Fringe contrast (from Rule Set 3)
        contrast = self.moire.compute_fringe_contrast(pattern)
        features.append(contrast)

        # Spatial frequency content
        fft_2d = np.abs(np.fft.fft2(I))
        fft_shifted = np.fft.fftshift(fft_2d)

        # Radial frequency profile
        cy, cx = fft_shifted.shape[0] // 2, fft_shifted.shape[1] // 2
        for r in [5, 10, 20]:
            y, x = np.ogrid[-cy:fft_shifted.shape[0]-cy,
                           -cx:fft_shifted.shape[1]-cx]
            mask = (x*x + y*y < r*r) & (x*x + y*y >= (r-5)**2)
            features.append(np.mean(fft_shifted[mask]) if np.any(mask) else 0)

        # Layer state statistics
        features.extend([
            np.mean(layer1_state),
            np.mean(layer2_state),
            np.corrcoef(layer1_state.flatten(), layer2_state.flatten())[0, 1]
        ])

        return np.array(features)

    def save_encoded(self, encoded: Dict, output_path: Path):
        """Save encoded pattern to file."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.save(output_path / "moire_intensity.npy", encoded["moire_intensity"])
        if encoded["moire_spectral"] is not None:
            np.save(output_path / "moire_spectral.npy", encoded["moire_spectral"])
        np.save(output_path / "features.npy", encoded["features"])

        # Save metadata
        metadata = {
            "twist_angle": float(encoded["twist_angle"]),
            "gap": float(encoded["gap"]),
            "logic_mode": encoded["logic_mode"]
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def run_demo():
    """Run demonstration of the optical encoder."""
    print("=" * 60)
    print("Optical Kirigami Moiré Encoder - Demonstration")
    print("=" * 60)

    # Initialize encoder
    config = EncoderConfig(
        grid_size=(64, 64),
        lattice_constant=1.0,
        cascade_steps=30
    )
    encoder = OpticalKirigamiEncoder(config)

    print("\n1. Encoding random image data...")
    test_image = np.random.rand(32, 32)
    result = encoder.encode_data(test_image)

    print(f"   - Moiré period: {encoder.moire.compute_moire_period(result['twist_angle']):.2f} μm")
    print(f"   - Twist angle: {result['twist_angle']:.2f}°")
    print(f"   - Talbot gap: {result['gap']:.2f} μm")
    print(f"   - Logic mode: {result['logic_mode']}")
    print(f"   - Feature vector size: {len(result['features'])}")

    print("\n2. Testing different operating modes...")
    for i, angle in enumerate(encoder.commensurate_angles):
        encoder.set_operating_mode(angle_index=i)
        result = encoder.encode_data(test_image)
        print(f"   Mode {i}: angle={angle:.2f}°, contrast={encoder.moire.compute_fringe_contrast(encoder.moire.generate_moire_field(angle)):.3f}")

    print("\n3. Testing sequence encoding with reservoir memory...")
    test_sequence = np.sin(np.linspace(0, 4*np.pi, 10).reshape(-1, 1))
    encoded_seq = encoder.encode_sequence(test_sequence, use_reservoir_memory=True)
    print(f"   Encoded {len(encoded_seq)} frames")
    print(f"   Feature evolution std: {np.std([e['features'][0] for e in encoded_seq]):.4f}")

    print("\n4. Physics validation...")
    # Validate Rule Set 1: Commensurate angles
    print("   Rule Set 1 - Commensurate angles:")
    for m, n in [(2, 1), (3, 1), (4, 1)]:
        angle = encoder.lattice.compute_commensurate_angle(m, n)
        period = encoder.lattice.get_superlattice_period(m, n)
        print(f"     (m={m}, n={n}): θ={angle:.2f}°, Σ={period}")

    # Validate Rule Set 3: Talbot gaps
    print("   Rule Set 3 - Talbot gaps:")
    for state in encoder.talbot_states[:4]:
        print(f"     Gap={state.gap:.2f}μm, Mode={state.mode.value}, Logic={state.logic_polarity}")

    print("\n" + "=" * 60)
    print("Demo complete. System ready for data encoding.")
    print("=" * 60)

    return encoder


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optical Kirigami Moiré Encoder"
    )
    parser.add_argument(
        "--mode", type=str, default="demo",
        choices=["demo", "encode", "batch"],
        help="Operating mode"
    )
    parser.add_argument(
        "--input", type=str,
        help="Input file path (audio/image)"
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory"
    )
    parser.add_argument(
        "--grid-size", type=int, default=64,
        help="Grid resolution"
    )
    parser.add_argument(
        "--architecture-mode", type=str, default="legacy",
        choices=["legacy", "quasicrystal"],
        help="Computation architecture mode"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()

    elif args.mode == "encode":
        if not args.input:
            print("Error: --input required for encode mode")
            return

        config = EncoderConfig(
            grid_size=(args.grid_size, args.grid_size),
            architecture_mode=args.architecture_mode,
        )
        encoder = OpticalKirigamiEncoder(config)

        input_path = Path(args.input)

        if input_path.suffix.lower() in [".wav", ".mp3", ".flac"]:
            # Audio encoding
            try:
                import soundfile as sf
                audio, sr = sf.read(str(input_path))
                if audio.ndim > 1:
                    audio = audio[:, 0]  # Mono
                encoded = encoder.encode_audio(audio, sample_rate=sr)
                print(f"Encoded {len(encoded)} audio frames")
            except ImportError:
                print("Install soundfile for audio: pip install soundfile")

        elif input_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            # Image encoding
            try:
                from PIL import Image
                img = Image.open(input_path).convert("L")
                img_array = np.array(img) / 255.0
                encoded = encoder.encode_data(img_array)
                encoder.save_encoded(encoded, Path(args.output))
                print(f"Encoded image saved to {args.output}")
            except ImportError:
                print("Install Pillow for images: pip install Pillow")

        else:
            # Try as numpy array
            try:
                data = np.load(str(input_path))
                encoded = encoder.encode_data(data)
                encoder.save_encoded(encoded, Path(args.output))
            except Exception as e:
                print(f"Could not load input: {e}")


if __name__ == "__main__":
    main()
