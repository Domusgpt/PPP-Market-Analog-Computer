"""
Local Phase Quantization (LPQ) Decoder for H4 Constellation.

LPQ is specifically designed for analyzing moiré interference patterns
because it extracts information from the phase of the Fourier spectrum,
which is exactly what moiré patterns encode.

Key advantages over traditional edge detection:
1. Blur-invariant (moiré fringes are inherently "soft" boundaries)
2. Phase-based (captures the interference phenomenon directly)
3. Quantized output (can be compared to target codes)

This module implements:
- Short-Term Fourier Transform (STFT) on local windows
- Phase extraction from low-frequency coefficients
- Quantization to descriptor codes
- Error signal computation for servo feedback
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
import numpy as np
from enum import Enum


class PhaseMode(Enum):
    """Phase extraction modes."""
    LOW_FREQUENCY = "low_freq"      # Only use low-frequency components
    MID_FREQUENCY = "mid_freq"      # Mid-frequency for finer detail
    FULL_SPECTRUM = "full"          # Use all frequency components


class QuantizationMethod(Enum):
    """Methods for phase quantization."""
    BINARY = "binary"              # 0 or 1 per coefficient (256 codes)
    TERNARY = "ternary"            # -1, 0, 1 per coefficient
    MULTI_LEVEL = "multi_level"    # Multiple quantization levels


@dataclass
class LPQConfig:
    """Configuration for LPQ analysis."""
    window_size: int = 7           # Local analysis window size
    num_neighbors: int = 8          # Number of spatial neighbors
    frequency_type: PhaseMode = PhaseMode.LOW_FREQUENCY
    quantization: QuantizationMethod = QuantizationMethod.BINARY
    decorrelation: bool = True      # Apply decorrelation
    blur_radius: float = 0.0        # Expected blur radius for robustness


@dataclass
class LPQDescriptor:
    """An LPQ descriptor for a local region."""
    code: int                      # Quantized descriptor code
    phase_values: np.ndarray       # Raw phase values
    magnitude: float               # Magnitude of response
    position: Tuple[int, int]      # Center position in image


class LPQDecoder:
    """
    Local Phase Quantization decoder for moiré pattern analysis.

    The decoder performs:
    1. STFT on local windows to extract frequency content
    2. Phase extraction from selected frequency components
    3. Decorrelation to reduce redundancy
    4. Quantization to produce discrete codes

    These codes can be compared to target codes to compute
    error signals for the quaternion control loop.
    """

    def __init__(self, config: Optional[LPQConfig] = None):
        """
        Initialize the LPQ decoder.

        Args:
            config: LPQ configuration (uses defaults if None)
        """
        self.config = config or LPQConfig()

        # Precompute frequency basis for STFT
        self._init_frequency_basis()

        # Decorrelation matrix (computed on first use)
        self._decorrelation_matrix = None

    def _init_frequency_basis(self):
        """Initialize the frequency basis functions for STFT."""
        w = self.config.window_size
        n = self.config.num_neighbors

        # Create 2D frequency sampling points
        # Using the four corner frequencies plus center frequencies
        self.freq_points = []

        if self.config.frequency_type == PhaseMode.LOW_FREQUENCY:
            # Low frequency: use smallest non-zero frequencies
            freqs = [1, 2]
        elif self.config.frequency_type == PhaseMode.MID_FREQUENCY:
            freqs = [2, 3, 4]
        else:
            freqs = list(range(1, w // 2 + 1))

        for fx in freqs[:2]:
            for fy in freqs[:2]:
                self.freq_points.append((fx, fy))

        # Compute basis functions (complex exponentials)
        x = np.arange(w) - w // 2
        y = np.arange(w) - w // 2
        xx, yy = np.meshgrid(x, y)

        self.basis_functions = []
        for fx, fy in self.freq_points:
            # Complex exponential basis
            basis = np.exp(2j * np.pi * (fx * xx + fy * yy) / w)
            self.basis_functions.append(basis)

        self.basis_functions = np.array(self.basis_functions)

    def _compute_stft(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute STFT coefficients for a local patch.

        Args:
            patch: Local image patch (window_size x window_size)

        Returns:
            Complex STFT coefficients
        """
        # Ensure patch is float
        patch = patch.astype(np.float64)

        # Apply window function (Gaussian)
        w = self.config.window_size
        sigma = w / 4
        x = np.arange(w) - w // 2
        window_1d = np.exp(-x**2 / (2 * sigma**2))
        window = np.outer(window_1d, window_1d)

        windowed_patch = patch * window

        # Compute coefficients using precomputed basis
        coefficients = np.zeros(len(self.basis_functions), dtype=np.complex128)

        for i, basis in enumerate(self.basis_functions):
            coefficients[i] = np.sum(windowed_patch * np.conj(basis))

        return coefficients

    def _extract_phase(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Extract phase from complex coefficients.

        Args:
            coefficients: Complex STFT coefficients

        Returns:
            Phase values in [-π, π]
        """
        return np.angle(coefficients)

    def _compute_decorrelation_matrix(self, sample_data: np.ndarray):
        """
        Compute decorrelation matrix from sample data.

        Uses whitening to decorrelate the phase features.

        Args:
            sample_data: Sample phase vectors (N x D)
        """
        # Compute covariance matrix
        cov = np.cov(sample_data.T)

        # Eigendecomposition for whitening
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Avoid division by zero
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # Whitening matrix: W = Λ^(-1/2) @ V^T
        self._decorrelation_matrix = np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    def _decorrelate(self, phase_values: np.ndarray) -> np.ndarray:
        """
        Apply decorrelation to phase values.

        Args:
            phase_values: Raw phase vector

        Returns:
            Decorrelated phase vector
        """
        if self._decorrelation_matrix is None:
            return phase_values

        return self._decorrelation_matrix @ phase_values

    def _quantize(self, phase_values: np.ndarray) -> int:
        """
        Quantize phase values to a descriptor code.

        Args:
            phase_values: Decorrelated phase values

        Returns:
            Integer code (0 to 255 for binary quantization)
        """
        if self.config.quantization == QuantizationMethod.BINARY:
            # Binary: each phase value → 0 or 1 based on sign
            bits = (phase_values >= 0).astype(int)
            # Convert to integer code
            code = 0
            for i, bit in enumerate(bits[:8]):  # Use up to 8 bits
                code |= (bit << i)
            return code

        elif self.config.quantization == QuantizationMethod.TERNARY:
            # Ternary: -1, 0, 1 based on thresholds
            threshold = np.pi / 4
            quantized = np.zeros_like(phase_values)
            quantized[phase_values > threshold] = 1
            quantized[phase_values < -threshold] = -1
            # Convert to base-3 code
            code = 0
            for i, val in enumerate(quantized[:8]):
                code += int(val + 1) * (3 ** i)
            return code

        else:
            # Multi-level: more quantization levels
            levels = 4
            quantized = np.floor((phase_values + np.pi) / (2 * np.pi / levels))
            quantized = np.clip(quantized, 0, levels - 1).astype(int)
            code = 0
            for i, val in enumerate(quantized[:8]):
                code += val * (levels ** i)
            return code

    def compute_descriptor(self, patch: np.ndarray) -> LPQDescriptor:
        """
        Compute LPQ descriptor for a local patch.

        Args:
            patch: Image patch (window_size x window_size)

        Returns:
            LPQDescriptor containing code and metadata
        """
        # Compute STFT
        coefficients = self._compute_stft(patch)

        # Extract phase
        phase_values = self._extract_phase(coefficients)

        # Compute magnitude (for weighting)
        magnitude = np.mean(np.abs(coefficients))

        # Decorrelate if enabled
        if self.config.decorrelation:
            processed_phase = self._decorrelate(phase_values)
        else:
            processed_phase = phase_values

        # Quantize to code
        code = self._quantize(processed_phase)

        return LPQDescriptor(
            code=code,
            phase_values=phase_values,
            magnitude=magnitude,
            position=(0, 0)
        )

    def compute_image_descriptors(self,
                                   image: np.ndarray,
                                   step: int = 1) -> List[LPQDescriptor]:
        """
        Compute LPQ descriptors for an entire image.

        Args:
            image: Input image (2D array)
            step: Step size between descriptor centers

        Returns:
            List of LPQDescriptors covering the image
        """
        h, w = image.shape
        half_win = self.config.window_size // 2

        descriptors = []

        for i in range(half_win, h - half_win, step):
            for j in range(half_win, w - half_win, step):
                # Extract patch
                patch = image[
                    i - half_win:i + half_win + 1,
                    j - half_win:j + half_win + 1
                ]

                if patch.shape[0] == self.config.window_size and \
                   patch.shape[1] == self.config.window_size:
                    desc = self.compute_descriptor(patch)
                    desc.position = (i, j)
                    descriptors.append(desc)

        return descriptors

    def compute_histogram(self,
                          descriptors: List[LPQDescriptor]) -> np.ndarray:
        """
        Compute histogram of LPQ codes.

        Args:
            descriptors: List of LPQ descriptors

        Returns:
            Histogram array (256 bins for binary quantization)
        """
        if self.config.quantization == QuantizationMethod.BINARY:
            num_bins = 256
        elif self.config.quantization == QuantizationMethod.TERNARY:
            num_bins = 3 ** 8
        else:
            num_bins = 4 ** 8

        histogram = np.zeros(min(num_bins, 256))

        for desc in descriptors:
            bin_idx = desc.code % len(histogram)
            histogram[bin_idx] += desc.magnitude

        # Normalize
        total = np.sum(histogram)
        if total > 0:
            histogram /= total

        return histogram


class MoireLPQAnalyzer:
    """
    Specialized LPQ analyzer for moiré interference patterns.

    This analyzer is tuned for the specific characteristics of
    moiré fringes from the Cyan/Magenta kirigami stack:
    - Low spatial frequencies (large-scale fringes)
    - Gradual intensity transitions (blur-like)
    - Color-separated channels
    """

    def __init__(self):
        """Initialize the moiré analyzer."""
        # Configure LPQ for moiré patterns
        self.config = LPQConfig(
            window_size=15,  # Larger window for moiré scale
            num_neighbors=8,
            frequency_type=PhaseMode.LOW_FREQUENCY,
            quantization=QuantizationMethod.BINARY,
            decorrelation=True
        )

        self.decoder = LPQDecoder(self.config)

        # Target codes for different quaternion states
        self.target_codes: Dict[str, np.ndarray] = {}

    def analyze_moire(self, moire_pattern: np.ndarray) -> Dict:
        """
        Analyze a moiré pattern and extract phase information.

        Args:
            moire_pattern: 2D moiré intensity array

        Returns:
            Dictionary with analysis results
        """
        # Normalize pattern
        pattern = moire_pattern.astype(np.float64)
        if pattern.max() > pattern.min():
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())

        # Compute descriptors
        descriptors = self.decoder.compute_image_descriptors(pattern, step=4)

        # Compute histogram
        histogram = self.decoder.compute_histogram(descriptors)

        # Extract global phase information
        phases = np.array([d.phase_values for d in descriptors])
        mean_phase = np.mean(phases, axis=0)
        phase_variance = np.var(phases, axis=0)

        # Compute "phason strain" metric
        # High variance indicates strain/distortion in the pattern
        phason_strain = np.mean(phase_variance)

        # Dominant code (mode of histogram)
        dominant_code = np.argmax(histogram)

        return {
            "histogram": histogram,
            "mean_phase": mean_phase,
            "phase_variance": phase_variance,
            "phason_strain": phason_strain,
            "dominant_code": dominant_code,
            "num_descriptors": len(descriptors)
        }

    def analyze_spectral_moire(self, rgb_moire: np.ndarray) -> Dict:
        """
        Analyze RGB spectral moiré pattern.

        Args:
            rgb_moire: HxWx3 RGB moiré pattern

        Returns:
            Per-channel and combined analysis
        """
        results = {}

        # Analyze each channel
        for i, channel_name in enumerate(["red", "green", "blue"]):
            channel_result = self.analyze_moire(rgb_moire[:, :, i])
            results[channel_name] = channel_result

        # Combined analysis (luminance)
        luminance = 0.299 * rgb_moire[:, :, 0] + \
                   0.587 * rgb_moire[:, :, 1] + \
                   0.114 * rgb_moire[:, :, 2]
        results["luminance"] = self.analyze_moire(luminance)

        # Cross-channel phase coherence
        r_phase = results["red"]["mean_phase"]
        g_phase = results["green"]["mean_phase"]
        b_phase = results["blue"]["mean_phase"]

        phase_coherence = np.mean([
            np.abs(np.cos(r_phase - g_phase)),
            np.abs(np.cos(g_phase - b_phase)),
            np.abs(np.cos(b_phase - r_phase))
        ])
        results["phase_coherence"] = phase_coherence

        return results

    def register_target_code(self, name: str, quaternion_state: np.ndarray):
        """
        Register a target code for a specific quaternion state.

        Args:
            name: Name/identifier for the state
            quaternion_state: The quaternion [w, x, y, z]
        """
        # Generate synthetic pattern for this quaternion
        # This would normally come from calibration
        self.target_codes[name] = quaternion_state

    def compute_error_signal(self,
                              current_analysis: Dict,
                              target_code: int) -> float:
        """
        Compute error signal between current pattern and target.

        Args:
            current_analysis: Result from analyze_moire()
            target_code: Target LPQ code

        Returns:
            Error signal in [0, 1] range
        """
        current_code = current_analysis["dominant_code"]

        # Hamming distance between codes
        xor = current_code ^ target_code
        hamming_distance = bin(xor).count('1')

        # Normalize by number of bits
        max_bits = 8
        error = hamming_distance / max_bits

        return error

    def compute_feedback_vector(self,
                                 current_analysis: Dict,
                                 target_histogram: np.ndarray) -> np.ndarray:
        """
        Compute feedback vector for PID control.

        Args:
            current_analysis: Current moiré analysis
            target_histogram: Target LPQ histogram

        Returns:
            Feedback vector for layer adjustment
        """
        current_histogram = current_analysis["histogram"]

        # Histogram difference (signed)
        diff = target_histogram - current_histogram

        # Map histogram difference to 6-layer feedback
        # using the trilatic structure
        feedback = np.zeros(6)

        # Pair A feedback (from low bins)
        feedback[0] = np.mean(diff[:85])   # Cyan
        feedback[1] = np.mean(diff[:85])   # Magenta

        # Pair B feedback (from mid bins)
        feedback[2] = np.mean(diff[85:170])   # Cyan
        feedback[3] = np.mean(diff[85:170])   # Magenta

        # Pair C feedback (from high bins)
        feedback[4] = np.mean(diff[170:])   # Cyan
        feedback[5] = np.mean(diff[170:])   # Magenta

        return feedback


class PhasonStrainDetector:
    """
    Detector for phason strain in quasicrystalline moiré patterns.

    In the H4 projection context, phason strain represents
    deviations from the ideal 4D→2D projection, which appear
    as characteristic "worm-like" defects in the moiré pattern.
    """

    def __init__(self, threshold: float = 0.1):
        """
        Initialize the phason detector.

        Args:
            threshold: Strain detection threshold
        """
        self.threshold = threshold
        self.lpq_analyzer = MoireLPQAnalyzer()

    def detect_phason_worms(self, moire_pattern: np.ndarray) -> Dict:
        """
        Detect phason worm defects in the pattern.

        Args:
            moire_pattern: 2D moiré intensity array

        Returns:
            Detection results with worm locations and strengths
        """
        analysis = self.lpq_analyzer.analyze_moire(moire_pattern)

        # Phason worms appear as regions of high phase variance
        # with characteristic linear structure

        # Compute local phase variance
        h, w = moire_pattern.shape
        window = 15
        variance_map = np.zeros((h, w))

        descriptors = self.lpq_analyzer.decoder.compute_image_descriptors(
            moire_pattern.astype(np.float64), step=2
        )

        for desc in descriptors:
            i, j = desc.position
            variance = np.var(desc.phase_values)
            variance_map[i, j] = variance

        # Threshold to find high-strain regions
        strain_mask = variance_map > self.threshold

        # Find connected components (worm segments)
        worm_count = self._count_connected_components(strain_mask)

        return {
            "strain_map": variance_map,
            "worm_mask": strain_mask,
            "worm_count": worm_count,
            "mean_strain": analysis["phason_strain"],
            "max_strain": np.max(variance_map)
        }

    def _count_connected_components(self, mask: np.ndarray) -> int:
        """Count connected components in a binary mask."""
        # Simple connected component counting
        from scipy import ndimage
        try:
            labeled, num_features = ndimage.label(mask)
            return num_features
        except ImportError:
            # Fallback: count high-strain pixels
            return int(np.sum(mask))

    def compute_ideal_pattern_code(self,
                                     deployment_state: float) -> int:
        """
        Compute the expected LPQ code for an ideal pattern.

        For a perfect 24-cell projection, the moiré should have
        specific symmetry properties that produce a characteristic code.

        Args:
            deployment_state: Current deployment (0, 0.5, or 1)

        Returns:
            Expected LPQ code
        """
        # Ideal codes based on deployment state
        # These would be calibrated experimentally
        ideal_codes = {
            0.0: 0b10101010,    # Locked state: regular pattern
            0.5: 0b11001100,    # Auxetic: transitional pattern
            1.0: 0b11110000,    # Deployed: 600-cell projection
        }

        # Interpolate for continuous states
        if deployment_state <= 0.25:
            return ideal_codes[0.0]
        elif deployment_state <= 0.75:
            return ideal_codes[0.5]
        else:
            return ideal_codes[1.0]
