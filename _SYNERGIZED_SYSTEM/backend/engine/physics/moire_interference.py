"""
Moiré Interference Engine
=========================

Implements the moiré pattern generation and bichromatic interference
calculations as described in Sections 2.1-2.2 of the specification.

Key equation: L_M = a / (2 * sin(theta/2))

This module handles:
- Moiré period calculation
- Bichromatic (opposite spectrum) interference
- Spectral logic gates (AND, XOR, NAND)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Callable
from enum import Enum


class SpectralBand(Enum):
    """
    Spectral bands for bichromatic moiré logic.

    "Opposite spectrum" refers to complementary colors:
    - Red vs Cyan (complementary)
    - Blue vs Yellow (complementary)
    """
    RED = "red"        # ~620-750 nm, Layer 2 (Dot Array)
    CYAN = "cyan"      # ~490-520 nm, Layer 1 (Hole Array)
    BLUE = "blue"
    YELLOW = "yellow"
    GREEN = "green"


class LogicGate(Enum):
    """
    Moiré logic gate types determined by Talbot gap.

    From Section 5.3:
    - Integer Talbot: POSITIVE (AND/OR)
    - Half-Integer Talbot: NEGATIVE (NAND/XOR)
    """
    AND = "and"
    OR = "or"
    XOR = "xor"
    NAND = "nand"


@dataclass
class MoirePattern:
    """Represents a computed moiré interference pattern."""
    period: float                    # L_M in micrometers
    intensity_field: np.ndarray      # 2D intensity distribution
    phase_field: np.ndarray          # 2D phase distribution
    spectral_field: Optional[np.ndarray] = None  # RGB chromatic output


class MoireInterference:
    """
    Moiré interference calculator for trilatic optical computing.

    This class computes moiré patterns from the superposition of two
    hexagonal lattices with configurable twist, displacement, and
    spectral properties.

    Parameters
    ----------
    lattice_constant : float
        Base lattice constant 'a' in micrometers
    wavelength_red : float
        Red channel wavelength in nm (default: 650 nm)
    wavelength_cyan : float
        Cyan channel wavelength in nm (default: 500 nm)
    """

    def __init__(
        self,
        lattice_constant: float = 1.0,
        wavelength_red: float = 650.0,
        wavelength_cyan: float = 500.0
    ):
        self.a = lattice_constant
        self.lambda_red = wavelength_red
        self.lambda_cyan = wavelength_cyan

    def compute_moire_period(self, twist_angle: float) -> float:
        """
        Compute moiré superlattice period.

        L_M = a / (2 * sin(theta / 2))

        From Section 2.1: For small angles, the moiré period becomes
        inversely proportional to the twist, allowing microscopic
        mechanical actuations to result in macroscopic optical shifts.

        Parameters
        ----------
        twist_angle : float
            Rotation angle theta in degrees

        Returns
        -------
        float
            Moiré period L_M in micrometers
        """
        if twist_angle == 0:
            return float('inf')  # No moiré pattern when aligned

        theta_rad = np.radians(twist_angle)
        L_M = self.a / (2 * np.sin(theta_rad / 2))

        return abs(L_M)

    def compute_amplification_factor(self, twist_angle: float) -> float:
        """
        Calculate the geometric amplification from twist.

        The amplification factor A = L_M / a shows how microscopic
        lattice changes translate to macroscopic moiré shifts.

        Parameters
        ----------
        twist_angle : float
            Twist angle in degrees

        Returns
        -------
        float
            Amplification factor (dimensionless)
        """
        L_M = self.compute_moire_period(twist_angle)
        if L_M == float('inf'):
            return float('inf')
        return L_M / self.a

    def generate_moire_field(
        self,
        twist_angle: float,
        grid_size: Tuple[int, int] = (256, 256),
        field_size: Tuple[float, float] = (50.0, 50.0),
        displacement: Tuple[float, float] = (0.0, 0.0)
    ) -> MoirePattern:
        """
        Generate 2D moiré interference pattern.

        Superimposes two hexagonal gratings with the specified twist
        and computes the resulting intensity modulation.

        Parameters
        ----------
        twist_angle : float
            Rotation angle between layers in degrees
        grid_size : Tuple[int, int]
            Output resolution (nx, ny)
        field_size : Tuple[float, float]
            Physical field size in micrometers (Lx, Ly)
        displacement : Tuple[float, float]
            Relative displacement (dx, dy) in micrometers

        Returns
        -------
        MoirePattern
            Pattern with intensity and phase fields
        """
        nx, ny = grid_size
        Lx, Ly = field_size
        dx, dy = displacement

        # Create coordinate grids
        x = np.linspace(-Lx/2, Lx/2, nx)
        y = np.linspace(-Ly/2, Ly/2, ny)
        X, Y = np.meshgrid(x, y)

        # Layer 1: Reference hexagonal grating (0 degrees)
        layer1 = self._hexagonal_grating(X, Y, 0.0)

        # Layer 2: Rotated and displaced grating
        theta_rad = np.radians(twist_angle)
        X2 = X * np.cos(theta_rad) + Y * np.sin(theta_rad) - dx
        Y2 = -X * np.sin(theta_rad) + Y * np.cos(theta_rad) - dy
        layer2 = self._hexagonal_grating(X2, Y2, 0.0)

        # Moiré interference: product of transmission functions
        intensity = layer1 * layer2

        # Phase from complex field
        complex_field = layer1 * np.exp(1j * layer2 * 2 * np.pi)
        phase = np.angle(complex_field)

        period = self.compute_moire_period(twist_angle)

        return MoirePattern(
            period=period,
            intensity_field=intensity,
            phase_field=phase
        )

    def _hexagonal_grating(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rotation: float = 0.0
    ) -> np.ndarray:
        """
        Generate hexagonal (trilatic) transmission grating.

        Uses three wave vectors at 120-degree separation:
        T(x,y) = (1/3) * sum_i cos(G_i · r)

        Parameters
        ----------
        X, Y : np.ndarray
            Coordinate meshgrids
        rotation : float
            Additional rotation in degrees

        Returns
        -------
        np.ndarray
            Transmission function (0 to 1)
        """
        k = 2 * np.pi / self.a  # Wave number
        theta = np.radians(rotation)

        # Three wave vectors at 0, 120, 240 degrees (trilatic)
        angles = [0, 120, 240]
        transmission = np.zeros_like(X)

        for angle in angles:
            phi = np.radians(angle) + theta
            Gx = k * np.cos(phi)
            Gy = k * np.sin(phi)
            transmission += np.cos(Gx * X + Gy * Y)

        # Normalize to [0, 1]
        transmission = (transmission / 3 + 1) / 2

        return transmission

    def compute_bichromatic_moire(
        self,
        twist_angle: float,
        grid_size: Tuple[int, int] = (256, 256),
        field_size: Tuple[float, float] = (50.0, 50.0),
        layer1_state: np.ndarray = None,
        layer2_state: np.ndarray = None
    ) -> MoirePattern:
        """
        Compute bichromatic (opposite spectrum) moiré pattern.

        From Section 2.2: The system creates a spatial map of spectral
        correlation. Where lattices align in phase, transmission is
        superposition. Where out of phase, structure becomes opaque.

        Parameters
        ----------
        twist_angle : float
            Rotation angle in degrees
        grid_size : Tuple[int, int]
            Output resolution
        field_size : Tuple[float, float]
            Physical field size in micrometers
        layer1_state : np.ndarray, optional
            State field for layer 1 (Cyan/Hole array)
        layer2_state : np.ndarray, optional
            State field for layer 2 (Red/Dot array)

        Returns
        -------
        MoirePattern
            Pattern with spectral (RGB) output field
        """
        base_pattern = self.generate_moire_field(
            twist_angle, grid_size, field_size
        )

        nx, ny = grid_size

        # Default: uniform state 1 if not provided
        if layer1_state is None:
            layer1_state = np.ones((ny, nx))
        if layer2_state is None:
            layer2_state = np.ones((ny, nx))

        # Cyan channel: Hole array (Layer 1) - transmissive
        # Wavelength-dependent transmission modulation
        cyan_response = base_pattern.intensity_field * layer1_state

        # Red channel: Dot array (Layer 2) - reflective/absorptive
        # Inverse response for complementary logic
        theta_rad = np.radians(twist_angle)
        x = np.linspace(-field_size[0]/2, field_size[0]/2, nx)
        y = np.linspace(-field_size[1]/2, field_size[1]/2, ny)
        X, Y = np.meshgrid(x, y)
        X2 = X * np.cos(theta_rad) + Y * np.sin(theta_rad)
        Y2 = -X * np.sin(theta_rad) + Y * np.cos(theta_rad)
        layer2_pattern = self._hexagonal_grating(X2, Y2, 0.0)
        red_response = layer2_pattern * layer2_state

        # Build RGB spectral field
        # R: Red channel from dot array
        # G: Mix of both (where both transmit)
        # B: Part of cyan channel
        spectral = np.zeros((ny, nx, 3))
        spectral[:, :, 0] = red_response * 0.8  # Red
        spectral[:, :, 1] = (cyan_response + red_response) / 2 * 0.5  # Green
        spectral[:, :, 2] = cyan_response * 0.9  # Blue (cyan component)

        # Normalize spectral
        spectral = np.clip(spectral, 0, 1)

        # Compute modulated intensity using additive interference (superposition)
        # This properly encodes the input information into the moiré pattern
        modulated_intensity = (cyan_response + red_response) / 2.0

        # Normalize to use full dynamic range while preserving spatial structure
        i_min, i_max = modulated_intensity.min(), modulated_intensity.max()
        if i_max > i_min:
            modulated_intensity = (modulated_intensity - i_min) / (i_max - i_min)

        return MoirePattern(
            period=base_pattern.period,
            intensity_field=modulated_intensity,
            phase_field=base_pattern.phase_field,
            spectral_field=spectral
        )

    def evaluate_logic_gate(
        self,
        pattern: MoirePattern,
        gate_type: LogicGate,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Evaluate moiré pattern as a spatial logic gate.

        From Section 2.2: Moiré output performs XOR, AND, or NAND
        operations simultaneously across the visual field.

        Parameters
        ----------
        pattern : MoirePattern
            Input moiré pattern
        gate_type : LogicGate
            Type of logic operation
        threshold : float
            Binary threshold for intensity

        Returns
        -------
        np.ndarray
            Binary logic output
        """
        I = pattern.intensity_field

        if gate_type == LogicGate.AND:
            # High output only where both layers transmit
            return (I > threshold).astype(float)

        elif gate_type == LogicGate.OR:
            # High output where either layer has signal
            return (I > threshold * 0.3).astype(float)

        elif gate_type == LogicGate.XOR:
            # High output at interference fringes (edges)
            mid_low = threshold * 0.3
            mid_high = threshold * 0.7
            return ((I > mid_low) & (I < mid_high)).astype(float)

        elif gate_type == LogicGate.NAND:
            # Inverse of AND
            return (I <= threshold).astype(float)

        return I

    def compute_fringe_contrast(self, pattern: MoirePattern) -> float:
        """
        Calculate Michelson contrast of moiré fringes.

        C = (I_max - I_min) / (I_max + I_min)

        High contrast is required for clear logic states.
        Rule Set 3 ensures this via Talbot resonance.

        Parameters
        ----------
        pattern : MoirePattern
            Moiré pattern to analyze

        Returns
        -------
        float
            Contrast value (0 to 1)
        """
        I = pattern.intensity_field
        I_max = np.max(I)
        I_min = np.min(I)

        if I_max + I_min == 0:
            return 0.0

        return (I_max - I_min) / (I_max + I_min)
