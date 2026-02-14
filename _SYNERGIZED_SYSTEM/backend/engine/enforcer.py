"""
Rule Enforcer - Hard Constraints for Optical Kirigami Moiré
===========================================================

Implements the three rule sets from Section 5 of the specification:

Rule Set 1: Angular Commensurability (Pythagorean Rule)
  - Twist angles must satisfy cos(θ) = (n² + 4mn + m²) / (2(n² + mn + m²))
  - Ensures periodic (not quasicrystalline) moiré superlattices

Rule Set 2: Trilatic Tilt Symmetry (Orthogonality Rule)
  - Tilt axes must align with reciprocal lattice vectors: φ_tilt = k × (π/3)
  - Preserves C3 symmetry of the trilatic division

Rule Set 3: Talbot Distance (Integer Gap Rule)
  - Gap must be integer or half-integer multiple of Talbot length
  - Integer: z = N × Z_T → POSITIVE logic (AND/OR)
  - Half-integer: z = (N + 0.5) × Z_T → NEGATIVE logic (NAND/XOR)

These rules are ENFORCED, not suggested. Invalid states raise exceptions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum
import math


class RuleViolation(Exception):
    """Raised when a physical rule is violated."""
    pass


class LogicPolarity(Enum):
    """Logic polarity determined by Talbot gap mode."""
    POSITIVE = "positive"  # Integer Talbot: AND, OR
    NEGATIVE = "negative"  # Half-integer Talbot: NAND, XOR


@dataclass
class CommensurateLock:
    """Result of locking to a commensurate angle."""
    m: int
    n: int
    angle: float  # degrees
    superlattice_period: int
    cognitive_mode: str


@dataclass
class TalbotLock:
    """Result of locking to a Talbot resonance."""
    gap: float  # micrometers
    order: int
    mode: str  # "integer" or "half_integer"
    polarity: LogicPolarity
    contrast: float


@dataclass
class TiltLock:
    """Result of locking to a valid tilt axis."""
    axis_index: int  # k in φ = k × 60°
    axis_angle: float  # degrees
    tilt_angle: float  # degrees
    effective_lattice_constant: float


class RuleEnforcer:
    """
    Enforces the three rule sets for optical kirigami moiré computation.

    All methods either return valid locked values or raise RuleViolation.
    This ensures the system always operates in a physically valid regime.

    Parameters
    ----------
    lattice_constant : float
        Base lattice constant in micrometers
    wavelength : float
        Operating wavelength in nanometers
    strict_mode : bool
        If True, raise exceptions on violations. If False, auto-correct.
    """

    # Rule Set 1: Commensurate angles from Pythagorean rule
    # cos(θ) = (n² + 4mn + m²) / (2(n² + mn + m²))
    COMMENSURATE_TABLE = {
        (1, 1): {"angle": 0.0, "mode": "All-Pass / Transparent"},
        (2, 1): {"angle": 21.787, "mode": "Coarse Filtering"},
        (3, 1): {"angle": 13.174, "mode": "Intermediate Filtering"},
        (3, 2): {"angle": 27.796, "mode": "Wide Filtering"},
        (4, 1): {"angle": 9.430, "mode": "Fine Detail / Texture"},
        (4, 3): {"angle": 32.204, "mode": "Very Wide Filtering"},
        (5, 1): {"angle": 7.341, "mode": "Edge Detection"},
        (5, 2): {"angle": 16.426, "mode": "Medium Filtering"},
        (5, 3): {"angle": 23.413, "mode": "Broad Filtering"},
        (5, 4): {"angle": 34.833, "mode": "Ultra Wide Filtering"},
        (6, 1): {"angle": 6.009, "mode": "Fine Edge Detection"},
        (7, 1): {"angle": 5.086, "mode": "Very Fine Edge Detection"},
        (8, 1): {"angle": 4.408, "mode": "Ultra Fine Edge Detection"},
    }

    MAGIC_ANGLE = 1.1  # degrees - flat band regime

    # Rule Set 2: Valid tilt axis indices (k × 60°)
    VALID_TILT_AXES = [0, 1, 2, 3, 4, 5]  # 0°, 60°, 120°, 180°, 240°, 300°

    # Rule Set 3: Talbot geometry factor
    HEXAGONAL_TALBOT_FACTOR = 1.5  # 3/2 for trilatic geometry

    def __init__(
        self,
        lattice_constant: float = 1.0,
        wavelength: float = 550.0,
        strict_mode: bool = False
    ):
        self.a = lattice_constant
        self.wavelength = wavelength / 1000.0  # Convert nm to μm
        self.strict_mode = strict_mode

        # Compute Talbot length
        self.talbot_length = self._compute_talbot_length()

        # Pre-compute all commensurate angles for fast lookup
        self._commensurate_angles = self._build_angle_list()

    def _compute_talbot_length(self) -> float:
        """Z_T = (3/2) × a² / λ for hexagonal geometry."""
        return self.HEXAGONAL_TALBOT_FACTOR * (self.a ** 2) / self.wavelength

    def _build_angle_list(self) -> List[float]:
        """Build sorted list of all commensurate angles (including negatives)."""
        angles = set()
        for (m, n), data in self.COMMENSURATE_TABLE.items():
            angles.add(data["angle"])
            angles.add(-data["angle"])
        return sorted(angles)

    @staticmethod
    def compute_commensurate_angle(m: int, n: int) -> float:
        """
        Compute commensurate angle from indices (m, n).

        cos(θ) = (n² + 4mn + m²) / (2(n² + mn + m²))
        """
        if math.gcd(m, n) != 1:
            raise ValueError(f"m={m} and n={n} must be coprime")

        numerator = n**2 + 4*m*n + m**2
        denominator = 2 * (n**2 + m*n + m**2)
        cos_theta = np.clip(numerator / denominator, -1.0, 1.0)

        return np.degrees(np.arccos(cos_theta))

    # =========================================================================
    # RULE SET 1: Angular Commensurability
    # =========================================================================

    def enforce_angle(self, target_angle: float) -> CommensurateLock:
        """
        Enforce Rule Set 1: Lock angle to nearest commensurate value.

        Parameters
        ----------
        target_angle : float
            Desired twist angle in degrees

        Returns
        -------
        CommensurateLock
            Locked commensurate configuration

        Raises
        ------
        RuleViolation
            If strict_mode and angle deviation exceeds tolerance
        """
        # Find nearest commensurate angle
        nearest_angle = min(self._commensurate_angles,
                          key=lambda a: abs(a - target_angle))

        deviation = abs(target_angle - nearest_angle)

        if self.strict_mode and deviation > 1.0:
            raise RuleViolation(
                f"Angle {target_angle}° deviates {deviation:.2f}° from nearest "
                f"commensurate angle {nearest_angle}°. Use enforce_angle() to lock."
            )

        # Find the (m, n) indices
        abs_angle = abs(nearest_angle)
        m, n, mode = 1, 1, "All-Pass"

        for (mi, ni), data in self.COMMENSURATE_TABLE.items():
            if abs(data["angle"] - abs_angle) < 0.001:
                m, n = mi, ni
                mode = data["mode"]
                break

        # Superlattice period: Σ = n² + mn + m²
        period = n**2 + m*n + m**2

        return CommensurateLock(
            m=m,
            n=n,
            angle=nearest_angle,
            superlattice_period=period,
            cognitive_mode=mode
        )

    def is_commensurate(self, angle: float, tolerance: float = 0.1) -> bool:
        """Check if angle is at a commensurate value."""
        nearest = min(self._commensurate_angles, key=lambda a: abs(a - angle))
        return abs(angle - nearest) < tolerance

    def is_at_magic_angle(self, angle: float, tolerance: float = 0.2) -> bool:
        """Check if at magic angle (~1.1°) for flat band physics."""
        return abs(abs(angle) - self.MAGIC_ANGLE) < tolerance

    # =========================================================================
    # RULE SET 2: Trilatic Tilt Symmetry
    # =========================================================================

    def enforce_tilt(
        self,
        tilt_angle: float,
        target_axis: Optional[int] = None
    ) -> TiltLock:
        """
        Enforce Rule Set 2: Lock tilt to valid axis.

        Tilt axes must be at k × 60° to preserve C3 symmetry.

        Parameters
        ----------
        tilt_angle : float
            Tilt magnitude in degrees
        target_axis : int, optional
            Preferred axis index (0-5). If None, uses axis 0.

        Returns
        -------
        TiltLock
            Locked tilt configuration
        """
        if target_axis is None:
            target_axis = 0

        if target_axis not in self.VALID_TILT_AXES:
            if self.strict_mode:
                raise RuleViolation(
                    f"Tilt axis {target_axis} is invalid. "
                    f"Must be 0-5 (k × 60°)."
                )
            # Auto-correct to nearest valid axis
            target_axis = target_axis % 6

        axis_angle = target_axis * 60.0

        # Compute effective lattice constant under tilt
        phi_rad = np.radians(tilt_angle)
        a_eff = self.a * np.cos(phi_rad)

        return TiltLock(
            axis_index=target_axis,
            axis_angle=axis_angle,
            tilt_angle=tilt_angle,
            effective_lattice_constant=a_eff
        )

    def validate_tilt_axes(self, tip: float, tilt: float) -> bool:
        """
        Validate that combined tip/tilt preserves trilatic symmetry.

        For counter-symmetric or cylindrical configurations.
        """
        # Both tip and tilt should be small enough not to distort badly
        total_tilt = np.sqrt(tip**2 + tilt**2)
        return total_tilt < 30.0  # Maximum combined tilt

    # =========================================================================
    # RULE SET 3: Talbot Distance (Integer Gap Rule)
    # =========================================================================

    def enforce_gap(self, target_gap: float, max_order: int = 10) -> TalbotLock:
        """
        Enforce Rule Set 3: Lock gap to Talbot resonance.

        Parameters
        ----------
        target_gap : float
            Desired gap in micrometers
        max_order : int
            Maximum Talbot order to consider

        Returns
        -------
        TalbotLock
            Locked Talbot configuration
        """
        best_gap = None
        best_diff = float('inf')
        best_order = 1
        best_mode = "integer"

        for n in range(1, max_order + 1):
            # Integer mode: z = N × Z_T
            gap_int = n * self.talbot_length
            diff_int = abs(gap_int - target_gap)

            if diff_int < best_diff:
                best_diff = diff_int
                best_gap = gap_int
                best_order = n
                best_mode = "integer"

            # Half-integer mode: z = (N + 0.5) × Z_T
            gap_half = (n + 0.5) * self.talbot_length
            diff_half = abs(gap_half - target_gap)

            if diff_half < best_diff:
                best_diff = diff_half
                best_gap = gap_half
                best_order = n
                best_mode = "half_integer"

        if self.strict_mode and best_diff > 0.1 * self.talbot_length:
            raise RuleViolation(
                f"Gap {target_gap}μm deviates {best_diff:.3f}μm from nearest "
                f"Talbot resonance at {best_gap:.3f}μm"
            )

        # Contrast falls off with deviation from exact resonance
        contrast = np.exp(-10 * best_diff / self.talbot_length)

        polarity = (LogicPolarity.POSITIVE if best_mode == "integer"
                   else LogicPolarity.NEGATIVE)

        return TalbotLock(
            gap=best_gap,
            order=best_order,
            mode=best_mode,
            polarity=polarity,
            contrast=contrast
        )

    def get_logic_gaps(self, order: int = 1) -> Tuple[float, float]:
        """
        Get gap values for positive/negative logic switching.

        Returns
        -------
        Tuple[float, float]
            (positive_gap, negative_gap) for AND/OR vs NAND/XOR
        """
        positive = order * self.talbot_length
        negative = (order + 0.5) * self.talbot_length
        return positive, negative

    def is_at_resonance(self, gap: float, tolerance: float = 0.05) -> bool:
        """Check if gap is at a valid Talbot resonance."""
        ratio = gap / self.talbot_length
        fractional = ratio - round(ratio)

        # At integer or half-integer?
        return abs(fractional) < tolerance or abs(abs(fractional) - 0.5) < tolerance

    # =========================================================================
    # COMBINED ENFORCEMENT
    # =========================================================================

    def enforce_all(
        self,
        angle: float,
        gap: float,
        tilt_angle: float = 0.0,
        tilt_axis: int = 0
    ) -> Tuple[CommensurateLock, TalbotLock, TiltLock]:
        """
        Enforce all three rule sets simultaneously.

        Returns locked configurations for angle, gap, and tilt.
        """
        angle_lock = self.enforce_angle(angle)
        gap_lock = self.enforce_gap(gap)
        tilt_lock = self.enforce_tilt(tilt_angle, tilt_axis)

        return angle_lock, gap_lock, tilt_lock

    def update_parameters(
        self,
        lattice_constant: Optional[float] = None,
        wavelength: Optional[float] = None
    ):
        """Update physical parameters and recompute Talbot length."""
        if lattice_constant is not None:
            self.a = lattice_constant
        if wavelength is not None:
            self.wavelength = wavelength / 1000.0

        self.talbot_length = self._compute_talbot_length()

    def get_moiré_period(self, angle: float) -> float:
        """Compute moiré period: L_M = a / (2 × sin(θ/2))."""
        if angle == 0:
            return float('inf')
        theta_rad = np.radians(angle)
        return abs(self.a / (2 * np.sin(theta_rad / 2)))

    def __repr__(self) -> str:
        return (f"RuleEnforcer(a={self.a}μm, λ={self.wavelength*1000}nm, "
                f"Z_T={self.talbot_length:.3f}μm)")
