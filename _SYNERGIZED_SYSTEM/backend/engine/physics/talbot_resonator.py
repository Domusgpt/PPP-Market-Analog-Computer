"""
Talbot Resonator - Gap Control for Moiré Logic
==============================================

Implements Rule Set 3: The Intersection and Talbot Distance
(The Integer Gap Rule) from Section 5.3.

The Talbot effect creates self-images of periodic structures at
integer multiples of the Talbot length. This module calculates
the required gap distances for high-contrast moiré interference.

Key equation: Z_T = (3/2) * a^2 / lambda
(Factor 3/2 for hexagonal geometry vs 2 for square)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum


class TalbotMode(Enum):
    """
    Talbot resonance modes for logic switching.

    From Section 5.3:
    - INTEGER: Self-image (constructive) -> POSITIVE logic (AND/OR)
    - HALF_INTEGER: Shifted image (inverted) -> NEGATIVE logic (NAND/XOR)
    """
    INTEGER = "integer"           # z = N * Z_T
    HALF_INTEGER = "half_integer"  # z = (N + 0.5) * Z_T


@dataclass
class TalbotState:
    """Represents a Talbot resonator configuration."""
    gap: float                    # Current gap in micrometers
    talbot_length: float          # Z_T for current wavelength
    mode: TalbotMode              # INTEGER or HALF_INTEGER
    order: int                    # N (Talbot order)
    contrast: float               # Expected fringe contrast (0-1)
    logic_polarity: str           # "POSITIVE" or "NEGATIVE"


class TalbotResonator:
    """
    Talbot resonator for controlling moiré interference contrast.

    Implements Rule Set 3: The gap between kirigami layers must be
    an integer or half-integer multiple of the Talbot length to
    maintain high-contrast self-imaging.

    Parameters
    ----------
    lattice_constant : float
        Base lattice constant 'a' in micrometers
    wavelength : float
        Operating wavelength in nanometers (default: 550 nm, green)
    geometry_factor : float
        Factor for lattice geometry (1.5 for hexagonal, 2.0 for square)
    """

    # Geometry factors for different lattice types
    HEXAGONAL_FACTOR = 1.5  # 3/2 for trilatic geometry
    SQUARE_FACTOR = 2.0

    def __init__(
        self,
        lattice_constant: float = 1.0,
        wavelength: float = 550.0,
        geometry_factor: float = 1.5
    ):
        self.a = lattice_constant
        self.wavelength = wavelength / 1000.0  # Convert nm to micrometers
        self.geometry_factor = geometry_factor
        self._compute_talbot_length()

    def _compute_talbot_length(self):
        """
        Compute the Talbot length for current parameters.

        Z_T = (3/2) * a^2 / lambda (for hexagonal)
        """
        self.Z_T = self.geometry_factor * (self.a ** 2) / self.wavelength

    @property
    def talbot_length(self) -> float:
        """Get the current Talbot length in micrometers."""
        return self.Z_T

    def compute_talbot_gap(
        self,
        order: int,
        mode: TalbotMode = TalbotMode.INTEGER
    ) -> float:
        """
        Calculate gap distance for a given Talbot order and mode.

        Implements Rule Set 3: Integer Gap Rule

        Parameters
        ----------
        order : int
            Talbot order N (must be positive)
        mode : TalbotMode
            INTEGER (self-image) or HALF_INTEGER (inverted)

        Returns
        -------
        float
            Required gap z_gap in micrometers
        """
        if order < 1:
            raise ValueError("Talbot order must be >= 1")

        if mode == TalbotMode.INTEGER:
            # z_gap = N * Z_T (self-image)
            return order * self.Z_T
        else:
            # z_gap = (N + 0.5) * Z_T (shifted image)
            return (order + 0.5) * self.Z_T

    def get_nearest_talbot_gap(
        self,
        target_gap: float,
        max_order: int = 10
    ) -> TalbotState:
        """
        Find the nearest valid Talbot gap to a target distance.

        This "snaps" arbitrary gaps to valid resonance positions.

        Parameters
        ----------
        target_gap : float
            Desired gap in micrometers
        max_order : int
            Maximum Talbot order to consider

        Returns
        -------
        TalbotState
            Nearest valid Talbot configuration
        """
        best_gap = None
        best_diff = float('inf')
        best_mode = None
        best_order = 1

        for n in range(1, max_order + 1):
            # Check integer mode
            gap_int = n * self.Z_T
            diff_int = abs(gap_int - target_gap)
            if diff_int < best_diff:
                best_diff = diff_int
                best_gap = gap_int
                best_mode = TalbotMode.INTEGER
                best_order = n

            # Check half-integer mode
            gap_half = (n + 0.5) * self.Z_T
            diff_half = abs(gap_half - target_gap)
            if diff_half < best_diff:
                best_diff = diff_half
                best_gap = gap_half
                best_mode = TalbotMode.HALF_INTEGER
                best_order = n

        # Compute expected contrast (decreases with distance from exact)
        deviation = best_diff / self.Z_T
        contrast = np.exp(-deviation * 10)  # Exponential falloff

        logic = "POSITIVE" if best_mode == TalbotMode.INTEGER else "NEGATIVE"

        return TalbotState(
            gap=best_gap,
            talbot_length=self.Z_T,
            mode=best_mode,
            order=best_order,
            contrast=contrast,
            logic_polarity=logic
        )

    def get_logic_gaps(self, base_order: int = 1) -> Tuple[float, float]:
        """
        Get gap values for logic switching.

        Returns the gap distances for toggling between POSITIVE
        (AND/OR) and NEGATIVE (NAND/XOR) logic modes.

        Parameters
        ----------
        base_order : int
            Base Talbot order

        Returns
        -------
        Tuple[float, float]
            (positive_gap, negative_gap) in micrometers
        """
        positive_gap = self.compute_talbot_gap(base_order, TalbotMode.INTEGER)
        negative_gap = self.compute_talbot_gap(base_order, TalbotMode.HALF_INTEGER)

        return positive_gap, negative_gap

    def compute_diffraction_blur(self, gap: float) -> float:
        """
        Estimate diffraction-induced blur at a given gap.

        The "washout" effect mentioned in Section 5.3 occurs when
        the gap is not at a Talbot resonance.

        Parameters
        ----------
        gap : float
            Layer gap in micrometers

        Returns
        -------
        float
            Blur diameter in micrometers
        """
        # Fresnel diffraction spreading
        # sigma ~ sqrt(lambda * z)
        blur = np.sqrt(self.wavelength * gap)
        return blur

    def is_valid_resonance(
        self,
        gap: float,
        tolerance: float = 0.05
    ) -> Tuple[bool, Optional[TalbotMode]]:
        """
        Check if a gap distance is at a valid Talbot resonance.

        Parameters
        ----------
        gap : float
            Gap to check in micrometers
        tolerance : float
            Fractional tolerance (default 5%)

        Returns
        -------
        Tuple[bool, Optional[TalbotMode]]
            (is_valid, mode if valid else None)
        """
        # Check against integer multiples
        ratio = gap / self.Z_T
        integer_part = round(ratio)
        fractional_part = abs(ratio - integer_part)

        if fractional_part < tolerance:
            return True, TalbotMode.INTEGER
        elif abs(fractional_part - 0.5) < tolerance:
            return True, TalbotMode.HALF_INTEGER

        return False, None

    def generate_resonance_ladder(
        self,
        max_gap: float,
        include_half_integer: bool = True
    ) -> List[TalbotState]:
        """
        Generate all valid Talbot gaps up to a maximum distance.

        Parameters
        ----------
        max_gap : float
            Maximum gap distance in micrometers
        include_half_integer : bool
            Whether to include half-integer modes

        Returns
        -------
        List[TalbotState]
            Ordered list of valid Talbot configurations
        """
        states = []
        n = 1

        while True:
            # Integer mode
            gap_int = n * self.Z_T
            if gap_int > max_gap:
                break

            states.append(TalbotState(
                gap=gap_int,
                talbot_length=self.Z_T,
                mode=TalbotMode.INTEGER,
                order=n,
                contrast=1.0,
                logic_polarity="POSITIVE"
            ))

            # Half-integer mode
            if include_half_integer:
                gap_half = (n + 0.5) * self.Z_T
                if gap_half <= max_gap:
                    states.append(TalbotState(
                        gap=gap_half,
                        talbot_length=self.Z_T,
                        mode=TalbotMode.HALF_INTEGER,
                        order=n,
                        contrast=1.0,
                        logic_polarity="NEGATIVE"
                    ))

            n += 1

        return sorted(states, key=lambda s: s.gap)

    def update_parameters(
        self,
        lattice_constant: Optional[float] = None,
        wavelength: Optional[float] = None
    ):
        """
        Update resonator parameters and recompute Talbot length.

        Parameters
        ----------
        lattice_constant : float, optional
            New lattice constant in micrometers
        wavelength : float, optional
            New wavelength in nanometers
        """
        if lattice_constant is not None:
            self.a = lattice_constant
        if wavelength is not None:
            self.wavelength = wavelength / 1000.0

        self._compute_talbot_length()
