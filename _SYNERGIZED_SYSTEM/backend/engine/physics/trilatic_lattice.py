"""
Trilatic Lattice - Hexagonal Lattice Geometry for Moiré Computing
=================================================================

Implements the triangular (hexagonal) lattice geometry with C6 rotational
symmetry as specified in Section 2.3 of the technical document.

The "trilatic" geometry provides:
- Isotropic bandgaps (uniform optical properties in all directions)
- Three wave vectors separated by 120 degrees
- Support for flat band physics at magic angles
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
import math


class CommensurateModeIndex(Enum):
    """
    Predefined commensurate angle indices (m, n) from Table 1.
    These ensure uniform periodic processing grids.

    Implements Rule Set 1: Angular Commensurability (Pythagorean Rule)
    """
    ALIGNED = (1, 1)           # 0 deg - All-Pass / Transparent Mode
    COARSE = (2, 1)            # 21.79 deg - Coarse Filtering
    INTERMEDIATE = (3, 1)      # 13.17 deg - Intermediate Filtering
    FINE = (4, 1)              # 9.43 deg - Fine Detail / Texture Analysis
    EDGE = (5, 1)              # 7.34 deg - Edge Detection
    # Magic angle ~1.1 deg is handled separately


@dataclass
class LatticeVector:
    """Represents a lattice vector in 2D with hexagonal coordinates."""
    a1: np.ndarray  # First primitive vector
    a2: np.ndarray  # Second primitive vector

    @property
    def a3(self) -> np.ndarray:
        """Third vector for trilatic symmetry (a3 = -(a1 + a2))"""
        return -(self.a1 + self.a2)


class TrilaticLattice:
    """
    Hexagonal lattice class implementing trilatic geometry for moiré computing.

    The lattice constant 'a' defines the base periodicity. For a hexagonal
    lattice, the primitive vectors are:
        a1 = a * (1, 0)
        a2 = a * (cos(60°), sin(60°)) = a * (0.5, sqrt(3)/2)

    Parameters
    ----------
    lattice_constant : float
        Base lattice constant 'a' in micrometers (default: 1.0 μm)
    """

    # Implements Rule Set 1: Pythagorean Commensurate Angles
    # cos(theta) = (n^2 + 4mn + m^2) / (2(n^2 + mn + m^2))
    COMMENSURATE_ANGLES = {
        (1, 1): 0.0,
        (2, 1): 21.787,      # degrees
        (3, 1): 13.174,
        (4, 1): 9.430,
        (5, 1): 7.341,
        (6, 1): 6.009,
        (7, 1): 5.086,
        (8, 1): 4.408,
    }

    MAGIC_ANGLE = 1.1  # degrees - flat band regime

    def __init__(self, lattice_constant: float = 1.0):
        """
        Initialize a trilatic (hexagonal) lattice.

        Parameters
        ----------
        lattice_constant : float
            Lattice spacing in micrometers
        """
        self.a = lattice_constant
        self._update_vectors()

    def _update_vectors(self):
        """Update primitive lattice vectors based on current lattice constant."""
        # Hexagonal lattice primitive vectors
        self.a1 = np.array([self.a, 0.0])
        self.a2 = np.array([self.a * 0.5, self.a * np.sqrt(3) / 2])
        self.lattice_vectors = LatticeVector(self.a1, self.a2)

        # Reciprocal lattice vectors (for k-space calculations)
        # b1 = 2π(a2 × z) / (a1 · (a2 × z))
        self.b1 = (2 * np.pi / self.a) * np.array([1.0, -1.0 / np.sqrt(3)])
        self.b2 = (2 * np.pi / self.a) * np.array([0.0, 2.0 / np.sqrt(3)])

    @staticmethod
    def compute_commensurate_angle(m: int, n: int) -> float:
        """
        Compute commensurate angle from indices (m, n).

        Implements Rule Set 1: Angular Commensurability (Pythagorean Rule)

        cos(theta) = (n^2 + 4mn + m^2) / (2(n^2 + mn + m^2))

        Parameters
        ----------
        m, n : int
            Coprime integers defining the commensurate superlattice

        Returns
        -------
        float
            Rotation angle in degrees
        """
        if math.gcd(m, n) != 1:
            raise ValueError(f"m={m} and n={n} must be coprime integers")

        numerator = n**2 + 4*m*n + m**2
        denominator = 2 * (n**2 + m*n + m**2)
        cos_theta = numerator / denominator

        # Clamp to valid range for numerical stability
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)

        return np.degrees(theta_rad)

    @staticmethod
    def get_superlattice_period(m: int, n: int) -> int:
        """
        Get the superlattice period index for commensurate angles.

        For hexagonal lattices, Σ = n^2 + mn + m^2

        Returns
        -------
        int
            Superlattice period (number of unit cells)
        """
        return n**2 + m*n + m**2

    def get_nearest_commensurate_angle(self, target_angle: float) -> Tuple[int, int, float]:
        """
        Find the nearest commensurate angle to a target.

        Parameters
        ----------
        target_angle : float
            Desired angle in degrees

        Returns
        -------
        Tuple[int, int, float]
            (m, n, actual_angle) for the nearest commensurate configuration
        """
        best_match = None
        min_diff = float('inf')

        for (m, n), angle in self.COMMENSURATE_ANGLES.items():
            diff = abs(angle - target_angle)
            if diff < min_diff:
                min_diff = diff
                best_match = (m, n, angle)

        return best_match

    def generate_lattice_points(
        self,
        n_cells: int = 10,
        rotation: float = 0.0
    ) -> np.ndarray:
        """
        Generate lattice points for visualization or calculation.

        Parameters
        ----------
        n_cells : int
            Number of unit cells in each direction
        rotation : float
            Rotation angle in degrees

        Returns
        -------
        np.ndarray
            Array of (x, y) coordinates, shape (N, 2)
        """
        points = []
        theta = np.radians(rotation)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        for i in range(-n_cells, n_cells + 1):
            for j in range(-n_cells, n_cells + 1):
                point = i * self.a1 + j * self.a2
                rotated = R @ point
                points.append(rotated)

        return np.array(points)

    def apply_tilt(self, tilt_angle: float, tilt_axis_index: int = 0) -> float:
        """
        Compute effective lattice constant under tilt.

        Implements Rule Set 2: Trilatic Tilt Symmetry
        The effective lattice constant is: a_eff = a * cos(phi)

        Tilt axes must be aligned with reciprocal lattice vectors:
        phi_tilt = k * (pi/3) where k is an integer

        Parameters
        ----------
        tilt_angle : float
            Tilt angle phi in degrees
        tilt_axis_index : int
            Index k (0-5) specifying tilt axis at k * 60 degrees

        Returns
        -------
        float
            Effective lattice constant
        """
        # Implements Rule Set 2: Orthogonality Rule
        if tilt_axis_index not in range(6):
            raise ValueError("Tilt axis index must be 0-5 (k * pi/3)")

        phi_rad = np.radians(tilt_angle)
        a_eff = self.a * np.cos(phi_rad)

        return a_eff

    def get_trilatic_wave_vectors(self) -> List[np.ndarray]:
        """
        Get the three wave vectors for trilatic moiré interference.

        These are separated by 120 degrees as specified in Section 2.3.

        Returns
        -------
        List[np.ndarray]
            Three wave vectors G1, G2, G3
        """
        G1 = self.b1
        G2 = self.b2
        G3 = -(self.b1 + self.b2)  # Closure condition

        return [G1, G2, G3]

    def is_at_magic_angle(self, theta: float, tolerance: float = 0.1) -> bool:
        """
        Check if the system is near the magic angle (~1.1 degrees).

        At the magic angle, flat bands emerge where the group velocity
        approaches zero, enabling intense non-linear memory updates.

        Parameters
        ----------
        theta : float
            Current twist angle in degrees
        tolerance : float
            Acceptable deviation in degrees

        Returns
        -------
        bool
            True if within tolerance of magic angle
        """
        return abs(theta - self.MAGIC_ANGLE) < tolerance
