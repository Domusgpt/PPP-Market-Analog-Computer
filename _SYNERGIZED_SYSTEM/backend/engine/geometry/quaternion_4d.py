"""
Quaternion4D - Four-dimensional rotation mathematics.

This module implements quaternion algebra for controlling 4D rotations
in the H4 Constellation prototype. Unlike 3D rotations (which rotate
about an axis), 4D rotations are ISOCLINIC - they occur in two orthogonal
planes simultaneously.

Any 4D rotation can be decomposed into:
    R(p) = q_L * p * q_R†

where q_L and q_R are the Left and Right rotation quaternions.
This decomposition is the mathematical basis for the "2 directions"
per layer actuation requirement.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
from enum import Enum


class RotationMode(Enum):
    """Types of 4D rotation."""
    SIMPLE = "simple"          # Rotation in a single plane
    DOUBLE = "double"          # Rotation in two orthogonal planes (general)
    ISOCLINIC = "isoclinic"    # Equal rotation in both planes
    CLIFFORD = "clifford"      # Special isoclinic (Clifford translation)


@dataclass
class Quaternion4D:
    """
    A unit quaternion for 4D rotation control.

    Quaternion: q = w + xi + yj + zk
    where:
        - w is the scalar (real) component
        - x, y, z are the vector (imaginary) components
        - i, j, k are the quaternion units with i² = j² = k² = ijk = -1

    For unit quaternions: w² + x² + y² + z² = 1
    """
    w: float = 1.0  # Scalar component
    x: float = 0.0  # i component
    y: float = 0.0  # j component
    z: float = 0.0  # k component

    def __post_init__(self):
        """Normalize to unit quaternion."""
        self._normalize()

    def _normalize(self):
        """Normalize quaternion to unit length."""
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 1e-10:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Quaternion4D":
        """Create quaternion from numpy array."""
        return cls(arr[0], arr[1], arr[2], arr[3])

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> "Quaternion4D":
        """
        Create quaternion from axis-angle representation (3D rotation).

        Args:
            axis: Unit vector defining rotation axis [x, y, z]
            angle: Rotation angle in radians

        Returns:
            Unit quaternion representing the rotation
        """
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        s = np.sin(half_angle)
        return cls(
            w=np.cos(half_angle),
            x=axis[0] * s,
            y=axis[1] * s,
            z=axis[2] * s
        )

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> "Quaternion4D":
        """
        Create quaternion from Euler angles.

        Args:
            roll: Rotation about x-axis (radians)
            pitch: Rotation about y-axis (radians)
            yaw: Rotation about z-axis (radians)

        Returns:
            Unit quaternion
        """
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)

        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )

    @classmethod
    def identity(cls) -> "Quaternion4D":
        """Return identity quaternion (no rotation)."""
        return cls(1.0, 0.0, 0.0, 0.0)

    def conjugate(self) -> "Quaternion4D":
        """Return conjugate quaternion q† = w - xi - yj - zk."""
        return Quaternion4D(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> "Quaternion4D":
        """Return inverse quaternion. For unit quaternions, inverse = conjugate."""
        return self.conjugate()

    def __mul__(self, other: "Quaternion4D") -> "Quaternion4D":
        """
        Quaternion multiplication (Hamilton product).

        The product is non-commutative: q1 * q2 ≠ q2 * q1
        """
        return Quaternion4D(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )

    def __add__(self, other: "Quaternion4D") -> "Quaternion4D":
        """Quaternion addition."""
        result = Quaternion4D(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
        return result

    def __neg__(self) -> "Quaternion4D":
        """Negation."""
        return Quaternion4D(-self.w, -self.x, -self.y, -self.z)

    def __sub__(self, other: "Quaternion4D") -> "Quaternion4D":
        """Quaternion subtraction."""
        return self + (-other)

    def scale(self, s: float) -> "Quaternion4D":
        """Scale quaternion by scalar (not normalized)."""
        return Quaternion4D(self.w * s, self.x * s, self.y * s, self.z * s)

    def dot(self, other: "Quaternion4D") -> float:
        """Dot product of quaternions."""
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        """Quaternion norm (magnitude)."""
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def to_rotation_matrix_3d(self) -> np.ndarray:
        """
        Convert to 3x3 rotation matrix (for 3D rotations only).

        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])

    def rotate_vector_3d(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate a 3D vector using this quaternion.

        Args:
            v: 3D vector [x, y, z]

        Returns:
            Rotated 3D vector
        """
        # Convert vector to pure quaternion
        p = Quaternion4D(0, v[0], v[1], v[2])

        # Rotation: q * p * q†
        result = self * p * self.conjugate()

        return np.array([result.x, result.y, result.z])

    def get_scalar_part(self) -> float:
        """Get scalar (real) component."""
        return self.w

    def get_vector_part(self) -> np.ndarray:
        """Get vector (imaginary) components."""
        return np.array([self.x, self.y, self.z])

    @classmethod
    def slerp(cls, q1: "Quaternion4D", q2: "Quaternion4D",
              t: float) -> "Quaternion4D":
        """
        Spherical linear interpolation between two quaternions.

        Args:
            q1: Start quaternion
            q2: End quaternion
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated quaternion
        """
        dot = q1.dot(q2)

        # Handle negative dot product (choose shorter path)
        if dot < 0:
            q2 = -q2
            dot = -dot

        # Clamp dot product
        dot = np.clip(dot, -1.0, 1.0)

        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = Quaternion4D(
                q1.w + t * (q2.w - q1.w),
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z)
            )
            return result

        # Spherical interpolation
        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return Quaternion4D(
            s0 * q1.w + s1 * q2.w,
            s0 * q1.x + s1 * q2.x,
            s0 * q1.y + s1 * q2.y,
            s0 * q1.z + s1 * q2.z
        )


@dataclass
class QuaternionRotation:
    """
    A complete 4D rotation specified by Left and Right quaternions.

    In 4D, any rotation can be expressed as:
        R(p) = q_L * p * q_R†

    where:
        - q_L is the "Left" quaternion
        - q_R is the "Right" quaternion
        - p is the point being rotated (as a quaternion)

    Special cases:
        - q_L = q_R: Simple rotation (rotation in one plane)
        - q_L ≠ q_R: Double rotation (rotation in two orthogonal planes)
        - |angle_L| = |angle_R|: Isoclinic rotation (Clifford translation)
    """
    q_left: Quaternion4D = field(default_factory=Quaternion4D.identity)
    q_right: Quaternion4D = field(default_factory=Quaternion4D.identity)
    mode: RotationMode = RotationMode.SIMPLE

    @classmethod
    def from_single_quaternion(cls, q: Quaternion4D) -> "QuaternionRotation":
        """Create a simple rotation from a single quaternion."""
        return cls(q_left=q, q_right=q, mode=RotationMode.SIMPLE)

    @classmethod
    def from_planes(cls,
                    plane1: Tuple[int, int],
                    angle1: float,
                    plane2: Tuple[int, int],
                    angle2: float) -> "QuaternionRotation":
        """
        Create a double rotation from two plane specifications.

        Args:
            plane1: First plane as coordinate pair (e.g., (0, 1) for xy-plane)
            angle1: Rotation angle in first plane (radians)
            plane2: Second plane as coordinate pair
            angle2: Rotation angle in second plane (radians)

        Returns:
            QuaternionRotation representing the double rotation
        """
        # Convert plane rotations to quaternion components
        # This is the key insight for 4D rotation control

        c1, s1 = np.cos(angle1 / 2), np.sin(angle1 / 2)
        c2, s2 = np.cos(angle2 / 2), np.sin(angle2 / 2)

        # Construct left and right quaternions based on planes
        # The mapping depends on which planes are specified

        if plane1 == (0, 1) and plane2 == (2, 3):  # xy and zw planes
            q_l = Quaternion4D(c1 * c2, s1 * c2, c1 * s2, s1 * s2)
            q_r = Quaternion4D(c1 * c2, s1 * c2, -c1 * s2, -s1 * s2)
        elif plane1 == (0, 2) and plane2 == (1, 3):  # xz and yw planes
            q_l = Quaternion4D(c1 * c2, c1 * s2, s1 * c2, s1 * s2)
            q_r = Quaternion4D(c1 * c2, -c1 * s2, s1 * c2, -s1 * s2)
        else:
            # General case - compute using rotation matrices
            q_l = Quaternion4D(c1, s1, 0, 0)
            q_r = Quaternion4D(c2, 0, s2, 0)

        mode = RotationMode.ISOCLINIC if np.isclose(angle1, angle2) else RotationMode.DOUBLE

        return cls(q_left=q_l, q_right=q_r, mode=mode)

    @classmethod
    def isoclinic(cls, angle: float, chirality: str = "left") -> "QuaternionRotation":
        """
        Create an isoclinic (Clifford) rotation.

        Isoclinic rotations rotate equally in two orthogonal planes,
        creating a "screw-like" motion through 4D space.

        Args:
            angle: Rotation angle (same for both planes)
            chirality: "left" or "right" isoclinic

        Returns:
            QuaternionRotation for isoclinic rotation
        """
        c, s = np.cos(angle / 2), np.sin(angle / 2)

        if chirality == "left":
            q_l = Quaternion4D(c, s, 0, 0)
            q_r = Quaternion4D(c, s, 0, 0)
        else:
            q_l = Quaternion4D(c, s, 0, 0)
            q_r = Quaternion4D(c, -s, 0, 0)

        return cls(q_left=q_l, q_right=q_r, mode=RotationMode.ISOCLINIC)

    def apply_to_point(self, point: np.ndarray) -> np.ndarray:
        """
        Apply 4D rotation to a point.

        Args:
            point: 4D point as [x, y, z, w]

        Returns:
            Rotated 4D point
        """
        # Convert point to quaternion representation
        p = Quaternion4D(point[3], point[0], point[1], point[2])

        # Apply rotation: q_L * p * q_R†
        rotated = self.q_left * p * self.q_right.conjugate()

        return np.array([rotated.x, rotated.y, rotated.z, rotated.w])

    def apply_to_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply 4D rotation to multiple points.

        Args:
            points: Array of 4D points (N x 4)

        Returns:
            Rotated points (N x 4)
        """
        return np.array([self.apply_to_point(p) for p in points])

    def to_4x4_matrix(self) -> np.ndarray:
        """
        Convert to 4x4 rotation matrix.

        Returns:
            4x4 orthogonal matrix representing the rotation
        """
        # Use basis vectors to construct the matrix
        basis = np.eye(4)
        rotated_basis = self.apply_to_points(basis)
        return rotated_basis.T

    def compose(self, other: "QuaternionRotation") -> "QuaternionRotation":
        """Compose two 4D rotations."""
        return QuaternionRotation(
            q_left=self.q_left * other.q_left,
            q_right=self.q_right * other.q_right,
            mode=RotationMode.DOUBLE
        )

    def inverse(self) -> "QuaternionRotation":
        """Get inverse rotation."""
        return QuaternionRotation(
            q_left=self.q_left.inverse(),
            q_right=self.q_right.inverse(),
            mode=self.mode
        )


@dataclass
class IsoclinicRotation:
    """
    Specialized class for isoclinic (Clifford) rotations.

    Isoclinic rotations are the 4D analogue of screw motions -
    they simultaneously rotate in two orthogonal planes by the same angle.
    This creates beautiful, symmetric trajectories on the 3-sphere.

    In the H4 Constellation, isoclinic rotations are used to:
    1. Transition between the five embedded 24-cells in a 600-cell
    2. Generate the "dimensionality overlapping" effect in layer pairs
    3. Create smooth animations of 4D polytope projections
    """
    angle: float = 0.0
    axis_pair: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 1), (2, 3))
    chirality: str = "left"  # "left" or "right"

    def __post_init__(self):
        """Validate axis pair orthogonality."""
        p1, p2 = self.axis_pair
        # Check that planes are orthogonal (no shared axes)
        if set(p1) & set(p2):
            raise ValueError("Axis pairs must be orthogonal (no shared axes)")

    def to_quaternion_rotation(self) -> QuaternionRotation:
        """Convert to QuaternionRotation representation."""
        return QuaternionRotation.isoclinic(self.angle, self.chirality)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply isoclinic rotation to points."""
        return self.to_quaternion_rotation().apply_to_points(points)

    def get_trajectory(self, point: np.ndarray, num_steps: int = 100) -> np.ndarray:
        """
        Generate trajectory of a point under continuous isoclinic rotation.

        Args:
            point: Starting 4D point
            num_steps: Number of steps for full rotation

        Returns:
            Array of points along trajectory (num_steps x 4)
        """
        trajectory = []
        for i in range(num_steps):
            partial_angle = self.angle * i / (num_steps - 1)
            partial_rot = IsoclinicRotation(
                partial_angle, self.axis_pair, self.chirality
            )
            trajectory.append(partial_rot.apply(point.reshape(1, -1))[0])
        return np.array(trajectory)

    @classmethod
    def golden_rotation(cls) -> "IsoclinicRotation":
        """
        Create a Golden Ratio isoclinic rotation.

        This rotation is special because it maps 24-cells to 24-cells
        within the 600-cell structure.
        """
        phi = (1 + np.sqrt(5)) / 2
        angle = 2 * np.arctan(phi)  # ≈ 116.57°
        return cls(angle=angle, axis_pair=((0, 1), (2, 3)), chirality="left")


class QuaternionController:
    """
    Main controller for quaternion-based 4D actuation.

    This class implements the "Dimensionality Overlapping Matrix" that
    maps 4D quaternion input to the 6 physical kirigami layers.

    The mapping logic:
        - Component i (x-vector) → Primary: Pair A, Secondary: Pair B
        - Component j (y-vector) → Primary: Pair B, Secondary: Pair C
        - Component k (z-vector) → Primary: Pair C, Secondary: Pair A
        - Component w (scalar) → Global baseline for all layers
    """

    def __init__(self):
        """Initialize the quaternion controller."""
        # Dimensionality Overlapping Matrix
        # Rows: quaternion components [w, x, y, z]
        # Cols: layer pairs [A, B, C]
        self.overlap_matrix = np.array([
            [1.0, 1.0, 1.0],   # w (scalar) affects all pairs equally
            [1.0, 0.5, 0.0],   # x affects A primarily, B secondarily
            [0.0, 1.0, 0.5],   # y affects B primarily, C secondarily
            [0.5, 0.0, 1.0],   # z affects C primarily, A secondarily
        ])

        # State registers for the three trilatic channels
        self.state_alpha = np.zeros(8)  # 16-cell A state
        self.state_beta = np.zeros(8)   # 16-cell B state
        self.state_gamma = np.zeros(8)  # 16-cell C state

        # Current quaternion state
        self.current_quaternion = Quaternion4D.identity()

    def decompose_quaternion(self,
                              q: Quaternion4D) -> Tuple[Quaternion4D, Quaternion4D]:
        """
        Decompose quaternion into Left and Right components.

        This decomposition is the key to mapping 4D rotations to
        layer pair actuation.

        Returns:
            (q_left, q_right) quaternion pair
        """
        # For a general quaternion, the decomposition is:
        # q = q_L * q_R where the split is determined by
        # the rotation plane structure

        # Simple decomposition: split by scalar/vector parts
        w = q.w
        v_norm = np.sqrt(q.x**2 + q.y**2 + q.z**2)

        if v_norm < 1e-10:
            # Pure scalar quaternion
            return Quaternion4D.identity(), Quaternion4D.identity()

        # Decompose into two half-angle rotations
        half_angle = np.arccos(w) / 2
        axis = q.get_vector_part() / v_norm

        c, s = np.cos(half_angle), np.sin(half_angle)

        q_left = Quaternion4D(c, axis[0] * s, axis[1] * s, 0)
        q_right = Quaternion4D(c, 0, axis[1] * s, axis[2] * s)

        return q_left, q_right

    def compute_layer_actuation(self,
                                 q: Quaternion4D) -> np.ndarray:
        """
        Compute actuation values for the 6 kirigami layers.

        Args:
            q: Input quaternion

        Returns:
            Array of 6 actuation values in [0, 1] range
        """
        # Get quaternion components
        components = q.to_array()  # [w, x, y, z]

        # Apply dimensionality overlapping matrix
        pair_values = self.overlap_matrix.T @ components

        # Normalize to [0, 1] range
        pair_values = np.clip(pair_values, 0, 1)

        # Decompose into left/right for each pair
        q_left, q_right = self.decompose_quaternion(q)

        # Layer values: alternating Cyan (left) and Magenta (right)
        layer_values = np.zeros(6)

        # Pair A (Layers 1-2)
        layer_values[0] = pair_values[0] * np.abs(q_left.x)  # Cyan
        layer_values[1] = pair_values[0] * np.abs(q_right.x)  # Magenta

        # Pair B (Layers 3-4)
        layer_values[2] = pair_values[1] * np.abs(q_left.y)  # Cyan
        layer_values[3] = pair_values[1] * np.abs(q_right.y)  # Magenta

        # Pair C (Layers 5-6)
        layer_values[4] = pair_values[2] * np.abs(q_left.z)  # Cyan
        layer_values[5] = pair_values[2] * np.abs(q_right.z)  # Magenta

        # Add global baseline from scalar component
        baseline = np.abs(q.w) * 0.1
        layer_values += baseline

        # Clamp to [0, 1]
        return np.clip(layer_values, 0, 1)

    def discretize_to_states(self,
                              actuation: np.ndarray) -> np.ndarray:
        """
        Discretize actuation values to 0, 0.5, 1 states.

        Args:
            actuation: Continuous actuation values [0, 1]

        Returns:
            Discretized state values {0, 0.5, 1}
        """
        states = np.zeros_like(actuation)

        for i, val in enumerate(actuation):
            if val < 0.25:
                states[i] = 0.0
            elif val < 0.75:
                states[i] = 0.5
            else:
                states[i] = 1.0

        return states

    def update_state(self, q: Quaternion4D):
        """Update internal state with new quaternion input."""
        self.current_quaternion = q
        actuation = self.compute_layer_actuation(q)
        states = self.discretize_to_states(actuation)

        # Update trilatic state registers
        self.state_alpha[:] = states[:2].mean()
        self.state_beta[:] = states[2:4].mean()
        self.state_gamma[:] = states[4:6].mean()

    def get_chain_reaction(self) -> np.ndarray:
        """
        Compute the "chain reaction" effect across layer pairs.

        The overlapping dimensionality creates a circular dependency
        where changing one pair affects the others.

        Returns:
            Coupling strengths between pairs (3x3 matrix)
        """
        # Coupling matrix derived from overlap structure
        coupling = np.array([
            [1.0, 0.3, 0.1],  # A → A, B, C
            [0.1, 1.0, 0.3],  # B → A, B, C
            [0.3, 0.1, 1.0],  # C → A, B, C
        ])

        return coupling

    def interpolate_quaternion(self,
                                q_target: Quaternion4D,
                                num_steps: int = 10) -> List[Quaternion4D]:
        """
        Generate smooth interpolation from current to target quaternion.

        Args:
            q_target: Target quaternion
            num_steps: Number of interpolation steps

        Returns:
            List of interpolated quaternions
        """
        result = []
        for i in range(num_steps + 1):
            t = i / num_steps
            q_interp = Quaternion4D.slerp(self.current_quaternion, q_target, t)
            result.append(q_interp)
        return result

    def process_data_stream(self,
                             data: np.ndarray) -> List[np.ndarray]:
        """
        Process a stream of 4D data vectors as quaternions.

        Args:
            data: Array of 4D vectors (N x 4)

        Returns:
            List of layer actuation arrays
        """
        actuations = []

        for vec in data:
            # Normalize to unit quaternion
            q = Quaternion4D.from_array(vec)
            actuation = self.compute_layer_actuation(q)
            actuations.append(actuation)

        return actuations
