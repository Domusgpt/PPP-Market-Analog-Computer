"""
Tripole Actuator System - Kinematic Control for Moiré Computing
===============================================================

Implements the Parallel Kinematic Tripole System from Section 6.

The tripole geometry uses three linear actuators arranged in an
equilateral triangle (120 degrees apart) to control:
- Piston: Global gap (Talbot logic switching)
- Tip: X-gradient (horizontal attention)
- Tilt: Y-gradient (vertical attention)

Plus a separate rotary stage for twist angle control.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
from enum import Enum


class ActuatorCommand(Enum):
    """
    High-level actuator commands from Table 2.

    These map cognitive intentions to actuator coordinates.
    """
    PISTON = "piston"           # Global phase shift (Talbot toggle)
    TIP_X = "tip_x"             # X-gradient attention
    TILT_Y = "tilt_y"           # Y-gradient attention
    ROTATE = "rotate"           # Twist angle (commensurate snap)


@dataclass
class ActuatorState:
    """Current state of the tripole actuator system."""
    z_a: float              # Actuator A extension (at 0 deg)
    z_b: float              # Actuator B extension (at 120 deg)
    z_c: float              # Actuator C extension (at 240 deg)
    theta: float            # Rotary stage angle (degrees)
    timestamp: float = 0.0


@dataclass
class PlaneState:
    """Resulting plane orientation from actuator state."""
    gap: float              # Mean gap (piston component)
    tip: float              # Tip angle about Y axis (degrees)
    tilt: float             # Tilt angle about X axis (degrees)
    twist: float            # Rotation angle (degrees)


class TripoleActuator:
    """
    Tripole kinematic actuator for optical kirigami control.

    From Section 6.1: Three linear actuators are arranged in an
    equilateral triangle configuration, spaced 120 degrees apart
    at radius R from the optical axis.

    This geometry is mathematically optimal for controlling the
    plane equation z = Ax + By + C.

    Parameters
    ----------
    radius : float
        Radius R from optical axis to actuators (mm)
    max_extension : float
        Maximum actuator extension (micrometers)
    resolution : float
        Actuator position resolution (nm)
    """

    # Actuator angular positions (radians)
    ANGLE_A = 0.0                    # 0 degrees
    ANGLE_B = 2 * np.pi / 3          # 120 degrees
    ANGLE_C = 4 * np.pi / 3          # 240 degrees

    def __init__(
        self,
        radius: float = 25.0,
        max_extension: float = 100.0,
        resolution: float = 1.0
    ):
        self.radius = radius  # mm
        self.max_extension = max_extension  # micrometers
        self.resolution = resolution / 1000.0  # Convert nm to micrometers

        # Current state
        self._state = ActuatorState(
            z_a=0.0, z_b=0.0, z_c=0.0, theta=0.0
        )

        # Actuator XY positions
        self._positions = self._compute_actuator_positions()

        # Command history for trajectory analysis
        self._history: List[ActuatorState] = []

    def _compute_actuator_positions(self) -> Dict[str, Tuple[float, float]]:
        """Compute XY positions of actuators on the mount."""
        return {
            'A': (self.radius * np.cos(self.ANGLE_A),
                  self.radius * np.sin(self.ANGLE_A)),
            'B': (self.radius * np.cos(self.ANGLE_B),
                  self.radius * np.sin(self.ANGLE_B)),
            'C': (self.radius * np.cos(self.ANGLE_C),
                  self.radius * np.sin(self.ANGLE_C))
        }

    @property
    def state(self) -> ActuatorState:
        """Get current actuator state."""
        return self._state

    @property
    def plane_state(self) -> PlaneState:
        """Get resulting plane orientation."""
        return self._compute_plane_state()

    def _compute_plane_state(self) -> PlaneState:
        """
        Compute plane orientation from actuator extensions.

        The plane equation z = Ax + By + C is determined by
        three points (actuator positions with their extensions).
        """
        # Mean height (piston/gap)
        gap = (self._state.z_a + self._state.z_b + self._state.z_c) / 3

        # Tip and tilt from differential extensions
        # Uses transformation matrix for tripole geometry
        R = self.radius

        # Tip about Y-axis: primarily controlled by A vs (B+C)/2
        delta_tip = self._state.z_a - (self._state.z_b + self._state.z_c) / 2
        tip_rad = np.arctan2(delta_tip, R * 1.5)
        tip = np.degrees(tip_rad)

        # Tilt about X-axis: primarily controlled by B vs C
        delta_tilt = self._state.z_b - self._state.z_c
        tilt_rad = np.arctan2(delta_tilt, R * np.sqrt(3))
        tilt = np.degrees(tilt_rad)

        return PlaneState(
            gap=gap,
            tip=tip,
            tilt=tilt,
            twist=self._state.theta
        )

    def command_piston(self, delta_z: float) -> ActuatorState:
        """
        Execute piston (global gap) command.

        From Table 2: Equal extension of A, B, C.
        This toggles Talbot logic (AND <-> XOR).

        Parameters
        ----------
        delta_z : float
            Gap change in micrometers

        Returns
        -------
        ActuatorState
            New actuator state
        """
        # Implements Rule Set 3: Integer Gap Rule
        # Uniform extension for piston motion
        self._state.z_a = self._clamp_extension(self._state.z_a + delta_z)
        self._state.z_b = self._clamp_extension(self._state.z_b + delta_z)
        self._state.z_c = self._clamp_extension(self._state.z_c + delta_z)

        self._record_state()
        return self._state

    def command_tip(self, angle: float) -> ActuatorState:
        """
        Execute tip (X-gradient) command.

        From Table 2: Differential extension of B and C vs A.
        This scans attention horizontally.

        Parameters
        ----------
        angle : float
            Tip angle in degrees

        Returns
        -------
        ActuatorState
            New actuator state
        """
        # Implements Rule Set 2: Trilatic Tilt Symmetry
        angle_rad = np.radians(angle)
        delta = self.radius * np.tan(angle_rad)

        # A moves opposite to (B+C)/2
        self._state.z_a = self._clamp_extension(self._state.z_a + delta)
        self._state.z_b = self._clamp_extension(self._state.z_b - delta / 2)
        self._state.z_c = self._clamp_extension(self._state.z_c - delta / 2)

        self._record_state()
        return self._state

    def command_tilt(self, angle: float) -> ActuatorState:
        """
        Execute tilt (Y-gradient) command.

        From Table 2: Differential extension of B vs C.
        This scans attention vertically.

        Parameters
        ----------
        angle : float
            Tilt angle in degrees

        Returns
        -------
        ActuatorState
            New actuator state
        """
        # Implements Rule Set 2: Trilatic Tilt Symmetry
        angle_rad = np.radians(angle)
        delta = self.radius * np.sqrt(3) * np.tan(angle_rad)

        # B and C move opposite directions
        self._state.z_b = self._clamp_extension(self._state.z_b + delta / 2)
        self._state.z_c = self._clamp_extension(self._state.z_c - delta / 2)

        self._record_state()
        return self._state

    def command_rotate(self, theta: float, snap_to_commensurate: bool = True) -> ActuatorState:
        """
        Execute rotation command.

        From Section 5.1: The actuation system must effectively
        "snap" to commensurate discrete angles.

        Parameters
        ----------
        theta : float
            Rotation angle in degrees
        snap_to_commensurate : bool
            If True, snap to nearest commensurate angle

        Returns
        -------
        ActuatorState
            New actuator state
        """
        # Implements Rule Set 1: Pythagorean Commensurability
        if snap_to_commensurate:
            theta = self._snap_to_commensurate(theta)

        self._state.theta = theta
        self._record_state()
        return self._state

    def _snap_to_commensurate(self, target_theta: float) -> float:
        """
        Snap angle to nearest commensurate value.

        Commensurate angles from Table 1:
        0, 21.79, 13.17, 9.43, 7.34 degrees (and their negatives)
        """
        # Commensurate angles for trilatic moiré
        commensurate_angles = [0.0, 7.34, 9.43, 13.17, 21.79]

        # Include negatives
        all_angles = []
        for a in commensurate_angles:
            all_angles.extend([a, -a])

        # Find nearest
        best_angle = min(all_angles, key=lambda a: abs(a - target_theta))

        return best_angle

    def set_state(
        self,
        z_a: Optional[float] = None,
        z_b: Optional[float] = None,
        z_c: Optional[float] = None,
        theta: Optional[float] = None
    ):
        """
        Directly set actuator positions.

        Parameters
        ----------
        z_a, z_b, z_c : float, optional
            Actuator extensions in micrometers
        theta : float, optional
            Rotation angle in degrees
        """
        if z_a is not None:
            self._state.z_a = self._clamp_extension(z_a)
        if z_b is not None:
            self._state.z_b = self._clamp_extension(z_b)
        if z_c is not None:
            self._state.z_c = self._clamp_extension(z_c)
        if theta is not None:
            self._state.theta = theta

        self._record_state()

    def set_gap(self, target_gap: float):
        """
        Set absolute gap value.

        Adjusts all actuators uniformly to achieve target.

        Parameters
        ----------
        target_gap : float
            Desired gap in micrometers
        """
        current_gap = self.plane_state.gap
        delta = target_gap - current_gap
        self.command_piston(delta)

    def set_plane(
        self,
        gap: float,
        tip: float = 0.0,
        tilt: float = 0.0,
        twist: float = 0.0
    ):
        """
        Set complete plane state.

        Parameters
        ----------
        gap : float
            Gap in micrometers
        tip : float
            Tip angle in degrees
        tilt : float
            Tilt angle in degrees
        twist : float
            Twist angle in degrees
        """
        # Reset to neutral
        self._state = ActuatorState(z_a=0, z_b=0, z_c=0, theta=0)

        # Apply gap (piston)
        self.command_piston(gap)

        # Apply tip
        if tip != 0:
            tip_rad = np.radians(tip)
            delta = self.radius * np.tan(tip_rad)
            self._state.z_a += delta
            self._state.z_b -= delta / 2
            self._state.z_c -= delta / 2

        # Apply tilt
        if tilt != 0:
            tilt_rad = np.radians(tilt)
            delta = self.radius * np.sqrt(3) * np.tan(tilt_rad)
            self._state.z_b += delta / 2
            self._state.z_c -= delta / 2

        # Clamp extensions
        self._state.z_a = self._clamp_extension(self._state.z_a)
        self._state.z_b = self._clamp_extension(self._state.z_b)
        self._state.z_c = self._clamp_extension(self._state.z_c)

        # Apply twist
        self.command_rotate(twist)

    def _clamp_extension(self, z: float) -> float:
        """Clamp extension to valid range."""
        return np.clip(z, 0, self.max_extension)

    def _record_state(self):
        """Record current state to history."""
        import copy
        self._state.timestamp = len(self._history)
        self._history.append(copy.copy(self._state))

    def get_attention_gradient(self) -> Tuple[float, float]:
        """
        Get current attention gradient direction.

        The tip/tilt angles define where the system is "looking"
        in the visual field.

        Returns
        -------
        Tuple[float, float]
            (gradient_x, gradient_y) normalized direction
        """
        plane = self.plane_state
        gx = np.sin(np.radians(plane.tip))
        gy = np.sin(np.radians(plane.tilt))

        # Normalize
        mag = np.sqrt(gx**2 + gy**2)
        if mag > 0:
            return gx / mag, gy / mag
        return 0.0, 0.0

    def compute_effective_lattice_constant(
        self,
        base_lattice_constant: float
    ) -> float:
        """
        Compute effective lattice constant under current tilt.

        a_eff = a * cos(phi) where phi is the total tilt angle.

        Parameters
        ----------
        base_lattice_constant : float
            Unmodified lattice constant

        Returns
        -------
        float
            Effective lattice constant
        """
        plane = self.plane_state
        total_tilt = np.sqrt(plane.tip**2 + plane.tilt**2)
        phi_rad = np.radians(total_tilt)

        return base_lattice_constant * np.cos(phi_rad)

    def reset(self):
        """Reset actuators to home position."""
        self._state = ActuatorState(z_a=0, z_b=0, z_c=0, theta=0)
        self._history.clear()
        self._record_state()

    def get_history(self) -> List[ActuatorState]:
        """Get command history."""
        return self._history.copy()

    def __repr__(self) -> str:
        plane = self.plane_state
        return (f"TripoleActuator(gap={plane.gap:.2f}μm, "
                f"tip={plane.tip:.2f}°, tilt={plane.tilt:.2f}°, "
                f"twist={plane.twist:.2f}°)")
