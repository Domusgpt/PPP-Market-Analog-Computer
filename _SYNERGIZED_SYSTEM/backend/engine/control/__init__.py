"""
Control Layer Module
====================

Implements the Tripole Actuator System for precise kinematic
control of the kirigami layers.

Section 6 of the specification defines the parallel kinematic
tripole system with 3-DOF control (Tip, Tilt, Piston).
"""

from .tripole_actuator import TripoleActuator, ActuatorCommand

__all__ = ["TripoleActuator", "ActuatorCommand"]
