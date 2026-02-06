"""
Physics Kernel for Optical Kirigami Moiré
==========================================

This module implements the core physics of moiré interference,
commensurate lattice calculations, and Talbot self-imaging effects.

Design Rules Implemented:
- Rule Set 1: Angular Commensurability (Pythagorean Rule)
- Rule Set 2: Trilatic Tilt Symmetry (Orthogonality Rule)
- Rule Set 3: Intersection and Talbot Distance (Integer Gap Rule)
"""

from .trilatic_lattice import TrilaticLattice
from .moire_interference import MoireInterference
from .talbot_resonator import TalbotResonator

__all__ = ["TrilaticLattice", "MoireInterference", "TalbotResonator"]
