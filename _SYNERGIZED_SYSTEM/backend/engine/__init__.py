"""
Optical Kirigami Moiré - Emergent Optical Cognition Simulation
==============================================================

A Python package for simulating the "Emergent Optical Cognition via
Tristable Kirigami-Moiré Architecture" as described in the technical
specification for visual modal machine cognition.

Modules:
- physics: Core physics calculations (moiré interference, Talbot effect)
- kirigami: Tristable metamaterial mechanics and reservoir computing
- control: Tripole actuator system for precise kinematic control
"""

from .physics import TrilaticLattice, MoireInterference, TalbotResonator
from .kirigami import TristableCell, KirigamiSheet
from .control import TripoleActuator

__version__ = "0.1.0"
__all__ = [
    "TrilaticLattice",
    "MoireInterference",
    "TalbotResonator",
    "TristableCell",
    "KirigamiSheet",
    "TripoleActuator",
]
