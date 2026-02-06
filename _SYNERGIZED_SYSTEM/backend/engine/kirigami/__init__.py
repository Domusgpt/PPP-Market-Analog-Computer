"""
Kirigami Logic Module
=====================

Implements the tristable mechanical metamaterial mechanics
and reservoir computing dynamics for optical cognition.

Key Components:
- TristableCell: Individual unit cell with states 0, 0.5, 1
- KirigamiSheet: Full lattice acting as a physical reservoir
- H4KirigamiStack: 6-layer stack for H4 Constellation prototype
- LayerPair: Cyan/Magenta layer pair with moiré interference
"""

from .tristable_cell import TristableCell, CellState
from .kirigami_sheet import KirigamiSheet
from .h4_kirigami import (
    H4KirigamiStack,
    KirigamiLayer,
    LayerPair,
    KirigamiCell,
    KirigamiMechanics,
    LayerColor,
    DeploymentState,
    CutPattern,
)

__all__ = [
    "TristableCell",
    "CellState",
    "KirigamiSheet",
    "H4KirigamiStack",
    "KirigamiLayer",
    "LayerPair",
    "KirigamiCell",
    "KirigamiMechanics",
    "LayerColor",
    "DeploymentState",
    "CutPattern",
]
