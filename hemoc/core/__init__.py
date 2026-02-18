"""
hemoc.core -- Mathematical Primitives (The Canonical Math Spine)
================================================================

This is the single source of truth for all mathematical objects in the
HEMOC/PPP system.  Every renderer, encoder, experiment, and verification
routine imports from here.

Modules
-------
e8_roots          E8 root system (240 roots in 8D)
phillips_matrix   The Phillips 8x8 projection operator and its theorems
h4_decomposition  H4_L + H4_R decomposition, 24-cell, trilatic split
cell_600          600-cell vertices and the true 5x24-cell partition
kernel_basis      Null space of the Phillips matrix, collision direction,
                  clean phason channels
"""

from hemoc.core.e8_roots import generate_e8_roots, E8Root, E8RootType
from hemoc.core.phillips_matrix import (
    PHILLIPS_MATRIX,
    PHILLIPS_U_L,
    PHILLIPS_U_R,
    PHI,
    PHI_INV,
    PLASTIC_RATIO,
    ENTRY_A,
    ENTRY_B,
    ENTRY_C,
    COLUMN_TRICHOTOMY,
)
from hemoc.core.h4_decomposition import (
    project_to_h4_left,
    project_to_h4_right,
    project_to_h4_full,
    project_e8_phillips,
)
from hemoc.core.cell_600 import generate_600_cell_vertices, partition_into_24_cells
from hemoc.core.kernel_basis import (
    compute_kernel_basis,
    COLLISION_DIRECTION,
    compute_clean_kernel_directions,
)

__all__ = [
    "generate_e8_roots", "E8Root", "E8RootType",
    "PHILLIPS_MATRIX", "PHILLIPS_U_L", "PHILLIPS_U_R",
    "PHI", "PHI_INV", "PLASTIC_RATIO",
    "ENTRY_A", "ENTRY_B", "ENTRY_C",
    "COLUMN_TRICHOTOMY",
    "project_to_h4_left", "project_to_h4_right", "project_to_h4_full",
    "project_e8_phillips",
    "generate_600_cell_vertices", "partition_into_24_cells",
    "compute_kernel_basis", "COLLISION_DIRECTION",
    "compute_clean_kernel_directions",
]
