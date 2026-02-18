"""
The Phillips 8x8 Projection Operator
======================================

Defines the Phillips matrix and all verified algebraic identities.

The Phillips matrix is an 8x8 dense, non-symmetric, rank-4 projection
operator for the E8 -> H4_L + H4_R decomposition.  It was constructed by
Paul Phillips for hyper-dimensional audio analysis and quaternion system
needs.  The golden-ratio block structure (U_R = phi * U_L) is a DISCOVERED
property of the projection geometry, not a design constraint.

Constants
---------
    a = 1/2                     = 0.500
    b = (phi - 1) / 2 = 1/(2phi) ~ 0.309
    c = phi / 2                 ~ 0.809

The entry set {a, b, c} forms a geometric progression with common ratio phi:
    b = a / phi,    c = a * phi.

Block structure
---------------
    U_L (rows 0-3): entries from {+-a, +-b}, row norm^2 = 3 - phi
    U_R (rows 4-7): entries from {+-a, +-c}, row norm^2 = phi + 2
    Fundamental identity: U_R = phi * U_L  (discovered, not designed)

Verified theorems (see hemoc.theory.invariant_verifier for proofs)
------------------------------------------------------------------
    Theorem 4.1 (Column Trichotomy):
        Column norms^2 follow the 2-4-2 pattern:
        dims {0,4} = phi+2,  dims {1,2,5,6} = 2.5,  dims {3,7} = 3-phi.

    Theorem 5.1 (Pentagonal Row Norms):
        sqrt(3-phi) = 2 * sin(36 deg)  ~ 1.17557.

    Frobenius identity:
        ||U||_F^2 = 20   (matches 600-cell vertex valence).

    phi-scaling:
        For ANY 8D vector x:  ||U_R x|| / ||U_L x|| = phi.

    sqrt(5)-coupling:
        Row norms satisfy: sqrt(3-phi) * sqrt(phi+2) = sqrt(5).

    Shell coincidence:
        phi * sqrt(3-phi) = sqrt(phi+2)  ~ 1.90211.

References
----------
    Phillips, "The Totalistic Geometry of E8" (2026, in preparation).
    Boyle & Steinhardt, arXiv:1608.08215 (Coxeter pairs).
"""

import numpy as np

# =============================================================================
# Fundamental constants
# =============================================================================

PHI = (1.0 + np.sqrt(5.0)) / 2.0          # Golden ratio  ~ 1.61803
PHI_INV = PHI - 1.0                        # 1/phi         ~ 0.61803
PLASTIC_RATIO = float(np.real(             # rho           ~ 1.32472
    np.roots([1, 0, -1, -1])[0]
))

# Phillips entry constants
ENTRY_A = 0.5                              # a  = 1/2
ENTRY_B = (PHI - 1.0) / 2.0               # b  = 1/(2phi)  ~ 0.30902
ENTRY_C = PHI / 2.0                        # c  = phi/2     ~ 0.80902

# Verify geometric progression: b, a, c with ratio phi
assert abs(ENTRY_A / ENTRY_B - PHI) < 1e-12, "a/b != phi"
assert abs(ENTRY_C / ENTRY_A - PHI) < 1e-12, "c/a != phi"

# =============================================================================
# The Phillips Matrix (8 x 8)
# =============================================================================

_a, _b, _c = ENTRY_A, ENTRY_B, ENTRY_C

PHILLIPS_MATRIX = np.array([
    # --- U_L block (rows 0-3): contracted, entries {+-a, +-b} ---
    [ _a,  _b,  _a,  _b,  _a, -_b,  _a, -_b],
    [ _a,  _a, -_b, -_b, -_a, -_a,  _b,  _b],
    [ _a, -_b, -_a,  _b,  _a, -_b, -_a,  _b],
    [ _a, -_a,  _b, -_b, -_a,  _a, -_b,  _b],
    # --- U_R block (rows 4-7): expanded, entries {+-a, +-c} ---
    [ _c,  _a,  _c,  _a,  _c, -_a,  _c, -_a],
    [ _c,  _c, -_a, -_a, -_c, -_c,  _a,  _a],
    [ _c, -_a, -_c,  _a,  _c, -_a, -_c,  _a],
    [ _c, -_c,  _a, -_a, -_c,  _c, -_a,  _a],
], dtype=np.float64)

PHILLIPS_U_L = PHILLIPS_MATRIX[:4]   # (4, 8) -- left H4 projection
PHILLIPS_U_R = PHILLIPS_MATRIX[4:]   # (4, 8) -- right H4 projection

# =============================================================================
# Column Trichotomy  (2-4-2 pattern)
# =============================================================================

COLUMN_TRICHOTOMY = {
    "expanded":   [0, 4],         # norm^2 = phi + 2
    "stable":     [1, 2, 5, 6],   # norm^2 = 2.5
    "contracted": [3, 7],         # norm^2 = 3 - phi
}

# =============================================================================
# Precomputed theorem values  (used by verifier, exported for convenience)
# =============================================================================

ROW_NORM_SQ_LEFT   = 3.0 - PHI            # ~ 1.38197
ROW_NORM_SQ_RIGHT  = PHI + 2.0            # ~ 3.61803
FROBENIUS_NORM_SQ  = 20.0
SQRT5              = np.sqrt(5.0)
SHELL_COINCIDENCE  = PHI * np.sqrt(3.0 - PHI)   # = sqrt(phi+2)  ~ 1.90211

# Coxeter angle interpretation of entry values
COXETER_ANGLES = {
    "b": {"value": ENTRY_B, "angle_deg": 72.0, "cos_check": np.cos(np.radians(72.0))},
    "a": {"value": ENTRY_A, "angle_deg": 60.0, "cos_check": np.cos(np.radians(60.0))},
    "c": {"value": ENTRY_C, "angle_deg": 36.0, "cos_check": np.cos(np.radians(36.0))},
}
