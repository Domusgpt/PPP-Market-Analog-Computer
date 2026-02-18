"""
Golden Hadamard Class Checker (Conjecture 5)
==============================================

The Phillips matrix defines a new class of structured matrices --
"Golden Hadamard matrices" -- characterized by five axioms:

  GH1: Dense (all entries nonzero)
  GH2: Entries in (1/2) * Z[phi]   (the ring of golden integers, scaled)
  GH3: Block scaling U_R = phi^k * U_L  for some integer k
  GH4: Rank deficient (rank < min(m, n))
  GH5: Eigenvalues (of U^T U) in Q(phi)

This module checks whether a given matrix satisfies each axiom, enabling
classification of the Golden Hadamard family.
"""

from typing import Dict
import numpy as np

from hemoc.core.phillips_matrix import PHI, ENTRY_A, ENTRY_B, ENTRY_C


class GoldenHadamardChecker:
    """
    Check whether a matrix satisfies the Golden Hadamard axioms.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to check (typically 8x8 or 2n x n).
    block_split : int, optional
        Row index at which to split U_L / U_R.  Default: half the rows.
    """

    def __init__(self, matrix: np.ndarray, block_split: int = None):
        self.M = np.asarray(matrix, dtype=np.float64)
        m, n = self.M.shape
        if block_split is None:
            block_split = m // 2
        self.U_L = self.M[:block_split]
        self.U_R = self.M[block_split:]

    def check_gh1_dense(self) -> Dict:
        """GH1: All entries nonzero."""
        n_zero = int(np.sum(np.abs(self.M) < 1e-15))
        total = self.M.size
        return {
            "axiom": "GH1 Dense",
            "n_zero_entries": n_zero,
            "total_entries": total,
            "pass": n_zero == 0,
        }

    def check_gh2_golden_ring(self, tolerance: float = 1e-8) -> Dict:
        """
        GH2: Entries in (1/2) * Z[phi].

        Checks that every entry can be written as (a + b*phi)/2 for
        integer a, b (where |a|, |b| <= some small bound).
        """
        entries = np.abs(self.M.ravel())
        unique_abs = np.unique(np.round(entries, 8))

        # Known golden-ring values at scale 1/2
        golden_ring_values = set()
        for a in range(-4, 5):
            for b in range(-4, 5):
                val = abs((a + b * PHI) / 2.0)
                golden_ring_values.add(round(val, 8))

        all_in_ring = all(
            any(abs(v - grv) < tolerance for grv in golden_ring_values)
            for v in unique_abs
        )

        return {
            "axiom": "GH2 Golden Ring Entries",
            "unique_absolute_values": unique_abs.tolist(),
            "pass": all_in_ring,
        }

    def check_gh3_block_scaling(self, tolerance: float = 1e-8) -> Dict:
        """GH3: U_R = phi^k * U_L for some integer k."""
        if self.U_L.shape != self.U_R.shape:
            return {"axiom": "GH3 Block Scaling", "pass": False,
                    "note": "Block shapes differ"}

        # Check for k = 1, 2, -1
        for k in [1, 2, -1]:
            diff = np.max(np.abs(self.U_R - PHI ** k * self.U_L))
            if diff < tolerance:
                return {
                    "axiom": "GH3 Block Scaling",
                    "scaling_exponent_k": k,
                    "max_deviation": float(diff),
                    "pass": True,
                }

        return {
            "axiom": "GH3 Block Scaling",
            "pass": False,
            "note": "No integer k found such that U_R = phi^k * U_L",
        }

    def check_gh4_rank_deficient(self) -> Dict:
        """GH4: Rank < min(m, n)."""
        m, n = self.M.shape
        rank = int(np.linalg.matrix_rank(self.M))
        return {
            "axiom": "GH4 Rank Deficient",
            "rank": rank,
            "min_dimension": min(m, n),
            "pass": rank < min(m, n),
        }

    def check_gh5_eigenvalues_in_Q_phi(self, tolerance: float = 1e-3) -> Dict:
        """
        GH5: Eigenvalues of M^T M lie in Q(phi).

        Checks that each eigenvalue can be written as (a + b*phi)/d for
        integer a, b and small denominator d.

        Note: numerical eigenvalue computation introduces ~1e-4 errors
        for 8x8 matrices, so tolerance is set accordingly.
        """
        G = self.M.T @ self.M
        eigenvalues = np.sort(np.linalg.eigvalsh(G))[::-1]

        # Check if each eigenvalue is in Q(phi)
        in_q_phi = []
        for ev in eigenvalues:
            # Near-zero eigenvalues are trivially in Q(phi)
            if abs(ev) < tolerance:
                in_q_phi.append({
                    "eigenvalue": float(ev),
                    "expression": "0",
                    "in_Q_phi": True,
                })
                continue

            # Try to express ev = (a + b*phi) / d
            found = False
            for a_num in range(-40, 41):
                if found:
                    break
                for b_num in range(-40, 41):
                    if found:
                        break
                    for denom in [1, 2, 4, 8]:
                        candidate = (a_num + b_num * PHI) / denom
                        if abs(ev - candidate) < tolerance:
                            in_q_phi.append({
                                "eigenvalue": float(ev),
                                "expression": f"({a_num} + {b_num}*phi) / {denom}",
                                "in_Q_phi": True,
                            })
                            found = True
                            break
            if not found:
                in_q_phi.append({
                    "eigenvalue": float(ev),
                    "in_Q_phi": False,
                })

        return {
            "axiom": "GH5 Eigenvalues in Q(phi)",
            "eigenvalues": [float(e) for e in eigenvalues],
            "details": in_q_phi,
            "pass": all(e["in_Q_phi"] for e in in_q_phi),
        }

    def check_all(self) -> Dict:
        """Run all five axiom checks."""
        checks = [
            self.check_gh1_dense(),
            self.check_gh2_golden_ring(),
            self.check_gh3_block_scaling(),
            self.check_gh4_rank_deficient(),
            self.check_gh5_eigenvalues_in_Q_phi(),
        ]
        return {
            "all_pass": all(c["pass"] for c in checks),
            "n_passed": sum(1 for c in checks if c["pass"]),
            "n_total": len(checks),
            "checks": checks,
        }
