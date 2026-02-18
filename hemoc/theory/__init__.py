"""
hemoc.theory -- Verification, Invariant Testing, and Conjecture Harness
========================================================================

Modules
-------
invariant_verifier    All Phillips matrix theorems + perturbation fuzz harness
galois_verifier       Dual-channel phi-coupled error detection
golden_hadamard       Golden Hadamard class axiom checker (Conjecture 5)
"""

from hemoc.theory.invariant_verifier import PhillipsInvariantVerifier
from hemoc.theory.galois_verifier import GaloisDualVerifier
from hemoc.theory.golden_hadamard import GoldenHadamardChecker

__all__ = [
    "PhillipsInvariantVerifier",
    "GaloisDualVerifier",
    "GoldenHadamardChecker",
]
