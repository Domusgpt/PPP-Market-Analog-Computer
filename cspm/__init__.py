"""
Cryptographically-Seeded Polytopal Modulation (CSPM) for Optical Networks

A physical-layer optical communication system that modulates signals onto
the vertices of a 600-cell polytope in 4D space, using:
- Polarization (Stokes parameters) for 3 dimensions
- Orbital Angular Momentum (OAM) mode for the 4th dimension

Key innovations:
1. Zero-overhead error correction via geometric vertex snapping
2. Physical-layer encryption via hash-chain lattice rotation
3. Topologically protected signal propagation
4. O(1) decoding latency vs O(n) for algebraic FEC

Copyright (c) 2025 Paul Phillips - Clear Seas Solutions LLC
"""

from .lattice import Cell600, PolychoralConstellation
from .transmitter import CSPMTransmitter, HashChainRotator
from .channel import OpticalChannel, FiberChannel, FreespaceChannel
from .receiver import CSPMReceiver, GeometricQuantizer
from .baseline import QAMModulator, QAMDemodulator
from .simulation import run_comparison, BERAnalysis

__version__ = "1.0.0"
__author__ = "Paul Phillips"

__all__ = [
    "Cell600",
    "PolychoralConstellation",
    "CSPMTransmitter",
    "HashChainRotator",
    "OpticalChannel",
    "FiberChannel",
    "FreespaceChannel",
    "CSPMReceiver",
    "GeometricQuantizer",
    "QAMModulator",
    "QAMDemodulator",
    "run_comparison",
    "BERAnalysis",
]
