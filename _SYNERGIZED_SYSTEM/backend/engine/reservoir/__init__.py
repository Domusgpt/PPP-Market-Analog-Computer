"""
Advanced Reservoir Module
=========================

Sophisticated reservoir computing implementations.

Modules:
- multiscale: Multi-timescale reservoir
- learnable: Trainable stiffness maps
- esn: Echo State Network formulation
- criticality: Criticality metrics and control
"""

from .multiscale import MultiScaleReservoir
from .learnable import LearnableReservoir
from .esn import EchoStateReservoir
from .criticality import CriticalityAnalyzer
from .readout import ReservoirReadout, ReadoutConfig, MoireFeatureExtractor

__all__ = [
    "MultiScaleReservoir",
    "LearnableReservoir",
    "EchoStateReservoir",
    "CriticalityAnalyzer",
    "ReservoirReadout",
    "ReadoutConfig",
    "MoireFeatureExtractor",
]
