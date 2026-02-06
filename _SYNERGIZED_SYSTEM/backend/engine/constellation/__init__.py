"""
H4 Constellation Module - Distributed 600-cell hyper-computer.

This module implements the networking and coordination layer for
multiple 24-cell modules to form the complete 600-cell structure.

Key Components:
- ConstellationNode: Individual 24-cell prototype unit
- ConstellationNetwork: Inter-module communication and coordination
- PhasonPropagation: Strain wave propagation across the network
"""

from .h4_constellation import (
    ConstellationNode,
    ConstellationNetwork,
    VertexPort,
    PhasonPropagator,
    ConstellationState,
    NodePosition,
    H4ConstellationController,
)

__all__ = [
    "ConstellationNode",
    "ConstellationNetwork",
    "VertexPort",
    "PhasonPropagator",
    "ConstellationState",
    "NodePosition",
    "H4ConstellationController",
]
