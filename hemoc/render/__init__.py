"""
hemoc.render -- Rendering Layer
================================

Defines the renderer contract (interface), the dual-channel Galois
renderer, and a renderer validation test suite.

Modules
-------
renderer_contract       Abstract contract all renderers must satisfy
dual_channel_renderer   Galois-verified dual h_L/h_R rendering
renderer_test_suite     Validation functions for any renderer implementation
"""

from hemoc.render.renderer_contract import RendererContract, RenderResult
from hemoc.render.dual_channel_renderer import DualChannelGaloisRenderer

__all__ = [
    "RendererContract",
    "RenderResult",
    "DualChannelGaloisRenderer",
]
