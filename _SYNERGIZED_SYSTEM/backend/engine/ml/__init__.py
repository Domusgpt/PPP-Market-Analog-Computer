"""
ML Integration Module
=====================

Machine learning framework integration for the encoder.

Modules:
- torch_encoder: PyTorch nn.Module wrapper
- tf_encoder: TensorFlow/Keras layer
- auto_tune: Automatic parameter optimization
- datasets: Dataset utilities
"""

from .torch_encoder import TorchMoireEncoder, TorchMoireLayer
from .auto_tune import AutoTuner, TunerConfig
from .datasets import MoireDataset, create_dataloader

__all__ = [
    "TorchMoireEncoder",
    "TorchMoireLayer",
    "AutoTuner",
    "TunerConfig",
    "MoireDataset",
    "create_dataloader"
]
