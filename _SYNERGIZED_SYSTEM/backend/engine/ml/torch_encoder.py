"""
PyTorch Integration - Neural Network Wrapper
============================================

PyTorch nn.Module wrappers for the moiré encoder.
"""

import numpy as np
from typing import Optional, Tuple, Dict

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Dummy classes for type hints
    class nn:
        class Module:
            pass


if HAS_TORCH:

    class TorchMoireLayer(nn.Module):
        """
        Differentiable moiré pattern layer.

        Approximates moiré computation using differentiable operations
        for end-to-end training.

        Parameters
        ----------
        grid_size : Tuple[int, int]
            Output pattern size
        n_angles : int
            Number of angle channels

        Example
        -------
        >>> layer = TorchMoireLayer(grid_size=(64, 64))
        >>> pattern = layer(input_tensor)
        """

        def __init__(
            self,
            grid_size: Tuple[int, int] = (64, 64),
            n_angles: int = 5
        ):
            super().__init__()

            self.grid_size = grid_size
            self.n_angles = n_angles

            # Learnable wave vectors (approximating hexagonal lattice)
            self.wave_vectors = nn.Parameter(
                torch.randn(3, 2) * 0.1
            )

            # Learnable angle embeddings
            angles = torch.tensor([0.0, 7.34, 9.43, 13.17, 21.79]) * np.pi / 180
            self.register_buffer('base_angles', angles)

            # Create coordinate grid
            y = torch.linspace(-1, 1, grid_size[0])
            x = torch.linspace(-1, 1, grid_size[1])
            Y, X = torch.meshgrid(y, x, indexing='ij')
            self.register_buffer('X', X)
            self.register_buffer('Y', Y)

        def forward(
            self,
            x: torch.Tensor,
            angle_idx: int = 2
        ) -> torch.Tensor:
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor, shape (batch, 1, H, W) or (batch, H, W)
            angle_idx : int
                Which angle to use

            Returns
            -------
            torch.Tensor
                Moiré pattern, shape (batch, 1, grid_H, grid_W)
            """
            if x.dim() == 3:
                x = x.unsqueeze(1)

            batch_size = x.shape[0]

            # Resize input to grid size
            x = F.interpolate(x, size=self.grid_size, mode='bilinear', align_corners=False)

            # Get rotation angle
            theta = self.base_angles[angle_idx]

            # Rotation matrix
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)

            # Compute grating 1 (no rotation)
            grating1 = self._compute_grating(self.X, self.Y, 0.0)

            # Compute grating 2 (rotated)
            X_rot = self.X * cos_t - self.Y * sin_t
            Y_rot = self.X * sin_t + self.Y * cos_t
            grating2 = self._compute_grating(X_rot, Y_rot, 0.0)

            # Modulate with input
            grating1 = grating1 * (0.5 + 0.5 * x)
            grating2 = grating2 * (0.5 + 0.5 * (1 - x))

            # Moiré pattern (product)
            pattern = grating1 * grating2

            return pattern

        def _compute_grating(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            phase: float
        ) -> torch.Tensor:
            """Compute hexagonal grating pattern."""
            # Sum of three cosines at 120° intervals
            k = 2 * np.pi * 5  # Spatial frequency

            angles = torch.tensor([0, 2*np.pi/3, 4*np.pi/3], device=X.device)

            grating = torch.zeros_like(X)
            for angle in angles:
                kx = k * torch.cos(angle)
                ky = k * torch.sin(angle)
                grating = grating + torch.cos(kx * X + ky * Y + phase)

            # Normalize to [0, 1]
            grating = (grating / 3 + 1) / 2

            return grating


    class TorchMoireEncoder(nn.Module):
        """
        Full moiré encoder as PyTorch module.

        Combines cascade dynamics (approximated) with moiré computation
        for feature extraction.

        Parameters
        ----------
        grid_size : Tuple[int, int]
            Processing grid size
        n_cascade_layers : int
            Number of cascade convolution layers
        n_features : int
            Output feature dimension

        Example
        -------
        >>> encoder = TorchMoireEncoder()
        >>> features = encoder(images)  # (batch, n_features)
        """

        def __init__(
            self,
            grid_size: Tuple[int, int] = (64, 64),
            n_cascade_layers: int = 3,
            n_features: int = 128
        ):
            super().__init__()

            self.grid_size = grid_size

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(16)
            )

            # Cascade layers (approximate reservoir dynamics)
            cascade_layers = []
            for i in range(n_cascade_layers):
                cascade_layers.extend([
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.Tanh(),  # Approximate tristable with tanh
                    nn.BatchNorm2d(16)
                ])
            self.cascade = nn.Sequential(*cascade_layers)

            # Moiré layer
            self.moire = TorchMoireLayer(grid_size)

            # Feature extraction
            self.feature_conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )

            self.fc = nn.Sequential(
                nn.Linear(128 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_features)
            )

        def forward(
            self,
            x: torch.Tensor,
            return_pattern: bool = False
        ) -> torch.Tensor:
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input images, shape (batch, 1, H, W) or (batch, H, W)
            return_pattern : bool
                Also return moiré pattern

            Returns
            -------
            torch.Tensor
                Features (batch, n_features) or tuple with pattern
            """
            if x.dim() == 3:
                x = x.unsqueeze(1)

            # Resize to grid
            x = F.interpolate(x, size=self.grid_size, mode='bilinear', align_corners=False)

            # Input projection
            h = self.input_proj(x)

            # Cascade dynamics
            h = self.cascade(h)

            # Reduce to single channel (state)
            state = h.mean(dim=1, keepdim=True)

            # Generate moiré pattern
            pattern = self.moire(state)

            # Extract features
            feat = self.feature_conv(pattern)
            feat = feat.view(feat.size(0), -1)
            features = self.fc(feat)

            if return_pattern:
                return features, pattern

            return features

        def encode_batch(
            self,
            images: np.ndarray,
            device: str = 'cpu'
        ) -> np.ndarray:
            """
            Encode batch of numpy images.

            Parameters
            ----------
            images : np.ndarray
                Batch of images (N, H, W)
            device : str
                Torch device

            Returns
            -------
            np.ndarray
                Feature matrix (N, n_features)
            """
            self.eval()

            with torch.no_grad():
                x = torch.from_numpy(images).float().to(device)
                if x.dim() == 3:
                    x = x.unsqueeze(1)

                features = self(x)

                return features.cpu().numpy()


else:
    # Dummy implementations when PyTorch not available

    class TorchMoireLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not installed. Install with: pip install torch")

    class TorchMoireEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not installed. Install with: pip install torch")
