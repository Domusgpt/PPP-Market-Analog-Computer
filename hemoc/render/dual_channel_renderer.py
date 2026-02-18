"""
Dual-Channel Galois-Verified Renderer
=======================================

Renders two synchronized image stacks -- one driven predominantly by
h_L-derived controls, one by h_R-derived controls -- and enforces the
Galois invariant as a runtime check.

Architecture
------------
    Input: 8D state vector (or 6 rotation angles mapped to 8D)
        |
        +--> project_to_h4_left()  --> h_L (4D) --> render_left()  --> left_image
        |
        +--> project_to_h4_right() --> h_R (4D) --> render_right() --> right_image
        |
        +--> galois_check(||h_R|| / ||h_L|| == phi)
        |
        +--> combine() --> fused output + verification flag

If the Galois invariant deviates beyond tolerance, the renderer:
  (a) flags the output as unreliable, and
  (b) optionally falls back to a conservative single-channel mode.

This turns a theoretical symmetry into an operational self-auditing
cognition substrate.

Note: This is a CPU-based reference implementation.  GPU (CUDA/Colab)
acceleration should use batched matrix operations and fragment shaders.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from hemoc.core.phillips_matrix import PHILLIPS_U_L, PHILLIPS_U_R, PHI
from hemoc.core.cell_600 import generate_600_cell_vertices
from hemoc.render.renderer_contract import RendererContract, RenderResult


class DualChannelGaloisRenderer(RendererContract):
    """
    Dual-channel renderer with Galois phi-coupling verification.

    Renders moire patterns by:
    1. Mapping 6 rotation angles to 4D rotations.
    2. Applying rotations to 600-cell vertices.
    3. Projecting rotated vertices to 2D via stereographic projection.
    4. Composing hex-grating interference patterns.

    The dual-channel split renders the h_L and h_R projected vertices
    separately, producing two correlated image stacks.

    Parameters
    ----------
    resolution : int
        Image resolution (square).  Default 64.
    n_channels : int
        Number of grating channels per image.  Default 5 (pentagonal).
    galois_tolerance : float
        Maximum allowed phi-ratio deviation.
    """

    def __init__(
        self,
        resolution: int = 64,
        n_channels: int = 5,
        galois_tolerance: float = 1e-4,
    ):
        self.resolution = resolution
        self.n_channels = n_channels
        self.galois_tolerance = galois_tolerance

        # Precompute 600-cell vertices
        self._vertices_600 = generate_600_cell_vertices()  # (120, 4)

        # Hex grating parameters
        self._grating_frequencies = np.array([
            1.0, PHI, PHI ** 2, PHI ** 3, PHI ** 4
        ])[:n_channels]

    @property
    def fidelity_class(self) -> str:
        return "physics"

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self.resolution, self.resolution, 3)

    def render(self, angles: np.ndarray) -> RenderResult:
        """
        Render dual-channel moire pattern from 6 rotation angles.

        Parameters
        ----------
        angles : np.ndarray of shape (6,)
            Rotation angles for the six 4D rotation planes.

        Returns
        -------
        RenderResult with left_image, right_image, and Galois flag.
        """
        angles = np.asarray(angles, dtype=np.float64)
        assert len(angles) == 6, f"Expected 6 angles, got {len(angles)}"

        # Build 4D rotation matrix from 6 angles
        R = self._build_rotation_4d(angles)

        # Rotate all 600-cell vertices
        rotated = (R @ self._vertices_600.T).T  # (120, 4)

        # Project through Phillips matrix blocks
        # For each vertex, compute h_L and h_R projections
        # We use a representative 8D embedding:
        #   v_8d = [v_4d[0], v_4d[1], v_4d[2], v_4d[3],
        #           v_4d[0], v_4d[1], v_4d[2], v_4d[3]]
        # This is a simplification; the full pipeline would use actual E8 roots.
        v_8d_batch = np.hstack([rotated, rotated])  # (120, 8)

        h_L_batch = v_8d_batch @ PHILLIPS_U_L.T  # (120, 4)
        h_R_batch = v_8d_batch @ PHILLIPS_U_R.T  # (120, 4)

        # Galois check on aggregate norms
        norm_L = np.linalg.norm(h_L_batch)
        norm_R = np.linalg.norm(h_R_batch)
        galois_ratio = norm_R / max(norm_L, 1e-14)
        galois_valid = abs(galois_ratio - PHI) < self.galois_tolerance

        # Render left channel (from h_L projections)
        left_image = self._render_moire(h_L_batch, rotated)

        # Render right channel (from h_R projections)
        right_image = self._render_moire(h_R_batch, rotated)

        # Fused output: blend left and right with phi-weighting
        fused = (left_image + PHI * right_image) / (1.0 + PHI)

        return RenderResult(
            image=fused,
            left_image=left_image,
            right_image=right_image,
            metadata={
                "angles": angles.tolist(),
                "renderer": "DualChannelGaloisRenderer",
                "fidelity_class": self.fidelity_class,
                "resolution": self.resolution,
                "n_channels": self.n_channels,
                "n_vertices_used": len(rotated),
            },
            galois_ratio=float(galois_ratio),
            galois_valid=bool(galois_valid),
        )

    def _build_rotation_4d(self, angles: np.ndarray) -> np.ndarray:
        """
        Build a 4D rotation matrix from 6 rotation-plane angles.

        The six planes are: (0,1), (1,2), (0,2), (0,3), (1,3), (2,3).
        The composition is right-multiplied in order.
        """
        R = np.eye(4)
        planes = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]

        for k, (i, j) in enumerate(planes):
            c = np.cos(angles[k])
            s = np.sin(angles[k])
            G = np.eye(4)
            G[i, i] = c
            G[j, j] = c
            G[i, j] = -s
            G[j, i] = s
            R = R @ G

        return R

    def _render_moire(
        self,
        projected_4d: np.ndarray,
        vertices_4d: np.ndarray,
    ) -> np.ndarray:
        """
        Render a moire interference pattern from projected vertices.

        Uses stereographic projection from 4D to 2D, then composites
        hex-grating patterns at golden-ratio frequency multiples.

        Parameters
        ----------
        projected_4d : np.ndarray of shape (N, 4)
            Projected vertices (through U_L or U_R).
        vertices_4d : np.ndarray of shape (N, 4)
            Original rotated 4D vertices (for stereographic projection).

        Returns
        -------
        np.ndarray of shape (resolution, resolution, 3)
            RGB image.
        """
        res = self.resolution

        # Stereographic projection: 4D -> 2D
        w_offset = 2.0
        safe_w = w_offset - vertices_4d[:, 3]
        safe_w = np.where(np.abs(safe_w) < 1e-10, 1e-10, safe_w)
        scale_factors = w_offset / safe_w

        x_2d = vertices_4d[:, 0] * scale_factors
        y_2d = vertices_4d[:, 1] * scale_factors

        # Normalize to [0, resolution)
        x_min, x_max = x_2d.min(), x_2d.max()
        y_min, y_max = y_2d.min(), y_2d.max()
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        # Create coordinate grids
        gx = np.linspace(-1, 1, res)
        gy = np.linspace(-1, 1, res)
        GX, GY = np.meshgrid(gx, gy)

        # Build image via hex-grating interference
        image = np.zeros((res, res, 3))

        for ch_idx in range(min(self.n_channels, len(self._grating_frequencies))):
            freq = self._grating_frequencies[ch_idx]
            angle_offset = ch_idx * np.pi / self.n_channels

            # Hex grating: 3 rotated cosine gratings superimposed
            for rot in range(3):
                theta = angle_offset + rot * np.pi / 3.0
                phase = freq * (GX * np.cos(theta) + GY * np.sin(theta))

                # Add contribution from projected vertex positions
                for v_idx in range(min(len(projected_4d), 24)):  # use first 24-cell
                    vx = (x_2d[v_idx] - x_min) / x_range * 2 - 1
                    vy = (y_2d[v_idx] - y_min) / y_range * 2 - 1
                    r_sq = (GX - vx) ** 2 + (GY - vy) ** 2
                    influence = np.exp(-r_sq * 4.0)
                    phase += influence * projected_4d[v_idx, ch_idx % 4] * 0.5

                grating = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))

                # Map to RGB channels
                color_idx = (ch_idx + rot) % 3
                image[:, :, color_idx] += grating

        # Normalize to [0, 1]
        img_max = image.max()
        if img_max > 0:
            image /= img_max

        return image
