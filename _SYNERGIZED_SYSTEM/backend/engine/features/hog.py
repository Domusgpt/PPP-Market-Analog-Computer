"""
HOG Descriptor - Histogram of Oriented Gradients
================================================

Edge-based feature extraction using oriented gradient histograms.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class HOGResult:
    """Result from HOG computation."""
    features: np.ndarray         # Final feature vector
    cell_histograms: np.ndarray  # Per-cell histograms
    gradient_magnitude: np.ndarray
    gradient_orientation: np.ndarray


class HOGDescriptor:
    """
    Histogram of Oriented Gradients descriptor.

    Computes edge orientation histograms in spatial cells,
    then normalizes across blocks for illumination invariance.

    Parameters
    ----------
    cell_size : Tuple[int, int]
        Size of histogram cells in pixels
    block_size : Tuple[int, int]
        Size of normalization blocks in cells
    n_bins : int
        Number of orientation bins
    signed : bool
        Use signed (0-360°) or unsigned (0-180°) gradients

    Example
    -------
    >>> hog = HOGDescriptor(cell_size=(8, 8), n_bins=9)
    >>> result = hog.compute(pattern)
    >>> features = result.features
    """

    def __init__(
        self,
        cell_size: Tuple[int, int] = (8, 8),
        block_size: Tuple[int, int] = (2, 2),
        n_bins: int = 9,
        signed: bool = False
    ):
        self.cell_size = cell_size
        self.block_size = block_size
        self.n_bins = n_bins
        self.signed = signed

        # Angle range
        self.angle_range = 2 * np.pi if signed else np.pi
        self.bin_width = self.angle_range / n_bins

    def _compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient magnitude and orientation."""
        # Sobel-like gradients
        gx = np.zeros_like(image)
        gy = np.zeros_like(image)

        gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gy[1:-1, :] = image[2:, :] - image[:-2, :]

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)

        # Map to positive range
        if not self.signed:
            orientation = orientation % np.pi

        return magnitude, orientation

    def _compute_cell_histogram(
        self,
        magnitude: np.ndarray,
        orientation: np.ndarray,
        cell_y: int,
        cell_x: int
    ) -> np.ndarray:
        """Compute histogram for a single cell."""
        cy, cx = self.cell_size
        y_start, y_end = cell_y * cy, (cell_y + 1) * cy
        x_start, x_end = cell_x * cx, (cell_x + 1) * cx

        cell_mag = magnitude[y_start:y_end, x_start:x_end]
        cell_ori = orientation[y_start:y_end, x_start:x_end]

        # Compute histogram with soft binning
        histogram = np.zeros(self.n_bins)

        for i in range(cell_mag.shape[0]):
            for j in range(cell_mag.shape[1]):
                angle = cell_ori[i, j]
                mag = cell_mag[i, j]

                # Find bin
                bin_idx = angle / self.bin_width
                lower_bin = int(bin_idx) % self.n_bins
                upper_bin = (lower_bin + 1) % self.n_bins

                # Soft assignment
                upper_weight = bin_idx - int(bin_idx)
                lower_weight = 1.0 - upper_weight

                histogram[lower_bin] += mag * lower_weight
                histogram[upper_bin] += mag * upper_weight

        return histogram

    def compute(self, image: np.ndarray) -> HOGResult:
        """
        Compute HOG descriptor for image.

        Parameters
        ----------
        image : np.ndarray
            2D input image

        Returns
        -------
        HOGResult
            HOG features and intermediate results
        """
        # Compute gradients
        magnitude, orientation = self._compute_gradients(image)

        # Number of cells
        n_cells_y = image.shape[0] // self.cell_size[0]
        n_cells_x = image.shape[1] // self.cell_size[1]

        # Compute cell histograms
        cell_histograms = np.zeros((n_cells_y, n_cells_x, self.n_bins))

        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                cell_histograms[cy, cx] = self._compute_cell_histogram(
                    magnitude, orientation, cy, cx
                )

        # Block normalization
        by, bx = self.block_size
        n_blocks_y = n_cells_y - by + 1
        n_blocks_x = n_cells_x - bx + 1

        features = []

        for block_y in range(max(1, n_blocks_y)):
            for block_x in range(max(1, n_blocks_x)):
                # Get block cells
                block = cell_histograms[
                    block_y:block_y + by,
                    block_x:block_x + bx
                ].flatten()

                # L2 normalization
                norm = np.sqrt(np.sum(block**2) + 1e-6)
                block = block / norm

                features.extend(block)

        return HOGResult(
            features=np.array(features, dtype=np.float32),
            cell_histograms=cell_histograms,
            gradient_magnitude=magnitude,
            gradient_orientation=orientation
        )

    def n_features_for_size(self, image_size: Tuple[int, int]) -> int:
        """Calculate number of features for given image size."""
        n_cells_y = image_size[0] // self.cell_size[0]
        n_cells_x = image_size[1] // self.cell_size[1]
        by, bx = self.block_size
        n_blocks_y = max(1, n_cells_y - by + 1)
        n_blocks_x = max(1, n_cells_x - bx + 1)
        return n_blocks_y * n_blocks_x * by * bx * self.n_bins


def extract_hog_features(
    image: np.ndarray,
    cell_size: Tuple[int, int] = (8, 8),
    n_bins: int = 9
) -> np.ndarray:
    """
    Quick function to extract HOG features.

    Parameters
    ----------
    image : np.ndarray
        2D input image
    cell_size : Tuple[int, int]
        Cell size in pixels
    n_bins : int
        Number of orientation bins

    Returns
    -------
    np.ndarray
        HOG feature vector
    """
    hog = HOGDescriptor(cell_size=cell_size, n_bins=n_bins)
    result = hog.compute(image)
    return result.features
