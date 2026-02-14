"""
Reservoir Readout Layer - Linear Classification from Moiré Patterns
===================================================================

Implements the readout layer for Physical Reservoir Computing as
described in Section 7.2.

The moiré pattern integrates the micro-state of the kirigami reservoir
into a macro-state image. A simple linear classifier (readout weights)
interprets these patterns for cognition.

Key principle: The reservoir does the hard work (nonlinear mixing,
temporal memory). The readout layer just needs to be linear.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
import warnings


@dataclass
class ReadoutConfig:
    """Configuration for readout layer."""
    n_features: int = 256       # Number of readout features
    n_outputs: int = 10         # Number of output classes
    regularization: float = 1e-4  # Ridge regression regularization
    use_bias: bool = True


@dataclass
class ReadoutResult:
    """Result of readout computation."""
    features: np.ndarray        # Extracted feature vector
    logits: np.ndarray          # Raw output scores
    prediction: int             # Argmax prediction
    confidence: float           # Softmax confidence


class ReservoirReadout:
    """
    Linear readout layer for kirigami reservoir.

    This implements the "readout weights" that interpret moiré patterns.
    Training uses ridge regression (closed-form solution) for speed.

    From Section 7.2: The "cognition" is achieved by training a simple
    linear classifier to interpret moiré fringes.

    Parameters
    ----------
    config : ReadoutConfig
        Configuration parameters
    feature_extractor : Callable, optional
        Function to extract features from moiré pattern
    """

    def __init__(
        self,
        config: Optional[ReadoutConfig] = None,
        feature_extractor: Optional[Callable] = None
    ):
        self.config = config or ReadoutConfig()
        self.feature_extractor = feature_extractor or self._default_extractor

        # Weights (initialized on first train)
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None

        # Training statistics
        self._is_trained = False
        self._train_accuracy = 0.0
        self._n_samples_seen = 0

    def _default_extractor(self, pattern: np.ndarray) -> np.ndarray:
        """
        Default feature extraction from moiré pattern.

        Extracts:
        - Downsampled intensity values
        - Local statistics (mean, std in patches)
        - Frequency content (DCT coefficients)
        """
        # Flatten and downsample if needed
        flat = pattern.flatten()

        # Target feature size
        target_size = self.config.n_features

        if len(flat) >= target_size:
            # Uniform sampling
            indices = np.linspace(0, len(flat) - 1, target_size, dtype=int)
            features = flat[indices]
        else:
            # Pad with zeros
            features = np.zeros(target_size)
            features[:len(flat)] = flat

        # Add global statistics
        n_stats = min(8, target_size // 4)
        if n_stats > 0:
            features[-n_stats:] = [
                np.mean(pattern),
                np.std(pattern),
                np.min(pattern),
                np.max(pattern),
                np.median(pattern),
                np.percentile(pattern, 25),
                np.percentile(pattern, 75),
                np.sum(np.abs(np.diff(pattern.flatten())))  # Total variation
            ][:n_stats]

        return features.astype(np.float32)

    def extract_features(self, pattern: np.ndarray) -> np.ndarray:
        """
        Extract features from a moiré pattern.

        Parameters
        ----------
        pattern : np.ndarray
            2D moiré intensity pattern

        Returns
        -------
        np.ndarray
            1D feature vector
        """
        return self.feature_extractor(pattern)

    def train(
        self,
        patterns: List[np.ndarray],
        labels: np.ndarray,
        verbose: bool = False
    ) -> float:
        """
        Train readout weights using ridge regression.

        Uses closed-form solution: W = (X^T X + λI)^{-1} X^T Y

        Parameters
        ----------
        patterns : List[np.ndarray]
            List of moiré patterns
        labels : np.ndarray
            Integer class labels (0 to n_outputs-1)
        verbose : bool
            Print training progress

        Returns
        -------
        float
            Training accuracy
        """
        n_samples = len(patterns)
        if n_samples == 0:
            raise ValueError("No training samples provided")

        # Extract features
        if verbose:
            print(f"Extracting features from {n_samples} patterns...")

        X = np.zeros((n_samples, self.config.n_features))
        for i, pattern in enumerate(patterns):
            X[i] = self.extract_features(pattern)

        # Add bias column
        if self.config.use_bias:
            X = np.hstack([X, np.ones((n_samples, 1))])

        # One-hot encode labels
        Y = np.zeros((n_samples, self.config.n_outputs))
        for i, label in enumerate(labels):
            if 0 <= label < self.config.n_outputs:
                Y[i, label] = 1.0
            else:
                warnings.warn(f"Label {label} out of range [0, {self.config.n_outputs})")

        # Ridge regression: W = (X^T X + λI)^{-1} X^T Y
        if verbose:
            print("Computing ridge regression solution...")

        λ = self.config.regularization
        n_cols = X.shape[1]

        XTX = X.T @ X
        XTY = X.T @ Y
        reg = λ * np.eye(n_cols)

        # Solve system
        try:
            W_full = np.linalg.solve(XTX + reg, XTY)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            W_full = np.linalg.lstsq(XTX + reg, XTY, rcond=None)[0]

        # Split weights and bias
        if self.config.use_bias:
            self.W = W_full[:-1, :]
            self.b = W_full[-1, :]
        else:
            self.W = W_full
            self.b = np.zeros(self.config.n_outputs)

        # Mark as trained before computing accuracy (predict requires it)
        self._is_trained = True

        # Compute training accuracy
        predictions = self.predict_batch(patterns)
        accuracy = np.mean(predictions == labels)
        self._train_accuracy = accuracy
        self._n_samples_seen = n_samples

        if verbose:
            print(f"Training complete. Accuracy: {accuracy:.2%}")

        return accuracy

    def predict(self, pattern: np.ndarray) -> ReadoutResult:
        """
        Predict class from moiré pattern.

        Parameters
        ----------
        pattern : np.ndarray
            2D moiré intensity pattern

        Returns
        -------
        ReadoutResult
            Prediction with features, logits, class, and confidence
        """
        if not self._is_trained:
            raise RuntimeError("Readout layer not trained. Call train() first.")

        # Extract features
        features = self.extract_features(pattern)

        # Compute logits
        logits = features @ self.W + self.b

        # Softmax for confidence
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        prediction = int(np.argmax(logits))
        confidence = float(probs[prediction])

        return ReadoutResult(
            features=features,
            logits=logits,
            prediction=prediction,
            confidence=confidence
        )

    def predict_batch(self, patterns: List[np.ndarray]) -> np.ndarray:
        """
        Predict classes for multiple patterns.

        Returns
        -------
        np.ndarray
            Array of predicted class indices
        """
        predictions = np.zeros(len(patterns), dtype=int)
        for i, pattern in enumerate(patterns):
            result = self.predict(pattern)
            predictions[i] = result.prediction
        return predictions

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get trained weights and bias."""
        if not self._is_trained:
            raise RuntimeError("Readout layer not trained.")
        return self.W.copy(), self.b.copy()

    def set_weights(self, W: np.ndarray, b: np.ndarray):
        """Set weights directly (for loading pre-trained models)."""
        self.W = W.copy()
        self.b = b.copy()
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def train_accuracy(self) -> float:
        return self._train_accuracy

    def reset(self):
        """Reset to untrained state."""
        self.W = None
        self.b = None
        self._is_trained = False
        self._train_accuracy = 0.0
        self._n_samples_seen = 0

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (f"ReservoirReadout({status}, features={self.config.n_features}, "
                f"outputs={self.config.n_outputs})")


class MoireFeatureExtractor:
    """
    Feature extraction from moiré patterns using Gabor, spectral, and HOG.

    Combines three complementary feature extraction strategies:
    - Gabor filter bank: orientation and spatial frequency selectivity
    - Spectral analysis: FFT-based radial/angular frequency bins
    - HOG: gradient orientation histograms for shape structure

    Plus: global statistics, patch grid, and fork detection.
    """

    def __init__(self, grid_size: Tuple[int, int] = (64, 64)):
        self.grid_size = grid_size

        # Initialize feature extractors (lazy — imported on first use)
        self._gabor = None
        self._spectral = None
        self._hog = None
        self._n_features = None

    def _ensure_extractors(self):
        """Lazy initialization of feature extraction modules."""
        if self._gabor is not None:
            return
        from ..features.gabor import GaborFilterBank
        from ..features.spectral import SpectralAnalyzer
        from ..features.hog import HOGDescriptor

        self._gabor = GaborFilterBank(n_orientations=6, n_scales=3)
        self._spectral = SpectralAnalyzer(n_radial_bins=8, n_angular_bins=6)
        self._hog = HOGDescriptor(cell_size=(8, 8), n_bins=9)

    def extract(self, pattern: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive features from moiré pattern.

        Parameters
        ----------
        pattern : np.ndarray
            2D or 3D (RGB) moiré pattern

        Returns
        -------
        np.ndarray
            Feature vector combining Gabor + spectral + HOG + statistics
        """
        self._ensure_extractors()
        features = []

        # Handle RGB vs grayscale
        if pattern.ndim == 3:
            gray = np.mean(pattern, axis=2)
            features.extend(self._channel_stats(pattern))
        else:
            gray = pattern

        # Resize if needed
        if gray.shape != self.grid_size:
            from scipy.ndimage import zoom
            zoom_factors = (self.grid_size[0] / gray.shape[0],
                          self.grid_size[1] / gray.shape[1])
            gray = zoom(gray, zoom_factors, order=1)

        # 1. Global statistics (5 features)
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray),
            (np.max(gray) - np.min(gray)) / (np.max(gray) + np.min(gray) + 1e-8),
        ])

        # 2. Gabor features — orientation × scale energy
        gabor_resp = self._gabor.apply(gray)
        features.extend(gabor_resp.features.tolist())

        # 3. Spectral features — radial × angular frequency bins
        spectral_res = self._spectral.analyze(gray)
        features.extend(spectral_res.features.tolist())

        # 4. HOG features — gradient orientation histograms
        hog_res = self._hog.compute(gray)
        features.extend(hog_res.features.tolist())

        # 5. Patch statistics (8×8 grid = 64 features)
        ph = max(1, gray.shape[0] // 8)
        pw = max(1, gray.shape[1] // 8)
        for i in range(8):
            for j in range(8):
                r0, r1 = i * ph, min((i + 1) * ph, gray.shape[0])
                c0, c1 = j * pw, min((j + 1) * pw, gray.shape[1])
                patch = gray[r0:r1, c0:c1]
                features.append(np.mean(patch) if patch.size > 0 else 0.0)

        # 6. Fork detection + dominant orientation (2 features)
        fork_count = self._detect_forks(gray)
        features.append(fork_count)
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        features.append(np.arctan2(np.sum(gy), np.sum(gx)))

        return np.array(features, dtype=np.float32)

    def _channel_stats(self, rgb: np.ndarray) -> List[float]:
        """Extract per-channel statistics from RGB pattern."""
        stats = []
        for c in range(3):
            channel = rgb[:, :, c]
            stats.extend([np.mean(channel), np.std(channel)])
        return stats

    def _detect_forks(self, pattern: np.ndarray, threshold: float = 0.3) -> int:
        """Simplified fork (topological defect) detection."""
        gx = np.gradient(pattern, axis=1)
        gy = np.gradient(pattern, axis=0)
        edge_mag = np.sqrt(gx**2 + gy**2)
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(edge_mag, size=3)
        peaks = (edge_mag == local_max) & (edge_mag > threshold * np.max(edge_mag))
        return int(np.sum(peaks))

    @property
    def n_features(self) -> int:
        """Total number of features extracted (computed on first call)."""
        if self._n_features is None:
            # Compute by running on a dummy pattern
            dummy = np.random.default_rng(0).random(self.grid_size)
            feats = self.extract(dummy)
            self._n_features = len(feats)
        return self._n_features
