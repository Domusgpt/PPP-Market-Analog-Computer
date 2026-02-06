"""
Auto Tuner - Automatic Parameter Selection
==========================================

Automatically selects optimal encoder parameters based on input data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.fast_moire import FastMoireComputer
from ..core.fast_cascade import FastCascadeSimulator


class InputType(Enum):
    """Detected input types."""
    TEXTURE = "texture"       # High frequency content
    EDGES = "edges"           # Strong edges
    SMOOTH = "smooth"         # Low frequency
    MIXED = "mixed"           # Combination
    UNKNOWN = "unknown"


@dataclass
class TunerConfig:
    """Configuration for auto-tuning."""
    # Search ranges
    angle_candidates: List[int] = None  # Indices into commensurate angles
    cascade_range: Tuple[int, int] = (10, 100)
    coupling_range: Tuple[float, float] = (0.1, 0.8)

    # Optimization
    n_trials: int = 10
    metric: str = "contrast"  # "contrast", "entropy", "variance"

    def __post_init__(self):
        if self.angle_candidates is None:
            self.angle_candidates = [0, 1, 2, 3, 4]


@dataclass
class TuningResult:
    """Result from auto-tuning."""
    best_angle_idx: int
    best_cascade_steps: int
    best_coupling: float
    detected_input_type: InputType
    metrics: Dict[str, float]
    all_trials: List[Dict]


class AutoTuner:
    """
    Automatic parameter tuner for the encoder.

    Analyzes input data and selects optimal encoding parameters
    based on content characteristics.

    Parameters
    ----------
    config : TunerConfig
        Tuning configuration

    Example
    -------
    >>> tuner = AutoTuner()
    >>> result = tuner.tune(input_data)
    >>> print(f"Best angle: {result.best_angle_idx}")
    >>> encoder.set_operating_mode(angle_index=result.best_angle_idx)
    """

    COMMENSURATE_ANGLES = [0.0, 7.34, 9.43, 13.17, 21.79]

    def __init__(self, config: Optional[TunerConfig] = None):
        self.config = config or TunerConfig()

        self.moire = FastMoireComputer()

    def analyze_input(self, data: np.ndarray) -> InputType:
        """
        Analyze input data to determine content type.

        Parameters
        ----------
        data : np.ndarray
            2D input array

        Returns
        -------
        InputType
            Detected content type
        """
        # Normalize
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)

        # Compute gradient magnitude
        gy, gx = np.gradient(data)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        # Frequency content
        fft = np.abs(np.fft.fftshift(np.fft.fft2(data)))
        cy, cx = fft.shape[0] // 2, fft.shape[1] // 2

        # Radial average
        y, x = np.ogrid[-cy:fft.shape[0]-cy, -cx:fft.shape[1]-cx]
        r = np.sqrt(x*x + y*y)

        low_freq = np.mean(fft[r < cy * 0.2])
        mid_freq = np.mean(fft[(r >= cy * 0.2) & (r < cy * 0.5)])
        high_freq = np.mean(fft[r >= cy * 0.5])

        # Statistics
        edge_strength = np.mean(gradient_mag)
        variance = np.var(data)

        # Classification
        freq_ratio = high_freq / (low_freq + 1e-8)

        if freq_ratio > 0.5 and edge_strength > 0.1:
            return InputType.TEXTURE
        elif edge_strength > 0.15:
            return InputType.EDGES
        elif variance < 0.1 and freq_ratio < 0.2:
            return InputType.SMOOTH
        elif freq_ratio > 0.3 or edge_strength > 0.08:
            return InputType.MIXED
        else:
            return InputType.UNKNOWN

    def suggest_parameters(self, input_type: InputType) -> Dict:
        """
        Suggest parameters based on input type.

        Parameters
        ----------
        input_type : InputType
            Detected input type

        Returns
        -------
        Dict
            Suggested parameters
        """
        suggestions = {
            InputType.TEXTURE: {
                'angle_idx': 2,      # 9.43° - fine detail
                'cascade_steps': 30,
                'coupling': 0.4
            },
            InputType.EDGES: {
                'angle_idx': 1,      # 7.34° - edge detection
                'cascade_steps': 40,
                'coupling': 0.3
            },
            InputType.SMOOTH: {
                'angle_idx': 3,      # 13.17° - intermediate
                'cascade_steps': 20,
                'coupling': 0.5
            },
            InputType.MIXED: {
                'angle_idx': 2,
                'cascade_steps': 35,
                'coupling': 0.35
            },
            InputType.UNKNOWN: {
                'angle_idx': 2,
                'cascade_steps': 30,
                'coupling': 0.4
            }
        }

        return suggestions.get(input_type, suggestions[InputType.UNKNOWN])

    def _compute_metric(
        self,
        pattern: np.ndarray,
        metric: str = "contrast"
    ) -> float:
        """Compute quality metric for pattern."""
        if metric == "contrast":
            # Michelson contrast
            return (pattern.max() - pattern.min()) / (pattern.max() + pattern.min() + 1e-8)

        elif metric == "entropy":
            # Histogram entropy
            hist, _ = np.histogram(pattern.flatten(), bins=256, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist + 1e-10))

        elif metric == "variance":
            return np.var(pattern)

        else:
            return 0.0

    def tune(
        self,
        data: np.ndarray,
        grid_size: Tuple[int, int] = (64, 64)
    ) -> TuningResult:
        """
        Tune parameters for given input.

        Parameters
        ----------
        data : np.ndarray
            Input data to tune for
        grid_size : Tuple[int, int]
            Grid size for evaluation

        Returns
        -------
        TuningResult
            Tuning results with best parameters
        """
        # Analyze input
        input_type = self.analyze_input(data)
        suggestions = self.suggest_parameters(input_type)

        # Create simulator
        sim = FastCascadeSimulator(grid_size)

        # Prepare input
        if data.shape != grid_size:
            from scipy.ndimage import zoom
            factors = (grid_size[0] / data.shape[0], grid_size[1] / data.shape[1])
            data = zoom(data, factors, order=1)

        data = (data - data.min()) / (data.max() - data.min() + 1e-8)

        # Trial results
        trials = []
        best_score = -float('inf')
        best_params = suggestions.copy()

        # Try suggested parameters first
        sim.reset()
        sim.coupling = suggestions['coupling']
        result = sim.run(data, n_steps=suggestions['cascade_steps'])

        pattern = self.moire.compute(
            twist_angle=self.COMMENSURATE_ANGLES[suggestions['angle_idx']],
            grid_size=grid_size,
            layer1_state=result.final_state
        )

        score = self._compute_metric(pattern.intensity, self.config.metric)
        trials.append({
            'angle_idx': suggestions['angle_idx'],
            'cascade_steps': suggestions['cascade_steps'],
            'coupling': suggestions['coupling'],
            'score': score
        })

        if score > best_score:
            best_score = score
            best_params = suggestions.copy()

        # Random search for better parameters
        for _ in range(self.config.n_trials - 1):
            # Random parameters
            angle_idx = np.random.choice(self.config.angle_candidates)
            cascade_steps = np.random.randint(*self.config.cascade_range)
            coupling = np.random.uniform(*self.config.coupling_range)

            # Evaluate
            sim.reset()
            sim.coupling = coupling
            result = sim.run(data, n_steps=cascade_steps)

            pattern = self.moire.compute(
                twist_angle=self.COMMENSURATE_ANGLES[angle_idx],
                grid_size=grid_size,
                layer1_state=result.final_state
            )

            score = self._compute_metric(pattern.intensity, self.config.metric)

            trials.append({
                'angle_idx': angle_idx,
                'cascade_steps': cascade_steps,
                'coupling': coupling,
                'score': score
            })

            if score > best_score:
                best_score = score
                best_params = {
                    'angle_idx': angle_idx,
                    'cascade_steps': cascade_steps,
                    'coupling': coupling
                }

        return TuningResult(
            best_angle_idx=best_params['angle_idx'],
            best_cascade_steps=best_params['cascade_steps'],
            best_coupling=best_params['coupling'],
            detected_input_type=input_type,
            metrics={'best_score': best_score, 'metric': self.config.metric},
            all_trials=trials
        )

    def tune_batch(
        self,
        data_samples: List[np.ndarray],
        grid_size: Tuple[int, int] = (64, 64)
    ) -> TuningResult:
        """
        Tune parameters for batch of samples.

        Finds parameters that work well across all samples.

        Parameters
        ----------
        data_samples : List[np.ndarray]
            List of input samples
        grid_size : Tuple[int, int]
            Grid size

        Returns
        -------
        TuningResult
            Best parameters for batch
        """
        # Tune each sample
        results = [self.tune(s, grid_size) for s in data_samples]

        # Aggregate scores
        param_scores = {}

        for result in results:
            key = (result.best_angle_idx, result.best_cascade_steps)
            if key not in param_scores:
                param_scores[key] = []
            param_scores[key].append(result.metrics['best_score'])

        # Find best average
        best_key = max(param_scores.keys(), key=lambda k: np.mean(param_scores[k]))

        return TuningResult(
            best_angle_idx=best_key[0],
            best_cascade_steps=best_key[1],
            best_coupling=np.mean([r.best_coupling for r in results]),
            detected_input_type=InputType.MIXED,  # Batch is mixed
            metrics={'mean_score': np.mean(param_scores[best_key])},
            all_trials=[t for r in results for t in r.all_trials]
        )
