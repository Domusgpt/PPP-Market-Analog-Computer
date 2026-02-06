"""
Batch Encoder - High-Throughput Data Encoding
=============================================

Optimized encoder for processing large batches of data
with parallel execution and memory-efficient chunking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Generator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

from .fast_moire import FastMoireComputer, FastMoirePattern
from .fast_cascade import FastCascadeSimulator, CascadeResult


@dataclass
class BatchEncodingResult:
    """Results from batch encoding."""
    patterns: np.ndarray          # Shape: (batch, ny, nx)
    spectral: np.ndarray          # Shape: (batch, ny, nx, 3)
    features: np.ndarray          # Shape: (batch, n_features)
    layer_states: np.ndarray      # Shape: (batch, ny, nx)
    metadata: Dict                # Encoding parameters used


class BatchEncoder:
    """
    High-throughput batch encoder.

    Optimized for processing large datasets with:
    - Parallel moiré computation
    - Memory-efficient chunking
    - Shared computation caching
    - Progress tracking

    Parameters
    ----------
    grid_size : Tuple[int, int]
        Output resolution
    cascade_steps : int
        Reservoir cascade steps
    n_workers : int
        Number of parallel workers (0 = auto)

    Example
    -------
    >>> encoder = BatchEncoder(grid_size=(64, 64))
    >>> results = encoder.encode_batch(images)
    >>> print(f"Encoded {len(results.patterns)} images")
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        cascade_steps: int = 30,
        coupling_strength: float = 0.3,
        n_workers: int = 0
    ):
        self.grid_size = grid_size
        self.cascade_steps = cascade_steps
        self.coupling_strength = coupling_strength

        # Auto-detect workers
        if n_workers <= 0:
            import os
            n_workers = os.cpu_count() or 4
        self.n_workers = n_workers

        # Thread-local storage for simulators
        self._local = threading.local()

        # Shared moiré computer
        self.moire = FastMoireComputer()

        # Pre-computed angles
        self.angles = [0.0, 7.34, 9.43, 13.17, 21.79]
        self.current_angle_idx = 2  # Default: 9.43°

    def _get_simulator(self) -> FastCascadeSimulator:
        """Get thread-local simulator instance."""
        if not hasattr(self._local, 'simulator'):
            self._local.simulator = FastCascadeSimulator(
                self.grid_size,
                coupling_strength=self.coupling_strength
            )
        return self._local.simulator

    def _encode_single(self, data: np.ndarray, angle_idx: int) -> Dict:
        """Encode a single input (for parallel execution)."""
        # Get thread-local simulator
        sim = self._get_simulator()
        sim.reset()

        # Run cascade
        cascade_result = sim.run(data, n_steps=self.cascade_steps)

        # Compute moiré
        angle = self.angles[angle_idx % len(self.angles)]
        moire_result = self.moire.compute(
            twist_angle=angle,
            grid_size=self.grid_size,
            layer1_state=cascade_result.final_state
        )

        # Extract features
        features = self._extract_features(
            moire_result.intensity,
            cascade_result.final_state
        )

        return {
            'pattern': moire_result.intensity,
            'spectral': moire_result.spectral,
            'features': features,
            'layer_state': cascade_result.final_state,
            'angle': angle
        }

    def _extract_features(
        self,
        pattern: np.ndarray,
        layer_state: np.ndarray
    ) -> np.ndarray:
        """Extract feature vector from encoded pattern."""
        features = []

        # Intensity statistics
        features.extend([
            np.mean(pattern),
            np.std(pattern),
            np.min(pattern),
            np.max(pattern)
        ])

        # Contrast
        contrast = (np.max(pattern) - np.min(pattern)) / (np.max(pattern) + np.min(pattern) + 1e-8)
        features.append(contrast)

        # Spatial frequency content (radial bins)
        fft = np.abs(np.fft.fftshift(np.fft.fft2(pattern)))
        cy, cx = fft.shape[0] // 2, fft.shape[1] // 2
        y, x = np.ogrid[-cy:fft.shape[0]-cy, -cx:fft.shape[1]-cx]
        r = np.sqrt(x*x + y*y)

        for r_max in [5, 10, 20, 40]:
            mask = r < r_max
            features.append(np.mean(fft[mask]))

        # Layer state statistics
        features.extend([
            np.mean(layer_state),
            np.std(layer_state)
        ])

        return np.array(features, dtype=np.float32)

    def encode_batch(
        self,
        data: Union[List[np.ndarray], np.ndarray],
        angle_indices: Optional[List[int]] = None,
        chunk_size: int = 100,
        show_progress: bool = False
    ) -> BatchEncodingResult:
        """
        Encode batch of inputs.

        Parameters
        ----------
        data : List[np.ndarray] or np.ndarray
            Input data (list or 3D array)
        angle_indices : List[int], optional
            Angle index per input (defaults to current_angle_idx)
        chunk_size : int
            Chunk size for memory efficiency
        show_progress : bool
            Print progress updates

        Returns
        -------
        BatchEncodingResult
            Batch encoding results
        """
        # Convert to list if array
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = [data]
            else:
                data = [data[i] for i in range(data.shape[0])]

        n_samples = len(data)

        # Default angle indices
        if angle_indices is None:
            angle_indices = [self.current_angle_idx] * n_samples

        # Allocate output arrays
        patterns = np.zeros((n_samples, *self.grid_size), dtype=np.float32)
        spectral = np.zeros((n_samples, *self.grid_size, 3), dtype=np.float32)
        features_list = []
        layer_states = np.zeros((n_samples, *self.grid_size), dtype=np.float32)

        # Process in chunks
        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_data = data[chunk_start:chunk_end]
            chunk_angles = angle_indices[chunk_start:chunk_end]

            if show_progress:
                print(f"Processing {chunk_start+1}-{chunk_end} of {n_samples}...")

            # Parallel encode chunk
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(self._encode_single, d, a)
                    for d, a in zip(chunk_data, chunk_angles)
                ]

                for i, future in enumerate(futures):
                    idx = chunk_start + i
                    result = future.result()

                    patterns[idx] = result['pattern']
                    spectral[idx] = result['spectral']
                    features_list.append(result['features'])
                    layer_states[idx] = result['layer_state']

        # Stack features
        features = np.stack(features_list, axis=0)

        return BatchEncodingResult(
            patterns=patterns,
            spectral=spectral,
            features=features,
            layer_states=layer_states,
            metadata={
                'grid_size': self.grid_size,
                'cascade_steps': self.cascade_steps,
                'angles_used': [self.angles[i % len(self.angles)] for i in set(angle_indices)]
            }
        )

    def encode_stream(
        self,
        data_generator: Generator[np.ndarray, None, None],
        buffer_size: int = 10
    ) -> Generator[Dict, None, None]:
        """
        Stream encode data from generator.

        Parameters
        ----------
        data_generator : Generator
            Yields input arrays
        buffer_size : int
            Buffer size for parallel processing

        Yields
        ------
        Dict
            Encoded result for each input
        """
        buffer = []

        for item in data_generator:
            buffer.append(item)

            if len(buffer) >= buffer_size:
                # Process buffer
                results = self.encode_batch(buffer, show_progress=False)

                for i in range(len(buffer)):
                    yield {
                        'pattern': results.patterns[i],
                        'spectral': results.spectral[i],
                        'features': results.features[i],
                        'layer_state': results.layer_states[i]
                    }

                buffer = []

        # Process remaining
        if buffer:
            results = self.encode_batch(buffer, show_progress=False)
            for i in range(len(buffer)):
                yield {
                    'pattern': results.patterns[i],
                    'spectral': results.spectral[i],
                    'features': results.features[i],
                    'layer_state': results.layer_states[i]
                }

    def encode_multiangle(
        self,
        data: np.ndarray,
        angles: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode single input at multiple angles.

        Parameters
        ----------
        data : np.ndarray
            Single input array
        angles : List[int], optional
            Angle indices to use (default: all 5)

        Returns
        -------
        Dict
            'patterns': (n_angles, ny, nx)
            'features': (n_angles, n_features)
            'combined_features': concatenated features
        """
        if angles is None:
            angles = list(range(len(self.angles)))

        # Run cascade once
        sim = self._get_simulator()
        sim.reset()
        cascade_result = sim.run(data, n_steps=self.cascade_steps)

        # Compute moiré at each angle
        patterns = []
        all_features = []

        for angle_idx in angles:
            angle = self.angles[angle_idx]
            moire_result = self.moire.compute(
                twist_angle=angle,
                grid_size=self.grid_size,
                layer1_state=cascade_result.final_state
            )

            patterns.append(moire_result.intensity)
            features = self._extract_features(
                moire_result.intensity,
                cascade_result.final_state
            )
            all_features.append(features)

        return {
            'patterns': np.stack(patterns, axis=0),
            'features': np.stack(all_features, axis=0),
            'combined_features': np.concatenate(all_features)
        }

    def set_angle(self, angle_idx: int):
        """Set default encoding angle."""
        self.current_angle_idx = angle_idx % len(self.angles)

    def benchmark(self, n_samples: int = 100) -> Dict:
        """
        Run encoding benchmark.

        Returns timing statistics.
        """
        import time

        # Generate random test data
        test_data = [np.random.rand(*self.grid_size) for _ in range(n_samples)]

        # Warm-up
        self.encode_batch(test_data[:5])

        # Timed run
        start = time.time()
        self.encode_batch(test_data)
        elapsed = time.time() - start

        return {
            'n_samples': n_samples,
            'total_time_s': elapsed,
            'time_per_sample_ms': (elapsed / n_samples) * 1000,
            'throughput_per_s': n_samples / elapsed,
            'grid_size': self.grid_size,
            'n_workers': self.n_workers
        }
