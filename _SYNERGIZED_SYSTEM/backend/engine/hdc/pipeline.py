"""
HDC Pipeline - Integrated Moiré + HDC Processing
=================================================

Complete pipeline that:
1. Takes raw sensor input (video, audio, etc.)
2. Encodes through moiré temporal encoder
3. Converts to hypervector with HDC
4. Stores/retrieves from associative memory

This is the "full stack" for robotics/SLAM applications.
"""

import numpy as np
from typing import Optional, Dict, List, Any, Tuple, Generator
from dataclasses import dataclass
import time

from ..streaming.temporal_encoder import TemporalStreamEncoder, TemporalConfig, TemporalFrame
from .encoder import HDCEncoder, HDCConfig, HDCResult
from .memory import AssociativeMemory, SpatialMemory, MemoryResult, MatchType


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Moiré encoder config
    grid_size: tuple = (64, 64)
    cascade_steps: int = 15
    decay_rate: float = 0.85

    # HDC config
    hdc_dim: int = 10000
    num_levels: int = 32
    spatial_sample: int = 16

    # Memory config
    exact_threshold: float = 0.7
    similar_threshold: float = 0.4
    min_loop_gap: int = 10  # Minimum frames before loop closure


@dataclass
class PipelineResult:
    """Result from pipeline processing."""
    # Moiré output
    moire_pattern: np.ndarray
    evolution_pattern: np.ndarray
    spectral_output: np.ndarray

    # HDC output
    hypervector: np.ndarray

    # Memory result
    memory_result: MemoryResult
    is_loop_closure: bool
    matched_label: Any

    # Metadata
    frame_index: int
    timestamp: float
    processing_time_ms: float


class MoireHDCPipeline:
    """
    Complete Moiré → HDC → Memory pipeline.

    This is the main interface for robotics/SLAM applications.
    Feed it sensor data, get back:
    - Visual moiré pattern (for debugging/visualization)
    - Hypervector representation (for efficient matching)
    - Memory query results (loop closure, anomaly detection)

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration

    Example
    -------
    >>> pipeline = MoireHDCPipeline()
    >>> pipeline.start()
    >>>
    >>> for frame in video_stream:
    ...     result = pipeline.process(frame)
    ...     if result.is_loop_closure:
    ...         print(f"Loop closure detected! Matching: {result.matched_label}")
    ...     # Store current location
    ...     pipeline.store_current("location_001")
    >>>
    >>> pipeline.stop()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Create temporal encoder config
        temporal_config = TemporalConfig(
            grid_size=self.config.grid_size,
            cascade_steps=self.config.cascade_steps,
            decay_rate=self.config.decay_rate
        )

        # Create HDC config
        hdc_config = HDCConfig(
            dim=self.config.hdc_dim,
            num_levels=self.config.num_levels,
            spatial_sample=self.config.spatial_sample
        )

        # Initialize components
        self.moire_encoder = TemporalStreamEncoder(temporal_config)
        self.hdc_encoder = HDCEncoder(hdc_config)
        self.memory = AssociativeMemory(
            dim=self.config.hdc_dim,
            exact_threshold=self.config.exact_threshold,
            similar_threshold=self.config.similar_threshold
        )

        # State
        self._running = False
        self._frame_count = 0
        self._last_hv: Optional[np.ndarray] = None
        self._last_moire: Optional[TemporalFrame] = None

    def start(self):
        """Start the pipeline."""
        self._running = True
        self._frame_count = 0
        self.moire_encoder.start()
        self.hdc_encoder.reset_streaming()

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        self.moire_encoder.stop()

    def process(
        self,
        input_data: np.ndarray,
        auto_store: bool = False,
        label: Any = None
    ) -> PipelineResult:
        """
        Process single input frame through full pipeline.

        Parameters
        ----------
        input_data : np.ndarray
            Raw input data (image, audio frame, sensor data)
        auto_store : bool
            Automatically store in memory
        label : Any
            Label for storage (if auto_store)

        Returns
        -------
        PipelineResult
            Complete processing result
        """
        start_time = time.time()

        # Step 1: Moiré encoding
        moire_frame = self.moire_encoder.process(input_data)

        # Step 2: HDC encoding (uses moiré pattern)
        hdc_result = self.hdc_encoder.encode_streaming(moire_frame.pattern)

        # Step 3: Memory query
        memory_result = self.memory.query(
            hdc_result.hypervector,
            min_temporal_gap=self.config.min_loop_gap
        )

        # Step 4: Optional storage
        if auto_store:
            self.memory.store(hdc_result.hypervector, label=label)

        # Update state
        self._last_hv = hdc_result.hypervector
        self._last_moire = moire_frame
        self._frame_count += 1

        processing_time = (time.time() - start_time) * 1000

        return PipelineResult(
            moire_pattern=moire_frame.pattern,
            evolution_pattern=moire_frame.evolution_pattern,
            spectral_output=moire_frame.spectral_output,
            hypervector=hdc_result.hypervector,
            memory_result=memory_result,
            is_loop_closure=memory_result.is_loop_closure,
            matched_label=memory_result.best_match_label,
            frame_index=self._frame_count,
            timestamp=moire_frame.timestamp,
            processing_time_ms=processing_time
        )

    def store_current(self, label: Any, metadata: Optional[Dict] = None):
        """
        Store the current state in memory.

        Parameters
        ----------
        label : Any
            Label for the stored pattern
        metadata : Dict
            Optional metadata
        """
        if self._last_hv is not None:
            self.memory.store(self._last_hv, label=label, metadata=metadata)

    def check_loop_closure(self) -> Tuple[bool, Any, float]:
        """
        Check if current state matches a previous state.

        Returns
        -------
        Tuple[bool, Any, float]
            (is_loop_closure, matched_label, similarity)
        """
        if self._last_hv is None:
            return False, None, 0.0

        result = self.memory.query(
            self._last_hv,
            min_temporal_gap=self.config.min_loop_gap
        )

        return (
            result.is_loop_closure,
            result.best_match_label,
            result.similarity
        )

    def process_stream(
        self,
        data_generator: Generator[np.ndarray, None, None],
        store_every: int = 10,
        label_prefix: str = "frame"
    ) -> Generator[PipelineResult, None, None]:
        """
        Process a stream of data.

        Parameters
        ----------
        data_generator : Generator
            Yields input data frames
        store_every : int
            Store every N frames in memory
        label_prefix : str
            Prefix for auto-generated labels

        Yields
        ------
        PipelineResult
            Results for each frame
        """
        self.start()

        for i, data in enumerate(data_generator):
            auto_store = (i % store_every == 0)
            label = f"{label_prefix}_{i}" if auto_store else None

            yield self.process(data, auto_store=auto_store, label=label)

        self.stop()

    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'frames_processed': self._frame_count,
            'memory_entries': len(self.memory),
            'memory_labels': self.memory.get_all_labels(),
            'moire_stats': self.moire_encoder.get_statistics(),
            'hdc_history_length': self.hdc_encoder.get_history_length()
        }

    def reset(self):
        """Reset all state."""
        self.moire_encoder.reset()
        self.hdc_encoder.reset_streaming()
        self.memory.clear()
        self._frame_count = 0
        self._last_hv = None
        self._last_moire = None


class SLAMPipeline(MoireHDCPipeline):
    """
    Specialized pipeline for SLAM applications.

    Adds spatial awareness and trajectory tracking.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(config)

        # Replace memory with spatial memory
        self.memory = SpatialMemory(
            dim=self.config.hdc_dim,
            exact_threshold=self.config.exact_threshold,
            similar_threshold=self.config.similar_threshold
        )

        # Trajectory tracking
        self._trajectory: List[Tuple[float, float, int]] = []  # (x, y, frame_idx)
        self._current_position = (0.0, 0.0)

    def set_position(self, x: float, y: float):
        """Set current robot/camera position."""
        self._current_position = (x, y)

    def process_with_position(
        self,
        input_data: np.ndarray,
        position: Tuple[float, float],
        auto_store: bool = True
    ) -> PipelineResult:
        """
        Process with known position.

        Parameters
        ----------
        input_data : np.ndarray
            Sensor input
        position : Tuple[float, float]
            Current (x, y) position
        auto_store : bool
            Store in spatial memory

        Returns
        -------
        PipelineResult
            Processing result
        """
        self._current_position = position

        # Process through base pipeline
        result = self.process(input_data, auto_store=False)

        # Spatial memory operations
        if auto_store:
            self.memory.store_with_position(
                result.hypervector,
                position,
                label=f"pos_{self._frame_count}",
                metadata={'frame': self._frame_count}
            )

        # Track trajectory
        self._trajectory.append((*position, self._frame_count))

        # Spatial loop closure check
        spatial_result = self.memory.query_spatial(
            result.hypervector,
            position,
            min_distance=5.0
        )

        # Update result with spatial info
        result.memory_result = spatial_result
        result.is_loop_closure = spatial_result.is_loop_closure
        result.matched_label = spatial_result.best_match_label

        return result

    def get_trajectory(self) -> List[Tuple[float, float, int]]:
        """Get recorded trajectory."""
        return self._trajectory.copy()

    def get_loop_closures(self) -> List[Tuple[int, int, float]]:
        """
        Get all detected loop closures.

        Returns
        -------
        List[Tuple[int, int, float]]
            List of (frame1, frame2, similarity) tuples
        """
        # This would track loop closures during processing
        # For now, return empty - would need to be tracked during process()
        return []
