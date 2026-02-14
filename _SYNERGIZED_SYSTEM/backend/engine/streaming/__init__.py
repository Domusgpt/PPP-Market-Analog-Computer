"""
Streaming Module - Real-Time Processing
=======================================

Real-time streaming encoder for audio, video, and sensor data.

Modules:
- stream_encoder: Core streaming API
- temporal_encoder: Temporal dynamics encoder with proper sequence discrimination
- audio_stream: Audio input handling
- video_stream: Video/webcam processing
- buffer: Ring buffer utilities
"""

from .stream_encoder import StreamEncoder, StreamConfig
from .temporal_encoder import TemporalStreamEncoder, TemporalConfig, TemporalFrame, LogicMode
from .audio_stream import AudioStreamProcessor
from .buffer import RingBuffer, FrameBuffer

__all__ = [
    "StreamEncoder",
    "StreamConfig",
    "TemporalStreamEncoder",
    "TemporalConfig",
    "TemporalFrame",
    "LogicMode",
    "AudioStreamProcessor",
    "RingBuffer",
    "FrameBuffer"
]
