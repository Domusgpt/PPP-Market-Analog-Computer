"""
Audio Stream Processor - Real-Time Audio Encoding
================================================

Real-time audio processing with spectrogram-based encoding.
"""

import numpy as np
from typing import Optional, Callable, Generator
from dataclasses import dataclass
import time

from .stream_encoder import StreamEncoder, StreamConfig, StreamFrame
from .buffer import RingBuffer


@dataclass
class AudioConfig:
    """Configuration for audio streaming."""
    sample_rate: int = 22050
    frame_size: int = 1024
    hop_size: int = 512
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 8000.0


class AudioStreamProcessor:
    """
    Real-time audio stream processor.

    Converts audio input to spectrograms and encodes
    as moiré patterns in real-time.

    Parameters
    ----------
    audio_config : AudioConfig
        Audio processing configuration
    stream_config : StreamConfig
        Encoder configuration

    Example
    -------
    >>> processor = AudioStreamProcessor()
    >>> processor.start()
    >>> # Feed audio samples
    >>> processor.feed(audio_chunk)
    >>> # Get latest pattern
    >>> pattern = processor.get_current_pattern()
    """

    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        stream_config: Optional[StreamConfig] = None
    ):
        self.audio_config = audio_config or AudioConfig()
        self.stream_config = stream_config or StreamConfig(
            grid_size=(64, 64),
            cascade_steps=15
        )

        # Audio buffer
        self.audio_buffer = RingBuffer(
            capacity=self.audio_config.sample_rate * 10  # 10 seconds
        )

        # Stream encoder
        self.encoder = StreamEncoder(self.stream_config)

        # Precompute mel filterbank
        self._mel_filters = self._create_mel_filterbank()

        # State
        self._last_process_idx = 0
        self._frames: list = []

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel-scale filterbank."""
        n_fft = self.audio_config.frame_size
        n_mels = self.audio_config.n_mels
        sr = self.audio_config.sample_rate
        fmin = self.audio_config.fmin
        fmax = min(self.audio_config.fmax, sr / 2)

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel points
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bins
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        # Create filterbank
        n_freqs = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, n_freqs))

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising edge
            for j in range(left, center):
                if j < n_freqs:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling edge
            for j in range(center, right):
                if j < n_freqs:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def feed(self, samples: np.ndarray) -> int:
        """
        Feed audio samples into processor.

        Parameters
        ----------
        samples : np.ndarray
            Audio samples (mono, float32)

        Returns
        -------
        int
            Number of frames processed
        """
        # Add to buffer
        self.audio_buffer.extend(samples.flatten())

        # Process available frames
        n_processed = 0
        hop = self.audio_config.hop_size
        frame_size = self.audio_config.frame_size

        while self.audio_buffer.count >= self._last_process_idx + frame_size:
            # Extract frame
            audio_data = self.audio_buffer.get_last(
                self._last_process_idx + frame_size
            )
            frame_audio = audio_data[-frame_size:]

            # Compute spectrogram frame
            spec_frame = self._compute_spectrogram_frame(frame_audio)

            # Encode
            stream_frame = self.encoder.process(spec_frame)
            self._frames.append(stream_frame)

            self._last_process_idx += hop
            n_processed += 1

        return n_processed

    def _compute_spectrogram_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram for single frame."""
        # Window
        window = np.hanning(len(audio_frame))
        windowed = audio_frame * window

        # FFT
        spectrum = np.abs(np.fft.rfft(windowed))

        # Apply mel filterbank
        mel_spectrum = np.dot(self._mel_filters, spectrum)

        # Log scale
        mel_spectrum = np.log(mel_spectrum + 1e-6)

        # Reshape to 2D (stack same frame to make square-ish)
        n_mels = self.audio_config.n_mels
        repeats = max(1, n_mels // 4)
        spec_2d = np.tile(mel_spectrum.reshape(-1, 1), (1, repeats))

        return spec_2d

    def start(self):
        """Start processing."""
        self.encoder.start()
        self._last_process_idx = 0
        self._frames.clear()

    def stop(self):
        """Stop processing."""
        self.encoder.stop()

    def reset(self):
        """Reset processor state."""
        self.audio_buffer.clear()
        self.encoder.reset()
        self._last_process_idx = 0
        self._frames.clear()

    def get_current_pattern(self) -> Optional[np.ndarray]:
        """Get most recent pattern."""
        return self.encoder.pattern_buffer.get_current()

    def get_recent_patterns(self, n: int = 10) -> np.ndarray:
        """Get recent patterns."""
        return self.encoder.get_recent_patterns(n)

    def get_frames(self) -> list:
        """Get all processed frames."""
        return self._frames.copy()

    def process_file(self, audio_path: str) -> list:
        """
        Process entire audio file.

        Parameters
        ----------
        audio_path : str
            Path to audio file

        Returns
        -------
        list
            List of StreamFrame objects
        """
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)

            # Resample if needed
            if sr != self.audio_config.sample_rate:
                from scipy.signal import resample
                n_samples = int(len(audio) * self.audio_config.sample_rate / sr)
                audio = resample(audio, n_samples)

            # Convert to mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Process
            self.start()
            self.feed(audio.astype(np.float32))
            self.stop()

            return self.get_frames()

        except ImportError:
            raise ImportError("Install soundfile: pip install soundfile")

    def process_stream(
        self,
        audio_generator: Generator[np.ndarray, None, None]
    ) -> Generator[StreamFrame, None, None]:
        """
        Process audio stream.

        Parameters
        ----------
        audio_generator : Generator
            Yields audio chunks

        Yields
        ------
        StreamFrame
            Encoded frames
        """
        self.start()

        for chunk in audio_generator:
            n_frames = self.feed(chunk)

            # Yield new frames
            for _ in range(n_frames):
                if self._frames:
                    yield self._frames[-1]

        self.stop()
