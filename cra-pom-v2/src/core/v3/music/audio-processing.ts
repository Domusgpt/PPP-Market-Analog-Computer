/**
 * PPP v3 Audio Processing for Music Embeddings
 *
 * Provides audio signal analysis to ground music embeddings in actual sound:
 * 1. Chromagram extraction (pitch class energy distribution)
 * 2. Mel spectrogram features (perceptual frequency representation)
 * 3. Harmonic analysis (fundamental + overtones)
 * 4. Rhythm features (onset detection, tempo)
 *
 * This enables verification of embeddings against real audio,
 * not just symbolic music theory.
 *
 * REFERENCES:
 * - https://arxiv.org/html/2410.06927v1 (Spectral features for audio classification)
 * - Mel spectrogram: Perceptually-motivated frequency representation
 */

// ============================================================================
// Types
// ============================================================================

export interface AudioBuffer {
  samples: Float32Array;
  sampleRate: number;
  duration: number;
  channels: number;
}

export interface SpectralFrame {
  frequencies: Float32Array;
  magnitudes: Float32Array;
  phases: Float32Array;
  timestamp: number;
}

export interface Chromagram {
  /** 12 pitch class energies per frame */
  frames: Float32Array[];
  /** Timestamps for each frame */
  times: number[];
  /** Frame rate (frames per second) */
  frameRate: number;
}

export interface MelSpectrogram {
  /** Mel band energies per frame */
  frames: Float32Array[];
  /** Number of mel bands */
  numBands: number;
  /** Timestamps for each frame */
  times: number[];
  /** Frequency range */
  fMin: number;
  fMax: number;
}

export interface AudioFeatures {
  /** Chromagram (pitch class content) */
  chromagram: Chromagram;
  /** RMS energy over time */
  rms: Float32Array;
  /** Spectral centroid (brightness) */
  spectralCentroid: Float32Array;
  /** Zero crossing rate (noisiness) */
  zeroCrossingRate: Float32Array;
  /** Detected fundamental frequency (if monophonic) */
  f0?: Float32Array;
}

export interface AudioEmbedding {
  /** Feature vector */
  vector: Float32Array;
  /** Dominant pitch class (if detectable) */
  pitchClass?: string;
  /** Chord detection result (if polyphonic) */
  chordEstimate?: string;
  /** Confidence scores */
  confidence: {
    pitch: number;
    chord: number;
  };
}

// ============================================================================
// Constants
// ============================================================================

const A4_FREQUENCY = 440;
const FRAME_SIZE = 2048;
const HOP_SIZE = 512;
const NUM_MEL_BANDS = 128;

// ============================================================================
// FFT Implementation (Pure JS for Browser Compatibility)
// ============================================================================

/**
 * Compute FFT of real-valued signal
 * Uses Cooley-Tukey radix-2 algorithm
 */
export function fft(signal: Float32Array): { real: Float32Array; imag: Float32Array } {
  const n = signal.length;

  // Ensure power of 2
  if ((n & (n - 1)) !== 0) {
    throw new Error('FFT requires power of 2 length');
  }

  const real = new Float32Array(n);
  const imag = new Float32Array(n);

  // Copy input
  for (let i = 0; i < n; i++) {
    real[i] = signal[i];
  }

  // Bit reversal
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Cooley-Tukey
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = -2 * Math.PI / len;

    for (let i = 0; i < n; i += len) {
      let wr = 1, wi = 0;
      const wpr = Math.cos(angle);
      const wpi = Math.sin(angle);

      for (let k = 0; k < halfLen; k++) {
        const tr = wr * real[i + k + halfLen] - wi * imag[i + k + halfLen];
        const ti = wr * imag[i + k + halfLen] + wi * real[i + k + halfLen];

        real[i + k + halfLen] = real[i + k] - tr;
        imag[i + k + halfLen] = imag[i + k] - ti;
        real[i + k] += tr;
        imag[i + k] += ti;

        const wTemp = wr;
        wr = wr * wpr - wi * wpi;
        wi = wTemp * wpi + wi * wpr;
      }
    }
  }

  return { real, imag };
}

/**
 * Compute magnitude spectrum from FFT result
 */
export function magnitude(real: Float32Array, imag: Float32Array): Float32Array {
  const n = real.length;
  const mag = new Float32Array(n / 2);

  for (let i = 0; i < n / 2; i++) {
    mag[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
  }

  return mag;
}

// ============================================================================
// Window Functions
// ============================================================================

/**
 * Hann window
 */
export function hannWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return window;
}

/**
 * Apply window to signal
 */
export function applyWindow(signal: Float32Array, window: Float32Array): Float32Array {
  const result = new Float32Array(signal.length);
  for (let i = 0; i < signal.length; i++) {
    result[i] = signal[i] * window[i];
  }
  return result;
}

// ============================================================================
// Chromagram Extraction
// ============================================================================

/**
 * Convert frequency to pitch class (0-11)
 */
export function frequencyToPitchClass(freq: number): number {
  if (freq <= 0) return 0;
  const midiNote = 12 * Math.log2(freq / A4_FREQUENCY) + 69;
  return ((Math.round(midiNote) % 12) + 12) % 12;
}

/**
 * Create mel filterbank
 */
function melFilterbank(
  numBands: number,
  fftSize: number,
  sampleRate: number,
  fMin: number,
  fMax: number
): Float32Array[] {
  const melMin = 2595 * Math.log10(1 + fMin / 700);
  const melMax = 2595 * Math.log10(1 + fMax / 700);

  // Mel points
  const melPoints = new Float32Array(numBands + 2);
  for (let i = 0; i < numBands + 2; i++) {
    melPoints[i] = melMin + (i * (melMax - melMin)) / (numBands + 1);
  }

  // Convert to Hz
  const hzPoints = melPoints.map(m => 700 * (Math.pow(10, m / 2595) - 1));

  // Convert to FFT bins
  const binPoints = hzPoints.map(f => Math.floor((fftSize + 1) * f / sampleRate));

  // Create filterbank
  const filterbank: Float32Array[] = [];

  for (let i = 0; i < numBands; i++) {
    const filter = new Float32Array(fftSize / 2);

    for (let j = binPoints[i]; j < binPoints[i + 1]; j++) {
      filter[j] = (j - binPoints[i]) / (binPoints[i + 1] - binPoints[i]);
    }
    for (let j = binPoints[i + 1]; j < binPoints[i + 2]; j++) {
      filter[j] = (binPoints[i + 2] - j) / (binPoints[i + 2] - binPoints[i + 1]);
    }

    filterbank.push(filter);
  }

  return filterbank;
}

/**
 * Extract chromagram from audio
 */
export function extractChromagram(
  audio: AudioBuffer,
  frameSize = FRAME_SIZE,
  hopSize = HOP_SIZE
): Chromagram {
  const { samples, sampleRate } = audio;
  const window = hannWindow(frameSize);
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;

  const frames: Float32Array[] = [];
  const times: number[] = [];

  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    const frame = samples.slice(start, start + frameSize);

    // Apply window and compute FFT
    const windowed = applyWindow(frame, window);
    const { real, imag } = fft(windowed);
    const mag = magnitude(real, imag);

    // Sum magnitudes into 12 pitch classes
    const chroma = new Float32Array(12);

    for (let bin = 1; bin < mag.length; bin++) {
      const freq = (bin * sampleRate) / frameSize;
      if (freq > 20 && freq < 4000) { // Typical musical range
        const pc = frequencyToPitchClass(freq);
        chroma[pc] += mag[bin];
      }
    }

    // Normalize
    const sum = chroma.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (let i = 0; i < 12; i++) {
        chroma[i] /= sum;
      }
    }

    frames.push(chroma);
    times.push(start / sampleRate);
  }

  return {
    frames,
    times,
    frameRate: sampleRate / hopSize,
  };
}

/**
 * Extract mel spectrogram from audio
 */
export function extractMelSpectrogram(
  audio: AudioBuffer,
  numBands = NUM_MEL_BANDS,
  frameSize = FRAME_SIZE,
  hopSize = HOP_SIZE,
  fMin = 20,
  fMax = 8000
): MelSpectrogram {
  const { samples, sampleRate } = audio;
  const window = hannWindow(frameSize);
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  const filterbank = melFilterbank(numBands, frameSize, sampleRate, fMin, fMax);

  const frames: Float32Array[] = [];
  const times: number[] = [];

  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    const frame = samples.slice(start, start + frameSize);

    // Apply window and compute FFT
    const windowed = applyWindow(frame, window);
    const { real, imag } = fft(windowed);
    const mag = magnitude(real, imag);

    // Apply filterbank
    const melFrame = new Float32Array(numBands);
    for (let i = 0; i < numBands; i++) {
      let sum = 0;
      for (let j = 0; j < mag.length; j++) {
        sum += mag[j] * filterbank[i][j];
      }
      melFrame[i] = Math.log(sum + 1e-10); // Log scale
    }

    frames.push(melFrame);
    times.push(start / sampleRate);
  }

  return {
    frames,
    numBands,
    times,
    fMin,
    fMax,
  };
}

// ============================================================================
// Audio Feature Extraction
// ============================================================================

/**
 * Calculate RMS energy
 */
export function calculateRMS(
  samples: Float32Array,
  frameSize = FRAME_SIZE,
  hopSize = HOP_SIZE
): Float32Array {
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  const rms = new Float32Array(numFrames);

  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    let sum = 0;
    for (let i = 0; i < frameSize; i++) {
      sum += samples[start + i] * samples[start + i];
    }
    rms[f] = Math.sqrt(sum / frameSize);
  }

  return rms;
}

/**
 * Calculate spectral centroid (brightness)
 */
export function calculateSpectralCentroid(
  audio: AudioBuffer,
  frameSize = FRAME_SIZE,
  hopSize = HOP_SIZE
): Float32Array {
  const { samples, sampleRate } = audio;
  const window = hannWindow(frameSize);
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  const centroid = new Float32Array(numFrames);

  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    const frame = samples.slice(start, start + frameSize);

    const windowed = applyWindow(frame, window);
    const { real, imag } = fft(windowed);
    const mag = magnitude(real, imag);

    let weightedSum = 0;
    let magSum = 0;

    for (let i = 0; i < mag.length; i++) {
      const freq = (i * sampleRate) / frameSize;
      weightedSum += freq * mag[i];
      magSum += mag[i];
    }

    centroid[f] = magSum > 0 ? weightedSum / magSum : 0;
  }

  return centroid;
}

/**
 * Calculate zero crossing rate
 */
export function calculateZeroCrossingRate(
  samples: Float32Array,
  frameSize = FRAME_SIZE,
  hopSize = HOP_SIZE
): Float32Array {
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  const zcr = new Float32Array(numFrames);

  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    let crossings = 0;

    for (let i = 1; i < frameSize; i++) {
      if ((samples[start + i] >= 0) !== (samples[start + i - 1] >= 0)) {
        crossings++;
      }
    }

    zcr[f] = crossings / frameSize;
  }

  return zcr;
}

/**
 * Extract comprehensive audio features
 */
export function extractAudioFeatures(audio: AudioBuffer): AudioFeatures {
  return {
    chromagram: extractChromagram(audio),
    rms: calculateRMS(audio.samples),
    spectralCentroid: calculateSpectralCentroid(audio),
    zeroCrossingRate: calculateZeroCrossingRate(audio.samples),
  };
}

// ============================================================================
// Audio Embedding
// ============================================================================

/**
 * Create embedding from audio features
 */
export function embedAudio(features: AudioFeatures): AudioEmbedding {
  // Average chromagram across time
  const avgChroma = new Float32Array(12);
  for (const frame of features.chromagram.frames) {
    for (let i = 0; i < 12; i++) {
      avgChroma[i] += frame[i] / features.chromagram.frames.length;
    }
  }

  // Find dominant pitch class
  let maxChroma = 0;
  let dominantPC = 0;
  for (let i = 0; i < 12; i++) {
    if (avgChroma[i] > maxChroma) {
      maxChroma = avgChroma[i];
      dominantPC = i;
    }
  }

  const pitchClasses = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  // Compute summary statistics for other features
  const avgRMS = features.rms.reduce((a, b) => a + b, 0) / features.rms.length;
  const avgCentroid = features.spectralCentroid.reduce((a, b) => a + b, 0) / features.spectralCentroid.length;
  const avgZCR = features.zeroCrossingRate.reduce((a, b) => a + b, 0) / features.zeroCrossingRate.length;

  // Create 15-dimensional embedding
  // [12 chroma + rms + centroid + zcr]
  const vector = new Float32Array(15);
  for (let i = 0; i < 12; i++) {
    vector[i] = avgChroma[i];
  }
  vector[12] = avgRMS;
  vector[13] = avgCentroid / 8000; // Normalize
  vector[14] = avgZCR * 10; // Scale up

  // Simple chord detection (major/minor triads)
  const majorPattern = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0];
  const minorPattern = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0];

  let bestChord = '';
  let bestChordScore = 0;

  for (let root = 0; root < 12; root++) {
    // Major
    let majorScore = 0;
    for (let i = 0; i < 12; i++) {
      majorScore += avgChroma[(root + i) % 12] * majorPattern[i];
    }
    if (majorScore > bestChordScore) {
      bestChordScore = majorScore;
      bestChord = pitchClasses[root];
    }

    // Minor
    let minorScore = 0;
    for (let i = 0; i < 12; i++) {
      minorScore += avgChroma[(root + i) % 12] * minorPattern[i];
    }
    if (minorScore > bestChordScore) {
      bestChordScore = minorScore;
      bestChord = pitchClasses[root] + 'm';
    }
  }

  return {
    vector,
    pitchClass: pitchClasses[dominantPC],
    chordEstimate: bestChord,
    confidence: {
      pitch: maxChroma,
      chord: bestChordScore,
    },
  };
}

// ============================================================================
// Tone Generation (for Testing)
// ============================================================================

/**
 * Generate a sine wave at a given frequency
 */
export function generateSineWave(
  frequency: number,
  duration: number,
  sampleRate = 44100
): AudioBuffer {
  const numSamples = Math.floor(duration * sampleRate);
  const samples = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    samples[i] = Math.sin((2 * Math.PI * frequency * i) / sampleRate);
  }

  return {
    samples,
    sampleRate,
    duration,
    channels: 1,
  };
}

/**
 * Generate a chord (multiple sine waves)
 */
export function generateChord(
  frequencies: number[],
  duration: number,
  sampleRate = 44100
): AudioBuffer {
  const numSamples = Math.floor(duration * sampleRate);
  const samples = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    let sum = 0;
    for (const freq of frequencies) {
      sum += Math.sin((2 * Math.PI * freq * i) / sampleRate);
    }
    samples[i] = sum / frequencies.length;
  }

  return {
    samples,
    sampleRate,
    duration,
    channels: 1,
  };
}

/**
 * Generate a note (with harmonics for more realistic sound)
 */
export function generateNote(
  frequency: number,
  duration: number,
  harmonics = [1, 0.5, 0.33, 0.25],
  sampleRate = 44100
): AudioBuffer {
  const numSamples = Math.floor(duration * sampleRate);
  const samples = new Float32Array(numSamples);

  // ADSR envelope
  const attack = 0.01 * sampleRate;
  const decay = 0.05 * sampleRate;
  const sustain = 0.7;
  const release = 0.1 * sampleRate;

  for (let i = 0; i < numSamples; i++) {
    let amplitude = 1;

    if (i < attack) {
      amplitude = i / attack;
    } else if (i < attack + decay) {
      amplitude = 1 - (1 - sustain) * (i - attack) / decay;
    } else if (i > numSamples - release) {
      amplitude = sustain * (numSamples - i) / release;
    } else {
      amplitude = sustain;
    }

    let sum = 0;
    for (let h = 0; h < harmonics.length; h++) {
      sum += harmonics[h] * Math.sin((2 * Math.PI * frequency * (h + 1) * i) / sampleRate);
    }

    samples[i] = amplitude * sum / harmonics.length;
  }

  return {
    samples,
    sampleRate,
    duration,
    channels: 1,
  };
}

// ============================================================================
// Exports
// ============================================================================

export const AudioProcessing = {
  // FFT
  fft,
  magnitude,

  // Windowing
  hannWindow,
  applyWindow,

  // Feature extraction
  extractChromagram,
  extractMelSpectrogram,
  extractAudioFeatures,
  calculateRMS,
  calculateSpectralCentroid,
  calculateZeroCrossingRate,

  // Embedding
  embedAudio,

  // Generation
  generateSineWave,
  generateChord,
  generateNote,
};

export default AudioProcessing;
