/**
 * PPP v3 Music Theory & Audio Processing
 *
 * Provides mathematically-grounded embeddings for music based on:
 * - Tonnetz geometry (Euler 1739)
 * - Fourier phase space (continuous harmony)
 * - Audio spectral features (mel, chromagram)
 *
 * Perfect for testing embedding quality because music has
 * verifiable mathematical ground truth.
 */

// Music theory (symbolic)
export {
  // Types
  type PitchClass,
  type Note,
  type Chord,
  type ChordType,
  type TonnetzCoordinate,
  type MusicEmbedding,

  // Constants
  PITCH_CLASSES,
  INTERVALS,
  JUST_RATIOS,
  A4_FREQUENCY,

  // Pitch operations
  pitchClassToSemitone,
  semitoneToPitchClass,
  intervalBetween,
  createNote,
  toMidi,
  midiToFrequency,

  // Tonnetz
  pitchClassToTonnetz,
  tonnetzDistance,

  // Chords
  createChord,
  chordName,

  // Fourier phase
  fourierPhase,

  // Embeddings
  embedPitchClass,
  embedChord,
  cosineSimilarity,
  euclideanDistance,

  // Analogies
  analogy,

  // Verification
  verifyMusicEmbeddings,

  // Namespace
  MusicTheory,
} from './music-theory';

// Audio processing
export {
  // Types
  type AudioBuffer,
  type SpectralFrame,
  type Chromagram,
  type MelSpectrogram,
  type AudioFeatures,
  type AudioEmbedding,

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

  // Tone generation
  generateSineWave,
  generateChord,
  generateNote,

  // Namespace
  AudioProcessing,
} from './audio-processing';
