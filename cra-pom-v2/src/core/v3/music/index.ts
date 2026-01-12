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

// 4D Polytopes (8-cell, 16-cell, 24-cell)
export {
  // Classes
  Cell8,
  Cell16,
  Cell24,

  // Vector operations
  createVector4D,
  distance4D as polytope_distance4D,
  scale4D as polytope_scale4D,
  add4D as polytope_add4D,
  midpoint4D,

  // Verification
  verifyConstruction,
  classifySymmetry,
  type SymmetryGroup,

  // Namespace
  Polytopes,
} from './polytopes';

// Music Geometry Domain (24-Cell musical mapping)
export {
  // Types
  type Vector4D,
  type Vertex24,
  type MusicalKey,
  type Edge24,
  type Path4D,
  type TensionResult,
  type CalibrationResult,

  // Core class
  Cell24 as MusicCell24,

  // Vector operations (from domain)
  distance4D,
  dot4D,
  magnitude4D,
  normalize4D,
  add4D,
  subtract4D,
  scale4D,
  lerp4D,

  // Tension analysis
  calculateTension,
  calculateResolution,

  // Path analysis
  createPath,
  analyzeVoiceLeading,
  detectPythagoreanComma,

  // Calibration
  runCalibrationSuite,

  // Semantic bridge
  blendSemanticAndStructural,

  // Namespace
  MusicGeometryDomain,
} from './music-geometry-domain';
