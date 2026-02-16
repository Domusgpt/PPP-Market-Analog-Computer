/**
 * PPP Math Core — Unified Barrel Export
 *
 * Main entry point for all TypeScript math modules.
 * Adapted from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * with import paths adjusted for PPP math_core directory layout.
 *
 * Module inventory:
 * - geometric_algebra/ — Types, GeometricAlgebra, Lattice24, CausalReasoningEngine
 * - topology/          — Cell600, E8Projection, GoldenRatioScaling
 * - tda/               — PersistentHomology, GhostFrequencyDetector
 * - engine/            — TrinityEngine, ChronomorphicEngine
 * - domains/           — MusicGeometryDomain
 * - encoding/          — HDCEncoder
 * - metacognition/     — StateClassifier, EmbeddingClassifier
 * - applications/      — MusicGenerationMode, AnomalyDetectionMode, RoboticsControlAdapter
 * - shaders/           — TrinityShaders, MultiLayerRenderer
 */

// =============================================================================
// TYPE EXPORTS
// =============================================================================

export * from './geometric_algebra/types.js';

// =============================================================================
// MATH EXPORTS
// =============================================================================

export {
  dot,
  magnitude,
  normalize,
  scale,
  add,
  subtract,
  centroid,
  distance,
  distanceSquared,
  wedge,
  bivectorMagnitude,
  normalizeBivector,
  scaleBivector,
  addBivector,
  unitBivector,
  createRotor,
  rotorInPlane,
  rotorReverse,
  rotorMultiply,
  normalizeRotor,
  applyRotorToVector,
  rotationMatrix4D,
  matrixVectorMultiply,
  matrixMultiply,
  combinedRotationMatrix,
  isIsoclinicRotation,
  createIsoclinicRotor,
  stereographicProject,
  stereographicUnproject,
  Multivector
} from './geometric_algebra/GeometricAlgebra.js';

// =============================================================================
// TOPOLOGY EXPORTS
// =============================================================================

export {
  Lattice24,
  getDefaultLattice,
  createLattice
} from './geometric_algebra/Lattice24.js';

// =============================================================================
// ENGINE EXPORTS
// =============================================================================

export {
  CausalReasoningEngine
} from './geometric_algebra/CausalReasoningEngine.js';

// =============================================================================
// DOMAIN EXPORTS
// =============================================================================

export {
  MusicGeometryDomain,
  createMusicGeometryDomain,
  PITCH_CLASSES,
  CIRCLE_OF_FIFTHS,
  OCTATONIC_COLLECTIONS,
  TRINITY_OCTATONIC_MAP
} from './domains/MusicGeometryDomain.js';

// =============================================================================
// ENCODING EXPORTS
// =============================================================================

export {
  HDCEncoder,
  createHDCEncoder,
  DEFAULT_HDC_CONFIG,
  cosineSimilarity,
  bind,
  bundle,
  permute,
  normalizeHypervector
} from './encoding/HDCEncoder.js';

export type { Hypervector, HDCConfig } from './encoding/HDCEncoder.js';

// =============================================================================
// METACOGNITION EXPORTS
// =============================================================================

export {
  StateClassifier,
  DEFAULT_CLASSIFIER_CONFIG
} from './metacognition/StateClassifier.js';

export type {
  StateCategory,
  ConfidenceLevel,
  StateClassification,
  StateSuggestion,
  StateFeatures,
  ClassifierConfig
} from './metacognition/StateClassifier.js';

export {
  EmbeddingClassifier,
  cosineSimilarity as embeddingCosineSimilarity,
  averageEmbeddings,
  DEFAULT_EXAMPLES
} from './metacognition/EmbeddingClassifier.js';

export type {
  Embedding,
  EmbeddingProvider,
  EmbeddingClassifierConfig,
  LabeledExample,
  EmbeddingClassificationResult
} from './metacognition/EmbeddingClassifier.js';

// =============================================================================
// APPLICATION MODE EXPORTS
// =============================================================================

export {
  MusicGenerationMode,
  DEFAULT_MUSIC_CONFIG
} from './applications/MusicGenerationMode.js';

export type {
  PitchClass,
  Note,
  Chord,
  MusicEvent,
  FFTResult,
  ChromaVector,
  KeyDetection,
  MusicGenerationConfig,
  GenerationOutput,
  ModulationEvent
} from './applications/MusicGenerationMode.js';

export {
  AnomalyDetectionMode,
  DEFAULT_ANOMALY_CONFIG
} from './applications/AnomalyDetectionMode.js';

export type {
  DataPoint,
  AnomalyResult,
  AnomalyType,
  AnomalyFactor,
  BaselineProfile,
  AnomalyDetectionConfig
} from './applications/AnomalyDetectionMode.js';

export {
  RoboticsControlAdapter,
  DEFAULT_ROBOTICS_CONFIG,
  QUADCOPTER_X_MIXING
} from './applications/RoboticsControlAdapter.js';

export type {
  Vector3D,
  IMUData,
  Quaternion,
  RobotPose,
  ControlOutput,
  ControlMode,
  Waypoint,
  DriftCorrection,
  RoboticsConfig
} from './applications/RoboticsControlAdapter.js';

// =============================================================================
// SHADER EXPORTS
// =============================================================================

export {
  TRINITY_VERTEX_SHADER,
  TRINITY_FRAGMENT_SHADER,
  EDGE_VERTEX_SHADER,
  EDGE_FRAGMENT_SHADER,
  compileShader,
  createShaderProgram,
  createTrinityVertexProgram,
  createTrinityEdgeProgram,
  ShaderExports
} from './shaders/TrinityShaders.js';

export {
  MultiLayerManager,
  CrossSectionSlicer,
  MULTI_LAYER_VERTEX_SHADER,
  MULTI_LAYER_FRAGMENT_SHADER,
  DEFAULT_LAYER_CONFIG,
  DEFAULT_SLICE_CONFIG,
  DEFAULT_RENDER_CONFIG
} from './shaders/MultiLayerRenderer.js';

export type {
  LayerConfig,
  SliceConfig,
  RenderConfig
} from './shaders/MultiLayerRenderer.js';
