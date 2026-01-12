/**
 * PPP v3 Geometry Module
 *
 * High-dimensional geometry for cognitive computation:
 *
 * POLYTOPES:
 * - 8-Cell (Tesseract): 16 vertices - base structure
 * - 16-Cell (Hexadecachoron): 8 vertices - dual of 8-cell
 * - 24-Cell (Icositetrachoron): 24 vertices - self-dual, maps to 24 keys
 * - 600-Cell (Hexacosichoron): 120 vertices - 5x finer granularity
 *
 * TRINITY:
 * - Thesis/Antithesis/Synthesis dialectic logic
 * - 24-cell decomposes into 3 orthogonal 16-cells
 *
 * ROTATIONS:
 * - Quaternion-based 4D rotations (SO(4))
 * - Isoclinic rotations (all points same angle)
 * - Musical modulation mappings
 */

// Base polytopes (from music module for now)
export {
  Cell8,
  Cell16,
  Cell24,
  verifyConstruction,
  classifySymmetry,
  type SymmetryGroup,
  createVector4D,
  distance4D,
  scale4D,
  add4D,
  midpoint4D,
  Polytopes,
} from '../music/polytopes';

// 600-Cell
export {
  Cell600,
  verify600Cell,
  PHI,
  PHI_INV,
  Cell600Module,
} from './cell600';

// Trinity Decomposition
export {
  Trinity24Cell,
  createDialecticPair,
  createOpposition,
  createConvergence,
  type TrinityRole,
  type TrinityVertex,
  type TrinityState,
  type DialecticResult,
  type OverlapResult,
  TrinityModule,
} from './trinity';

// Quaternion 4D Rotations
export {
  // Quaternion operations
  createQuaternion,
  identityQuaternion,
  quaternionNorm,
  normalizeQuaternion,
  quaternionConjugate,
  quaternionInverse,
  quaternionMultiply,
  quaternionFromAxisAngle,
  quaternionSlerp,

  // 4D rotation operations
  rotate4D,
  createIsoclinicRotation,
  createSimpleRotation,
  createDoubleRotation,
  composeRotations,
  identityRotation,
  inverseRotation,

  // Batch operations
  rotatePoints4D,
  animateRotation,

  // Special rotations
  rotationBetweenPoints,
  musicalModulationRotation,

  // Types
  type Quaternion,
  type RotationPlane,
  type Rotation4D,

  // Module namespace
  Quaternion4DModule,
} from './quaternion4d';

// E8 Projection and φ-Scaling
export {
  E8Projection,
  createMusicalE8,
  createUniformLayer1,
  E8ProjectionModule,
  type Vector8D,
  type AlignmentPoint,
  type LayerBlendResult,
  type E8ProjectionState,
} from './e8-projection';

// Projection Pipeline (4D → 3D → 2D)
export {
  // 4D → 3D
  project4DTo3D,
  project4DTo3DBatch,

  // 3D → 2D
  project3DTo2D,
  project3DTo2DBatch,
  defaultCamera,

  // Full pipeline
  projectFull,
  projectFullBatch,
  projectFullSorted,

  // Multi-view
  multiViewProject,

  // Viewport
  toViewport,
  toViewportBatch,

  // Utilities
  distance3D,
  distance2D,

  // Class
  ProjectionPipeline,

  // Module
  ProjectionModule,

  // Types
  type Vector3D,
  type Vector2D,
  type Camera3D,
  type Viewport,
  type ProjectionMethod4D,
  type ProjectionMethod3D,
  type ProjectionOptions4D,
  type ProjectionOptions3D,
  type ProjectedPoint,
  type MultiViewResult,
  type AnimationFrame,
} from './projection';

// Homological Analysis
export {
  // Core computation
  computeBetti,
  computeDistanceMatrix,
  buildRipsComplex,

  // Persistence
  computePersistence,
  getIntervalPersistence,
  filterPersistence,

  // Event detection
  detectTransition,

  // Class
  HomologyAnalyzer,

  // Factory
  createPolytopeAnalyzer,
  quickBetti,

  // Module
  HomologyModule,

  // Types
  type BettiNumbers,
  type PersistenceInterval,
  type PersistenceDiagram,
  type TopologicalEventType,
  type TopologicalEvent,
  type SimplexData,
} from './homology';

// ============================================================================
// Re-export Vector4D type
// ============================================================================

export type { Vector4D } from '../music/music-geometry-domain';
