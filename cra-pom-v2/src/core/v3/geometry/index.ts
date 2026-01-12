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

// ============================================================================
// Re-export Vector4D type
// ============================================================================

export type { Vector4D } from '../music/music-geometry-domain';
