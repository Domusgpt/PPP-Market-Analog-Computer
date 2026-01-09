// File: src/core/index.ts
// Core module exports

export {
  Quaternion,
  Lattice24,
  CognitiveManifold,
  PHI,
  PHI_INV,
  CONVEXITY_TOLERANCE,
  type QuaternionState,
  type LatticeState,
  type ConvexityStatus,
  type ConvexityResult,
  type ManifoldStepResult,
  type GeometricState,
} from './geometry';

export {
  AuditChain,
  getGlobalAuditChain,
  resetGlobalAuditChain,
  type AuditEventType,
  type LogEntry,
  type LogEntryCompact,
  type ChainValidation,
  type ChainStatistics,
} from './trace';

export {
  stereographicProject4Dto3D,
  perspectiveProject3Dto2D,
  projectQuaternionTo2D,
  projectLattice,
  ProjectionAnimator,
  getProjectionState,
  DEFAULT_PROJECTION_PARAMS,
  type Point2D,
  type Point3D,
  type ProjectedVertex,
  type ProjectedEdge,
  type ProjectionResult,
  type ProjectionParams,
  type ProjectionState,
} from './projection';
