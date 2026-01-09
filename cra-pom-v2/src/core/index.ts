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

export {
  TrajectoryHistory,
  type TrajectoryPoint,
  type TrajectoryStats,
  type TrajectoryExport,
} from './trajectory';

export {
  computeCoherenceMetrics,
  computeGeometricInvariants,
  analyzeLatticePosition,
  computeTrajectoryQuality,
  formatMetricValue,
  getMetricAssessment,
  type CoherenceMetrics,
  type GeometricInvariants,
  type LatticeAnalysis,
  type TrajectoryQuality,
} from './analysis';

// ============================================
// PPP (Polytopal Projection Processing) Modules
// ============================================

// Hyperdimensional Computing (HDC)
export {
  Hypervector,
  VSA,
  SemanticMemory,
  DEFAULT_DIMENSION,
} from './hdc';

// Fourier Holographic Reduced Representations (FHRR)
export {
  PhasorVector,
  FHRR_VSA,
  ResonatorNetwork,
  polar,
  phasor,
  type Complex,
} from './fhrr';

// Concept Polytopes
export {
  ConvexPolytope,
  VoronoiCell,
  VoronoiTessellation,
  PolytopeBundle,
  type HalfSpace,
  type ContainmentResult,
} from './polytope';

// Rules as Rotors (Clifford Algebra)
export {
  Rotor,
  RuleLibrary,
  InferenceChain,
  type Rule,
  type InferenceStep,
} from './rotor';

// Garden of Forking Paths (Multi-Future Prediction)
export {
  GardenOfForkingPaths,
  TimeHorizon,
  type FuturePath,
  type Fork,
  type FutureBundle,
  type FutureConstraint,
} from './garden';

// PPP Reasoning Engine
export {
  PPPReasoningEngine,
  type Concept,
  type ReasoningQuery,
  type ReasoningResult,
} from './reasoner';

// PPP System State (Machine API)
export {
  PPPStateManager,
  // Serialization functions
  serializeHypervector,
  deserializeHypervector,
  serializePhasorVector,
  deserializePhasorVector,
  serializePolytope,
  deserializePolytope,
  serializeRotor,
  deserializeRotor,
  serializeRule,
  deserializeRule,
  serializeInferenceChain,
  serializeFutureBundle,
  serializeReasoningResult,
  // JSON Schema for validation
  PPP_SYSTEM_STATE_SCHEMA,
  // Types
  type SerializedHypervector,
  type SerializedPhasorVector,
  type SerializedPolytope,
  type SerializedRotor,
  type SerializedRule,
  type SerializedConcept,
  type SerializedInferenceChain,
  type SerializedFuturePath,
  type SerializedFutureBundle,
  type SerializedReasoningResult,
  type PPPSystemState,
  type MachineAPIRequest,
  type MachineAPIResponse,
} from './state';

// PPP LLM Integration (Agentic Tools + Verification)
export {
  PPPIntegration,
  PPP_TOOLS,
  PPP_SYSTEM_PROMPT,
  GROUNDING_PROTOCOL,
  type VerificationProof,
  type VerifiedResponse,
  type GroundingCitation,
  type ToolDefinition,
} from './llm-integration';
