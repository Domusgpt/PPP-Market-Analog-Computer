/**
 * Unified Type System for the Chronomorphic Polytopal Engine
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module defines the complete type hierarchy for the CPE, synthesizing:
 * - Vib3-CORE visualization types (6D rotation, 24 geometry variants)
 * - PPP-info-site engine types (CausalReasoningEngine, Lattice24)
 * - Trinity Architecture types (Alpha/Beta/Gamma 16-cells)
 * - Phillips GIT types (Entropic drives, Convexity constraints)
 *
 * Theoretical Foundations:
 * - P1: Information is ontologically primary
 * - P2: Dual entropic drives (EPO-D dispersive, EPO-I integrative)
 * - P3: Universe is self-contained and self-referential
 * - D1: Convexity is mathematically required for stable equilibrium
 * - D3: 24-cell is uniquely maximal integration structure
 */

// =============================================================================
// MATHEMATICAL PRIMITIVES
// =============================================================================

/**
 * 4D vector representing a point in Euclidean 4-space.
 * Used for positions, velocities, and general 4D coordinates.
 * Components: [x, y, z, w] or equivalently [x₁, x₂, x₃, x₄]
 */
export type Vector4D = [number, number, number, number];

/**
 * 3D vector for projected coordinates or 3-space operations.
 */
export type Vector3D = [number, number, number];

/**
 * 6-component bivector in 4D (Cl₄,₀ geometric algebra).
 * Components correspond to the 6 rotation planes:
 * [e₁₂, e₁₃, e₁₄, e₂₃, e₂₄, e₃₄] = [XY, XZ, XW, YZ, YW, ZW]
 *
 * Mathematical Note: In 4D, rotations occur in PLANES, not around axes.
 * Each component represents the "amount" of rotation in that plane.
 */
export type Bivector4D = [number, number, number, number, number, number];

/**
 * Rotation plane indices for the 6 independent rotation planes in 4D.
 */
export enum RotationPlane {
  XY = 0, // e₁₂ - familiar 3D rotation in XY plane
  XZ = 1, // e₁₃ - familiar 3D rotation in XZ plane
  XW = 2, // e₁₄ - "exotic" 4D rotation involving W
  YZ = 3, // e₂₃ - familiar 3D rotation in YZ plane
  YW = 4, // e₂₄ - "exotic" 4D rotation involving W
  ZW = 5  // e₃₄ - "exotic" 4D rotation involving W
}

/**
 * Rotor in 4D geometric algebra (Cl₄,₀).
 * Represents orientation via R = exp(-θ/2 · B) where B is a unit bivector.
 *
 * Mathematical Representation:
 * - scalar: The real/scalar part (grade 0)
 * - bivector: The bivector part (grade 2)
 * - isUnit: Whether the rotor is normalized (|R| = 1)
 *
 * Unitary Update: State transforms via sandwich product S' = R·S·R̃
 * This preserves information (norm) during reasoning/rotation.
 */
export interface Rotor {
  readonly scalar: number;
  readonly bivector: Bivector4D;
  readonly isUnit: boolean;
}

/**
 * Unit quaternion for representing 3D rotations.
 * Related to SO(4) decomposition: SO(4) ≅ (SU(2) × SU(2))/ℤ₂
 */
export interface Quaternion {
  readonly w: number; // Real/scalar part
  readonly x: number; // i component
  readonly y: number; // j component
  readonly z: number; // k component
}

// =============================================================================
// TRINITY ARCHITECTURE TYPES
// =============================================================================

/**
 * The three axis pairs that partition the 24-cell into three 16-cells.
 *
 * Mathematical Foundation (W(D₄) ⊂ W(F₄) with index 3):
 * The 24-cell has F₄ symmetry (order 1152). It decomposes into 3 copies
 * of the 16-cell, each corresponding to a different pairing of coordinate axes.
 *
 * - Alpha: axes (1,2) and (3,4) → vertices (±1, ±1, 0, 0) and (0, 0, ±1, ±1)
 * - Beta:  axes (1,3) and (2,4) → vertices (±1, 0, ±1, 0) and (0, ±1, 0, ±1)
 * - Gamma: axes (1,4) and (2,3) → vertices (±1, 0, 0, ±1) and (0, ±1, ±1, 0)
 *
 * Musical Mapping (Octatonic Collections):
 * - Alpha → OCT₀,₁ = {C, C♯, E♭, E, F♯, G, A, B♭}
 * - Beta  → OCT₁,₂
 * - Gamma → OCT₀,₂
 */
export type TrinityAxis = 'alpha' | 'beta' | 'gamma';

/**
 * Multi-state superposition for the Trinity architecture.
 * Represents the current "harmonic context" as a weighted blend of all three axes.
 *
 * weights: [w_α, w_β, w_γ] where Σwᵢ = 1 (normalized)
 *
 * Interpretation:
 * - Pure state: One weight = 1, others = 0 (single key/mode)
 * - Polytonal: Multiple non-zero weights (harmonic ambiguity)
 * - Phase shift: Transition from one dominant axis to another
 */
export interface TrinityState {
  readonly activeAxis: TrinityAxis;
  readonly weights: [number, number, number]; // [w_α, w_β, w_γ]
  readonly tension: number; // Inter-axis tension (0-1)
  readonly phaseProgress: number; // Progress through current phase shift (0-1)
}

/**
 * Information about a phase shift between Trinity axes.
 * Analogous to musical modulation between keys.
 */
export interface PhaseShiftInfo {
  readonly from: TrinityAxis;
  readonly to: TrinityAxis;
  readonly crossAxisVertices: number[]; // Vertices shared between axes
  readonly rotationPlane: RotationPlane; // Which 4D plane the shift occurs in
  readonly direction: 1 | -1; // Rotation direction
}

/**
 * Decomposition of the 24-cell into three 16-cells.
 */
export interface TrinityDecomposition {
  readonly alpha: readonly number[]; // 8 vertex indices for Alpha 16-cell
  readonly beta: readonly number[];  // 8 vertex indices for Beta 16-cell
  readonly gamma: readonly number[]; // 8 vertex indices for Gamma 16-cell
}

// =============================================================================
// LATTICE AND TOPOLOGY TYPES
// =============================================================================

/**
 * A vertex of the 24-cell lattice (Orthocognitum).
 *
 * The 24 vertices are all permutations of (±1, ±1, 0, 0).
 * Each vertex has exactly 8 neighbors at distance √2.
 */
export interface LatticeVertex {
  readonly id: number;
  readonly coordinates: Vector4D;
  readonly neighbors: readonly number[]; // 8 neighbor vertex IDs
  readonly trinityAxis?: TrinityAxis; // Which 16-cell this vertex belongs to
}

/**
 * An octahedral cell of the 24-cell.
 * The 24-cell has 24 octahedral cells (3-faces).
 */
export interface LatticeCell {
  readonly id: number;
  readonly vertices: readonly number[]; // 6 vertex IDs forming the octahedron
  readonly centroid: Vector4D;
}

/**
 * Voronoi region around a lattice vertex.
 * Used for nearest-neighbor classification and coherence computation.
 */
export interface VoronoiRegion {
  readonly vertexId: number;
  readonly center: Vector4D;
  readonly radius: number; // Effective radius (half edge length)
}

/**
 * Result of convexity/validity checking (Epistaorthognition).
 *
 * From Phillips GIT (D1): Convexity is mathematically required for stable equilibrium.
 * States outside the convex hull represent invalid/incoherent configurations.
 */
export interface ConvexityResult {
  readonly isValid: boolean;           // Inside the convex hull?
  readonly coherence: number;          // 0-1 coherence score
  readonly nearestVertex: number;      // Index of nearest lattice vertex
  readonly distance: number;           // Distance to nearest vertex
  readonly centroid: Vector4D;         // Centroid of k-nearest vertices
  readonly activeVertices: number[];   // Indices of k-nearest vertices
}

// =============================================================================
// POLYTOPE TYPES
// =============================================================================

/**
 * The six regular convex 4-polytopes (polychora).
 *
 * From the research: These form an "alphabet of state-space" where:
 * - 5-cell: Minimal connectivity (4 neighbors per vertex)
 * - 8-cell: Orthogonal hypercubic structure
 * - 16-cell: Dual of 8-cell, octahedral cells
 * - 24-cell: Self-dual, maximal integration structure (D3)
 * - 120-cell: Dodecahedral cells, F₄ × 10 symmetry elements
 * - 600-cell: Tetrahedral cells, contains 25 inscribed 24-cells
 */
export type PolychoronType =
  | '5-cell'   // 4-simplex (pentatope)
  | '8-cell'   // Tesseract (hypercube)
  | '16-cell'  // 4-orthoplex (hexadecachoron)
  | '24-cell'  // Icositetrachoron (self-dual)
  | '120-cell' // Hecatonicosachoron
  | '600-cell';// Hexacosichoron

/**
 * Geometry variant encoding (from Vib3-CORE).
 * geometry = coreIndex × 8 + baseIndex
 *
 * Base geometries (0-7): Tetrahedron, Hypercube, Sphere, Torus, Klein, Fractal, Wave, Crystal
 * Core types (0-2): Base, Hypersphere Core, Hypertetrahedron Core
 */
export interface GeometryVariant {
  readonly baseIndex: number;    // 0-7: Which base geometry
  readonly coreIndex: number;    // 0-2: Which core type wrapping
  readonly combined: number;     // 0-23: Combined geometry index
}

// =============================================================================
// ENGINE STATE AND PHYSICS
// =============================================================================

/**
 * Complete state of the Causal Reasoning Engine.
 *
 * Core Principles (from PPP White Paper):
 * - "Reasoning is Rotation": Logical inference = applying rotor R to state S
 * - "Force ∧ State = Torque": Input generates rotation via wedge product
 * - "Unitary Update: R·S·R̃": Transformations preserve norm (truth value)
 */
export interface EngineState {
  readonly position: Vector4D;           // Current 4D position in Orthocognitum
  readonly orientation: Rotor;           // Current 4D orientation
  readonly velocity: Vector4D;           // Linear velocity
  readonly angularVelocity: Bivector4D;  // Angular velocity (6 components)
  readonly trinityState?: TrinityState;  // Trinity architecture state
  readonly timestamp: number;            // Timestamp of this state
}

/**
 * Force applied to the engine.
 *
 * Forces generate torque via wedge product: τ = position ∧ force
 * This implements "Context is constructed by the wedge product" -
 * the rotation plane emerges from the relationship between state and input.
 */
export interface Force {
  readonly linear: Vector4D;         // Linear force component
  readonly rotational: Bivector4D;   // Direct rotational force
  readonly magnitude: number;        // Total force magnitude
  readonly source: string;           // Source identifier
}

/**
 * Torque computed from force and state.
 * τ = position ∧ force (bivector defining rotation plane)
 */
export interface Torque {
  readonly plane: Bivector4D;              // Rotation plane (normalized)
  readonly magnitude: number;              // Torque magnitude
  readonly angularAcceleration: Bivector4D;// Resulting angular acceleration
}

/**
 * Result of an engine update step.
 */
export interface UpdateResult {
  readonly state: EngineState;
  readonly torque: Torque;
  readonly convexity: ConvexityResult;
  readonly deltaTime: number;
  readonly wasClamped: boolean;
}

/**
 * Engine configuration parameters.
 *
 * The three causal constraints (from Gärdenfors fpsyg-11-00630):
 * 1. MONOTONICITY: Larger forces → larger results
 * 2. CONTINUITY: Small force changes → small result changes
 * 3. CONVEXITY: Intermediate forces → intermediate results
 */
export interface EngineConfig {
  readonly fixedTimestep: number;      // Physics update timestep
  readonly damping: number;            // Velocity damping (0-1)
  readonly inertia: number;            // Rotational inertia (0-1)
  readonly maxLinearVelocity: number;  // Max linear velocity
  readonly maxAngularVelocity: number; // Max angular velocity
  readonly kNearest: number;           // k for k-nearest coherence check
  readonly autoClamp: boolean;         // Auto-clamp to valid region
}

/**
 * Default engine configuration.
 */
export const DEFAULT_ENGINE_CONFIG: EngineConfig = {
  fixedTimestep: 1 / 60,
  damping: 0.02,
  inertia: 0.3,
  maxLinearVelocity: 2.0,
  maxAngularVelocity: Math.PI,
  kNearest: 4,
  autoClamp: true
};

// =============================================================================
// TELEMETRY AND EVENTS
// =============================================================================

/**
 * Telemetry event types for engine monitoring.
 */
export enum TelemetryEventType {
  ENGINE_INITIALIZED = 'engine:initialized',
  ENGINE_RESET = 'engine:reset',
  STATE_UPDATE = 'state:update',
  FORCE_APPLIED = 'force:applied',
  TOPOLOGY_VIOLATION = 'topology:violation',
  COHERENCE_CHANGE = 'coherence:change',
  LATTICE_TRANSITION = 'lattice:transition',
  PHASE_SHIFT_START = 'phase:shift:start',
  PHASE_SHIFT_COMPLETE = 'phase:shift:complete',
  TDA_ANALYSIS = 'tda:analysis'
}

/**
 * Telemetry event payload.
 */
export interface TelemetryEvent {
  readonly timestamp: number;
  readonly eventType: TelemetryEventType;
  readonly payload: Record<string, unknown>;
}

/**
 * Telemetry subscriber callback.
 */
export type TelemetrySubscriber = (event: TelemetryEvent) => void;

// =============================================================================
// TDA (TOPOLOGICAL DATA ANALYSIS) TYPES
// =============================================================================

/**
 * Betti number profile from persistent homology analysis.
 *
 * Interpretation:
 * - β₀: Number of connected components (cohesion/texture)
 * - β₁: Number of 1D holes/loops (cycles/repetition)
 * - β₂: Number of 2D voids (ambiguity/"ghost frequencies")
 */
export interface BettiProfile {
  readonly beta0: number; // Connected components
  readonly beta1: number; // 1D holes
  readonly beta2: number; // 2D voids
  readonly persistence: Float32Array; // Persistence pairs
}

/**
 * Persistence pair from TDA (birth, death) times.
 */
export interface PersistencePair {
  readonly birth: number;
  readonly death: number;
  readonly dimension: number; // 0, 1, or 2
}

/**
 * Topological void detected via β₂ analysis.
 * Used for "ghost frequency" detection in musical applications.
 */
export interface TopologicalVoid {
  readonly center: Vector4D;      // Estimated center of the void
  readonly radius: number;        // Effective radius
  readonly persistence: number;   // How persistent this feature is
  readonly nearestVertex: number; // Nearest lattice vertex ("ghost vertex")
}

// =============================================================================
// VISUALIZATION TYPES (FROM VIB3-CORE)
// =============================================================================

/**
 * 6D rotation state for visualization.
 * Corresponds to the six independent rotation planes in 4D.
 */
export interface RotationState {
  readonly rot4dXY: number; // Familiar 3D rotation
  readonly rot4dXZ: number; // Familiar 3D rotation
  readonly rot4dYZ: number; // Familiar 3D rotation
  readonly rot4dXW: number; // Exotic 4D rotation
  readonly rot4dYW: number; // Exotic 4D rotation
  readonly rot4dZW: number; // Exotic 4D rotation
}

/**
 * Visualization parameters for the holographic shader system.
 */
export interface VisualizationParams {
  readonly polychoronType: PolychoronType;
  readonly geometry: GeometryVariant;
  readonly rotation: RotationState;
  readonly projectionDistance: number;    // Stereographic projection R
  readonly sliceW: number;                // W-coordinate for hyperplane slice
  readonly lineThickness: number;         // Edge line thickness
  readonly opacity: number;               // Base opacity
  readonly hue: number;                   // Color hue (0-360)
  readonly saturation: number;            // Color saturation (0-1)
  readonly intensity: number;             // Brightness intensity (0-1)
  readonly showTrinityColors: boolean;    // Color by Trinity axis
  readonly showCrossAxisBridges: boolean; // Highlight inter-axis edges
}

// =============================================================================
// MUSIC-GEOMETRY DOMAIN TYPES
// =============================================================================

/**
 * Musical key representation.
 */
export interface MusicalKey {
  readonly root: number;      // 0-11 (C=0, C#=1, ..., B=11)
  readonly mode: 'major' | 'minor';
  readonly vertexId: number;  // Corresponding 24-cell vertex
}

/**
 * Chord representation with geometric mapping.
 */
export interface GeometricChord {
  readonly pitchClasses: number[];     // Pitch classes (0-11)
  readonly activeVertices: number[];   // Active 24-cell vertices
  readonly centroid: Vector4D;         // Geometric centroid
  readonly trinityWeights: [number, number, number]; // Trinity superposition
}

/**
 * Octatonic collection (8-note scale).
 * Each Trinity axis maps to one of the three octatonic collections.
 */
export interface OctatonicCollection {
  readonly axis: TrinityAxis;
  readonly pitchClasses: readonly number[]; // 8 pitch classes
  readonly name: string; // e.g., "OCT₀,₁"
}

// =============================================================================
// MATHEMATICAL CONSTANTS
// =============================================================================

/**
 * Mathematical constants used throughout the engine.
 */
export const MATH_CONSTANTS = {
  EPSILON: 1e-10,
  PHI: (1 + Math.sqrt(5)) / 2,        // Golden ratio φ ≈ 1.618
  PHI_INV: 2 / (1 + Math.sqrt(5)),    // 1/φ ≈ 0.618
  SQRT2: Math.SQRT2,                  // √2 ≈ 1.414
  SQRT5: Math.sqrt(5),                // √5 ≈ 2.236
  TAU: 2 * Math.PI,                   // τ = 2π ≈ 6.283

  // 24-Cell specific
  CIRCUMRADIUS_24: Math.SQRT2,        // Circumradius of unit 24-cell
  EDGE_LENGTH_24: Math.SQRT2,         // Edge length of unit 24-cell
  INRADIUS_24: 1,                     // Inradius of unit 24-cell

  // 600-Cell specific (from E₈→H₄ projection)
  CIRCUMRADIUS_600: Math.sqrt(2 + ((1 + Math.sqrt(5)) / 2)), // ≈ 1.902

  // Hurwitz quaternion norm
  HURWITZ_NORM: 1                     // Unit Hurwitz quaternions
} as const;

// =============================================================================
// TYPE GUARDS
// =============================================================================

/** Type guard for Vector4D */
export function isVector4D(v: unknown): v is Vector4D {
  return Array.isArray(v) && v.length === 4 && v.every(x => typeof x === 'number');
}

/** Type guard for Bivector4D */
export function isBivector4D(v: unknown): v is Bivector4D {
  return Array.isArray(v) && v.length === 6 && v.every(x => typeof x === 'number');
}

/** Type guard for Rotor */
export function isRotor(r: unknown): r is Rotor {
  return (
    typeof r === 'object' &&
    r !== null &&
    'scalar' in r &&
    'bivector' in r &&
    typeof (r as Rotor).scalar === 'number' &&
    isBivector4D((r as Rotor).bivector)
  );
}

/** Type guard for TrinityAxis */
export function isTrinityAxis(a: unknown): a is TrinityAxis {
  return a === 'alpha' || a === 'beta' || a === 'gamma';
}

/** Type guard for PolychoronType */
export function isPolychoronType(p: unknown): p is PolychoronType {
  return ['5-cell', '8-cell', '16-cell', '24-cell', '120-cell', '600-cell'].includes(p as string);
}
