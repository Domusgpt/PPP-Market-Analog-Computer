/**
 * Chronomorphic Polytopal Engine - Core Type Definitions
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module defines the foundational types for Geometric Cognition,
 * implementing the mathematical structures described in the PPP white papers.
 *
 * Key Concepts:
 * - Multivectors: Elements of Clifford Algebra Cl(4,0)
 * - Polytopes: Bounded convex regions defining valid concept spaces
 * - Rotors: Unitary transformations for reasoning operations
 * - Forces: Input vectors that drive state transitions
 */

// =============================================================================
// PRIMITIVE GEOMETRIC TYPES
// =============================================================================

/**
 * A 4-dimensional vector in Euclidean space.
 * Components: [x, y, z, w] where w is the fourth spatial dimension.
 */
export type Vector4D = [number, number, number, number];

/**
 * A 3-dimensional vector for spatial operations and projections.
 */
export type Vector3D = [number, number, number];

/**
 * A unit quaternion for efficient rotation representation.
 * Format: [w, x, y, z] where w is the scalar part.
 */
export type Quaternion = [number, number, number, number];

/**
 * A bivector in 4D represents a 2D plane of rotation.
 * There are 6 basis bivectors in Cl(4,0): e12, e13, e14, e23, e24, e34
 */
export type Bivector4D = [number, number, number, number, number, number];

/**
 * Grade decomposition for multivectors in Cl(4,0).
 * Grade 0: Scalar (1 component)
 * Grade 1: Vector (4 components)
 * Grade 2: Bivector (6 components)
 * Grade 3: Trivector (4 components)
 * Grade 4: Pseudoscalar (1 component)
 */
export enum Grade {
    SCALAR = 0,
    VECTOR = 1,
    BIVECTOR = 2,
    TRIVECTOR = 3,
    PSEUDOSCALAR = 4
}

// =============================================================================
// MULTIVECTOR TYPES
// =============================================================================

/**
 * Full multivector in Clifford Algebra Cl(4,0).
 * Total dimension: 2^4 = 16 basis elements.
 *
 * Indices:
 * [0]      : scalar (1)
 * [1-4]    : vectors (e1, e2, e3, e4)
 * [5-10]   : bivectors (e12, e13, e14, e23, e24, e34)
 * [11-14]  : trivectors (e123, e124, e134, e234)
 * [15]     : pseudoscalar (e1234)
 */
export type MultivectorComponents = Float64Array;

/**
 * Rotor: An even-grade multivector used for rotations.
 * R = s + B where s is scalar and B is bivector.
 * Must satisfy ||R|| = 1 (unit norm).
 */
export interface Rotor {
    readonly scalar: number;
    readonly bivector: Bivector4D;
    readonly isUnit: boolean;
}

/**
 * Spinor: A half-angle representation for continuous rotations.
 * Used for interpolating between orientations.
 */
export interface Spinor {
    readonly left: Quaternion;
    readonly right: Quaternion;
}

// =============================================================================
// TOPOLOGICAL TYPES
// =============================================================================

/**
 * A vertex in the 24-Cell lattice.
 * The 24-Cell has 24 vertices, all permutations of (±1, ±1, 0, 0).
 */
export interface LatticeVertex {
    readonly id: number;
    readonly coordinates: Vector4D;
    readonly neighbors: number[]; // IDs of adjacent vertices
}

/**
 * A cell (3-face) of the 24-Cell.
 * Each cell is an octahedron with 6 vertices.
 */
export interface LatticeCell {
    readonly id: number;
    readonly vertices: number[]; // 6 vertex IDs
    readonly centroid: Vector4D;
}

/**
 * Voronoi region around a lattice vertex.
 * Represents the "concept cell" in the Orthocognitum.
 */
export interface VoronoiRegion {
    readonly vertexId: number;
    readonly center: Vector4D;
    readonly radius: number; // Distance to nearest boundary
}

/**
 * Result of a convexity check against the Orthocognitum.
 */
export interface ConvexityResult {
    /** Whether the point lies within the convex hull */
    readonly isValid: boolean;
    /** Coherence score: 0.0 (boundary) to 1.0 (centroid) */
    readonly coherence: number;
    /** Nearest lattice vertex ID */
    readonly nearestVertex: number;
    /** Distance to nearest vertex */
    readonly distance: number;
    /** The centroid of active concepts */
    readonly centroid: Vector4D;
    /** Active concept vertices (k-nearest) */
    readonly activeVertices: number[];
}

/**
 * Supported topology stages for metamorphic reasoning.
 */
export type TopologyStage = 'SIMPLEX' | 'HYPERCUBE' | 'CELL24';

/**
 * Topology provider interface for dynamic manifold selection.
 */
export interface TopologyProvider {
    /** Human-readable name */
    readonly name: string;
    /** Ordered vertex list */
    readonly vertices: Vector4D[];
    /** Neighbor indices per vertex */
    readonly neighbors: number[][];
    /** Circumradius for boundary proximity calculations */
    readonly circumradius: number;
    /** Convexity check against the active manifold */
    checkConvexity(point: Vector4D, kNearest?: number): ConvexityResult;
    /** Coherence score for a position */
    computeCoherence(point: Vector4D, kNearest?: number): number;
}

// =============================================================================
// CAUSAL ENGINE TYPES
// =============================================================================

/**
 * The current state of the geometric reasoning engine.
 */
export interface EngineState {
    /** Current position in the Orthocognitum (4D vector) */
    readonly position: Vector4D;
    /** Current orientation (even multivector / rotor) */
    readonly orientation: Rotor;
    /** Velocity: rate of position change */
    readonly velocity: Vector4D;
    /** Angular velocity: bivector representing rotation rate */
    readonly angularVelocity: Bivector4D;
    /** Timestamp of last update (ms) */
    readonly timestamp: number;
}

/**
 * Force applied to the engine (input).
 * Forces generate torque via wedge product with current state.
 */
export interface Force {
    /** Linear force component (translational) */
    readonly linear: Vector4D;
    /** Rotational force component (bivector) */
    readonly rotational: Bivector4D;
    /** Magnitude of the force */
    readonly magnitude: number;
    /** Source identifier for telemetry */
    readonly source: string;
}

/**
 * Torque generated by force application.
 * Torque = State ∧ Force (wedge product creating rotation plane)
 */
export interface Torque {
    /** The bivector defining the plane of rotation */
    readonly plane: Bivector4D;
    /** Magnitude of the torque */
    readonly magnitude: number;
    /** Angular acceleration this produces */
    readonly angularAcceleration: Bivector4D;
}

/**
 * Result of a physics update step.
 */
export interface UpdateResult {
    /** New state after update */
    readonly state: EngineState;
    /** Torque that was applied */
    readonly torque: Torque;
    /** Validity check result */
    readonly convexity: ConvexityResult;
    /** Delta time of this update (seconds) */
    readonly deltaTime: number;
    /** Was the state clamped to valid region? */
    readonly wasClamped: boolean;
}

// =============================================================================
// TELEMETRY TYPES
// =============================================================================

/**
 * Telemetry event for observability and debugging.
 */
export interface TelemetryEvent {
    readonly timestamp: number;
    readonly eventType: TelemetryEventType;
    readonly payload: Record<string, unknown>;
}

export enum TelemetryEventType {
    STATE_UPDATE = 'STATE_UPDATE',
    FORCE_APPLIED = 'FORCE_APPLIED',
    TOPOLOGY_VIOLATION = 'TOPOLOGY_VIOLATION',
    COHERENCE_CHANGE = 'COHERENCE_CHANGE',
    LATTICE_TRANSITION = 'LATTICE_TRANSITION',
    ENGINE_INITIALIZED = 'ENGINE_INITIALIZED',
    ENGINE_RESET = 'ENGINE_RESET'
}

/**
 * Telemetry stream subscriber callback.
 */
export type TelemetrySubscriber = (event: TelemetryEvent) => void;

// =============================================================================
// ENGINE CONFIGURATION
// =============================================================================

/**
 * Configuration for the Chronomorphic Polytopal Engine.
 */
export interface EngineConfig {
    /** Inertia coefficient: resistance to state change (0-1) */
    readonly inertia: number;
    /** Damping coefficient: velocity decay rate (0-1) */
    readonly damping: number;
    /** Maximum angular velocity magnitude */
    readonly maxAngularVelocity: number;
    /** Maximum linear velocity magnitude */
    readonly maxLinearVelocity: number;
    /** Number of nearest neighbors for convexity check */
    readonly kNearest: number;
    /** Coherence threshold for topology violation */
    readonly coherenceThreshold: number;
    /** Whether to clamp state to valid region automatically */
    readonly autoClamp: boolean;
    /** Fixed timestep for deterministic physics (seconds) */
    readonly fixedTimestep: number;
}

/**
 * Default engine configuration.
 */
export const DEFAULT_ENGINE_CONFIG: EngineConfig = {
    inertia: 0.92,
    damping: 0.15,
    maxAngularVelocity: Math.PI * 2,
    maxLinearVelocity: 2.0,
    kNearest: 4,
    coherenceThreshold: 0.3,
    autoClamp: true,
    fixedTimestep: 1 / 60
} as const;

// =============================================================================
// UTILITY TYPES
// =============================================================================

/**
 * Mathematical constants for the engine.
 */
export const MATH_CONSTANTS = {
    /** Machine epsilon for floating point comparisons */
    EPSILON: 1e-10,
    /** Golden ratio (used in 600-cell construction) */
    PHI: (1 + Math.sqrt(5)) / 2,
    /** 1/sqrt(2) (used in 24-cell vertices) */
    SQRT2_INV: 1 / Math.sqrt(2),
    /** 2*PI */
    TAU: Math.PI * 2
} as const;

/**
 * Basis bivector indices in Cl(4,0).
 */
export const BIVECTOR_BASIS = {
    E12: 0,
    E13: 1,
    E14: 2,
    E23: 3,
    E24: 4,
    E34: 5
} as const;

/**
 * Rotation plane names for SO(4).
 */
export const ROTATION_PLANES = [
    'XY', 'XZ', 'XW', 'YZ', 'YW', 'ZW'
] as const;

export type RotationPlane = typeof ROTATION_PLANES[number];
