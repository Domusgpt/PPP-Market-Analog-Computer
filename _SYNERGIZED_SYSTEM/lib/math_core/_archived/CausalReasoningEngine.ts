/**
 * Causal Reasoning Engine
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 *
 * Implements geometric causal reasoning through force-torque-rotor dynamics.
 *
 * Core Principles (from PPP White Paper):
 * - "Reasoning is Rotation": Logical inference = applying rotor R to state S
 * - "Force ∧ State = Torque": Input generates rotation via wedge product
 * - "Unitary Update: R·S·R̃": Transformations preserve norm (truth value)
 *
 * The Three Causal Constraints (Gärdenfors):
 * 1. MONOTONICITY: Larger forces → larger results
 * 2. CONTINUITY: Small force changes → small result changes
 * 3. CONVEXITY: Intermediate forces → intermediate results
 */

import {
  Vector4D,
  Bivector4D,
  Rotor,
  EngineState,
  EngineConfig,
  Force,
  Torque,
  UpdateResult,
  ConvexityResult,
  DEFAULT_ENGINE_CONFIG,
  TelemetryEvent,
  TelemetryEventType,
  TelemetrySubscriber,
  MATH_CONSTANTS
} from '../types/index.js';

import {
  dot,
  magnitude,
  normalize,
  scale,
  add,
  subtract,
  wedge,
  bivectorMagnitude,
  normalizeBivector,
  scaleBivector,
  addBivector,
  createRotor,
  rotorMultiply,
  normalizeRotor,
  applyRotorToVector
} from '../math/GeometricAlgebra.js';

import { Lattice24, getDefaultLattice } from '../topology/Lattice24.js';

// =============================================================================
// CONSTANTS
// =============================================================================

const VELOCITY_EPSILON = 1e-8;
const ANGULAR_VELOCITY_EPSILON = 1e-8;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Clamp a value between min and max.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Apply damping to a value.
 */
function applyDamping(value: number, damping: number, dt: number): number {
  return value * Math.pow(1 - damping, dt * 60);
}

/**
 * Clamp vector magnitude.
 */
function clampVectorMagnitude(v: Vector4D, maxMag: number): Vector4D {
  const mag = magnitude(v);
  if (mag > maxMag) {
    return scale(v, maxMag / mag);
  }
  return v;
}

/**
 * Clamp bivector magnitude.
 */
function clampBivectorMagnitude(b: Bivector4D, maxMag: number): Bivector4D {
  const mag = bivectorMagnitude(b);
  if (mag > maxMag) {
    return scaleBivector(b, maxMag / mag);
  }
  return b;
}

// =============================================================================
// CAUSAL REASONING ENGINE
// =============================================================================

/**
 * CausalReasoningEngine implements physics-based geometric reasoning.
 *
 * Usage:
 * ```typescript
 * const engine = new CausalReasoningEngine();
 * engine.applyForce({ linear: [1, 0, 0, 0], rotational: [0,0,0,0,0,0], magnitude: 1, source: 'input' });
 * const result = engine.update(1/60);
 * console.log(`New position: ${result.state.position}`);
 * ```
 */
export class CausalReasoningEngine {
  private _state: EngineState;
  private _config: EngineConfig;
  private _lattice: Lattice24;
  private _pendingForces: Force[];
  private _subscribers: Set<TelemetrySubscriber>;
  private _updateCount: number;
  private _accumulator: number;

  constructor(config: Partial<EngineConfig> = {}, lattice?: Lattice24) {
    this._config = { ...DEFAULT_ENGINE_CONFIG, ...config };
    this._lattice = lattice ?? getDefaultLattice();
    this._state = this._createInitialState();
    this._pendingForces = [];
    this._subscribers = new Set();
    this._updateCount = 0;
    this._accumulator = 0;
  }

  // =========================================================================
  // INITIALIZATION
  // =========================================================================

  private _createInitialState(): EngineState {
    return {
      position: [0, 0, 0, 0],
      orientation: { scalar: 1, bivector: [0, 0, 0, 0, 0, 0], isUnit: true },
      velocity: [0, 0, 0, 0],
      angularVelocity: [0, 0, 0, 0, 0, 0],
      timestamp: Date.now()
    };
  }

  // =========================================================================
  // ACCESSORS
  // =========================================================================

  get state(): EngineState { return this._state; }
  get config(): EngineConfig { return this._config; }
  get lattice(): Lattice24 { return this._lattice; }
  get updateCount(): number { return this._updateCount; }

  // =========================================================================
  // FORCE APPLICATION
  // =========================================================================

  /**
   * Apply a force to the engine.
   * Forces are accumulated and processed during the next update.
   */
  applyForce(force: Force): void {
    this._pendingForces.push(force);

    this._emitEvent(TelemetryEventType.FORCE_APPLIED, {
      force: {
        linear: [...force.linear],
        rotational: [...force.rotational],
        magnitude: force.magnitude,
        source: force.source
      }
    });
  }

  /**
   * Apply a linear force (convenience method).
   */
  applyLinearForce(direction: Vector4D, magnitude: number, source: string = 'linear'): void {
    const normalized = normalize(direction);
    this.applyForce({
      linear: scale(normalized, magnitude),
      rotational: [0, 0, 0, 0, 0, 0],
      magnitude,
      source
    });
  }

  /**
   * Apply a rotational force (convenience method).
   */
  applyRotationalForce(bivector: Bivector4D, magnitude: number, source: string = 'rotational'): void {
    const normalized = normalizeBivector(bivector);
    this.applyForce({
      linear: [0, 0, 0, 0],
      rotational: scaleBivector(normalized, magnitude),
      magnitude,
      source
    });
  }

  /**
   * Apply an impulse (instant velocity change).
   */
  applyImpulse(linearImpulse: Vector4D, angularImpulse: Bivector4D): void {
    this._state = {
      ...this._state,
      velocity: add(this._state.velocity, linearImpulse),
      angularVelocity: addBivector(this._state.angularVelocity, angularImpulse)
    };
  }

  // =========================================================================
  // TORQUE COMPUTATION
  // =========================================================================

  /**
   * Compute torque from force and current state.
   * τ = position ∧ force (wedge product creates rotation plane)
   *
   * This implements "Context is constructed by the wedge product" -
   * the rotation plane emerges from the relationship between state and input.
   */
  private _computeTorque(force: Force): Torque {
    // Linear force contribution: position ∧ force
    const linearTorque = wedge(this._state.position, force.linear);

    // Combined with direct rotational force
    const totalBivector = addBivector(linearTorque, force.rotational);

    // Normalize to get rotation plane
    const mag = bivectorMagnitude(totalBivector);
    const plane = mag > MATH_CONSTANTS.EPSILON
      ? normalizeBivector(totalBivector)
      : [0, 0, 0, 0, 0, 0] as Bivector4D;

    // Angular acceleration = torque / inertia
    const angularAcceleration = scaleBivector(
      totalBivector,
      1 / (this._config.inertia + MATH_CONSTANTS.EPSILON)
    );

    return {
      plane,
      magnitude: mag,
      angularAcceleration
    };
  }

  // =========================================================================
  // PHYSICS UPDATE
  // =========================================================================

  /**
   * Update the engine state.
   * Uses fixed timestep integration for stability.
   */
  update(deltaTime: number): UpdateResult {
    this._accumulator += deltaTime;

    let result: UpdateResult | null = null;

    // Fixed timestep integration
    while (this._accumulator >= this._config.fixedTimestep) {
      result = this._fixedUpdate(this._config.fixedTimestep);
      this._accumulator -= this._config.fixedTimestep;
    }

    // Return last result or create one for current state
    if (!result) {
      result = {
        state: this._state,
        torque: { plane: [0, 0, 0, 0, 0, 0], magnitude: 0, angularAcceleration: [0, 0, 0, 0, 0, 0] },
        convexity: this._lattice.checkConvexity(this._state.position, this._config.kNearest),
        deltaTime,
        wasClamped: false
      };
    }

    return result;
  }

  /**
   * Fixed timestep physics update.
   */
  private _fixedUpdate(dt: number): UpdateResult {
    this._updateCount++;

    // Sum all pending forces
    let totalForce: Force = {
      linear: [0, 0, 0, 0],
      rotational: [0, 0, 0, 0, 0, 0],
      magnitude: 0,
      source: 'combined'
    };

    for (const force of this._pendingForces) {
      totalForce.linear = add(totalForce.linear, force.linear);
      totalForce.rotational = addBivector(totalForce.rotational, force.rotational);
      totalForce.magnitude += force.magnitude;
    }
    this._pendingForces = [];

    // Compute torque
    const torque = this._computeTorque(totalForce);

    // Update angular velocity
    let newAngularVelocity = addBivector(
      this._state.angularVelocity,
      scaleBivector(torque.angularAcceleration, dt)
    );

    // Apply damping
    newAngularVelocity = scaleBivector(
      newAngularVelocity,
      Math.pow(1 - this._config.damping, dt * 60)
    );

    // Clamp angular velocity
    newAngularVelocity = clampBivectorMagnitude(
      newAngularVelocity,
      this._config.maxAngularVelocity
    );

    // Update orientation via rotor
    const angularMag = bivectorMagnitude(newAngularVelocity);
    let newOrientation = this._state.orientation;

    if (angularMag > ANGULAR_VELOCITY_EPSILON) {
      const deltaRotor = createRotor(newAngularVelocity, angularMag * dt);
      newOrientation = normalizeRotor(rotorMultiply(deltaRotor, this._state.orientation));
    }

    // Update linear velocity
    let newVelocity = add(
      this._state.velocity,
      scale(totalForce.linear, dt)
    );

    // Apply damping
    newVelocity = scale(
      newVelocity,
      Math.pow(1 - this._config.damping, dt * 60)
    );

    // Clamp linear velocity
    newVelocity = clampVectorMagnitude(newVelocity, this._config.maxLinearVelocity);

    // Update position
    let newPosition = add(this._state.position, scale(newVelocity, dt));

    // Check convexity
    const convexity = this._lattice.checkConvexity(newPosition, this._config.kNearest);

    // Auto-clamp if configured and outside hull
    let wasClamped = false;
    if (this._config.autoClamp && !convexity.isValid) {
      newPosition = this._lattice.clamp(newPosition);
      wasClamped = true;

      // Reflect velocity at boundary
      const toCenter = subtract([0, 0, 0, 0], newPosition);
      const reflection = dot(newVelocity, normalize(toCenter));
      if (reflection < 0) {
        newVelocity = add(newVelocity, scale(normalize(toCenter), -2 * reflection));
        newVelocity = scale(newVelocity, 0.5); // Energy loss on reflection
      }

      this._emitEvent(TelemetryEventType.TOPOLOGY_VIOLATION, {
        originalPosition: this._state.position,
        clampedPosition: newPosition,
        coherence: convexity.coherence
      });
    }

    // Create new state
    const newState: EngineState = {
      position: newPosition,
      orientation: newOrientation,
      velocity: newVelocity,
      angularVelocity: newAngularVelocity,
      timestamp: Date.now()
    };

    this._state = newState;

    // Emit update event
    this._emitEvent(TelemetryEventType.STATE_UPDATE, {
      position: [...newPosition],
      coherence: convexity.coherence,
      angularMagnitude: angularMag
    });

    return {
      state: newState,
      torque,
      convexity,
      deltaTime: dt,
      wasClamped
    };
  }

  // =========================================================================
  // STATE MANIPULATION
  // =========================================================================

  /**
   * Set the position directly.
   */
  setPosition(position: Vector4D): void {
    const clamped = this._config.autoClamp
      ? this._lattice.clamp(position)
      : position;

    this._state = { ...this._state, position: clamped };
  }

  /**
   * Set the orientation directly.
   */
  setOrientation(orientation: Rotor): void {
    this._state = { ...this._state, orientation: normalizeRotor(orientation) };
  }

  /**
   * Reset to initial state.
   */
  reset(): void {
    this._state = this._createInitialState();
    this._pendingForces = [];
    this._accumulator = 0;

    this._emitEvent(TelemetryEventType.ENGINE_RESET, {});
  }

  /**
   * Reset to a specific vertex.
   */
  resetToVertex(vertexId: number): void {
    const vertex = this._lattice.getVertex(vertexId);
    if (vertex) {
      this._state = {
        ...this._createInitialState(),
        position: [...vertex.coordinates] as Vector4D
      };
    }
  }

  // =========================================================================
  // NAVIGATION
  // =========================================================================

  /**
   * Navigate towards a target position.
   */
  navigateTowards(target: Vector4D, strength: number = 1): void {
    const direction = subtract(target, this._state.position);
    const dist = magnitude(direction);

    if (dist > MATH_CONSTANTS.EPSILON) {
      this.applyLinearForce(direction, strength * dist, 'navigation');
    }
  }

  /**
   * Navigate towards a vertex.
   */
  navigateToVertex(vertexId: number, strength: number = 1): void {
    const vertex = this._lattice.getVertex(vertexId);
    if (vertex) {
      this.navigateTowards(vertex.coordinates, strength);
    }
  }

  /**
   * Navigate towards the nearest vertex.
   */
  navigateToNearestVertex(strength: number = 1): void {
    const nearestId = this._lattice.findNearest(this._state.position);
    this.navigateToVertex(nearestId, strength);
  }

  // =========================================================================
  // ROTATION CONTROL
  // =========================================================================

  /**
   * Rotate in a specific plane.
   */
  rotateInPlane(planeIndex: number, angularVelocity: number): void {
    const bivector: Bivector4D = [0, 0, 0, 0, 0, 0];
    bivector[planeIndex] = 1;
    this.applyRotationalForce(bivector, angularVelocity, `plane_${planeIndex}`);
  }

  /**
   * Stop all rotation.
   */
  stopRotation(): void {
    this._state = {
      ...this._state,
      angularVelocity: [0, 0, 0, 0, 0, 0]
    };
  }

  /**
   * Stop all motion.
   */
  stopAll(): void {
    this._state = {
      ...this._state,
      velocity: [0, 0, 0, 0],
      angularVelocity: [0, 0, 0, 0, 0, 0]
    };
  }

  // =========================================================================
  // QUERY METHODS
  // =========================================================================

  /**
   * Get the current coherence score.
   */
  getCoherence(): number {
    return this._lattice.computeCoherence(this._state.position, this._config.kNearest);
  }

  /**
   * Get the nearest vertex to current position.
   */
  getNearestVertex(): number {
    return this._lattice.findNearest(this._state.position);
  }

  /**
   * Get distance to nearest vertex.
   */
  getDistanceToNearestVertex(): number {
    const nearestId = this._lattice.findNearest(this._state.position);
    const vertex = this._lattice.getVertex(nearestId);
    if (!vertex) return Infinity;

    return magnitude(subtract(this._state.position, vertex.coordinates));
  }

  /**
   * Check if currently at rest.
   */
  isAtRest(): boolean {
    const linearSpeed = magnitude(this._state.velocity);
    const angularSpeed = bivectorMagnitude(this._state.angularVelocity);

    return linearSpeed < VELOCITY_EPSILON && angularSpeed < ANGULAR_VELOCITY_EPSILON;
  }

  // =========================================================================
  // CONFIGURATION
  // =========================================================================

  /**
   * Update configuration.
   */
  configure(config: Partial<EngineConfig>): void {
    this._config = { ...this._config, ...config };
  }

  // =========================================================================
  // TELEMETRY
  // =========================================================================

  subscribe(callback: TelemetrySubscriber): () => void {
    this._subscribers.add(callback);
    return () => this._subscribers.delete(callback);
  }

  unsubscribe(callback: TelemetrySubscriber): void {
    this._subscribers.delete(callback);
  }

  private _emitEvent(eventType: TelemetryEventType, payload: Record<string, unknown>): void {
    const event: TelemetryEvent = {
      timestamp: Date.now(),
      eventType,
      payload: { ...payload, updateCount: this._updateCount }
    };

    for (const subscriber of this._subscribers) {
      try {
        subscriber(event);
      } catch (error) {
        console.error('CausalReasoningEngine telemetry error:', error);
      }
    }
  }

  // =========================================================================
  // STATISTICS
  // =========================================================================

  getStats(): Record<string, unknown> {
    return {
      updateCount: this._updateCount,
      position: [...this._state.position],
      velocity: [...this._state.velocity],
      linearSpeed: magnitude(this._state.velocity),
      angularSpeed: bivectorMagnitude(this._state.angularVelocity),
      coherence: this.getCoherence(),
      nearestVertex: this.getNearestVertex(),
      isAtRest: this.isAtRest(),
      pendingForces: this._pendingForces.length,
      config: { ...this._config }
    };
  }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

export function createCausalReasoningEngine(
  config?: Partial<EngineConfig>
): CausalReasoningEngine {
  return new CausalReasoningEngine(config);
}

export function createCausalReasoningEngineAt(
  position: Vector4D,
  config?: Partial<EngineConfig>
): CausalReasoningEngine {
  const engine = new CausalReasoningEngine(config);
  engine.setPosition(position);
  return engine;
}
