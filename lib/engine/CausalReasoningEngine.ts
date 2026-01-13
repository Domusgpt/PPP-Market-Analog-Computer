/**
 * Causal Reasoning Engine for Geometric Cognition
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module implements the core physics loop of the Chronomorphic Polytopal Engine.
 * It replaces interpolation-based animation with rigorous algebraic physics where:
 *
 *   "Reasoning is Rotation" - Logical inference is applying a rotor R to state S
 *   "Force ∧ State = Torque" - Input generates rotation via wedge product
 *   "Unitary Update: R·S·R~" - Transformations preserve the norm (truth value)
 *
 * The Three Causal Constraints (from Gärdenfors fpsyg-11-00630):
 * 1. MONOTONICITY: Larger forces → larger results (qualitative causal thinking)
 * 2. CONTINUITY: Small force changes → small result changes (action control)
 * 3. CONVEXITY: Intermediate forces → intermediate results (generalization)
 *
 * Architecture:
 * - Forces enter as semantic vectors (from HDC encoder or direct input)
 * - Torque is computed via wedge product with current state
 * - Rotor is derived from torque magnitude and plane
 * - State is updated via sandwich product: S' = R·S·R~
 * - Epistaorthognition validates result against Orthocognitum
 *
 * References:
 * - Gärdenfors (2020) "Events and Causal Mappings in Conceptual Spaces"
 * - PPP White Paper: "Chronomorphic Polytopal Engine"
 * - Hestenes (1999) "New Foundations for Classical Mechanics"
 */

import {
    Vector4D,
    Bivector4D,
    Rotor,
    Force,
    Torque,
    EngineState,
    EngineConfig,
    UpdateResult,
    ConvexityResult,
    TelemetryEvent,
    TelemetryEventType,
    TelemetrySubscriber,
    DEFAULT_ENGINE_CONFIG,
    MATH_CONSTANTS
} from '../../types/index.js';

import {
    Multivector,
    wedge,
    dot,
    centroid,
    normalize,
    magnitude,
    bivectorMagnitude
} from '../math/GeometricAlgebra.js';

import {
    Lattice24,
    getDefaultLattice,
    checkConvexity as latticeCheckConvexity
} from '../topology/Lattice24.js';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Maximum forces that can be queued */
const MAX_FORCE_QUEUE = 100;

/** Minimum timestep to prevent numerical instability */
const MIN_TIMESTEP = 1e-6;

/** Maximum timestep to prevent large jumps */
const MAX_TIMESTEP = 0.1;

// =============================================================================
// STATE INITIALIZATION
// =============================================================================

/**
 * Create initial engine state at the origin.
 */
function createInitialState(): EngineState {
    return {
        position: [0, 0, 0, 0],
        orientation: {
            scalar: 1,
            bivector: [0, 0, 0, 0, 0, 0],
            isUnit: true
        },
        velocity: [0, 0, 0, 0],
        angularVelocity: [0, 0, 0, 0, 0, 0],
        timestamp: Date.now()
    };
}

/**
 * Create state at a specific position.
 */
function createStateAt(position: Vector4D): EngineState {
    return {
        position,
        orientation: {
            scalar: 1,
            bivector: [0, 0, 0, 0, 0, 0],
            isUnit: true
        },
        velocity: [0, 0, 0, 0],
        angularVelocity: [0, 0, 0, 0, 0, 0],
        timestamp: Date.now()
    };
}

// =============================================================================
// FORCE PROCESSING
// =============================================================================

/**
 * Compute torque from force and current state via wedge product.
 *
 * Torque = Position ∧ Force (creates bivector defining rotation plane)
 *
 * This implements the core insight: "Context is constructed by the wedge product"
 * The plane of rotation emerges from the relationship between where we are
 * (state) and what's pushing us (force).
 */
function computeTorque(
    state: EngineState,
    force: Force,
    config: EngineConfig
): Torque {
    // Compute linear torque: position ∧ linear force
    const linearBivector = wedge(state.position, force.linear);

    // Add rotational force component directly
    const totalBivector: Bivector4D = [
        linearBivector[0] + force.rotational[0],
        linearBivector[1] + force.rotational[1],
        linearBivector[2] + force.rotational[2],
        linearBivector[3] + force.rotational[3],
        linearBivector[4] + force.rotational[4],
        linearBivector[5] + force.rotational[5]
    ];

    // Compute magnitude
    const torqueMagnitude = bivectorMagnitude(totalBivector);

    // Apply inertia: actual angular acceleration is inversely proportional to inertia
    // Higher inertia = more resistance to rotation
    const inertiaFactor = 1 - config.inertia;
    const scaledMagnitude = torqueMagnitude * inertiaFactor;

    // Compute angular acceleration (bivector scaled by inertia)
    const angularAcceleration: Bivector4D = torqueMagnitude > MATH_CONSTANTS.EPSILON
        ? [
            totalBivector[0] / torqueMagnitude * scaledMagnitude,
            totalBivector[1] / torqueMagnitude * scaledMagnitude,
            totalBivector[2] / torqueMagnitude * scaledMagnitude,
            totalBivector[3] / torqueMagnitude * scaledMagnitude,
            totalBivector[4] / torqueMagnitude * scaledMagnitude,
            totalBivector[5] / torqueMagnitude * scaledMagnitude
          ]
        : [0, 0, 0, 0, 0, 0];

    return {
        plane: totalBivector,
        magnitude: torqueMagnitude,
        angularAcceleration
    };
}

/**
 * Derive a rotor from angular velocity and timestep.
 *
 * The rotor R = exp(-θ/2 · B) where:
 * - θ is the rotation angle (|ω| * dt)
 * - B is the unit bivector (normalized angular velocity)
 *
 * This gives us the rotation operator for the unitary update.
 */
function angularVelocityToRotor(
    angularVelocity: Bivector4D,
    dt: number
): Multivector {
    // Compute rotation angle: magnitude of angular velocity * time
    const omega = bivectorMagnitude(angularVelocity);
    const angle = omega * dt;

    if (angle < MATH_CONSTANTS.EPSILON) {
        return Multivector.scalar(1); // Identity rotor
    }

    // Create rotor: R = cos(θ/2) - sin(θ/2)·B (with negative for correct direction)
    // Using the bivector as the rotation plane
    return Multivector.rotor(angularVelocity, angle);
}

// =============================================================================
// STATE UPDATE (UNITARY TRANSFORMATION)
// =============================================================================

/**
 * Apply a rotor to the state via sandwich product.
 *
 * S' = R·S·R~ (the unitary update)
 *
 * This preserves the norm of the state vector, ensuring that
 * information is not lost during reasoning.
 */
function applyRotor(
    state: EngineState,
    rotor: Multivector,
    torque: Torque,
    dt: number,
    config: EngineConfig
): EngineState {
    // Convert position to multivector
    const positionMV = Multivector.vector(state.position);

    // Apply sandwich product: R·position·R~
    const rotatedPositionMV = positionMV.sandwich(rotor);
    const newPosition = rotatedPositionMV.vector;

    // Update linear velocity (integrate linear force contribution)
    const newVelocity: Vector4D = [
        state.velocity[0] * (1 - config.damping),
        state.velocity[1] * (1 - config.damping),
        state.velocity[2] * (1 - config.damping),
        state.velocity[3] * (1 - config.damping)
    ];

    // Update angular velocity: integrate angular acceleration, apply damping
    const newAngularVelocity: Bivector4D = [
        (state.angularVelocity[0] + torque.angularAcceleration[0] * dt) * (1 - config.damping),
        (state.angularVelocity[1] + torque.angularAcceleration[1] * dt) * (1 - config.damping),
        (state.angularVelocity[2] + torque.angularAcceleration[2] * dt) * (1 - config.damping),
        (state.angularVelocity[3] + torque.angularAcceleration[3] * dt) * (1 - config.damping),
        (state.angularVelocity[4] + torque.angularAcceleration[4] * dt) * (1 - config.damping),
        (state.angularVelocity[5] + torque.angularAcceleration[5] * dt) * (1 - config.damping)
    ];

    // Clamp angular velocity
    const angMag = bivectorMagnitude(newAngularVelocity);
    const clampedAngularVelocity: Bivector4D = angMag > config.maxAngularVelocity
        ? newAngularVelocity.map(v => v / angMag * config.maxAngularVelocity) as Bivector4D
        : newAngularVelocity;

    // Update orientation: compose current orientation with new rotation
    const currentOrientationMV = Multivector.scalar(state.orientation.scalar)
        .add(Multivector.bivector(state.orientation.bivector));
    const newOrientationMV = rotor.mul(currentOrientationMV).normalized();

    return {
        position: newPosition,
        orientation: newOrientationMV.toRotor(),
        velocity: newVelocity,
        angularVelocity: clampedAngularVelocity,
        timestamp: state.timestamp + dt * 1000
    };
}

/**
 * Apply linear velocity to position.
 * Separate from rotation to allow different integration strategies.
 */
function integrateLinearMotion(
    state: EngineState,
    force: Force,
    dt: number,
    config: EngineConfig
): EngineState {
    // Compute acceleration from force (F = ma, assume m = 1 for simplicity)
    const inertiaFactor = 1 - config.inertia;
    const acceleration: Vector4D = [
        force.linear[0] * inertiaFactor,
        force.linear[1] * inertiaFactor,
        force.linear[2] * inertiaFactor,
        force.linear[3] * inertiaFactor
    ];

    // Update velocity: v' = v + a*dt
    const newVelocity: Vector4D = [
        (state.velocity[0] + acceleration[0] * dt) * (1 - config.damping),
        (state.velocity[1] + acceleration[1] * dt) * (1 - config.damping),
        (state.velocity[2] + acceleration[2] * dt) * (1 - config.damping),
        (state.velocity[3] + acceleration[3] * dt) * (1 - config.damping)
    ];

    // Clamp velocity
    const velMag = magnitude(newVelocity);
    const clampedVelocity: Vector4D = velMag > config.maxLinearVelocity
        ? newVelocity.map(v => v / velMag * config.maxLinearVelocity) as Vector4D
        : newVelocity;

    // Update position: p' = p + v*dt
    const newPosition: Vector4D = [
        state.position[0] + clampedVelocity[0] * dt,
        state.position[1] + clampedVelocity[1] * dt,
        state.position[2] + clampedVelocity[2] * dt,
        state.position[3] + clampedVelocity[3] * dt
    ];

    return {
        ...state,
        position: newPosition,
        velocity: clampedVelocity
    };
}

// =============================================================================
// VALIDATION AND CLAMPING
// =============================================================================

/**
 * Clamp state to valid region if auto-clamping is enabled.
 */
function clampState(
    state: EngineState,
    lattice: Lattice24,
    config: EngineConfig
): { state: EngineState; wasClamped: boolean } {
    if (!config.autoClamp) {
        return { state, wasClamped: false };
    }

    if (lattice.isInside(state.position)) {
        return { state, wasClamped: false };
    }

    // Project position to valid region
    const clampedPosition = lattice.clamp(state.position);

    // Reduce velocity when clamped (collision response)
    const dampenedVelocity: Vector4D = [
        state.velocity[0] * 0.5,
        state.velocity[1] * 0.5,
        state.velocity[2] * 0.5,
        state.velocity[3] * 0.5
    ];

    return {
        state: {
            ...state,
            position: clampedPosition,
            velocity: dampenedVelocity
        },
        wasClamped: true
    };
}

// =============================================================================
// CAUSAL REASONING ENGINE CLASS
// =============================================================================

/**
 * The CausalReasoningEngine implements geometric cognition through physics.
 *
 * Core principles:
 * - State is a 4D vector in the Orthocognitum
 * - Forces (inputs) generate torque via wedge product
 * - Reasoning is rotation via sandwich product R·S·R~
 * - Validity is checked against the 24-Cell lattice
 *
 * Usage:
 * ```typescript
 * const engine = new CausalReasoningEngine();
 * engine.applyForce({ linear: [0.5, 0, 0, 0], ... });
 * const result = engine.update();
 * console.log(`Coherence: ${result.convexity.coherence}`);
 * ```
 */
export class CausalReasoningEngine {
    /** Current engine state */
    private _state: EngineState;

    /** Engine configuration */
    private _config: EngineConfig;

    /** Reference to the 24-Cell lattice */
    private readonly _lattice: Lattice24;

    /** Queue of forces to apply */
    private _forceQueue: Force[];

    /** Accumulated force (sum of queued forces) */
    private _accumulatedForce: Force;

    /** Last update timestamp (for variable timestep) */
    private _lastUpdateTime: number;

    /** Telemetry subscribers */
    private _subscribers: Set<TelemetrySubscriber>;

    /** Update counter for telemetry */
    private _updateCount: number;

    /** Running flag */
    private _isRunning: boolean;

    /** Previous coherence for change detection */
    private _previousCoherence: number;

    /** Previous nearest vertex for transition detection */
    private _previousNearestVertex: number;

    constructor(config: Partial<EngineConfig> = {}, lattice?: Lattice24) {
        this._config = { ...DEFAULT_ENGINE_CONFIG, ...config };
        this._lattice = lattice ?? getDefaultLattice();
        this._state = createInitialState();
        this._forceQueue = [];
        this._accumulatedForce = this._createZeroForce();
        this._lastUpdateTime = performance.now();
        this._subscribers = new Set();
        this._updateCount = 0;
        this._isRunning = false;
        this._previousCoherence = 1.0;
        this._previousNearestVertex = 0;

        this._emitEvent(TelemetryEventType.ENGINE_INITIALIZED, {
            config: this._config,
            initialState: this._state
        });
    }

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    /** Get current configuration */
    get config(): EngineConfig {
        return this._config;
    }

    /** Update configuration */
    setConfig(config: Partial<EngineConfig>): void {
        this._config = { ...this._config, ...config };
    }

    /** Get current state (readonly) */
    get state(): EngineState {
        return this._state;
    }

    /** Get the lattice */
    get lattice(): Lattice24 {
        return this._lattice;
    }

    /** Get running status */
    get isRunning(): boolean {
        return this._isRunning;
    }

    /** Get update count */
    get updateCount(): number {
        return this._updateCount;
    }

    // =========================================================================
    // FORCE APPLICATION
    // =========================================================================

    /**
     * Apply a force to the engine.
     * Forces are queued and processed on the next update.
     */
    applyForce(force: Force): void {
        if (this._forceQueue.length >= MAX_FORCE_QUEUE) {
            // Merge oldest forces to prevent unbounded growth
            this._consolidateForces();
        }

        this._forceQueue.push(force);

        this._emitEvent(TelemetryEventType.FORCE_APPLIED, {
            force,
            queueLength: this._forceQueue.length
        });
    }

    /**
     * Apply a simple linear force.
     */
    applyLinearForce(direction: Vector4D, mag: number = 1): void {
        const normalizedDir = normalize(direction);
        this.applyForce({
            linear: [
                normalizedDir[0] * mag,
                normalizedDir[1] * mag,
                normalizedDir[2] * mag,
                normalizedDir[3] * mag
            ],
            rotational: [0, 0, 0, 0, 0, 0],
            magnitude: mag,
            source: 'linear'
        });
    }

    /**
     * Apply a rotational force in a specific plane.
     */
    applyRotationalForce(planeIndex: number, mag: number = 1): void {
        const rotational: Bivector4D = [0, 0, 0, 0, 0, 0];
        rotational[planeIndex] = mag;

        this.applyForce({
            linear: [0, 0, 0, 0],
            rotational,
            magnitude: Math.abs(mag),
            source: 'rotational'
        });
    }

    /**
     * Apply an impulse (instantaneous force).
     * This bypasses the queue and applies immediately.
     */
    applyImpulse(force: Force): UpdateResult {
        // Store current accumulated force
        const prevAccumulated = this._accumulatedForce;

        // Apply impulse as accumulated force
        this._accumulatedForce = force;

        // Force immediate update
        const result = this._performUpdate(this._config.fixedTimestep);

        // Restore accumulated force
        this._accumulatedForce = prevAccumulated;

        return result;
    }

    /**
     * Clear all queued forces.
     */
    clearForces(): void {
        this._forceQueue = [];
        this._accumulatedForce = this._createZeroForce();
    }

    // =========================================================================
    // UPDATE LOOP
    // =========================================================================

    /**
     * Process one physics update.
     * This is the core "reasoning step" of the engine.
     *
     * @param dt - Optional timestep override (uses config.fixedTimestep if not provided)
     * @returns UpdateResult with new state and validation
     */
    update(dt?: number): UpdateResult {
        // Calculate timestep
        const now = performance.now();
        const elapsed = (now - this._lastUpdateTime) / 1000;
        this._lastUpdateTime = now;

        const timestep = dt ?? Math.min(Math.max(elapsed, MIN_TIMESTEP), MAX_TIMESTEP);

        // Process force queue
        this._processForceQueue();

        // Perform physics update
        return this._performUpdate(timestep);
    }

    /**
     * Run continuous updates at fixed timestep.
     * @param callback - Called after each update
     */
    start(callback?: (result: UpdateResult) => void): void {
        if (this._isRunning) return;
        this._isRunning = true;

        const loop = () => {
            if (!this._isRunning) return;

            const result = this.update(this._config.fixedTimestep);
            callback?.(result);

            requestAnimationFrame(loop);
        };

        requestAnimationFrame(loop);
    }

    /**
     * Stop continuous updates.
     */
    stop(): void {
        this._isRunning = false;
    }

    /**
     * Reset engine to initial state.
     */
    reset(position?: Vector4D): void {
        this._state = position ? createStateAt(position) : createInitialState();
        this.clearForces();
        this._lastUpdateTime = performance.now();
        this._updateCount = 0;
        this._previousCoherence = 1.0;
        this._previousNearestVertex = 0;

        this._emitEvent(TelemetryEventType.ENGINE_RESET, {
            state: this._state
        });
    }

    /**
     * Set state directly (for initialization or external control).
     */
    setState(state: Partial<EngineState>): void {
        this._state = {
            ...this._state,
            ...state,
            timestamp: Date.now()
        };
    }

    // =========================================================================
    // TELEMETRY
    // =========================================================================

    /**
     * Subscribe to telemetry events.
     */
    subscribe(callback: TelemetrySubscriber): () => void {
        this._subscribers.add(callback);
        return () => this._subscribers.delete(callback);
    }

    /**
     * Unsubscribe from telemetry events.
     */
    unsubscribe(callback: TelemetrySubscriber): void {
        this._subscribers.delete(callback);
    }

    /**
     * Get current convexity/validity status.
     */
    checkConvexity(): ConvexityResult {
        return this._lattice.checkConvexity(this._state.position, this._config.kNearest);
    }

    // =========================================================================
    // INTERNAL UPDATE LOGIC
    // =========================================================================

    private _performUpdate(dt: number): UpdateResult {
        this._updateCount++;

        // 1. Compute torque from accumulated force
        const torque = computeTorque(this._state, this._accumulatedForce, this._config);

        // 2. Derive rotor from angular velocity
        // Combine existing angular velocity with new torque-induced acceleration
        const effectiveAngularVelocity: Bivector4D = [
            this._state.angularVelocity[0] + torque.angularAcceleration[0] * dt,
            this._state.angularVelocity[1] + torque.angularAcceleration[1] * dt,
            this._state.angularVelocity[2] + torque.angularAcceleration[2] * dt,
            this._state.angularVelocity[3] + torque.angularAcceleration[3] * dt,
            this._state.angularVelocity[4] + torque.angularAcceleration[4] * dt,
            this._state.angularVelocity[5] + torque.angularAcceleration[5] * dt
        ];

        const rotor = angularVelocityToRotor(effectiveAngularVelocity, dt);

        // 3. Apply rotor via sandwich product (rotational update)
        let newState = applyRotor(this._state, rotor, torque, dt, this._config);

        // 4. Integrate linear motion
        newState = integrateLinearMotion(newState, this._accumulatedForce, dt, this._config);

        // 5. Validate against topology (Epistaorthognition)
        const convexity = this._lattice.checkConvexity(newState.position, this._config.kNearest);

        // 6. Clamp to valid region if needed
        const { state: clampedState, wasClamped } = clampState(newState, this._lattice, this._config);
        newState = clampedState;

        // 7. Emit telemetry events
        this._emitStateUpdate(newState, convexity, torque, dt);

        // Check for topology violations
        if (!convexity.isValid) {
            this._emitEvent(TelemetryEventType.TOPOLOGY_VIOLATION, {
                position: newState.position,
                coherence: convexity.coherence,
                nearestVertex: convexity.nearestVertex
            });
        }

        // Check for coherence changes
        const coherenceDelta = Math.abs(convexity.coherence - this._previousCoherence);
        if (coherenceDelta > 0.1) {
            this._emitEvent(TelemetryEventType.COHERENCE_CHANGE, {
                previous: this._previousCoherence,
                current: convexity.coherence,
                delta: coherenceDelta
            });
        }
        this._previousCoherence = convexity.coherence;

        // Check for lattice transitions
        if (convexity.nearestVertex !== this._previousNearestVertex) {
            this._emitEvent(TelemetryEventType.LATTICE_TRANSITION, {
                from: this._previousNearestVertex,
                to: convexity.nearestVertex,
                position: newState.position
            });
        }
        this._previousNearestVertex = convexity.nearestVertex;

        // 8. Update state
        this._state = newState;

        // 9. Clear accumulated force for next frame
        this._accumulatedForce = this._createZeroForce();

        return {
            state: this._state,
            torque,
            convexity,
            deltaTime: dt,
            wasClamped
        };
    }

    private _processForceQueue(): void {
        if (this._forceQueue.length === 0) return;

        // Sum all queued forces
        let totalLinear: Vector4D = [0, 0, 0, 0];
        let totalRotational: Bivector4D = [0, 0, 0, 0, 0, 0];
        let totalMagnitude = 0;

        for (const force of this._forceQueue) {
            totalLinear[0] += force.linear[0];
            totalLinear[1] += force.linear[1];
            totalLinear[2] += force.linear[2];
            totalLinear[3] += force.linear[3];

            totalRotational[0] += force.rotational[0];
            totalRotational[1] += force.rotational[1];
            totalRotational[2] += force.rotational[2];
            totalRotational[3] += force.rotational[3];
            totalRotational[4] += force.rotational[4];
            totalRotational[5] += force.rotational[5];

            totalMagnitude += force.magnitude;
        }

        this._accumulatedForce = {
            linear: totalLinear,
            rotational: totalRotational,
            magnitude: totalMagnitude,
            source: 'accumulated'
        };

        // Clear the queue
        this._forceQueue = [];
    }

    private _consolidateForces(): void {
        // Merge first half of queue into single force
        const halfLength = Math.floor(this._forceQueue.length / 2);
        const toConsolidate = this._forceQueue.splice(0, halfLength);

        let totalLinear: Vector4D = [0, 0, 0, 0];
        let totalRotational: Bivector4D = [0, 0, 0, 0, 0, 0];
        let totalMagnitude = 0;

        for (const force of toConsolidate) {
            totalLinear[0] += force.linear[0];
            totalLinear[1] += force.linear[1];
            totalLinear[2] += force.linear[2];
            totalLinear[3] += force.linear[3];

            totalRotational[0] += force.rotational[0];
            totalRotational[1] += force.rotational[1];
            totalRotational[2] += force.rotational[2];
            totalRotational[3] += force.rotational[3];
            totalRotational[4] += force.rotational[4];
            totalRotational[5] += force.rotational[5];

            totalMagnitude += force.magnitude;
        }

        // Insert consolidated force at beginning
        this._forceQueue.unshift({
            linear: totalLinear,
            rotational: totalRotational,
            magnitude: totalMagnitude,
            source: 'consolidated'
        });
    }

    private _createZeroForce(): Force {
        return {
            linear: [0, 0, 0, 0],
            rotational: [0, 0, 0, 0, 0, 0],
            magnitude: 0,
            source: 'none'
        };
    }

    private _emitEvent(eventType: TelemetryEventType, payload: Record<string, unknown>): void {
        const event: TelemetryEvent = {
            timestamp: Date.now(),
            eventType,
            payload
        };

        for (const subscriber of this._subscribers) {
            try {
                subscriber(event);
            } catch (error) {
                console.error('Telemetry subscriber error:', error);
            }
        }
    }

    private _emitStateUpdate(
        state: EngineState,
        convexity: ConvexityResult,
        torque: Torque,
        dt: number
    ): void {
        this._emitEvent(TelemetryEventType.STATE_UPDATE, {
            updateCount: this._updateCount,
            position: state.position,
            orientation: state.orientation,
            velocity: state.velocity,
            angularVelocity: state.angularVelocity,
            coherence: convexity.coherence,
            isValid: convexity.isValid,
            nearestVertex: convexity.nearestVertex,
            torqueMagnitude: torque.magnitude,
            deltaTime: dt
        });
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /**
     * Get engine statistics.
     */
    getStats(): Record<string, unknown> {
        const convexity = this.checkConvexity();
        return {
            updateCount: this._updateCount,
            isRunning: this._isRunning,
            position: this._state.position,
            velocity: magnitude(this._state.velocity),
            angularVelocity: bivectorMagnitude(this._state.angularVelocity),
            coherence: convexity.coherence,
            isValid: convexity.isValid,
            nearestVertex: convexity.nearestVertex,
            forceQueueLength: this._forceQueue.length,
            subscriberCount: this._subscribers.size
        };
    }

    /**
     * Create a snapshot of current state for serialization.
     */
    snapshot(): {
        state: EngineState;
        config: EngineConfig;
        convexity: ConvexityResult;
        updateCount: number;
    } {
        return {
            state: { ...this._state },
            config: { ...this._config },
            convexity: this.checkConvexity(),
            updateCount: this._updateCount
        };
    }

    /**
     * Restore from a snapshot.
     */
    restore(snapshot: { state: EngineState; config?: EngineConfig }): void {
        this._state = { ...snapshot.state };
        if (snapshot.config) {
            this._config = { ...snapshot.config };
        }
        this.clearForces();
    }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/**
 * Create a new CausalReasoningEngine with default configuration.
 */
export function createEngine(config?: Partial<EngineConfig>): CausalReasoningEngine {
    return new CausalReasoningEngine(config);
}

/**
 * Create an engine initialized at a specific position.
 */
export function createEngineAt(
    position: Vector4D,
    config?: Partial<EngineConfig>
): CausalReasoningEngine {
    const engine = new CausalReasoningEngine(config);
    engine.reset(position);
    return engine;
}

/**
 * Create an engine at a random valid position.
 */
export function createEngineRandom(config?: Partial<EngineConfig>): CausalReasoningEngine {
    const engine = new CausalReasoningEngine(config);
    const randomPos = engine.lattice.randomInside();
    engine.reset(randomPos);
    return engine;
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
    createInitialState,
    createStateAt,
    computeTorque,
    angularVelocityToRotor,
    applyRotor,
    integrateLinearMotion,
    clampState,
    MIN_TIMESTEP,
    MAX_TIMESTEP,
    MAX_FORCE_QUEUE
};
