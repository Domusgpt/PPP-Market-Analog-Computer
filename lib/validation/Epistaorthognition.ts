/**
 * Epistaorthognition - Cognitive Validity Validation Module
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * "Epistaorthognition" (from Greek: episteme = knowledge, orthos = correct/straight,
 * gnosis = cognition) is the process of validating that reasoning stays within
 * the bounds of coherent thought - the Orthocognitum.
 *
 * This module provides:
 * 1. State validation against the 24-Cell topological constraints
 * 2. Coherence metrics measuring alignment with concept lattice
 * 3. Anomaly detection for reasoning drift
 * 4. Correction suggestions to return to valid regions
 *
 * Key Metrics:
 * - Coherence: Distance-weighted alignment with k-nearest lattice vertices
 * - Stability: Rate of change of coherence over time
 * - Boundary Proximity: How close to leaving the Orthocognitum
 * - Concept Membership: Which Voronoi region(s) the state occupies
 *
 * Use Case: AI safety auditing - verify reasoning trajectories stay within valid bounds.
 *
 * References:
 * - PPP White Paper: "The Orthocognitum is the Shape of the Known"
 * - Gärdenfors "Conceptual Spaces" (2000)
 */

import {
    Vector4D,
    Bivector4D,
    EngineState,
    ConvexityResult,
    TopologyProvider,
    MATH_CONSTANTS
} from '../../types/index.js';

import {
    dot,
    magnitude,
    normalize,
    centroid,
    bivectorMagnitude
} from '../math/GeometricAlgebra.js';

import {
    Lattice24,
    getDefaultLattice,
    isInsideConvexHull,
    CIRCUMRADIUS
} from '../topology/Lattice24.js';

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

/**
 * Result of a full state validation.
 */
export interface ValidationResult {
    /** Overall validity: is the state within acceptable bounds? */
    readonly isValid: boolean;

    /** Coherence score: 0.0 (invalid) to 1.0 (perfectly coherent) */
    readonly coherence: number;

    /** Stability score: 0.0 (unstable) to 1.0 (stable) */
    readonly stability: number;

    /** Boundary proximity: 0.0 (center) to 1.0 (at boundary) */
    readonly boundaryProximity: number;

    /** Concept membership: Voronoi regions the state belongs to */
    readonly conceptMembership: ConceptMembership;

    /** Detailed convexity information */
    readonly convexity: ConvexityResult;

    /** Validation timestamp */
    readonly timestamp: number;

    /** Warning messages if any */
    readonly warnings: string[];
}

/**
 * Batch-level metrics for metamorphic topology decisions.
 */
export interface BatchMetrics {
    /** Mean coherence across batch */
    readonly coherenceMean: number;
    /** Variance of coherence across batch */
    readonly coherenceVariance: number;
    /** 95th percentile boundary proximity */
    readonly boundaryRiskP95: number;
    /** Minimum coherence observed */
    readonly minCoherence: number;
    /** Maximum coherence observed */
    readonly maxCoherence: number;
}

/**
 * Concept membership information.
 */
export interface ConceptMembership {
    /** Primary concept (nearest vertex) */
    readonly primary: number;

    /** Secondary concepts (k-nearest vertices) */
    readonly secondary: number[];

    /** Weights for each concept (distance-based) */
    readonly weights: number[];

    /** Interpolated concept position */
    readonly interpolated: Vector4D;
}

/**
 * Anomaly report for trajectory analysis.
 */
export interface AnomalyReport {
    /** Were anomalies detected? */
    readonly hasAnomalies: boolean;

    /** Anomaly severity: 0.0 (none) to 1.0 (critical) */
    readonly severity: number;

    /** List of detected anomalies */
    readonly anomalies: Anomaly[];

    /** Trajectory statistics */
    readonly statistics: TrajectoryStatistics;

    /** Recommended actions */
    readonly recommendations: string[];
}

/**
 * Single anomaly detection.
 */
export interface Anomaly {
    /** Type of anomaly */
    readonly type: AnomalyType;

    /** Index in trajectory where anomaly occurred */
    readonly index: number;

    /** Severity: 0.0 to 1.0 */
    readonly severity: number;

    /** Description of the anomaly */
    readonly description: string;

    /** State at anomaly point */
    readonly state: EngineState;
}

/**
 * Types of anomalies that can be detected.
 */
export enum AnomalyType {
    /** Coherence dropped below threshold */
    COHERENCE_DROP = 'COHERENCE_DROP',

    /** State left valid region */
    BOUNDARY_VIOLATION = 'BOUNDARY_VIOLATION',

    /** Sudden large state change */
    DISCONTINUITY = 'DISCONTINUITY',

    /** Coherence oscillating rapidly */
    INSTABILITY = 'INSTABILITY',

    /** State stuck in same region */
    STAGNATION = 'STAGNATION',

    /** Velocity exceeds safe limits */
    VELOCITY_SPIKE = 'VELOCITY_SPIKE',

    /** Angular velocity exceeds limits */
    ROTATION_SPIKE = 'ROTATION_SPIKE'
}

/**
 * Trajectory statistics.
 */
export interface TrajectoryStatistics {
    /** Number of states analyzed */
    readonly count: number;

    /** Average coherence */
    readonly meanCoherence: number;

    /** Coherence standard deviation */
    readonly stdCoherence: number;

    /** Minimum coherence observed */
    readonly minCoherence: number;

    /** Maximum coherence observed */
    readonly maxCoherence: number;

    /** Number of boundary violations */
    readonly boundaryViolations: number;

    /** Number of lattice transitions */
    readonly latticeTransitions: number;

    /** Total path length */
    readonly pathLength: number;

    /** Average velocity */
    readonly meanVelocity: number;
}

/**
 * Correction vector to return to valid region.
 */
export interface CorrectionVector {
    /** Linear correction (direction to move) */
    readonly linear: Vector4D;

    /** Rotational correction (rotation to apply) */
    readonly rotational: Bivector4D;

    /** Magnitude of correction needed */
    readonly magnitude: number;

    /** Target position (projected valid point) */
    readonly target: Vector4D;

    /** Target vertex (nearest valid concept) */
    readonly targetVertex: number;

    /** Estimated steps to reach valid region */
    readonly estimatedSteps: number;

    /** Urgency: 0.0 (no rush) to 1.0 (critical) */
    readonly urgency: number;
}

/**
 * Configuration for validation.
 */
export interface ValidationConfig {
    /** Coherence threshold for validity */
    readonly coherenceThreshold: number;

    /** Stability threshold for validity */
    readonly stabilityThreshold: number;

    /** Boundary proximity warning threshold */
    readonly boundaryWarningThreshold: number;

    /** Number of neighbors for concept membership */
    readonly kNearest: number;

    /** Velocity spike threshold */
    readonly velocitySpikeThreshold: number;

    /** Angular velocity spike threshold */
    readonly angularVelocitySpikeThreshold: number;

    /** Discontinuity threshold (position jump) */
    readonly discontinuityThreshold: number;

    /** Stagnation threshold (steps without movement) */
    readonly stagnationThreshold: number;

    /** History length for stability calculation */
    readonly stabilityWindow: number;
}

/**
 * Default validation configuration.
 */
export const DEFAULT_VALIDATION_CONFIG: ValidationConfig = {
    coherenceThreshold: 0.3,
    stabilityThreshold: 0.5,
    boundaryWarningThreshold: 0.8,
    kNearest: 4,
    velocitySpikeThreshold: 2.0,
    angularVelocitySpikeThreshold: Math.PI * 2,
    discontinuityThreshold: 0.5,
    stagnationThreshold: 50,
    stabilityWindow: 10
} as const;

// =============================================================================
// EPISTAORTHOGNITION CLASS
// =============================================================================

/**
 * Epistaorthognition validator for cognitive state validation.
 *
 * Usage:
 * ```typescript
 * const validator = new Epistaorthognition();
 * const result = validator.validateState(engineState);
 * if (!result.isValid) {
 *   const correction = validator.suggestCorrection(engineState);
 *   engine.applyForce(correction.linear);
 * }
 * ```
 */
export class Epistaorthognition {
    /** Reference to the 24-Cell lattice */
    private readonly _lattice: Lattice24;

    /** Validation configuration */
    private _config: ValidationConfig;

    /** History of coherence values for stability calculation */
    private _coherenceHistory: number[];

    /** History of positions for trajectory analysis */
    private _positionHistory: Vector4D[];

    /** Last validation result */
    private _lastValidation: ValidationResult | null;

    constructor(config: Partial<ValidationConfig> = {}, lattice?: Lattice24) {
        this._config = { ...DEFAULT_VALIDATION_CONFIG, ...config };
        this._lattice = lattice ?? getDefaultLattice();
        this._coherenceHistory = [];
        this._positionHistory = [];
        this._lastValidation = null;
    }

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    /** Get current configuration */
    get config(): ValidationConfig {
        return this._config;
    }

    /** Update configuration */
    setConfig(config: Partial<ValidationConfig>): void {
        this._config = { ...this._config, ...config };
    }

    /** Get the lattice */
    get lattice(): Lattice24 {
        return this._lattice;
    }

    /** Get last validation result */
    get lastValidation(): ValidationResult | null {
        return this._lastValidation;
    }

    // =========================================================================
    // CORE VALIDATION FUNCTIONS
    // =========================================================================

    /**
     * Validate the current engine state.
     * This is the primary validation function.
     *
     * @param state - Engine state to validate
     * @returns Full validation result
     */
    validateState(state: EngineState): ValidationResult {
        const warnings: string[] = [];

        // 1. Check convexity (inside 24-cell?)
        const convexity = this._lattice.checkConvexity(
            state.position,
            this._config.kNearest
        );

        // 2. Compute coherence
        const coherence = convexity.coherence;

        // 3. Compute stability (rate of coherence change)
        this._coherenceHistory.push(coherence);
        if (this._coherenceHistory.length > this._config.stabilityWindow) {
            this._coherenceHistory.shift();
        }
        const stability = this._computeStability();

        // 4. Compute boundary proximity
        const boundaryProximity = this._computeBoundaryProximity(state.position);

        // 5. Compute concept membership
        const conceptMembership = this._computeConceptMembership(
            state.position,
            convexity
        );

        // 6. Generate warnings
        if (coherence < this._config.coherenceThreshold) {
            warnings.push(`Low coherence: ${coherence.toFixed(3)} < ${this._config.coherenceThreshold}`);
        }

        if (stability < this._config.stabilityThreshold) {
            warnings.push(`Unstable: stability ${stability.toFixed(3)} < ${this._config.stabilityThreshold}`);
        }

        if (boundaryProximity > this._config.boundaryWarningThreshold) {
            warnings.push(`Near boundary: proximity ${boundaryProximity.toFixed(3)}`);
        }

        if (!convexity.isValid) {
            warnings.push('Outside valid region (Orthocognitum violated)');
        }

        // 7. Determine overall validity
        const isValid = convexity.isValid &&
            coherence >= this._config.coherenceThreshold &&
            stability >= this._config.stabilityThreshold * 0.5; // More lenient on stability

        // 8. Store position history
        this._positionHistory.push(state.position);
        if (this._positionHistory.length > 100) {
            this._positionHistory.shift();
        }

        const result: ValidationResult = {
            isValid,
            coherence,
            stability,
            boundaryProximity,
            conceptMembership,
            convexity,
            timestamp: Date.now(),
            warnings
        };

        this._lastValidation = result;
        return result;
    }

    /**
     * Compute coherence for a position.
     * Coherence measures how well the state aligns with the local concept structure.
     *
     * @param position - 4D position vector
     * @returns Coherence score 0.0 to 1.0
     */
    computeCoherence(position: Vector4D): number {
        const convexity = this._lattice.checkConvexity(position, this._config.kNearest);
        return convexity.coherence;
    }

    /**
     * Detect anomalies in a trajectory.
     * Analyzes a sequence of states for reasoning drift and violations.
     *
     * @param trajectory - Array of engine states to analyze
     * @returns Anomaly report with detected issues
     */
    detectAnomaly(trajectory: EngineState[]): AnomalyReport {
        if (trajectory.length === 0) {
            return this._createEmptyAnomalyReport();
        }

        const anomalies: Anomaly[] = [];
        const coherences: number[] = [];
        const velocities: number[] = [];
        let boundaryViolations = 0;
        let latticeTransitions = 0;
        let pathLength = 0;
        let lastVertex = -1;

        // Analyze each state
        for (let i = 0; i < trajectory.length; i++) {
            const state = trajectory[i];
            const convexity = this._lattice.checkConvexity(
                state.position,
                this._config.kNearest
            );

            coherences.push(convexity.coherence);

            // Check for boundary violation
            if (!convexity.isValid) {
                boundaryViolations++;
                anomalies.push({
                    type: AnomalyType.BOUNDARY_VIOLATION,
                    index: i,
                    severity: 1.0 - convexity.coherence,
                    description: `State outside Orthocognitum at index ${i}`,
                    state
                });
            }

            // Check for coherence drop
            if (convexity.coherence < this._config.coherenceThreshold) {
                anomalies.push({
                    type: AnomalyType.COHERENCE_DROP,
                    index: i,
                    severity: (this._config.coherenceThreshold - convexity.coherence) /
                        this._config.coherenceThreshold,
                    description: `Coherence dropped to ${convexity.coherence.toFixed(3)} at index ${i}`,
                    state
                });
            }

            // Check for lattice transition
            if (lastVertex >= 0 && convexity.nearestVertex !== lastVertex) {
                latticeTransitions++;
            }
            lastVertex = convexity.nearestVertex;

            // Check velocity
            const velocity = magnitude(state.velocity);
            velocities.push(velocity);

            if (velocity > this._config.velocitySpikeThreshold) {
                anomalies.push({
                    type: AnomalyType.VELOCITY_SPIKE,
                    index: i,
                    severity: Math.min(1, velocity / (this._config.velocitySpikeThreshold * 2)),
                    description: `Velocity spike: ${velocity.toFixed(3)} at index ${i}`,
                    state
                });
            }

            // Check angular velocity
            const angularVelocity = bivectorMagnitude(state.angularVelocity);
            if (angularVelocity > this._config.angularVelocitySpikeThreshold) {
                anomalies.push({
                    type: AnomalyType.ROTATION_SPIKE,
                    index: i,
                    severity: Math.min(1, angularVelocity /
                        (this._config.angularVelocitySpikeThreshold * 2)),
                    description: `Angular velocity spike: ${angularVelocity.toFixed(3)} at index ${i}`,
                    state
                });
            }

            // Check for discontinuity
            if (i > 0) {
                const prevPos = trajectory[i - 1].position;
                const jump = Math.sqrt(
                    (state.position[0] - prevPos[0]) ** 2 +
                    (state.position[1] - prevPos[1]) ** 2 +
                    (state.position[2] - prevPos[2]) ** 2 +
                    (state.position[3] - prevPos[3]) ** 2
                );

                pathLength += jump;

                if (jump > this._config.discontinuityThreshold) {
                    anomalies.push({
                        type: AnomalyType.DISCONTINUITY,
                        index: i,
                        severity: Math.min(1, jump / (this._config.discontinuityThreshold * 2)),
                        description: `Position jump of ${jump.toFixed(3)} at index ${i}`,
                        state
                    });
                }
            }
        }

        // Check for instability (coherence oscillation)
        if (coherences.length >= 3) {
            let oscillations = 0;
            for (let i = 2; i < coherences.length; i++) {
                const d1 = coherences[i - 1] - coherences[i - 2];
                const d2 = coherences[i] - coherences[i - 1];
                if (d1 * d2 < 0 && Math.abs(d1) > 0.1 && Math.abs(d2) > 0.1) {
                    oscillations++;
                }
            }

            if (oscillations > coherences.length * 0.3) {
                anomalies.push({
                    type: AnomalyType.INSTABILITY,
                    index: -1,
                    severity: Math.min(1, oscillations / coherences.length),
                    description: `Coherence instability: ${oscillations} oscillations`,
                    state: trajectory[trajectory.length - 1]
                });
            }
        }

        // Check for stagnation
        if (trajectory.length >= this._config.stagnationThreshold) {
            const recentPositions = trajectory.slice(-this._config.stagnationThreshold);
            const startPos = recentPositions[0].position;
            const endPos = recentPositions[recentPositions.length - 1].position;
            const totalMovement = Math.sqrt(
                (endPos[0] - startPos[0]) ** 2 +
                (endPos[1] - startPos[1]) ** 2 +
                (endPos[2] - startPos[2]) ** 2 +
                (endPos[3] - startPos[3]) ** 2
            );

            if (totalMovement < 0.01) {
                anomalies.push({
                    type: AnomalyType.STAGNATION,
                    index: trajectory.length - 1,
                    severity: 0.5,
                    description: `Stagnation detected over ${this._config.stagnationThreshold} steps`,
                    state: trajectory[trajectory.length - 1]
                });
            }
        }

        // Compute statistics
        const meanCoherence = coherences.reduce((a, b) => a + b, 0) / coherences.length;
        const stdCoherence = Math.sqrt(
            coherences.reduce((sum, c) => sum + (c - meanCoherence) ** 2, 0) / coherences.length
        );
        const meanVelocity = velocities.reduce((a, b) => a + b, 0) / velocities.length;

        const statistics: TrajectoryStatistics = {
            count: trajectory.length,
            meanCoherence,
            stdCoherence,
            minCoherence: Math.min(...coherences),
            maxCoherence: Math.max(...coherences),
            boundaryViolations,
            latticeTransitions,
            pathLength,
            meanVelocity
        };

        // Compute overall severity
        const severity = anomalies.length > 0
            ? Math.min(1, anomalies.reduce((sum, a) => sum + a.severity, 0) / anomalies.length)
            : 0;

        // Generate recommendations
        const recommendations = this._generateRecommendations(anomalies, statistics);

        return {
            hasAnomalies: anomalies.length > 0,
            severity,
            anomalies,
            statistics,
            recommendations
        };
    }

    /**
     * Suggest a correction to return to valid region.
     * Provides guidance on how to restore valid reasoning state.
     *
     * @param state - Current (potentially invalid) state
     * @returns Correction vector with target and urgency
     */
    suggestCorrection(state: EngineState): CorrectionVector {
        // Check current validity
        const convexity = this._lattice.checkConvexity(
            state.position,
            this._config.kNearest
        );

        // If already valid and coherent, no correction needed
        if (convexity.isValid && convexity.coherence >= this._config.coherenceThreshold) {
            return {
                linear: [0, 0, 0, 0],
                rotational: [0, 0, 0, 0, 0, 0],
                magnitude: 0,
                target: state.position,
                targetVertex: convexity.nearestVertex,
                estimatedSteps: 0,
                urgency: 0
            };
        }

        // Find target: either the projected valid point or nearest vertex
        let target: Vector4D;
        let targetVertex: number;

        if (!convexity.isValid) {
            // Outside valid region: project to boundary
            target = this._lattice.clamp(state.position);
            targetVertex = this._lattice.findNearest(target);
        } else {
            // Low coherence: move toward centroid of active concepts
            target = convexity.centroid;
            targetVertex = convexity.nearestVertex;
        }

        // Compute linear correction (direction to target)
        const dx = target[0] - state.position[0];
        const dy = target[1] - state.position[1];
        const dz = target[2] - state.position[2];
        const dw = target[3] - state.position[3];

        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz + dw * dw);

        const linear: Vector4D = distance > MATH_CONSTANTS.EPSILON
            ? [dx / distance, dy / distance, dz / distance, dw / distance]
            : [0, 0, 0, 0];

        // Compute rotational correction (reduce angular velocity if spinning too fast)
        const angularMag = bivectorMagnitude(state.angularVelocity);
        const rotational: Bivector4D = angularMag > this._config.angularVelocitySpikeThreshold
            ? state.angularVelocity.map(v => -v * 0.5) as Bivector4D
            : [0, 0, 0, 0, 0, 0];

        // Estimate steps to reach valid region
        // Assume average step moves ~0.1 units
        const estimatedSteps = Math.ceil(distance / 0.1);

        // Compute urgency based on severity of violation
        let urgency: number;
        if (!convexity.isValid) {
            urgency = 1.0; // Critical: outside valid region
        } else if (convexity.coherence < this._config.coherenceThreshold * 0.5) {
            urgency = 0.8; // High: very low coherence
        } else if (convexity.coherence < this._config.coherenceThreshold) {
            urgency = 0.5; // Medium: below threshold
        } else {
            urgency = 0.2; // Low: just needs minor adjustment
        }

        return {
            linear,
            rotational,
            magnitude: distance,
            target,
            targetVertex,
            estimatedSteps,
            urgency
        };
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /**
     * Compute stability from coherence history.
     */
    private _computeStability(): number {
        if (this._coherenceHistory.length < 2) {
            return 1.0; // Assume stable if not enough history
        }

        // Compute variance of coherence
        const mean = this._coherenceHistory.reduce((a, b) => a + b, 0) /
            this._coherenceHistory.length;
        const variance = this._coherenceHistory.reduce(
            (sum, c) => sum + (c - mean) ** 2, 0
        ) / this._coherenceHistory.length;

        // Stability is inverse of variance (clamped)
        // Low variance = high stability
        const stability = Math.max(0, 1 - Math.sqrt(variance) * 5);
        return stability;
    }

    /**
     * Compute boundary proximity (0 = center, 1 = at boundary).
     */
    private _computeBoundaryProximity(position: Vector4D): number {
        const distFromCenter = magnitude(position);
        const proximity = distFromCenter / CIRCUMRADIUS;
        return Math.min(1, proximity);
    }

    /**
     * Compute concept membership (which vertices the state is near).
     */
    private _computeConceptMembership(
        position: Vector4D,
        convexity: ConvexityResult
    ): ConceptMembership {
        const activeVertices = convexity.activeVertices;
        const weights: number[] = [];

        // Compute distance-based weights
        let totalWeight = 0;
        for (const vertexId of activeVertices) {
            const vertex = this._lattice.getVertex(vertexId);
            if (!vertex) continue;

            const dist = Math.sqrt(
                (position[0] - vertex.coordinates[0]) ** 2 +
                (position[1] - vertex.coordinates[1]) ** 2 +
                (position[2] - vertex.coordinates[2]) ** 2 +
                (position[3] - vertex.coordinates[3]) ** 2
            );

            // Inverse distance weighting
            const weight = 1 / (dist + MATH_CONSTANTS.EPSILON);
            weights.push(weight);
            totalWeight += weight;
        }

        // Normalize weights
        const normalizedWeights = weights.map(w => w / totalWeight);

        // Compute interpolated position
        const interpolated: Vector4D = [0, 0, 0, 0];
        for (let i = 0; i < activeVertices.length; i++) {
            const vertex = this._lattice.getVertex(activeVertices[i]);
            if (!vertex) continue;

            interpolated[0] += vertex.coordinates[0] * normalizedWeights[i];
            interpolated[1] += vertex.coordinates[1] * normalizedWeights[i];
            interpolated[2] += vertex.coordinates[2] * normalizedWeights[i];
            interpolated[3] += vertex.coordinates[3] * normalizedWeights[i];
        }

        return {
            primary: convexity.nearestVertex,
            secondary: activeVertices.filter(v => v !== convexity.nearestVertex),
            weights: normalizedWeights,
            interpolated
        };
    }

    /**
     * Generate recommendations based on anomalies.
     */
    private _generateRecommendations(
        anomalies: Anomaly[],
        statistics: TrajectoryStatistics
    ): string[] {
        const recommendations: string[] = [];

        // Group anomalies by type
        const typeCount = new Map<AnomalyType, number>();
        for (const anomaly of anomalies) {
            typeCount.set(anomaly.type, (typeCount.get(anomaly.type) || 0) + 1);
        }

        if (typeCount.get(AnomalyType.BOUNDARY_VIOLATION)) {
            recommendations.push('Increase damping or reduce force magnitude to prevent boundary violations');
        }

        if (typeCount.get(AnomalyType.COHERENCE_DROP)) {
            recommendations.push('Apply corrective forces toward lattice vertices to restore coherence');
        }

        if (typeCount.get(AnomalyType.DISCONTINUITY)) {
            recommendations.push('Use smaller timesteps or reduce impulse forces to ensure continuity');
        }

        if (typeCount.get(AnomalyType.INSTABILITY)) {
            recommendations.push('Increase inertia coefficient to stabilize coherence oscillations');
        }

        if (typeCount.get(AnomalyType.STAGNATION)) {
            recommendations.push('Apply exploratory forces to escape local minimum');
        }

        if (typeCount.get(AnomalyType.VELOCITY_SPIKE)) {
            recommendations.push('Lower maxLinearVelocity or increase damping');
        }

        if (typeCount.get(AnomalyType.ROTATION_SPIKE)) {
            recommendations.push('Lower maxAngularVelocity or increase rotational damping');
        }

        if (statistics.meanCoherence < 0.5) {
            recommendations.push('Overall low coherence: consider resetting to a lattice vertex');
        }

        if (recommendations.length === 0 && anomalies.length === 0) {
            recommendations.push('Trajectory appears healthy - no corrective action needed');
        }

        return recommendations;
    }

    /**
     * Create empty anomaly report.
     */
    private _createEmptyAnomalyReport(): AnomalyReport {
        return {
            hasAnomalies: false,
            severity: 0,
            anomalies: [],
            statistics: {
                count: 0,
                meanCoherence: 1,
                stdCoherence: 0,
                minCoherence: 1,
                maxCoherence: 1,
                boundaryViolations: 0,
                latticeTransitions: 0,
                pathLength: 0,
                meanVelocity: 0
            },
            recommendations: ['No trajectory data to analyze']
        };
    }

    /**
     * Clear validation history.
     */
    clearHistory(): void {
        this._coherenceHistory = [];
        this._positionHistory = [];
        this._lastValidation = null;
    }

    /**
     * Get validation statistics.
     */
    getStats(): Record<string, unknown> {
        return {
            historyLength: this._coherenceHistory.length,
            positionHistoryLength: this._positionHistory.length,
            lastCoherence: this._coherenceHistory[this._coherenceHistory.length - 1] ?? null,
            averageCoherence: this._coherenceHistory.length > 0
                ? this._coherenceHistory.reduce((a, b) => a + b, 0) / this._coherenceHistory.length
                : null,
            lastValidation: this._lastValidation
        };
    }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/** Singleton instance */
let _defaultValidator: Epistaorthognition | null = null;

/**
 * Get or create the default validator instance.
 */
export function getDefaultValidator(): Epistaorthognition {
    if (!_defaultValidator) {
        _defaultValidator = new Epistaorthognition();
    }
    return _defaultValidator;
}

/**
 * Create a new validator instance.
 */
export function createValidator(config?: Partial<ValidationConfig>): Epistaorthognition {
    return new Epistaorthognition(config);
}

// =============================================================================
// STANDALONE FUNCTIONS
// =============================================================================

/**
 * Quick validation of a state.
 */
export function validateState(state: EngineState): ValidationResult {
    return getDefaultValidator().validateState(state);
}

/**
 * Quick coherence check.
 */
export function computeCoherence(position: Vector4D): number {
    return getDefaultValidator().computeCoherence(position);
}

function percentile(values: number[], p: number): number {
    if (values.length === 0) {
        return 0;
    }
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
    return sorted[index];
}

/**
 * Compute batch-level coherence and boundary risk metrics.
 *
 * @param positions - Array of 4D positions to analyze
 */
export function computeBatchMetrics(
    positions: Vector4D[],
    topology?: Pick<TopologyProvider, 'checkConvexity' | 'circumradius'>
): BatchMetrics {
    const provider = topology ?? getDefaultLattice();
    const coherences: number[] = [];
    const boundaryProximities: number[] = [];

    for (const position of positions) {
        const convexity = provider.checkConvexity(position);
        coherences.push(convexity.coherence);
        const distFromCenter = magnitude(position);
        boundaryProximities.push(Math.min(1, distFromCenter / provider.circumradius));
    }

    if (coherences.length === 0) {
        return {
            coherenceMean: 0,
            coherenceVariance: 0,
            boundaryRiskP95: 0,
            minCoherence: 0,
            maxCoherence: 0
        };
    }

    const mean = coherences.reduce((sum, value) => sum + value, 0) / coherences.length;
    const variance = coherences.reduce((sum, value) => sum + (value - mean) ** 2, 0) / coherences.length;

    return {
        coherenceMean: mean,
        coherenceVariance: variance,
        boundaryRiskP95: percentile(boundaryProximities, 0.95),
        minCoherence: Math.min(...coherences),
        maxCoherence: Math.max(...coherences)
    };
}

/**
 * Quick anomaly detection.
 */
export function detectAnomaly(trajectory: EngineState[]): AnomalyReport {
    return getDefaultValidator().detectAnomaly(trajectory);
}

/**
 * Quick correction suggestion.
 */
export function suggestCorrection(state: EngineState): CorrectionVector {
    return getDefaultValidator().suggestCorrection(state);
}
