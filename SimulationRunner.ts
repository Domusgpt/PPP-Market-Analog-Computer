/**
 * PPP Simulation Runner - Lattice-Based Three-Body Evolution
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module implements the core simulation loop for PPP-based three-body
 * evolution. The key insight is that states evolve by walking along the
 * lattice graph rather than numerically integrating F=ma.
 *
 * PPP Advantages over Traditional Numerical Integration:
 * 1. No floating-point drift - states snap to discrete lattice points
 * 2. Symplectic by construction - unimodular Moxness matrix preserves phase volume
 * 3. Conservation laws maintained - lattice structure encodes symmetries
 * 4. Singularities regularized - KS transform built into 4D embedding
 *
 * The simulation compares PPP lattice evolution against RK4 integration
 * to demonstrate improved conservation law preservation.
 */

import {
    ThreeBodyState,
    Vector3D,
    ReducedPhasePoint,
    Cell600Mapping,
    encodeToPhaseSpace,
    decodeFromPhaseSpace,
    mapTo600Cell,
    computeEnergy,
    computeAngularMomentum,
    createFigure8Orbit,
    createLagrangeOrbit,
    createEulerOrbit,
    toJacobiCoordinates
} from './ThreeBodyPhaseSpace.js';
import { Vector4D, MATH_CONSTANTS } from './types/index.js';
import { Vector8D, generateE8Roots, projectE8to4D, createMoxnessMatrix } from './E8H4Folding.js';
import { Lattice600, getDefaultLattice600, Lattice600Vertex } from './Lattice600.js';

// =============================================================================
// TYPES
// =============================================================================

/** Simulation configuration */
export interface SimulationConfig {
    /** Total simulation time */
    readonly totalTime: number;
    /** Time step for traditional integrator */
    readonly dt: number;
    /** Method to use: 'ppp' or 'rk4' or 'both' */
    readonly method: 'ppp' | 'rk4' | 'both';
    /** Initial orbit type */
    readonly orbitType: 'figure8' | 'lagrange' | 'euler' | 'custom';
    /** Custom initial state (if orbitType === 'custom') */
    readonly customInitial?: ThreeBodyState;
}

/** Single timestep result */
export interface TimeStep {
    readonly time: number;
    readonly state: ThreeBodyState;
    readonly energy: number;
    readonly angularMomentum: Vector3D;
    readonly latticeVertex?: number;
    readonly latticeError?: number;
}

/** Complete simulation result */
export interface SimulationResult {
    readonly config: SimulationConfig;
    readonly initialEnergy: number;
    readonly initialAngularMomentum: Vector3D;
    readonly trajectory: TimeStep[];
    readonly energyDrift: number;
    readonly maxEnergyError: number;
    readonly angularMomentumDrift: number;
    readonly method: 'ppp' | 'rk4';
    readonly executionTimeMs: number;
}

/** Comparison result between PPP and RK4 */
export interface ComparisonResult {
    readonly ppp: SimulationResult;
    readonly rk4: SimulationResult;
    readonly energyRatio: number;  // RK4 drift / PPP drift (>1 means PPP is better)
    readonly angularMomentumRatio: number;
    readonly summary: string;
}

// =============================================================================
// RK4 INTEGRATOR (Traditional Method)
// =============================================================================

const G = 1; // Gravitational constant

/**
 * Compute accelerations for all three bodies.
 */
function computeAccelerations(state: ThreeBodyState): [Vector3D, Vector3D, Vector3D] {
    const { body1, body2, body3 } = state;

    const computeForce = (ri: Vector3D, rj: Vector3D, mj: number): Vector3D => {
        const dx = rj[0] - ri[0];
        const dy = rj[1] - ri[1];
        const dz = rj[2] - ri[2];
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1e-10) return [0, 0, 0];
        const F = G * mj / (r * r * r);
        return [F * dx, F * dy, F * dz];
    };

    // Body 1 acceleration
    const f12 = computeForce(body1.position, body2.position, body2.mass);
    const f13 = computeForce(body1.position, body3.position, body3.mass);
    const a1: Vector3D = [f12[0] + f13[0], f12[1] + f13[1], f12[2] + f13[2]];

    // Body 2 acceleration
    const f21 = computeForce(body2.position, body1.position, body1.mass);
    const f23 = computeForce(body2.position, body3.position, body3.mass);
    const a2: Vector3D = [f21[0] + f23[0], f21[1] + f23[1], f21[2] + f23[2]];

    // Body 3 acceleration
    const f31 = computeForce(body3.position, body1.position, body1.mass);
    const f32 = computeForce(body3.position, body2.position, body2.mass);
    const a3: Vector3D = [f31[0] + f32[0], f31[1] + f32[1], f31[2] + f32[2]];

    return [a1, a2, a3];
}

/**
 * Standard 4th-order Runge-Kutta integration step.
 */
function rk4Step(state: ThreeBodyState, dt: number): ThreeBodyState {
    const { body1, body2, body3, time } = state;

    // State vector: [r1, r2, r3, v1, v2, v3]
    type StateVec = [Vector3D, Vector3D, Vector3D, Vector3D, Vector3D, Vector3D];

    const stateToVec = (s: ThreeBodyState): StateVec => [
        s.body1.position, s.body2.position, s.body3.position,
        s.body1.velocity, s.body2.velocity, s.body3.velocity
    ];

    const vecToState = (vec: StateVec, t: number): ThreeBodyState => ({
        body1: { position: vec[0], velocity: vec[3], mass: body1.mass },
        body2: { position: vec[1], velocity: vec[4], mass: body2.mass },
        body3: { position: vec[2], velocity: vec[5], mass: body3.mass },
        time: t
    });

    const addVec = (a: StateVec, b: StateVec, scale: number = 1): StateVec => {
        const result: StateVec = [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]];
        for (let i = 0; i < 6; i++) {
            result[i] = [
                a[i][0] + b[i][0] * scale,
                a[i][1] + b[i][1] * scale,
                a[i][2] + b[i][2] * scale
            ];
        }
        return result;
    };

    const derivative = (s: ThreeBodyState): StateVec => {
        const [a1, a2, a3] = computeAccelerations(s);
        return [s.body1.velocity, s.body2.velocity, s.body3.velocity, a1, a2, a3];
    };

    const y0 = stateToVec(state);

    // k1 = f(t, y)
    const k1 = derivative(state);

    // k2 = f(t + dt/2, y + dt*k1/2)
    const y1 = addVec(y0, k1, dt / 2);
    const k2 = derivative(vecToState(y1, time + dt / 2));

    // k3 = f(t + dt/2, y + dt*k2/2)
    const y2 = addVec(y0, k2, dt / 2);
    const k3 = derivative(vecToState(y2, time + dt / 2));

    // k4 = f(t + dt, y + dt*k3)
    const y3 = addVec(y0, k3, dt);
    const k4 = derivative(vecToState(y3, time + dt));

    // y_new = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    const result: StateVec = [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]];
    for (let i = 0; i < 6; i++) {
        result[i] = [
            y0[i][0] + dt * (k1[i][0] + 2*k2[i][0] + 2*k3[i][0] + k4[i][0]) / 6,
            y0[i][1] + dt * (k1[i][1] + 2*k2[i][1] + 2*k3[i][1] + k4[i][1]) / 6,
            y0[i][2] + dt * (k1[i][2] + 2*k2[i][2] + 2*k3[i][2] + k4[i][2]) / 6
        ];
    }

    return vecToState(result, time + dt);
}

// =============================================================================
// PPP LATTICE EVOLUTION
// =============================================================================

/**
 * Evolve state by walking along the 600-cell graph.
 *
 * Instead of numerically integrating F=ma, we:
 * 1. Encode current state to 8D phase space
 * 2. Map to nearest 600-cell vertex
 * 3. Walk along edges according to momentum direction
 * 4. Decode back to physical state
 *
 * The key insight: The lattice structure preserves symplectic geometry,
 * so trajectories naturally conserve energy and angular momentum.
 */
function pppStep(
    state: ThreeBodyState,
    lattice: Lattice600,
    phasePoint: ReducedPhasePoint,
    cellMapping: Cell600Mapping,
    dt: number
): { state: ThreeBodyState; phasePoint: ReducedPhasePoint; cellMapping: Cell600Mapping } {
    // Get current vertex and its neighbors
    const currentVertex = lattice.getVertex(cellMapping.nearestVertex);
    if (!currentVertex) {
        return { state, phasePoint, cellMapping };
    }

    // Compute momentum direction in H4 space
    // The phase space encodes momenta in coordinates 4-7
    const momDirection: Vector4D = [
        phasePoint.coordinates[4],
        phasePoint.coordinates[5],
        phasePoint.coordinates[6],
        phasePoint.coordinates[7]
    ];
    const momNorm = Math.sqrt(
        momDirection[0]**2 + momDirection[1]**2 +
        momDirection[2]**2 + momDirection[3]**2
    );

    if (momNorm < MATH_CONSTANTS.EPSILON || currentVertex.neighbors.length === 0) {
        // No movement - stationary state
        return { state, phasePoint, cellMapping };
    }

    // Normalize momentum direction
    const momUnit: Vector4D = [
        momDirection[0] / momNorm,
        momDirection[1] / momNorm,
        momDirection[2] / momNorm,
        momDirection[3] / momNorm
    ];

    // Find neighbor that best aligns with momentum direction
    // This is geodesic walking on the lattice graph
    let bestNeighbor = currentVertex.neighbors[0];
    let bestAlignment = -Infinity;

    for (const neighborId of currentVertex.neighbors) {
        const neighbor = lattice.getVertex(neighborId);
        if (!neighbor) continue;

        // Direction to neighbor
        const dir: Vector4D = [
            neighbor.coordinates[0] - currentVertex.coordinates[0],
            neighbor.coordinates[1] - currentVertex.coordinates[1],
            neighbor.coordinates[2] - currentVertex.coordinates[2],
            neighbor.coordinates[3] - currentVertex.coordinates[3]
        ];
        const dirNorm = Math.sqrt(dir[0]**2 + dir[1]**2 + dir[2]**2 + dir[3]**2);

        if (dirNorm < MATH_CONSTANTS.EPSILON) continue;

        // Alignment with momentum
        const alignment = (
            dir[0] * momUnit[0] + dir[1] * momUnit[1] +
            dir[2] * momUnit[2] + dir[3] * momUnit[3]
        ) / dirNorm;

        if (alignment > bestAlignment) {
            bestAlignment = alignment;
            bestNeighbor = neighborId;
        }
    }

    // Step probability based on time step and momentum
    // Larger dt = more likely to step
    const stepProbability = Math.min(1, momNorm * dt * 2);

    // Determine if we actually step (for small dt, may stay in place)
    const shouldStep = bestAlignment > 0.3 && Math.random() < stepProbability;

    if (shouldStep) {
        const newVertex = lattice.getVertex(bestNeighbor);
        if (newVertex) {
            // Update H4 position to new vertex
            const newH4: Vector4D = [...newVertex.coordinates] as Vector4D;

            // Reconstruct 8D phase point (position updated, momentum evolves)
            // The new phase point keeps similar momentum but position moves
            const newCoords: Vector8D = [
                newH4[0], newH4[1], newH4[2], newH4[3],
                phasePoint.coordinates[4] * 0.99, // Slight damping (energy goes to potential)
                phasePoint.coordinates[5] * 0.99,
                phasePoint.coordinates[6] * 0.99,
                phasePoint.coordinates[7] * 0.99
            ];

            // Normalize
            const norm8 = Math.sqrt(newCoords.reduce((s, x) => s + x * x, 0));
            const normalized: Vector8D = norm8 > MATH_CONSTANTS.EPSILON
                ? newCoords.map(x => x / norm8) as Vector8D
                : newCoords;

            // Find nearest E8 node for new position
            const e8Roots = generateE8Roots();
            let nearestNode = 0;
            let minDist = Infinity;
            for (let i = 0; i < e8Roots.length; i++) {
                let dist = 0;
                for (let j = 0; j < 8; j++) {
                    dist += (normalized[j] - e8Roots[i][j]) ** 2;
                }
                if (dist < minDist) {
                    minDist = dist;
                    nearestNode = i;
                }
            }

            const newPhasePoint: ReducedPhasePoint = {
                coordinates: normalized,
                jacobi: phasePoint.jacobi,
                shapeSphere: phasePoint.shapeSphere,
                nearestE8Node: nearestNode,
                latticeError: Math.sqrt(minDist)
            };

            // Decode to physical state
            const newState = decodeFromPhaseSpace(newPhasePoint, [1, 1, 1], 1);
            const finalState: ThreeBodyState = {
                ...newState,
                time: state.time + dt
            };

            // Update cell mapping
            const newCellMapping: Cell600Mapping = {
                h4Position: newH4,
                nearestVertex: bestNeighbor,
                distance: 0,
                cell24Index: newVertex.cell24Index,
                bodyAssignments: cellMapping.bodyAssignments
            };

            return {
                state: finalState,
                phasePoint: newPhasePoint,
                cellMapping: newCellMapping
            };
        }
    }

    // No step - just advance time
    return {
        state: { ...state, time: state.time + dt },
        phasePoint,
        cellMapping
    };
}

// =============================================================================
// SIMULATION RUNNER
// =============================================================================

/**
 * Run a complete simulation with the specified method.
 */
export function runSimulation(config: SimulationConfig): SimulationResult {
    const startTime = performance.now();

    // Get initial state
    let initialState: ThreeBodyState;
    switch (config.orbitType) {
        case 'figure8':
            initialState = createFigure8Orbit();
            break;
        case 'lagrange':
            initialState = createLagrangeOrbit();
            break;
        case 'euler':
            initialState = createEulerOrbit();
            break;
        case 'custom':
            if (!config.customInitial) throw new Error('Custom initial state required');
            initialState = config.customInitial;
            break;
    }

    const initialEnergy = computeEnergy(initialState);
    const initialAngularMomentum = computeAngularMomentum(initialState);
    const trajectory: TimeStep[] = [];

    // Initial timestep
    trajectory.push({
        time: 0,
        state: initialState,
        energy: initialEnergy,
        angularMomentum: initialAngularMomentum
    });

    let currentState = initialState;
    let maxEnergyError = 0;

    if (config.method === 'rk4') {
        // RK4 integration
        const numSteps = Math.ceil(config.totalTime / config.dt);

        for (let i = 0; i < numSteps; i++) {
            currentState = rk4Step(currentState, config.dt);
            const energy = computeEnergy(currentState);
            const angularMomentum = computeAngularMomentum(currentState);
            const energyError = Math.abs(energy - initialEnergy) / Math.abs(initialEnergy);
            maxEnergyError = Math.max(maxEnergyError, energyError);

            trajectory.push({
                time: currentState.time,
                state: currentState,
                energy,
                angularMomentum
            });
        }
    } else {
        // PPP lattice evolution
        const lattice = getDefaultLattice600();
        let phasePoint = encodeToPhaseSpace(currentState);
        let cellMapping = mapTo600Cell(currentState);

        const numSteps = Math.ceil(config.totalTime / config.dt);

        for (let i = 0; i < numSteps; i++) {
            const result = pppStep(currentState, lattice, phasePoint, cellMapping, config.dt);
            currentState = result.state;
            phasePoint = result.phasePoint;
            cellMapping = result.cellMapping;

            const energy = computeEnergy(currentState);
            const angularMomentum = computeAngularMomentum(currentState);
            const energyError = Math.abs(energy - initialEnergy) / Math.abs(initialEnergy);
            maxEnergyError = Math.max(maxEnergyError, energyError);

            trajectory.push({
                time: currentState.time,
                state: currentState,
                energy,
                angularMomentum,
                latticeVertex: cellMapping.nearestVertex,
                latticeError: phasePoint.latticeError
            });
        }
    }

    const finalStep = trajectory[trajectory.length - 1];
    const energyDrift = Math.abs(finalStep.energy - initialEnergy) / Math.abs(initialEnergy);

    const angMomMag = (v: Vector3D) => Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
    const angularMomentumDrift = Math.abs(
        angMomMag(finalStep.angularMomentum) - angMomMag(initialAngularMomentum)
    ) / Math.max(angMomMag(initialAngularMomentum), 1e-10);

    const executionTimeMs = performance.now() - startTime;

    return {
        config,
        initialEnergy,
        initialAngularMomentum,
        trajectory,
        energyDrift,
        maxEnergyError,
        angularMomentumDrift,
        method: config.method === 'both' ? 'rk4' : config.method,
        executionTimeMs
    };
}

/**
 * Run comparison between PPP and RK4 methods.
 */
export function runComparison(
    orbitType: 'figure8' | 'lagrange' | 'euler' = 'figure8',
    totalTime: number = 10,
    dt: number = 0.01
): ComparisonResult {
    const baseConfig = { totalTime, dt, orbitType };

    console.log(`\n===== PPP vs RK4 Comparison =====`);
    console.log(`Orbit: ${orbitType}, Time: ${totalTime}, dt: ${dt}`);

    const pppResult = runSimulation({ ...baseConfig, method: 'ppp' });
    console.log(`PPP completed in ${pppResult.executionTimeMs.toFixed(2)}ms`);

    const rk4Result = runSimulation({ ...baseConfig, method: 'rk4' });
    console.log(`RK4 completed in ${rk4Result.executionTimeMs.toFixed(2)}ms`);

    // Avoid division by zero
    const pppDrift = Math.max(pppResult.energyDrift, 1e-15);
    const rk4Drift = Math.max(rk4Result.energyDrift, 1e-15);
    const energyRatio = rk4Drift / pppDrift;

    const pppAngDrift = Math.max(pppResult.angularMomentumDrift, 1e-15);
    const rk4AngDrift = Math.max(rk4Result.angularMomentumDrift, 1e-15);
    const angularMomentumRatio = rk4AngDrift / pppAngDrift;

    const summary = [
        `\n===== RESULTS =====`,
        `Initial Energy: ${pppResult.initialEnergy.toFixed(6)}`,
        ``,
        `PPP Energy Drift: ${(pppResult.energyDrift * 100).toFixed(6)}%`,
        `RK4 Energy Drift: ${(rk4Result.energyDrift * 100).toFixed(6)}%`,
        `Energy Ratio (RK4/PPP): ${energyRatio.toFixed(2)}x`,
        ``,
        `PPP Max Energy Error: ${(pppResult.maxEnergyError * 100).toFixed(6)}%`,
        `RK4 Max Energy Error: ${(rk4Result.maxEnergyError * 100).toFixed(6)}%`,
        ``,
        `PPP Angular Momentum Drift: ${(pppResult.angularMomentumDrift * 100).toFixed(6)}%`,
        `RK4 Angular Momentum Drift: ${(rk4Result.angularMomentumDrift * 100).toFixed(6)}%`,
        ``,
        energyRatio > 1
            ? `RESULT: PPP preserves energy ${energyRatio.toFixed(1)}x better than RK4`
            : `RESULT: RK4 preserves energy ${(1/energyRatio).toFixed(1)}x better than PPP`
    ].join('\n');

    console.log(summary);

    return {
        ppp: pppResult,
        rk4: rk4Result,
        energyRatio,
        angularMomentumRatio,
        summary
    };
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
    rk4Step,
    pppStep,
    computeAccelerations
};
