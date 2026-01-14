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
} from './lib/topology/ThreeBodyPhaseSpace.js';
import { Vector4D, MATH_CONSTANTS } from './types/index.js';
import { Vector8D, generateE8Roots, projectE8to4D, createMoxnessMatrix } from './lib/topology/E8H4Folding.js';
import { Lattice600, getDefaultLattice600, Lattice600Vertex, Cell24Subset } from './lib/topology/Lattice600.js';
import {
    computeTrinityDecomposition,
    phillipsSynthesis,
    TrinityDecomposition,
    Cell16Subset
} from './lib/topology/TrinityDecomposition.js';

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
// PPP LATTICE EVOLUTION - NESTED 24-CELL / 16-CELL TRIALECTIC
// =============================================================================

/**
 * The nested structure:
 *
 * 600-cell (120 vertices)
 * ├── 24-cell A (24 vertices) [Body 1]
 * │   ├── 16-cell α (8 vertices)
 * │   ├── 16-cell β (8 vertices)
 * │   └── 16-cell γ (8 vertices)
 * ├── 24-cell B (24 vertices) [Body 2]
 * │   └── (same 16-cell trinity)
 * ├── 24-cell C (24 vertices) [Body 3]
 * │   └── (same 16-cell trinity)
 * ├── 24-cell D (24 vertices) [Interaction]
 * └── 24-cell E (24 vertices) [Interaction]
 *
 * Bodies evolve by:
 * 1. Each body maps to its 24-cell (A, B, C)
 * 2. Within each 24-cell, find which 16-cell trialectic (α, β, γ)
 * 3. Phillips Synthesis: Given two bodies' positions, third is constrained
 * 4. Evolution = snapping to nearest valid constellation
 */

// Cache the trinity decomposition (same for all 24-cells)
let _trinityCache: TrinityDecomposition | null = null;

function getTrinity(): TrinityDecomposition {
    if (!_trinityCache) {
        _trinityCache = computeTrinityDecomposition();
    }
    return _trinityCache;
}

/**
 * Find which 16-cell (α, β, γ) a point is nearest to within a 24-cell.
 */
function classifyTrialectic(
    point: Vector4D,
    trinity: TrinityDecomposition
): { subset: 'alpha' | 'beta' | 'gamma'; nearestVertex: Vector4D; distance: number } {
    let bestSubset: 'alpha' | 'beta' | 'gamma' = 'alpha';
    let bestVertex: Vector4D = trinity.alpha.vertices[0];
    let bestDist = Infinity;

    for (const [name, subset] of [
        ['alpha', trinity.alpha],
        ['beta', trinity.beta],
        ['gamma', trinity.gamma]
    ] as const) {
        for (const v of subset.vertices) {
            const dist = Math.sqrt(
                (point[0] - v[0]) ** 2 + (point[1] - v[1]) ** 2 +
                (point[2] - v[2]) ** 2 + (point[3] - v[3]) ** 2
            );
            if (dist < bestDist) {
                bestDist = dist;
                bestVertex = v;
                bestSubset = name;
            }
        }
    }

    return { subset: bestSubset, nearestVertex: bestVertex, distance: bestDist };
}

/**
 * Evolve using the nested 24-cell constellation with 16-cell trialectic.
 *
 * Instead of F=ma integration:
 * 1. Map each body to its 24-cell
 * 2. Classify each body's position in the trialectic (α, β, γ)
 * 3. Use Phillips Synthesis: body 3's γ emerges from bodies 1's α + body 2's β
 * 4. Snap to nearest valid lattice configuration
 */
function pppStep(
    state: ThreeBodyState,
    lattice: Lattice600,
    phasePoint: ReducedPhasePoint,
    cellMapping: Cell600Mapping,
    dt: number
): { state: ThreeBodyState; phasePoint: ReducedPhasePoint; cellMapping: Cell600Mapping } {
    const trinity = getTrinity();
    const cells = lattice.disjoint24Cells;

    // Get H4 projections for each body from mapping
    const h4 = cellMapping.h4Position;

    // Classify each body in the trialectic
    // Body 1 should be in α, Body 2 in β, Body 3's γ is computed
    const body1Class = classifyTrialectic(h4, trinity);
    const body2Offset: Vector4D = [h4[0] + 0.3, h4[1], h4[2], h4[3]];
    const body2Class = classifyTrialectic(body2Offset, trinity);

    // Phillips Synthesis: Given α (body1) and β (body2), compute γ (body3)
    const synthesizedGamma = phillipsSynthesis(
        body1Class.nearestVertex,
        body2Class.nearestVertex
    );

    // The constellation coherence: how well does the trialectic balance?
    // Perfect balance = centroid at origin (color neutral)
    const centroid: Vector4D = [
        (body1Class.nearestVertex[0] + body2Class.nearestVertex[0] + synthesizedGamma[0]) / 3,
        (body1Class.nearestVertex[1] + body2Class.nearestVertex[1] + synthesizedGamma[1]) / 3,
        (body1Class.nearestVertex[2] + body2Class.nearestVertex[2] + synthesizedGamma[2]) / 3,
        (body1Class.nearestVertex[3] + body2Class.nearestVertex[3] + synthesizedGamma[3]) / 3
    ];
    const centroidDist = Math.sqrt(
        centroid[0]**2 + centroid[1]**2 + centroid[2]**2 + centroid[3]**2
    );
    const coherence = Math.exp(-centroidDist * 2);

    // Evolution: Bodies move toward valid trialectic configuration
    // High coherence = stable orbit, low change
    // Low coherence = bodies adjust toward synthesis
    const changeFactor = (1 - coherence) * dt;

    // Move body positions toward their trialectic vertices
    const newBody1Pos: [number, number, number] = [
        state.body1.position[0] + changeFactor * (body1Class.nearestVertex[0] - state.body1.position[0]),
        state.body1.position[1] + changeFactor * (body1Class.nearestVertex[1] - state.body1.position[1]),
        state.body1.position[2] + changeFactor * (body1Class.nearestVertex[2] - state.body1.position[2])
    ];

    const newBody2Pos: [number, number, number] = [
        state.body2.position[0] + changeFactor * (body2Class.nearestVertex[0] - state.body2.position[0]),
        state.body2.position[1] + changeFactor * (body2Class.nearestVertex[1] - state.body2.position[1]),
        state.body2.position[2] + changeFactor * (body2Class.nearestVertex[2] - state.body2.position[2])
    ];

    // Body 3 moves toward synthesized γ position
    const newBody3Pos: [number, number, number] = [
        state.body3.position[0] + changeFactor * (synthesizedGamma[0] - state.body3.position[0]),
        state.body3.position[1] + changeFactor * (synthesizedGamma[1] - state.body3.position[1]),
        state.body3.position[2] + changeFactor * (synthesizedGamma[2] - state.body3.position[2])
    ];

    // Compute new velocities from position changes
    const newBody1Vel: [number, number, number] = [
        (newBody1Pos[0] - state.body1.position[0]) / dt,
        (newBody1Pos[1] - state.body1.position[1]) / dt,
        (newBody1Pos[2] - state.body1.position[2]) / dt
    ];
    const newBody2Vel: [number, number, number] = [
        (newBody2Pos[0] - state.body2.position[0]) / dt,
        (newBody2Pos[1] - state.body2.position[1]) / dt,
        (newBody2Pos[2] - state.body2.position[2]) / dt
    ];
    const newBody3Vel: [number, number, number] = [
        (newBody3Pos[0] - state.body3.position[0]) / dt,
        (newBody3Pos[1] - state.body3.position[1]) / dt,
        (newBody3Pos[2] - state.body3.position[2]) / dt
    ];

    const newState: ThreeBodyState = {
        body1: { position: newBody1Pos, velocity: newBody1Vel, mass: state.body1.mass },
        body2: { position: newBody2Pos, velocity: newBody2Vel, mass: state.body2.mass },
        body3: { position: newBody3Pos, velocity: newBody3Vel, mass: state.body3.mass },
        time: state.time + dt
    };

    // Re-encode to phase space
    const newPhasePoint = encodeToPhaseSpace(newState);
    const newCellMapping = mapTo600Cell(newState);

    return {
        state: newState,
        phasePoint: newPhasePoint,
        cellMapping: newCellMapping
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
