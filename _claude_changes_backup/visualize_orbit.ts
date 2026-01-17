#!/usr/bin/env npx tsx
/**
 * PPP Orbit Visualization
 *
 * Shows how real orbits trace paths on the 600-cell lattice.
 * The key insight: PPP produces PATTERNS that should match physics.
 *
 * What we're looking for:
 * 1. Stable orbits (Figure-8, Lagrange) should stay near lattice nodes
 * 2. The lattice error should be low and periodic for periodic orbits
 * 3. The 24-cell transitions should form coherent patterns
 */

import {
    ThreeBodyState,
    encodeToPhaseSpace,
    mapTo600Cell,
    computeEnergy,
    createFigure8Orbit,
    createLagrangeOrbit,
} from './lib/topology/ThreeBodyPhaseSpace.js';
import { getDefaultLattice600 } from './lib/topology/Lattice600.js';

// RK4 step (same as SimulationRunner)
function rk4Step(state: ThreeBodyState, dt: number): ThreeBodyState {
    const G = 1;
    type Vector3D = [number, number, number];
    type StateVec = [Vector3D, Vector3D, Vector3D, Vector3D, Vector3D, Vector3D];

    const computeAccel = (s: ThreeBodyState): [Vector3D, Vector3D, Vector3D] => {
        const force = (ri: Vector3D, rj: Vector3D, mj: number): Vector3D => {
            const dx = rj[0] - ri[0], dy = rj[1] - ri[1], dz = rj[2] - ri[2];
            const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (r < 1e-10) return [0, 0, 0];
            const F = G * mj / (r * r * r);
            return [F * dx, F * dy, F * dz];
        };
        const f12 = force(s.body1.position, s.body2.position, s.body2.mass);
        const f13 = force(s.body1.position, s.body3.position, s.body3.mass);
        const f21 = force(s.body2.position, s.body1.position, s.body1.mass);
        const f23 = force(s.body2.position, s.body3.position, s.body3.mass);
        const f31 = force(s.body3.position, s.body1.position, s.body1.mass);
        const f32 = force(s.body3.position, s.body2.position, s.body2.mass);
        return [
            [f12[0]+f13[0], f12[1]+f13[1], f12[2]+f13[2]],
            [f21[0]+f23[0], f21[1]+f23[1], f21[2]+f23[2]],
            [f31[0]+f32[0], f31[1]+f32[1], f31[2]+f32[2]]
        ];
    };

    const stateToVec = (s: ThreeBodyState): StateVec => [
        s.body1.position as Vector3D, s.body2.position as Vector3D, s.body3.position as Vector3D,
        s.body1.velocity as Vector3D, s.body2.velocity as Vector3D, s.body3.velocity as Vector3D
    ];
    const vecToState = (v: StateVec, t: number): ThreeBodyState => ({
        body1: { position: v[0], velocity: v[3], mass: state.body1.mass },
        body2: { position: v[1], velocity: v[4], mass: state.body2.mass },
        body3: { position: v[2], velocity: v[5], mass: state.body3.mass },
        time: t
    });
    const addVec = (a: StateVec, b: StateVec, s: number): StateVec => {
        const r: StateVec = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]];
        for (let i = 0; i < 6; i++) r[i] = [a[i][0]+b[i][0]*s, a[i][1]+b[i][1]*s, a[i][2]+b[i][2]*s];
        return r;
    };
    const deriv = (s: ThreeBodyState): StateVec => {
        const [a1, a2, a3] = computeAccel(s);
        return [s.body1.velocity as Vector3D, s.body2.velocity as Vector3D, s.body3.velocity as Vector3D, a1, a2, a3];
    };

    const y0 = stateToVec(state);
    const k1 = deriv(state);
    const k2 = deriv(vecToState(addVec(y0, k1, dt/2), state.time + dt/2));
    const k3 = deriv(vecToState(addVec(y0, k2, dt/2), state.time + dt/2));
    const k4 = deriv(vecToState(addVec(y0, k3, dt), state.time + dt));

    const result: StateVec = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]];
    for (let i = 0; i < 6; i++) {
        result[i] = [
            y0[i][0] + dt*(k1[i][0] + 2*k2[i][0] + 2*k3[i][0] + k4[i][0])/6,
            y0[i][1] + dt*(k1[i][1] + 2*k2[i][1] + 2*k3[i][1] + k4[i][1])/6,
            y0[i][2] + dt*(k1[i][2] + 2*k2[i][2] + 2*k3[i][2] + k4[i][2])/6
        ];
    }
    return vecToState(result, state.time + dt);
}

// Track orbit on lattice
interface LatticeTrace {
    time: number;
    e8Node: number;
    latticeError: number;
    cell24: number;
    energy: number;
}

function traceOrbit(
    initial: ThreeBodyState,
    totalTime: number,
    dt: number
): LatticeTrace[] {
    const trace: LatticeTrace[] = [];
    let state = initial;
    const lattice = getDefaultLattice600();

    const numSteps = Math.ceil(totalTime / dt);
    for (let i = 0; i <= numSteps; i++) {
        const phase = encodeToPhaseSpace(state);
        const cellMap = mapTo600Cell(state);
        const energy = computeEnergy(state);

        trace.push({
            time: state.time,
            e8Node: phase.nearestE8Node,
            latticeError: phase.latticeError,
            cell24: cellMap.cell24Index,
            energy
        });

        if (i < numSteps) {
            state = rk4Step(state, dt);
        }
    }

    return trace;
}

// ASCII visualization of lattice error over time
function plotLatticeError(trace: LatticeTrace[], title: string) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`${title}`);
    console.log(`${'='.repeat(60)}`);

    const width = 60;
    const height = 15;

    // Find min/max error
    const errors = trace.map(t => t.latticeError);
    const minErr = Math.min(...errors);
    const maxErr = Math.max(...errors);
    const range = maxErr - minErr || 1;

    // Create ASCII plot
    const grid: string[][] = Array(height).fill(null).map(() => Array(width).fill(' '));

    // Plot error line
    for (let i = 0; i < trace.length; i++) {
        const x = Math.floor(i / trace.length * (width - 1));
        const y = height - 1 - Math.floor((trace[i].latticeError - minErr) / range * (height - 1));
        if (y >= 0 && y < height) {
            grid[y][x] = '●';
        }
    }

    // Add axes
    for (let y = 0; y < height; y++) {
        const errVal = maxErr - (y / (height - 1)) * range;
        const label = errVal.toFixed(3).padStart(6);
        console.log(`${label} │${grid[y].join('')}│`);
    }
    console.log(`${'─'.repeat(7)}┼${'─'.repeat(width)}┤`);
    console.log(`  Time  │ 0${' '.repeat(width/2 - 3)}${(trace[trace.length-1].time).toFixed(1)}${' '.repeat(width/2 - 3)}│`);

    // Statistics
    const avgError = errors.reduce((a, b) => a + b, 0) / errors.length;
    const stdDev = Math.sqrt(errors.map(e => (e - avgError) ** 2).reduce((a, b) => a + b, 0) / errors.length);

    console.log(`\nLattice Error Statistics:`);
    console.log(`  Mean: ${avgError.toFixed(6)}`);
    console.log(`  Std:  ${stdDev.toFixed(6)}`);
    console.log(`  Min:  ${minErr.toFixed(6)}`);
    console.log(`  Max:  ${maxErr.toFixed(6)}`);

    // Energy conservation
    const energies = trace.map(t => t.energy);
    const initialE = energies[0];
    const finalE = energies[energies.length - 1];
    const maxEnergyDrift = Math.max(...energies.map(e => Math.abs(e - initialE)));

    console.log(`\nEnergy Conservation:`);
    console.log(`  Initial: ${initialE.toFixed(6)}`);
    console.log(`  Final:   ${finalE.toFixed(6)}`);
    console.log(`  Drift:   ${((finalE - initialE) / Math.abs(initialE) * 100).toFixed(6)}%`);
    console.log(`  Max Err: ${(maxEnergyDrift / Math.abs(initialE) * 100).toFixed(6)}%`);

    // 24-cell transitions
    const cellTransitions = new Map<string, number>();
    for (let i = 1; i < trace.length; i++) {
        if (trace[i].cell24 !== trace[i-1].cell24) {
            const key = `${trace[i-1].cell24}→${trace[i].cell24}`;
            cellTransitions.set(key, (cellTransitions.get(key) || 0) + 1);
        }
    }

    console.log(`\n24-Cell Transitions (pattern coherence):`);
    if (cellTransitions.size === 0) {
        console.log(`  Orbit stays in single 24-cell (very stable)`);
    } else {
        const sorted = [...cellTransitions.entries()].sort((a, b) => b[1] - a[1]);
        for (const [trans, count] of sorted.slice(0, 5)) {
            console.log(`  ${trans}: ${count} times`);
        }
    }
}

// Main
console.log('╔════════════════════════════════════════════════════════════╗');
console.log('║     PPP ORBIT PATTERN VISUALIZATION                       ║');
console.log('║     Tracking how orbits trace paths on E8/600-cell        ║');
console.log('╚════════════════════════════════════════════════════════════╝');

// Initialize lattice
const lattice = getDefaultLattice600();

// Test Figure-8 orbit
const figure8 = createFigure8Orbit();
const figure8Trace = traceOrbit(figure8, 6.3, 0.01);  // ~1 period
plotLatticeError(figure8Trace, 'FIGURE-8 ORBIT (Chenciner-Montgomery)');

// Test Lagrange orbit
const lagrange = createLagrangeOrbit();
const lagrangeTrace = traceOrbit(lagrange, 6.3, 0.01);
plotLatticeError(lagrangeTrace, 'LAGRANGE EQUILATERAL ORBIT');

console.log('\n' + '═'.repeat(60));
console.log('INTERPRETATION:');
console.log('─'.repeat(60));
console.log('• Low, stable lattice error = orbit follows lattice structure');
console.log('• Periodic error pattern = orbit is periodic on lattice');
console.log('• Few 24-cell transitions = orbit is confined (stable)');
console.log('• Energy conservation shows RK4 accuracy');
console.log('═'.repeat(60));
