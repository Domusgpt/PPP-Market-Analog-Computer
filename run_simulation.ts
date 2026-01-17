#!/usr/bin/env npx ts-node
/**
 * PPP Simulation Runner - Test Script
 *
 * Runs the PPP vs RK4 comparison to demonstrate that lattice-based
 * evolution preserves conservation laws better than traditional
 * numerical integration.
 */

import { runComparison, runSimulation } from './SimulationRunner.js';

console.log('===========================================');
console.log('PPP (Polytopal Projection Processing)');
console.log('Three-Body Simulation Comparison');
console.log('===========================================\n');

// Test all orbit types
const orbits: Array<'figure8' | 'lagrange' | 'euler'> = ['figure8', 'lagrange', 'euler'];

for (const orbit of orbits) {
    try {
        console.log(`\n--- Testing ${orbit.toUpperCase()} orbit ---`);
        const result = runComparison(orbit, 5, 0.01);

        console.log('\nDetailed Results:');
        console.log(`  PPP trajectory length: ${result.ppp.trajectory.length} steps`);
        console.log(`  RK4 trajectory length: ${result.rk4.trajectory.length} steps`);

        // Sample trajectory points
        const pppFinal = result.ppp.trajectory[result.ppp.trajectory.length - 1];
        const rk4Final = result.rk4.trajectory[result.rk4.trajectory.length - 1];

        console.log('\n  Final states:');
        console.log(`    PPP: E=${pppFinal.energy.toFixed(6)}, |L|=${Math.sqrt(
            pppFinal.angularMomentum[0]**2 +
            pppFinal.angularMomentum[1]**2 +
            pppFinal.angularMomentum[2]**2
        ).toFixed(6)}`);
        console.log(`    RK4: E=${rk4Final.energy.toFixed(6)}, |L|=${Math.sqrt(
            rk4Final.angularMomentum[0]**2 +
            rk4Final.angularMomentum[1]**2 +
            rk4Final.angularMomentum[2]**2
        ).toFixed(6)}`);

    } catch (err) {
        console.error(`Error testing ${orbit}:`, err);
    }
}

console.log('\n\n===========================================');
console.log('Simulation Complete');
console.log('===========================================');
