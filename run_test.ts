/**
 * Quick test of PPP vs RK4 comparison
 */
import { runComparison } from './SimulationRunner.js';

console.log('Running PPP vs RK4 comparison on Figure-8 orbit...\n');

// Run shorter simulation for quick test
const result = runComparison('figure8', 5, 0.01);

console.log('\n--- Lattice Error Analysis ---');
const pppTrajectory = result.ppp.trajectory;
const lastFew = pppTrajectory.slice(-5);
for (const step of lastFew) {
    const t = step.time.toFixed(2);
    const le = step.latticeError?.toFixed(4) ?? 'N/A';
    const e = step.energy.toFixed(6);
    console.log(`t=${t}: latticeError=${le}, E=${e}`);
}
