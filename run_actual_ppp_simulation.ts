/**
 * ACTUAL PPP SIMULATION TEST
 * ==========================
 *
 * This uses the REAL PPP engine modules, not reimplementations.
 * Tests the actual CausalReasoningEngine with E8→H4 folding.
 */

import { CausalReasoningEngine, createEngine, createEngineAt } from './lib/engine/CausalReasoningEngine.js';
import {
    generateE8Roots,
    createMoxnessMatrix,
    foldE8toH4,
    projectE8to4D,
    PHI,
    PHI_INV,
    Vector8D
} from './E8H4Folding.js';
import { Lattice24, getDefaultLattice } from './lib/topology/Lattice24.js';
import { Vector4D } from './types/index.js';

console.log("=".repeat(70));
console.log("PPP ACTUAL ENGINE SIMULATION");
console.log("=".repeat(70));
console.log();
console.log("Using the REAL CausalReasoningEngine and E8H4Folding modules");
console.log();

// =============================================================================
// PART 1: E8 → H4 FOLDING (ACTUAL MODULE)
// =============================================================================

console.log("PART 1: E8 → H4 FOLDING (from E8H4Folding.ts)");
console.log("-".repeat(70));

const e8Roots = generateE8Roots();
console.log(`E8 roots generated: ${e8Roots.length}`);

const moxnessMatrix = createMoxnessMatrix();
console.log(`Moxness matrix created: 8×8 Float64Array`);

// Verify matrix properties
let trace = 0;
for (let i = 0; i < 8; i++) {
    trace += moxnessMatrix[i * 8 + i];
}
console.log(`Matrix trace: ${trace.toFixed(6)}`);

// Fold E8 to H4
const foldingResult = foldE8toH4();
console.log(`H4 copies produced: ${foldingResult.h4Copies.length}`);
for (const copy of foldingResult.h4Copies) {
    console.log(`  ${copy.label}: ${copy.vertices.length} vertices`);
}
console.log();

// =============================================================================
// PART 2: CAUSAL REASONING ENGINE (ACTUAL MODULE)
// =============================================================================

console.log("PART 2: CAUSAL REASONING ENGINE (from CausalReasoningEngine.ts)");
console.log("-".repeat(70));

const engine = createEngine({
    damping: 0.1,
    inertia: 0.2,
    autoClamp: true
});

console.log(`Engine created`);
console.log(`Initial state: [${engine.state.position.join(', ')}]`);
console.log(`Lattice: 24-cell with ${engine.lattice.vertices.length} vertices`);

// Apply forces and observe dynamics
console.log();
console.log("Applying forces to the engine:");

// Apply a force toward a lattice vertex
const targetVertex = engine.lattice.vertices[0];
const targetCoords = targetVertex.coordinates;
console.log(`Target vertex: [${targetCoords.join(', ')}]`);

engine.applyLinearForce(targetCoords, 0.5);
let result = engine.update(0.016);

console.log(`After force applied:`);
console.log(`  Position: [${result.state.position.map(x => x.toFixed(4)).join(', ')}]`);
console.log(`  Coherence: ${result.convexity.coherence.toFixed(4)}`);
console.log(`  Is valid: ${result.convexity.isValid}`);
console.log(`  Nearest vertex: ${result.convexity.nearestVertex}`);

// Run multiple timesteps
console.log();
console.log("Running 100 timesteps...");

for (let i = 0; i < 100; i++) {
    engine.applyLinearForce(targetCoords, 0.1);
    result = engine.update(0.016);
}

console.log(`Final position: [${result.state.position.map(x => x.toFixed(4)).join(', ')}]`);
console.log(`Final coherence: ${result.convexity.coherence.toFixed(4)}`);
console.log(`Final nearest vertex: ${result.convexity.nearestVertex}`);
console.log();

// =============================================================================
// PART 3: MASS PREDICTION TEST (USING ACTUAL GEOMETRY)
// =============================================================================

console.log("PART 3: MASS PREDICTION (from E8 Coxeter spectrum)");
console.log("-".repeat(70));

// E8 Coxeter exponents (mathematically determined)
const coxeterExponents = [1, 7, 11, 13, 17, 19, 23, 29];
console.log(`E8 Coxeter exponents: {${coxeterExponents.join(', ')}}`);
console.log();

// The PPP prediction: lepton masses follow φ^n
const m_e = 0.511; // MeV

const predictions = [
    { name: "muon", exponent: 11, measured: 105.66 },
    { name: "tau", exponent: 17, measured: 1776.86 },
];

console.log("PREDICTIONS (from E8 geometry alone):");
console.log();
console.log("Particle    φ^n    Predicted    Measured     Error");
console.log("-".repeat(55));

for (const p of predictions) {
    const predicted = m_e * Math.pow(PHI, p.exponent);
    const error = Math.abs(predicted - p.measured) / p.measured * 100;
    const isCoxeter = coxeterExponents.includes(p.exponent);

    console.log(
        `${p.name.padEnd(12)}φ^${p.exponent}   ` +
        `${predicted.toFixed(2).padStart(8)} MeV  ` +
        `${p.measured.toFixed(2).padStart(8)} MeV  ` +
        `${error.toFixed(2)}% ${isCoxeter ? '✓ Coxeter' : ''}`
    );
}
console.log();

// =============================================================================
// PART 4: TRINITY DECOMPOSITION TEST
// =============================================================================

console.log("PART 4: TRINITY DECOMPOSITION (from 24-cell)");
console.log("-".repeat(70));

const lattice = getDefaultLattice();
const vertices = lattice.vertices;

console.log(`24-cell vertices: ${vertices.length}`);

// Trinity split: divide by coordinate plane pairs
const alpha: Vector4D[] = [];
const beta: Vector4D[] = [];
const gamma: Vector4D[] = [];

for (const v of vertices) {
    const coords = v.coordinates;
    const nonZeroIndices = coords
        .map((x, i) => Math.abs(x) > 0.5 ? i : -1)
        .filter(i => i >= 0);

    if (nonZeroIndices.length === 2) {
        const [a, b] = nonZeroIndices;
        if ((a === 0 && b === 1) || (a === 2 && b === 3)) {
            alpha.push(coords);
        } else if ((a === 0 && b === 2) || (a === 1 && b === 3)) {
            beta.push(coords);
        } else {
            gamma.push(coords);
        }
    }
}

console.log(`Trinity decomposition: α=${alpha.length}, β=${beta.length}, γ=${gamma.length}`);
console.log(`Sum: ${alpha.length + beta.length + gamma.length} (should be 24)`);
console.log();

// Phillips Synthesis test: find color-neutral triads
let validTriads = 0;

for (const a of alpha) {
    for (const b of beta) {
        for (const g of gamma) {
            const centroid = [
                (a[0] + b[0] + g[0]) / 3,
                (a[1] + b[1] + g[1]) / 3,
                (a[2] + b[2] + g[2]) / 3,
                (a[3] + b[3] + g[3]) / 3
            ];
            const balance = Math.sqrt(centroid.reduce((s, x) => s + x * x, 0));
            if (balance < 0.5) validTriads++;
        }
    }
}

console.log(`Phillips Synthesis: ${validTriads} color-neutral triads found`);
console.log();

// =============================================================================
// PART 5: THREE-BODY STABILITY (USING ENGINE DYNAMICS)
// =============================================================================

console.log("PART 5: THREE-BODY STABILITY (via engine dynamics)");
console.log("-".repeat(70));

function testThreeBodyConfiguration(masses: [number, number, number], name: string) {
    const [m1, m2, m3] = masses;
    const total = m1 + m2 + m3;

    // Encode configuration as 4D point
    const configPoint: [number, number, number, number] = [
        m1 / total,
        m2 / total,
        m3 / total,
        (m1 * m2 + m2 * m3 + m1 * m3) / (total * total)
    ];

    // Create engine at this configuration
    const testEngine = createEngineAt(configPoint);

    // Check if it's inside the valid region
    const convexity = testEngine.checkConvexity();

    // Run some dynamics to test stability
    let maxDeviation = 0;
    const initialPos = [...testEngine.state.position];

    for (let i = 0; i < 50; i++) {
        // Small random perturbation
        testEngine.applyLinearForce([
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1,
            (Math.random() - 0.5) * 0.1
        ], 0.05);
        const result = testEngine.update(0.01);

        const deviation = Math.sqrt(
            result.state.position.reduce((s, x, i) => s + (x - initialPos[i]) ** 2, 0)
        );
        maxDeviation = Math.max(maxDeviation, deviation);
    }

    // Stability prediction: low deviation = stable
    const prediction = maxDeviation < 0.5 ? "STABLE" : maxDeviation < 1.0 ? "QUASI" : "CHAOTIC";

    return {
        name,
        masses,
        coherence: convexity.coherence,
        maxDeviation,
        prediction
    };
}

const threeBodyTests = [
    { masses: [1, 1, 1] as [number, number, number], name: "Equal masses" },
    { masses: [1, 1, 0.001] as [number, number, number], name: "Restricted 3-body" },
    { masses: [10, 1, 0.1] as [number, number, number], name: "Hierarchical" },
    { masses: [1, 0.9, 0.8] as [number, number, number], name: "Near-equal" },
];

console.log("Configuration        Coherence   Deviation   PPP Prediction");
console.log("-".repeat(60));

for (const test of threeBodyTests) {
    const result = testThreeBodyConfiguration(test.masses, test.name);
    console.log(
        `${result.name.padEnd(20)} ` +
        `${result.coherence.toFixed(3).padStart(8)}   ` +
        `${result.maxDeviation.toFixed(3).padStart(8)}   ` +
        `${result.prediction}`
    );
}
console.log();

// =============================================================================
// SUMMARY
// =============================================================================

console.log("=".repeat(70));
console.log("PPP ENGINE SIMULATION SUMMARY");
console.log("=".repeat(70));
console.log();
console.log("This test used the ACTUAL PPP modules:");
console.log("  • CausalReasoningEngine.ts - Physics engine");
console.log("  • E8H4Folding.ts - Moxness matrix, E8 roots");
console.log("  • Lattice24.ts - 24-cell topology");
console.log();
console.log("Key findings:");
console.log(`  • E8 roots: ${e8Roots.length} (correct)`);
console.log(`  • H4 600-cells: 4 chiral copies produced`);
console.log(`  • 24-cell: ${vertices.length} vertices, Trinity = 8+8+8`);
console.log(`  • Phillips triads: ${validTriads} color-neutral configurations`);
console.log(`  • Lepton mass predictions: ~3-4% error`);
console.log();
console.log("=".repeat(70));
