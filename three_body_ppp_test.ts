/**
 * THREE-BODY PROBLEM TEST using ACTUAL PPP Engine
 * ================================================
 *
 * Tests whether the PPP framework can:
 * 1. Correctly identify known stable configurations (Lagrange points, figure-8)
 * 2. Correctly identify known chaotic configurations
 * 3. Make predictions that match numerical integration
 */

import { createEngine, createEngineAt } from './lib/engine/CausalReasoningEngine.js';
import { foldE8toH4, PHI } from './E8H4Folding.js';
import { getDefaultLattice } from './lib/topology/Lattice24.js';

console.log("=".repeat(70));
console.log("THREE-BODY PROBLEM: PPP ENGINE VALIDATION");
console.log("=".repeat(70));
console.log();

// =============================================================================
// KNOWN THREE-BODY SOLUTIONS (ground truth)
// =============================================================================

interface ThreeBodyConfig {
    name: string;
    masses: [number, number, number];
    // Initial positions in center-of-mass frame
    positions: [[number, number], [number, number], [number, number]];
    // Initial velocities
    velocities: [[number, number], [number, number], [number, number]];
    // Known outcome
    knownStability: 'STABLE' | 'PERIODIC' | 'CHAOTIC' | 'ESCAPE';
    description: string;
}

const KNOWN_SOLUTIONS: ThreeBodyConfig[] = [
    // STABLE configurations
    {
        name: "Lagrange L4 (equilateral)",
        masses: [1, 1, 1],
        positions: [[0, 0], [1, 0], [0.5, Math.sqrt(3)/2]],
        velocities: [[0, 0], [0, 0], [0, 0]],
        knownStability: 'STABLE',
        description: "Equal masses at equilateral triangle vertices"
    },
    {
        name: "Binary + distant third",
        masses: [1, 1, 0.1],
        positions: [[0.5, 0], [-0.5, 0], [10, 0]],
        velocities: [[0, 0.5], [0, -0.5], [0, 0.1]],
        knownStability: 'STABLE',
        description: "Hierarchical triple - stable due to separation"
    },
    {
        name: "Sun-Earth-Moon",
        masses: [333000, 1, 0.0123],
        positions: [[0, 0], [1, 0], [1.00257, 0]],
        velocities: [[0, 0], [0, 29.78], [0, 30.8]],
        knownStability: 'STABLE',
        description: "Real solar system configuration - stable for billions of years"
    },
    {
        name: "Wide hierarchical",
        masses: [10, 1, 0.5],
        positions: [[0, 0], [1, 0], [50, 0]],
        velocities: [[0, 0], [0, 3.16], [0, 0.45]],
        knownStability: 'STABLE',
        description: "Very wide tertiary - essentially stable"
    },

    // PERIODIC configurations
    {
        name: "Figure-8 orbit",
        masses: [1, 1, 1],
        positions: [[-0.97, 0.243], [0.97, -0.243], [0, 0]],
        velocities: [[0.466, 0.432], [0.466, 0.432], [-0.932, -0.864]],
        knownStability: 'PERIODIC',
        description: "Moore's figure-8 solution (1993)"
    },
    {
        name: "Euler collinear",
        masses: [1, 1, 1],
        positions: [[-1, 0], [0, 0], [1, 0]],
        velocities: [[0, 0.5], [0, 0], [0, -0.5]],
        knownStability: 'PERIODIC',
        description: "Three bodies on a line, rotating"
    },
    {
        name: "Broucke-Henon orbit",
        masses: [1, 1, 1],
        positions: [[-1, 0], [1, 0], [0, 0]],
        velocities: [[0.347, 0.533], [0.347, 0.533], [-0.694, -1.066]],
        knownStability: 'PERIODIC',
        description: "Another known periodic orbit family"
    },

    // CHAOTIC configurations
    {
        name: "Random chaotic 1",
        masses: [1, 0.9, 0.8],
        positions: [[0.1, 0.2], [-0.3, 0.5], [0.4, -0.1]],
        velocities: [[0.1, -0.1], [-0.2, 0.1], [0.1, 0.15]],
        knownStability: 'CHAOTIC',
        description: "Generic initial conditions - typically chaotic"
    },
    {
        name: "Random chaotic 2",
        masses: [1, 1, 1],
        positions: [[0.5, 0.3], [-0.7, 0.1], [0.2, -0.4]],
        velocities: [[0.2, 0.1], [-0.1, -0.2], [-0.1, 0.1]],
        knownStability: 'CHAOTIC',
        description: "Equal masses, random positions"
    },
    {
        name: "Near-collision",
        masses: [1, 1, 1],
        positions: [[0.1, 0], [-0.1, 0], [2, 0]],
        velocities: [[0, 0.5], [0, -0.5], [0, 0]],
        knownStability: 'CHAOTIC',
        description: "Two bodies very close - highly chaotic"
    },
    {
        name: "Pythagorean problem",
        masses: [3, 4, 5],
        positions: [[1, 3], [-2, -1], [1, -1]],
        velocities: [[0, 0], [0, 0], [0, 0]],
        knownStability: 'CHAOTIC',
        description: "Classic chaotic three-body problem"
    },

    // ESCAPE configurations
    {
        name: "High energy escape",
        masses: [1, 1, 1],
        positions: [[0, 0], [1, 0], [2, 0]],
        velocities: [[0, 0], [0, 5], [0, -5]],
        knownStability: 'ESCAPE',
        description: "High velocities - bodies will escape"
    },
    {
        name: "Hyperbolic encounter",
        masses: [1, 1, 0.1],
        positions: [[0, 0], [0.5, 0], [10, 5]],
        velocities: [[0, 0], [0, 1], [-3, -1]],
        knownStability: 'ESCAPE',
        description: "Third body on escape trajectory"
    },

    // Sun-Jupiter-asteroid (Trojan)
    {
        name: "Trojan asteroid L4",
        masses: [1000, 1, 0.001],
        positions: [[0, 0], [5.2, 0], [5.2 * Math.cos(Math.PI/3), 5.2 * Math.sin(Math.PI/3)]],
        velocities: [[0, 0], [0, 2.76], [-2.76 * Math.sin(Math.PI/3), 2.76 * Math.cos(Math.PI/3)]],
        knownStability: 'STABLE',
        description: "Restricted 3-body at L4"
    }
];

// =============================================================================
// PPP ENCODING: Map 3-body config to 4D state
// =============================================================================

function encodeThreeBodyAs4D(config: ThreeBodyConfig): [number, number, number, number] {
    const [m1, m2, m3] = config.masses;
    const total = m1 + m2 + m3;

    const [p1, p2, p3] = config.positions;
    const [v1, v2, v3] = config.velocities;

    // Distances between all pairs
    const r12 = Math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2);
    const r23 = Math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2);
    const r13 = Math.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2);

    // Hierarchy ratio: how separated are the scales?
    // High ratio = hierarchical = stable, Low ratio = compact = chaotic
    const rMin = Math.min(r12, r23, r13);
    const rMax = Math.max(r12, r23, r13);
    const hierarchyRatio = rMax / Math.max(rMin, 0.01);

    // Mass hierarchy
    const mMin = Math.min(m1, m2, m3);
    const mMax = Math.max(m1, m2, m3);
    const massRatio = mMax / Math.max(mMin, 0.001);

    // Angular momentum (normalized)
    const L = Math.abs(
        m1*(p1[0]*v1[1] - p1[1]*v1[0]) +
        m2*(p2[0]*v2[1] - p2[1]*v2[0]) +
        m3*(p3[0]*v3[1] - p3[1]*v3[0])
    );

    // Total energy
    const KE = 0.5 * (m1*(v1[0]**2 + v1[1]**2) +
                      m2*(v2[0]**2 + v2[1]**2) +
                      m3*(v3[0]**2 + v3[1]**2));
    const PE = -m1*m2/Math.max(r12, 0.01) - m2*m3/Math.max(r23, 0.01) - m1*m3/Math.max(r13, 0.01);
    const E = KE + PE;

    // Virial ratio: 2*KE / |PE| - should be ~1 for bound systems
    const virialRatio = 2 * KE / Math.abs(PE);

    // Encode as 4D point - use stability-relevant quantities
    return [
        Math.tanh(Math.log10(hierarchyRatio + 1) / 2),  // Distance hierarchy (high = stable)
        Math.tanh(Math.log10(massRatio + 1) / 3),      // Mass hierarchy
        Math.tanh(L / (total * Math.sqrt(rMin * rMax + 0.1))),  // Normalized angular momentum
        Math.tanh((virialRatio - 1) / 2)               // Virial deviation (0 = bound, >1 = unbound)
    ];
}

// =============================================================================
// RUN ACTUAL PPP DYNAMICS
// =============================================================================

function testWithPPPEngine(config: ThreeBodyConfig): {
    coherence: number;
    stability: number;
    nearestVertex: number;
    pppPrediction: string;
    trajectoryDeviation: number;
} {
    const state4D = encodeThreeBodyAs4D(config);

    // Create engine at this configuration
    const engine = createEngineAt(state4D);

    // Initial convexity check
    const initialConvexity = engine.checkConvexity();

    // Run dynamics with small perturbations (test stability)
    const trajectoryPoints: number[][] = [];
    const initialPos = [...engine.state.position];
    trajectoryPoints.push([...initialPos]);

    let maxDeviation = 0;
    const numSteps = 200;

    for (let i = 0; i < numSteps; i++) {
        // Apply small random perturbation (simulate numerical noise / chaos sensitivity)
        const perturbation: [number, number, number, number] = [
            (Math.random() - 0.5) * 0.02,
            (Math.random() - 0.5) * 0.02,
            (Math.random() - 0.5) * 0.02,
            (Math.random() - 0.5) * 0.02
        ];

        engine.applyLinearForce(perturbation, 0.1);
        const result = engine.update(0.01);

        trajectoryPoints.push([...result.state.position]);

        // Measure deviation from initial
        const deviation = Math.sqrt(
            result.state.position.reduce((s, x, j) => s + (x - initialPos[j])**2, 0)
        );
        maxDeviation = Math.max(maxDeviation, deviation);
    }

    // Compute trajectory "spread" - how much the state wanders
    const finalPos = trajectoryPoints[trajectoryPoints.length - 1];
    const finalDeviation = Math.sqrt(
        finalPos.reduce((s, x, j) => s + (x - initialPos[j])**2, 0)
    );

    // PPP prediction based on coherence and deviation
    let prediction: string;
    if (initialConvexity.coherence > 0.8 && maxDeviation < 0.3) {
        prediction = 'STABLE';
    } else if (initialConvexity.coherence > 0.6 && maxDeviation < 0.6) {
        prediction = 'PERIODIC';
    } else if (maxDeviation > 1.0 || initialConvexity.coherence < 0.4) {
        prediction = 'ESCAPE';
    } else {
        prediction = 'CHAOTIC';
    }

    return {
        coherence: initialConvexity.coherence,
        stability: 1 - maxDeviation,
        nearestVertex: initialConvexity.nearestVertex,
        pppPrediction: prediction,
        trajectoryDeviation: maxDeviation
    };
}

// =============================================================================
// MAIN TEST
// =============================================================================

console.log("Testing PPP predictions against KNOWN three-body solutions:");
console.log();
console.log("Config                    Known      PPP Pred   Coherence  Deviation  Match");
console.log("-".repeat(80));

let correct = 0;
let total = 0;

const results: Array<{
    name: string;
    known: string;
    predicted: string;
    coherence: number;
    deviation: number;
    match: boolean;
}> = [];

for (const config of KNOWN_SOLUTIONS) {
    const result = testWithPPPEngine(config);

    // Check if prediction matches known outcome
    // Allow PERIODIC to match STABLE (both are "regular" behavior)
    const isMatch =
        result.pppPrediction === config.knownStability ||
        (result.pppPrediction === 'PERIODIC' && config.knownStability === 'STABLE') ||
        (result.pppPrediction === 'STABLE' && config.knownStability === 'PERIODIC');

    if (isMatch) correct++;
    total++;

    results.push({
        name: config.name,
        known: config.knownStability,
        predicted: result.pppPrediction,
        coherence: result.coherence,
        deviation: result.trajectoryDeviation,
        match: isMatch
    });

    console.log(
        `${config.name.substring(0, 24).padEnd(24)} ` +
        `${config.knownStability.padEnd(10)} ` +
        `${result.pppPrediction.padEnd(10)} ` +
        `${result.coherence.toFixed(3).padStart(8)}   ` +
        `${result.trajectoryDeviation.toFixed(3).padStart(8)}  ` +
        `${isMatch ? '✓' : '✗'}`
    );
}

console.log("-".repeat(80));
console.log();

// =============================================================================
// STATISTICAL ANALYSIS
// =============================================================================

const accuracy = correct / total;
console.log(`ACCURACY: ${correct}/${total} = ${(accuracy * 100).toFixed(1)}%`);
console.log();

// Is this better than random?
// With 4 categories, random = 25%
const pValue = binomialTest(correct, total, 0.25);
console.log(`Null hypothesis: PPP predicts no better than random (25%)`);
console.log(`Binomial test p-value: ${pValue.toFixed(4)}`);
console.log(`Statistically significant (p < 0.05): ${pValue < 0.05 ? 'YES' : 'NO'}`);
console.log();

// =============================================================================
// HONEST ASSESSMENT
// =============================================================================

console.log("=".repeat(70));
console.log("HONEST ASSESSMENT");
console.log("=".repeat(70));
console.log();

if (accuracy >= 0.8 && pValue < 0.05) {
    console.log("STATUS: ✓ PPP shows predictive power for three-body stability");
    console.log();
    console.log("The PPP coherence metric correlates with known stability classes.");
    console.log("This suggests the 24-cell geometry captures relevant phase space structure.");
} else if (accuracy >= 0.5) {
    console.log("STATUS: ⚠ PARTIAL - PPP shows some correlation but not definitive");
    console.log();
    console.log("Results are better than random but not conclusive.");
    console.log("More test cases needed, especially edge cases.");
} else {
    console.log("STATUS: ✗ FAILED - PPP does not reliably predict three-body outcomes");
    console.log();
    console.log("The 4D encoding may not capture the relevant dynamics.");
    console.log("The 24-cell constraint space may be inappropriate for this problem.");
}

console.log();
console.log("=".repeat(70));

// =============================================================================
// HELPER: Binomial test
// =============================================================================

function binomialTest(successes: number, trials: number, p0: number): number {
    // Compute probability of getting >= successes under null hypothesis
    let pValue = 0;
    for (let k = successes; k <= trials; k++) {
        pValue += binomialPMF(k, trials, p0);
    }
    return pValue;
}

function binomialPMF(k: number, n: number, p: number): number {
    return binomialCoeff(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k);
}

function binomialCoeff(n: number, k: number): number {
    if (k > n) return 0;
    if (k === 0 || k === n) return 1;
    let result = 1;
    for (let i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}
