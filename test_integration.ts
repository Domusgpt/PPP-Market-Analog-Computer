/**
 * Quick integration test for 3-body modules
 */
import {
    // E8H4Folding
    generateE8Roots,
    projectE8to4D,
    foldE8toH4,

    // Lattice600
    Lattice600,
    getDefaultLattice600,

    // ThreeBodyPhaseSpace
    createFigure8Orbit,
    encodeToPhaseSpace,
    mapTo600Cell,
    computeEnergy,

    // TrinityDecomposition - Physics (Ali)
    computeTrinityDecomposition,
    phillipsSynthesis,

    // TrinityDecomposition - Musical (Phillips)
    computeMusicalTrialectic,
    computeTrinityStateVector,
    classifyTrinityState,
    detectPhaseShift,
    mapThreeBodiesToTrinity
} from './lib/topology/index.js';

console.log('Testing 3-body module integration...\n');

// Test E8 roots
const e8Roots = generateE8Roots();
console.log('E8 roots: ' + e8Roots.length + ' vectors (expected 240)');

// Test E8→H4 folding
const foldResult = foldE8toH4();
console.log('H4 folding: ' + foldResult.h4L.length + ' (L) + ' + foldResult.h4R.length + ' (R) vertices');

// Test 600-cell
const lattice = getDefaultLattice600();
console.log('600-cell: ' + lattice.vertices.length + ' vertices, ' + lattice.edges.length + ' edges');

// Test Figure-8 orbit
const figure8 = createFigure8Orbit();
const pos = figure8.body1.position;
console.log('Figure-8 orbit: body1 at [' + pos[0].toFixed(3) + ', ' + pos[1].toFixed(3) + ', ' + pos[2].toFixed(3) + ']');

// Test phase space encoding
const phasePoint = encodeToPhaseSpace(figure8);
console.log('Phase space: nearest E8 node = ' + phasePoint.nearestE8Node + ', error = ' + phasePoint.latticeError.toFixed(4));

// Test energy computation
const energy = computeEnergy(figure8);
console.log('Energy: ' + energy.toFixed(6));

// Test 600-cell mapping
const cellMapping = mapTo600Cell(figure8);
console.log('600-cell mapping: bodies in 24-cells [' + cellMapping.bodyAssignments.body1 + ', ' + cellMapping.bodyAssignments.body2 + ', ' + cellMapping.bodyAssignments.body3 + ']');

// Test Trinity decomposition
const trinity = computeTrinityDecomposition();
console.log('Trinity: α=' + trinity.alpha.vertices.length + ', β=' + trinity.beta.vertices.length + ', γ=' + trinity.gamma.vertices.length + ' vertices');

// Test Phillips synthesis
const synth = phillipsSynthesis(trinity.alpha.vertices[0], trinity.beta.vertices[0]);
console.log('Phillips synthesis: γ vertex = [' + synth[0].toFixed(3) + ', ' + synth[1].toFixed(3) + ', ' + synth[2].toFixed(3) + ', ' + synth[3].toFixed(3) + ']');

// Test Musical Trialectic (Phillips derivation)
console.log('\n--- Musical Trialectic (Phillips) ---');

const musicalTrinity = computeMusicalTrialectic();
console.log('Musical Trinity: α=' + musicalTrinity.alpha.dialectic + ' (' + musicalTrinity.alpha.harmonic + ')');
console.log('                 β=' + musicalTrinity.beta.dialectic + ' (' + musicalTrinity.beta.harmonic + ')');
console.log('                 γ=' + musicalTrinity.gamma.dialectic + ' (' + musicalTrinity.gamma.harmonic + ')');
console.log('Octatonic collections: [' + musicalTrinity.alpha.octatonic + ', ' + musicalTrinity.beta.octatonic + ', ' + musicalTrinity.gamma.octatonic + ']');

// Test Trinity State Vector
const testPoint: [number, number, number, number] = [0.5, 0.5, 0.3, 0.3];
const stateVector = computeTrinityStateVector(testPoint);
console.log('Trinity State Ψ = [α:' + stateVector.alpha.toFixed(3) + ', β:' + stateVector.beta.toFixed(3) + ', γ:' + stateVector.gamma.toFixed(3) + ']');

// Test state classification
const classification = classifyTrinityState(stateVector);
console.log('Classification: dominant=' + classification.dominant + ', entropy=' + classification.entropy.toFixed(3) + ', ' + classification.description);

// Test phase shift detection
const stateVector2 = computeTrinityStateVector([0.8, 0.1, 0.1, 0.0] as [number, number, number, number]);
const shift = detectPhaseShift(stateVector, stateVector2);
if (shift) {
    console.log('Phase shift: ' + shift.from + ' → ' + shift.to + ' (' + shift.type + ', tension=' + shift.tension.toFixed(3) + ')');
} else {
    console.log('Phase shift: none detected');
}

// Test 3-body to Trinity mapping
const body1: [number, number, number, number] = [1.0, 0.0, 0.0, 0.0];
const body2: [number, number, number, number] = [0.0, 1.0, 0.0, 0.0];
const body3: [number, number, number, number] = [0.0, 0.0, 1.0, 0.0];
const trinityMapping = mapThreeBodiesToTrinity(body1, body2, body3);
console.log('3-body → Trinity states:');
console.log('  Body1 Ψ = [α:' + trinityMapping.body1State.alpha.toFixed(3) + ', β:' + trinityMapping.body1State.beta.toFixed(3) + ', γ:' + trinityMapping.body1State.gamma.toFixed(3) + ']');
console.log('  Body2 Ψ = [α:' + trinityMapping.body2State.alpha.toFixed(3) + ', β:' + trinityMapping.body2State.beta.toFixed(3) + ', γ:' + trinityMapping.body2State.gamma.toFixed(3) + ']');
console.log('  Body3 Ψ = [α:' + trinityMapping.body3State.alpha.toFixed(3) + ', β:' + trinityMapping.body3State.beta.toFixed(3) + ', γ:' + trinityMapping.body3State.gamma.toFixed(3) + ']');
console.log('System locked: ' + trinityMapping.isLocked + ' (score: ' + trinityMapping.lockingScore.toFixed(3) + ')');

console.log('\n✓ All modules integrated successfully!');
