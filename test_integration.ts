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

    // TrinityDecomposition
    computeTrinityDecomposition,
    phillipsSynthesis
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
console.log('600-cell mapping: bodies in 24-cells [' + cellMapping.body1Cell24 + ', ' + cellMapping.body2Cell24 + ', ' + cellMapping.body3Cell24 + ']');

// Test Trinity decomposition
const trinity = computeTrinityDecomposition();
console.log('Trinity: α=' + trinity.alpha.vertices.length + ', β=' + trinity.beta.vertices.length + ', γ=' + trinity.gamma.vertices.length + ' vertices');

// Test Phillips synthesis
const synth = phillipsSynthesis(trinity.alpha.vertices[0], trinity.beta.vertices[0]);
console.log('Phillips synthesis: γ vertex = [' + synth[0].toFixed(3) + ', ' + synth[1].toFixed(3) + ', ' + synth[2].toFixed(3) + ', ' + synth[3].toFixed(3) + ']');

console.log('\n✓ All modules integrated successfully!');
