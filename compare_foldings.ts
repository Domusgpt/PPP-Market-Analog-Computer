/**
 * Comparison Test: φ-Coupled vs Orthonormal E8→H4 Folding
 *
 * This script compares the two matrix implementations to understand
 * what each produces and whether the φ-coupled version is revealing
 * interesting sub-structure or is simply incorrect.
 */

import {
    createMoxnessMatrix,
    generateE8Roots as generateE8Roots_Original,
    foldE8toH4,
    applyMoxnessMatrix,
    extractH4Left
} from './lib/topology/E8H4Folding.js';

import {
    createOrthonormalMoxnessMatrix,
    createCoxeterProjectionMatrix,
    generateE8Roots as generateE8Roots_Ortho,
    foldE8toH4_Orthonormal,
    foldE8toH4_Normalized,
    verifyMatrixProperties
} from './lib/topology/E8H4Folding_Orthonormal.js';

const PHI = (1 + Math.sqrt(5)) / 2;

console.log('╔════════════════════════════════════════════════════════════════╗');
console.log('║     E8 → H4 Folding Matrix Comparison                          ║');
console.log('║     φ-Coupled (Current) vs Orthonormal (Corrected)             ║');
console.log('╚════════════════════════════════════════════════════════════════╝\n');

// =============================================================================
// PART 1: Matrix Properties
// =============================================================================

console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('PART 1: Matrix Properties\n');

const phiMatrix = createMoxnessMatrix();
const orthoMatrix = createOrthonormalMoxnessMatrix();
const coxeterMatrix = createCoxeterProjectionMatrix();

console.log('1a. φ-Coupled Matrix (Current):');
const phiProps = verifyMatrixProperties(phiMatrix);
console.log('    Row norms: [' + phiProps.rowNorms.map(n => n.toFixed(3)).join(', ') + ']');
console.log('    Max off-diagonal in M×Mᵀ: ' + phiProps.maxOffDiagonal.toFixed(4));
console.log('    Is orthonormal: ' + phiProps.isOrthonormal);

console.log('\n1b. Orthonormal Matrix (Row-normalized):');
const orthoProps = verifyMatrixProperties(orthoMatrix);
console.log('    Row norms: [' + orthoProps.rowNorms.map(n => n.toFixed(3)).join(', ') + ']');
console.log('    Max off-diagonal in M×Mᵀ: ' + orthoProps.maxOffDiagonal.toFixed(4));
console.log('    Is orthonormal: ' + orthoProps.isOrthonormal);

console.log('\n1c. Coxeter Projection Matrix:');
const coxeterProps = verifyMatrixProperties(coxeterMatrix);
console.log('    Row norms: [' + coxeterProps.rowNorms.map(n => n.toFixed(3)).join(', ') + ']');
console.log('    Max off-diagonal in M×Mᵀ: ' + coxeterProps.maxOffDiagonal.toFixed(4));
console.log('    Is orthonormal: ' + coxeterProps.isOrthonormal);

// =============================================================================
// PART 2: Cross-Block Coupling Analysis
// =============================================================================

console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('PART 2: Cross-Block Coupling (Row0·Row4)\n');

function computeRow0Row4Dot(matrix: Float64Array): number {
    let dot = 0;
    for (let k = 0; k < 8; k++) {
        dot += matrix[0 * 8 + k] * matrix[4 * 8 + k];
    }
    return dot;
}

console.log('φ-Coupled:   Row0·Row4 = ' + computeRow0Row4Dot(phiMatrix).toFixed(6));
console.log('             Note: φ - 1/φ = ' + (PHI - 1/PHI).toFixed(6) + ' = 1 exactly');
console.log('Orthonormal: Row0·Row4 = ' + computeRow0Row4Dot(orthoMatrix).toFixed(6));
console.log('Coxeter:     Row0·Row4 = ' + computeRow0Row4Dot(coxeterMatrix).toFixed(6));

// =============================================================================
// PART 3: Projection Output Comparison
// =============================================================================

console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('PART 3: Projection Output Counts\n');

// Original φ-coupled
const phiResult = foldE8toH4();
console.log('3a. φ-Coupled Matrix:');
console.log('    H4L unit scale:  ' + phiResult.h4L.length + ' vertices');
console.log('    H4L φ-scale:     ' + phiResult.h4L_phi.length + ' vertices');
console.log('    H4R unit scale:  ' + phiResult.h4R.length + ' vertices');
console.log('    H4R φ-scale:     ' + phiResult.h4R_phi.length + ' vertices');
console.log('    TOTAL:           ' + (phiResult.h4L.length + phiResult.h4L_phi.length +
                                       phiResult.h4R.length + phiResult.h4R_phi.length));

// Orthonormal
const orthoResult = foldE8toH4_Orthonormal();
console.log('\n3b. Orthonormal Matrix:');
console.log('    Unique 4D points:  ' + orthoResult.uniqueVertices.length);
console.log('    Unit scale:        ' + orthoResult.unitScale.length);
console.log('    φ-scale:           ' + orthoResult.phiScale.length);

// Normalized output (φ-coupled matrix with output normalization)
const normalizedResult = foldE8toH4_Normalized();
console.log('\n3c. φ-Coupled + Output Normalization:');
console.log('    Unique normalized: ' + normalizedResult.uniqueVertices.length);

// =============================================================================
// PART 4: Norm Distribution Analysis
// =============================================================================

console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('PART 4: Norm Distribution of Projected Vectors\n');

function getNormDistribution(projections: [number, number, number, number][]): Map<number, number> {
    const counts = new Map<number, number>();
    for (const p of projections) {
        const norm = Math.sqrt(p[0]**2 + p[1]**2 + p[2]**2 + p[3]**2);
        const rounded = Math.round(norm * 100) / 100;
        counts.set(rounded, (counts.get(rounded) || 0) + 1);
    }
    return new Map([...counts.entries()].sort((a, b) => a[0] - b[0]));
}

// Get norms from φ-coupled projections
const e8Roots = generateE8Roots_Original();
const phiProjections: [number, number, number, number][] = [];
for (const root of e8Roots) {
    const rotated = applyMoxnessMatrix(root, phiMatrix);
    phiProjections.push(extractH4Left(rotated));
}

const phiNorms = getNormDistribution(phiProjections);
console.log('4a. φ-Coupled norm distribution:');
for (const [norm, count] of phiNorms) {
    const bar = '█'.repeat(Math.min(count / 4, 20));
    const phiRelation = Math.abs(norm - PHI) < 0.01 ? ' (≈φ)' :
                        Math.abs(norm - 1/PHI) < 0.01 ? ' (≈1/φ)' :
                        Math.abs(norm - PHI*PHI) < 0.01 ? ' (≈φ²)' :
                        Math.abs(norm - 1) < 0.01 ? ' (≈1)' :
                        Math.abs(norm - Math.sqrt(2)) < 0.01 ? ' (≈√2)' :
                        Math.abs(norm - Math.sqrt(3)) < 0.01 ? ' (≈√3)' : '';
    console.log('    ' + norm.toFixed(2) + ': ' + count.toString().padStart(3) + ' ' + bar + phiRelation);
}

// Get norms from orthonormal projections
const orthoNorms = getNormDistribution(orthoResult.allProjections);
console.log('\n4b. Orthonormal norm distribution:');
for (const [norm, count] of orthoNorms) {
    const bar = '█'.repeat(Math.min(count / 4, 20));
    console.log('    ' + norm.toFixed(2) + ': ' + count.toString().padStart(3) + ' ' + bar);
}

// =============================================================================
// PART 5: Tesseract Check (16 vertices)
// =============================================================================

console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('PART 5: Tesseract Structure Check\n');

// Check if the 16 H4L vertices form a tesseract
// Tesseract has 32 edges, each of length = edge length
// And specific distance patterns

function computeDistances(vertices: [number, number, number, number][]): number[] {
    const distances: number[] = [];
    for (let i = 0; i < vertices.length; i++) {
        for (let j = i + 1; j < vertices.length; j++) {
            const d = Math.sqrt(
                (vertices[i][0] - vertices[j][0])**2 +
                (vertices[i][1] - vertices[j][1])**2 +
                (vertices[i][2] - vertices[j][2])**2 +
                (vertices[i][3] - vertices[j][3])**2
            );
            distances.push(Math.round(d * 1000) / 1000);
        }
    }
    return distances;
}

function getDistanceCounts(distances: number[]): Map<number, number> {
    const counts = new Map<number, number>();
    for (const d of distances) {
        counts.set(d, (counts.get(d) || 0) + 1);
    }
    return new Map([...counts.entries()].sort((a, b) => a[0] - b[0]));
}

if (phiResult.h4L.length >= 8) {
    const h4LDistances = computeDistances(phiResult.h4L.slice(0, 16));
    const h4LDistCounts = getDistanceCounts(h4LDistances);

    console.log('5a. φ-Coupled H4L (' + phiResult.h4L.length + ' vertices) distance distribution:');
    console.log('    (Tesseract should have: 32 edges at d=edge, 24 face diags, 8 space diags)');
    for (const [dist, count] of h4LDistCounts) {
        console.log('    d=' + dist.toFixed(3) + ': ' + count + ' pairs');
    }
}

// =============================================================================
// PART 6: Summary and Conclusions
// =============================================================================

console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('PART 6: Summary\n');

console.log('┌─────────────────────┬──────────────┬──────────────┬──────────────┐');
console.log('│ Property            │ φ-Coupled    │ Orthonormal  │ With NormOut │');
console.log('├─────────────────────┼──────────────┼──────────────┼──────────────┤');
console.log('│ Row norms           │ 1.18, 1.90   │ ' +
    orthoProps.rowNorms[0].toFixed(2) + ', ' + orthoProps.rowNorms[4].toFixed(2) + '   │ N/A          │');
console.log('│ Row0·Row4           │ 1.0 (=φ-1/φ) │ ' +
    computeRow0Row4Dot(orthoMatrix).toFixed(4) + '       │ N/A          │');
console.log('│ Unique 4D vertices  │ ' +
    (phiResult.h4L.length + phiResult.h4L_phi.length).toString().padStart(4) + '         │ ' +
    orthoResult.uniqueVertices.length.toString().padStart(4) + '         │ ' +
    normalizedResult.uniqueVertices.length.toString().padStart(4) + '         │');
console.log('│ Is orthonormal?     │ No           │ ' +
    (orthoProps.isOrthonormal ? 'Yes' : 'No ').padEnd(12) + ' │ N/A          │');
console.log('└─────────────────────┴──────────────┴──────────────┴──────────────┘');

console.log('\n═══════════════════════════════════════════════════════════════════');
console.log('KEY FINDINGS:');
console.log('═══════════════════════════════════════════════════════════════════');

if (normalizedResult.uniqueVertices.length > 100) {
    console.log('✓ Output normalization yields ~' + normalizedResult.uniqueVertices.length +
                ' vertices (close to 600-cell\'s 120)');
    console.log('  → The φ-coupled matrix STRUCTURE is correct, just needs output normalization');
} else {
    console.log('✗ Even with normalization, only ' + normalizedResult.uniqueVertices.length +
                ' unique vertices');
    console.log('  → The matrix coefficients themselves may need adjustment');
}

if (phiResult.h4L.length === 16) {
    console.log('\n? The 16 H4L vertices may form an inscribed tesseract');
    console.log('  → Check distance distribution above for tesseract signature');
}

console.log('\n');
