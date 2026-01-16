/**
 * Compute the determinant of the φ-coupled folding matrix
 * and analyze its structure
 */

const PHI = (1 + Math.sqrt(5)) / 2;
const a = 0.5;
const b = (PHI - 1) / 2;
const c = PHI / 2;

// Construct the full 8x8 matrix
const U: number[][] = [
    // H4L rows (0-3)
    [a, a, a, a, b, b, -b, -b],
    [a, a, -a, -a, b, -b, b, -b],
    [a, -a, a, -a, b, -b, -b, b],
    [a, -a, -a, a, b, b, -b, -b],
    // H4R rows (4-7)
    [c, c, c, c, -a, -a, a, a],
    [c, c, -c, -c, -a, a, -a, a],
    [c, -c, c, -c, -a, a, a, -a],
    [c, -c, -c, c, -a, -a, a, a]
];

// LU decomposition for determinant calculation
function determinant(matrix: number[][]): number {
    const n = matrix.length;
    const M = matrix.map(row => [...row]); // Deep copy
    let det = 1;

    for (let i = 0; i < n; i++) {
        // Find pivot
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) {
                maxRow = k;
            }
        }

        // Swap rows if needed
        if (maxRow !== i) {
            [M[i], M[maxRow]] = [M[maxRow], M[i]];
            det *= -1;
        }

        // Check for zero pivot
        if (Math.abs(M[i][i]) < 1e-15) {
            return 0;
        }

        det *= M[i][i];

        // Eliminate below
        for (let k = i + 1; k < n; k++) {
            const factor = M[k][i] / M[i][i];
            for (let j = i; j < n; j++) {
                M[k][j] -= factor * M[i][j];
            }
        }
    }

    return det;
}

// Compute determinant
const det = determinant(U);

console.log('╔════════════════════════════════════════════════════════════════╗');
console.log('║  DETERMINANT ANALYSIS OF THE φ-COUPLED FOLDING MATRIX         ║');
console.log('╚════════════════════════════════════════════════════════════════╝\n');

console.log('Matrix coefficients:');
console.log(`  a = 1/2 = ${a}`);
console.log(`  b = (φ-1)/2 = ${b}`);
console.log(`  c = φ/2 = ${c}\n`);

console.log('Computed determinant:');
console.log(`  det(U) = ${det}`);
console.log(`  |det(U)| = ${Math.abs(det)}\n`);

// Try to express in terms of φ
console.log('Attempting to express det(U) in terms of φ:');

const candidates = [
    { name: '1', value: 1 },
    { name: 'φ', value: PHI },
    { name: 'φ²', value: PHI * PHI },
    { name: 'φ³', value: PHI * PHI * PHI },
    { name: 'φ⁴', value: Math.pow(PHI, 4) },
    { name: '1/φ', value: 1/PHI },
    { name: '1/φ²', value: 1/(PHI*PHI) },
    { name: '√5', value: Math.sqrt(5) },
    { name: '5', value: 5 },
    { name: '√5/4', value: Math.sqrt(5)/4 },
    { name: '5/16', value: 5/16 },
    { name: '1/16', value: 1/16 },
    { name: '1/4', value: 1/4 },
    { name: '(φ-1/φ)⁴ = 1', value: 1 },
    { name: '((3-φ)(φ+2))² = 25', value: 25 },
    { name: '5²/16 = 25/16', value: 25/16 },
    { name: '√5/16', value: Math.sqrt(5)/16 },
    { name: '5/4', value: 5/4 },
    { name: 'φ⁴/16', value: Math.pow(PHI,4)/16 },
    { name: '(3-φ)²(φ+2)²/16 = 25/16', value: 25/16 },
    { name: '1/256', value: 1/256 },
    { name: '1/64', value: 1/64 },
    { name: '5/64', value: 5/64 },
    { name: '25/64', value: 25/64 },
    { name: '5/256', value: 5/256 },
    { name: '25/256', value: 25/256 },
];

const absDet = Math.abs(det);
for (const c of candidates) {
    if (Math.abs(absDet - c.value) < 1e-10) {
        console.log(`  |det(U)| = ${c.name} = ${c.value} ✓`);
    }
    if (Math.abs(absDet - c.value/4) < 1e-10) {
        console.log(`  |det(U)| = ${c.name}/4 = ${c.value/4} ✓`);
    }
    if (Math.abs(absDet - c.value*4) < 1e-10) {
        console.log(`  |det(U)| = ${c.name}×4 = ${c.value*4} ✓`);
    }
}

// More systematic search
console.log('\nSearching for pattern a^m × b^n × c^p × 2^q:');
for (let m = 0; m <= 8; m++) {
    for (let n = 0; n <= 8; n++) {
        for (let p = 0; p <= 8; p++) {
            if (m + n + p <= 8) {
                for (let q = -8; q <= 8; q++) {
                    const val = Math.pow(a, m) * Math.pow(b, n) * Math.pow(c, p) * Math.pow(2, q);
                    if (Math.abs(absDet - val) < 1e-10) {
                        console.log(`  a^${m} × b^${n} × c^${p} × 2^${q} = ${val}`);
                    }
                }
            }
        }
    }
}

// Try expressions with (3-φ) and (φ+2)
console.log('\nSearching with (3-φ) and (φ+2):');
const threeMinusPhi = 3 - PHI;
const phiPlusTwo = PHI + 2;
for (let m = 0; m <= 4; m++) {
    for (let n = 0; n <= 4; n++) {
        for (let q = -8; q <= 8; q++) {
            const val = Math.pow(threeMinusPhi, m) * Math.pow(phiPlusTwo, n) * Math.pow(2, q);
            if (Math.abs(absDet - val) < 1e-10) {
                console.log(`  (3-φ)^${m} × (φ+2)^${n} × 2^${q} = ${val}`);
            }
        }
    }
}

// Check eigenvalues
console.log('\n' + '═'.repeat(66));
console.log('EIGENVALUE ANALYSIS (approximate):');
console.log('═'.repeat(66));

// Power iteration for largest eigenvalue (rough estimate)
function matVecMult(M: number[][], v: number[]): number[] {
    return M.map(row => row.reduce((sum, val, i) => sum + val * v[i], 0));
}

function norm(v: number[]): number {
    return Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
}

// For a more complete analysis, compute U^T U
const UT = U[0].map((_, i) => U.map(row => row[i]));
const UTU: number[][] = [];
for (let i = 0; i < 8; i++) {
    UTU[i] = [];
    for (let j = 0; j < 8; j++) {
        UTU[i][j] = 0;
        for (let k = 0; k < 8; k++) {
            UTU[i][j] += UT[i][k] * U[k][j];
        }
    }
}

console.log('\nU^T U diagonal (squared singular values):');
for (let i = 0; i < 8; i++) {
    console.log(`  (U^T U)_{${i}${i}} = ${UTU[i][i].toFixed(6)}`);
}

console.log('\n' + '═'.repeat(66));
console.log('BLOCK STRUCTURE ANALYSIS:');
console.log('═'.repeat(66));

// Extract blocks
const A = U.slice(0, 4).map(row => row.slice(0, 4));  // Top-left
const B = U.slice(0, 4).map(row => row.slice(4, 8));  // Top-right
const C = U.slice(4, 8).map(row => row.slice(0, 4));  // Bottom-left
const D = U.slice(4, 8).map(row => row.slice(4, 8));  // Bottom-right

console.log('\nBlock determinants:');
console.log(`  det(A) = ${determinant(A).toFixed(6)} (top-left, a-coefficients)`);
console.log(`  det(B) = ${determinant(B).toFixed(6)} (top-right, b-coefficients)`);
console.log(`  det(C) = ${determinant(C).toFixed(6)} (bottom-left, c-coefficients)`);
console.log(`  det(D) = ${determinant(D).toFixed(6)} (bottom-right, a-coefficients)`);

const detA = determinant(A);
const detD = determinant(D);

console.log('\nBlock structure insight:');
console.log(`  det(A) = ${detA.toFixed(6)} = a⁴ × det(sign pattern) = (1/2)⁴ × 4 = 1/4`);
console.log(`  Checking: a⁴ × 4 = ${Math.pow(a, 4) * 4}`);

console.log('\n' + '═'.repeat(66));
console.log('FINAL DETERMINANT EXPRESSION:');
console.log('═'.repeat(66));

// The determinant value
console.log(`\n  det(U) = ${det}`);
console.log(`\n  Expressed as fraction of known constants:`);
console.log(`    det(U) / (1/16) = ${det / (1/16)}`);
console.log(`    det(U) / ((3-φ)²/16) = ${det / (threeMinusPhi*threeMinusPhi/16)}`);
console.log(`    det(U) / ((φ+2)²/16) = ${det / (phiPlusTwo*phiPlusTwo/16)}`);
console.log(`    det(U) / (5/16) = ${det / (5/16)}`);

// Check if det = ±(some nice expression)
const sqrtDet = Math.sqrt(Math.abs(det));
console.log(`\n  √|det(U)| = ${sqrtDet}`);
console.log(`    Compared to 1/4 = ${1/4}`);
console.log(`    Compared to √(3-φ)/4 = ${Math.sqrt(3-PHI)/4}`);
console.log(`    Compared to (3-φ)/4 = ${(3-PHI)/4}`);
