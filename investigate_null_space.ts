/**
 * DEEP INVESTIGATION: Null Space and Linear Dependency
 *
 * Since rank(U) = 7 and nullity = 1, there is exactly ONE vector v
 * such that U × v = 0. What is this vector? What does it mean?
 */

const PHI = (1 + Math.sqrt(5)) / 2;
const a = 0.5;
const b = (PHI - 1) / 2;
const c = PHI / 2;

const U: number[][] = [
    [a, a, a, a, b, b, -b, -b],
    [a, a, -a, -a, b, -b, b, -b],
    [a, -a, a, -a, b, -b, -b, b],
    [a, -a, -a, a, b, b, -b, -b],
    [c, c, c, c, -a, -a, a, a],
    [c, c, -c, -c, -a, a, -a, a],
    [c, -c, c, -c, -a, a, a, -a],
    [c, -c, -c, c, -a, -a, a, a]
];

console.log('╔══════════════════════════════════════════════════════════════════════╗');
console.log('║  DEEP INVESTIGATION: Null Space Analysis                             ║');
console.log('╚══════════════════════════════════════════════════════════════════════╝\n');

// =============================================================================
// PART 1: Find the null space via reduced row echelon form
// =============================================================================
console.log('═'.repeat(74));
console.log('PART 1: Finding the Null Space');
console.log('═'.repeat(74) + '\n');

// Augment U with identity to find null space
function findNullSpace(matrix: number[][]): number[][] {
    const m = matrix.length;
    const n = matrix[0].length;
    const M = matrix.map(row => [...row]);

    // Gaussian elimination with full pivoting
    let pivotCol = 0;
    const pivotCols: number[] = [];

    for (let row = 0; row < m && pivotCol < n; row++) {
        // Find pivot
        let maxRow = row;
        for (let r = row + 1; r < m; r++) {
            if (Math.abs(M[r][pivotCol]) > Math.abs(M[maxRow][pivotCol])) {
                maxRow = r;
            }
        }

        if (Math.abs(M[maxRow][pivotCol]) < 1e-10) {
            pivotCol++;
            row--;
            continue;
        }

        // Swap rows
        [M[row], M[maxRow]] = [M[maxRow], M[row]];
        pivotCols.push(pivotCol);

        // Scale pivot row
        const pivot = M[row][pivotCol];
        for (let j = 0; j < n; j++) {
            M[row][j] /= pivot;
        }

        // Eliminate column
        for (let r = 0; r < m; r++) {
            if (r !== row) {
                const factor = M[r][pivotCol];
                for (let j = 0; j < n; j++) {
                    M[r][j] -= factor * M[row][j];
                }
            }
        }

        pivotCol++;
    }

    // Find free variables (columns not in pivotCols)
    const freeCols: number[] = [];
    for (let j = 0; j < n; j++) {
        if (!pivotCols.includes(j)) {
            freeCols.push(j);
        }
    }

    console.log('Pivot columns:', pivotCols);
    console.log('Free columns (null space basis):', freeCols);
    console.log('');

    // Construct null space basis
    const nullBasis: number[][] = [];
    for (const freeCol of freeCols) {
        const v = new Array(n).fill(0);
        v[freeCol] = 1;

        // Back-substitute to find other components
        for (let i = pivotCols.length - 1; i >= 0; i--) {
            const pivCol = pivotCols[i];
            v[pivCol] = -M[i][freeCol];
        }

        nullBasis.push(v);
    }

    return nullBasis;
}

// Work with U^T to find null space of U (left null space of U^T)
// Actually, we want null space of U: vectors v where Uv = 0
// This is the same as right null space, found from row echelon of U^T
const UT = U[0].map((_, j) => U.map(row => row[j]));
const nullSpace = findNullSpace(U);

console.log('Null space basis vectors:');
for (let i = 0; i < nullSpace.length; i++) {
    console.log(`  v${i} = [${nullSpace[i].map(x => x.toFixed(6)).join(', ')}]`);
}

// Verify: U × v should = 0
console.log('\nVerification U × v = 0:');
for (const v of nullSpace) {
    const result: number[] = [];
    for (let i = 0; i < 8; i++) {
        let sum = 0;
        for (let j = 0; j < 8; j++) {
            sum += U[i][j] * v[j];
        }
        result.push(sum);
    }
    console.log('  U × v = [' + result.map(x => x.toFixed(10)).join(', ') + ']');
    const isZero = result.every(x => Math.abs(x) < 1e-10);
    console.log('  All zeros? ' + (isZero ? '✓ YES' : '✗ NO'));
}

// =============================================================================
// PART 2: Understand the null space geometrically
// =============================================================================
console.log('\n' + '═'.repeat(74));
console.log('PART 2: Geometric Interpretation of Null Space');
console.log('═'.repeat(74) + '\n');

// The null space vector represents an 8D direction that gets collapsed to 0
// Let's analyze its structure
if (nullSpace.length > 0) {
    const v = nullSpace[0];

    console.log('Null space vector structure:');
    console.log('  First 4 components (multiply with cols 0-3):');
    for (let i = 0; i < 4; i++) {
        console.log(`    v[${i}] = ${v[i].toFixed(10)}`);
    }
    console.log('  Last 4 components (multiply with cols 4-7):');
    for (let i = 4; i < 8; i++) {
        console.log(`    v[${i}] = ${v[i].toFixed(10)}`);
    }

    // Check if components have golden ratio relationships
    console.log('\nLooking for φ relationships in null vector:');
    const components = v.filter(x => Math.abs(x) > 1e-10);
    if (components.length > 0) {
        const c0 = components[0];
        for (let i = 1; i < components.length; i++) {
            const ratio = components[i] / c0;
            console.log(`  v[${v.indexOf(components[i])}] / v[${v.indexOf(c0)}] = ${ratio.toFixed(10)}`);
            if (Math.abs(ratio - PHI) < 1e-6) console.log('    ≈ φ');
            if (Math.abs(ratio - 1/PHI) < 1e-6) console.log('    ≈ 1/φ');
            if (Math.abs(ratio + PHI) < 1e-6) console.log('    ≈ -φ');
            if (Math.abs(ratio + 1/PHI) < 1e-6) console.log('    ≈ -1/φ');
            if (Math.abs(ratio - 1) < 1e-6) console.log('    ≈ 1');
            if (Math.abs(ratio + 1) < 1e-6) console.log('    ≈ -1');
        }
    }
}

// =============================================================================
// PART 3: The Linear Dependency Between H4L and H4R
// =============================================================================
console.log('\n' + '═'.repeat(74));
console.log('PART 3: Linear Dependency Between H4L and H4R Subspaces');
console.log('═'.repeat(74) + '\n');

// Since rank = 7 and each block has rank 4, there must be exactly one
// linear dependency. Let's find which row is dependent on others.

console.log('Checking if any row is a linear combination of others...\n');

// Try to express Row 7 in terms of Row 0, 3, 4
// Based on the coupling pattern (Row0·Row7 = -1/φ, Row3·Row7 = 1)
// there might be a relationship

console.log('Coupling pattern suggests examining Row0, Row3, Row4, Row7:');
console.log('  Row0·Row4 = 1');
console.log('  Row0·Row7 = -1/φ');
console.log('  Row3·Row4 = -1/φ');
console.log('  Row3·Row7 = 1\n');

// Let's check: is Row3 + Row4 proportional to Row0 + Row7?
const row0 = U[0];
const row3 = U[3];
const row4 = U[4];
const row7 = U[7];

console.log('Checking Row0 + αRow4 vs Row3 + αRow7 for some α:');

// Row 0: [a, a, a, a, b, b, -b, -b]
// Row 3: [a, -a, -a, a, b, b, -b, -b]  <- same last 4 as Row 0!
// Row 4: [c, c, c, c, -a, -a, a, a]
// Row 7: [c, -c, -c, c, -a, -a, a, a]  <- same last 4 as Row 4!

console.log('\nObservation: Rows 0 and 3 have IDENTICAL last 4 entries');
console.log('             Rows 4 and 7 have IDENTICAL last 4 entries\n');

console.log('Row 0: [' + row0.map(x => x.toFixed(3)).join(', ') + ']');
console.log('Row 3: [' + row3.map(x => x.toFixed(3)).join(', ') + ']');
console.log('Row 4: [' + row4.map(x => x.toFixed(3)).join(', ') + ']');
console.log('Row 7: [' + row7.map(x => x.toFixed(3)).join(', ') + ']');

// Check: Row0 - Row3 only differs in first 4 entries
const diff03 = row0.map((x, i) => x - row3[i]);
const diff47 = row4.map((x, i) => x - row7[i]);
console.log('\nRow0 - Row3: [' + diff03.map(x => x.toFixed(3)).join(', ') + ']');
console.log('Row4 - Row7: [' + diff47.map(x => x.toFixed(3)).join(', ') + ']');

// These should be proportional!
const ratio_check = diff03.map((x, i) => Math.abs(diff47[i]) > 1e-10 ? x / diff47[i] : null);
console.log('\n(Row0 - Row3) / (Row4 - Row7) element-wise:');
console.log('  ' + ratio_check.map(x => x !== null ? x.toFixed(6) : 'N/A').join(', '));

// Find the actual linear dependency
console.log('\n' + '═'.repeat(74));
console.log('PART 4: The Exact Linear Dependency');
console.log('═'.repeat(74) + '\n');

// Let's try: α*Row0 + β*Row3 + γ*Row4 + δ*Row7 = 0
// Using the structure we observed

// Since last 4 entries of Row0 = last 4 of Row3, and same for 4,7
// Row0 - Row3 has zeros in last 4
// Row4 - Row7 has zeros in last 4
// These two differences span a 2D subspace of the first 4 coordinates

// The ratio for the first 4 entries:
// (Row0-Row3)[0:4] / (Row4-Row7)[0:4] = [2a, 2a, 2a, 0] / [2c, 2c, 2c, 0] = a/c = 1/φ

console.log('Key insight: First 4 entries of (Row0-Row3) vs (Row4-Row7):');
console.log(`  Row0-Row3: [${diff03.slice(0,4).map(x=>x.toFixed(4)).join(', ')}]`);
console.log(`  Row4-Row7: [${diff47.slice(0,4).map(x=>x.toFixed(4)).join(', ')}]`);

const ratio_first4 = diff03[0] / diff47[0];
console.log(`\n  Ratio: ${diff03[0]} / ${diff47[0]} = ${ratio_first4}`);
console.log(`  This equals a/c = ${a}/${c} = ${a/c}`);
console.log(`  Which equals 1/φ = ${1/PHI}`);
console.log(`  Match: ${Math.abs(ratio_first4 - 1/PHI) < 1e-10 ? '✓ YES' : '✗ NO'}`);

console.log('\nTherefore the linear dependency is:');
console.log('  (Row0 - Row3) = (1/φ) × (Row4 - Row7)');
console.log('  Or equivalently:');
console.log('  φ×Row0 - φ×Row3 - Row4 + Row7 = 0');

// Verify this
const verify = row0.map((_, i) => PHI*row0[i] - PHI*row3[i] - row4[i] + row7[i]);
console.log('\nVerification: φ×Row0 - φ×Row3 - Row4 + Row7 =');
console.log('  [' + verify.map(x => x.toFixed(10)).join(', ') + ']');
const isZero = verify.every(x => Math.abs(x) < 1e-10);
console.log('  All zeros? ' + (isZero ? '✓ YES - This is the linear dependency!' : '✗ NO'));

// =============================================================================
// SUMMARY
// =============================================================================
console.log('\n\n' + '═'.repeat(74));
console.log('SUMMARY: The Linear Dependency');
console.log('═'.repeat(74) + '\n');

console.log('The 8 rows of U satisfy exactly ONE linear relationship:');
console.log('');
console.log('    φ × Row₀  -  φ × Row₃  -  Row₄  +  Row₇  =  0');
console.log('');
console.log('This can be rewritten as:');
console.log('');
console.log('    Row₀ - Row₃  =  (1/φ) × (Row₄ - Row₇)');
console.log('');
console.log('INTERPRETATION:');
console.log('  The difference between H4L rows 0 and 3 is proportional');
console.log('  to the difference between H4R rows 4 and 7.');
console.log('  The proportionality constant is 1/φ (the inverse golden ratio).');
console.log('');
console.log('  This is why rank(U) = 7, not 8.');
console.log('  The golden ratio appears in the linear dependency itself!');
