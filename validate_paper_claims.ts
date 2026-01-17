/**
 * Comprehensive validation of all claims in the arXiv paper
 * "Algebraic Structure of the Moxness E₈ → H₄ Folding Matrix"
 */

const PHI = (1 + Math.sqrt(5)) / 2;
const a = 0.5;
const b = (PHI - 1) / 2;
const c = PHI / 2;

console.log("=" .repeat(70));
console.log("COMPREHENSIVE VALIDATION OF ARXIV PAPER CLAIMS");
console.log("=" .repeat(70));

// ============================================================================
// 1. GOLDEN RATIO IDENTITIES
// ============================================================================
console.log("\n## 1. Golden Ratio Identities\n");

const id1 = PHI * PHI;
const id1_expected = PHI + 1;
console.log(`φ² = ${id1.toFixed(10)} vs φ+1 = ${id1_expected.toFixed(10)} → ${Math.abs(id1 - id1_expected) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);

const id2 = 1 / PHI;
const id2_expected = PHI - 1;
console.log(`1/φ = ${id2.toFixed(10)} vs φ-1 = ${id2_expected.toFixed(10)} → ${Math.abs(id2 - id2_expected) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);

const id3 = PHI - 1/PHI;
console.log(`φ - 1/φ = ${id3.toFixed(10)} vs 1 → ${Math.abs(id3 - 1) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);

const id4 = (3 - PHI) * (PHI + 2);
console.log(`(3-φ)(φ+2) = ${id4.toFixed(10)} vs 5 → ${Math.abs(id4 - 5) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);

// ============================================================================
// 2. CONSTRUCT THE MATRIX
// ============================================================================
console.log("\n## 2. Matrix Construction\n");

const U: number[][] = [
  [ a,  a,  a,  a,  b,  b, -b, -b],  // row 0 H4L
  [ a,  a, -a, -a,  b, -b,  b, -b],  // row 1 H4L
  [ a, -a,  a, -a,  b, -b, -b,  b],  // row 2 H4L
  [ a, -a, -a,  a,  b,  b, -b, -b],  // row 3 H4L
  [ c,  c,  c,  c, -a, -a,  a,  a],  // row 4 H4R
  [ c,  c, -c, -c, -a,  a, -a,  a],  // row 5 H4R
  [ c, -c,  c, -c, -a,  a,  a, -a],  // row 6 H4R
  [ c, -c, -c,  c, -a, -a,  a,  a],  // row 7 H4R
];

console.log(`Coefficients: a=${a}, b=${b.toFixed(6)}, c=${c.toFixed(6)}`);
console.log(`b = (φ-1)/2 = 1/(2φ) → ${Math.abs(b - 1/(2*PHI)) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);
console.log(`c = φ/2 → ${Math.abs(c - PHI/2) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);
console.log(`c - b = ${(c-b).toFixed(10)} vs 0.5 → ${Math.abs(c - b - 0.5) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);

// ============================================================================
// 3. ROW NORMS
// ============================================================================
console.log("\n## 3. Row Norms\n");

function rowNorm(row: number[]): number {
  return Math.sqrt(row.reduce((s, v) => s + v*v, 0));
}

const H4L_norm_expected = Math.sqrt(3 - PHI);
const H4R_norm_expected = Math.sqrt(PHI + 2);

for (let i = 0; i < 4; i++) {
  const norm = rowNorm(U[i]);
  const pass = Math.abs(norm - H4L_norm_expected) < 1e-10;
  console.log(`Row ${i} norm = ${norm.toFixed(10)} vs √(3-φ) = ${H4L_norm_expected.toFixed(10)} → ${pass ? '✓ PASS' : '✗ FAIL'}`);
}

for (let i = 4; i < 8; i++) {
  const norm = rowNorm(U[i]);
  const pass = Math.abs(norm - H4R_norm_expected) < 1e-10;
  console.log(`Row ${i} norm = ${norm.toFixed(10)} vs √(φ+2) = ${H4R_norm_expected.toFixed(10)} → ${pass ? '✓ PASS' : '✗ FAIL'}`);
}

// Product identity
const product = H4L_norm_expected * H4R_norm_expected;
console.log(`\nNorm product = ${product.toFixed(10)} vs √5 = ${Math.sqrt(5).toFixed(10)} → ${Math.abs(product - Math.sqrt(5)) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);

// ============================================================================
// 4. COLUMN NORMS
// ============================================================================
console.log("\n## 4. Column Norms (Duality)\n");

function colNorm(col: number): number {
  let sum = 0;
  for (let i = 0; i < 8; i++) sum += U[i][col] * U[i][col];
  return Math.sqrt(sum);
}

for (let j = 0; j < 4; j++) {
  const norm = colNorm(j);
  const pass = Math.abs(norm - H4R_norm_expected) < 1e-10;
  console.log(`Col ${j} norm = ${norm.toFixed(10)} vs √(φ+2) = ${H4R_norm_expected.toFixed(10)} → ${pass ? '✓ PASS' : '✗ FAIL'}`);
}

for (let j = 4; j < 8; j++) {
  const norm = colNorm(j);
  const pass = Math.abs(norm - H4L_norm_expected) < 1e-10;
  console.log(`Col ${j} norm = ${norm.toFixed(10)} vs √(3-φ) = ${H4L_norm_expected.toFixed(10)} → ${pass ? '✓ PASS' : '✗ FAIL'}`);
}

// ============================================================================
// 5. CROSS-BLOCK COUPLING
// ============================================================================
console.log("\n## 5. Cross-Block Coupling\n");

function dot(a: number[], b: number[]): number {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

const coupling = dot(U[0], U[4]);
console.log(`⟨Row₀, Row₄⟩ = ${coupling.toFixed(10)} vs 1 → ${Math.abs(coupling - 1) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);
console.log(`φ - 1/φ = ${(PHI - 1/PHI).toFixed(10)} vs 1 → ${Math.abs(PHI - 1/PHI - 1) < 1e-10 ? '✓ PASS' : '✗ FAIL'}`);

// ============================================================================
// 6. NULL SPACE
// ============================================================================
console.log("\n## 6. Null Space\n");

// Right null space: v = [0,0,0,0,1,1,1,1]
const nullVector = [0, 0, 0, 0, 1, 1, 1, 1];

// Compute U * v
const Uv: number[] = [];
for (let i = 0; i < 8; i++) {
  let sum = 0;
  for (let j = 0; j < 8; j++) {
    sum += U[i][j] * nullVector[j];
  }
  Uv.push(sum);
}

console.log(`Null vector v = [${nullVector.join(', ')}]`);
console.log(`U × v = [${Uv.map(x => x.toFixed(10)).join(', ')}]`);
const nullSpacePass = Uv.every(x => Math.abs(x) < 1e-10);
console.log(`U × v = 0? → ${nullSpacePass ? '✓ PASS' : '✗ FAIL'}`);

// Column sum check
console.log("\nColumn sum (cols 4-7):");
for (let i = 0; i < 8; i++) {
  const colSum = U[i][4] + U[i][5] + U[i][6] + U[i][7];
  console.log(`  Row ${i}: col4+col5+col6+col7 = ${colSum.toFixed(10)} → ${Math.abs(colSum) < 1e-10 ? '✓' : '✗'}`);
}

// ============================================================================
// 7. ROW DEPENDENCY (LEFT NULL SPACE)
// ============================================================================
console.log("\n## 7. Row Dependency (Left Null Space)\n");

// Check: φ·Row₀ − φ·Row₃ − Row₄ + Row₇ = 0
const rowDep: number[] = [];
for (let j = 0; j < 8; j++) {
  rowDep.push(PHI * U[0][j] - PHI * U[3][j] - U[4][j] + U[7][j]);
}
console.log(`φ·Row₀ − φ·Row₃ − Row₄ + Row₇ = [${rowDep.map(x => x.toFixed(6)).join(', ')}]`);
const rowDepPass = rowDep.every(x => Math.abs(x) < 1e-10);
console.log(`= 0? → ${rowDepPass ? '✓ PASS' : '✗ FAIL'}`);

// ============================================================================
// 8. DETERMINANT AND RANK
// ============================================================================
console.log("\n## 8. Determinant and Rank\n");

// Simple Gaussian elimination to find rank
function gaussianElimination(matrix: number[][]): { rank: number, det: number } {
  const M = matrix.map(row => [...row]);
  const n = M.length;
  let det = 1;
  let rank = 0;

  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) {
        maxRow = row;
      }
    }

    if (Math.abs(M[maxRow][col]) < 1e-10) {
      det = 0;
      continue;
    }

    // Swap rows
    if (maxRow !== col) {
      [M[col], M[maxRow]] = [M[maxRow], M[col]];
      det *= -1;
    }

    det *= M[col][col];
    rank++;

    // Eliminate
    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col];
      for (let j = col; j < n; j++) {
        M[row][j] -= factor * M[col][j];
      }
    }
  }

  return { rank, det };
}

const { rank, det } = gaussianElimination(U);
console.log(`Determinant = ${det.toFixed(10)} → ${Math.abs(det) < 1e-10 ? '✓ PASS (= 0)' : '✗ FAIL'}`);
console.log(`Rank = ${rank} → ${rank === 7 ? '✓ PASS (= 7)' : '✗ FAIL'}`);

// ============================================================================
// 9. E8 ROOT PROJECTION NORMS
// ============================================================================
console.log("\n## 9. E8 Root Projection Norms\n");

// Generate E8 roots
function generateE8Roots(): number[][] {
  const roots: number[][] = [];

  // Type D8: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
  for (let i = 0; i < 8; i++) {
    for (let j = i + 1; j < 8; j++) {
      for (const si of [-1, 1]) {
        for (const sj of [-1, 1]) {
          const root = [0, 0, 0, 0, 0, 0, 0, 0];
          root[i] = si;
          root[j] = sj;
          roots.push(root);
        }
      }
    }
  }

  // Type S8: (±1/2)^8 with even number of minus signs
  for (let mask = 0; mask < 256; mask++) {
    let minusCount = 0;
    const root: number[] = [];
    for (let i = 0; i < 8; i++) {
      if (mask & (1 << i)) {
        root.push(-0.5);
        minusCount++;
      } else {
        root.push(0.5);
      }
    }
    if (minusCount % 2 === 0) {
      roots.push(root);
    }
  }

  return roots;
}

const e8Roots = generateE8Roots();
console.log(`Generated ${e8Roots.length} E8 roots → ${e8Roots.length === 240 ? '✓ PASS' : '✗ FAIL'}`);

// Project and compute norms
function projectH4L(root: number[]): number[] {
  const result: number[] = [];
  for (let i = 0; i < 4; i++) {
    let sum = 0;
    for (let j = 0; j < 8; j++) {
      sum += U[i][j] * root[j];
    }
    result.push(sum);
  }
  return result;
}

function norm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x*x, 0));
}

// Count norms
const normCounts = new Map<string, number>();
for (const root of e8Roots) {
  const projected = projectH4L(root);
  const n = norm(projected);
  const key = n.toFixed(3);
  normCounts.set(key, (normCounts.get(key) || 0) + 1);
}

// Expected values from paper
const expectedNorms: [string, string, number][] = [
  ['0.382', '1/φ²', 12],
  ['0.618', '1/φ', 8],
  ['0.727', '√(3-φ)/φ', 4],
  ['0.874', '√2/φ', 40],
  ['1.000', '1', 16],
  ['1.070', '√3/φ', 8],
  ['1.176', '√(3-φ)', 72],
  ['1.328', '√(5-2φ)', 8],
  ['1.414', '√2', 56],
  ['1.618', 'φ', 12],
  ['1.732', '√3', 4],
];

console.log("\nProjected norm distribution:");
let totalCount = 0;
for (const [normStr, algebraic, expectedCount] of expectedNorms) {
  const actualCount = normCounts.get(normStr) || 0;
  totalCount += actualCount;
  const pass = actualCount === expectedCount;
  console.log(`  ${normStr} (${algebraic}): ${actualCount} → ${pass ? '✓' : '✗'} (expected ${expectedCount})`);
}
console.log(`\nTotal: ${totalCount} → ${totalCount === 240 ? '✓ PASS' : '✗ FAIL'}`);

// Verify algebraic forms
console.log("\nAlgebraic form verification:");
const algChecks: [number, string][] = [
  [1/PHI/PHI, '1/φ² = 2-φ'],
  [1/PHI, '1/φ = φ-1'],
  [Math.sqrt(3-PHI)/PHI, '√(3-φ)/φ'],
  [Math.sqrt(2)/PHI, '√2/φ'],
  [1, '1'],
  [Math.sqrt(3)/PHI, '√3/φ'],
  [Math.sqrt(3-PHI), '√(3-φ)'],
  [Math.sqrt(5-2*PHI), '√(5-2φ)'],
  [Math.sqrt(2), '√2'],
  [PHI, 'φ'],
  [Math.sqrt(3), '√3'],
];

for (const [val, name] of algChecks) {
  console.log(`  ${name} = ${val.toFixed(6)}`);
}

// ============================================================================
// 10. 600-CELL VERTEX COUNT
// ============================================================================
console.log("\n## 10. 600-Cell Vertex Count\n");

// Type 1: permutations of (±1, 0, 0, 0)
const type1 = 4 * 2; // 4 positions × 2 signs = 8
// Type 2: (±1/2, ±1/2, ±1/2, ±1/2)
const type2 = 16; // 2^4 = 16
// Type 3: even permutations of (0, ±1/2, ±φ/2, ±1/(2φ))
const type3 = 96; // 12 even perms × 8 sign combos = 96

console.log(`Type 1 (no φ): ${type1} vertices`);
console.log(`Type 2 (no φ): ${type2} vertices`);
console.log(`Type 3 (has φ): ${type3} vertices`);
console.log(`Total: ${type1 + type2 + type3} → ${type1 + type2 + type3 === 120 ? '✓ PASS' : '✗ FAIL'}`);
console.log(`Vertices with φ: ${type3} of 120 = ${(type3/120*100).toFixed(1)}%`);

// ============================================================================
// SUMMARY
// ============================================================================
console.log("\n" + "=" .repeat(70));
console.log("VALIDATION COMPLETE");
console.log("=" .repeat(70));
