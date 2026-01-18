/**
 * Test: The ACTUAL Moxness Matrix from Literature
 *
 * This implements the real Moxness folding matrix as documented in:
 * - Moxness, J.G. "The 3D Visualization of E8 using an H4 Folding Matrix" (2014)
 * - Baez, J. "The 600-Cell" blog series (2020)
 *
 * The actual Moxness matrix is 4×8 and uses coefficients {0, ±1, φ, 1/φ, φ²}
 * NOT the Hadamard-like structure in PPP.
 *
 * @date January 2026
 */

const PHI = (1 + Math.sqrt(5)) / 2;
const PHI_INV = 1 / PHI;  // = φ - 1 ≈ 0.618
const PHI_SQ = PHI * PHI; // = φ + 1 ≈ 2.618

console.log("=" .repeat(70));
console.log("TEST: THE ACTUAL MOXNESS MATRIX FROM LITERATURE");
console.log("=" .repeat(70));

console.log(`\nGolden ratio constants:`);
console.log(`  φ = ${PHI.toFixed(6)}`);
console.log(`  1/φ = ${PHI_INV.toFixed(6)}`);
console.log(`  φ² = ${PHI_SQ.toFixed(6)}`);

/**
 * The ACTUAL Moxness 4×8 folding matrix S (from Baez/Conway-Sloane)
 *
 * This maps ℝ⁸ → ℝ⁴ such that:
 * - 120 E8 roots → unit quaternions (600-cell)
 * - 120 E8 roots → quaternions of norm 1/φ² (scaled 600-cell)
 *
 * Matrix from: https://johncarlosbaez.wordpress.com/2020/11/30/the-600-cell-part-4/
 *
 * S = [ 1   φ   0  -1   φ   0   0    0  ]
 *     [ φ   0   1   φ   0  -1   0    0  ]
 *     [ 0   1   φ   0  -1   φ   0    0  ]
 *     [ 0   0   0   0   0   0   φ²  1/φ ]
 *
 * Each row gives one 4D coordinate component.
 */
const MOXNESS_ACTUAL: number[][] = [
  [ 1,   PHI,   0,  -1,   PHI,   0,    0,       0     ],  // x-component
  [ PHI, 0,     1,   PHI, 0,    -1,    0,       0     ],  // y-component
  [ 0,   1,     PHI, 0,  -1,     PHI,  0,       0     ],  // z-component
  [ 0,   0,     0,   0,   0,     0,    PHI_SQ,  PHI_INV],  // w-component
];

// Also test the normalized version (divide by √(1+φ²) for unit E8 to unit H4)
const NORM_FACTOR = Math.sqrt(1 + PHI * PHI);  // ≈ 1.902
console.log(`\nNormalization factor √(1+φ²) = ${NORM_FACTOR.toFixed(6)}`);

// =============================================================================
// Generate E8 Roots
// =============================================================================

function generateE8Roots(): number[][] {
  const roots: number[][] = [];

  // Type D8: (±1, ±1, 0, 0, 0, 0, 0, 0) - 112 roots
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

  // Type S8: (±1/2)^8 with even parity - 128 roots
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

// =============================================================================
// Generate 600-Cell Vertices
// =============================================================================

function generate600CellVertices(): number[][] {
  const vertices: number[][] = [];

  // Type 1: 8 vertices - permutations of (±1, 0, 0, 0)
  for (let axis = 0; axis < 4; axis++) {
    for (const sign of [-1, 1]) {
      const v = [0, 0, 0, 0];
      v[axis] = sign;
      vertices.push(v);
    }
  }

  // Type 2: 16 vertices - (±1/2, ±1/2, ±1/2, ±1/2)
  for (let mask = 0; mask < 16; mask++) {
    vertices.push([
      (mask & 1) ? 0.5 : -0.5,
      (mask & 2) ? 0.5 : -0.5,
      (mask & 4) ? 0.5 : -0.5,
      (mask & 8) ? 0.5 : -0.5
    ]);
  }

  // Type 3: 96 vertices - even permutations of (0, ±1/(2φ), ±1/2, ±φ/2)
  const evenPerms = [
    [0,1,2,3], [0,2,3,1], [0,3,1,2],
    [1,0,3,2], [1,2,0,3], [1,3,2,0],
    [2,0,1,3], [2,1,3,0], [2,3,0,1],
    [3,0,2,1], [3,1,0,2], [3,2,1,0]
  ];

  const vals = [0, PHI_INV/2, 0.5, PHI/2];

  for (const perm of evenPerms) {
    for (let s1 = 0; s1 < 2; s1++) {
      for (let s2 = 0; s2 < 2; s2++) {
        for (let s3 = 0; s3 < 2; s3++) {
          const v = [0, 0, 0, 0];
          for (let i = 0; i < 4; i++) {
            const valIdx = perm[i];
            if (valIdx === 0) {
              v[i] = 0;
            } else if (valIdx === 1) {
              v[i] = (s1 ? -1 : 1) * vals[1];
            } else if (valIdx === 2) {
              v[i] = (s2 ? -1 : 1) * vals[2];
            } else {
              v[i] = (s3 ? -1 : 1) * vals[3];
            }
          }
          vertices.push(v);
        }
      }
    }
  }

  return vertices;
}

// =============================================================================
// Projection and Analysis
// =============================================================================

function projectMoxness(v8: number[]): number[] {
  const result: number[] = [];
  for (let i = 0; i < 4; i++) {
    let sum = 0;
    for (let j = 0; j < 8; j++) {
      sum += MOXNESS_ACTUAL[i][j] * v8[j];
    }
    result.push(sum);
  }
  return result;
}

function norm4D(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

function distance4D(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < 4; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

function quantize(v: number[], precision: number = 1000): string {
  return v.map(x => Math.round(x * precision) / precision).join(',');
}

// =============================================================================
// Run the Test
// =============================================================================

const e8Roots = generateE8Roots();
console.log(`\n## Generated ${e8Roots.length} E8 roots`);

const cell600 = generate600CellVertices();
console.log(`## Generated ${cell600.length} 600-cell vertices`);

// Project all E8 roots
const projected: number[][] = [];
const uniquePoints = new Map<string, number[]>();
const normCounts = new Map<string, number>();

for (const root of e8Roots) {
  const p = projectMoxness(root);
  projected.push(p);

  const key = quantize(p);
  if (!uniquePoints.has(key)) {
    uniquePoints.set(key, p);
  }

  const n = norm4D(p).toFixed(3);
  normCounts.set(n, (normCounts.get(n) || 0) + 1);
}

console.log(`\n## Projection results (actual Moxness):`);
console.log(`   Unique points: ${uniquePoints.size}`);
console.log(`   Expected for valid E8→H4: 120 or 240`);

console.log(`\n## Norm distribution:`);
const sortedNorms = Array.from(normCounts.entries()).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]));
for (const [n, count] of sortedNorms) {
  // Check if this norm is algebraically meaningful
  const nVal = parseFloat(n);
  let algebraic = '';
  if (Math.abs(nVal - 1) < 0.001) algebraic = ' (= 1)';
  if (Math.abs(nVal - PHI) < 0.001) algebraic = ' (= φ)';
  if (Math.abs(nVal - PHI_INV) < 0.001) algebraic = ' (= 1/φ)';
  if (Math.abs(nVal - PHI_SQ) < 0.001) algebraic = ' (= φ²)';
  if (Math.abs(nVal - Math.sqrt(2)) < 0.001) algebraic = ' (= √2)';
  if (Math.abs(nVal - NORM_FACTOR) < 0.001) algebraic = ' (= √(1+φ²))';

  console.log(`   ${n}: ${count} projections${algebraic}`);
}

// Check if projections match 600-cell (at various scales)
console.log("\n" + "=" .repeat(70));
console.log("CHECKING IF PROJECTIONS MATCH 600-CELL VERTICES");
console.log("=" .repeat(70));

// Group by norm
const projByNorm = new Map<string, number[][]>();
for (const [key, point] of uniquePoints) {
  const normKey = norm4D(point).toFixed(3);
  if (!projByNorm.has(normKey)) projByNorm.set(normKey, []);
  projByNorm.get(normKey)!.push(point);
}

let totalMatches = 0;
let totalPoints = 0;

for (const [normKey, points] of projByNorm) {
  const scale = parseFloat(normKey);

  // Scale 600-cell to this norm and check matches
  const scaledCell600 = cell600.map(v => v.map(x => x * scale));

  let matchCount = 0;
  for (const point of points) {
    let found = false;
    for (const vertex of scaledCell600) {
      if (distance4D(point, vertex) < 0.001) {
        found = true;
        break;
      }
    }
    if (found) matchCount++;
  }

  console.log(`   Norm ${normKey}: ${matchCount}/${points.length} match scaled 600-cell`);
  totalMatches += matchCount;
  totalPoints += points.length;
}

console.log(`\n   TOTAL: ${totalMatches}/${totalPoints} projections match 600-cell vertices`);

// =============================================================================
// Final Verdict
// =============================================================================

console.log("\n" + "=" .repeat(70));
console.log("COMPARISON: ACTUAL MOXNESS vs PPP MATRIX");
console.log("=" .repeat(70));

const moxnessWorks = totalMatches === totalPoints && (uniquePoints.size === 120 || uniquePoints.size === 240);

console.log(`
ACTUAL MOXNESS MATRIX:
  - Unique points: ${uniquePoints.size}
  - Norm shells: ${sortedNorms.length}
  - Match 600-cell: ${totalMatches}/${totalPoints}
  - Valid E8→H4: ${moxnessWorks ? '✓ YES' : '✗ NO'}

PPP MATRIX (from previous test):
  - Unique points: 226
  - Norm shells: 11
  - Match 600-cell: 36/226 (16%)
  - Valid E8→H4: ✗ NO

CONCLUSION: ${moxnessWorks ?
  'The ACTUAL Moxness matrix correctly projects E8 to H4 600-cell.' :
  'Neither matrix produces clean 600-cell projection (need to check matrix construction).'}
`);

// Additional analysis: look at the structure of the actual Moxness matrix
console.log("\n## Actual Moxness matrix structure:");
console.log("   Row 0:", MOXNESS_ACTUAL[0].map(x => x.toFixed(3)).join("  "));
console.log("   Row 1:", MOXNESS_ACTUAL[1].map(x => x.toFixed(3)).join("  "));
console.log("   Row 2:", MOXNESS_ACTUAL[2].map(x => x.toFixed(3)).join("  "));
console.log("   Row 3:", MOXNESS_ACTUAL[3].map(x => x.toFixed(3)).join("  "));
console.log(`
   Observations:
   - Coefficients: {0, ±1, φ, 1/φ, φ²}
   - Sparse structure (many zeros)
   - First 3 rows work on first 6 coordinates
   - Row 4 only uses last 2 coordinates (φ² and 1/φ)
`);
