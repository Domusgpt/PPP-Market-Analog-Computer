/**
 * Test: Does the PPP Matrix Actually Project E8 → H4 Correctly?
 *
 * CONTEXT: We discovered that the "Moxness matrix" in PPP is NOT the actual
 * Moxness folding matrix from the literature. This test determines whether
 * the PPP matrix has any practical utility for E8 → H4 projection.
 *
 * EXPECTED (if true E8→H4 projection):
 * - 240 E8 roots → 120 unique 4D points (2:1 mapping)
 * - Output should be exactly 600-cell vertices (possibly at 2 scales)
 *
 * @date January 2026
 */

const PHI = (1 + Math.sqrt(5)) / 2;
const PHI_INV = PHI - 1;

// The PPP matrix (NOT Moxness)
const a = 0.5;
const b = (PHI - 1) / 2;  // = 1/(2φ) ≈ 0.309
const c = PHI / 2;        // = φ/2 ≈ 0.809

const PPP_MATRIX: number[][] = [
  [ a,  a,  a,  a,  b,  b, -b, -b],  // row 0
  [ a,  a, -a, -a,  b, -b,  b, -b],  // row 1
  [ a, -a,  a, -a,  b, -b, -b,  b],  // row 2
  [ a, -a, -a,  a,  b,  b, -b, -b],  // row 3
  [ c,  c,  c,  c, -a, -a,  a,  a],  // row 4
  [ c,  c, -c, -c, -a,  a, -a,  a],  // row 5
  [ c, -c,  c, -c, -a,  a,  a, -a],  // row 6
  [ c, -c, -c,  c, -a, -a,  a,  a],  // row 7
];

console.log("=" .repeat(70));
console.log("TEST: DOES PPP MATRIX PROJECT E8 → H4 CORRECTLY?");
console.log("=" .repeat(70));

// =============================================================================
// 1. GENERATE E8 ROOTS
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
// 2. GENERATE 600-CELL VERTICES
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
    // Generate sign combinations for non-zero values
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
// 3. PROJECTION AND COMPARISON
// =============================================================================

function project8Dto4D(v8: number[]): [number[], number[]] {
  // Project using first 4 rows (H4L) and last 4 rows (H4R)
  const h4L: number[] = [];
  const h4R: number[] = [];

  for (let i = 0; i < 4; i++) {
    let sumL = 0, sumR = 0;
    for (let j = 0; j < 8; j++) {
      sumL += PPP_MATRIX[i][j] * v8[j];
      sumR += PPP_MATRIX[i + 4][j] * v8[j];
    }
    h4L.push(sumL);
    h4R.push(sumR);
  }

  return [h4L, h4R];
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
// RUN THE TEST
// =============================================================================

const e8Roots = generateE8Roots();
console.log(`\n## Generated ${e8Roots.length} E8 roots`);
console.log(`   Expected: 240 → ${e8Roots.length === 240 ? '✓ PASS' : '✗ FAIL'}`);

const cell600 = generate600CellVertices();
console.log(`\n## Generated ${cell600.length} 600-cell vertices`);
console.log(`   Expected: 120 → ${cell600.length === 120 ? '✓ PASS' : '✗ FAIL'}`);

// Project all E8 roots
console.log("\n## Projecting E8 roots through PPP matrix...");

const uniqueH4L = new Map<string, number[]>();
const uniqueH4R = new Map<string, number[]>();
const normCountsL = new Map<string, number>();
const normCountsR = new Map<string, number>();

for (const root of e8Roots) {
  const [h4L, h4R] = project8Dto4D(root);

  const keyL = quantize(h4L);
  const keyR = quantize(h4R);

  if (!uniqueH4L.has(keyL)) uniqueH4L.set(keyL, h4L);
  if (!uniqueH4R.has(keyR)) uniqueH4R.set(keyR, h4R);

  const normL = norm4D(h4L).toFixed(3);
  const normR = norm4D(h4R).toFixed(3);

  normCountsL.set(normL, (normCountsL.get(normL) || 0) + 1);
  normCountsR.set(normR, (normCountsR.get(normR) || 0) + 1);
}

console.log(`\n## Unique H4 Left projections: ${uniqueH4L.size}`);
console.log(`   (If E8→H4 works: should be 120 or 240)`);
console.log(`\n## Unique H4 Right projections: ${uniqueH4R.size}`);

console.log("\n## H4 Left norm distribution:");
const sortedNormsL = Array.from(normCountsL.entries()).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]));
for (const [norm, count] of sortedNormsL) {
  console.log(`   ${norm}: ${count} projections`);
}

console.log("\n## H4 Right norm distribution:");
const sortedNormsR = Array.from(normCountsR.entries()).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]));
for (const [norm, count] of sortedNormsR) {
  console.log(`   ${norm}: ${count} projections`);
}

// =============================================================================
// 4. CHECK IF PROJECTIONS MATCH 600-CELL VERTICES
// =============================================================================

console.log("\n" + "=" .repeat(70));
console.log("KEY TEST: DO PROJECTIONS LAND ON 600-CELL VERTICES?");
console.log("=" .repeat(70));

// Compute 600-cell norms (should all be 1)
const cell600Norms = cell600.map(v => norm4D(v).toFixed(4));
console.log(`\n600-cell vertex norms (sample): ${cell600Norms.slice(0, 5).join(', ')}...`);
console.log(`All norms = 1? ${cell600Norms.every(n => n === '1.0000') ? '✓ YES' : '✗ NO (mixed)'}`);

// For each unique projected point, find closest 600-cell vertex
function findClosest600CellVertex(point: number[]): [number, number] {
  let minDist = Infinity;
  let closestIdx = -1;

  for (let i = 0; i < cell600.length; i++) {
    const dist = distance4D(point, cell600[i]);
    if (dist < minDist) {
      minDist = dist;
      closestIdx = i;
    }
  }

  return [closestIdx, minDist];
}

console.log("\n## Testing if H4L projections match 600-cell vertices...");

// Separate by norm (might be scaled copies)
const projByNorm = new Map<string, number[][]>();
for (const [key, point] of uniqueH4L) {
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
console.log(`   ${totalMatches === totalPoints ? '✓ PPP MATRIX PRODUCES VALID 600-CELL' : '✗ PPP MATRIX DOES NOT PRODUCE 600-CELL'}`);

// =============================================================================
// 5. COMPARE WITH WHAT ACTUAL MOXNESS SHOULD PRODUCE
// =============================================================================

console.log("\n" + "=" .repeat(70));
console.log("COMPARISON: WHAT SHOULD PROPER E8→H4 PRODUCE?");
console.log("=" .repeat(70));

console.log(`
EXPECTED from actual Moxness (literature):
  - 240 E8 roots → 2 × 120 vertices (two 600-cells at scales 1 and 1/φ²)
  - Each 600-cell should have norm-1 vertices
  - The two 600-cells should be φ-related

WHAT PPP MATRIX PRODUCES:
  - ${uniqueH4L.size} unique H4L points
  - ${uniqueH4R.size} unique H4R points
  - Multiple norm shells (not clean φ-scaling)
`);

// Check if ANY norm matches φ-scaling
console.log("## Checking for φ-scaling relationship between norm shells...");
const normsArray = sortedNormsL.map(([n, _]) => parseFloat(n));
console.log(`   Observed norms: ${normsArray.join(', ')}`);

for (let i = 0; i < normsArray.length - 1; i++) {
  const ratio = normsArray[i + 1] / normsArray[i];
  const isPhiRelated = Math.abs(ratio - PHI) < 0.01 || Math.abs(ratio - PHI_INV) < 0.01;
  console.log(`   ${normsArray[i].toFixed(3)} → ${normsArray[i+1].toFixed(3)}: ratio = ${ratio.toFixed(4)} ${isPhiRelated ? '(φ-related!)' : ''}`);
}

// =============================================================================
// 6. FINAL VERDICT
// =============================================================================

console.log("\n" + "=" .repeat(70));
console.log("FINAL VERDICT");
console.log("=" .repeat(70));

const isValid600Cell = totalMatches === totalPoints;
const uniqueCountValid = uniqueH4L.size === 120 || uniqueH4L.size === 240;
const normDistributionSimple = sortedNormsL.length <= 2;

console.log(`
✓/✗ Produces valid 600-cell vertices: ${isValid600Cell ? '✓' : '✗'}
✓/✗ Correct unique point count (120 or 240): ${uniqueCountValid ? '✓' : '✗'}
✓/✗ Simple norm distribution (≤2 shells): ${normDistributionSimple ? '✓' : '✗'}

CONCLUSION: ${isValid600Cell && uniqueCountValid ?
  'PPP matrix IS a valid E8→H4 projection (produces 600-cell vertices)' :
  'PPP matrix is NOT a valid E8→H4 projection (does not produce 600-cell)'}

If NOT valid:
  - The PPP matrix is a φ-coupled Hadamard construction
  - It has interesting algebraic properties (√5 coupling, rank-7)
  - But it does NOT project E8 roots to H4/600-cell vertices
  - The "Moxness matrix" label is a misnomer
`);
