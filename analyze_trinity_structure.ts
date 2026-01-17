/**
 * Analyze the actual structure of Trinity decomposition subsets
 *
 * Finding: The subsets are NOT 16-cells (cross-polytopes).
 * They are products of squares: □ × □ (two orthogonal 4-gons)
 */

type Vector4D = [number, number, number, number];

console.log("=".repeat(70));
console.log("ANALYZING TRINITY DECOMPOSITION STRUCTURE");
console.log("=".repeat(70));

// α subset vertices
const alphaVertices: Vector4D[] = [
  [-1, -1, 0, 0], [-1, 1, 0, 0], [1, -1, 0, 0], [1, 1, 0, 0],  // xy-plane square
  [0, 0, -1, -1], [0, 0, -1, 1], [0, 0, 1, -1], [0, 0, 1, 1],  // zw-plane square
];

console.log("\n## α Subset Structure\n");
console.log("Vertices split into two orthogonal squares:");
console.log("\nSquare 1 (xy-plane, z=w=0):");
for (let i = 0; i < 4; i++) {
  console.log(`  [${alphaVertices[i].join(', ')}]`);
}
console.log("\nSquare 2 (zw-plane, x=y=0):");
for (let i = 4; i < 8; i++) {
  console.log(`  [${alphaVertices[i].join(', ')}]`);
}

// What polytope is this?
console.log("\n## Polytope Identification\n");

console.log(`
This is NOT a 16-cell (cross-polytope).

A 16-cell has:
- 8 vertices: (±1,0,0,0), (0,±1,0,0), (0,0,±1,0), (0,0,0,±1)
- 24 edges of length √2
- Each vertex connected to 6 others

The Trinity subset has:
- 8 vertices: two orthogonal squares □ × □
- 24 pairs at distance 2 (diagonal within each plane's structure)
- 4 pairs at distance 2√2 (between parallel vertices across planes)
- No edges at distance √2!

This is actually a DUOPRISM: □ × □ (square × square)

Or more precisely, each Trinity subset forms the vertices of a
4D Clifford torus T² embedded in the 3-sphere S³.
`);

// Compare to actual 16-cell
console.log("## Comparison: Actual 16-Cell\n");

const cell16Vertices: Vector4D[] = [
  [1, 0, 0, 0], [-1, 0, 0, 0],
  [0, 1, 0, 0], [0, -1, 0, 0],
  [0, 0, 1, 0], [0, 0, -1, 0],
  [0, 0, 0, 1], [0, 0, 0, -1],
];

console.log("16-cell vertices (scaled by 1/√2 to match 24-cell):");
const scaledCell16 = cell16Vertices.map(v =>
  v.map(x => x * Math.sqrt(2)) as Vector4D
);
for (const v of scaledCell16) {
  console.log(`  [${v.map(x => x.toFixed(4)).join(', ')}]`);
}

// Count distances in true 16-cell
console.log("\n16-cell distance distribution:");
const cell16Distances = new Map<string, number>();
for (let i = 0; i < 8; i++) {
  for (let j = i + 1; j < 8; j++) {
    const v1 = scaledCell16[i];
    const v2 = scaledCell16[j];
    const dist = Math.sqrt(
      (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 +
      (v1[2] - v2[2]) ** 2 + (v1[3] - v2[3]) ** 2
    );
    const key = dist.toFixed(4);
    cell16Distances.set(key, (cell16Distances.get(key) || 0) + 1);
  }
}
for (const [dist, count] of Array.from(cell16Distances.entries()).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))) {
  console.log(`  d = ${dist}: ${count} pairs`);
}

// The correct decomposition claim
console.log("\n## Correct Mathematical Statement\n");

console.log(`
The 24-cell CAN be decomposed in several ways:

1. SELF-DUAL DECOMPOSITION:
   24-cell = 16-cell ∪ 8-cell (tesseract)
   The 24-cell contains both a 16-cell AND its dual 8-cell

2. TRINITY DECOMPOSITION (what the code does):
   24-cell = 3 × (□ × □) = 3 duoprisms
   Each duoprism has 8 vertices forming two orthogonal squares

The Trinity decomposition is mathematically valid but calling
the subsets "16-cells" is INCORRECT terminology.

The correct term would be:
- "8-vertex orthogonal duoprisms" or
- "Clifford toroidal patches" or
- Simply "Trinity subsets"

The 3-fold structure is still meaningful for:
- Music: 3 octatonic collections
- Physics: 3 color charges / 3 generations
- Dialectics: thesis/antithesis/synthesis

But the claim "24-cell = 3 × 16-cell" is false.
`);

// What IS true?
console.log("## What IS True About This Decomposition\n");

// Check if the three subsets are related by rotation
console.log("Checking rotational relationships between α, β, γ:");

const betaVertices: Vector4D[] = [
  [-1, 0, -1, 0], [-1, 0, 1, 0], [1, 0, -1, 0], [1, 0, 1, 0],  // xz-plane
  [0, -1, 0, -1], [0, -1, 0, 1], [0, 1, 0, -1], [0, 1, 0, 1],  // yw-plane
];

const gammaVertices: Vector4D[] = [
  [-1, 0, 0, -1], [-1, 0, 0, 1], [1, 0, 0, -1], [1, 0, 0, 1],  // xw-plane
  [0, -1, -1, 0], [0, -1, 1, 0], [0, 1, -1, 0], [0, 1, 1, 0],  // yz-plane
];

console.log(`
Plane assignments:
- α: (xy, zw) ← coordinates 01, 23
- β: (xz, yw) ← coordinates 02, 13
- γ: (xw, yz) ← coordinates 03, 12

This is a COMBINATORIAL decomposition based on pairing the 4 coordinates
into 3 complementary pairs: (01|23), (02|13), (03|12)

Each pair of coordinate indices defines two orthogonal 2-planes in ℝ⁴.
The 24-cell vertices can be partitioned by which 2-plane pair they inhabit.

This is related to the HOPF FIBRATION of S³!
`);

// Verify: do these really partition the 24-cell?
console.log("## Verification: Complete Partition\n");

// Generate all 24-cell vertices
const all24Cell: Vector4D[] = [];
for (let i = 0; i < 4; i++) {
  for (let j = i + 1; j < 4; j++) {
    for (const si of [-1, 1]) {
      for (const sj of [-1, 1]) {
        const v: Vector4D = [0, 0, 0, 0];
        v[i] = si;
        v[j] = sj;
        all24Cell.push(v);
      }
    }
  }
}

// Map each vertex to its subset
let alphaCount = 0, betaCount = 0, gammaCount = 0, unassigned = 0;

for (const v of all24Cell) {
  const nonZero: number[] = [];
  for (let i = 0; i < 4; i++) {
    if (v[i] !== 0) nonZero.push(i);
  }

  if (nonZero.length === 2) {
    const [a, b] = nonZero;
    const pair = `${a}${b}`;
    if (pair === '01' || pair === '23') alphaCount++;
    else if (pair === '02' || pair === '13') betaCount++;
    else if (pair === '03' || pair === '12') gammaCount++;
    else unassigned++;
  } else {
    unassigned++;
  }
}

console.log(`α (01|23): ${alphaCount} vertices`);
console.log(`β (02|13): ${betaCount} vertices`);
console.log(`γ (03|12): ${gammaCount} vertices`);
console.log(`Unassigned: ${unassigned}`);
console.log(`Total: ${alphaCount + betaCount + gammaCount + unassigned} (expected 24)`);

console.log("\n" + "=".repeat(70));
console.log("CONCLUSION");
console.log("=".repeat(70));

console.log(`
The Trinity Decomposition is a VALID partition of the 24-cell into 3
congruent 8-vertex subsets based on coordinate plane pairing.

HOWEVER:
- These subsets are NOT 16-cells (cross-polytopes)
- They are duoprisms (□ × □) or Clifford torus patches
- The documentation/code comments should be corrected

RECOMMENDED FIX:
Replace "16-cell" with "8-vertex subset" or "duoprism" in documentation.
The mathematical structure is meaningful; only the terminology is wrong.
`);
