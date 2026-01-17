/**
 * Test to validate that TrinityDecomposition produces actual 16-cells
 *
 * The 24-cell can be decomposed into 3 disjoint 16-cells.
 * Each 16-cell (cross-polytope/orthoplex) has:
 * - 8 vertices
 * - 24 edges
 * - Vertices at distance √2 from each other (for unit 24-cell)
 */

type Vector4D = [number, number, number, number];

console.log("=".repeat(70));
console.log("VALIDATING TRINITY DECOMPOSITION (24-cell → 3×16-cell)");
console.log("=".repeat(70));

// Generate 24-cell vertices
function generate24CellVertices(): Vector4D[] {
  const vertices: Vector4D[] = [];

  for (let i = 0; i < 4; i++) {
    for (let j = i + 1; j < 4; j++) {
      for (const si of [-1, 1]) {
        for (const sj of [-1, 1]) {
          const v: Vector4D = [0, 0, 0, 0];
          v[i] = si;
          v[j] = sj;
          vertices.push(v);
        }
      }
    }
  }

  return vertices;
}

// Replicate the Trinity decomposition logic from TrinityDecomposition.ts
function computeTrinityDecomposition(): {
  alpha: { vertices: Vector4D[], ids: number[] },
  beta: { vertices: Vector4D[], ids: number[] },
  gamma: { vertices: Vector4D[], ids: number[] }
} {
  const vertices = generate24CellVertices();

  const alphaIds: number[] = [];
  const betaIds: number[] = [];
  const gammaIds: number[] = [];

  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i];

    // Determine which coordinates are non-zero
    const nonZeroIndices: number[] = [];
    for (let j = 0; j < 4; j++) {
      if (Math.abs(v[j]) > 0.5) {
        nonZeroIndices.push(j);
      }
    }

    // Classify into α, β, γ based on index pairs
    if (nonZeroIndices.length === 2) {
      const [a, b] = nonZeroIndices;
      if ((a === 0 && b === 1) || (a === 2 && b === 3)) {
        alphaIds.push(i);
      } else if ((a === 0 && b === 2) || (a === 1 && b === 3)) {
        betaIds.push(i);
      } else {
        gammaIds.push(i);
      }
    }
  }

  return {
    alpha: { vertices: alphaIds.map(i => vertices[i]), ids: alphaIds },
    beta: { vertices: betaIds.map(i => vertices[i]), ids: betaIds },
    gamma: { vertices: gammaIds.map(i => vertices[i]), ids: gammaIds }
  };
}

// Validate that a set of vertices forms a valid 16-cell
function validate16Cell(vertices: Vector4D[], label: string): boolean {
  console.log(`\n## Validating ${label}\n`);

  // Check vertex count
  console.log(`Vertices: ${vertices.length} (expected 8)`);
  if (vertices.length !== 8) {
    console.log(`  ERROR: Wrong vertex count`);
    return false;
  }

  // A 16-cell has vertices on 4 axes: ±e₁, ±e₂, ±e₃, ±e₄
  // In the context of 24-cell decomposition, the structure is different
  // Let's check edge structure

  // Count edges: In a 16-cell, each vertex is connected to all others EXCEPT its antipode
  // So 8 vertices, each with 6 neighbors = 24 edges
  const edgeLengthSq = 2.0; // √2 for unit coordinates
  const tolerance = 0.01;

  let edgeCount = 0;
  const edgeLengths = new Map<string, number>();

  for (let i = 0; i < vertices.length; i++) {
    for (let j = i + 1; j < vertices.length; j++) {
      const v1 = vertices[i];
      const v2 = vertices[j];
      const distSq = (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 +
                     (v1[2] - v2[2]) ** 2 + (v1[3] - v2[3]) ** 2;

      const key = Math.sqrt(distSq).toFixed(4);
      edgeLengths.set(key, (edgeLengths.get(key) || 0) + 1);

      if (Math.abs(distSq - edgeLengthSq) < tolerance) {
        edgeCount++;
      }
    }
  }

  console.log(`Edge count: ${edgeCount} (expected 24 for 16-cell)`);

  console.log(`Distance distribution:`);
  for (const [dist, count] of Array.from(edgeLengths.entries()).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))) {
    const distVal = parseFloat(dist);
    let label = "";
    if (Math.abs(distVal - Math.sqrt(2)) < 0.01) label = " (edge)";
    else if (Math.abs(distVal - 2) < 0.01) label = " (diameter/antipode)";
    else if (Math.abs(distVal - Math.sqrt(6)) < 0.01) label = " (across)";
    console.log(`  d = ${dist}: ${count} pairs${label}`);
  }

  // For a proper 16-cell:
  // - 24 edges at distance √2
  // - 4 diameters at distance 2 (antipodal pairs)

  const isValid16Cell = edgeCount === 24;

  // Also check that vertices form the correct structure
  // A 16-cell has vertex coordinates that are permutations of (±1, 0, 0, 0)
  // In the decomposed form, they may have different structure

  console.log(`\nVertices:`);
  for (const v of vertices) {
    console.log(`  [${v.map(x => x.toFixed(2)).join(', ')}]`);
  }

  return isValid16Cell;
}

// Run validation
const trinity = computeTrinityDecomposition();

console.log("\n## Trinity Decomposition Summary\n");
console.log(`α (Red/Gen1):   ${trinity.alpha.ids.length} vertices (ids: ${trinity.alpha.ids.join(', ')})`);
console.log(`β (Green/Gen2): ${trinity.beta.ids.length} vertices (ids: ${trinity.beta.ids.join(', ')})`);
console.log(`γ (Blue/Gen3):  ${trinity.gamma.ids.length} vertices (ids: ${trinity.gamma.ids.join(', ')})`);

const total = trinity.alpha.ids.length + trinity.beta.ids.length + trinity.gamma.ids.length;
console.log(`Total: ${total} (expected 24)`);

// Check for disjointness
const allIds = new Set<number>();
let duplicates = 0;
for (const id of [...trinity.alpha.ids, ...trinity.beta.ids, ...trinity.gamma.ids]) {
  if (allIds.has(id)) duplicates++;
  allIds.add(id);
}
console.log(`Disjoint: ${duplicates === 0 ? '✓' : `✗ (${duplicates} duplicates)`}`);

// Validate each cell
const alphaValid = validate16Cell(trinity.alpha.vertices, 'α (Red/Gen1)');
const betaValid = validate16Cell(trinity.beta.vertices, 'β (Green/Gen2)');
const gammaValid = validate16Cell(trinity.gamma.vertices, 'γ (Blue/Gen3)');

// Summary
console.log("\n" + "=".repeat(70));
console.log("SUMMARY");
console.log("=".repeat(70));

if (alphaValid && betaValid && gammaValid && total === 24 && duplicates === 0) {
  console.log("\n✓ Trinity decomposition is CORRECT");
  console.log("  - 24-cell splits into 3 disjoint 16-cells");
  console.log("  - Each 16-cell has 8 vertices and 24 edges");
  console.log("  - Corresponds to: 3 octatonic collections (music)");
  console.log("  - Corresponds to: 3 fermion generations (physics)");
} else {
  console.log("\n✗ Trinity decomposition has issues:");
  if (!alphaValid) console.log("  - α is not a valid 16-cell");
  if (!betaValid) console.log("  - β is not a valid 16-cell");
  if (!gammaValid) console.log("  - γ is not a valid 16-cell");
  if (total !== 24) console.log(`  - Total vertices = ${total}, expected 24`);
  if (duplicates > 0) console.log(`  - ${duplicates} duplicate vertex assignments`);
}
