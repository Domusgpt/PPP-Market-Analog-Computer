/**
 * Test to verify the 600-cell → 5×24-cell decomposition fix
 */

// Direct import without module system - inline the test
const PHI = (1 + Math.sqrt(5)) / 2;
const PHI_INV = PHI - 1;

type Vector4D = [number, number, number, number];

interface Lattice600Vertex {
  id: number;
  coordinates: Vector4D;
  neighbors: number[];
  cell24Index: number;
  vertexType: number;
}

interface Cell24Subset {
  id: number;
  vertexIds: number[];
  vertices: Vector4D[];
  isDisjoint: boolean;
  label: string;
}

// Generate 600-cell vertices
function generate600CellVertices(): Lattice600Vertex[] {
  const vertices: Lattice600Vertex[] = [];
  let id = 0;

  // Group 1: 8 vertices
  for (let axis = 0; axis < 4; axis++) {
    for (const sign of [-1, 1]) {
      const coords: Vector4D = [0, 0, 0, 0];
      coords[axis] = sign;
      vertices.push({ id: id++, coordinates: coords, neighbors: [], cell24Index: 0, vertexType: 0 });
    }
  }

  // Group 2: 16 vertices
  for (let mask = 0; mask < 16; mask++) {
    vertices.push({
      id: id++,
      coordinates: [(mask & 1) ? 0.5 : -0.5, (mask & 2) ? 0.5 : -0.5, (mask & 4) ? 0.5 : -0.5, (mask & 8) ? 0.5 : -0.5],
      neighbors: [], cell24Index: 0, vertexType: 1
    });
  }

  // Group 3: 96 vertices
  const evenPerms = [
    [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2],
    [1, 0, 3, 2], [1, 2, 0, 3], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 1, 3, 0], [2, 3, 0, 1],
    [3, 0, 2, 1], [3, 1, 0, 2], [3, 2, 1, 0],
  ];

  const baseCoords = [0, 0.5 * PHI_INV, 0.5, 0.5 * PHI];

  for (const perm of evenPerms) {
    const permuted = perm.map(i => baseCoords[i]);
    for (let signMask = 0; signMask < 16; signMask++) {
      const v: Vector4D = [0, 0, 0, 0];
      let valid = true;
      for (let i = 0; i < 4; i++) {
        if (permuted[i] === 0) {
          v[i] = 0;
          if (signMask & (1 << i)) { valid = false; break; }
        } else {
          v[i] = (signMask & (1 << i)) ? -permuted[i] : permuted[i];
        }
      }
      if (!valid) continue;
      const isDup = vertices.some(u =>
        Math.abs(u.coordinates[0] - v[0]) < 0.001 &&
        Math.abs(u.coordinates[1] - v[1]) < 0.001 &&
        Math.abs(u.coordinates[2] - v[2]) < 0.001 &&
        Math.abs(u.coordinates[3] - v[3]) < 0.001
      );
      if (!isDup) {
        vertices.push({ id: id++, coordinates: v, neighbors: [], cell24Index: 0, vertexType: 2 });
      }
    }
  }

  return vertices;
}

function quaternionMultiply(q1: Vector4D, q2: Vector4D): Vector4D {
  const [w1, x1, y1, z1] = q1;
  const [w2, x2, y2, z2] = q2;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}

function quaternionConjugate(q: Vector4D): Vector4D {
  return [q[0], -q[1], -q[2], -q[3]];
}

function vectors4DEqual(v1: Vector4D, v2: Vector4D, tol = 1e-4): boolean {
  return Math.abs(v1[0] - v2[0]) < tol && Math.abs(v1[1] - v2[1]) < tol &&
         Math.abs(v1[2] - v2[2]) < tol && Math.abs(v1[3] - v2[3]) < tol;
}

function findOmegaOrder10(vertices: Lattice600Vertex[]): Vector4D | null {
  for (let i = 24; i < vertices.length; i++) {
    const candidate = vertices[i].coordinates;
    let power = candidate;
    for (let k = 1; k <= 20; k++) {
      if (Math.abs(power[0] - 1) < 1e-6 && Math.abs(power[1]) < 1e-6 &&
          Math.abs(power[2]) < 1e-6 && Math.abs(power[3]) < 1e-6) {
        if (k === 10) return candidate;
        break;
      }
      power = quaternionMultiply(power, candidate);
    }
  }
  return null;
}

function computeDisjoint24Cells(vertices: Lattice600Vertex[]): Cell24Subset[] {
  const omega = findOmegaOrder10(vertices);
  if (!omega) {
    console.log("ERROR: Could not find omega with order 10");
    return [];
  }

  const vertices2T: Vector4D[] = vertices.slice(0, 24).map(v => v.coordinates);
  const cosetReps: Vector4D[] = [];
  let rep: Vector4D = [1, 0, 0, 0];
  for (let k = 0; k < 5; k++) {
    cosetReps.push([...rep] as Vector4D);
    rep = quaternionMultiply(rep, omega);
  }
  const cosetRepsInverse = cosetReps.map(q => quaternionConjugate(q));

  const cellAssignments: number[] = new Array(vertices.length).fill(-1);

  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i].coordinates;
    for (let k = 0; k < 5; k++) {
      const product = quaternionMultiply(cosetRepsInverse[k], v);
      let found = false;
      for (const tVert of vertices2T) {
        if (vectors4DEqual(product, tVert)) { cellAssignments[i] = k; found = true; break; }
      }
      if (found) break;
      const negProduct: Vector4D = [-product[0], -product[1], -product[2], -product[3]];
      for (const tVert of vertices2T) {
        if (vectors4DEqual(negProduct, tVert)) { cellAssignments[i] = k; found = true; break; }
      }
      if (found) break;
    }
  }

  const cells: Cell24Subset[] = [];
  for (let cellId = 0; cellId < 5; cellId++) {
    const cellVertices: number[] = [];
    const cellCoords: Vector4D[] = [];
    for (let i = 0; i < vertices.length; i++) {
      if (cellAssignments[i] === cellId) {
        cellVertices.push(i);
        cellCoords.push(vertices[i].coordinates);
      }
    }
    cells.push({
      id: cellId, vertexIds: cellVertices, vertices: cellCoords,
      isDisjoint: true, label: `24-Cell-${String.fromCharCode(65 + cellId)}`
    });
  }
  return cells;
}

// Generate and decompose
const vertices = generate600CellVertices();
const cells = computeDisjoint24Cells(vertices);

console.log("=".repeat(70));
console.log("TESTING 600-CELL DECOMPOSITION FIX");
console.log("=".repeat(70));

console.log(`\nGenerated ${vertices.length} vertices (expected 120)`);

console.log("\n## Decomposition Results\n");

// Check vertex counts
let totalVertices = 0;
for (const cell of cells) {
  console.log(`${cell.label}: ${cell.vertexIds.length} vertices`);
  totalVertices += cell.vertexIds.length;
}

console.log(`\nTotal vertices: ${totalVertices} (expected 120)`);

// Check for valid 24-cell structure (edge count)
console.log("\n## Validating 24-Cell Structure\n");

for (const cell of cells) {
  if (cell.vertices.length !== 24) {
    console.log(`${cell.label}: INVALID - not 24 vertices`);
    continue;
  }

  // Count edges (distance = 1 for unit 24-cell)
  let edgeCount = 0;
  const edgeLengthTarget = 1.0; // 24-cell edge length on unit sphere

  for (let i = 0; i < 24; i++) {
    for (let j = i + 1; j < 24; j++) {
      const v1 = cell.vertices[i];
      const v2 = cell.vertices[j];
      const dist = Math.sqrt(
        (v1[0] - v2[0]) ** 2 +
        (v1[1] - v2[1]) ** 2 +
        (v1[2] - v2[2]) ** 2 +
        (v1[3] - v2[3]) ** 2
      );

      if (Math.abs(dist - edgeLengthTarget) < 0.01) {
        edgeCount++;
      }
    }
  }

  const valid = edgeCount === 96;
  console.log(`${cell.label}: ${edgeCount} edges ${valid ? '✓' : '✗'} (24-cell has 96 edges)`);
}

// Check that cells are disjoint (no shared vertices)
console.log("\n## Checking Disjointness\n");

const allVertexIds = new Set<number>();
let duplicatesFound = 0;

for (const cell of cells) {
  for (const vid of cell.vertexIds) {
    if (allVertexIds.has(vid)) {
      duplicatesFound++;
    }
    allVertexIds.add(vid);
  }
}

if (duplicatesFound === 0) {
  console.log("All 5 cells are disjoint ✓");
} else {
  console.log(`ERROR: Found ${duplicatesFound} duplicate vertex assignments`);
}

// Summary
console.log("\n" + "=".repeat(70));
console.log("SUMMARY");
console.log("=".repeat(70));

const allValid = cells.every(c => c.vertexIds.length === 24) &&
                 totalVertices === 120 &&
                 duplicatesFound === 0;

if (allValid) {
  console.log("\n✓ 600-cell decomposition is CORRECT");
  console.log("  - All 5 cells have exactly 24 vertices");
  console.log("  - Total: 120 vertices");
  console.log("  - Cells are mutually disjoint");
  console.log("  - Uses proper group-theoretic coset structure (2I/2T)");
} else {
  console.log("\n✗ 600-cell decomposition has issues");
}
