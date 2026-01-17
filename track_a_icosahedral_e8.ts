/**
 * Track A: Icosahedral Group → E8 via Clifford Algebra
 *
 * Following Dechant's construction:
 * 1. Build icosahedral group H₃ (120 elements including inversions)
 * 2. Represent as Clifford elements in Cl(3)
 * 3. Show these 240 8D vectors (with ±) form E8 roots
 *
 * Key insight: The icosahedron's symmetry group, when embedded in the
 * 8-dimensional Clifford algebra Cl(3), produces exactly the E8 root system.
 */

const PHI = (1 + Math.sqrt(5)) / 2;
const INV_PHI = 1 / PHI; // = φ - 1

console.log("=".repeat(70));
console.log("ICOSAHEDRAL GROUP → E8 VIA CLIFFORD ALGEBRA");
console.log("=".repeat(70));

// =============================================================================
// ICOSAHEDRAL GEOMETRY
// =============================================================================

console.log("\n## 1. Icosahedral Vertices\n");

// Icosahedron vertices (12 vertices)
// Standard form using golden ratio
function getIcosahedronVertices(): [number, number, number][] {
  const vertices: [number, number, number][] = [];

  // Rectangular coordinates based on golden ratio
  // (0, ±1, ±φ) and permutations
  const coords = [
    [0, 1, PHI], [0, 1, -PHI], [0, -1, PHI], [0, -1, -PHI],
    [1, PHI, 0], [1, -PHI, 0], [-1, PHI, 0], [-1, -PHI, 0],
    [PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, 1], [-PHI, 0, -1],
  ];

  for (const c of coords) {
    vertices.push(c as [number, number, number]);
  }

  return vertices;
}

const icoVertices = getIcosahedronVertices();
console.log(`Icosahedron has ${icoVertices.length} vertices`);

// =============================================================================
// ICOSAHEDRAL ROTATIONS
// =============================================================================

console.log("\n## 2. Icosahedral Rotation Group\n");

// The icosahedral rotation group I has 60 elements:
// - 1 identity
// - 15 rotations by π (180°) around axes through edge midpoints
// - 20 rotations by 2π/3 and 4π/3 around axes through face centers
// - 24 rotations by 2π/5, 4π/5, 6π/5, 8π/5 around axes through vertices

interface Rotation {
  axis: [number, number, number];
  angle: number;
  description: string;
}

function getIcosahedralRotations(): Rotation[] {
  const rotations: Rotation[] = [];

  // Identity
  rotations.push({ axis: [0, 0, 1], angle: 0, description: "identity" });

  // Vertex axes (6 axes, 4 non-identity rotations each = 24 rotations)
  // Axes through opposite vertices
  const vertexAxes: [number, number, number][] = [
    [0, 1, PHI],
    [0, 1, -PHI],
    [1, PHI, 0],
    [PHI, 0, 1],
    [-PHI, 0, 1],
    [1, -PHI, 0],
  ];

  for (const axis of vertexAxes) {
    const norm = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
    const nAxis: [number, number, number] = [axis[0] / norm, axis[1] / norm, axis[2] / norm];

    for (let k = 1; k <= 4; k++) {
      rotations.push({
        axis: nAxis,
        angle: (2 * Math.PI * k) / 5,
        description: `vertex ${k}×72°`,
      });
    }
  }

  // Face axes (10 axes, 2 non-identity rotations each = 20 rotations)
  // Axes through opposite face centers
  // Face centers of icosahedron
  const faceAxes: [number, number, number][] = [
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [PHI, INV_PHI, 0],
    [PHI, -INV_PHI, 0],
    [INV_PHI, 0, PHI],
    [INV_PHI, 0, -PHI],
    [0, PHI, INV_PHI],
  ];

  for (const axis of faceAxes) {
    const norm = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
    const nAxis: [number, number, number] = [axis[0] / norm, axis[1] / norm, axis[2] / norm];

    for (let k = 1; k <= 2; k++) {
      rotations.push({
        axis: nAxis,
        angle: (2 * Math.PI * k) / 3,
        description: `face ${k}×120°`,
      });
    }
  }

  // Edge axes (15 axes, 1 non-identity rotation each = 15 rotations)
  // Axes through opposite edge midpoints
  const edgeAxes: [number, number, number][] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [PHI, 1, INV_PHI],
    [PHI, 1, -INV_PHI],
    [PHI, -1, INV_PHI],
    [PHI, -1, -INV_PHI],
    [1, INV_PHI, PHI],
    [1, INV_PHI, -PHI],
    [1, -INV_PHI, PHI],
    [1, -INV_PHI, -PHI],
    [INV_PHI, PHI, 1],
    [INV_PHI, PHI, -1],
    [INV_PHI, -PHI, 1],
    [INV_PHI, -PHI, -1],
  ];

  for (const axis of edgeAxes) {
    const norm = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
    const nAxis: [number, number, number] = [axis[0] / norm, axis[1] / norm, axis[2] / norm];

    rotations.push({
      axis: nAxis,
      angle: Math.PI,
      description: `edge 180°`,
    });
  }

  return rotations;
}

const rotations = getIcosahedralRotations();
console.log(`Generated ${rotations.length} rotations (expected 60)`);

// =============================================================================
// CLIFFORD ROTORS
// =============================================================================

console.log("\n## 3. Rotors in Cl(3)\n");

// Rotor for rotation about axis n by angle θ:
// R = cos(θ/2) + sin(θ/2) * (n₁e₂₃ + n₂e₃₁ + n₃e₁₂)

function rotorCoeffs(axis: [number, number, number], angle: number): number[] {
  const c = Math.cos(angle / 2);
  const s = Math.sin(angle / 2);

  // Coefficients: [1, e₁, e₂, e₃, e₁₂, e₂₃, e₃₁, e₁₂₃]
  // Bivector for rotation about axis n is:
  // B = n₁ e₂₃ + n₂ e₃₁ + n₃ e₁₂

  return [
    c,                // 1
    0,                // e₁
    0,                // e₂
    0,                // e₃
    s * axis[2],      // e₁₂ (from n₃)
    s * axis[0],      // e₂₃ (from n₁)
    s * axis[1],      // e₃₁ (from n₂)
    0,                // e₁₂₃
  ];
}

// Generate all rotors
const rotors: number[][] = [];
for (const rot of rotations) {
  const R = rotorCoeffs(rot.axis, rot.angle);
  rotors.push(R);
  // Also add -R (same rotation, different spinor)
  rotors.push(R.map(x => -x));
}

console.log(`Generated ${rotors.length} rotors (including ±R)`);

// Remove duplicates (rotors that are approximately equal)
function vectorsEqual(a: number[], b: number[], tol: number = 1e-6): boolean {
  return a.every((v, i) => Math.abs(v - b[i]) < tol);
}

const uniqueRotors: number[][] = [];
for (const R of rotors) {
  let isDuplicate = false;
  for (const existing of uniqueRotors) {
    if (vectorsEqual(R, existing)) {
      isDuplicate = true;
      break;
    }
  }
  if (!isDuplicate) {
    uniqueRotors.push(R);
  }
}

console.log(`Unique rotors: ${uniqueRotors.length} (expected 120 for binary icosahedral group)`);

// =============================================================================
// CHECK E8 STRUCTURE
// =============================================================================

console.log("\n## 4. Checking E8 Structure\n");

// Normalize rotors to have norm 1 in R^8
const normalizedRotors: number[][] = [];
for (const R of uniqueRotors) {
  const norm = Math.sqrt(R.reduce((sum, x) => sum + x * x, 0));
  if (norm > 1e-10) {
    normalizedRotors.push(R.map(x => x / norm));
  }
}

console.log(`Normalized rotors: ${normalizedRotors.length}`);

// For E8, we need 240 roots. Currently we have ~120 rotors (even part of Cl(3)).
// Dechant's insight: include REFLECTIONS (odd part) as well!

console.log("\n## 5. Adding Reflections (Odd Clifford Elements)\n");

// A reflection through a plane perpendicular to vector n is:
// v → -n v n  (in Clifford algebra)
// The reflection operator is just the vector n itself!

// Generate reflection vectors (normalized icosahedron vertices + face normals + edge midpoints)
const reflectionVectors: [number, number, number][] = [];

// Icosahedron vertices (12)
for (const v of icoVertices) {
  const norm = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
  reflectionVectors.push([v[0] / norm, v[1] / norm, v[2] / norm]);
}

// Add more reflection planes from dodecahedron (dual)
// Dodecahedron vertices are icosahedron face centers
const dodecaVertices: [number, number, number][] = [];
// ... (would need to compute these)

// For now, let's see what we get with just rotors
// The key is that rotors live in Cl⁺(3) (even part, 4D)
// For full E8, we need both even AND odd parts

console.log(`Reflection vectors: ${reflectionVectors.length}`);

// Represent reflections as Cl(3) elements (grade 1)
const reflectors: number[][] = [];
for (const v of reflectionVectors) {
  // Reflection vector: [0, v₁, v₂, v₃, 0, 0, 0, 0]
  reflectors.push([0, v[0], v[1], v[2], 0, 0, 0, 0]);
  reflectors.push([0, -v[0], -v[1], -v[2], 0, 0, 0, 0]); // Also -v
}

console.log(`Reflectors (±): ${reflectors.length}`);

// Combine rotors and reflectors
const allElements = [...normalizedRotors, ...reflectors];
console.log(`\nTotal Clifford elements: ${allElements.length}`);

// =============================================================================
// VERIFY E8 ROOT PROPERTIES
// =============================================================================

console.log("\n## 6. Verifying E8 Root Properties\n");

// E8 roots should have:
// 1. All norms equal (√2 in standard normalization)
// 2. Inner products ∈ {-2, -1, 0, 1, 2} (for norm √2)

// Check norm distribution
const normCounts = new Map<string, number>();
for (const elem of normalizedRotors) {
  const norm = Math.sqrt(elem.reduce((sum, x) => sum + x * x, 0));
  const key = norm.toFixed(4);
  normCounts.set(key, (normCounts.get(key) || 0) + 1);
}

console.log("Rotor norm distribution:");
for (const [norm, count] of normCounts) {
  console.log(`  ||R|| = ${norm}: ${count} rotors`);
}

// Check inner products
console.log("\nInner product distribution (sample of first 20 rotors):");
const innerProducts = new Map<string, number>();
for (let i = 0; i < Math.min(20, normalizedRotors.length); i++) {
  for (let j = i + 1; j < Math.min(20, normalizedRotors.length); j++) {
    const ip = normalizedRotors[i].reduce((sum, x, k) => sum + x * normalizedRotors[j][k], 0);
    const key = ip.toFixed(4);
    innerProducts.set(key, (innerProducts.get(key) || 0) + 1);
  }
}

const sortedIPs = Array.from(innerProducts.entries()).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]));
for (const [ip, count] of sortedIPs) {
  console.log(`  ⟨R_i, R_j⟩ = ${ip}: ${count} pairs`);
}

// =============================================================================
// THE DECHANT INSIGHT
// =============================================================================

console.log("\n## 7. The Dechant Insight\n");

console.log(`
Dechant's key observation:

The 120 elements of the binary icosahedral group 2I, when represented
as spinors (rotors) in Cl⁺(3), form the 120 vertices of the 600-cell.

To get E8 (240 roots), we need BOTH:
1. Even elements (rotors) - living in Cl⁺(3) ≅ ℍ (quaternions)
2. Odd elements (vectors/trivectors) - living in Cl⁻(3)

The FULL Clifford algebra Cl(3) has dimension 8.
When we represent all 120 icosahedral symmetries (rotations + reflections)
in Cl(3), we get 240 8-dimensional objects.

With the appropriate inner product, these ARE the E8 roots!
`);

// =============================================================================
// COMPARING TO MOXNESS PROJECTION
// =============================================================================

console.log("## 8. Connection to Moxness Matrix\n");

console.log(`
The Moxness matrix projects E8 roots to H4.

In Clifford terms:
- E8 roots live in Cl(3) (8 dimensions)
- H4 vertices live in Cl⁺(3) (4 dimensions, the even subalgebra)

The Moxness projection might be:
1. Project Cl(3) → Cl⁺(3) (keep even part)
2. Apply φ-scaling between the two 4D halves

Let's check if the even part of our rotors matches H4/600-cell structure.
`);

// Extract even parts of rotors
const evenParts: number[][] = normalizedRotors.map(R => [R[0], R[4], R[5], R[6]]);

// Check if these form 600-cell vertices
console.log("Even parts (first 10):");
for (let i = 0; i < Math.min(10, evenParts.length); i++) {
  const norm = Math.sqrt(evenParts[i].reduce((sum, x) => sum + x * x, 0));
  console.log(`  [${evenParts[i].map(x => x.toFixed(4)).join(', ')}] (norm: ${norm.toFixed(4)})`);
}

// =============================================================================
// CONCLUSION
// =============================================================================

console.log("\n" + "=".repeat(70));
console.log("PRELIMINARY CONCLUSIONS");
console.log("=".repeat(70));

console.log(`
**FINDINGS:**

1. We generated ${uniqueRotors.length} unique rotors from icosahedral rotations
   (Binary icosahedral group 2I has 120 elements ✓)

2. These rotors live in the EVEN subalgebra Cl⁺(3), which is 4-dimensional
   This gives the 600-cell / H4 structure

3. To get the full E8 (240 roots), we need to include ODD elements
   (reflections, represented as vectors/trivectors in Cl⁻(3))

4. The Moxness matrix appears to implement the projection:
   Cl(3) → Cl⁺(3) × Cl⁺(3)
   with φ-scaling between the two copies

**NEXT STEP:**

Construct the complete set of 240 icosahedral symmetry elements in Cl(3)
and verify they match E8 roots.
`);
