/**
 * Track A: Deriving the Moxness Matrix from Clifford Algebra
 *
 * Goal: Show that the Moxness matrix arises naturally from the
 * Clifford algebra projection Cl(3) → Cl⁺(3) × Cl⁺(3)
 *
 * Key insight: The 8D Clifford algebra Cl(3) decomposes as:
 * - Even part Cl⁺(3): {1, e₁₂, e₂₃, e₃₁} → first 4D (H4_L)
 * - Odd part Cl⁻(3): {e₁, e₂, e₃, e₁₂₃} → second 4D (H4_R)
 *
 * But the Moxness matrix mixes these with φ-scaling!
 */

const PHI = (1 + Math.sqrt(5)) / 2;
const INV_PHI = PHI - 1; // = 1/φ

console.log("=".repeat(70));
console.log("DERIVING THE MOXNESS MATRIX FROM CLIFFORD ALGEBRA");
console.log("=".repeat(70));

// =============================================================================
// 1. CLIFFORD BASIS ORDERING
// =============================================================================

console.log("\n## 1. Clifford Basis Ordering\n");

console.log(`
Standard Cl(3) basis ordering:
  Index 0: 1     (grade 0, scalar)
  Index 1: e₁    (grade 1, vector)
  Index 2: e₂    (grade 1, vector)
  Index 3: e₃    (grade 1, vector)
  Index 4: e₁₂   (grade 2, bivector)
  Index 5: e₂₃   (grade 2, bivector)
  Index 6: e₃₁   (grade 2, bivector)
  Index 7: e₁₂₃  (grade 3, trivector)

Even subalgebra Cl⁺(3) = {1, e₁₂, e₂₃, e₃₁} → indices {0, 4, 5, 6}
Odd subalgebra Cl⁻(3) = {e₁, e₂, e₃, e₁₂₃} → indices {1, 2, 3, 7}
`);

// =============================================================================
// 2. THE MOXNESS MATRIX STRUCTURE
// =============================================================================

console.log("## 2. Moxness Matrix Structure\n");

const a = 0.5;
const b = (PHI - 1) / 2;  // = 1/(2φ)
const c = PHI / 2;

console.log(`Moxness coefficients:`);
console.log(`  a = ${a.toFixed(6)}`);
console.log(`  b = ${b.toFixed(6)} = 1/(2φ)`);
console.log(`  c = ${c.toFixed(6)} = φ/2`);
console.log(`  Ratio: b:a:c = 1:φ:φ² (geometric progression)`);

// Build Moxness matrix
const U: number[][] = [
  [a, a, a, a, b, b, -b, -b],
  [a, a, -a, -a, b, -b, b, -b],
  [a, -a, a, -a, b, -b, -b, b],
  [a, -a, -a, a, b, b, -b, -b],
  [c, c, c, c, -a, -a, a, a],
  [c, c, -c, -c, -a, a, -a, a],
  [c, -c, c, -c, -a, a, a, -a],
  [c, -c, -c, c, -a, -a, a, a],
];

console.log("\nMoxness matrix (8×8):");
for (let i = 0; i < 8; i++) {
  console.log(`  [${U[i].map(x => x >= 0 ? ' ' + x.toFixed(3) : x.toFixed(3)).join(', ')}]`);
}

// =============================================================================
// 3. ANALYZING THE BLOCK STRUCTURE
// =============================================================================

console.log("\n## 3. Block Structure Analysis\n");

// Extract 4×4 blocks
const A = U.slice(0, 4).map(row => row.slice(0, 4)); // Upper-left
const B = U.slice(0, 4).map(row => row.slice(4, 8)); // Upper-right
const C = U.slice(4, 8).map(row => row.slice(0, 4)); // Lower-left
const D = U.slice(4, 8).map(row => row.slice(4, 8)); // Lower-right

console.log("Block A (H4_L ← even Cl⁺):");
for (const row of A) console.log(`  [${row.map(x => x.toFixed(3)).join(', ')}]`);

console.log("\nBlock B (H4_L ← odd Cl⁻):");
for (const row of B) console.log(`  [${row.map(x => x.toFixed(3)).join(', ')}]`);

console.log("\nBlock C (H4_R ← even Cl⁺):");
for (const row of C) console.log(`  [${row.map(x => x.toFixed(3)).join(', ')}]`);

console.log("\nBlock D (H4_R ← odd Cl⁻):");
for (const row of D) console.log(`  [${row.map(x => x.toFixed(3)).join(', ')}]`);

// =============================================================================
// 4. HADAMARD STRUCTURE
// =============================================================================

console.log("\n## 4. Hadamard Structure\n");

// The sign pattern looks like H₂ ⊗ H₂
// H₂ = [[1,1],[1,-1]]
// H₂ ⊗ H₂ = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]

const H2 = [[1, 1], [1, -1]];
const H4 = [
  [1, 1, 1, 1],
  [1, -1, 1, -1],
  [1, 1, -1, -1],
  [1, -1, -1, 1],
];

console.log("H₂ ⊗ H₂ =");
for (const row of H4) console.log(`  [${row.map(x => x >= 0 ? '+' : '-').join(' ')}]`);

console.log("\nBlock A sign pattern:");
for (const row of A) console.log(`  [${row.map(x => x >= 0 ? '+' : '-').join(' ')}]`);

console.log("\nBlock B sign pattern:");
for (const row of B) console.log(`  [${row.map(x => x >= 0 ? '+' : '-').join(' ')}]`);

// Check if A = a * H4
const isAHadamard = A.every((row, i) => row.every((val, j) => Math.abs(Math.abs(val) - a) < 0.001));
console.log(`\nBlock A = a × (sign pattern)? ${isAHadamard ? '✓' : '✗'}`);

// =============================================================================
// 5. THE COLUMN INTERPRETATION
// =============================================================================

console.log("\n## 5. Column Interpretation\n");

console.log(`
Consider what each column of U does:

Columns 0-3: Transform EVEN Cl⁺(3) = {1, e₁₂, e₂₃, e₃₁}
  - But wait, the Clifford ordering is {1, e₁, e₂, e₃, e₁₂, e₂₃, e₃₁, e₁₂₃}
  - Columns 0-3 correspond to indices 0,1,2,3 = {1, e₁, e₂, e₃}
  - This is NOT the even subalgebra!

Columns 4-7: Transform ODD Cl⁻(3) = {e₁, e₂, e₃, e₁₂₃}
  - Columns 4-7 correspond to indices 4,5,6,7 = {e₁₂, e₂₃, e₃₁, e₁₂₃}
  - This is MIXED even/odd!

**INSIGHT: The Moxness matrix does NOT directly align with Cl⁺/Cl⁻ decomposition!**

Instead, it appears to use a DIFFERENT basis reordering:
  - Columns 0-3: {1, e₁, e₂, e₃} (scalar + vectors)
  - Columns 4-7: {e₁₂, e₂₃, e₃₁, e₁₂₃} (bivectors + pseudoscalar)

This is a GRADE-based grouping:
  - Grades 0+1 → columns 0-3
  - Grades 2+3 → columns 4-7
`);

// =============================================================================
// 6. NULL SPACE AND GRADE STRUCTURE
// =============================================================================

console.log("## 6. Null Space and Grade Structure\n");

const nullSpace = [0, 0, 0, 0, 1, 1, 1, 1];

console.log(`Moxness null space: [${nullSpace.join(', ')}]`);
console.log(`
This says: columns 4-7 sum to zero.

In grade terms:
  - Columns 4-7 = {e₁₂, e₂₃, e₃₁, e₁₂₃}
  - Grade 2 components (bivectors) + Grade 3 (pseudoscalar) must balance

This is a CHIRALITY constraint in 4D!
The sum e₁₂ + e₂₃ + e₃₁ + e₁₂₃ represents a specific "handedness" balance.
`);

// Verify null space
const Uv = U.map(row => row.reduce((sum, val, j) => sum + val * nullSpace[j], 0));
console.log(`Verification: U × [0,0,0,0,1,1,1,1]ᵀ = [${Uv.map(x => x.toFixed(6)).join(', ')}]`);

// =============================================================================
// 7. DERIVING FROM FIRST PRINCIPLES
// =============================================================================

console.log("\n## 7. Attempting First-Principles Derivation\n");

console.log(`
To derive the Moxness matrix from Clifford algebra, we need:

1. A map from E8 roots (in Cl(3)) to H4 vertices (in Cl⁺(3) × Cl⁺(3))

2. The map should:
   - Preserve H4 symmetry
   - Scale by φ between the two H4 copies
   - Have the observed null space

Let's try to construct this:

Step 1: Represent E8 roots as Cl(3) elements
  - 112 roots of form (±1, ±1, 0, 0, 0, 0, 0, 0) permutations
  - 128 roots of form (±1/2)⁸ with even parity

Step 2: Project to Cl⁺(3) × Cl⁺(3) with φ-scaling
`);

// Generate a few E8 roots and project them
console.log("\n### Testing Projection on E8 Roots\n");

type Vector8D = number[];

function generateSomeE8Roots(): Vector8D[] {
  const roots: Vector8D[] = [];

  // Type I: First few
  roots.push([1, 1, 0, 0, 0, 0, 0, 0]);
  roots.push([1, -1, 0, 0, 0, 0, 0, 0]);
  roots.push([1, 0, 1, 0, 0, 0, 0, 0]);
  roots.push([1, 0, 0, 1, 0, 0, 0, 0]);

  // Type II: First few with even parity
  roots.push([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  roots.push([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5]);
  roots.push([0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]);
  roots.push([0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]);

  return roots;
}

const testRoots = generateSomeE8Roots();

console.log("E8 Root → H4_L × H4_R projection:");
for (const root of testRoots) {
  const H4L: number[] = [];
  const H4R: number[] = [];

  for (let i = 0; i < 4; i++) {
    H4L.push(U[i].reduce((sum, val, j) => sum + val * root[j], 0));
    H4R.push(U[i + 4].reduce((sum, val, j) => sum + val * root[j], 0));
  }

  const normL = Math.sqrt(H4L.reduce((sum, x) => sum + x * x, 0));
  const normR = Math.sqrt(H4R.reduce((sum, x) => sum + x * x, 0));

  console.log(`  ${JSON.stringify(root)}`);
  console.log(`    → H4_L: [${H4L.map(x => x.toFixed(4)).join(', ')}] (||·|| = ${normL.toFixed(4)})`);
  console.log(`    → H4_R: [${H4R.map(x => x.toFixed(4)).join(', ')}] (||·|| = ${normR.toFixed(4)})`);
}

// =============================================================================
// 8. THE REORDERING HYPOTHESIS
// =============================================================================

console.log("\n## 8. The Reordering Hypothesis\n");

console.log(`
**HYPOTHESIS:**

The Moxness matrix is NOT a direct Clifford projection, but rather:

1. A REORDERING of the Cl(3) basis to group by "grades 0+1" vs "grades 2+3"
2. A φ-SCALING between the reordered blocks
3. A HADAMARD-like mixing within each block

The φ-scaling arises because:
- H4 has icosahedral symmetry (involves φ)
- The projection must preserve this symmetry
- The ratio c/a = φ encodes the icosahedral structure

**THEOREM (Conjectured):**

The Moxness matrix is the UNIQUE (up to symmetry) linear projection
E8 → H4 × H4 that:
1. Preserves H4 symmetry in each factor
2. Scales by φ between factors
3. Has minimal rank deficiency (rank 7, not 6 or less)
`);

// =============================================================================
// 9. VERIFICATION: COMPARE TO STANDARD PROJECTION
// =============================================================================

console.log("\n## 9. Alternative: Direct Grade Projection\n");

// What if we just project to Cl⁺(3) (even part)?
// That would be: keep indices {0, 4, 5, 6}

console.log("Direct grade projection (keep even part {1, e₁₂, e₂₃, e₃₁}):");

const gradeProjection = [
  [1, 0, 0, 0, 0, 0, 0, 0],  // 1 → 1
  [0, 0, 0, 0, 1, 0, 0, 0],  // e₁₂ → e₁₂
  [0, 0, 0, 0, 0, 1, 0, 0],  // e₂₃ → e₂₃
  [0, 0, 0, 0, 0, 0, 1, 0],  // e₃₁ → e₃₁
];

console.log("This would give a 4×8 matrix that discards the odd part entirely.");
console.log("The Moxness matrix is MORE SUBTLE - it MIXES even and odd with φ-scaling.");

// =============================================================================
// CONCLUSION
// =============================================================================

console.log("\n" + "=".repeat(70));
console.log("CONCLUSION: MOXNESS MATRIX DERIVATION");
console.log("=".repeat(70));

console.log(`
**KEY FINDINGS:**

1. The Moxness matrix does NOT directly correspond to the Cl⁺(3)/Cl⁻(3) split.

2. Instead, it groups the Cl(3) basis as:
   - Columns 0-3: "Base" part (includes scalar + vectors)
   - Columns 4-7: "Dual" part (includes bivectors + pseudoscalar)

3. The φ-scaling (a, b=a/φ, c=aφ) encodes ICOSAHEDRAL GEOMETRY.
   This is necessary because H4 has icosahedral symmetry.

4. The null space [0,0,0,0,1,1,1,1] says the "dual part" must balance.
   This is a CHIRALITY or PARITY constraint.

5. The Moxness matrix appears to be a HYBRID:
   - Hadamard structure (sign pattern)
   - φ-scaling (icosahedral geometry)
   - Grade mixing (not pure Cl⁺ projection)

**REMAINING QUESTION:**

Can we derive the EXACT coefficients (a=1/2, b=(φ-1)/2, c=φ/2) from:
- Requirement of H4 symmetry preservation
- Requirement of φ-scaling between copies
- Requirement of minimal rank deficiency

This would complete the first-principles derivation.
`);
