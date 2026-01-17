/**
 * Track A: McKay Correspondence Investigation
 *
 * The McKay correspondence relates:
 * - Binary icosahedral group 2I (order 120) ↔ Affine E₈ Dynkin diagram
 * - The 240 E8 roots correspond to edges of the McKay graph
 * - The 120 vertices of 600-cell correspond to elements of 2I
 *
 * Question: Does the Moxness null space [0,0,0,0,1,1,1,1]ᵀ relate to
 * the imaginary root δ of affine E₈?
 */

console.log("=".repeat(70));
console.log("TRACK A: McKAY CORRESPONDENCE INVESTIGATION");
console.log("=".repeat(70));

// =============================================================================
// 1. THE E₈ CARTAN MATRIX
// =============================================================================

console.log("\n## 1. The E₈ Cartan Matrix\n");

// E8 Cartan matrix (8×8) - standard labeling
// Dynkin diagram: 1-3-4-5-6-7-8 with 2 attached to 4
const E8_CARTAN: number[][] = [
  [2, 0, -1, 0, 0, 0, 0, 0],   // α₁
  [0, 2, 0, -1, 0, 0, 0, 0],   // α₂
  [-1, 0, 2, -1, 0, 0, 0, 0],  // α₃
  [0, -1, -1, 2, -1, 0, 0, 0], // α₄
  [0, 0, 0, -1, 2, -1, 0, 0],  // α₅
  [0, 0, 0, 0, -1, 2, -1, 0],  // α₆
  [0, 0, 0, 0, 0, -1, 2, -1],  // α₇
  [0, 0, 0, 0, 0, 0, -1, 2],   // α₈
];

console.log("E₈ Cartan matrix (8×8):");
for (const row of E8_CARTAN) {
  console.log(`  [${row.map(x => x.toString().padStart(2)).join(', ')}]`);
}

// Compute determinant
function det(M: number[][]): number {
  const n = M.length;
  if (n === 1) return M[0][0];
  if (n === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];

  let d = 0;
  for (let j = 0; j < n; j++) {
    const minor: number[][] = [];
    for (let i = 1; i < n; i++) {
      minor.push([...M[i].slice(0, j), ...M[i].slice(j + 1)]);
    }
    d += (j % 2 === 0 ? 1 : -1) * M[0][j] * det(minor);
  }
  return d;
}

console.log(`\ndet(E₈ Cartan) = ${det(E8_CARTAN)}`);
console.log("(Should be 1 for E₈ - unimodular lattice)");

// =============================================================================
// 2. THE AFFINE E₈ CARTAN MATRIX
// =============================================================================

console.log("\n## 2. The Affine Ê₈ Cartan Matrix\n");

// Affine E8 adds node 0 connected to node 1
// The highest root θ = 2α₁ + 3α₂ + 4α₃ + 6α₄ + 5α₅ + 4α₆ + 3α₇ + 2α₈
// Affine root α₀ = -θ

// Affine Cartan matrix (9×9)
const AFFINE_E8_CARTAN: number[][] = [
  [2, -1, 0, 0, 0, 0, 0, 0, 0],   // α₀ (affine)
  [-1, 2, 0, -1, 0, 0, 0, 0, 0],  // α₁
  [0, 0, 2, 0, -1, 0, 0, 0, 0],   // α₂
  [0, -1, 0, 2, -1, 0, 0, 0, 0],  // α₃
  [0, 0, -1, -1, 2, -1, 0, 0, 0], // α₄
  [0, 0, 0, 0, -1, 2, -1, 0, 0],  // α₅
  [0, 0, 0, 0, 0, -1, 2, -1, 0],  // α₆
  [0, 0, 0, 0, 0, 0, -1, 2, -1],  // α₇
  [0, 0, 0, 0, 0, 0, 0, -1, 2],   // α₈
];

console.log("Affine Ê₈ Cartan matrix (9×9):");
for (const row of AFFINE_E8_CARTAN) {
  console.log(`  [${row.map(x => x.toString().padStart(2)).join(', ')}]`);
}

console.log(`\ndet(Affine Ê₈ Cartan) = ${det(AFFINE_E8_CARTAN)}`);
console.log("(Should be 0 for affine algebras - singular)");

// =============================================================================
// 3. NULL SPACE OF AFFINE E₈
// =============================================================================

console.log("\n## 3. Null Space of Affine Ê₈\n");

// The null space is spanned by the vector of marks (Coxeter labels)
// For E₈: marks = [1, 2, 3, 4, 6, 5, 4, 3, 2] for nodes [0, 1, 2, 3, 4, 5, 6, 7, 8]
// These are the coefficients of the imaginary root δ = Σ mᵢαᵢ

const AFFINE_E8_MARKS = [1, 2, 3, 4, 6, 5, 4, 3, 2]; // Standard ordering

console.log("Coxeter marks (null space generator):");
console.log(`  δ = ${AFFINE_E8_MARKS.join('·α₀ + ').replace(/·α₀/g, (_, i) => `α${i}`)}`);
console.log(`  = [${AFFINE_E8_MARKS.join(', ')}]`);

// Verify it's in the null space
const product: number[] = [];
for (let i = 0; i < 9; i++) {
  let sum = 0;
  for (let j = 0; j < 9; j++) {
    sum += AFFINE_E8_CARTAN[i][j] * AFFINE_E8_MARKS[j];
  }
  product.push(sum);
}

console.log(`\nVerification: Cartan × marks = [${product.join(', ')}]`);
console.log(`Is null: ${product.every(x => x === 0) ? '✓ YES' : '✗ NO'}`);

// =============================================================================
// 4. COMPARISON WITH MOXNESS NULL SPACE
// =============================================================================

console.log("\n## 4. Comparison with Moxness Null Space\n");

const MOXNESS_NULL = [0, 0, 0, 0, 1, 1, 1, 1];

console.log("Moxness matrix null space:");
console.log(`  v = [${MOXNESS_NULL.join(', ')}]`);
console.log(`  Dimension: 8 (lives in ℝ⁸)`);

console.log("\nAffine E₈ null space:");
console.log(`  δ = [${AFFINE_E8_MARKS.join(', ')}]`);
console.log(`  Dimension: 9 (lives in ℝ⁹)`);

console.log("\n**OBSERVATION: Dimensional Mismatch**");
console.log("The Moxness null space is in ℝ⁸ while the affine null space is in ℝ⁹.");
console.log("Direct comparison is not meaningful without a projection.");

// =============================================================================
// 5. PROJECTING AFFINE NULL SPACE TO E₈
// =============================================================================

console.log("\n## 5. Projecting Affine Null Space to E₈\n");

// If we remove the affine node (α₀), we get the marks for E₈ simple roots
const E8_MARKS = AFFINE_E8_MARKS.slice(1); // [2, 3, 4, 6, 5, 4, 3, 2]

console.log("E₈ marks (affine marks with α₀ removed):");
console.log(`  [${E8_MARKS.join(', ')}]`);

console.log("\nComparison:");
console.log(`  E₈ marks:      [${E8_MARKS.join(', ')}]`);
console.log(`  Moxness null:  [${MOXNESS_NULL.join(', ')}]`);

// Check if proportional
const ratio = E8_MARKS[0] / MOXNESS_NULL[0]; // undefined if MOXNESS_NULL[0] = 0
console.log(`\n**OBSERVATION: NOT Proportional**`);
console.log("The Moxness null space [0,0,0,0,1,1,1,1] has zeros in positions 0-3,");
console.log("while the E₈ marks [2,3,4,6,5,4,3,2] are all positive.");

// =============================================================================
// 6. ALTERNATIVE INTERPRETATION
// =============================================================================

console.log("\n## 6. Alternative Interpretation: Column Grouping\n");

console.log("The Moxness matrix has structure:");
console.log("  - Columns 0-3: 'E₈ half' (rational-like structure)");
console.log("  - Columns 4-7: 'H₄ half' (φ-dependent structure)");
console.log("");
console.log("The null space [0,0,0,0,1,1,1,1]ᵀ says:");
console.log("  - Sum of columns 4-7 is zero");
console.log("  - Columns 0-3 are unconstrained");
console.log("");
console.log("This suggests the null space encodes the 'folding' constraint:");
console.log("  The φ-dependent half (columns 4-7) must sum to zero.");

// =============================================================================
// 7. D₄ TRIALITY CONNECTION
// =============================================================================

console.log("\n## 7. D₄ Triality Connection\n");

// The E8 Dynkin diagram has a D4 subdiagram at node 4
// D4 has triality symmetry (S3)
// The three "arms" of D4 are nodes 2, 3, 5

console.log("E₈ Dynkin diagram:");
console.log("        α₂");
console.log("         |");
console.log("  α₁-α₃-α₄-α₅-α₆-α₇-α₈");
console.log("");
console.log("D₄ subdiagram (nodes 2,3,4,5):");
console.log("    α₂");
console.log("     |");
console.log("  α₃-α₄-α₅");
console.log("");
console.log("The three 'arms' α₂, α₃, α₅ are permuted by triality.");

// Check if Moxness null space respects some D4 structure
console.log("\nMoxness null space grouped by potential D₄ structure:");
console.log("  Positions [0,1,2,3]: [0, 0, 0, 0] - all zero");
console.log("  Positions [4,5,6,7]: [1, 1, 1, 1] - all equal");
console.log("");
console.log("The symmetry is compatible with D₄ triality if we view");
console.log("positions 4,5,6,7 as forming a 'tetrahedral' structure.");

// =============================================================================
// 8. QUATERNIONIC INTERPRETATION
// =============================================================================

console.log("\n## 8. Quaternionic Interpretation\n");

console.log("The H₄ group is closely related to quaternions.");
console.log("The 24-cell vertices are the 24 unit Hurwitz quaternions.");
console.log("");
console.log("The Moxness null space [0,0,0,0,1,1,1,1] can be interpreted as:");
console.log("  - First 4 components (scalar + vector part 1): unconstrained");
console.log("  - Last 4 components (vector part 2): sum to zero");
console.log("");
console.log("In quaternionic terms, this might represent:");
console.log("  - A constraint on the 'imaginary' quaternion directions");
console.log("  - The 'real' part (positions 0-3) is free");
console.log("  - The 'imaginary' balance (positions 4-7) must be preserved");

// =============================================================================
// CONCLUSION
// =============================================================================

console.log("\n" + "=".repeat(70));
console.log("CONCLUSION");
console.log("=".repeat(70));

console.log(`
**FINDING: No Direct McKay Correspondence Connection**

The Moxness null space [0,0,0,0,1,1,1,1]ᵀ does NOT directly correspond to
the imaginary root δ of affine E₈:
- δ = [1,2,3,4,6,5,4,3,2] (all positive, different magnitudes)
- Moxness null = [0,0,0,0,1,1,1,1] (half zeros, half ones)

**ALTERNATIVE INTERPRETATION:**

The Moxness null space encodes a structural property of the folding:
1. It splits ℝ⁸ into two halves (0-3 and 4-7)
2. The 'φ-dependent half' (columns 4-7) must sum to zero
3. This constraint arises from the φ-scaling relationship c = φa

**CONNECTION TO D₄ TRIALITY:**

The structure [0,0,0,0,1,1,1,1] is compatible with viewing:
- Positions 0-3 as the 'base' of E₈
- Positions 4-7 as the 'triality-invariant' part

This warrants further investigation into how the D₄ subgroup of E₈
relates to the H₄ folding.

**NEXT STEP:** Investigate the Clifford algebra connection (Dechant)
to understand WHY the null space has this specific structure.
`);
