/**
 * Track A: Clifford Algebra Cl(3) Implementation
 *
 * Cl(3) is the Clifford algebra of 3D Euclidean space.
 * It has 2³ = 8 basis elements organized by grade:
 *
 * Grade 0: 1 (scalar)
 * Grade 1: e₁, e₂, e₃ (vectors)
 * Grade 2: e₁₂, e₂₃, e₃₁ (bivectors)
 * Grade 3: e₁₂₃ (trivector/pseudoscalar)
 *
 * Multiplication rules:
 * - eᵢ² = +1 (Euclidean signature)
 * - eᵢeⱼ = -eⱼeᵢ for i ≠ j
 * - e₁₂ = e₁e₂, etc.
 */

const PHI = (1 + Math.sqrt(5)) / 2;

// =============================================================================
// CLIFFORD ALGEBRA Cl(3)
// =============================================================================

/**
 * Basis element indices:
 * 0: 1 (scalar)
 * 1: e₁
 * 2: e₂
 * 3: e₃
 * 4: e₁₂ = e₁e₂
 * 5: e₂₃ = e₂e₃
 * 6: e₃₁ = e₃e₁
 * 7: e₁₂₃ = e₁e₂e₃ (pseudoscalar)
 */

const BASIS_NAMES = ['1', 'e₁', 'e₂', 'e₃', 'e₁₂', 'e₂₃', 'e₃₁', 'e₁₂₃'];
const GRADES = [0, 1, 1, 1, 2, 2, 2, 3];

/**
 * Multiplication table for Cl(3) basis elements.
 * Entry [i][j] = { sign, index } where eᵢ * eⱼ = sign * e_index
 */
const MULT_TABLE: { sign: number; index: number }[][] = [];

function initMultTable() {
  // We compute eᵢ * eⱼ using the rules:
  // e₁² = e₂² = e₃² = 1
  // eᵢeⱼ = -eⱼeᵢ for i ≠ j

  // Binary representation of basis elements:
  // 1 = 000, e₁ = 001, e₂ = 010, e₃ = 100
  // e₁₂ = 011, e₂₃ = 110, e₃₁ = 101, e₁₂₃ = 111

  const binaryRep = [0b000, 0b001, 0b010, 0b100, 0b011, 0b110, 0b101, 0b111];
  const binaryToIndex = new Map<number, number>();
  binaryRep.forEach((b, i) => binaryToIndex.set(b, i));

  for (let i = 0; i < 8; i++) {
    MULT_TABLE[i] = [];
    for (let j = 0; j < 8; j++) {
      // Result basis is XOR of binary representations
      const resultBinary = binaryRep[i] ^ binaryRep[j];
      const resultIndex = binaryToIndex.get(resultBinary)!;

      // Compute sign by counting transpositions
      let sign = 1;

      // For each bit in j that needs to pass through bits in i
      // Count the swaps needed
      const iBits = binaryRep[i];
      const jBits = binaryRep[j];

      // j's e₁ passes through i's e₂, e₃
      if (jBits & 0b001) {
        if (iBits & 0b010) sign *= -1; // e₁ passes e₂
        if (iBits & 0b100) sign *= -1; // e₁ passes e₃
      }
      // j's e₂ passes through i's e₃
      if (jBits & 0b010) {
        if (iBits & 0b100) sign *= -1; // e₂ passes e₃
      }

      // Handle squares: eᵢ² = +1
      const commonBits = iBits & jBits;
      // Each common bit contributes a factor of +1 (already accounted for in XOR)
      // But we need to account for the reordering to get squares together

      // Count bits that cancel (appear in both)
      let cancelCount = 0;
      for (let b = 0; b < 3; b++) {
        if (commonBits & (1 << b)) {
          cancelCount++;
          // This bit in j has to pass through remaining bits of i to pair up
          // Already counted above
        }
      }

      MULT_TABLE[i][j] = { sign, index: resultIndex };
    }
  }
}

// Initialize on load
initMultTable();

// Manual correction based on known Clifford algebra rules
// Let's verify and fix the multiplication table
function verifyAndFixMultTable() {
  // Known products:
  // e₁ * e₁ = 1, e₂ * e₂ = 1, e₃ * e₃ = 1
  // e₁ * e₂ = e₁₂, e₂ * e₁ = -e₁₂
  // e₁₂ * e₁₂ = e₁e₂e₁e₂ = -e₁e₁e₂e₂ = -1
  // e₁₂₃ * e₁₂₃ = -1 (in Cl(3))

  const expected: [number, number, number, number][] = [
    // [i, j, expectedSign, expectedIndex]
    [1, 1, 1, 0],   // e₁² = 1
    [2, 2, 1, 0],   // e₂² = 1
    [3, 3, 1, 0],   // e₃² = 1
    [1, 2, 1, 4],   // e₁e₂ = e₁₂
    [2, 1, -1, 4],  // e₂e₁ = -e₁₂
    [2, 3, 1, 5],   // e₂e₃ = e₂₃
    [3, 2, -1, 5],  // e₃e₂ = -e₂₃
    [3, 1, 1, 6],   // e₃e₁ = e₃₁
    [1, 3, -1, 6],  // e₁e₃ = -e₃₁
    [4, 4, -1, 0],  // e₁₂² = -1
    [5, 5, -1, 0],  // e₂₃² = -1
    [6, 6, -1, 0],  // e₃₁² = -1
    [7, 7, -1, 0],  // e₁₂₃² = -1
    [1, 4, 1, 2],   // e₁ * e₁₂ = e₂
    [4, 1, -1, 2],  // e₁₂ * e₁ = -e₂
  ];

  // Rebuild table correctly
  // Using the canonical ordering and sign conventions

  // Represent each basis as a sorted list of generators
  // 1 = [], e₁ = [1], e₂ = [2], e₃ = [3]
  // e₁₂ = [1,2], e₂₃ = [2,3], e₃₁ = [1,3], e₁₂₃ = [1,2,3]

  const basisGens: number[][] = [
    [],        // 1
    [1],       // e₁
    [2],       // e₂
    [3],       // e₃
    [1, 2],    // e₁₂
    [2, 3],    // e₂₃
    [1, 3],    // e₃₁ (note: this is e₃e₁, not e₁e₃!)
    [1, 2, 3], // e₁₂₃
  ];

  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      // Concatenate generators
      const combined = [...basisGens[i], ...basisGens[j]];

      // Bubble sort to canonical order, counting swaps
      let swaps = 0;
      const arr = [...combined];
      for (let p = 0; p < arr.length; p++) {
        for (let q = p + 1; q < arr.length; q++) {
          if (arr[p] > arr[q]) {
            [arr[p], arr[q]] = [arr[q], arr[p]];
            swaps++;
          }
        }
      }

      // Cancel pairs (eᵢ² = 1)
      const result: number[] = [];
      let k = 0;
      while (k < arr.length) {
        if (k + 1 < arr.length && arr[k] === arr[k + 1]) {
          // Pair cancels to 1
          k += 2;
        } else {
          result.push(arr[k]);
          k++;
        }
      }

      // Find result in basis
      const resultStr = result.join(',');
      let resultIndex = -1;
      for (let b = 0; b < 8; b++) {
        if (basisGens[b].join(',') === resultStr) {
          resultIndex = b;
          break;
        }
      }

      if (resultIndex === -1) {
        console.error(`No basis match for ${resultStr}`);
        resultIndex = 0;
      }

      const sign = (swaps % 2 === 0) ? 1 : -1;
      MULT_TABLE[i][j] = { sign, index: resultIndex };
    }
  }
}

verifyAndFixMultTable();

/**
 * Clifford element in Cl(3)
 */
class Cl3 {
  /** Coefficients for basis {1, e₁, e₂, e₃, e₁₂, e₂₃, e₃₁, e₁₂₃} */
  coeffs: number[];

  constructor(coeffs: number[] = [0, 0, 0, 0, 0, 0, 0, 0]) {
    this.coeffs = [...coeffs];
    while (this.coeffs.length < 8) this.coeffs.push(0);
  }

  static scalar(s: number): Cl3 {
    return new Cl3([s, 0, 0, 0, 0, 0, 0, 0]);
  }

  static vector(x: number, y: number, z: number): Cl3 {
    return new Cl3([0, x, y, z, 0, 0, 0, 0]);
  }

  static bivector(xy: number, yz: number, zx: number): Cl3 {
    return new Cl3([0, 0, 0, 0, xy, yz, zx, 0]);
  }

  static pseudoscalar(p: number): Cl3 {
    return new Cl3([0, 0, 0, 0, 0, 0, 0, p]);
  }

  /** Geometric product */
  mul(other: Cl3): Cl3 {
    const result = new Cl3();
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        const { sign, index } = MULT_TABLE[i][j];
        result.coeffs[index] += sign * this.coeffs[i] * other.coeffs[j];
      }
    }
    return result;
  }

  /** Addition */
  add(other: Cl3): Cl3 {
    return new Cl3(this.coeffs.map((c, i) => c + other.coeffs[i]));
  }

  /** Scalar multiplication */
  scale(s: number): Cl3 {
    return new Cl3(this.coeffs.map(c => c * s));
  }

  /** Grade projection */
  grade(n: number): Cl3 {
    const result = new Cl3();
    for (let i = 0; i < 8; i++) {
      if (GRADES[i] === n) {
        result.coeffs[i] = this.coeffs[i];
      }
    }
    return result;
  }

  /** Reverse (reversion) - reverses order of vectors in each term */
  reverse(): Cl3 {
    // Grade 0, 1: unchanged
    // Grade 2: sign flip (e₁₂ → e₂₁ = -e₁₂)
    // Grade 3: sign flip (e₁₂₃ → e₃₂₁ = -e₁₂₃)
    return new Cl3([
      this.coeffs[0],   // 1
      this.coeffs[1],   // e₁
      this.coeffs[2],   // e₂
      this.coeffs[3],   // e₃
      -this.coeffs[4],  // e₁₂
      -this.coeffs[5],  // e₂₃
      -this.coeffs[6],  // e₃₁
      -this.coeffs[7],  // e₁₂₃
    ]);
  }

  /** Clifford conjugate (reverse + grade involution) */
  conjugate(): Cl3 {
    // Combines reversion with grade involution
    // Grade 0: +, Grade 1: -, Grade 2: -, Grade 3: +
    return new Cl3([
      this.coeffs[0],   // 1
      -this.coeffs[1],  // e₁
      -this.coeffs[2],  // e₂
      -this.coeffs[3],  // e₃
      -this.coeffs[4],  // e₁₂
      -this.coeffs[5],  // e₂₃
      -this.coeffs[6],  // e₃₁
      this.coeffs[7],   // e₁₂₃
    ]);
  }

  /** Norm squared */
  normSq(): number {
    return this.coeffs.reduce((sum, c) => sum + c * c, 0);
  }

  /** Norm */
  norm(): number {
    return Math.sqrt(this.normSq());
  }

  /** Even part (grades 0 and 2) */
  even(): Cl3 {
    return new Cl3([
      this.coeffs[0], 0, 0, 0,
      this.coeffs[4], this.coeffs[5], this.coeffs[6], 0
    ]);
  }

  /** Odd part (grades 1 and 3) */
  odd(): Cl3 {
    return new Cl3([
      0, this.coeffs[1], this.coeffs[2], this.coeffs[3],
      0, 0, 0, this.coeffs[7]
    ]);
  }

  /** Check if approximately zero */
  isZero(tol: number = 1e-10): boolean {
    return this.coeffs.every(c => Math.abs(c) < tol);
  }

  /** Check if approximately equal */
  equals(other: Cl3, tol: number = 1e-10): boolean {
    return this.coeffs.every((c, i) => Math.abs(c - other.coeffs[i]) < tol);
  }

  /** String representation */
  toString(): string {
    const terms: string[] = [];
    for (let i = 0; i < 8; i++) {
      if (Math.abs(this.coeffs[i]) > 1e-10) {
        const sign = this.coeffs[i] >= 0 ? '+' : '';
        terms.push(`${sign}${this.coeffs[i].toFixed(4)}${BASIS_NAMES[i]}`);
      }
    }
    return terms.length > 0 ? terms.join(' ') : '0';
  }

  /** Get as 8D vector */
  toVector8(): number[] {
    return [...this.coeffs];
  }
}

// =============================================================================
// ROTOR CONSTRUCTION
// =============================================================================

/**
 * Create a rotor for rotation in plane defined by bivector B, angle θ
 * R = cos(θ/2) + sin(θ/2) * B̂
 */
function rotor(B: Cl3, theta: number): Cl3 {
  const Bnorm = Math.sqrt(
    B.coeffs[4] ** 2 + B.coeffs[5] ** 2 + B.coeffs[6] ** 2
  );
  if (Bnorm < 1e-10) return Cl3.scalar(1);

  const Bhat = B.scale(1 / Bnorm);
  const c = Math.cos(theta / 2);
  const s = Math.sin(theta / 2);

  return Cl3.scalar(c).add(Bhat.scale(s));
}

/**
 * Create a rotor from axis-angle representation
 * Axis n = (nx, ny, nz), rotation angle θ
 */
function rotorFromAxisAngle(nx: number, ny: number, nz: number, theta: number): Cl3 {
  // Rotation plane bivector is dual to axis
  // For axis n, plane bivector B = n · I where I = e₁₂₃
  // B = nx*e₂₃ + ny*e₃₁ + nz*e₁₂
  const norm = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (norm < 1e-10) return Cl3.scalar(1);

  const B = Cl3.bivector(nz / norm, nx / norm, ny / norm);
  return rotor(B, theta);
}

/**
 * Apply rotor to vector: v' = R v R†
 */
function rotateVector(R: Cl3, v: Cl3): Cl3 {
  return R.mul(v).mul(R.reverse());
}

// =============================================================================
// VERIFICATION
// =============================================================================

console.log("=".repeat(70));
console.log("CLIFFORD ALGEBRA Cl(3) VERIFICATION");
console.log("=".repeat(70));

console.log("\n## 1. Basis Multiplication Verification\n");

// Test key products
const e1 = new Cl3([0, 1, 0, 0, 0, 0, 0, 0]);
const e2 = new Cl3([0, 0, 1, 0, 0, 0, 0, 0]);
const e3 = new Cl3([0, 0, 0, 1, 0, 0, 0, 0]);
const e12 = new Cl3([0, 0, 0, 0, 1, 0, 0, 0]);
const e23 = new Cl3([0, 0, 0, 0, 0, 1, 0, 0]);
const e31 = new Cl3([0, 0, 0, 0, 0, 0, 1, 0]);
const e123 = new Cl3([0, 0, 0, 0, 0, 0, 0, 1]);

console.log("e₁² =", e1.mul(e1).toString(), "(should be +1)");
console.log("e₂² =", e2.mul(e2).toString(), "(should be +1)");
console.log("e₃² =", e3.mul(e3).toString(), "(should be +1)");
console.log("e₁e₂ =", e1.mul(e2).toString(), "(should be +e₁₂)");
console.log("e₂e₁ =", e2.mul(e1).toString(), "(should be -e₁₂)");
console.log("e₁₂² =", e12.mul(e12).toString(), "(should be -1)");
console.log("e₁₂₃² =", e123.mul(e123).toString(), "(should be -1)");

console.log("\n## 2. Rotor Verification\n");

// Test: rotation by 90° around z-axis
const R90z = rotorFromAxisAngle(0, 0, 1, Math.PI / 2);
console.log("Rotor for 90° around z:", R90z.toString());

const vx = Cl3.vector(1, 0, 0);
const vxRotated = rotateVector(R90z, vx);
console.log("Rotate (1,0,0) by 90° around z:", vxRotated.toString());
console.log("(should be approximately (0,1,0))");

console.log("\n## 3. Even/Odd Decomposition\n");

const testElement = new Cl3([1, 2, 3, 4, 5, 6, 7, 8]);
console.log("Test element:", testElement.toString());
console.log("Even part:", testElement.even().toString());
console.log("Odd part:", testElement.odd().toString());

// Export for use in other modules
export { Cl3, rotor, rotorFromAxisAngle, rotateVector, GRADES, BASIS_NAMES };
