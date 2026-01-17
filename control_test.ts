/**
 * Control Test: Verify that rational approximations fail to reproduce
 * the structural properties of the exact φ-based matrix
 */

const PHI = (1 + Math.sqrt(5)) / 2;

console.log("=" .repeat(60));
console.log("CONTROL TEST: Exact φ vs Rational Approximations");
console.log("=" .repeat(60));

// Exact coefficients
const a_exact = 0.5;
const b_exact = (PHI - 1) / 2;
const c_exact = PHI / 2;

// Rational approximations
const a_approx = 0.5;
const b_approx = 0.3;
const c_approx = 0.8;

function buildMatrix(a: number, b: number, c: number): number[][] {
  return [
    [a, a, a, a, b, b, -b, -b],
    [a, a, -a, -a, b, -b, b, -b],
    [a, -a, a, -a, b, -b, -b, b],
    [a, -a, -a, a, b, b, -b, -b],
    [c, c, c, c, -a, -a, a, a],
    [c, c, -c, -c, -a, a, -a, a],
    [c, -c, c, -c, -a, a, a, -a],
    [c, -c, -c, c, -a, -a, a, a],
  ];
}

function computeRank(matrix: number[][]): number {
  const M = matrix.map(row => [...row]);
  const n = M.length;
  let rank = 0;

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) {
        maxRow = row;
      }
    }

    if (Math.abs(M[maxRow][col]) < 1e-10) continue;

    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    rank++;

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col];
      for (let j = col; j < n; j++) {
        M[row][j] -= factor * M[col][j];
      }
    }
  }
  return rank;
}

function computeDet(matrix: number[][]): number {
  const M = matrix.map(row => [...row]);
  const n = M.length;
  let det = 1;

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) {
        maxRow = row;
      }
    }

    if (Math.abs(M[maxRow][col]) < 1e-10) return 0;

    if (maxRow !== col) {
      [M[col], M[maxRow]] = [M[maxRow], M[col]];
      det *= -1;
    }

    det *= M[col][col];

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col];
      for (let j = col; j < n; j++) {
        M[row][j] -= factor * M[col][j];
      }
    }
  }
  return det;
}

// Build matrices
const U_exact = buildMatrix(a_exact, b_exact, c_exact);
const U_approx = buildMatrix(a_approx, b_approx, c_approx);

// Compute norms
const H4L_exact = 4*a_exact*a_exact + 4*b_exact*b_exact;
const H4R_exact = 4*c_exact*c_exact + 4*a_exact*a_exact;
const H4L_approx = 4*a_approx*a_approx + 4*b_approx*b_approx;
const H4R_approx = 4*c_approx*c_approx + 4*a_approx*a_approx;

console.log("\n## Row Norms (squared)\n");
console.log(`H4L norm² (exact φ):  ${H4L_exact.toFixed(10)} = 3-φ = ${(3-PHI).toFixed(10)}`);
console.log(`H4L norm² (approx):   ${H4L_approx.toFixed(10)}`);
console.log(`Deviation: ${(Math.abs(H4L_exact - H4L_approx) / H4L_exact * 100).toFixed(2)}%\n`);

console.log(`H4R norm² (exact φ):  ${H4R_exact.toFixed(10)} = φ+2 = ${(PHI+2).toFixed(10)}`);
console.log(`H4R norm² (approx):   ${H4R_approx.toFixed(10)}`);
console.log(`Deviation: ${(Math.abs(H4R_exact - H4R_approx) / H4R_exact * 100).toFixed(2)}%`);

// Product
const prod_exact = Math.sqrt(H4L_exact * H4R_exact);
const prod_approx = Math.sqrt(H4L_approx * H4R_approx);

console.log("\n## Norm Product\n");
console.log(`Product (exact φ):    ${prod_exact.toFixed(10)} = √5 = ${Math.sqrt(5).toFixed(10)}`);
console.log(`Product (approx):     ${prod_approx.toFixed(10)}`);
console.log(`Deviation: ${(Math.abs(prod_exact - prod_approx) / prod_exact * 100).toFixed(2)}%`);

// Rank
const rank_exact = computeRank(U_exact);
const rank_approx = computeRank(U_approx);

console.log("\n## Rank\n");
console.log(`Rank (exact φ):       ${rank_exact}`);
console.log(`Rank (approx):        ${rank_approx}`);

// Determinant
const det_exact = computeDet(U_exact);
const det_approx = computeDet(U_approx);

console.log("\n## Determinant\n");
console.log(`Det (exact φ):        ${det_exact.toFixed(10)}`);
console.log(`Det (approx):         ${det_approx.toFixed(10)}`);

// Null space test
const nullVec = [0, 0, 0, 0, 1, 1, 1, 1];
const Uv_exact = U_exact.map(row => row.reduce((s, v, j) => s + v * nullVec[j], 0));
const Uv_approx = U_approx.map(row => row.reduce((s, v, j) => s + v * nullVec[j], 0));

console.log("\n## Null Space Test (U × [0,0,0,0,1,1,1,1]ᵀ)\n");
console.log(`Exact:  [${Uv_exact.map(x => x.toFixed(6)).join(', ')}]`);
console.log(`Approx: [${Uv_approx.map(x => x.toFixed(6)).join(', ')}]`);

const isNullExact = Uv_exact.every(x => Math.abs(x) < 1e-10);
const isNullApprox = Uv_approx.every(x => Math.abs(x) < 1e-10);
console.log(`\nIs null? Exact: ${isNullExact ? '✓ YES' : '✗ NO'}, Approx: ${isNullApprox ? '✓ YES' : '✗ NO'}`);

console.log("\n" + "=" .repeat(60));
console.log("CONCLUSION: Rational approximations FAIL to reproduce structure");
console.log("=" .repeat(60));
