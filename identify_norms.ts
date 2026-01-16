// Identify the missing norm values algebraically
const PHI = (1 + Math.sqrt(5)) / 2;

const norms = [0.382, 0.618, 0.727, 0.874, 1.000, 1.070, 1.176, 1.328, 1.414, 1.618, 1.732];

console.log("Identifying all norm values algebraically:\n");

for (const n of norms) {
  const n2 = n * n;
  console.log(`Norm ≈ ${n.toFixed(3)}, norm² ≈ ${n2.toFixed(4)}`);

  // Check various φ expressions for the norm itself
  const candidates = [
    { expr: "1/φ²", val: 1/(PHI*PHI) },
    { expr: "1/φ", val: 1/PHI },
    { expr: "2-φ", val: 2-PHI },
    { expr: "φ-1", val: PHI-1 },
    { expr: "√((3-φ)/2)", val: Math.sqrt((3-PHI)/2) },
    { expr: "1", val: 1 },
    { expr: "√(3-φ)", val: Math.sqrt(3-PHI) },
    { expr: "√2", val: Math.sqrt(2) },
    { expr: "φ", val: PHI },
    { expr: "√3", val: Math.sqrt(3) },
    { expr: "√(φ+2)", val: Math.sqrt(PHI+2) },
    { expr: "√2/φ", val: Math.sqrt(2)/PHI },
    { expr: "√(2/φ)", val: Math.sqrt(2/PHI) },
    { expr: "√((φ+1)/2)", val: Math.sqrt((PHI+1)/2) },
    { expr: "√(φ/2)", val: Math.sqrt(PHI/2) },
    { expr: "1/√φ", val: 1/Math.sqrt(PHI) },
    { expr: "√(1/φ)", val: Math.sqrt(1/PHI) },
    { expr: "√(1+1/φ²)", val: Math.sqrt(1 + 1/(PHI*PHI)) },
    { expr: "√(2-1/φ)", val: Math.sqrt(2 - 1/PHI) },
    { expr: "√(1+1/φ)", val: Math.sqrt(1 + 1/PHI) },
    { expr: "√(φ-1/φ)", val: Math.sqrt(PHI - 1/PHI) },
    { expr: "√(2φ-1)/φ", val: Math.sqrt(2*PHI-1)/PHI },
    { expr: "√((5-φ)/2)", val: Math.sqrt((5-PHI)/2) },
    { expr: "√(5-2φ)", val: Math.sqrt(5-2*PHI) },
  ];

  let found = false;
  for (const c of candidates) {
    if (Math.abs(n - c.val) < 0.002) {
      console.log(`  → MATCH: ${c.expr} = ${c.val.toFixed(6)}`);
      found = true;
    }
  }

  if (!found) {
    console.log(`  → NO SIMPLE MATCH FOUND`);
    // Try to find what norm² equals
    console.log(`  → Searching for norm² = ${n2.toFixed(6)}...`);

    // Brute force check
    for (let a = -3; a <= 3; a++) {
      for (let b = -3; b <= 3; b++) {
        const testVal = a + b * PHI;
        if (Math.abs(n2 - testVal) < 0.001 && testVal > 0) {
          console.log(`  → norm² = ${a} + ${b}φ = ${testVal.toFixed(6)}`);
        }
        const testVal2 = (a + b * PHI) / 2;
        if (Math.abs(n2 - testVal2) < 0.001 && testVal2 > 0) {
          console.log(`  → norm² = (${a} + ${b}φ)/2 = ${testVal2.toFixed(6)}`);
        }
      }
    }
  }
  console.log();
}

// Analyze the cross-block inner products
console.log("=".repeat(60));
console.log("CROSS-BLOCK INNER PRODUCT ANALYSIS");
console.log("=".repeat(60));

console.log("\nValue -0.618 appearing in cross-block:");
console.log(`  -1/φ = ${(-1/PHI).toFixed(6)}`);
console.log(`  1-φ = ${(1-PHI).toFixed(6)}`);
console.log(`  These are equal! (since 1/φ = φ-1, so -1/φ = 1-φ)`);

console.log("\nValue 0.382 appearing in within-block (H4L):");
console.log(`  1/φ² = ${(1/(PHI*PHI)).toFixed(6)}`);
console.log(`  2-φ = ${(2-PHI).toFixed(6)}`);
console.log(`  These are equal! (since 1/φ² = 1/(φ+1) = ... = 2-φ)`);

console.log("\nValue 1.000 appearing in within-block (H4R) off-diagonal:");
console.log(`  This is just 1 = φ - 1/φ`);

console.log("\n" + "=".repeat(60));
console.log("COMPLETE INNER PRODUCT STRUCTURE");
console.log("=".repeat(60));

console.log(`
H4L within-block Gram matrix (norm² = 3-φ ≈ 1.382):
  Diagonal: 3-φ
  Off-diagonal (0,3) and (3,0): 2-φ = 1/φ²
  Others: 0

H4R within-block Gram matrix (norm² = φ+2 ≈ 3.618):
  Diagonal: φ+2
  Off-diagonal (4,7) and (7,4): 1
  Others: 0

Cross-block Gram matrix:
  (0,4), (1,5), (2,6), (3,7): 1 = φ - 1/φ  [corresponding pairs]
  (0,7), (3,4): 1-φ = -1/φ                 [swapped pairs]
  Others: 0
`);
