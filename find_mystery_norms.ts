const PHI = (1 + Math.sqrt(5)) / 2;

// The mystery norms
const n1 = 0.727;
const n2_val = 1.070;

console.log("Mystery norm 0.727:");
console.log("  norm² =", n1*n1);

console.log("\nMystery norm 1.070:");
console.log("  norm² =", n2_val*n2_val);

// Compute exact from matrix
const a = 0.5;
const b = (PHI - 1) / 2;
const c = PHI / 2;

const U: number[][] = [
  [ a,  a,  a,  a,  b,  b, -b, -b],
  [ a,  a, -a, -a,  b, -b,  b, -b],
  [ a, -a,  a, -a,  b, -b, -b,  b],
  [ a, -a, -a,  a,  b,  b, -b, -b],
  [ c,  c,  c,  c, -a, -a,  a,  a],
  [ c,  c, -c, -c, -a,  a, -a,  a],
  [ c, -c,  c, -c, -a,  a,  a, -a],
  [ c, -c, -c,  c, -a, -a,  a,  a]
];

function project(root: number[]): number[] {
  return U.map(row => row.reduce((s, v, i) => s + v * root[i], 0));
}

function h4lNorm(root: number[]): number {
  const p = project(root);
  return Math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2] + p[3]*p[3]);
}

console.log("\n\nSearching D8 roots for mystery norms...\n");

// Try all D8 roots
for (let i = 0; i < 8; i++) {
  for (let j = i+1; j < 8; j++) {
    for (const si of [-1, 1]) {
      for (const sj of [-1, 1]) {
        const root = [0,0,0,0,0,0,0,0];
        root[i] = si;
        root[j] = sj;
        const n = h4lNorm(root);
        if (Math.abs(n - 0.727) < 0.02) {
          console.log("Found 0.727 norm:");
          console.log("  Root position: i=" + i + ", j=" + j + ", signs: " + si + "," + sj);
          console.log("  Exact norm² = " + (n*n));
          const p = project(root);
          console.log("  H4L projection: [" + p.slice(0,4).map(x => x.toFixed(4)).join(", ") + "]");
          console.log();
        }
        if (Math.abs(n - 1.070) < 0.02) {
          console.log("Found 1.070 norm:");
          console.log("  Root position: i=" + i + ", j=" + j + ", signs: " + si + "," + sj);
          console.log("  Exact norm² = " + (n*n));
          const p = project(root);
          console.log("  H4L projection: [" + p.slice(0,4).map(x => x.toFixed(4)).join(", ") + "]");
          console.log();
        }
      }
    }
  }
}

// Check what 0.7265... squared is in terms of φ
console.log("\n\nAlgebraic identification:");
const testNorm = Math.sqrt((3-PHI)/2);
console.log("√((3-φ)/2) = " + testNorm);

const testNorm2 = Math.sqrt((5-2*PHI)/2);
console.log("√((5-2φ)/2) = " + testNorm2);

const testNorm3 = Math.sqrt(3 - 2*PHI);
console.log("√(3-2φ) = " + testNorm3 + " (imaginary if negative)");

// Check 1/√φ scaled things
console.log("1/√φ = " + (1/Math.sqrt(PHI)));
console.log("√(3-φ)/√2 = " + Math.sqrt(3-PHI)/Math.sqrt(2));

// For 1.070
console.log("\nFor norm ≈ 1.070:");
console.log("  √(3-φ-1/φ) = " + Math.sqrt(3 - PHI - 1/PHI));
console.log("  (1+1/φ)/√2 = " + (1 + 1/PHI)/Math.sqrt(2));

// Let me compute more carefully
// norm² ≈ 0.5285 for 0.727
// norm² ≈ 1.1449 for 1.070

console.log("\n\nChecking (3-φ)/2 = " + ((3-PHI)/2));
console.log("Checking 1 + 1/φ² - 1 = 1/φ² = " + (1/(PHI*PHI)));
console.log("Checking 3/φ² = " + (3/(PHI*PHI)));
console.log("√(3/φ²) = √3/φ = " + (Math.sqrt(3)/PHI));

// Actually the mystery norms
console.log("\n\n=== EXACT COMPUTATION ===");
// A root like (1, 0, 0, 0, 1, 0, 0, 0) - positions 0 and 4
const root04 = [1, 0, 0, 0, 1, 0, 0, 0];
const p04 = project(root04);
console.log("\nRoot (1,0,0,0,1,0,0,0):");
console.log("  H4L = [" + p04.slice(0,4).map(x => x.toFixed(6)).join(", ") + "]");
console.log("  H4L norm = " + h4lNorm(root04));
console.log("  H4L norm² = " + Math.pow(h4lNorm(root04), 2));
console.log("  Expected: a+b = " + (a+b) + " in first component");
console.log("  norm² = 4*(a+b)² = " + (4*(a+b)*(a+b)));

// For a root like (1, 0, 0, 0, 0, 0, 1, 0) - positions 0 and 6
const root06 = [1, 0, 0, 0, 0, 0, 1, 0];
const p06 = project(root06);
console.log("\nRoot (1,0,0,0,0,0,1,0):");
console.log("  H4L = [" + p06.slice(0,4).map(x => x.toFixed(6)).join(", ") + "]");
console.log("  H4L norm = " + h4lNorm(root06));
console.log("  H4L norm² = " + Math.pow(h4lNorm(root06), 2));

// Check algebraically
console.log("\n\n=== ALGEBRAIC ANALYSIS ===");
console.log("a = 1/2, b = (φ-1)/2, c = φ/2");
console.log("a + b = 1/2 + (φ-1)/2 = φ/2 = c");
console.log("a - b = 1/2 - (φ-1)/2 = (2-φ)/2 = 1/φ/2 = 1/(2φ)");
console.log("");
console.log("For root at positions i, j with values si, sj:");
console.log("  If both i,j < 4: H4L gets ±a terms → sum involves a");
console.log("  If both i,j ≥ 4: H4L gets ±b terms → sum involves b");
console.log("  If i < 4 and j ≥ 4: H4L gets one ±a and one ±b → involves a±b");
