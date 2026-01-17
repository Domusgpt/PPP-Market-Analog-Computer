/**
 * Verify Moxness matrix properties and diagnose the 16+4 problem
 */
import {
    generateE8Roots,
    createMoxnessMatrix,
    applyMoxnessMatrix,
    extractH4Left,
    extractH4Right,
    foldE8toH4
} from './lib/topology/E8H4Folding.js';

const PHI = (1 + Math.sqrt(5)) / 2;

console.log("=== Moxness Matrix Verification ===\n");

// 1. Check the folding result
const result = foldE8toH4();
console.log("foldE8toH4() results:");
console.log("  h4L (unit scale):  " + result.h4L.length + " vertices");
console.log("  h4L_phi (φ scale): " + result.h4L_phi.length + " vertices");
console.log("  h4R (unit scale):  " + result.h4R.length + " vertices");
console.log("  h4R_phi (φ scale): " + result.h4R_phi.length + " vertices");
console.log("  TOTAL unique: " + (result.h4L.length + result.h4L_phi.length + result.h4R.length + result.h4R_phi.length));

// 2. Check norm distribution
const e8Roots = generateE8Roots();
const matrix = createMoxnessMatrix();

const leftNorms: number[] = [];
const rightNorms: number[] = [];

for (const root of e8Roots) {
    const rotated = applyMoxnessMatrix(root, matrix);
    const left = extractH4Left(rotated);
    const right = extractH4Right(rotated);

    const leftNorm = Math.sqrt(left[0]**2 + left[1]**2 + left[2]**2 + left[3]**2);
    const rightNorm = Math.sqrt(right[0]**2 + right[1]**2 + right[2]**2 + right[3]**2);

    leftNorms.push(leftNorm);
    rightNorms.push(rightNorm);
}

// Find unique norms
const uniqueLeftNorms = [...new Set(leftNorms.map(n => Math.round(n * 1000) / 1000))].sort((a,b) => a-b);
const uniqueRightNorms = [...new Set(rightNorms.map(n => Math.round(n * 1000) / 1000))].sort((a,b) => a-b);

console.log("\n=== Norm Distribution from 240 E8 projections ===");
console.log("LEFT projection unique norms: " + uniqueLeftNorms.join(", "));
console.log("RIGHT projection unique norms: " + uniqueRightNorms.join(", "));

console.log("\n=== Current Filter Thresholds ===");
console.log("Unit scale: |norm - 1| < 0.1  → [0.9, 1.1]");
console.log("φ-scale:    |norm - φ| < 0.1  → [" + (PHI-0.1).toFixed(3) + ", " + (PHI+0.1).toFixed(3) + "]");

// Count by category
let leftUnit = 0, leftPhi = 0, leftMissed = 0;
let rightUnit = 0, rightPhi = 0, rightMissed = 0;

for (const n of leftNorms) {
    if (Math.abs(n - 1) < 0.1) leftUnit++;
    else if (Math.abs(n - PHI) < 0.1) leftPhi++;
    else leftMissed++;
}

for (const n of rightNorms) {
    if (Math.abs(n - 1) < 0.1) rightUnit++;
    else if (Math.abs(n - PHI) < 0.1) rightPhi++;
    else rightMissed++;
}

console.log("\n=== Projections by Filter Category ===");
console.log("LEFT:  unit=" + leftUnit + ", φ-scale=" + leftPhi + ", MISSED=" + leftMissed + " of 240");
console.log("RIGHT: unit=" + rightUnit + ", φ-scale=" + rightPhi + ", MISSED=" + rightMissed + " of 240");

// Show what norms are being missed
const missedLeftNorms = uniqueLeftNorms.filter(n => Math.abs(n - 1) >= 0.1 && Math.abs(n - PHI) >= 0.1);
const missedRightNorms = uniqueRightNorms.filter(n => Math.abs(n - 1) >= 0.1 && Math.abs(n - PHI) >= 0.1);

if (missedLeftNorms.length > 0) {
    console.log("\nMISSED LEFT norms (not caught by filters): " + missedLeftNorms.join(", "));
}
if (missedRightNorms.length > 0) {
    console.log("MISSED RIGHT norms (not caught by filters): " + missedRightNorms.join(", "));
}
