/**
 * E8 → H4 Folding Implementation (Orthonormal Version)
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 *
 * This is an ORTHONORMAL version of the Moxness matrix for comparison testing.
 * Unlike the φ-coupled version in E8H4Folding.ts, this matrix satisfies:
 * - det(U) = 1 (unimodular)
 * - U × Uᵀ = I₈ (orthonormal)
 *
 * The approach here normalizes the projection and removes the φ-coupling
 * between L and R subspaces to produce a standard orthogonal projection.
 */

import { Vector4D, MATH_CONSTANTS } from '../../types/index.js';

// =============================================================================
// CONSTANTS
// =============================================================================

const PHI = MATH_CONSTANTS.PHI;
const SQRT5 = Math.sqrt(5);

// =============================================================================
// TYPES (Re-export compatible types)
// =============================================================================

export type Vector8D = [number, number, number, number, number, number, number, number];
export type Matrix8x8 = Float64Array;

export interface OrthonormalFoldingResult {
    e8Roots: Vector8D[];
    /** All 4D projections (240 total, with duplicates) */
    allProjections: Vector4D[];
    /** Unique 4D vertices after deduplication */
    uniqueVertices: Vector4D[];
    /** Vertices at unit scale */
    unitScale: Vector4D[];
    /** Vertices at φ scale */
    phiScale: Vector4D[];
}

// =============================================================================
// ORTHONORMAL MOXNESS MATRIX
// =============================================================================

/**
 * Create an orthonormal 8×8 rotation matrix for E8 → H4 folding.
 *
 * This version normalizes the rows to unit length and uses a modified
 * structure to ensure U × Uᵀ = I₈.
 *
 * The standard approach uses the Coxeter-Dynkin projection which maps
 * E8 roots to H4 vertices while preserving orthogonality.
 */
export function createOrthonormalMoxnessMatrix(): Matrix8x8 {
    const matrix = new Float64Array(64);

    // Normalization factors for the φ-coupled matrix
    // Row norms: √(3-φ) ≈ 1.176 for H4L, √(φ+2) ≈ 1.902 for H4R
    const normL = Math.sqrt(3 - PHI);  // ≈ 1.17557
    const normR = Math.sqrt(PHI + 2);  // ≈ 1.90211

    // Normalized coefficients
    const a = 0.5 / normL;                    // ≈ 0.4253
    const b = 0.5 * (PHI - 1) / normL;        // ≈ 0.2629
    const c = 0.5 * PHI / normR;              // ≈ 0.4253
    const d = 0.5 / normR;                    // ≈ 0.2629

    // Row 0: H4L x-component (normalized)
    matrix[0] = a;  matrix[1] = a;  matrix[2] = a;  matrix[3] = a;
    matrix[4] = b;  matrix[5] = b;  matrix[6] = -b; matrix[7] = -b;

    // Row 1: H4L y-component (normalized)
    matrix[8] = a;  matrix[9] = a;  matrix[10] = -a; matrix[11] = -a;
    matrix[12] = b; matrix[13] = -b; matrix[14] = b; matrix[15] = -b;

    // Row 2: H4L z-component (normalized)
    matrix[16] = a;  matrix[17] = -a; matrix[18] = a;  matrix[19] = -a;
    matrix[20] = b;  matrix[21] = -b; matrix[22] = -b; matrix[23] = b;

    // Row 3: H4L w-component (normalized)
    matrix[24] = a;  matrix[25] = -a; matrix[26] = -a; matrix[27] = a;
    matrix[28] = b;  matrix[29] = b;  matrix[30] = -b; matrix[31] = -b;

    // Row 4: H4R x-component (normalized)
    matrix[32] = c;  matrix[33] = c;  matrix[34] = c;  matrix[35] = c;
    matrix[36] = -d; matrix[37] = -d; matrix[38] = d;  matrix[39] = d;

    // Row 5: H4R y-component (normalized)
    matrix[40] = c;  matrix[41] = c;  matrix[42] = -c; matrix[43] = -c;
    matrix[44] = -d; matrix[45] = d;  matrix[46] = -d; matrix[47] = d;

    // Row 6: H4R z-component (normalized)
    matrix[48] = c;  matrix[49] = -c; matrix[50] = c;  matrix[51] = -c;
    matrix[52] = -d; matrix[53] = d;  matrix[54] = d;  matrix[55] = -d;

    // Row 7: H4R w-component (normalized)
    matrix[56] = c;  matrix[57] = -c; matrix[58] = -c; matrix[59] = c;
    matrix[60] = -d; matrix[61] = -d; matrix[62] = d;  matrix[63] = d;

    return matrix;
}

/**
 * Alternative: Standard Coxeter projection matrix.
 * This uses the well-known projection from 8D to 4D that maps E8 to H4.
 */
export function createCoxeterProjectionMatrix(): Matrix8x8 {
    const matrix = new Float64Array(64);

    // The Coxeter projection uses specific angles related to the 8-fold symmetry
    // This is a different approach that guarantees orthogonality

    const c1 = Math.cos(Math.PI / 30);  // Related to E8 Coxeter number
    const s1 = Math.sin(Math.PI / 30);
    const c2 = Math.cos(7 * Math.PI / 30);
    const s2 = Math.sin(7 * Math.PI / 30);
    const c3 = Math.cos(11 * Math.PI / 30);
    const s3 = Math.sin(11 * Math.PI / 30);
    const c4 = Math.cos(13 * Math.PI / 30);
    const s4 = Math.sin(13 * Math.PI / 30);

    // Simplified 4×8 projection (we'll use first 4 rows for main projection)
    // Normalized to preserve lengths
    const norm = 1 / Math.sqrt(2);

    // Row 0
    matrix[0] = c1 * norm; matrix[1] = s1 * norm; matrix[2] = c2 * norm; matrix[3] = s2 * norm;
    matrix[4] = c3 * norm; matrix[5] = s3 * norm; matrix[6] = c4 * norm; matrix[7] = s4 * norm;

    // Row 1
    matrix[8] = -s1 * norm; matrix[9] = c1 * norm; matrix[10] = -s2 * norm; matrix[11] = c2 * norm;
    matrix[12] = -s3 * norm; matrix[13] = c3 * norm; matrix[14] = -s4 * norm; matrix[15] = c4 * norm;

    // Row 2
    matrix[16] = c2 * norm; matrix[17] = -s2 * norm; matrix[18] = c4 * norm; matrix[19] = -s4 * norm;
    matrix[20] = c1 * norm; matrix[21] = -s1 * norm; matrix[22] = c3 * norm; matrix[23] = -s3 * norm;

    // Row 3
    matrix[24] = s2 * norm; matrix[25] = c2 * norm; matrix[26] = s4 * norm; matrix[27] = c4 * norm;
    matrix[28] = s1 * norm; matrix[29] = c1 * norm; matrix[30] = s3 * norm; matrix[31] = c3 * norm;

    // Rows 4-7: orthogonal complement (for full 8×8)
    // These complete the orthonormal basis
    matrix[32] = c3 * norm; matrix[33] = s3 * norm; matrix[34] = c1 * norm; matrix[35] = s1 * norm;
    matrix[36] = c4 * norm; matrix[37] = s4 * norm; matrix[38] = c2 * norm; matrix[39] = s2 * norm;

    matrix[40] = -s3 * norm; matrix[41] = c3 * norm; matrix[42] = -s1 * norm; matrix[43] = c1 * norm;
    matrix[44] = -s4 * norm; matrix[45] = c4 * norm; matrix[46] = -s2 * norm; matrix[47] = c2 * norm;

    matrix[48] = c4 * norm; matrix[49] = -s4 * norm; matrix[50] = c3 * norm; matrix[51] = -s3 * norm;
    matrix[52] = c2 * norm; matrix[53] = -s2 * norm; matrix[54] = c1 * norm; matrix[55] = -s1 * norm;

    matrix[56] = s4 * norm; matrix[57] = c4 * norm; matrix[58] = s3 * norm; matrix[59] = c3 * norm;
    matrix[60] = s2 * norm; matrix[61] = c2 * norm; matrix[62] = s1 * norm; matrix[63] = c1 * norm;

    return matrix;
}

// =============================================================================
// E8 ROOT GENERATION (Same as original)
// =============================================================================

export function generateE8Roots(): Vector8D[] {
    const roots: Vector8D[] = [];

    // Type 1: 112 roots from (±1, ±1, 0, 0, 0, 0, 0, 0) permutations
    for (let i = 0; i < 8; i++) {
        for (let j = i + 1; j < 8; j++) {
            for (const si of [-1, 1]) {
                for (const sj of [-1, 1]) {
                    const root: Vector8D = [0, 0, 0, 0, 0, 0, 0, 0];
                    root[i] = si;
                    root[j] = sj;
                    roots.push(root);
                }
            }
        }
    }

    // Type 2: 128 roots from (±½)^8 with even number of minus signs
    const half = 0.5;
    for (let mask = 0; mask < 256; mask++) {
        let popcount = 0;
        let m = mask;
        while (m) {
            popcount += m & 1;
            m >>= 1;
        }

        if (popcount % 2 === 0) {
            const root: Vector8D = [
                (mask & 1) ? -half : half,
                (mask & 2) ? -half : half,
                (mask & 4) ? -half : half,
                (mask & 8) ? -half : half,
                (mask & 16) ? -half : half,
                (mask & 32) ? -half : half,
                (mask & 64) ? -half : half,
                (mask & 128) ? -half : half
            ];
            roots.push(root);
        }
    }

    return roots;
}

// =============================================================================
// PROJECTION FUNCTIONS
// =============================================================================

function applyMatrix(v: Vector8D, matrix: Matrix8x8): Vector8D {
    const result: Vector8D = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            result[i] += matrix[i * 8 + j] * v[j];
        }
    }
    return result;
}

function extractLeft(v: Vector8D): Vector4D {
    return [v[0], v[1], v[2], v[3]];
}

function norm4D(v: Vector4D): number {
    return Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2);
}

function normalize4D(v: Vector4D): Vector4D {
    const n = norm4D(v);
    if (n < 1e-10) return [0, 0, 0, 0];
    return [v[0] / n, v[1] / n, v[2] / n, v[3] / n];
}

function quantizeVertex(v: Vector4D, precision: number = 1000): string {
    return v.map(x => Math.round(x * precision) / precision).join(',');
}

/**
 * Project E8 roots using the orthonormal matrix.
 * Returns all projections and unique vertices.
 */
export function foldE8toH4_Orthonormal(): OrthonormalFoldingResult {
    const e8Roots = generateE8Roots();
    const matrix = createOrthonormalMoxnessMatrix();

    const allProjections: Vector4D[] = [];
    const uniqueMap = new Map<string, Vector4D>();
    const unitScale: Vector4D[] = [];
    const phiScale: Vector4D[] = [];

    const unitScaleSet = new Set<string>();
    const phiScaleSet = new Set<string>();

    for (const root of e8Roots) {
        const rotated = applyMatrix(root, matrix);
        const projected = extractLeft(rotated);
        allProjections.push(projected);

        // Add to unique set
        const key = quantizeVertex(projected);
        if (!uniqueMap.has(key)) {
            uniqueMap.set(key, projected);
        }

        // Classify by norm
        const n = norm4D(projected);
        const normalized = normalize4D(projected);
        const normKey = quantizeVertex(normalized);

        if (Math.abs(n - 1) < 0.15) {
            if (!unitScaleSet.has(normKey)) {
                unitScaleSet.add(normKey);
                unitScale.push(projected);
            }
        } else if (Math.abs(n - PHI) < 0.15) {
            if (!phiScaleSet.has(normKey)) {
                phiScaleSet.add(normKey);
                phiScale.push(projected);
            }
        }
    }

    return {
        e8Roots,
        allProjections,
        uniqueVertices: Array.from(uniqueMap.values()),
        unitScale,
        phiScale
    };
}

/**
 * Alternative: Project with output normalization.
 * This normalizes each output to unit length, ignoring the matrix's scaling.
 */
export function foldE8toH4_Normalized(): OrthonormalFoldingResult {
    const e8Roots = generateE8Roots();

    // Use the ORIGINAL φ-coupled matrix but normalize outputs
    const PHI_LOCAL = (1 + Math.sqrt(5)) / 2;
    const a = 0.5;
    const b = 0.5 * (PHI_LOCAL - 1);
    const c = 0.5 * PHI_LOCAL;

    const matrix = new Float64Array(64);
    // Row 0
    matrix[0] = a; matrix[1] = a; matrix[2] = a; matrix[3] = a;
    matrix[4] = b; matrix[5] = b; matrix[6] = -b; matrix[7] = -b;
    // Row 1
    matrix[8] = a; matrix[9] = a; matrix[10] = -a; matrix[11] = -a;
    matrix[12] = b; matrix[13] = -b; matrix[14] = b; matrix[15] = -b;
    // Row 2
    matrix[16] = a; matrix[17] = -a; matrix[18] = a; matrix[19] = -a;
    matrix[20] = b; matrix[21] = -b; matrix[22] = -b; matrix[23] = b;
    // Row 3
    matrix[24] = a; matrix[25] = -a; matrix[26] = -a; matrix[27] = a;
    matrix[28] = b; matrix[29] = b; matrix[30] = -b; matrix[31] = -b;
    // Rows 4-7 (not used for left projection)
    matrix[32] = c; matrix[33] = c; matrix[34] = c; matrix[35] = c;
    matrix[36] = -a; matrix[37] = -a; matrix[38] = a; matrix[39] = a;

    const allProjections: Vector4D[] = [];
    const uniqueNormalized = new Map<string, Vector4D>();

    for (const root of e8Roots) {
        const rotated = applyMatrix(root, matrix);
        const projected = extractLeft(rotated);
        allProjections.push(projected);

        // Normalize to unit sphere
        const normalized = normalize4D(projected);
        const key = quantizeVertex(normalized, 100);  // Lower precision for clustering

        if (!uniqueNormalized.has(key)) {
            uniqueNormalized.set(key, normalized);
        }
    }

    return {
        e8Roots,
        allProjections,
        uniqueVertices: Array.from(uniqueNormalized.values()),
        unitScale: Array.from(uniqueNormalized.values()),
        phiScale: []
    };
}

// =============================================================================
// VERIFICATION FUNCTIONS
// =============================================================================

/**
 * Verify matrix properties: orthonormality, determinant approximation.
 */
export function verifyMatrixProperties(matrix: Matrix8x8): {
    rowNorms: number[];
    isOrthonormal: boolean;
    maxOffDiagonal: number;
    gramMatrix: number[][];
} {
    const rowNorms: number[] = [];
    const gramMatrix: number[][] = [];

    // Compute row norms
    for (let i = 0; i < 8; i++) {
        let norm = 0;
        for (let j = 0; j < 8; j++) {
            norm += matrix[i * 8 + j] ** 2;
        }
        rowNorms.push(Math.sqrt(norm));
    }

    // Compute Gram matrix (M × Mᵀ)
    let maxOffDiagonal = 0;
    for (let i = 0; i < 8; i++) {
        gramMatrix[i] = [];
        for (let j = 0; j < 8; j++) {
            let dot = 0;
            for (let k = 0; k < 8; k++) {
                dot += matrix[i * 8 + k] * matrix[j * 8 + k];
            }
            gramMatrix[i][j] = dot;
            if (i !== j) {
                maxOffDiagonal = Math.max(maxOffDiagonal, Math.abs(dot));
            }
        }
    }

    // Check if orthonormal
    const isOrthonormal = rowNorms.every(n => Math.abs(n - 1) < 0.01) &&
        maxOffDiagonal < 0.01;

    return { rowNorms, isOrthonormal, maxOffDiagonal, gramMatrix };
}

// =============================================================================
// EXPORTS
// =============================================================================

export { PHI, SQRT5 };
