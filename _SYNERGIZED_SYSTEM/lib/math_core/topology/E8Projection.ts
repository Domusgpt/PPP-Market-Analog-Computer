/**
 * E₈→H₄ Projection Pipeline
 *
 * Implements the dimensional cascade from the E₈ root system to 4D H₄ polytopes.
 * Based on the Baez-style 4×8 projection matrix and Conway-Sloane icosian norm.
 *
 * Key mathematical structures:
 * - E₈: 240 roots in 8D, split into two φ-scaled 600-cells
 * - H₄: Symmetry group of 600-cell (order 14400)
 * - φ (golden ratio): (1 + √5) / 2 ≈ 1.618
 *
 * The projection uses Galois conjugation: φ ↔ -1/φ to generate the two
 * nested 600-cells that emerge from E₈.
 */

import type { Vector4D } from '../geometric_algebra/types.js';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Golden ratio φ = (1 + √5) / 2 */
export const PHI = (1 + Math.sqrt(5)) / 2;

/** Conjugate golden ratio φ' = -1/φ = (1 - √5) / 2 */
export const PHI_CONJUGATE = (1 - Math.sqrt(5)) / 2;

/** √2 for normalization */
const SQRT2 = Math.sqrt(2);

/** 1/√2 */
const INV_SQRT2 = 1 / SQRT2;

// =============================================================================
// TYPES
// =============================================================================

/** 8-dimensional vector for E₈ */
export type Vector8D = [number, number, number, number, number, number, number, number];

/** E₈ root with metadata */
export interface E8Root {
  /** 8D coordinates */
  readonly coordinates: Vector8D;
  /** Squared norm (should be 2 for E₈ roots) */
  readonly normSquared: number;
  /** Root type: 'permutation' or 'half-integer' */
  readonly type: 'permutation' | 'half-integer';
}

/** Projected 4D point from E₈ */
export interface ProjectedPoint {
  /** Primary 4D coordinates (φ-scaled) */
  readonly outer: Vector4D;
  /** Secondary 4D coordinates (φ'-scaled) */
  readonly inner: Vector4D;
  /** Original E₈ root */
  readonly source: E8Root;
  /** Distance from origin in outer space */
  readonly outerRadius: number;
  /** Distance from origin in inner space */
  readonly innerRadius: number;
}

/** Nested 600-cell structure from E₈ projection */
export interface Nested600Cells {
  /** Outer 600-cell (φ-scaled) - 120 vertices */
  readonly outer: Vector4D[];
  /** Inner 600-cell (φ'-scaled) - 120 vertices */
  readonly inner: Vector4D[];
  /** All 240 E₈ roots with projections */
  readonly roots: ProjectedPoint[];
  /** Scale factor for outer */
  readonly outerScale: number;
  /** Scale factor for inner */
  readonly innerScale: number;
}

// =============================================================================
// BAEZ PROJECTION MATRIX
// =============================================================================

/**
 * The 4×8 projection matrix from Baez's construction.
 * Projects E₈ roots onto H₄ (4D) using the icosian representation.
 *
 * The matrix encodes the relationship between E₈ and H₄ via the
 * binary icosahedral group 2I.
 */
const BAEZ_PROJECTION_MATRIX: number[][] = [
  [1, PHI, 0, -1, PHI, 0, 0, 0],
  [PHI, 0, 1, PHI, 0, -1, 0, 0],
  [0, 1, PHI, 0, -1, PHI, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, PHI]
].map(row => row.map(x => x / 2)); // Normalize

/**
 * Alternative projection matrix using Galois conjugate φ' = -1/φ.
 */
const BAEZ_CONJUGATE_MATRIX: number[][] = [
  [1, PHI_CONJUGATE, 0, -1, PHI_CONJUGATE, 0, 0, 0],
  [PHI_CONJUGATE, 0, 1, PHI_CONJUGATE, 0, -1, 0, 0],
  [0, 1, PHI_CONJUGATE, 0, -1, PHI_CONJUGATE, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, PHI_CONJUGATE]
].map(row => row.map(x => x / 2));

// =============================================================================
// E₈ ROOT GENERATION
// =============================================================================

/**
 * Generate all 240 roots of the E₈ lattice.
 *
 * The E₈ roots come in two types:
 * 1. 112 roots: Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
 * 2. 128 roots: Half-integer coordinates with even number of minus signs
 *    (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2)
 */
export function generateE8Roots(): E8Root[] {
  const roots: E8Root[] = [];

  // Type 1: Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
  for (let i = 0; i < 8; i++) {
    for (let j = i + 1; j < 8; j++) {
      for (const si of [-1, 1]) {
        for (const sj of [-1, 1]) {
          const coords: Vector8D = [0, 0, 0, 0, 0, 0, 0, 0];
          coords[i] = si;
          coords[j] = sj;
          roots.push({
            coordinates: coords,
            normSquared: 2,
            type: 'permutation'
          });
        }
      }
    }
  }

  // Type 2: Half-integer coordinates with even number of minus signs
  // Generate all 2^8 = 256 sign patterns
  for (let pattern = 0; pattern < 256; pattern++) {
    // Count minus signs
    let minusCount = 0;
    for (let bit = 0; bit < 8; bit++) {
      if ((pattern >> bit) & 1) minusCount++;
    }

    // Only keep patterns with even number of minus signs
    if (minusCount % 2 === 0) {
      const coords: Vector8D = [0, 0, 0, 0, 0, 0, 0, 0];
      for (let bit = 0; bit < 8; bit++) {
        coords[bit] = ((pattern >> bit) & 1) ? -0.5 : 0.5;
      }
      roots.push({
        coordinates: coords,
        normSquared: 2,
        type: 'half-integer'
      });
    }
  }

  return roots;
}

// =============================================================================
// PROJECTION FUNCTIONS
// =============================================================================

/**
 * Apply 4×8 matrix to 8D vector to get 4D vector.
 */
function applyProjectionMatrix(matrix: number[][], v: Vector8D): Vector4D {
  const result: Vector4D = [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 8; j++) {
      result[i] += matrix[i][j] * v[j];
    }
  }
  return result;
}

/**
 * Compute Euclidean norm of 4D vector.
 */
function norm4D(v: Vector4D): number {
  return Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]);
}

/**
 * Project a single E₈ root to 4D.
 * Returns both the φ-scaled (outer) and φ'-scaled (inner) projections.
 */
export function projectE8Root(root: E8Root): ProjectedPoint {
  const outer = applyProjectionMatrix(BAEZ_PROJECTION_MATRIX, root.coordinates);
  const inner = applyProjectionMatrix(BAEZ_CONJUGATE_MATRIX, root.coordinates);

  return {
    outer,
    inner,
    source: root,
    outerRadius: norm4D(outer),
    innerRadius: norm4D(inner)
  };
}

/**
 * Project all E₈ roots and organize into nested 600-cells.
 */
export function projectE8ToNested600Cells(): Nested600Cells {
  const e8Roots = generateE8Roots();
  const projections = e8Roots.map(projectE8Root);

  // Separate into outer and inner based on radius
  // The outer 600-cell has larger radius (φ-scaled)
  // The inner 600-cell has smaller radius (φ'-scaled)

  const outerVertices: Vector4D[] = [];
  const innerVertices: Vector4D[] = [];

  // Track unique vertices (some roots project to same point)
  const outerSet = new Set<string>();
  const innerSet = new Set<string>();

  const tolerance = 0.0001;
  const toKey = (v: Vector4D) =>
    `${v[0].toFixed(4)},${v[1].toFixed(4)},${v[2].toFixed(4)},${v[3].toFixed(4)}`;

  for (const proj of projections) {
    const outerKey = toKey(proj.outer);
    const innerKey = toKey(proj.inner);

    if (!outerSet.has(outerKey)) {
      outerSet.add(outerKey);
      outerVertices.push(proj.outer);
    }

    if (!innerSet.has(innerKey)) {
      innerSet.add(innerKey);
      innerVertices.push(proj.inner);
    }
  }

  // Calculate scale factors
  const outerScale = outerVertices.length > 0
    ? norm4D(outerVertices[0])
    : PHI;
  const innerScale = innerVertices.length > 0
    ? norm4D(innerVertices[0])
    : Math.abs(PHI_CONJUGATE);

  return {
    outer: outerVertices,
    inner: innerVertices,
    roots: projections,
    outerScale,
    innerScale
  };
}

// =============================================================================
// GALOIS CONJUGATION
// =============================================================================

/**
 * Apply Galois conjugation to a 4D point.
 * Swaps φ ↔ -1/φ in the coordinate expressions.
 *
 * This maps points on the outer 600-cell to the inner 600-cell.
 */
export function galoisConjugate4D(v: Vector4D): Vector4D {
  // The conjugation effectively scales by φ' / φ = 1/φ²
  const scale = 1 / (PHI * PHI);
  return [
    v[0] * scale,
    v[1] * scale,
    v[2] * scale,
    v[3] * scale
  ];
}

/**
 * Check if a point lies on the outer or inner 600-cell.
 */
export function classifyPoint(
  v: Vector4D,
  nested: Nested600Cells,
  tolerance: number = 0.01
): 'outer' | 'inner' | 'unknown' {
  const radius = norm4D(v);

  if (Math.abs(radius - nested.outerScale) < tolerance) {
    return 'outer';
  }
  if (Math.abs(radius - nested.innerScale) < tolerance) {
    return 'inner';
  }
  return 'unknown';
}

// =============================================================================
// CONWAY-SLOANE ICOSIAN NORM
// =============================================================================

/**
 * Compute the icosian norm of a quaternion-like 4D vector.
 *
 * The icosian ring consists of quaternions of the form:
 * a + bi + cj + dk where a,b,c,d are in Z[φ] (integers extended by φ)
 *
 * The icosian norm is: N(q) = q * q̄ (quaternion multiplication)
 */
export function icosianNorm(v: Vector4D): number {
  // For a quaternion q = w + xi + yj + zk
  // The norm is w² + x² + y² + z²
  return v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3];
}

/**
 * Check if a point has unit icosian norm (lies on unit 3-sphere).
 */
export function hasUnitIcosianNorm(v: Vector4D, tolerance: number = 0.0001): boolean {
  return Math.abs(icosianNorm(v) - 1) < tolerance;
}

/**
 * Normalize a 4D vector to have unit icosian norm.
 */
export function normalizeIcosian(v: Vector4D): Vector4D {
  const n = Math.sqrt(icosianNorm(v));
  if (n < 0.0001) return [1, 0, 0, 0];
  return [v[0]/n, v[1]/n, v[2]/n, v[3]/n];
}

// =============================================================================
// 600-CELL VERTEX GENERATION (DIRECT)
// =============================================================================

/**
 * Generate the 120 vertices of a 600-cell directly using the
 * binary icosahedral group 2I.
 *
 * The vertices include:
 * - 8 vertices: permutations of (±1, 0, 0, 0)
 * - 16 vertices: (±1/2, ±1/2, ±1/2, ±1/2)
 * - 96 vertices: even permutations of (0, ±1/2, ±φ/2, ±1/(2φ))
 */
export function generate600CellVertices(): Vector4D[] {
  const vertices: Vector4D[] = [];
  const invPhi = 1 / PHI;

  // 8 vertices: permutations of (±1, 0, 0, 0)
  for (let i = 0; i < 4; i++) {
    for (const s of [-1, 1]) {
      const v: Vector4D = [0, 0, 0, 0];
      v[i] = s;
      vertices.push(v);
    }
  }

  // 16 vertices: (±1/2, ±1/2, ±1/2, ±1/2)
  for (let pattern = 0; pattern < 16; pattern++) {
    const v: Vector4D = [
      (pattern & 1) ? -0.5 : 0.5,
      (pattern & 2) ? -0.5 : 0.5,
      (pattern & 4) ? -0.5 : 0.5,
      (pattern & 8) ? -0.5 : 0.5
    ];
    vertices.push(v);
  }

  // 96 vertices: even permutations of (0, ±1/2, ±φ/2, ±1/(2φ))
  const base = [0, 0.5, PHI/2, invPhi/2];

  // Even permutations of 4 elements (12 total)
  const evenPerms = [
    [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2],
    [1, 0, 3, 2], [1, 2, 0, 3], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 1, 3, 0], [2, 3, 0, 1],
    [3, 0, 2, 1], [3, 1, 0, 2], [3, 2, 1, 0]
  ];

  for (const perm of evenPerms) {
    // Apply sign patterns to non-zero components
    for (let signs = 0; signs < 8; signs++) {
      const v: Vector4D = [0, 0, 0, 0];
      let signIdx = 0;

      for (let i = 0; i < 4; i++) {
        const val = base[perm[i]];
        if (val !== 0) {
          v[i] = ((signs >> signIdx) & 1) ? -val : val;
          signIdx++;
        } else {
          v[i] = 0;
        }
      }

      vertices.push(v);
    }
  }

  return vertices;
}

// =============================================================================
// E₈ PROJECTION CLASS
// =============================================================================

/**
 * E₈ to H₄ Projection Pipeline.
 *
 * Manages the dimensional cascade from E₈ (8D) to H₄ (4D) polytopes.
 */
export class E8ProjectionPipeline {
  private _e8Roots: E8Root[];
  private _nested600Cells: Nested600Cells;

  constructor() {
    this._e8Roots = generateE8Roots();
    this._nested600Cells = projectE8ToNested600Cells();
  }

  /** Get all E₈ roots */
  get e8Roots(): E8Root[] {
    return this._e8Roots;
  }

  /** Get nested 600-cell structure */
  get nested600Cells(): Nested600Cells {
    return this._nested600Cells;
  }

  /** Get outer 600-cell vertices */
  get outerVertices(): Vector4D[] {
    return this._nested600Cells.outer;
  }

  /** Get inner 600-cell vertices */
  get innerVertices(): Vector4D[] {
    return this._nested600Cells.inner;
  }

  /**
   * Project an arbitrary 8D vector to 4D.
   */
  project(v8: Vector8D): { outer: Vector4D; inner: Vector4D } {
    return {
      outer: applyProjectionMatrix(BAEZ_PROJECTION_MATRIX, v8),
      inner: applyProjectionMatrix(BAEZ_CONJUGATE_MATRIX, v8)
    };
  }

  /**
   * Find the nearest E₈ root to an 8D point.
   */
  findNearestRoot(v8: Vector8D): E8Root {
    let best = this._e8Roots[0];
    let bestDist = Infinity;

    for (const root of this._e8Roots) {
      let dist = 0;
      for (let i = 0; i < 8; i++) {
        const d = v8[i] - root.coordinates[i];
        dist += d * d;
      }
      if (dist < bestDist) {
        bestDist = dist;
        best = root;
      }
    }

    return best;
  }

  /**
   * Get vertices at a specific scale level.
   */
  getVerticesAtScale(scale: 'outer' | 'inner' | 'both'): Vector4D[] {
    switch (scale) {
      case 'outer':
        return this._nested600Cells.outer;
      case 'inner':
        return this._nested600Cells.inner;
      case 'both':
        return [...this._nested600Cells.outer, ...this._nested600Cells.inner];
    }
  }

  /**
   * Interpolate between outer and inner 600-cells.
   *
   * @param t - Interpolation factor (0 = outer, 1 = inner)
   */
  interpolateScale(t: number): Vector4D[] {
    const result: Vector4D[] = [];
    const n = Math.min(this._nested600Cells.outer.length, this._nested600Cells.inner.length);

    for (let i = 0; i < n; i++) {
      const outer = this._nested600Cells.outer[i];
      const inner = this._nested600Cells.inner[i];

      result.push([
        outer[0] * (1 - t) + inner[0] * t,
        outer[1] * (1 - t) + inner[1] * t,
        outer[2] * (1 - t) + inner[2] * t,
        outer[3] * (1 - t) + inner[3] * t
      ]);
    }

    return result;
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export default E8ProjectionPipeline;
