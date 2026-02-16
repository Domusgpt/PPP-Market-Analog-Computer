/**
 * Persistent Homology for Topological Data Analysis
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 *
 * Implements persistent homology computation for analyzing
 * the topological structure of point clouds in the Orthocognitum.
 *
 * Betti Number Interpretation:
 * - β₀: Connected components (cohesion/texture)
 * - β₁: 1D holes/loops (cycles/repetition)
 * - β₂: 2D voids (ambiguity/"ghost frequencies")
 *
 * This is a simplified implementation. For production use,
 * consider integrating Ripser via WASM for performance.
 */

import {
  Vector4D,
  BettiProfile,
  PersistencePair,
  TopologicalVoid,
  MATH_CONSTANTS
} from '../geometric_algebra/types.js';

import { Lattice24, getDefaultLattice } from '../geometric_algebra/Lattice24.js';
import { distance, centroid } from '../geometric_algebra/GeometricAlgebra.js';

// =============================================================================
// TYPES
// =============================================================================

/**
 * Distance matrix representation.
 */
export type DistanceMatrix = Float32Array;

/**
 * Simplex representation.
 */
interface Simplex {
  vertices: number[];
  dimension: number;
  filtrationValue: number;
}

/**
 * Filtration of simplicial complex.
 */
interface Filtration {
  simplices: Simplex[];
  maxDimension: number;
}

// =============================================================================
// DISTANCE MATRIX
// =============================================================================

/**
 * Compute pairwise distance matrix for a set of 4D points.
 */
export function computeDistanceMatrix(points: Vector4D[]): DistanceMatrix {
  const n = points.length;
  const matrix = new Float32Array(n * n);

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const d = distance(points[i], points[j]);
      matrix[i * n + j] = d;
      matrix[j * n + i] = d; // Symmetric
    }
  }

  return matrix;
}

/**
 * Get distance from matrix.
 */
function getDistance(matrix: DistanceMatrix, n: number, i: number, j: number): number {
  return matrix[i * n + j];
}

// =============================================================================
// VIETORIS-RIPS FILTRATION
// =============================================================================

/**
 * Build Vietoris-Rips filtration up to dimension maxDim.
 *
 * At each filtration value ε, the complex contains:
 * - All points (0-simplices)
 * - All edges (i,j) where d(i,j) ≤ ε (1-simplices)
 * - All triangles where all edges are present (2-simplices)
 * - etc.
 */
function buildVietorisRipsFiltration(
  distMatrix: DistanceMatrix,
  n: number,
  maxDim: number,
  maxRadius: number
): Filtration {
  const simplices: Simplex[] = [];

  // 0-simplices (vertices) - all at filtration 0
  for (let i = 0; i < n; i++) {
    simplices.push({
      vertices: [i],
      dimension: 0,
      filtrationValue: 0
    });
  }

  // 1-simplices (edges)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = getDistance(distMatrix, n, i, j);
      if (d <= maxRadius) {
        simplices.push({
          vertices: [i, j],
          dimension: 1,
          filtrationValue: d
        });
      }
    }
  }

  // 2-simplices (triangles) if maxDim >= 2
  if (maxDim >= 2) {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        for (let k = j + 1; k < n; k++) {
          const dij = getDistance(distMatrix, n, i, j);
          const dik = getDistance(distMatrix, n, i, k);
          const djk = getDistance(distMatrix, n, j, k);

          const maxEdge = Math.max(dij, dik, djk);

          if (maxEdge <= maxRadius) {
            simplices.push({
              vertices: [i, j, k],
              dimension: 2,
              filtrationValue: maxEdge
            });
          }
        }
      }
    }
  }

  // Sort by filtration value
  simplices.sort((a, b) => a.filtrationValue - b.filtrationValue);

  return {
    simplices,
    maxDimension: maxDim
  };
}

// =============================================================================
// UNION-FIND (DISJOINT SET)
// =============================================================================

class UnionFind {
  private parent: number[];
  private rank: number[];

  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = Array(n).fill(0);
  }

  find(x: number): number {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]); // Path compression
    }
    return this.parent[x];
  }

  union(x: number, y: number): boolean {
    const rootX = this.find(x);
    const rootY = this.find(y);

    if (rootX === rootY) return false; // Already in same component

    // Union by rank
    if (this.rank[rootX] < this.rank[rootY]) {
      this.parent[rootX] = rootY;
    } else if (this.rank[rootX] > this.rank[rootY]) {
      this.parent[rootY] = rootX;
    } else {
      this.parent[rootY] = rootX;
      this.rank[rootX]++;
    }

    return true;
  }

  getComponentCount(): number {
    const roots = new Set<number>();
    for (let i = 0; i < this.parent.length; i++) {
      roots.add(this.find(i));
    }
    return roots.size;
  }
}

// =============================================================================
// PERSISTENCE COMPUTATION
// =============================================================================

/**
 * Compute persistent homology (simplified algorithm).
 *
 * This implements a basic version focused on H₀ (connected components)
 * and approximates H₁ and H₂ using heuristics.
 */
function computePersistence(
  filtration: Filtration,
  n: number
): PersistencePair[] {
  const pairs: PersistencePair[] = [];

  // H₀: Use union-find to track connected components
  const uf = new UnionFind(n);
  const birthTimes = new Map<number, number>(); // Component root → birth time

  // All vertices are born at time 0
  for (let i = 0; i < n; i++) {
    birthTimes.set(i, 0);
  }

  // Process edges in order
  for (const simplex of filtration.simplices) {
    if (simplex.dimension === 1) {
      const [i, j] = simplex.vertices;
      const rootI = uf.find(i);
      const rootJ = uf.find(j);

      if (rootI !== rootJ) {
        // Merging two components
        const birthI = birthTimes.get(rootI) ?? 0;
        const birthJ = birthTimes.get(rootJ) ?? 0;

        // The younger component dies
        const [younger, older] = birthI > birthJ ? [rootI, rootJ] : [rootJ, rootI];
        const youngerBirth = birthI > birthJ ? birthI : birthJ;

        // Add persistence pair for H₀
        if (simplex.filtrationValue > youngerBirth + MATH_CONSTANTS.EPSILON) {
          pairs.push({
            birth: youngerBirth,
            death: simplex.filtrationValue,
            dimension: 0
          });
        }

        uf.union(i, j);

        // Update birth time for merged component
        const newRoot = uf.find(i);
        birthTimes.set(newRoot, Math.min(birthI, birthJ));
      }
    }
  }

  // Remaining components persist to infinity (represented by large value)
  const finalComponents = uf.getComponentCount();
  if (finalComponents > 0) {
    // One component persists forever
    const infinityValue = 999;

    // Find the oldest component
    for (let i = 0; i < n; i++) {
      if (uf.find(i) === i) {
        const birth = birthTimes.get(i) ?? 0;
        pairs.push({
          birth,
          death: infinityValue,
          dimension: 0
        });
      }
    }
  }

  // H₁ approximation: Look for triangles that don't "fill in" loops
  // This is a heuristic - real implementation would use boundary matrices
  const triangleMap = new Map<string, number>(); // Edge key → triangle count

  for (const simplex of filtration.simplices) {
    if (simplex.dimension === 2) {
      const [i, j, k] = simplex.vertices;
      const edges = [`${i}-${j}`, `${i}-${k}`, `${j}-${k}`];

      for (const edge of edges) {
        triangleMap.set(edge, (triangleMap.get(edge) ?? 0) + 1);
      }
    }
  }

  // Cycles that persist: edges in exactly one triangle
  let cycleCount = 0;
  for (const simplex of filtration.simplices) {
    if (simplex.dimension === 1) {
      const [i, j] = simplex.vertices;
      const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
      const triangles = triangleMap.get(key) ?? 0;

      if (triangles === 1 && cycleCount < 10) {
        pairs.push({
          birth: simplex.filtrationValue,
          death: simplex.filtrationValue + 0.5, // Heuristic death time
          dimension: 1
        });
        cycleCount++;
      }
    }
  }

  // H₂ approximation: Look for "cavities" formed by triangles
  // Count triangles that form closed surfaces
  const triangleCount = filtration.simplices.filter(s => s.dimension === 2).length;
  if (triangleCount > 4) {
    // Heuristic: more triangles suggest potential voids
    const voidCount = Math.floor(triangleCount / 10);
    for (let i = 0; i < voidCount && i < 5; i++) {
      const triangleSimplex = filtration.simplices.find(s => s.dimension === 2);
      if (triangleSimplex) {
        pairs.push({
          birth: triangleSimplex.filtrationValue,
          death: triangleSimplex.filtrationValue + 1.0,
          dimension: 2
        });
      }
    }
  }

  return pairs;
}

// =============================================================================
// BETTI NUMBERS
// =============================================================================

/**
 * Compute Betti numbers from persistence pairs at a given scale.
 */
function computeBettiNumbers(
  pairs: PersistencePair[],
  scale: number
): { beta0: number; beta1: number; beta2: number } {
  let beta0 = 0;
  let beta1 = 0;
  let beta2 = 0;

  for (const pair of pairs) {
    if (pair.birth <= scale && pair.death > scale) {
      switch (pair.dimension) {
        case 0: beta0++; break;
        case 1: beta1++; break;
        case 2: beta2++; break;
      }
    }
  }

  return { beta0, beta1, beta2 };
}

// =============================================================================
// VOID DETECTION
// =============================================================================

/**
 * Detect topological voids (β₂ features) and estimate their centers.
 */
function detectVoids(
  points: Vector4D[],
  pairs: PersistencePair[],
  lattice: Lattice24
): TopologicalVoid[] {
  const voids: TopologicalVoid[] = [];

  // Get β₂ pairs (voids)
  const voidPairs = pairs.filter(p => p.dimension === 2);

  for (const pair of voidPairs) {
    const persistence = pair.death - pair.birth;

    if (persistence > 0.1) { // Significant persistence
      // Estimate void center as centroid of points at this scale
      const scale = (pair.birth + pair.death) / 2;

      // Find points within scale
      const activePoints: Vector4D[] = [];
      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          if (distance(points[i], points[j]) <= scale) {
            if (!activePoints.includes(points[i])) activePoints.push(points[i]);
            if (!activePoints.includes(points[j])) activePoints.push(points[j]);
          }
        }
      }

      if (activePoints.length > 0) {
        const center = centroid(activePoints);

        // Find nearest lattice vertex ("ghost vertex")
        const nearestVertex = lattice.findNearest(center);

        voids.push({
          center,
          radius: scale,
          persistence,
          nearestVertex
        });
      }
    }
  }

  // Sort by persistence (most persistent first)
  voids.sort((a, b) => b.persistence - a.persistence);

  return voids;
}

// =============================================================================
// HARMONIC TOPOLOGIST CLASS
// =============================================================================

/**
 * HarmonicTopologist provides TDA analysis for the CPE.
 *
 * Usage:
 * ```typescript
 * const topologist = new HarmonicTopologist();
 * const profile = topologist.analyze([[0,0,0,0], [1,1,0,0], [0,1,1,0]]);
 * console.log(`β₀: ${profile.beta0}, β₁: ${profile.beta1}, β₂: ${profile.beta2}`);
 * ```
 */
export class HarmonicTopologist {
  private _lattice: Lattice24;
  private _maxDimension: number;
  private _maxRadius: number;

  constructor(
    lattice?: Lattice24,
    maxDimension: number = 2,
    maxRadius: number = 3.0
  ) {
    this._lattice = lattice ?? getDefaultLattice();
    this._maxDimension = maxDimension;
    this._maxRadius = maxRadius;
  }

  /**
   * Analyze a set of 4D points.
   */
  analyze(points: Vector4D[]): BettiProfile {
    if (points.length < 2) {
      return {
        beta0: points.length,
        beta1: 0,
        beta2: 0,
        persistence: new Float32Array(0)
      };
    }

    // Compute distance matrix
    const distMatrix = computeDistanceMatrix(points);

    // Build filtration
    const filtration = buildVietorisRipsFiltration(
      distMatrix,
      points.length,
      this._maxDimension,
      this._maxRadius
    );

    // Compute persistence
    const pairs = computePersistence(filtration, points.length);

    // Compute Betti numbers at characteristic scale
    const characteristicScale = this._lattice.edgeLength;
    const betti = computeBettiNumbers(pairs, characteristicScale);

    // Flatten persistence pairs into Float32Array
    const persistence = new Float32Array(pairs.length * 3);
    for (let i = 0; i < pairs.length; i++) {
      persistence[i * 3] = pairs[i].birth;
      persistence[i * 3 + 1] = pairs[i].death;
      persistence[i * 3 + 2] = pairs[i].dimension;
    }

    return {
      ...betti,
      persistence
    };
  }

  /**
   * Analyze points at multiple scales.
   */
  analyzeMultiScale(
    points: Vector4D[],
    scales: number[]
  ): Map<number, { beta0: number; beta1: number; beta2: number }> {
    const results = new Map();

    if (points.length < 2) {
      for (const scale of scales) {
        results.set(scale, { beta0: points.length, beta1: 0, beta2: 0 });
      }
      return results;
    }

    const distMatrix = computeDistanceMatrix(points);
    const filtration = buildVietorisRipsFiltration(
      distMatrix,
      points.length,
      this._maxDimension,
      this._maxRadius
    );
    const pairs = computePersistence(filtration, points.length);

    for (const scale of scales) {
      results.set(scale, computeBettiNumbers(pairs, scale));
    }

    return results;
  }

  /**
   * Detect topological voids.
   */
  detectVoids(points: Vector4D[]): TopologicalVoid[] {
    if (points.length < 4) {
      return [];
    }

    const distMatrix = computeDistanceMatrix(points);
    const filtration = buildVietorisRipsFiltration(
      distMatrix,
      points.length,
      this._maxDimension,
      this._maxRadius
    );
    const pairs = computePersistence(filtration, points.length);

    return detectVoids(points, pairs, this._lattice);
  }

  /**
   * Get "ghost vertices" - vertices that should be activated to fill voids.
   */
  getGhostVertices(points: Vector4D[]): number[] {
    const voids = this.detectVoids(points);
    return voids.map(v => v.nearestVertex);
  }

  /**
   * Compute ambiguity score based on β₂.
   * Higher β₂ = more ambiguity.
   */
  computeAmbiguity(points: Vector4D[]): number {
    const profile = this.analyze(points);
    // Normalize by number of points
    return profile.beta2 / Math.max(1, points.length / 4);
  }

  /**
   * Compute cohesion score based on β₀.
   * Lower β₀ = higher cohesion.
   */
  computeCohesion(points: Vector4D[]): number {
    if (points.length === 0) return 0;

    const profile = this.analyze(points);
    // 1 component = perfect cohesion, many components = low cohesion
    return 1 / profile.beta0;
  }

  /**
   * Compute cyclicity score based on β₁.
   */
  computeCyclicity(points: Vector4D[]): number {
    const profile = this.analyze(points);
    return profile.beta1 / Math.max(1, points.length / 3);
  }

  // =========================================================================
  // STATISTICS
  // =========================================================================

  getStats(): Record<string, unknown> {
    return {
      maxDimension: this._maxDimension,
      maxRadius: this._maxRadius,
      latticeVertexCount: this._lattice.vertexCount
    };
  }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

export function createHarmonicTopologist(
  maxDimension?: number,
  maxRadius?: number
): HarmonicTopologist {
  return new HarmonicTopologist(undefined, maxDimension, maxRadius);
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  computeDistanceMatrix,
  buildVietorisRipsFiltration,
  computePersistence,
  computeBettiNumbers,
  detectVoids
};
