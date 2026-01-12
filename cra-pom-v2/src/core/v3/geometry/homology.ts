/**
 * Homological Analysis for Geometric Cognition
 *
 * Extracts topological signals (Betti numbers) from geometric states.
 * This enables detection of structural patterns that are invariant under
 * continuous deformation.
 *
 * BETTI NUMBERS:
 * - b₀: Connected components (clusters of related concepts)
 * - b₁: 1-dimensional holes (loops/cycles in relationships)
 * - b₂: 2-dimensional voids (surfaces enclosing empty space)
 * - b₃: 3-dimensional voids (volumes in 4D)
 *
 * COGNITIVE SIGNIFICANCE:
 * - b₀ change = new concept cluster formed/merged
 * - b₁ increase = cyclic relationship detected (e.g., circle of fifths!)
 * - b₂ void = conceptual gap or contradiction
 * - Persistence = stability of topological features across scales
 *
 * IMPLEMENTATION:
 * Uses simplicial complex approximation via Rips complex construction.
 * At each radius threshold, we build simplices from nearby points and
 * compute homology groups.
 *
 * REFERENCES:
 * - Edelsbrunner & Harer, "Computational Topology"
 * - https://en.wikipedia.org/wiki/Persistent_homology
 */

import type { Vector4D } from '../music/music-geometry-domain';
import { distance4D } from '../music/polytopes';

// ============================================================================
// Types
// ============================================================================

export interface BettiNumbers {
  b0: number;  // Connected components
  b1: number;  // 1D holes (loops)
  b2: number;  // 2D voids (cavities)
  b3: number;  // 3D voids (in 4D)
}

export interface PersistenceInterval {
  birth: number;    // Threshold where feature appears
  death: number;    // Threshold where feature disappears (Infinity = never)
  dimension: number;  // Which Betti number (0, 1, 2, or 3)
}

export interface PersistenceDiagram {
  intervals: PersistenceInterval[];
  maxRadius: number;
}

export type TopologicalEventType =
  | 'cluster_formed'      // b₀ increased
  | 'cluster_merged'      // b₀ decreased
  | 'loop_formed'         // b₁ increased
  | 'loop_filled'         // b₁ decreased
  | 'void_formed'         // b₂ increased
  | 'void_filled'         // b₂ decreased
  | 'hypervoid_formed'    // b₃ increased
  | 'hypervoid_filled';   // b₃ decreased

export interface TopologicalEvent {
  type: TopologicalEventType;
  dimension: number;
  delta: number;
  threshold: number;
}

export interface SimplexData {
  vertices: number[];   // Vertex indices
  dimension: number;    // 0 = point, 1 = edge, 2 = triangle, 3 = tetrahedron
}

// ============================================================================
// Distance Matrix
// ============================================================================

/**
 * Compute pairwise distance matrix for points
 */
export function computeDistanceMatrix(points: Vector4D[]): number[][] {
  const n = points.length;
  const matrix: number[][] = [];

  for (let i = 0; i < n; i++) {
    matrix[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = 0;
      } else if (j < i) {
        matrix[i][j] = matrix[j][i];
      } else {
        matrix[i][j] = distance4D(points[i], points[j]);
      }
    }
  }

  return matrix;
}

// ============================================================================
// Rips Complex Construction
// ============================================================================

/**
 * Build Rips complex at given threshold
 * A simplex is included if all pairwise distances are ≤ threshold
 */
export function buildRipsComplex(
  distanceMatrix: number[][],
  threshold: number,
  maxDimension: number = 3
): SimplexData[] {
  const n = distanceMatrix.length;
  const simplices: SimplexData[] = [];

  // 0-simplices (vertices) - all points
  for (let i = 0; i < n; i++) {
    simplices.push({ vertices: [i], dimension: 0 });
  }

  // 1-simplices (edges) - pairs within threshold
  const edges: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (distanceMatrix[i][j] <= threshold) {
        simplices.push({ vertices: [i, j], dimension: 1 });
        edges.push([i, j]);
      }
    }
  }

  if (maxDimension < 2) return simplices;

  // 2-simplices (triangles) - check all triples
  const triangles: number[][] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (distanceMatrix[i][j] > threshold) continue;
      for (let k = j + 1; k < n; k++) {
        if (distanceMatrix[i][k] <= threshold &&
            distanceMatrix[j][k] <= threshold) {
          simplices.push({ vertices: [i, j, k], dimension: 2 });
          triangles.push([i, j, k]);
        }
      }
    }
  }

  if (maxDimension < 3) return simplices;

  // 3-simplices (tetrahedra) - check all quadruples
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (distanceMatrix[i][j] > threshold) continue;
      for (let k = j + 1; k < n; k++) {
        if (distanceMatrix[i][k] > threshold ||
            distanceMatrix[j][k] > threshold) continue;
        for (let l = k + 1; l < n; l++) {
          if (distanceMatrix[i][l] <= threshold &&
              distanceMatrix[j][l] <= threshold &&
              distanceMatrix[k][l] <= threshold) {
            simplices.push({ vertices: [i, j, k, l], dimension: 3 });
          }
        }
      }
    }
  }

  return simplices;
}

// ============================================================================
// Homology Computation (Simplified)
// ============================================================================

/**
 * Compute connected components (b₀) using union-find
 */
function computeB0(simplices: SimplexData[], n: number): number {
  // Union-find for connected components
  const parent = new Array(n).fill(-1);

  function find(x: number): number {
    if (parent[x] < 0) return x;
    parent[x] = find(parent[x]);
    return parent[x];
  }

  function union(x: number, y: number): void {
    const px = find(x);
    const py = find(y);
    if (px !== py) {
      // Union by rank
      if (parent[px] < parent[py]) {
        parent[px] += parent[py];
        parent[py] = px;
      } else {
        parent[py] += parent[px];
        parent[px] = py;
      }
    }
  }

  // Process all edges
  for (const simplex of simplices) {
    if (simplex.dimension === 1) {
      union(simplex.vertices[0], simplex.vertices[1]);
    }
  }

  // Count roots (negative parent values)
  let components = 0;
  for (let i = 0; i < n; i++) {
    if (parent[i] < 0) components++;
  }

  return components;
}

/**
 * Compute b₁ (loops) using Euler characteristic approximation
 * χ = V - E + F - T + ...
 * For simplicial complex: b₀ - b₁ + b₂ - b₃ = χ
 */
function computeBettiApprox(simplices: SimplexData[], n: number): BettiNumbers {
  // Count simplices by dimension
  const counts = [0, 0, 0, 0]; // vertices, edges, faces, tetrahedra
  for (const simplex of simplices) {
    if (simplex.dimension <= 3) {
      counts[simplex.dimension]++;
    }
  }

  const b0 = computeB0(simplices, n);

  // Use Euler characteristic relationship for approximation
  // This is a simplification; true computation requires boundary operator
  const euler = counts[0] - counts[1] + counts[2] - counts[3];

  // Approximate b₁ from Euler characteristic: χ ≈ b₀ - b₁ + b₂ - b₃
  // For typical point clouds, b₂ and b₃ are often small or zero initially
  // Rough estimate: b₁ ≈ b₀ - χ (assuming b₂, b₃ small)
  const b1Est = Math.max(0, b0 - euler);

  // b₂ approximation based on void detection
  // A void requires a closed surface of triangles with no tetrahedra inside
  // Simplified: b₂ ≈ triangles that don't belong to any tetrahedron / factor
  const b2Est = Math.max(0, Math.floor(counts[2] / 20) - Math.floor(counts[3] / 2));

  // b₃ for 4D voids (very rare in typical data)
  const b3Est = 0;

  return {
    b0,
    b1: b1Est,
    b2: b2Est,
    b3: b3Est,
  };
}

/**
 * Compute Betti numbers from point cloud at given threshold
 */
export function computeBetti(points: Vector4D[], threshold: number): BettiNumbers {
  if (points.length === 0) {
    return { b0: 0, b1: 0, b2: 0, b3: 0 };
  }

  const distMatrix = computeDistanceMatrix(points);
  const simplices = buildRipsComplex(distMatrix, threshold, 3);
  return computeBettiApprox(simplices, points.length);
}

// ============================================================================
// Persistent Homology
// ============================================================================

/**
 * Compute persistence diagram by tracking Betti numbers across thresholds
 */
export function computePersistence(
  points: Vector4D[],
  numThresholds: number = 20,
  maxRadius?: number
): PersistenceDiagram {
  if (points.length === 0) {
    return { intervals: [], maxRadius: 0 };
  }

  // Determine max radius from data if not provided
  const distMatrix = computeDistanceMatrix(points);
  let computedMaxRadius = maxRadius ?? 0;

  if (!maxRadius) {
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        if (distMatrix[i][j] > computedMaxRadius) {
          computedMaxRadius = distMatrix[i][j];
        }
      }
    }
  }

  // Sample thresholds
  const thresholds: number[] = [];
  for (let i = 0; i <= numThresholds; i++) {
    thresholds.push((i / numThresholds) * computedMaxRadius);
  }

  // Track Betti numbers at each threshold
  const history: { threshold: number; betti: BettiNumbers }[] = [];
  for (const threshold of thresholds) {
    const simplices = buildRipsComplex(distMatrix, threshold, 3);
    const betti = computeBettiApprox(simplices, points.length);
    history.push({ threshold, betti });
  }

  // Extract persistence intervals
  const intervals: PersistenceInterval[] = [];

  // Track birth/death of features
  for (let dim = 0; dim <= 3; dim++) {
    let prevCount = 0;
    const births: number[] = [];

    for (const { threshold, betti } of history) {
      const bettiKey = `b${dim}` as keyof BettiNumbers;
      const count = betti[bettiKey];

      // New features born
      while (births.length < count) {
        births.push(threshold);
      }

      // Features died
      while (births.length > count) {
        const birth = births.pop()!;
        intervals.push({
          birth,
          death: threshold,
          dimension: dim,
        });
      }

      prevCount = count;
    }

    // Remaining features persist to infinity
    for (const birth of births) {
      intervals.push({
        birth,
        death: Infinity,
        dimension: dim,
      });
    }
  }

  return {
    intervals,
    maxRadius: computedMaxRadius,
  };
}

/**
 * Get persistence (death - birth) for an interval
 */
export function getIntervalPersistence(interval: PersistenceInterval): number {
  if (interval.death === Infinity) return Infinity;
  return interval.death - interval.birth;
}

/**
 * Filter persistence diagram to significant features
 */
export function filterPersistence(
  diagram: PersistenceDiagram,
  minPersistence: number
): PersistenceInterval[] {
  return diagram.intervals.filter(
    interval => getIntervalPersistence(interval) >= minPersistence
  );
}

// ============================================================================
// Topological Event Detection
// ============================================================================

/**
 * Detect topological changes between two states
 */
export function detectTransition(
  before: BettiNumbers,
  after: BettiNumbers,
  threshold: number = 0
): TopologicalEvent[] {
  const events: TopologicalEvent[] = [];

  // Check each dimension
  const checks: { dim: number; key: keyof BettiNumbers; incType: TopologicalEventType; decType: TopologicalEventType }[] = [
    { dim: 0, key: 'b0', incType: 'cluster_formed', decType: 'cluster_merged' },
    { dim: 1, key: 'b1', incType: 'loop_formed', decType: 'loop_filled' },
    { dim: 2, key: 'b2', incType: 'void_formed', decType: 'void_filled' },
    { dim: 3, key: 'b3', incType: 'hypervoid_formed', decType: 'hypervoid_filled' },
  ];

  for (const { dim, key, incType, decType } of checks) {
    const delta = after[key] - before[key];
    if (delta > 0) {
      events.push({
        type: incType,
        dimension: dim,
        delta,
        threshold,
      });
    } else if (delta < 0) {
      events.push({
        type: decType,
        dimension: dim,
        delta: Math.abs(delta),
        threshold,
      });
    }
  }

  return events;
}

// ============================================================================
// HomologyAnalyzer Class
// ============================================================================

export class HomologyAnalyzer {
  private points: Vector4D[];
  private distanceMatrix: number[][] | null = null;
  private cachedBetti: Map<number, BettiNumbers> = new Map();

  constructor(points: Vector4D[] = []) {
    this.points = points;
  }

  /**
   * Set/update the point cloud
   */
  setPoints(points: Vector4D[]): void {
    this.points = points;
    this.distanceMatrix = null;
    this.cachedBetti.clear();
  }

  /**
   * Get distance matrix (cached)
   */
  getDistanceMatrix(): number[][] {
    if (!this.distanceMatrix) {
      this.distanceMatrix = computeDistanceMatrix(this.points);
    }
    return this.distanceMatrix;
  }

  /**
   * Compute Betti numbers at threshold (cached)
   */
  computeBetti(threshold: number): BettiNumbers {
    // Round threshold for caching
    const key = Math.round(threshold * 1000) / 1000;
    if (this.cachedBetti.has(key)) {
      return this.cachedBetti.get(key)!;
    }

    const distMatrix = this.getDistanceMatrix();
    const simplices = buildRipsComplex(distMatrix, threshold, 3);
    const betti = computeBettiApprox(simplices, this.points.length);

    this.cachedBetti.set(key, betti);
    return betti;
  }

  /**
   * Compute persistence diagram
   */
  computePersistence(numThresholds: number = 20): PersistenceDiagram {
    return computePersistence(this.points, numThresholds);
  }

  /**
   * Find optimal threshold where b₁ first appears (loop detection)
   */
  findFirstLoop(): number | null {
    const persistence = this.computePersistence(50);
    const loopIntervals = persistence.intervals.filter(i => i.dimension === 1);

    if (loopIntervals.length === 0) return null;

    // Find minimum birth time
    let minBirth = Infinity;
    for (const interval of loopIntervals) {
      if (interval.birth < minBirth) {
        minBirth = interval.birth;
      }
    }

    return minBirth === Infinity ? null : minBirth;
  }

  /**
   * Detect circle of fifths structure
   * Should show b₁ = 1 at appropriate threshold
   */
  detectCyclicStructure(): { threshold: number; loopCount: number } | null {
    const persistence = this.computePersistence(30);

    // Find persistent loops (long-lived b₁ features)
    const significantLoops = filterPersistence(persistence, persistence.maxRadius * 0.1)
      .filter(i => i.dimension === 1);

    if (significantLoops.length === 0) return null;

    // Find threshold where we have the most loops
    let bestThreshold = significantLoops[0].birth;
    let maxLoops = 0;

    for (let t = 0; t <= persistence.maxRadius; t += persistence.maxRadius / 30) {
      const betti = this.computeBetti(t);
      if (betti.b1 > maxLoops) {
        maxLoops = betti.b1;
        bestThreshold = t;
      }
    }

    return { threshold: bestThreshold, loopCount: maxLoops };
  }

  /**
   * Get summary of topological features
   */
  getSummary(threshold: number): {
    betti: BettiNumbers;
    description: string[];
  } {
    const betti = this.computeBetti(threshold);
    const description: string[] = [];

    if (betti.b0 === 1) {
      description.push('All points connected (single cluster)');
    } else if (betti.b0 > 1) {
      description.push(`${betti.b0} separate clusters`);
    }

    if (betti.b1 > 0) {
      description.push(`${betti.b1} loop(s)/cycle(s) detected`);
    }

    if (betti.b2 > 0) {
      description.push(`${betti.b2} void(s)/cavity(ies) detected`);
    }

    if (betti.b3 > 0) {
      description.push(`${betti.b3} 4D hypervoid(s) detected`);
    }

    if (description.length === 0) {
      description.push('No topological features at this threshold');
    }

    return { betti, description };
  }

  /**
   * Get number of points
   */
  get pointCount(): number {
    return this.points.length;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create analyzer for polytope vertices
 */
export function createPolytopeAnalyzer(vertices: Vector4D[]): HomologyAnalyzer {
  return new HomologyAnalyzer(vertices);
}

/**
 * Quick Betti computation for a set of points
 */
export function quickBetti(points: Vector4D[], threshold: number): BettiNumbers {
  return computeBetti(points, threshold);
}

// ============================================================================
// Exports
// ============================================================================

export const HomologyModule = {
  // Core computation
  computeBetti,
  computeDistanceMatrix,
  buildRipsComplex,

  // Persistence
  computePersistence,
  getIntervalPersistence,
  filterPersistence,

  // Event detection
  detectTransition,

  // Class
  HomologyAnalyzer,

  // Factory
  createPolytopeAnalyzer,
  quickBetti,
};

export default HomologyModule;
