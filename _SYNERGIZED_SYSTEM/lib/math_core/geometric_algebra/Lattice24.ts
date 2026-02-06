/**
 * 24-Cell Lattice with Trinity Decomposition (Enhanced Orthocognitum)
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module implements the 24-cell with Trinity decomposition into three 16-cells.
 * It extends the PPP-info-site implementation with:
 * - Trinity axis assignment (Alpha/Beta/Gamma)
 * - Phase shift detection between axes
 * - Musical mapping preparation (Octatonic collections)
 *
 * Mathematical Foundation (D3 from Phillips GIT):
 * The 24-cell UNIQUELY satisfies:
 * - Self-duality: Vertices ↔ cells interchangeable
 * - Quaternionic closure: 24 vertices = 24 unit Hurwitz quaternions
 * - Optimal packing: D₄ lattice achieves densest 4D sphere packing
 * - Tripartite decomposition: Exactly 3 disjoint 16-cells partition the 24 vertices
 *
 * Trinity Decomposition (W(D₄) ⊂ W(F₄) with index 3):
 * - Alpha: axes (1,2) and (3,4) → vertices (±1, ±1, 0, 0) and (0, 0, ±1, ±1)
 * - Beta:  axes (1,3) and (2,4) → vertices (±1, 0, ±1, 0) and (0, ±1, 0, ±1)
 * - Gamma: axes (1,4) and (2,3) → vertices (±1, 0, 0, ±1) and (0, ±1, ±1, 0)
 *
 * References:
 * - Coxeter, H.S.M. "Regular Polytopes" (1973)
 * - Gärdenfors "Conceptual Spaces" (2000), Criterion P for convexity
 * - PPP White Paper: "The Orthocognitum is the Shape of the Known"
 */

import {
  Vector4D,
  LatticeVertex,
  LatticeCell,
  VoronoiRegion,
  ConvexityResult,
  TrinityAxis,
  TrinityDecomposition,
  PhaseShiftInfo,
  RotationPlane,
  MATH_CONSTANTS
} from '../types/index.js';

import {
  dot,
  magnitude,
  normalize,
  centroid as computeCentroid,
  distance,
  distanceSquared
} from './GeometricAlgebra.js';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Distance between adjacent vertices in the 24-cell: √2 */
const EDGE_LENGTH = MATH_CONSTANTS.SQRT2;

/** Circumradius of the unit 24-cell (distance from center to vertex): √2 */
const CIRCUMRADIUS = MATH_CONSTANTS.SQRT2;

/** Inradius: distance from center to cell center: 1 */
const INRADIUS = 1;

/** Number of vertices in the 24-cell */
const NUM_VERTICES = 24;

/** Number of edges in the 24-cell */
const NUM_EDGES = 96;

/** Number of nearest neighbors per vertex */
const NEIGHBORS_PER_VERTEX = 8;

// =============================================================================
// TRINITY DECOMPOSITION STRUCTURE
// =============================================================================

/**
 * The Trinity decomposition partitions the 24 vertices into 3 groups of 8.
 * Each group forms a 16-cell (cross-polytope in 4D).
 *
 * Alpha: Uses axis pairs (X,Y) and (Z,W)
 *   Vertices: (±1, ±1, 0, 0) and (0, 0, ±1, ±1)
 *
 * Beta: Uses axis pairs (X,Z) and (Y,W)
 *   Vertices: (±1, 0, ±1, 0) and (0, ±1, 0, ±1)
 *
 * Gamma: Uses axis pairs (X,W) and (Y,Z)
 *   Vertices: (±1, 0, 0, ±1) and (0, ±1, ±1, 0)
 */

/**
 * Determine which Trinity axis a vertex belongs to based on its coordinates.
 */
function determineTrinityAxis(coords: Vector4D): TrinityAxis {
  const nonZeroIndices: number[] = [];

  for (let i = 0; i < 4; i++) {
    if (Math.abs(coords[i]) > MATH_CONSTANTS.EPSILON) {
      nonZeroIndices.push(i);
    }
  }

  if (nonZeroIndices.length !== 2) {
    // This shouldn't happen for valid 24-cell vertices
    return 'alpha';
  }

  const [i, j] = nonZeroIndices;

  // Alpha: (0,1) or (2,3) → pairs (X,Y) or (Z,W)
  if ((i === 0 && j === 1) || (i === 2 && j === 3)) {
    return 'alpha';
  }

  // Beta: (0,2) or (1,3) → pairs (X,Z) or (Y,W)
  if ((i === 0 && j === 2) || (i === 1 && j === 3)) {
    return 'beta';
  }

  // Gamma: (0,3) or (1,2) → pairs (X,W) or (Y,Z)
  return 'gamma';
}

// =============================================================================
// VERTEX GENERATION WITH TRINITY ASSIGNMENT
// =============================================================================

/**
 * Generate all 24 vertices of the 24-cell with Trinity axis assignment.
 */
function generate24CellVertices(): LatticeVertex[] {
  const vertices: LatticeVertex[] = [];
  let id = 0;

  // Generate all permutations of (±1, ±1, 0, 0)
  for (let i = 0; i < 4; i++) {
    for (let j = i + 1; j < 4; j++) {
      // All 4 sign combinations
      for (const si of [-1, 1]) {
        for (const sj of [-1, 1]) {
          const coords: Vector4D = [0, 0, 0, 0];
          coords[i] = si;
          coords[j] = sj;

          const axis = determineTrinityAxis(coords);

          vertices.push({
            id,
            coordinates: coords,
            neighbors: [], // Computed after all vertices exist
            trinityAxis: axis
          });
          id++;
        }
      }
    }
  }

  return vertices;
}

/**
 * Compute neighbor relationships for all vertices.
 * Two vertices are neighbors if their distance is √2 (edge length).
 * Each vertex has exactly 8 neighbors.
 */
function computeNeighbors(vertices: LatticeVertex[]): void {
  const edgeLengthSq = 2; // √2 squared
  const tolerance = MATH_CONSTANTS.EPSILON;

  for (let i = 0; i < vertices.length; i++) {
    const neighbors: number[] = [];
    const vi = vertices[i].coordinates;

    for (let j = 0; j < vertices.length; j++) {
      if (i === j) continue;

      const vj = vertices[j].coordinates;
      const distSq = distanceSquared(vi, vj);

      if (Math.abs(distSq - edgeLengthSq) < tolerance) {
        neighbors.push(j);
      }
    }

    // Type assertion to modify readonly property during initialization
    (vertices[i] as { neighbors: number[] }).neighbors = neighbors;
  }
}

/**
 * Build the Trinity decomposition from the generated vertices.
 */
function buildTrinityDecomposition(vertices: LatticeVertex[]): TrinityDecomposition {
  const alpha: number[] = [];
  const beta: number[] = [];
  const gamma: number[] = [];

  for (const v of vertices) {
    switch (v.trinityAxis) {
      case 'alpha':
        alpha.push(v.id);
        break;
      case 'beta':
        beta.push(v.id);
        break;
      case 'gamma':
        gamma.push(v.id);
        break;
    }
  }

  return { alpha, beta, gamma };
}

// =============================================================================
// CELL GENERATION
// =============================================================================

/**
 * Generate the 24 octahedral cells of the 24-cell.
 */
function generate24CellCells(vertices: LatticeVertex[]): LatticeCell[] {
  const cells: LatticeCell[] = [];

  for (let i = 0; i < vertices.length; i++) {
    const center = vertices[i].coordinates;
    const cellVertices: number[] = [];

    for (let j = 0; j < vertices.length; j++) {
      if (i === j) continue;
      const v = vertices[j].coordinates;

      // Vertices of cell are those sharing exactly one coordinate with center
      const dotProduct = dot(center, v);

      if (Math.abs(dotProduct - 1) < MATH_CONSTANTS.EPSILON) {
        cellVertices.push(j);
      }
    }

    // Compute cell centroid
    const cellCoords = cellVertices.map(idx => vertices[idx].coordinates);
    const cellCentroid = cellCoords.length > 0
      ? computeCentroid(cellCoords)
      : center;

    cells.push({
      id: i,
      vertices: cellVertices,
      centroid: cellCentroid
    });
  }

  return cells;
}

// =============================================================================
// CONVEXITY AND COHERENCE
// =============================================================================

/**
 * Check if a point lies within the convex hull of the 24-cell.
 *
 * For vertices at (±1, ±1, 0, 0):
 * A point is inside if |xᵢ| + |xⱼ| ≤ √2 for all pairs (i,j)
 */
function isInsideConvexHull(point: Vector4D): boolean {
  const [x, y, z, w] = point;

  // First check: distance to origin must be ≤ circumradius
  const distToOrigin = magnitude(point);
  if (distToOrigin > CIRCUMRADIUS + MATH_CONSTANTS.EPSILON) {
    return false;
  }

  // Second check: all pair constraints must be satisfied
  const constraints = [
    Math.abs(x) + Math.abs(y),
    Math.abs(x) + Math.abs(z),
    Math.abs(x) + Math.abs(w),
    Math.abs(y) + Math.abs(z),
    Math.abs(y) + Math.abs(w),
    Math.abs(z) + Math.abs(w)
  ];

  for (const c of constraints) {
    if (c > CIRCUMRADIUS + MATH_CONSTANTS.EPSILON) {
      return false;
    }
  }

  return true;
}

/**
 * Find the nearest vertex to a point.
 */
function findNearestVertex(point: Vector4D, vertices: LatticeVertex[]): number {
  let minDistSq = Infinity;
  let nearestIdx = 0;

  for (let i = 0; i < vertices.length; i++) {
    const distSq = distanceSquared(point, vertices[i].coordinates);
    if (distSq < minDistSq) {
      minDistSq = distSq;
      nearestIdx = i;
    }
  }

  return nearestIdx;
}

/**
 * Find k-nearest vertices.
 */
function findKNearestVertices(
  point: Vector4D,
  vertices: LatticeVertex[],
  k: number
): number[] {
  const distances = vertices.map((v, idx) => ({
    idx,
    distSq: distanceSquared(point, v.coordinates)
  }));

  distances.sort((a, b) => a.distSq - b.distSq);
  return distances.slice(0, Math.min(k, vertices.length)).map(d => d.idx);
}

/**
 * Compute coherence with Trinity-aware weighting.
 */
function computeCoherence(
  state: Vector4D,
  vertices: LatticeVertex[],
  k: number
): {
  coherence: number;
  centroid: Vector4D;
  activeVertices: number[];
  trinityWeights: [number, number, number];
} {
  const activeVertices = findKNearestVertices(state, vertices, k);

  // Compute Trinity weights based on active vertex distribution
  let alphaCount = 0, betaCount = 0, gammaCount = 0;

  for (const idx of activeVertices) {
    switch (vertices[idx].trinityAxis) {
      case 'alpha': alphaCount++; break;
      case 'beta': betaCount++; break;
      case 'gamma': gammaCount++; break;
    }
  }

  const total = activeVertices.length || 1;
  const trinityWeights: [number, number, number] = [
    alphaCount / total,
    betaCount / total,
    gammaCount / total
  ];

  // Compute centroid
  const activeCoords = activeVertices.map(idx => vertices[idx].coordinates);
  const localCentroid = computeCentroid(activeCoords);

  // Compute distance to centroid
  const distToCentroid = distance(state, localCentroid);
  const maxExpectedDist = EDGE_LENGTH;

  // Base coherence from distance
  let coherence = 1 - Math.min(distToCentroid / maxExpectedDist, 1);

  // Penalty for being outside convex hull
  if (!isInsideConvexHull(state)) {
    coherence *= 0.5;
  }

  // Alignment bonus
  const centroidNorm = magnitude(localCentroid);
  const stateNorm = magnitude(state);

  if (centroidNorm > MATH_CONSTANTS.EPSILON && stateNorm > MATH_CONSTANTS.EPSILON) {
    const alignment = dot(normalize(state), normalize(localCentroid));
    const alignmentFactor = (alignment + 1) / 2;
    coherence = coherence * 0.7 + alignmentFactor * 0.3;
  }

  return {
    coherence: Math.max(0, Math.min(1, coherence)),
    centroid: localCentroid,
    activeVertices,
    trinityWeights
  };
}

/**
 * Full convexity check with Trinity information.
 */
function checkConvexity(
  state: Vector4D,
  vertices: LatticeVertex[],
  k: number = 4
): ConvexityResult & { trinityWeights: [number, number, number] } {
  const nearestIdx = findNearestVertex(state, vertices);
  const nearestVertex = vertices[nearestIdx];

  const distToNearest = distance(state, nearestVertex.coordinates);

  const { coherence, centroid, activeVertices, trinityWeights } =
    computeCoherence(state, vertices, k);

  const isValid = isInsideConvexHull(state) && coherence > MATH_CONSTANTS.EPSILON;

  return {
    isValid,
    coherence,
    nearestVertex: nearestIdx,
    distance: distToNearest,
    centroid,
    activeVertices,
    trinityWeights
  };
}

// =============================================================================
// PHASE SHIFT DETECTION
// =============================================================================

/**
 * Detect phase shift between two vertices based on Trinity axis.
 */
function detectPhaseShift(
  fromVertex: number,
  toVertex: number,
  vertices: LatticeVertex[]
): PhaseShiftInfo | null {
  const from = vertices[fromVertex];
  const to = vertices[toVertex];

  if (from.trinityAxis === to.trinityAxis) {
    return null; // No phase shift within same axis
  }

  // Find cross-axis vertices (vertices shared between the two 16-cells)
  // In reality, the 16-cells don't share vertices, but they share edges
  // with the complementary 16-cell
  const crossAxisVertices = from.neighbors.filter(n =>
    vertices[n].trinityAxis !== from.trinityAxis &&
    vertices[n].trinityAxis !== to.trinityAxis
  );

  // Determine which rotation plane corresponds to this phase shift
  let rotationPlane: RotationPlane;

  if ((from.trinityAxis === 'alpha' && to.trinityAxis === 'beta') ||
      (from.trinityAxis === 'beta' && to.trinityAxis === 'alpha')) {
    rotationPlane = RotationPlane.YZ; // XY↔XZ transition uses YZ rotation
  } else if ((from.trinityAxis === 'alpha' && to.trinityAxis === 'gamma') ||
             (from.trinityAxis === 'gamma' && to.trinityAxis === 'alpha')) {
    rotationPlane = RotationPlane.ZW; // XY↔XW transition uses ZW rotation
  } else {
    rotationPlane = RotationPlane.XW; // XZ↔YZ transition uses XW rotation
  }

  return {
    from: from.trinityAxis!,
    to: to.trinityAxis!,
    crossAxisVertices,
    rotationPlane,
    direction: 1 // Could be determined from actual trajectory
  };
}

/**
 * Calculate inter-axis tension from active vertices.
 * Higher tension means the state is near a phase shift boundary.
 */
function calculateInterAxisTension(
  activeVertices: number[],
  vertices: LatticeVertex[]
): number {
  if (activeVertices.length === 0) return 0;

  // Count how many different axes are represented
  const axes = new Set<TrinityAxis>();
  for (const idx of activeVertices) {
    if (vertices[idx].trinityAxis) {
      axes.add(vertices[idx].trinityAxis);
    }
  }

  // Maximum tension when all 3 axes are equally represented
  // Minimum tension when only 1 axis is present
  if (axes.size === 1) return 0;
  if (axes.size === 3) return 1;

  // Two axes: intermediate tension
  return 0.5;
}

// =============================================================================
// PROJECTION AND CLAMPING
// =============================================================================

/**
 * Project a point to the convex hull boundary.
 */
function projectToConvexHull(point: Vector4D, _vertices: LatticeVertex[]): Vector4D {
  if (isInsideConvexHull(point)) {
    return point;
  }

  // Binary search along line from origin to point
  let inside: Vector4D = [0, 0, 0, 0];
  let outside = point;

  for (let i = 0; i < 20; i++) {
    const mid: Vector4D = [
      (inside[0] + outside[0]) / 2,
      (inside[1] + outside[1]) / 2,
      (inside[2] + outside[2]) / 2,
      (inside[3] + outside[3]) / 2
    ];

    if (isInsideConvexHull(mid)) {
      inside = mid;
    } else {
      outside = mid;
    }
  }

  return inside;
}

// =============================================================================
// LATTICE24 CLASS
// =============================================================================

/**
 * The enhanced Lattice24 class with Trinity decomposition.
 */
export class Lattice24 {
  private readonly _vertices: LatticeVertex[];
  private readonly _cells: LatticeCell[];
  private readonly _trinity: TrinityDecomposition;
  private readonly _voronoiRegions: Map<number, VoronoiRegion>;
  private readonly _nearestCache: Map<string, number[]>;

  constructor() {
    this._vertices = generate24CellVertices();
    computeNeighbors(this._vertices);
    this._cells = generate24CellCells(this._vertices);
    this._trinity = buildTrinityDecomposition(this._vertices);
    this._voronoiRegions = new Map();
    this._nearestCache = new Map();

    // Compute Voronoi regions
    for (let i = 0; i < this._vertices.length; i++) {
      this._voronoiRegions.set(i, {
        vertexId: i,
        center: this._vertices[i].coordinates,
        radius: EDGE_LENGTH / 2
      });
    }
  }

  // =========================================================================
  // ACCESSORS
  // =========================================================================

  get vertices(): readonly LatticeVertex[] { return this._vertices; }
  get vertexCount(): number { return NUM_VERTICES; }
  get cells(): readonly LatticeCell[] { return this._cells; }
  get trinity(): TrinityDecomposition { return this._trinity; }
  get circumradius(): number { return CIRCUMRADIUS; }
  get edgeLength(): number { return EDGE_LENGTH; }

  // =========================================================================
  // VERTEX LOOKUPS
  // =========================================================================

  getVertex(id: number): LatticeVertex | undefined {
    return this._vertices[id];
  }

  getVoronoiRegion(vertexId: number): VoronoiRegion | undefined {
    return this._voronoiRegions.get(vertexId);
  }

  findNearest(point: Vector4D): number {
    return findNearestVertex(point, this._vertices);
  }

  findKNearest(point: Vector4D, k: number): number[] {
    const quantize = (v: number) => Math.round(v * 100) / 100;
    const key = `${quantize(point[0])},${quantize(point[1])},${quantize(point[2])},${quantize(point[3])},${k}`;

    const cached = this._nearestCache.get(key);
    if (cached) return cached;

    const result = findKNearestVertices(point, this._vertices, k);

    if (this._nearestCache.size > 10000) {
      this._nearestCache.clear();
    }
    this._nearestCache.set(key, result);

    return result;
  }

  // =========================================================================
  // TRINITY OPERATIONS
  // =========================================================================

  /**
   * Get all vertices for a specific Trinity axis.
   */
  getAxisVertices(axis: TrinityAxis): number[] {
    return [...this._trinity[axis]];
  }

  /**
   * Get the Trinity axis for a vertex.
   */
  getVertexAxis(vertexId: number): TrinityAxis | undefined {
    return this._vertices[vertexId]?.trinityAxis;
  }

  /**
   * Detect phase shift between two vertices.
   */
  detectPhaseShift(fromVertex: number, toVertex: number): PhaseShiftInfo | null {
    return detectPhaseShift(fromVertex, toVertex, this._vertices);
  }

  /**
   * Calculate inter-axis tension from current position.
   */
  calculateTension(position: Vector4D, k: number = 4): number {
    const activeVertices = this.findKNearest(position, k);
    return calculateInterAxisTension(activeVertices, this._vertices);
  }

  /**
   * Get Trinity weights for a position.
   */
  getTrinityWeights(position: Vector4D, k: number = 4): [number, number, number] {
    const activeVertices = this.findKNearest(position, k);

    let alpha = 0, beta = 0, gamma = 0;
    for (const idx of activeVertices) {
      switch (this._vertices[idx].trinityAxis) {
        case 'alpha': alpha++; break;
        case 'beta': beta++; break;
        case 'gamma': gamma++; break;
      }
    }

    const total = activeVertices.length || 1;
    return [alpha / total, beta / total, gamma / total];
  }

  /**
   * Get the dominant Trinity axis for a position.
   */
  getDominantAxis(position: Vector4D, k: number = 4): TrinityAxis {
    const [alpha, beta, gamma] = this.getTrinityWeights(position, k);

    if (alpha >= beta && alpha >= gamma) return 'alpha';
    if (beta >= gamma) return 'beta';
    return 'gamma';
  }

  // =========================================================================
  // VALIDATION (EPISTAORTHOGNITION)
  // =========================================================================

  isInside(point: Vector4D): boolean {
    return isInsideConvexHull(point);
  }

  computeCoherence(state: Vector4D, k: number = 4): number {
    return computeCoherence(state, this._vertices, k).coherence;
  }

  checkConvexity(state: Vector4D, k: number = 4): ConvexityResult & {
    trinityWeights: [number, number, number];
  } {
    return checkConvexity(state, this._vertices, k);
  }

  // =========================================================================
  // PROJECTION AND CLAMPING
  // =========================================================================

  project(point: Vector4D): Vector4D {
    return projectToConvexHull(point, this._vertices);
  }

  clamp(point: Vector4D): Vector4D {
    if (isInsideConvexHull(point)) return point;
    return projectToConvexHull(point, this._vertices);
  }

  // =========================================================================
  // NAVIGATION
  // =========================================================================

  getNeighbors(vertexId: number): number[] {
    return [...(this._vertices[vertexId]?.neighbors ?? [])];
  }

  areNeighbors(v1: number, v2: number): boolean {
    return this._vertices[v1]?.neighbors.includes(v2) ?? false;
  }

  /**
   * Get cross-axis neighbors (neighbors from a different Trinity axis).
   */
  getCrossAxisNeighbors(vertexId: number): number[] {
    const vertex = this._vertices[vertexId];
    if (!vertex) return [];

    return vertex.neighbors.filter(n =>
      this._vertices[n].trinityAxis !== vertex.trinityAxis
    );
  }

  /**
   * Geodesic distance between two vertices (edge hops).
   */
  geodesicDistance(from: number, to: number): number {
    if (from === to) return 0;

    const visited = new Set<number>([from]);
    const queue: { id: number; dist: number }[] = [{ id: from, dist: 0 }];

    while (queue.length > 0) {
      const current = queue.shift()!;

      for (const neighborId of this._vertices[current.id].neighbors) {
        if (neighborId === to) {
          return current.dist + 1;
        }

        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          queue.push({ id: neighborId, dist: current.dist + 1 });
        }
      }
    }

    return Infinity;
  }

  // =========================================================================
  // UTILITY
  // =========================================================================

  randomInside(): Vector4D {
    for (let attempt = 0; attempt < 1000; attempt++) {
      const point: Vector4D = [
        (Math.random() - 0.5) * 2 * CIRCUMRADIUS,
        (Math.random() - 0.5) * 2 * CIRCUMRADIUS,
        (Math.random() - 0.5) * 2 * CIRCUMRADIUS,
        (Math.random() - 0.5) * 2 * CIRCUMRADIUS
      ];

      if (isInsideConvexHull(point)) return point;
    }

    return [0, 0, 0, 0];
  }

  randomVertex(): LatticeVertex {
    const idx = Math.floor(Math.random() * NUM_VERTICES);
    return this._vertices[idx];
  }

  /**
   * Random vertex from a specific Trinity axis.
   */
  randomAxisVertex(axis: TrinityAxis): LatticeVertex {
    const axisVertices = this._trinity[axis];
    const idx = axisVertices[Math.floor(Math.random() * axisVertices.length)];
    return this._vertices[idx];
  }

  clearCache(): void {
    this._nearestCache.clear();
  }

  getStats(): Record<string, number | Record<string, number>> {
    return {
      vertices: NUM_VERTICES,
      edges: NUM_EDGES,
      cells: this._cells.length,
      neighborsPerVertex: NEIGHBORS_PER_VERTEX,
      circumradius: CIRCUMRADIUS,
      edgeLength: EDGE_LENGTH,
      inradius: INRADIUS,
      cacheSize: this._nearestCache.size,
      trinity: {
        alpha: this._trinity.alpha.length,
        beta: this._trinity.beta.length,
        gamma: this._trinity.gamma.length
      }
    };
  }
}

// =============================================================================
// SINGLETON AND EXPORTS
// =============================================================================

let _defaultLattice: Lattice24 | null = null;

export function getDefaultLattice(): Lattice24 {
  if (!_defaultLattice) {
    _defaultLattice = new Lattice24();
  }
  return _defaultLattice;
}

export function createLattice(): Lattice24 {
  return new Lattice24();
}

export {
  isInsideConvexHull,
  findNearestVertex,
  findKNearestVertices,
  checkConvexity,
  projectToConvexHull,
  determineTrinityAxis,
  detectPhaseShift,
  calculateInterAxisTension,
  CIRCUMRADIUS,
  EDGE_LENGTH,
  INRADIUS,
  NUM_VERTICES,
  NUM_EDGES
};
