/**
 * 24-Cell Lattice Implementation for the Orthocognitum
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module implements the 24-Cell (icositetrachoron), the unique self-dual
 * regular 4-polytope with no 3D analogue. It serves as the "Topological Governor"
 * of the Chronomorphic Polytopal Engine, defining the boundaries of valid thought.
 *
 * The Orthocognitum:
 * - 24 vertices define the "Concept Cells" of the valid semantic space
 * - Voronoi tessellation partitions 4D space into convex regions
 * - States must remain within the convex hull for "Epistaorthognition"
 *
 * Geometric Properties:
 * - Vertices: 24 (permutations of ±1, ±1, 0, 0)
 * - Edges: 96 (connecting vertices at distance √2)
 * - Faces: 96 triangles
 * - Cells: 24 octahedra
 * - Vertex figure: Cube
 * - Each vertex has 8 nearest neighbors
 *
 * References:
 * - Coxeter, H.S.M. "Regular Polytopes" (1973)
 * - Gärdenfors "Conceptual Spaces" (2000)
 * - PPP White Paper: "The Orthocognitum is the Shape of the Known"
 */

import {
    Vector4D,
    LatticeVertex,
    LatticeCell,
    VoronoiRegion,
    ConvexityResult,
    MATH_CONSTANTS
} from '../../types/index.js';

import {
    dot,
    magnitude,
    normalize,
    centroid as computeCentroid
} from '../math/GeometricAlgebra.js';

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * Distance between adjacent vertices in the 24-cell.
 * √2 ≈ 1.414... for vertices at distance √(1² + 1²) = √2
 */
const EDGE_LENGTH = Math.SQRT2;

/**
 * Circumradius of the unit 24-cell (distance from center to vertex).
 * For vertices (±1, ±1, 0, 0): √(1² + 1²) = √2
 */
const CIRCUMRADIUS = Math.SQRT2;

/**
 * Inradius: distance from center to cell center.
 * For 24-cell with circumradius √2: inradius = 1
 */
const INRADIUS = 1;

/**
 * Number of vertices in the 24-cell.
 */
const NUM_VERTICES = 24;

/**
 * Number of edges in the 24-cell.
 */
const NUM_EDGES = 96;

/**
 * Number of nearest neighbors per vertex.
 */
const NEIGHBORS_PER_VERTEX = 8;

// =============================================================================
// VERTEX GENERATION
// =============================================================================

/**
 * Generate all 24 vertices of the 24-cell.
 *
 * The vertices are all permutations of (±1, ±1, 0, 0).
 * This gives 4C2 = 6 ways to choose which coordinates are non-zero,
 * times 2² = 4 sign combinations = 24 total.
 *
 * @returns Array of 24 vertices with coordinates and metadata
 */
function generate24CellVertices(): LatticeVertex[] {
    const vertices: LatticeVertex[] = [];
    let id = 0;

    // For each pair of dimensions (i, j) where i < j
    for (let i = 0; i < 4; i++) {
        for (let j = i + 1; j < 4; j++) {
            // All 4 sign combinations
            for (const si of [-1, 1]) {
                for (const sj of [-1, 1]) {
                    const coords: Vector4D = [0, 0, 0, 0];
                    coords[i] = si;
                    coords[j] = sj;

                    vertices.push({
                        id,
                        coordinates: coords,
                        neighbors: [] // Computed after all vertices exist
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
            const distSq = (vi[0] - vj[0]) ** 2 +
                          (vi[1] - vj[1]) ** 2 +
                          (vi[2] - vj[2]) ** 2 +
                          (vi[3] - vj[3]) ** 2;

            if (Math.abs(distSq - edgeLengthSq) < tolerance) {
                neighbors.push(j);
            }
        }

        // Type assertion to modify readonly property during initialization
        (vertices[i] as { neighbors: number[] }).neighbors = neighbors;
    }
}

// =============================================================================
// CELL GENERATION
// =============================================================================

/**
 * Generate the 24 octahedral cells of the 24-cell.
 * Each cell is centered on a point (±1, 0, 0, 0) or permutation.
 * The 6 vertices of each octahedron are the 6 neighbors of that center point
 * among the 24-cell vertices.
 */
function generate24CellCells(vertices: LatticeVertex[]): LatticeCell[] {
    const cells: LatticeCell[] = [];

    // Cell centers are at (±1, 0, 0, 0) and permutations = 8 positions
    // But 24-cell has 24 cells... each octahedral cell corresponds to
    // a vertex of the dual 24-cell.

    // For the 24-cell, cells are centered at distance 1 from origin
    // along positive/negative coordinate axes and their combinations.

    // Actually, for regular 24-cell cells, we use the dual vertices
    // The dual of 24-cell is 24-cell, so dual vertices are also
    // permutations of (±1, ±1, 0, 0).

    // Alternative approach: Each vertex of the 24-cell is surrounded
    // by 8 neighbors, and is shared by multiple cells.
    // For simplicity, we generate cells based on octahedral structure.

    // Cell centers in normalized 24-cell:
    const cellCenters: Vector4D[] = [];

    // The 24-cell has 24 octahedral cells
    // Centers can be computed as dual vertex positions (scaled)
    // For unit edge length cell, centers are at dual lattice positions

    // Generate using the dual 24-cell vertices (same as primal for self-dual)
    for (let i = 0; i < vertices.length; i++) {
        const center = vertices[i].coordinates;

        // Find 6 vertices that form the octahedron around this dual center
        // These are vertices at specific relationship to the center
        const cellVertices: number[] = [];

        for (let j = 0; j < vertices.length; j++) {
            if (i === j) continue;
            const v = vertices[j].coordinates;

            // Vertices of cell are those orthogonal to center direction
            // For 24-cell, specific geometric relationship
            const dotProduct = dot(center, v);

            // Vertices sharing exactly one coordinate with center
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
// VORONOI TESSELLATION
// =============================================================================

/**
 * Find the Voronoi region (nearest vertex) for a point in 4D space.
 * The Voronoi tessellation partitions space based on nearest-neighbor distance.
 *
 * @param point - Query point in 4D
 * @param vertices - Lattice vertices
 * @returns Index of nearest vertex
 */
function findNearestVertex(point: Vector4D, vertices: LatticeVertex[]): number {
    let minDistSq = Infinity;
    let nearestIdx = 0;

    for (let i = 0; i < vertices.length; i++) {
        const v = vertices[i].coordinates;
        const distSq = (point[0] - v[0]) ** 2 +
                      (point[1] - v[1]) ** 2 +
                      (point[2] - v[2]) ** 2 +
                      (point[3] - v[3]) ** 2;

        if (distSq < minDistSq) {
            minDistSq = distSq;
            nearestIdx = i;
        }
    }

    return nearestIdx;
}

/**
 * Find the k-nearest vertices for a point in 4D space.
 * Used for computing local centroid and coherence.
 *
 * @param point - Query point in 4D
 * @param vertices - Lattice vertices
 * @param k - Number of nearest neighbors to find
 * @returns Indices of k-nearest vertices, sorted by distance
 */
function findKNearestVertices(
    point: Vector4D,
    vertices: LatticeVertex[],
    k: number
): number[] {
    // Compute all distances
    const distances: { idx: number; distSq: number }[] = vertices.map((v, idx) => ({
        idx,
        distSq: (point[0] - v.coordinates[0]) ** 2 +
               (point[1] - v.coordinates[1]) ** 2 +
               (point[2] - v.coordinates[2]) ** 2 +
               (point[3] - v.coordinates[3]) ** 2
    }));

    // Sort by distance
    distances.sort((a, b) => a.distSq - b.distSq);

    // Return first k indices
    return distances.slice(0, Math.min(k, vertices.length)).map(d => d.idx);
}

/**
 * Compute the Voronoi region for a vertex.
 * In the 24-cell, Voronoi regions around vertices are truncated cubes.
 *
 * @param vertexId - Index of the vertex
 * @param vertices - All lattice vertices
 * @returns VoronoiRegion with center and effective radius
 */
function computeVoronoiRegion(vertexId: number, vertices: LatticeVertex[]): VoronoiRegion {
    const center = vertices[vertexId].coordinates;

    // The Voronoi radius is half the edge length (distance to boundary)
    const radius = EDGE_LENGTH / 2;

    return {
        vertexId,
        center,
        radius
    };
}

// =============================================================================
// CONVEXITY CHECKING (EPISTAORTHOGNITION)
// =============================================================================

/**
 * Check if a point lies within the convex hull of the 24-cell.
 *
 * The 24-cell can be defined by the intersection of half-spaces.
 * A point p is inside if it satisfies all bounding inequalities.
 *
 * For the 24-cell with vertices at (±1, ±1, 0, 0) permutations:
 * The convex hull is bounded by |x| + |y| + |z| + |w| ≤ 2 (roughly)
 * More precisely, the faces define half-space constraints.
 *
 * @param point - Point to check
 * @returns true if inside the convex hull
 */
function isInsideConvexHull(point: Vector4D): boolean {
    const [x, y, z, w] = point;

    // The 24-cell bounds: combinations of ±xi ± xj ≤ √2 for all pairs
    // More specifically: all permutations of ±x ± y ≤ √2
    const bound = CIRCUMRADIUS;

    // Check all 32 bounding half-spaces (8 per pair of coordinates)
    // Each face of the octahedral cells defines a constraint

    // Simplified check: Manhattan-type constraint adapted for 24-cell
    // The 24-cell dual is itself, so we can use vertex distances

    // A point is inside if its distance to the origin is less than
    // the circumradius AND it satisfies face constraints
    const distToOrigin = magnitude(point);
    if (distToOrigin > CIRCUMRADIUS + MATH_CONSTANTS.EPSILON) {
        return false;
    }

    // Additional face constraints:
    // For 24-cell, faces are triangular and defined by triplets of vertices
    // Simplified: check against "expanded" octahedral structure

    // Constraint: |xi| + |xj| ≤ √2 for all pairs (i,j)
    const constraints = [
        Math.abs(x) + Math.abs(y),
        Math.abs(x) + Math.abs(z),
        Math.abs(x) + Math.abs(w),
        Math.abs(y) + Math.abs(z),
        Math.abs(y) + Math.abs(w),
        Math.abs(z) + Math.abs(w)
    ];

    for (const c of constraints) {
        if (c > bound + MATH_CONSTANTS.EPSILON) {
            return false;
        }
    }

    return true;
}

/**
 * Compute a coherence score for a state vector.
 *
 * Coherence measures how well the state aligns with the local
 * concept structure (k-nearest lattice vertices).
 *
 * From the White Paper:
 * "If you normalize this sum that represents the centroid...
 *  By definition the centroid must lie within the convex hull."
 *
 * The coherence score is based on:
 * 1. Distance from the centroid of k-nearest vertices
 * 2. Whether the state lies within their convex hull
 * 3. Alignment with the local "concept direction"
 *
 * @param state - Current state vector
 * @param vertices - Lattice vertices
 * @param k - Number of nearest neighbors
 * @returns Coherence score 0.0 (invalid) to 1.0 (perfectly coherent)
 */
function computeCoherence(
    state: Vector4D,
    vertices: LatticeVertex[],
    k: number
): { coherence: number; centroid: Vector4D; activeVertices: number[] } {
    // Find k-nearest vertices
    const activeVertices = findKNearestVertices(state, vertices, k);

    // Compute centroid of active vertices
    const activeCoords = activeVertices.map(idx => vertices[idx].coordinates);
    const localCentroid = computeCentroid(activeCoords);

    // Compute distance from state to centroid
    const distToCentroid = Math.sqrt(
        (state[0] - localCentroid[0]) ** 2 +
        (state[1] - localCentroid[1]) ** 2 +
        (state[2] - localCentroid[2]) ** 2 +
        (state[3] - localCentroid[3]) ** 2
    );

    // Compute maximum expected distance (roughly half the region diameter)
    const maxExpectedDist = EDGE_LENGTH;

    // Coherence: 1 at centroid, 0 at boundary
    let coherence = 1 - Math.min(distToCentroid / maxExpectedDist, 1);

    // Penalty for being outside convex hull
    if (!isInsideConvexHull(state)) {
        coherence *= 0.5; // Reduce coherence but don't zero out
    }

    // Additional alignment factor: dot product with normalized centroid
    const centroidNorm = magnitude(localCentroid);
    const stateNorm = magnitude(state);

    if (centroidNorm > MATH_CONSTANTS.EPSILON && stateNorm > MATH_CONSTANTS.EPSILON) {
        const alignment = dot(
            normalize(state),
            normalize(localCentroid)
        );
        // Blend in alignment factor (range -1 to 1 → 0 to 1)
        const alignmentFactor = (alignment + 1) / 2;
        coherence = coherence * 0.7 + alignmentFactor * 0.3;
    }

    return {
        coherence: Math.max(0, Math.min(1, coherence)),
        centroid: localCentroid,
        activeVertices
    };
}

/**
 * Full convexity check with detailed result.
 * This is the core "Epistaorthognition" operation.
 *
 * @param state - State vector to validate
 * @param vertices - Lattice vertices
 * @param k - Number of nearest neighbors for local analysis
 * @returns Complete ConvexityResult
 */
function checkConvexity(
    state: Vector4D,
    vertices: LatticeVertex[],
    k: number = 4
): ConvexityResult {
    // Find nearest vertex
    const nearestIdx = findNearestVertex(state, vertices);
    const nearestVertex = vertices[nearestIdx];

    // Compute distance to nearest vertex
    const distToNearest = Math.sqrt(
        (state[0] - nearestVertex.coordinates[0]) ** 2 +
        (state[1] - nearestVertex.coordinates[1]) ** 2 +
        (state[2] - nearestVertex.coordinates[2]) ** 2 +
        (state[3] - nearestVertex.coordinates[3]) ** 2
    );

    // Compute coherence
    const { coherence, centroid, activeVertices } = computeCoherence(state, vertices, k);

    // Determine validity
    const isValid = isInsideConvexHull(state) && coherence > MATH_CONSTANTS.EPSILON;

    return {
        isValid,
        coherence,
        nearestVertex: nearestIdx,
        distance: distToNearest,
        centroid,
        activeVertices
    };
}

// =============================================================================
// PROJECTION AND CLAMPING
// =============================================================================

/**
 * Project a point to the nearest valid position on the 24-cell boundary.
 * Used when a state vector exits the Orthocognitum.
 *
 * @param point - Point potentially outside the convex hull
 * @param vertices - Lattice vertices
 * @returns Projected point on or inside the boundary
 */
function projectToConvexHull(point: Vector4D, vertices: LatticeVertex[]): Vector4D {
    // If already inside, return as-is
    if (isInsideConvexHull(point)) {
        return point;
    }

    // Find nearest vertex and project towards center
    const nearestIdx = findNearestVertex(point, vertices);
    const nearest = vertices[nearestIdx].coordinates;

    // Binary search along the line from origin to point
    // to find the boundary intersection
    let inside: Vector4D = [0, 0, 0, 0]; // Origin is always inside
    let outside = point;

    for (let i = 0; i < 20; i++) { // 20 iterations for good precision
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

/**
 * Scale a point to lie on the boundary of the 24-cell.
 * Preserves direction, adjusts magnitude.
 *
 * @param point - Point to scale
 * @returns Point scaled to boundary
 */
function scaleToHull(point: Vector4D): Vector4D {
    const norm = magnitude(point);
    if (norm < MATH_CONSTANTS.EPSILON) {
        return [1, 0, 0, 0]; // Default to a vertex
    }

    // Normalize and scale to circumradius
    return [
        point[0] / norm * CIRCUMRADIUS,
        point[1] / norm * CIRCUMRADIUS,
        point[2] / norm * CIRCUMRADIUS,
        point[3] / norm * CIRCUMRADIUS
    ];
}

// =============================================================================
// LATTICE24 CLASS
// =============================================================================

/**
 * The Lattice24 class encapsulates the 24-Cell geometry and operations.
 * It serves as the "Topological Governor" of the Chronomorphic Polytopal Engine.
 *
 * Usage:
 * ```typescript
 * const lattice = new Lattice24();
 * const result = lattice.checkConvexity([0.5, 0.5, 0, 0]);
 * console.log(`Coherence: ${result.coherence}`);
 * ```
 */
export class Lattice24 {
    /** The 24 vertices of the 24-cell */
    private readonly _vertices: LatticeVertex[];

    /** The 24 octahedral cells */
    private readonly _cells: LatticeCell[];

    /** Voronoi regions for each vertex */
    private readonly _voronoiRegions: Map<number, VoronoiRegion>;

    /** Cache for k-nearest lookups */
    private readonly _nearestCache: Map<string, number[]>;

    constructor() {
        // Generate vertices
        this._vertices = generate24CellVertices();
        computeNeighbors(this._vertices);

        // Generate cells
        this._cells = generate24CellCells(this._vertices);

        // Compute Voronoi regions
        this._voronoiRegions = new Map();
        for (let i = 0; i < this._vertices.length; i++) {
            this._voronoiRegions.set(i, computeVoronoiRegion(i, this._vertices));
        }

        // Initialize cache
        this._nearestCache = new Map();
    }

    // =========================================================================
    // ACCESSORS
    // =========================================================================

    /** Get all vertices */
    get vertices(): readonly LatticeVertex[] {
        return this._vertices;
    }

    /** Get vertex count */
    get vertexCount(): number {
        return NUM_VERTICES;
    }

    /** Get all cells */
    get cells(): readonly LatticeCell[] {
        return this._cells;
    }

    /** Get circumradius */
    get circumradius(): number {
        return CIRCUMRADIUS;
    }

    /** Get edge length */
    get edgeLength(): number {
        return EDGE_LENGTH;
    }

    // =========================================================================
    // LOOKUPS
    // =========================================================================

    /**
     * Get a vertex by ID.
     */
    getVertex(id: number): LatticeVertex | undefined {
        return this._vertices[id];
    }

    /**
     * Get Voronoi region for a vertex.
     */
    getVoronoiRegion(vertexId: number): VoronoiRegion | undefined {
        return this._voronoiRegions.get(vertexId);
    }

    /**
     * Find the nearest vertex to a point.
     */
    findNearest(point: Vector4D): number {
        return findNearestVertex(point, this._vertices);
    }

    /**
     * Find k-nearest vertices with caching.
     */
    findKNearest(point: Vector4D, k: number): number[] {
        // Create cache key (quantized to reduce cache size)
        const quantize = (v: number) => Math.round(v * 100) / 100;
        const key = `${quantize(point[0])},${quantize(point[1])},${quantize(point[2])},${quantize(point[3])},${k}`;

        const cached = this._nearestCache.get(key);
        if (cached) {
            return cached;
        }

        const result = findKNearestVertices(point, this._vertices, k);

        // Limit cache size
        if (this._nearestCache.size > 10000) {
            this._nearestCache.clear();
        }
        this._nearestCache.set(key, result);

        return result;
    }

    // =========================================================================
    // VALIDATION (EPISTAORTHOGNITION)
    // =========================================================================

    /**
     * Check if a point is inside the convex hull.
     */
    isInside(point: Vector4D): boolean {
        return isInsideConvexHull(point);
    }

    /**
     * Compute coherence score for a state.
     */
    computeCoherence(state: Vector4D, k: number = 4): number {
        return computeCoherence(state, this._vertices, k).coherence;
    }

    /**
     * Full convexity check (Epistaorthognition).
     */
    checkConvexity(state: Vector4D, k: number = 4): ConvexityResult {
        return checkConvexity(state, this._vertices, k);
    }

    // =========================================================================
    // PROJECTION AND CLAMPING
    // =========================================================================

    /**
     * Project a point to the convex hull boundary.
     */
    project(point: Vector4D): Vector4D {
        return projectToConvexHull(point, this._vertices);
    }

    /**
     * Clamp a point to be inside the convex hull.
     * Returns the point if inside, projected point otherwise.
     */
    clamp(point: Vector4D): Vector4D {
        if (isInsideConvexHull(point)) {
            return point;
        }
        return projectToConvexHull(point, this._vertices);
    }

    /**
     * Scale a point to the hull boundary.
     */
    scale(point: Vector4D): Vector4D {
        return scaleToHull(point);
    }

    // =========================================================================
    // NAVIGATION
    // =========================================================================

    /**
     * Get neighbor vertices of a vertex.
     */
    getNeighbors(vertexId: number): number[] {
        const vertex = this._vertices[vertexId];
        return vertex ? vertex.neighbors : [];
    }

    /**
     * Check if two vertices are neighbors (connected by edge).
     */
    areNeighbors(v1: number, v2: number): boolean {
        const vertex = this._vertices[v1];
        return vertex ? vertex.neighbors.includes(v2) : false;
    }

    /**
     * Compute geodesic distance (edge hops) between two vertices.
     * Uses BFS for shortest path.
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

        return Infinity; // Should never happen in connected lattice
    }

    // =========================================================================
    // UTILITY
    // =========================================================================

    /**
     * Get a random point inside the 24-cell.
     */
    randomInside(): Vector4D {
        // Sample until inside
        for (let attempt = 0; attempt < 1000; attempt++) {
            const point: Vector4D = [
                (Math.random() - 0.5) * 2 * CIRCUMRADIUS,
                (Math.random() - 0.5) * 2 * CIRCUMRADIUS,
                (Math.random() - 0.5) * 2 * CIRCUMRADIUS,
                (Math.random() - 0.5) * 2 * CIRCUMRADIUS
            ];

            if (isInsideConvexHull(point)) {
                return point;
            }
        }

        // Fallback: return origin
        return [0, 0, 0, 0];
    }

    /**
     * Get a random vertex.
     */
    randomVertex(): LatticeVertex {
        const idx = Math.floor(Math.random() * NUM_VERTICES);
        return this._vertices[idx];
    }

    /**
     * Clear the nearest-neighbor cache.
     */
    clearCache(): void {
        this._nearestCache.clear();
    }

    /**
     * Get lattice statistics.
     */
    getStats(): Record<string, number> {
        return {
            vertices: NUM_VERTICES,
            edges: NUM_EDGES,
            cells: this._cells.length,
            neighborsPerVertex: NEIGHBORS_PER_VERTEX,
            circumradius: CIRCUMRADIUS,
            edgeLength: EDGE_LENGTH,
            inradius: INRADIUS,
            cacheSize: this._nearestCache.size
        };
    }
}

// =============================================================================
// FACTORY AND EXPORTS
// =============================================================================

/** Singleton instance for shared use */
let _defaultLattice: Lattice24 | null = null;

/**
 * Get or create the default Lattice24 instance.
 */
export function getDefaultLattice(): Lattice24 {
    if (!_defaultLattice) {
        _defaultLattice = new Lattice24();
    }
    return _defaultLattice;
}

/**
 * Create a fresh Lattice24 instance.
 */
export function createLattice(): Lattice24 {
    return new Lattice24();
}

// Export helper functions for direct use
export {
    isInsideConvexHull,
    findNearestVertex,
    findKNearestVertices,
    computeCoherence,
    checkConvexity,
    projectToConvexHull,
    CIRCUMRADIUS,
    EDGE_LENGTH,
    INRADIUS,
    NUM_VERTICES,
    NUM_EDGES
};
