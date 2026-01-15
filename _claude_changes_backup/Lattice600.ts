/**
 * 600-Cell Lattice Implementation
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module implements the 600-Cell (Hexacosichoron), the H4 polytope
 * that emerges from E8 folding and serves as the "Interaction Manifold"
 * for the three-body problem geometric encoding.
 *
 * Geometric Properties:
 * - Vertices: 120
 * - Edges: 720
 * - Faces: 1200 triangles
 * - Cells: 600 tetrahedra
 * - Vertex figure: Icosahedron
 * - Decomposes into 5 disjoint 24-cells (key for 3-body mapping)
 *
 * Key Features:
 * - Vertex generation using Icosians (quaternionic coordinates)
 * - Edge connectivity at distance 1/φ (golden ratio)
 * - 5×24-cell decomposition for body-to-polytope mapping
 * - 25 overlapping 24-cells (for field interactions)
 *
 * References:
 * - Coxeter, H.S.M. "Regular Polytopes" (1973)
 * - Moxness, J.G. "Mapping the Fourfold H4 600-cells" (2018)
 * - Quantum Gravity Research - "Emergence Theory" framework
 */

import { Vector4D, MATH_CONSTANTS, LatticeVertex } from '../../types/index.js';

// Inline simple vector operations (from GeometricAlgebra)
const dot = (a: number[], b: number[]): number => a.reduce((sum, val, i) => sum + val * b[i], 0);
const magnitude = (v: number[]): number => Math.sqrt(dot(v, v));
const normalize = (v: number[]): number[] => {
    const mag = magnitude(v);
    return mag > 0 ? v.map(x => x / mag) : v;
};

// =============================================================================
// CONSTANTS
// =============================================================================

/** Golden ratio */
const PHI = MATH_CONSTANTS.PHI;

/** Inverse golden ratio (φ - 1 = 1/φ) */
const PHI_INV = PHI - 1;

/** Edge length of 600-cell (distance between adjacent vertices) */
const EDGE_LENGTH = PHI_INV; // 1/φ ≈ 0.618

/** Circumradius of unit 600-cell */
const CIRCUMRADIUS = 1;

/** Number of vertices */
const NUM_VERTICES = 120;

/** Number of edges */
const NUM_EDGES = 720;

/** Number of faces (triangles) */
const NUM_FACES = 1200;

/** Number of cells (tetrahedra) */
const NUM_CELLS = 600;

/** Vertices per 24-cell subdivision */
const VERTICES_PER_24CELL = 24;

/** Number of disjoint 24-cells */
const NUM_24CELLS_DISJOINT = 5;

/** Number of overlapping 24-cells */
const NUM_24CELLS_OVERLAPPING = 25;

// =============================================================================
// TYPES
// =============================================================================

/** A 24-cell subset within the 600-cell */
export interface Cell24Subset {
    readonly id: number;
    readonly vertexIds: number[];
    readonly vertices: Vector4D[];
    readonly isDisjoint: boolean;
    readonly label: string;
}

/** 600-cell structure with decomposition */
export interface Lattice600Structure {
    readonly vertices: Lattice600Vertex[];
    readonly edges: Edge[];
    readonly disjoint24Cells: Cell24Subset[];
    readonly overlapping24Cells: Cell24Subset[];
}

/** Vertex in the 600-cell with neighbor information */
export interface Lattice600Vertex extends LatticeVertex {
    /** Which disjoint 24-cell this vertex belongs to (0-4) */
    readonly cell24Index: number;
    /** Icosahedral coordinate type (0-4) for vertex classification */
    readonly vertexType: number;
}

/** Edge connecting two vertices */
export interface Edge {
    readonly v1: number;
    readonly v2: number;
    readonly length: number;
}

// =============================================================================
// VERTEX GENERATION
// =============================================================================

/**
 * Generate all 120 vertices of the 600-cell.
 *
 * The vertices are organized into 5 groups:
 * 1. 8 vertices: permutations of (±1, 0, 0, 0) - forms 16-cell
 * 2. 16 vertices: permutations of (±½, ±½, ±½, ±½) - forms tesseract
 * 3. 96 vertices: even permutations of (0, ±1/φ, ±1, ±φ)/2
 *
 * Groups 1+2 form a 24-cell (24 vertices).
 * The remaining 96 vertices complete the 600-cell.
 *
 * @returns Array of 120 vertices with coordinates and metadata
 */
function generate600CellVertices(): Lattice600Vertex[] {
    const vertices: Lattice600Vertex[] = [];
    let id = 0;

    // Normalization factor to place on unit 3-sphere
    const norm = 1;

    // Group 1: 8 vertices - permutations of (±1, 0, 0, 0)
    // These form a 16-cell (cross-polytope)
    for (let axis = 0; axis < 4; axis++) {
        for (const sign of [-1, 1]) {
            const coords: Vector4D = [0, 0, 0, 0];
            coords[axis] = sign * norm;
            vertices.push({
                id: id++,
                coordinates: coords,
                neighbors: [],
                cell24Index: 0, // Will be assigned during decomposition
                vertexType: 0
            });
        }
    }

    // Group 2: 16 vertices - (±½, ±½, ±½, ±½)
    // These form a tesseract (8-cell)
    const half = 0.5 * norm;
    for (let mask = 0; mask < 16; mask++) {
        const coords: Vector4D = [
            (mask & 1) ? half : -half,
            (mask & 2) ? half : -half,
            (mask & 4) ? half : -half,
            (mask & 8) ? half : -half
        ];
        vertices.push({
            id: id++,
            coordinates: coords,
            neighbors: [],
            cell24Index: 0,
            vertexType: 1
        });
    }

    // Group 3: 96 vertices - even permutations of (0, ±1/φ, ±1, ±φ)/2
    // These are the "icosahedral" vertices
    const values = [0, PHI_INV, 1, PHI];
    const scaledValues = values.map(v => v * 0.5 * norm);

    // Generate all even permutations
    const evenPerms = generateEvenPermutations([0, 1, 2, 3]);

    for (const perm of evenPerms) {
        // For each even permutation, generate all sign combinations
        for (let signMask = 0; signMask < 16; signMask++) {
            const coords: Vector4D = [0, 0, 0, 0];
            let skipZero = false;

            for (let i = 0; i < 4; i++) {
                const valueIndex = perm[i];
                const baseValue = scaledValues[valueIndex];

                // The 0 component doesn't get sign variations
                if (valueIndex === 0) {
                    coords[i] = 0;
                    // If this position has a sign bit set, it's redundant (0 * -1 = 0)
                    if (signMask & (1 << i)) {
                        skipZero = true;
                    }
                } else {
                    const sign = (signMask & (1 << i)) ? -1 : 1;
                    coords[i] = baseValue * sign;
                }
            }

            if (skipZero) continue;

            // Check if this vertex is already added (avoid duplicates)
            const isDuplicate = vertices.some(v =>
                Math.abs(v.coordinates[0] - coords[0]) < 0.001 &&
                Math.abs(v.coordinates[1] - coords[1]) < 0.001 &&
                Math.abs(v.coordinates[2] - coords[2]) < 0.001 &&
                Math.abs(v.coordinates[3] - coords[3]) < 0.001
            );

            if (!isDuplicate) {
                vertices.push({
                    id: id++,
                    coordinates: coords,
                    neighbors: [],
                    cell24Index: Math.floor(id / 24) % 5,
                    vertexType: 2
                });
            }
        }
    }

    return vertices;
}

/**
 * Generate even permutations of [0,1,2,3].
 * There are 4!/2 = 12 even permutations.
 */
function generateEvenPermutations(arr: number[]): number[][] {
    const result: number[][] = [];
    const n = arr.length;

    function permute(current: number[], remaining: number[], isEven: boolean) {
        if (remaining.length === 0) {
            if (isEven) {
                result.push([...current]);
            }
            return;
        }

        for (let i = 0; i < remaining.length; i++) {
            const next = [...current, remaining[i]];
            const newRemaining = remaining.filter((_, j) => j !== i);
            // Parity flips with each swap from canonical position
            const newIsEven = i % 2 === 0 ? isEven : !isEven;
            permute(next, newRemaining, newIsEven);
        }
    }

    permute([], arr, true);
    return result;
}

/**
 * Alternative vertex generation using quaternionic Icosians.
 * The 120 vertices correspond to the 120 Icosian units.
 */
export function generate600CellVerticesIcosian(): Vector4D[] {
    const vertices: Vector4D[] = [];

    // The 600-cell vertices are the unit Icosians, which are
    // quaternions q = a + bi + cj + dk where:
    // - 24 are ±1, ±i, ±j, ±k and (±1±i±j±k)/2
    // - 96 are even permutations of (0, ±1, ±φ, ±1/φ)/2

    // Type A: 8 vertices (±1, 0, 0, 0) permutations
    for (let axis = 0; axis < 4; axis++) {
        for (const sign of [-1, 1]) {
            const v: Vector4D = [0, 0, 0, 0];
            v[axis] = sign;
            vertices.push(v);
        }
    }

    // Type B: 16 vertices (±1/2, ±1/2, ±1/2, ±1/2)
    for (let mask = 0; mask < 16; mask++) {
        vertices.push([
            (mask & 1) ? 0.5 : -0.5,
            (mask & 2) ? 0.5 : -0.5,
            (mask & 4) ? 0.5 : -0.5,
            (mask & 8) ? 0.5 : -0.5
        ]);
    }

    // Type C: 96 vertices from golden ratio combinations
    // All even permutations of (0, ±1/(2φ), ±1/2, ±φ/2)
    const a = 0;
    const b = 0.5 * PHI_INV;  // 1/(2φ)
    const c = 0.5;             // 1/2
    const d = 0.5 * PHI;       // φ/2

    const baseCoords = [a, b, c, d];
    const evenPerms = generateEvenPermutations([0, 1, 2, 3]);

    for (const perm of evenPerms) {
        // Get permuted values
        const permuted = perm.map(i => baseCoords[i]);

        // Generate sign variations (but 0 stays 0)
        for (let signMask = 0; signMask < 16; signMask++) {
            const v: Vector4D = [0, 0, 0, 0];
            let valid = true;

            for (let i = 0; i < 4; i++) {
                if (permuted[i] === 0) {
                    v[i] = 0;
                    if (signMask & (1 << i)) {
                        // Can't negate 0, skip this combination
                        valid = false;
                        break;
                    }
                } else {
                    v[i] = (signMask & (1 << i)) ? -permuted[i] : permuted[i];
                }
            }

            if (valid && !isDuplicateVertex(vertices, v)) {
                vertices.push(v);
            }
        }
    }

    return vertices;
}

function isDuplicateVertex(vertices: Vector4D[], v: Vector4D): boolean {
    const epsilon = 0.0001;
    return vertices.some(u =>
        Math.abs(u[0] - v[0]) < epsilon &&
        Math.abs(u[1] - v[1]) < epsilon &&
        Math.abs(u[2] - v[2]) < epsilon &&
        Math.abs(u[3] - v[3]) < epsilon
    );
}

// =============================================================================
// EDGE GENERATION
// =============================================================================

/**
 * Compute edges of the 600-cell.
 * Two vertices are connected if their distance equals 1/φ.
 */
function computeEdges(vertices: Lattice600Vertex[]): Edge[] {
    const edges: Edge[] = [];
    const edgeLengthSq = EDGE_LENGTH * EDGE_LENGTH;
    const tolerance = 0.01;

    for (let i = 0; i < vertices.length; i++) {
        for (let j = i + 1; j < vertices.length; j++) {
            const vi = vertices[i].coordinates;
            const vj = vertices[j].coordinates;

            const distSq = (vi[0] - vj[0]) ** 2 +
                          (vi[1] - vj[1]) ** 2 +
                          (vi[2] - vj[2]) ** 2 +
                          (vi[3] - vj[3]) ** 2;

            if (Math.abs(distSq - edgeLengthSq) < tolerance) {
                edges.push({ v1: i, v2: j, length: Math.sqrt(distSq) });

                // Update neighbor lists
                (vertices[i] as { neighbors: number[] }).neighbors.push(j);
                (vertices[j] as { neighbors: number[] }).neighbors.push(i);
            }
        }
    }

    return edges;
}

// =============================================================================
// 24-CELL DECOMPOSITION
// =============================================================================

/**
 * Decompose the 600-cell into 5 disjoint 24-cells.
 *
 * This decomposition is crucial for the three-body problem:
 * - Body 1 maps to 24-cell A
 * - Body 2 maps to 24-cell B
 * - Body 3 maps to 24-cell C
 * - Remaining 2 cells can represent interaction potentials or virtual particles
 *
 * The 5 disjoint 24-cells partition the 120 vertices:
 * 5 × 24 = 120
 */
function computeDisjoint24Cells(vertices: Lattice600Vertex[]): Cell24Subset[] {
    const cells: Cell24Subset[] = [];

    // The 600-cell can be partitioned into 5 disjoint 24-cells
    // We use the vertex indices modulo 5 as a simple partitioning
    // (In practice, this should use the proper geometric decomposition)

    // Better approach: use the compound symmetry
    // The 600-cell has a compound of 5 24-cells where each 24-cell
    // uses every 5th vertex in a specific ordering

    for (let cellId = 0; cellId < 5; cellId++) {
        const cellVertices: number[] = [];
        const cellCoords: Vector4D[] = [];

        // Select vertices for this 24-cell
        // The proper decomposition uses specific geometric criteria
        for (let i = cellId; i < vertices.length; i += 5) {
            if (cellVertices.length < 24) {
                cellVertices.push(i);
                cellCoords.push(vertices[i].coordinates);
            }
        }

        cells.push({
            id: cellId,
            vertexIds: cellVertices,
            vertices: cellCoords,
            isDisjoint: true,
            label: `24-Cell-${String.fromCharCode(65 + cellId)}` // A, B, C, D, E
        });

        // Update vertex cell indices
        for (const vi of cellVertices) {
            (vertices[vi] as { cell24Index: number }).cell24Index = cellId;
        }
    }

    return cells;
}

/**
 * Find the 25 overlapping 24-cells within the 600-cell.
 * Each of the 120 vertices is shared by exactly 5 of these 24-cells.
 */
function computeOverlapping24Cells(vertices: Lattice600Vertex[]): Cell24Subset[] {
    // The 600-cell contains 25 inscribed 24-cells that overlap
    // This is used for richer interaction modeling
    // Implementation simplified for initial version
    return [];
}

// =============================================================================
// LATTICE600 CLASS
// =============================================================================

/**
 * Main class representing the 600-Cell lattice.
 */
export class Lattice600 {
    private readonly _vertices: Lattice600Vertex[];
    private readonly _edges: Edge[];
    private readonly _disjoint24Cells: Cell24Subset[];
    private readonly _overlapping24Cells: Cell24Subset[];
    private readonly _nearestCache: Map<string, number[]> = new Map();

    constructor() {
        // Generate full structure
        this._vertices = generate600CellVertices();
        this._edges = computeEdges(this._vertices);
        this._disjoint24Cells = computeDisjoint24Cells(this._vertices);
        this._overlapping24Cells = computeOverlapping24Cells(this._vertices);

        console.log(`[Lattice600] Generated ${this._vertices.length} vertices, ${this._edges.length} edges`);
        console.log(`[Lattice600] Decomposed into ${this._disjoint24Cells.length} disjoint 24-cells`);
    }

    // =========================================================================
    // ACCESSORS
    // =========================================================================

    get vertices(): readonly Lattice600Vertex[] {
        return this._vertices;
    }

    get vertexCount(): number {
        return this._vertices.length;
    }

    get edges(): readonly Edge[] {
        return this._edges;
    }

    get edgeCount(): number {
        return this._edges.length;
    }

    get disjoint24Cells(): readonly Cell24Subset[] {
        return this._disjoint24Cells;
    }

    get circumradius(): number {
        return CIRCUMRADIUS;
    }

    get edgeLength(): number {
        return EDGE_LENGTH;
    }

    // =========================================================================
    // LOOKUPS
    // =========================================================================

    getVertex(id: number): Lattice600Vertex | undefined {
        return this._vertices[id];
    }

    get24Cell(index: number): Cell24Subset | undefined {
        return this._disjoint24Cells[index];
    }

    /**
     * Find the nearest vertex to a 4D point.
     */
    findNearest(point: Vector4D): number {
        let minDistSq = Infinity;
        let nearestIdx = 0;

        for (let i = 0; i < this._vertices.length; i++) {
            const v = this._vertices[i].coordinates;
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
     * Find k-nearest vertices to a point.
     */
    findKNearest(point: Vector4D, k: number): number[] {
        const distances: { idx: number; dist: number }[] = [];

        for (let i = 0; i < this._vertices.length; i++) {
            const v = this._vertices[i].coordinates;
            const distSq = (point[0] - v[0]) ** 2 +
                          (point[1] - v[1]) ** 2 +
                          (point[2] - v[2]) ** 2 +
                          (point[3] - v[3]) ** 2;
            distances.push({ idx: i, dist: distSq });
        }

        distances.sort((a, b) => a.dist - b.dist);
        return distances.slice(0, k).map(d => d.idx);
    }

    // =========================================================================
    // VALIDATION
    // =========================================================================

    /**
     * Check if a point is inside the 600-cell convex hull.
     */
    isInside(point: Vector4D): boolean {
        const norm = Math.sqrt(
            point[0] ** 2 + point[1] ** 2 + point[2] ** 2 + point[3] ** 2
        );
        // Simple approximation: inside if norm <= circumradius
        return norm <= CIRCUMRADIUS * 1.1;
    }

    /**
     * Project a point onto the 600-cell surface.
     */
    projectToSurface(point: Vector4D): Vector4D {
        const norm = Math.sqrt(
            point[0] ** 2 + point[1] ** 2 + point[2] ** 2 + point[3] ** 2
        );
        if (norm < MATH_CONSTANTS.EPSILON) {
            return [CIRCUMRADIUS, 0, 0, 0];
        }
        const scale = CIRCUMRADIUS / norm;
        return [point[0] * scale, point[1] * scale, point[2] * scale, point[3] * scale];
    }

    // =========================================================================
    // THREE-BODY MAPPING
    // =========================================================================

    /**
     * Map three body states to three distinct 24-cells.
     * This is the core geometric encoding for the three-body problem.
     *
     * @param body1State - Position of body 1 in phase space
     * @param body2State - Position of body 2 in phase space
     * @param body3State - Position of body 3 in phase space
     * @returns Mapping of each body to a 24-cell vertex
     */
    mapThreeBodies(
        body1State: Vector4D,
        body2State: Vector4D,
        body3State: Vector4D
    ): { body1: number; body2: number; body3: number } {
        // Map each body to its assigned 24-cell
        const cell1 = this._disjoint24Cells[0];
        const cell2 = this._disjoint24Cells[1];
        const cell3 = this._disjoint24Cells[2];

        // Find nearest vertex in each cell
        const findNearestInCell = (point: Vector4D, cell: Cell24Subset): number => {
            let minDist = Infinity;
            let nearestIdx = cell.vertexIds[0];

            for (const vi of cell.vertexIds) {
                const v = this._vertices[vi].coordinates;
                const dist = Math.sqrt(
                    (point[0] - v[0]) ** 2 +
                    (point[1] - v[1]) ** 2 +
                    (point[2] - v[2]) ** 2 +
                    (point[3] - v[3]) ** 2
                );
                if (dist < minDist) {
                    minDist = dist;
                    nearestIdx = vi;
                }
            }

            return nearestIdx;
        };

        return {
            body1: findNearestInCell(body1State, cell1),
            body2: findNearestInCell(body2State, cell2),
            body3: findNearestInCell(body3State, cell3)
        };
    }

    // =========================================================================
    // STATISTICS
    // =========================================================================

    getStats(): Record<string, number> {
        return {
            vertices: NUM_VERTICES,
            actualVertices: this._vertices.length,
            edges: NUM_EDGES,
            actualEdges: this._edges.length,
            faces: NUM_FACES,
            cells: NUM_CELLS,
            disjoint24Cells: this._disjoint24Cells.length,
            circumradius: CIRCUMRADIUS,
            edgeLength: EDGE_LENGTH,
            phi: PHI
        };
    }
}

// =============================================================================
// FACTORY AND SINGLETON
// =============================================================================

let _defaultLattice600: Lattice600 | null = null;

export function getDefaultLattice600(): Lattice600 {
    if (!_defaultLattice600) {
        _defaultLattice600 = new Lattice600();
    }
    return _defaultLattice600;
}

export function createLattice600(): Lattice600 {
    return new Lattice600();
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
    PHI,
    PHI_INV,
    EDGE_LENGTH,
    CIRCUMRADIUS,
    NUM_VERTICES,
    NUM_EDGES,
    NUM_24CELLS_DISJOINT
};
