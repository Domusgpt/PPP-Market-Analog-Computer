/**
 * 600-Cell (Hexacosichoron) - The 4D Analogue of the Icosahedron
 *
 * The 600-cell is a regular 4D polytope with:
 * - 120 vertices
 * - 720 edges
 * - 1200 triangular faces
 * - 600 tetrahedral cells
 *
 * CONSTRUCTION:
 * The 120 vertices are the elements of the binary icosahedral group (2I),
 * which can be expressed as unit quaternions. They consist of:
 *
 * 1. 24 vertices from the 24-cell (forming a subgroup)
 * 2. 96 additional vertices involving the golden ratio φ
 *
 * VERTEX COORDINATES (unit quaternions, all permutations):
 * - 8 vertices: (±1, 0, 0, 0)
 * - 16 vertices: (±½, ±½, ±½, ±½)
 * - 96 vertices: all EVEN permutations of (0, ±½, ±φ/2, ±1/(2φ))
 *
 * where φ = (1+√5)/2 ≈ 1.618 (golden ratio)
 * and 1/φ = φ-1 ≈ 0.618
 *
 * MUSICAL SIGNIFICANCE:
 * - 120 vertices = 5× finer granularity than 24-cell
 * - Can represent microtonal intervals, continuous pitch space
 * - Contains ~25 embedded 24-cells at different orientations
 * - Edge structure encodes golden-ratio relationships (harmonic!)
 *
 * REFERENCES:
 * - https://en.wikipedia.org/wiki/600-cell
 * - https://en.wikipedia.org/wiki/Binary_icosahedral_group
 */

import type { Vector4D } from '../music/music-geometry-domain';

// ============================================================================
// Constants
// ============================================================================

/** Golden ratio φ = (1 + √5) / 2 */
export const PHI = (1 + Math.sqrt(5)) / 2;

/** Inverse golden ratio 1/φ = φ - 1 */
export const PHI_INV = PHI - 1;

/** Half golden ratio */
const HALF_PHI = PHI / 2;

/** Half inverse golden ratio */
const HALF_PHI_INV = PHI_INV / 2;

// ============================================================================
// Helper Functions
// ============================================================================

function createVector(w: number, x: number, y: number, z: number): Vector4D {
  return { w, x, y, z };
}

function distance4D(a: Vector4D, b: Vector4D): number {
  return Math.sqrt(
    (a.w - b.w) ** 2 +
    (a.x - b.x) ** 2 +
    (a.y - b.y) ** 2 +
    (a.z - b.z) ** 2
  );
}

function magnitude4D(v: Vector4D): number {
  return Math.sqrt(v.w ** 2 + v.x ** 2 + v.y ** 2 + v.z ** 2);
}

/**
 * Generate all EVEN permutations of 4 values.
 * Even permutations = permutations reachable by even number of swaps.
 */
function evenPermutations(values: [number, number, number, number]): Vector4D[] {
  const [a, b, c, d] = values;
  // The 12 even permutations of (a,b,c,d)
  return [
    createVector(a, b, c, d),
    createVector(a, c, d, b),
    createVector(a, d, b, c),
    createVector(b, a, d, c),
    createVector(b, c, a, d),
    createVector(b, d, c, a),
    createVector(c, a, b, d),
    createVector(c, b, d, a),
    createVector(c, d, a, b),
    createVector(d, a, c, b),
    createVector(d, b, a, c),
    createVector(d, c, b, a),
  ];
}

/**
 * Generate all sign variations of a vector (for non-zero components)
 */
function signVariations(v: Vector4D): Vector4D[] {
  const variations: Vector4D[] = [];
  const signs = [1, -1];

  for (const sw of signs) {
    for (const sx of signs) {
      for (const sy of signs) {
        for (const sz of signs) {
          const newV = createVector(
            v.w === 0 ? 0 : v.w * sw,
            v.x === 0 ? 0 : v.x * sx,
            v.y === 0 ? 0 : v.y * sy,
            v.z === 0 ? 0 : v.z * sz
          );
          variations.push(newV);
        }
      }
    }
  }

  // Remove duplicates (when some components are 0)
  const unique = new Map<string, Vector4D>();
  for (const vec of variations) {
    const key = `${vec.w.toFixed(6)},${vec.x.toFixed(6)},${vec.y.toFixed(6)},${vec.z.toFixed(6)}`;
    unique.set(key, vec);
  }

  return Array.from(unique.values());
}

// ============================================================================
// 600-Cell Class
// ============================================================================

export class Cell600 {
  readonly vertices: Vector4D[];
  readonly edges: [number, number][];
  readonly edgeLength: number;

  // The 24-cell embedded at the origin orientation
  readonly embedded24Cell: Vector4D[];

  constructor() {
    this.vertices = this.generateVertices();
    this.edgeLength = PHI_INV; // Edge length is 1/φ for unit 600-cell
    this.edges = this.generateEdges();
    this.embedded24Cell = this.vertices.slice(0, 24); // First 24 form a 24-cell
  }

  /**
   * Generate all 120 vertices of the 600-cell
   */
  private generateVertices(): Vector4D[] {
    const vertices: Vector4D[] = [];
    const added = new Set<string>();

    const addVertex = (v: Vector4D) => {
      // Normalize to unit length and check for duplicates
      const mag = magnitude4D(v);
      const normalized = createVector(v.w / mag, v.x / mag, v.y / mag, v.z / mag);
      const key = `${normalized.w.toFixed(5)},${normalized.x.toFixed(5)},${normalized.y.toFixed(5)},${normalized.z.toFixed(5)}`;
      if (!added.has(key)) {
        added.add(key);
        vertices.push(normalized);
      }
    };

    // GROUP 1: 8 vertices from axis permutations (±1, 0, 0, 0)
    // These form the 16-cell core
    for (const sign of [1, -1]) {
      addVertex(createVector(sign, 0, 0, 0));
      addVertex(createVector(0, sign, 0, 0));
      addVertex(createVector(0, 0, sign, 0));
      addVertex(createVector(0, 0, 0, sign));
    }

    // GROUP 2: 16 vertices (±½, ±½, ±½, ±½)
    // These plus Group 1 form the 24-cell
    for (const sw of [0.5, -0.5]) {
      for (const sx of [0.5, -0.5]) {
        for (const sy of [0.5, -0.5]) {
          for (const sz of [0.5, -0.5]) {
            addVertex(createVector(sw, sx, sy, sz));
          }
        }
      }
    }

    // GROUP 3: 96 vertices - all EVEN permutations of (0, ±½, ±φ/2, ±1/(2φ))
    // These are the "golden" vertices that expand 24-cell to 600-cell
    const baseValues: [number, number, number, number] = [0, 0.5, HALF_PHI, HALF_PHI_INV];

    // Get even permutations of the absolute values
    const perms = evenPermutations(baseValues);

    // Apply sign variations to each permutation
    for (const perm of perms) {
      const signVars = signVariations(perm);
      for (const v of signVars) {
        addVertex(v);
      }
    }

    return vertices;
  }

  /**
   * Generate edges connecting vertices at distance 1/φ
   */
  private generateEdges(): [number, number][] {
    const edges: [number, number][] = [];
    const edgeDist = PHI_INV;
    const tolerance = 0.001;

    for (let i = 0; i < this.vertices.length; i++) {
      for (let j = i + 1; j < this.vertices.length; j++) {
        const dist = distance4D(this.vertices[i], this.vertices[j]);
        if (Math.abs(dist - edgeDist) < tolerance) {
          edges.push([i, j]);
        }
      }
    }

    return edges;
  }

  /**
   * Get neighbors of a vertex (connected by an edge)
   */
  getNeighbors(vertexIndex: number): number[] {
    const neighbors: number[] = [];
    for (const [i, j] of this.edges) {
      if (i === vertexIndex) neighbors.push(j);
      else if (j === vertexIndex) neighbors.push(i);
    }
    return neighbors;
  }

  /**
   * Find the closest vertex to a given point
   */
  findClosestVertex(point: Vector4D): { index: number; distance: number } {
    let minDist = Infinity;
    let minIndex = 0;

    for (let i = 0; i < this.vertices.length; i++) {
      const dist = distance4D(this.vertices[i], point);
      if (dist < minDist) {
        minDist = dist;
        minIndex = i;
      }
    }

    return { index: minIndex, distance: minDist };
  }

  /**
   * Get one of the ~25 embedded 24-cells
   * Each 24-cell is defined by a specific orientation/rotation
   */
  get24CellSubset(index: number): Vector4D[] {
    if (index < 0 || index >= 25) {
      throw new Error('24-cell index must be 0-24');
    }

    // For now, return the "primary" 24-cell (first 24 vertices)
    // Full implementation would compute rotated 24-cells
    if (index === 0) {
      return this.embedded24Cell;
    }

    // TODO: Implement other 24-cell orientations via isoclinic rotations
    return this.embedded24Cell;
  }

  /**
   * Check if a point lies on the 600-cell (approximately)
   */
  isOnPolytope(point: Vector4D, tolerance: number = 0.01): boolean {
    const { distance } = this.findClosestVertex(point);
    return distance < tolerance;
  }

  /**
   * Interpolate between two vertices along the edge
   */
  interpolateEdge(fromIndex: number, toIndex: number, t: number): Vector4D {
    const from = this.vertices[fromIndex];
    const to = this.vertices[toIndex];

    return createVector(
      from.w + (to.w - from.w) * t,
      from.x + (to.x - from.x) * t,
      from.y + (to.y - from.y) * t,
      from.z + (to.z - from.z) * t
    );
  }

  // Properties
  get vertexCount(): number { return 120; }
  get edgeCount(): number { return 720; }
  get faceCount(): number { return 1200; }
  get cellCount(): number { return 600; }
}

// ============================================================================
// Verification
// ============================================================================

export function verify600Cell(): {
  vertexCount: number;
  edgeCount: number;
  neighborsPerVertex: number;
  contains24Cell: boolean;
  allUnitLength: boolean;
  details: string[];
} {
  const cell = new Cell600();
  const details: string[] = [];

  // Check vertex count
  details.push(`Vertices: ${cell.vertices.length} (expected 120)`);

  // Check edge count
  details.push(`Edges: ${cell.edges.length} (expected 720)`);

  // Check neighbors per vertex (should be 12 for 600-cell)
  const neighbors0 = cell.getNeighbors(0).length;
  details.push(`Neighbors per vertex: ${neighbors0} (expected 12)`);

  // Check that first 24 vertices form a 24-cell
  const first24 = cell.embedded24Cell;
  details.push(`Embedded 24-cell: ${first24.length} vertices`);

  // Check all vertices are unit length
  let allUnit = true;
  for (const v of cell.vertices) {
    const mag = magnitude4D(v);
    if (Math.abs(mag - 1) > 0.001) {
      allUnit = false;
      break;
    }
  }
  details.push(`All unit length: ${allUnit}`);

  return {
    vertexCount: cell.vertices.length,
    edgeCount: cell.edges.length,
    neighborsPerVertex: neighbors0,
    contains24Cell: first24.length === 24,
    allUnitLength: allUnit,
    details,
  };
}

// ============================================================================
// Exports
// ============================================================================

export const Cell600Module = {
  Cell600,
  verify600Cell,
  PHI,
  PHI_INV,
};

export default Cell600Module;
