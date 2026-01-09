// File: src/core/polytope.ts
// Concept Polytopes - Convex regions in high-dimensional semantic space
// Implements the core PPP principle: "concepts are convex polytopes"

import { Hypervector, DEFAULT_DIMENSION } from './hdc';

/**
 * Half-space constraint: ax ≤ b
 * Defines one face of a polytope
 */
export interface HalfSpace {
  normal: Float32Array; // The 'a' vector (points outward from polytope)
  offset: number; // The 'b' value
  name?: string; // Optional name for this constraint
}

/**
 * Result of containment test
 */
export interface ContainmentResult {
  isInside: boolean;
  distance: number; // Negative = inside, positive = outside
  nearestConstraint?: string;
  penetrationDepth: number; // How far inside/outside
}

/**
 * Convex Polytope - A concept represented as intersection of half-spaces
 *
 * In PPP, a concept like "DOG" is not a single point, but a convex region.
 * Any point inside this region "is a dog" (satisfies the concept).
 * The boundary facets represent the constraints that define the concept.
 *
 * Mathematical definition:
 * P = { x ∈ R^d : Ax ≤ b }
 * where A is a matrix of constraint normals and b is the offset vector
 */
export class ConvexPolytope {
  readonly name: string;
  readonly dimension: number;
  readonly constraints: HalfSpace[];
  readonly centroid: Float32Array;
  readonly vertices: Float32Array[]; // Approximate vertices (for visualization)

  constructor(
    name: string,
    dimension: number,
    constraints: HalfSpace[],
    centroid?: Float32Array,
    vertices?: Float32Array[]
  ) {
    this.name = name;
    this.dimension = dimension;
    this.constraints = constraints;
    this.centroid = centroid ?? this.computeCentroid();
    this.vertices = vertices ?? [];
  }

  /**
   * Create a polytope from a prototype vector and radius
   * This creates a hypersphere-like region (technically a cross-polytope approximation)
   */
  static fromPrototype(
    name: string,
    prototype: Hypervector,
    radius: number = 0.5
  ): ConvexPolytope {
    const dimension = prototype.dimension;
    const constraints: HalfSpace[] = [];

    // Create constraints for an L∞ ball (hypercube) centered at prototype
    // This is simpler than a true L2 ball but still convex
    for (let i = 0; i < Math.min(dimension, 100); i++) {
      // Limit constraints for efficiency
      // Upper bound: x_i ≤ prototype_i + radius
      constraints.push({
        normal: createUnitVector(dimension, i),
        offset: prototype.data[i] + radius,
        name: `upper_${i}`,
      });
      // Lower bound: -x_i ≤ -prototype_i + radius, i.e., x_i ≥ prototype_i - radius
      const negUnit = createUnitVector(dimension, i);
      for (let j = 0; j < dimension; j++) negUnit[j] *= -1;
      constraints.push({
        normal: negUnit,
        offset: -prototype.data[i] + radius,
        name: `lower_${i}`,
      });
    }

    return new ConvexPolytope(name, dimension, constraints, Float32Array.from(prototype.data));
  }

  /**
   * Create a polytope from multiple example vectors (convex hull approach)
   * The polytope encompasses all examples
   */
  static fromExamples(name: string, examples: Hypervector[], margin: number = 0.1): ConvexPolytope {
    if (examples.length === 0) {
      throw new Error('Need at least one example');
    }

    const dimension = examples[0].dimension;

    // Compute centroid
    const centroid = new Float32Array(dimension);
    for (const ex of examples) {
      for (let i = 0; i < dimension; i++) {
        centroid[i] += ex.data[i];
      }
    }
    for (let i = 0; i < dimension; i++) {
      centroid[i] /= examples.length;
    }

    // Create constraints based on bounding box with margin
    const constraints: HalfSpace[] = [];
    const numDims = Math.min(dimension, 100); // Limit for efficiency

    for (let i = 0; i < numDims; i++) {
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (const ex of examples) {
        minVal = Math.min(minVal, ex.data[i]);
        maxVal = Math.max(maxVal, ex.data[i]);
      }

      // Add margin
      minVal -= margin;
      maxVal += margin;

      // Upper bound
      constraints.push({
        normal: createUnitVector(dimension, i),
        offset: maxVal,
        name: `max_${i}`,
      });

      // Lower bound
      const negUnit = createUnitVector(dimension, i);
      for (let j = 0; j < dimension; j++) negUnit[j] *= -1;
      constraints.push({
        normal: negUnit,
        offset: -minVal,
        name: `min_${i}`,
      });
    }

    const vertices = examples.map((e) => Float32Array.from(e.data));
    return new ConvexPolytope(name, dimension, constraints, centroid, vertices);
  }

  /**
   * Compute centroid (average of constraint offsets, simplified)
   */
  private computeCentroid(): Float32Array {
    // For a well-defined polytope, centroid would require linear programming
    // This is a simplified version assuming balanced constraints
    const centroid = new Float32Array(this.dimension);
    return centroid;
  }

  /**
   * Check if a point is inside this polytope
   * A point is inside if it satisfies ALL half-space constraints: Ax ≤ b
   */
  containsPoint(point: Float32Array | Hypervector): ContainmentResult {
    const data = point instanceof Hypervector ? point.data : point;

    if (data.length !== this.dimension) {
      throw new Error('Dimension mismatch');
    }

    let maxViolation = -Infinity;
    let violatingConstraint: string | undefined;

    for (const constraint of this.constraints) {
      // Compute dot product: a · x
      let dotProduct = 0;
      for (let i = 0; i < this.dimension; i++) {
        dotProduct += constraint.normal[i] * data[i];
      }

      // Check constraint: a · x ≤ b
      // Violation = a · x - b (positive means outside)
      const violation = dotProduct - constraint.offset;

      if (violation > maxViolation) {
        maxViolation = violation;
        violatingConstraint = constraint.name;
      }
    }

    return {
      isInside: maxViolation <= 0,
      distance: maxViolation,
      nearestConstraint: violatingConstraint,
      penetrationDepth: -maxViolation,
    };
  }

  /**
   * Project a point onto this polytope (find nearest point inside)
   * Uses iterative projection onto half-spaces
   */
  project(point: Float32Array | Hypervector, maxIterations: number = 50): Float32Array {
    const data = point instanceof Hypervector ? Float32Array.from(point.data) : Float32Array.from(point);

    for (let iter = 0; iter < maxIterations; iter++) {
      let maxViolation = 0;
      let worstConstraint: HalfSpace | null = null;

      for (const constraint of this.constraints) {
        let dotProduct = 0;
        for (let i = 0; i < this.dimension; i++) {
          dotProduct += constraint.normal[i] * data[i];
        }
        const violation = dotProduct - constraint.offset;

        if (violation > maxViolation) {
          maxViolation = violation;
          worstConstraint = constraint;
        }
      }

      if (maxViolation <= 1e-6 || !worstConstraint) {
        break; // Already inside
      }

      // Project onto the violated half-space
      // x_new = x - (a · x - b) * a / ||a||²
      let normSq = 0;
      for (let i = 0; i < this.dimension; i++) {
        normSq += worstConstraint.normal[i] * worstConstraint.normal[i];
      }

      const scale = maxViolation / normSq;
      for (let i = 0; i < this.dimension; i++) {
        data[i] -= scale * worstConstraint.normal[i];
      }
    }

    return data;
  }

  /**
   * Compute distance from point to polytope boundary
   * Negative = inside, Positive = outside
   */
  distanceToBoundary(point: Float32Array | Hypervector): number {
    return this.containsPoint(point).distance;
  }

  /**
   * Get the "membership strength" - how centrally located is the point
   * Returns 0-1, where 1 = at centroid, 0 = at boundary or outside
   */
  membershipStrength(point: Float32Array | Hypervector): number {
    const result = this.containsPoint(point);
    if (!result.isInside) return 0;

    // Compute distance to centroid
    const data = point instanceof Hypervector ? point.data : point;
    let distToCentroid = 0;
    for (let i = 0; i < this.dimension; i++) {
      const diff = data[i] - this.centroid[i];
      distToCentroid += diff * diff;
    }
    distToCentroid = Math.sqrt(distToCentroid);

    // Normalize by penetration depth
    const maxDist = Math.abs(result.penetrationDepth) + distToCentroid;
    if (maxDist === 0) return 1;

    return 1 - distToCentroid / maxDist;
  }
}

/**
 * Helper: create a unit vector with 1 at index i
 */
function createUnitVector(dimension: number, index: number): Float32Array {
  const v = new Float32Array(dimension);
  v[index] = 1;
  return v;
}

/**
 * Voronoi Cell - A polytope defined by proximity to a prototype
 * Part of a Voronoi tessellation of the semantic space
 */
export class VoronoiCell extends ConvexPolytope {
  readonly prototype: Hypervector;

  constructor(name: string, prototype: Hypervector, neighbors: Map<string, Hypervector>) {
    // Create half-space constraints from bisecting hyperplanes
    const constraints: HalfSpace[] = [];

    for (const [neighborName, neighborProto] of neighbors) {
      // The bisecting hyperplane between prototype and neighbor
      // Normal points from prototype toward neighbor
      // Equation: (neighbor - prototype) · (x - midpoint) ≤ 0

      const normal = new Float32Array(prototype.dimension);
      const midpoint = new Float32Array(prototype.dimension);
      let normalNorm = 0;

      for (let i = 0; i < prototype.dimension; i++) {
        normal[i] = neighborProto.data[i] - prototype.data[i];
        midpoint[i] = (prototype.data[i] + neighborProto.data[i]) / 2;
        normalNorm += normal[i] * normal[i];
      }

      // Normalize the normal
      normalNorm = Math.sqrt(normalNorm);
      if (normalNorm > 0) {
        for (let i = 0; i < prototype.dimension; i++) {
          normal[i] /= normalNorm;
        }
      }

      // Compute offset: normal · midpoint
      let offset = 0;
      for (let i = 0; i < prototype.dimension; i++) {
        offset += normal[i] * midpoint[i];
      }

      constraints.push({
        normal,
        offset,
        name: `bisector_${neighborName}`,
      });
    }

    super(name, prototype.dimension, constraints, Float32Array.from(prototype.data));
    this.prototype = prototype;
  }
}

/**
 * Voronoi Tessellation - Partitions semantic space into concept regions
 * Each concept is the region of points closer to its prototype than any other
 *
 * This implements the PPP principle of geometric categorization:
 * A query belongs to whichever concept's Voronoi cell contains it.
 */
export class VoronoiTessellation {
  readonly dimension: number;
  readonly prototypes: Map<string, Hypervector>;
  private cells: Map<string, VoronoiCell>;

  constructor(dimension: number = DEFAULT_DIMENSION) {
    this.dimension = dimension;
    this.prototypes = new Map();
    this.cells = new Map();
  }

  /**
   * Add a concept prototype
   */
  addConcept(name: string, prototype: Hypervector): void {
    if (prototype.dimension !== this.dimension) {
      throw new Error('Dimension mismatch');
    }
    this.prototypes.set(name, prototype);
    this.rebuildCells();
  }

  /**
   * Add multiple concepts at once
   */
  addConcepts(concepts: Map<string, Hypervector>): void {
    for (const [name, proto] of concepts) {
      if (proto.dimension !== this.dimension) {
        throw new Error(`Dimension mismatch for concept ${name}`);
      }
      this.prototypes.set(name, proto);
    }
    this.rebuildCells();
  }

  /**
   * Rebuild Voronoi cells after adding concepts
   */
  private rebuildCells(): void {
    this.cells.clear();

    for (const [name, prototype] of this.prototypes) {
      // Get all other prototypes as neighbors
      const neighbors = new Map<string, Hypervector>();
      for (const [otherName, otherProto] of this.prototypes) {
        if (otherName !== name) {
          neighbors.set(otherName, otherProto);
        }
      }

      this.cells.set(name, new VoronoiCell(name, prototype, neighbors));
    }
  }

  /**
   * Classify a query point - which concept does it belong to?
   * Uses simple nearest-neighbor (equivalent to Voronoi cell membership)
   */
  classify(query: Hypervector): {
    concept: string;
    similarity: number;
    membership: number;
    alternatives: Array<{ concept: string; similarity: number }>;
  } {
    if (this.prototypes.size === 0) {
      throw new Error('No concepts defined');
    }

    const similarities: Array<{ concept: string; similarity: number }> = [];

    for (const [name, prototype] of this.prototypes) {
      similarities.push({
        concept: name,
        similarity: query.cosineSimilarity(prototype),
      });
    }

    // Sort by similarity (descending)
    similarities.sort((a, b) => b.similarity - a.similarity);

    const best = similarities[0];
    const cell = this.cells.get(best.concept)!;

    return {
      concept: best.concept,
      similarity: best.similarity,
      membership: cell.membershipStrength(query),
      alternatives: similarities.slice(1, 4), // Top 3 alternatives
    };
  }

  /**
   * Get the Voronoi cell for a concept
   */
  getCell(name: string): VoronoiCell | undefined {
    return this.cells.get(name);
  }

  /**
   * Get all concept names
   */
  getConceptNames(): string[] {
    return Array.from(this.prototypes.keys());
  }

  /**
   * Get the number of concepts
   */
  get size(): number {
    return this.prototypes.size;
  }
}

/**
 * Polytope Bundle - A collection of related polytopes forming a "concept family"
 * Used for representing taxonomies and hierarchies
 */
export class PolytopeBundle {
  readonly name: string;
  readonly polytopes: Map<string, ConvexPolytope>;
  readonly hierarchy: Map<string, string[]>; // parent -> children

  constructor(name: string) {
    this.name = name;
    this.polytopes = new Map();
    this.hierarchy = new Map();
  }

  /**
   * Add a polytope to the bundle
   */
  add(polytope: ConvexPolytope, parent?: string): void {
    this.polytopes.set(polytope.name, polytope);

    if (parent) {
      if (!this.hierarchy.has(parent)) {
        this.hierarchy.set(parent, []);
      }
      this.hierarchy.get(parent)!.push(polytope.name);
    }
  }

  /**
   * Get all polytopes that contain a point
   */
  findContaining(point: Hypervector): ConvexPolytope[] {
    const containing: ConvexPolytope[] = [];

    for (const polytope of this.polytopes.values()) {
      if (polytope.containsPoint(point).isInside) {
        containing.push(polytope);
      }
    }

    return containing;
  }

  /**
   * Get the most specific (deepest in hierarchy) polytope containing a point
   */
  classifyHierarchical(point: Hypervector): ConvexPolytope | null {
    const containing = this.findContaining(point);
    if (containing.length === 0) return null;

    // Find the one with no children also containing the point
    for (const polytope of containing) {
      const children = this.hierarchy.get(polytope.name) ?? [];
      const hasContainingChild = children.some((childName) => {
        const child = this.polytopes.get(childName);
        return child && child.containsPoint(point).isInside;
      });

      if (!hasContainingChild) {
        return polytope;
      }
    }

    return containing[0];
  }
}

export default { ConvexPolytope, VoronoiCell, VoronoiTessellation, PolytopeBundle };
