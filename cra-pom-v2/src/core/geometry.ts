// File: src/core/geometry.ts
// Geometric Cognition Kernel - Mathematical Core
// Implements Quaternion operations, 24-Cell lattice, and Cognitive Manifold

/**
 * Golden Ratio constant - used for ergodic (non-repeating) trajectories
 * phi = (1 + sqrt(5)) / 2
 */
export const PHI = (1 + Math.sqrt(5)) / 2;
export const PHI_INV = 1 / PHI;

/**
 * Safety/convexity tolerance margin for boundary detection
 */
export const CONVEXITY_TOLERANCE = 0.001;

/**
 * Quaternion class implementing Hamilton algebra
 * Represents a point in 4D space as (w, x, y, z) where w is the scalar part
 */
export class Quaternion {
  constructor(
    public readonly w: number,
    public readonly x: number,
    public readonly y: number,
    public readonly z: number
  ) {}

  /**
   * Create quaternion from array [w, x, y, z]
   */
  static fromArray(arr: readonly [number, number, number, number]): Quaternion {
    return new Quaternion(arr[0], arr[1], arr[2], arr[3]);
  }

  /**
   * Create identity quaternion (1, 0, 0, 0)
   */
  static identity(): Quaternion {
    return new Quaternion(1, 0, 0, 0);
  }

  /**
   * Create quaternion from axis-angle representation
   * @param axis - Normalized 3D vector [x, y, z]
   * @param angle - Rotation angle in radians
   */
  static fromAxisAngle(
    axis: readonly [number, number, number],
    angle: number
  ): Quaternion {
    const halfAngle = angle / 2;
    const sinHalf = Math.sin(halfAngle);
    const cosHalf = Math.cos(halfAngle);
    return new Quaternion(
      cosHalf,
      axis[0] * sinHalf,
      axis[1] * sinHalf,
      axis[2] * sinHalf
    );
  }

  /**
   * Create a purely imaginary quaternion (0, x, y, z)
   */
  static pure(x: number, y: number, z: number): Quaternion {
    return new Quaternion(0, x, y, z);
  }

  /**
   * Convert to array representation
   */
  toArray(): [number, number, number, number] {
    return [this.w, this.x, this.y, this.z];
  }

  /**
   * Hamilton product: this * other
   * (a1, b1, c1, d1) * (a2, b2, c2, d2)
   */
  multiply(other: Quaternion): Quaternion {
    return new Quaternion(
      this.w * other.w - this.x * other.x - this.y * other.y - this.z * other.z,
      this.w * other.x + this.x * other.w + this.y * other.z - this.z * other.y,
      this.w * other.y - this.x * other.z + this.y * other.w + this.z * other.x,
      this.w * other.z + this.x * other.y - this.y * other.x + this.z * other.w
    );
  }

  /**
   * Conjugate: (w, -x, -y, -z)
   */
  conjugate(): Quaternion {
    return new Quaternion(this.w, -this.x, -this.y, -this.z);
  }

  /**
   * Squared norm (magnitude squared): w² + x² + y² + z²
   */
  normSquared(): number {
    return this.w * this.w + this.x * this.x + this.y * this.y + this.z * this.z;
  }

  /**
   * Euclidean norm (magnitude): sqrt(w² + x² + y² + z²)
   */
  norm(): number {
    return Math.sqrt(this.normSquared());
  }

  /**
   * Normalize to unit quaternion
   */
  normalize(): Quaternion {
    const n = this.norm();
    if (n < 1e-10) {
      return Quaternion.identity();
    }
    return new Quaternion(this.w / n, this.x / n, this.y / n, this.z / n);
  }

  /**
   * Inverse: conjugate / normSquared
   */
  inverse(): Quaternion {
    const ns = this.normSquared();
    if (ns < 1e-10) {
      return Quaternion.identity();
    }
    const c = this.conjugate();
    return new Quaternion(c.w / ns, c.x / ns, c.y / ns, c.z / ns);
  }

  /**
   * Scalar multiplication
   */
  scale(s: number): Quaternion {
    return new Quaternion(this.w * s, this.x * s, this.y * s, this.z * s);
  }

  /**
   * Addition
   */
  add(other: Quaternion): Quaternion {
    return new Quaternion(
      this.w + other.w,
      this.x + other.x,
      this.y + other.y,
      this.z + other.z
    );
  }

  /**
   * Subtraction
   */
  subtract(other: Quaternion): Quaternion {
    return new Quaternion(
      this.w - other.w,
      this.x - other.x,
      this.y - other.y,
      this.z - other.z
    );
  }

  /**
   * Dot product (inner product)
   */
  dot(other: Quaternion): number {
    return (
      this.w * other.w + this.x * other.x + this.y * other.y + this.z * other.z
    );
  }

  /**
   * Euclidean distance to another quaternion
   */
  distanceTo(other: Quaternion): number {
    return this.subtract(other).norm();
  }

  /**
   * Linear interpolation (LERP)
   */
  lerp(other: Quaternion, t: number): Quaternion {
    return this.scale(1 - t).add(other.scale(t));
  }

  /**
   * Spherical linear interpolation (SLERP)
   */
  slerp(other: Quaternion, t: number): Quaternion {
    let dot = this.dot(other);
    let target = other;

    // Handle negative dot product (take shorter path)
    if (dot < 0) {
      target = other.scale(-1);
      dot = -dot;
    }

    // If very close, use linear interpolation
    if (dot > 0.9995) {
      return this.lerp(target, t).normalize();
    }

    const theta0 = Math.acos(dot);
    const theta = theta0 * t;
    const sinTheta = Math.sin(theta);
    const sinTheta0 = Math.sin(theta0);

    const s0 = Math.cos(theta) - (dot * sinTheta) / sinTheta0;
    const s1 = sinTheta / sinTheta0;

    return this.scale(s0).add(target.scale(s1));
  }

  /**
   * Check if approximately equal to another quaternion
   */
  equals(other: Quaternion, epsilon = 1e-10): boolean {
    return (
      Math.abs(this.w - other.w) < epsilon &&
      Math.abs(this.x - other.x) < epsilon &&
      Math.abs(this.y - other.y) < epsilon &&
      Math.abs(this.z - other.z) < epsilon
    );
  }

  /**
   * String representation for debugging
   */
  toString(): string {
    return `Quaternion(${this.w.toFixed(6)}, ${this.x.toFixed(6)}, ${this.y.toFixed(6)}, ${this.z.toFixed(6)})`;
  }

  /**
   * Get serializable state
   */
  toState(): QuaternionState {
    return {
      w: this.w,
      x: this.x,
      y: this.y,
      z: this.z,
      norm: this.norm(),
    };
  }
}

/**
 * Serializable quaternion state for TRACE logging
 */
export interface QuaternionState {
  w: number;
  x: number;
  y: number;
  z: number;
  norm: number;
}

/**
 * Lattice24 - The 24-Cell (D4 Lattice)
 *
 * The 24-cell is a regular 4D polytope with 24 vertices, 96 edges, 96 triangular
 * faces, and 24 octahedral cells. It is self-dual and has the symmetry of the
 * D4 root system.
 *
 * Vertices are all permutations of (±1, ±1, 0, 0)
 * Two vertices are connected by an edge if their distance is √2
 */
export class Lattice24 {
  public readonly vertices: readonly Quaternion[];
  public readonly edges: readonly [number, number][];
  public readonly radius: number;

  constructor() {
    this.vertices = this.generateVertices();
    this.edges = this.generateEdges();
    this.radius = Math.sqrt(2); // All vertices at distance √2 from origin
  }

  /**
   * Generate all 24 vertices of the 24-cell
   * All permutations of (±1, ±1, 0, 0)
   */
  private generateVertices(): Quaternion[] {
    const vertices: Quaternion[] = [];
    const values = [1, -1, 0, 0];

    // Generate all unique permutations of [1, -1, 0, 0] and [-1, 1, 0, 0]
    // and their sign variants
    const signs = [1, -1];

    // For each pair of positions to place ±1
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        // Try all sign combinations for the two non-zero entries
        for (const si of signs) {
          for (const sj of signs) {
            const coords: [number, number, number, number] = [0, 0, 0, 0];
            coords[i] = si;
            coords[j] = sj;
            vertices.push(new Quaternion(coords[0], coords[1], coords[2], coords[3]));
          }
        }
      }
    }

    return vertices;
  }

  /**
   * Generate edges connecting vertices at distance √2
   * In the 24-cell, vertices at distance √2 are connected
   */
  private generateEdges(): [number, number][] {
    const edges: [number, number][] = [];
    const targetDist = Math.sqrt(2);
    const epsilon = 0.001;

    for (let i = 0; i < this.vertices.length; i++) {
      for (let j = i + 1; j < this.vertices.length; j++) {
        const dist = this.vertices[i].distanceTo(this.vertices[j]);
        if (Math.abs(dist - targetDist) < epsilon) {
          edges.push([i, j]);
        }
      }
    }

    return edges;
  }

  /**
   * Get a vertex by index
   */
  getVertex(index: number): Quaternion {
    return this.vertices[index];
  }

  /**
   * Find the nearest vertex to a given point
   */
  findNearestVertex(point: Quaternion): { vertex: Quaternion; index: number; distance: number } {
    let minDist = Infinity;
    let nearestIdx = 0;

    for (let i = 0; i < this.vertices.length; i++) {
      const dist = point.distanceTo(this.vertices[i]);
      if (dist < minDist) {
        minDist = dist;
        nearestIdx = i;
      }
    }

    return {
      vertex: this.vertices[nearestIdx],
      index: nearestIdx,
      distance: minDist,
    };
  }

  /**
   * Check if a point is inside the convex hull of the 24-cell
   *
   * For the 24-cell with vertices at permutations of (±1, ±1, 0, 0),
   * a point (w, x, y, z) is inside if for every vertex V,
   * the dot product P · V ≤ 2 (the vertex norm squared)
   *
   * Equivalently, |w| + |x| + |y| + |z| ≤ 2 for this vertex arrangement
   */
  isInsideConvexHull(point: Quaternion): boolean {
    // The 24-cell can be characterized by the L1 norm constraint
    // |w| + |x| + |y| + |z| ≤ 2
    const l1Norm =
      Math.abs(point.w) + Math.abs(point.x) + Math.abs(point.y) + Math.abs(point.z);
    return l1Norm <= 2 + CONVEXITY_TOLERANCE;
  }

  /**
   * Calculate the penetration depth (negative if inside, positive if outside)
   */
  getPenetrationDepth(point: Quaternion): number {
    const l1Norm =
      Math.abs(point.w) + Math.abs(point.x) + Math.abs(point.y) + Math.abs(point.z);
    return l1Norm - 2;
  }

  /**
   * Project a point back onto the surface of the convex hull
   */
  projectOntoSurface(point: Quaternion): Quaternion {
    const l1Norm =
      Math.abs(point.w) + Math.abs(point.x) + Math.abs(point.y) + Math.abs(point.z);

    if (l1Norm < 1e-10) {
      return new Quaternion(1, 1, 0, 0).scale(1 / Math.sqrt(2));
    }

    // Scale to have L1 norm = 2
    const scale = 2 / l1Norm;
    return point.scale(scale);
  }

  /**
   * Get lattice state for logging
   */
  toState(): LatticeState {
    return {
      vertexCount: this.vertices.length,
      edgeCount: this.edges.length,
      radius: this.radius,
    };
  }
}

export interface LatticeState {
  vertexCount: number;
  edgeCount: number;
  radius: number;
}

/**
 * Convexity status enum
 */
export type ConvexityStatus = 'SAFE' | 'WARNING' | 'VIOLATION';

/**
 * Result of convexity check
 */
export interface ConvexityResult {
  status: ConvexityStatus;
  isInside: boolean;
  penetrationDepth: number;
  distanceFromOrigin: number;
  nearestVertex: {
    index: number;
    distance: number;
  };
}

/**
 * CognitiveManifold - The reasoning engine
 *
 * Implements geometric cognition through:
 * 1. Isoclinic rotations (Clifford translations) for movement
 * 2. Golden ratio in rotation axes for ergodic (non-repeating) trajectories
 * 3. Convexity constraints for "safety" (staying within the polytope)
 */
export class CognitiveManifold {
  private _position: Quaternion;
  private _leftRotor: Quaternion;
  private _rightRotor: Quaternion;
  private _stepCount: number;
  private readonly _lattice: Lattice24;

  /**
   * Warning threshold - distance from boundary
   */
  private readonly WARNING_THRESHOLD = 0.5;

  constructor(initialPosition?: Quaternion) {
    this._lattice = new Lattice24();
    this._position = initialPosition ?? new Quaternion(0.5, 0.5, 0.5, 0.5);
    this._stepCount = 0;

    // Initialize isoclinic rotation rotors using golden ratio for ergodicity
    this._leftRotor = this.createGoldenRotor('left');
    this._rightRotor = this.createGoldenRotor('right');
  }

  /**
   * Create a rotation quaternion using the golden ratio
   * This ensures non-repeating (ergodic) trajectories through 4D space
   */
  private createGoldenRotor(type: 'left' | 'right'): Quaternion {
    // Use golden ratio proportions in the rotation axis
    // This creates an irrational rotation that never exactly repeats
    const baseAngle = Math.PI / 12; // 15 degrees base rotation

    if (type === 'left') {
      // Left rotor: rotation in the WX-YZ planes
      const axis: [number, number, number] = [PHI_INV, 1, PHI];
      const axisNorm = Math.sqrt(
        axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
      );
      return Quaternion.fromAxisAngle(
        [axis[0] / axisNorm, axis[1] / axisNorm, axis[2] / axisNorm],
        baseAngle
      );
    } else {
      // Right rotor: rotation in complementary planes
      const axis: [number, number, number] = [1, PHI, PHI_INV];
      const axisNorm = Math.sqrt(
        axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
      );
      return Quaternion.fromAxisAngle(
        [axis[0] / axisNorm, axis[1] / axisNorm, axis[2] / axisNorm],
        baseAngle * PHI_INV
      );
    }
  }

  /**
   * Get current position
   */
  get position(): Quaternion {
    return this._position;
  }

  /**
   * Get the lattice
   */
  get lattice(): Lattice24 {
    return this._lattice;
  }

  /**
   * Get step count
   */
  get stepCount(): number {
    return this._stepCount;
  }

  /**
   * Perform an inference step - advance the thought vector through 4D space
   *
   * Uses double isoclinic rotation (Clifford translation):
   * P' = qL × P × qR
   *
   * This is distinct from ordinary 3D rotation because it operates
   * simultaneously in two orthogonal planes - a property unique to 4D.
   *
   * @param delta - Scaling factor for the rotation (default 1.0)
   * @param entropy - Optional entropy injection to perturb trajectory
   */
  inferenceStep(delta = 1.0, entropy?: [number, number, number, number]): ManifoldStepResult {
    this._stepCount++;

    // Scale the rotation by delta (uses SLERP for proper interpolation)
    const scaledLeft = Quaternion.identity().slerp(this._leftRotor, delta);
    const scaledRight = Quaternion.identity().slerp(this._rightRotor, delta);

    // Apply double isoclinic rotation: P' = qL × P × qR
    let newPosition = scaledLeft.multiply(this._position).multiply(scaledRight);

    // Apply entropy injection if provided
    if (entropy) {
      const entropyQ = new Quaternion(
        entropy[0] * 0.1,
        entropy[1] * 0.1,
        entropy[2] * 0.1,
        entropy[3] * 0.1
      );
      newPosition = newPosition.add(entropyQ);
    }

    // Check convexity before committing
    const convexityBefore = this.checkConvexity();
    this._position = newPosition;
    const convexityAfter = this.checkConvexity();

    return {
      step: this._stepCount,
      previousPosition: this._position,
      newPosition: newPosition,
      leftRotor: scaledLeft.toState(),
      rightRotor: scaledRight.toState(),
      delta,
      entropyApplied: entropy !== undefined,
      convexityBefore,
      convexityAfter,
    };
  }

  /**
   * Check if current position satisfies convexity constraint
   * This is the mathematical definition of "Safety"
   */
  checkConvexity(): ConvexityResult {
    const isInside = this._lattice.isInsideConvexHull(this._position);
    const penetrationDepth = this._lattice.getPenetrationDepth(this._position);
    const distanceFromOrigin = this._position.norm();
    const nearest = this._lattice.findNearestVertex(this._position);

    let status: ConvexityStatus;
    if (isInside && penetrationDepth < -this.WARNING_THRESHOLD) {
      status = 'SAFE';
    } else if (isInside) {
      status = 'WARNING';
    } else {
      status = 'VIOLATION';
    }

    return {
      status,
      isInside,
      penetrationDepth,
      distanceFromOrigin,
      nearestVertex: {
        index: nearest.index,
        distance: nearest.distance,
      },
    };
  }

  /**
   * Constrain position to stay within the convex hull
   * Call this after detecting a VIOLATION to restore safety
   */
  constrainToSafe(): Quaternion {
    if (!this._lattice.isInsideConvexHull(this._position)) {
      this._position = this._lattice.projectOntoSurface(this._position);
    }
    return this._position;
  }

  /**
   * Reset to initial state
   */
  reset(position?: Quaternion): void {
    this._position = position ?? new Quaternion(0.5, 0.5, 0.5, 0.5);
    this._stepCount = 0;
  }

  /**
   * Set custom rotors for controlled experiments
   */
  setRotors(left: Quaternion, right: Quaternion): void {
    this._leftRotor = left.normalize();
    this._rightRotor = right.normalize();
  }

  /**
   * Get current geometric state for TRACE logging
   */
  getGeometricState(): GeometricState {
    const convexity = this.checkConvexity();
    return {
      position: this._position.toState(),
      stepCount: this._stepCount,
      convexity,
      leftRotor: this._leftRotor.toState(),
      rightRotor: this._rightRotor.toState(),
      lattice: this._lattice.toState(),
    };
  }
}

/**
 * Result of a manifold inference step
 */
export interface ManifoldStepResult {
  step: number;
  previousPosition: Quaternion;
  newPosition: Quaternion;
  leftRotor: QuaternionState;
  rightRotor: QuaternionState;
  delta: number;
  entropyApplied: boolean;
  convexityBefore: ConvexityResult;
  convexityAfter: ConvexityResult;
}

/**
 * Complete geometric state for TRACE logging
 */
export interface GeometricState {
  position: QuaternionState;
  stepCount: number;
  convexity: ConvexityResult;
  leftRotor: QuaternionState;
  rightRotor: QuaternionState;
  lattice: LatticeState;
}
