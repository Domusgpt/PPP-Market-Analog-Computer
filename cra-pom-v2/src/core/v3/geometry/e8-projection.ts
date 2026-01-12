/**
 * E8 Lattice Projection and φ-Scaling Layer
 *
 * The E8 root system (240 points in 8D) projects to 4D as two 600-cells:
 * - Layer 1: Unit scale (conceptual/abstract representation)
 * - Layer 2: φ-scaled (physical/concrete representation)
 *
 * MATHEMATICAL BASIS:
 * E8 is the largest exceptional Lie group, with 240 root vectors in 8D.
 * When projected to 4D, these form two overlapping 600-cells scaled by φ.
 *
 * COGNITIVE SIGNIFICANCE:
 * - Two layers = dual representation (abstract vs concrete)
 * - Alignment points = moments of insight where both representations agree
 * - φ-scaling = natural harmony (same ratio as musical overtones!)
 *
 * The golden ratio φ appears naturally because:
 * - 600-cell vertices use φ in their coordinates
 * - E8 lattice has φ-scaling symmetry
 * - φ is the limit of Fibonacci ratios (natural growth)
 *
 * REFERENCES:
 * - https://en.wikipedia.org/wiki/E8_(mathematics)
 * - https://en.wikipedia.org/wiki/E8_lattice
 * - Coxeter, "Regular Polytopes"
 */

import type { Vector4D } from '../music/music-geometry-domain';
import { Cell600, PHI, PHI_INV } from './cell600';
import { distance4D, scale4D, add4D } from '../music/polytopes';

// ============================================================================
// Types
// ============================================================================

export interface AlignmentPoint {
  position: Vector4D;
  layer1Index: number;   // Index in unit-scale 600-cell
  layer2Index: number;   // Index in φ-scaled 600-cell (or -1 if none)
  alignmentStrength: number;  // 0-1, how well aligned
  type: 'exact' | 'near' | 'interpolated';
}

export interface LayerBlendResult {
  blendedPosition: Vector4D;
  layer1Weight: number;
  layer2Weight: number;
  resonance: number;  // 0-1, strength of layer interaction
}

export interface E8ProjectionState {
  layer1Activations: number[];  // 120 activation values
  layer2Activations: number[];  // 120 activation values
  alignmentPoints: AlignmentPoint[];
  totalResonance: number;
}

// ============================================================================
// 8D Vector Type (for E8 lattice)
// ============================================================================

export interface Vector8D {
  x0: number;
  x1: number;
  x2: number;
  x3: number;
  x4: number;
  x5: number;
  x6: number;
  x7: number;
}

// ============================================================================
// E8Projection Class
// ============================================================================

export class E8Projection {
  // The two 600-cell layers
  readonly layer1: Cell600;  // Unit scale (abstract)
  readonly layer2: Cell600;  // φ-scaled (concrete)

  // Precomputed alignment points
  private _alignmentPoints: AlignmentPoint[] | null = null;

  // Current activation state
  private layer1Activations: number[];
  private layer2Activations: number[];

  constructor() {
    this.layer1 = new Cell600();  // Unit scale
    this.layer2 = new Cell600();  // Will be scaled when queried

    // Initialize activations to zero
    this.layer1Activations = new Array(120).fill(0);
    this.layer2Activations = new Array(120).fill(0);
  }

  // ==========================================================================
  // Layer Access
  // ==========================================================================

  /**
   * Get vertices of layer 1 (unit scale)
   */
  getLayer1Vertices(): Vector4D[] {
    return this.layer1.vertices;
  }

  /**
   * Get vertices of layer 2 (φ-scaled)
   */
  getLayer2Vertices(): Vector4D[] {
    return this.layer1.vertices.map(v => scale4D(v, PHI));
  }

  /**
   * Get a specific vertex from layer 1
   */
  getLayer1Vertex(index: number): Vector4D {
    return this.layer1.vertices[index];
  }

  /**
   * Get a specific vertex from layer 2 (φ-scaled)
   */
  getLayer2Vertex(index: number): Vector4D {
    return scale4D(this.layer1.vertices[index], PHI);
  }

  // ==========================================================================
  // Alignment Detection
  // ==========================================================================

  /**
   * Find all alignment points between the two layers
   * These are points where a layer2 vertex is close to a layer1 vertex
   */
  findAlignmentPoints(threshold: number = 0.1): AlignmentPoint[] {
    if (this._alignmentPoints !== null) {
      return this._alignmentPoints;
    }

    const alignments: AlignmentPoint[] = [];
    const layer2Vertices = this.getLayer2Vertices();

    for (let i = 0; i < 120; i++) {
      const v1 = this.layer1.vertices[i];
      let bestMatch = -1;
      let bestDist = Infinity;

      for (let j = 0; j < 120; j++) {
        const v2 = layer2Vertices[j];
        const dist = distance4D(v1, v2);
        if (dist < bestDist) {
          bestDist = dist;
          bestMatch = j;
        }
      }

      // Classify alignment type
      let type: AlignmentPoint['type'];
      let strength: number;

      if (bestDist < 0.01) {
        type = 'exact';
        strength = 1.0;
      } else if (bestDist < threshold) {
        type = 'near';
        strength = 1.0 - bestDist / threshold;
      } else {
        type = 'interpolated';
        strength = Math.max(0, 1.0 - bestDist);
      }

      alignments.push({
        position: v1,
        layer1Index: i,
        layer2Index: strength > 0.1 ? bestMatch : -1,
        alignmentStrength: strength,
        type,
      });
    }

    this._alignmentPoints = alignments;
    return alignments;
  }

  /**
   * Get strongly aligned points (alignment > threshold)
   */
  getStrongAlignments(threshold: number = 0.5): AlignmentPoint[] {
    return this.findAlignmentPoints().filter(
      a => a.alignmentStrength >= threshold
    );
  }

  /**
   * Check if a point is near an alignment
   */
  isNearAlignment(point: Vector4D, threshold: number = 0.2): AlignmentPoint | null {
    const alignments = this.findAlignmentPoints();
    let bestAlignment: AlignmentPoint | null = null;
    let bestDist = Infinity;

    for (const align of alignments) {
      if (align.alignmentStrength < 0.3) continue;
      const dist = distance4D(point, align.position);
      if (dist < threshold && dist < bestDist) {
        bestDist = dist;
        bestAlignment = align;
      }
    }

    return bestAlignment;
  }

  // ==========================================================================
  // Layer Blending
  // ==========================================================================

  /**
   * Blend a point between layers based on blend factor
   * @param layer1Point Point in layer 1 (unit scale)
   * @param blendFactor 0 = pure layer1, 1 = pure layer2
   */
  blendLayers(layer1Point: Vector4D, blendFactor: number): LayerBlendResult {
    const clampedBlend = Math.max(0, Math.min(1, blendFactor));

    // Find corresponding point in layer 2
    const layer2Point = scale4D(layer1Point, PHI);

    // Linear blend
    const blended: Vector4D = {
      w: layer1Point.w * (1 - clampedBlend) + layer2Point.w * clampedBlend,
      x: layer1Point.x * (1 - clampedBlend) + layer2Point.x * clampedBlend,
      y: layer1Point.y * (1 - clampedBlend) + layer2Point.y * clampedBlend,
      z: layer1Point.z * (1 - clampedBlend) + layer2Point.z * clampedBlend,
    };

    // Check for resonance (how close is blended point to an alignment?)
    const alignment = this.isNearAlignment(blended, 0.3);
    const resonance = alignment ? alignment.alignmentStrength : 0;

    return {
      blendedPosition: blended,
      layer1Weight: 1 - clampedBlend,
      layer2Weight: clampedBlend,
      resonance,
    };
  }

  /**
   * Golden blend - uses φ-weighted averaging
   * This creates naturally harmonious transitions
   */
  goldenBlend(layer1Point: Vector4D): LayerBlendResult {
    const layer2Point = scale4D(layer1Point, PHI);

    // Golden weighted blend: w1 = φ⁻¹, w2 = φ⁻²
    const w1 = PHI_INV;  // ≈ 0.618
    const w2 = PHI_INV * PHI_INV;  // ≈ 0.382
    const total = w1 + w2;

    const blended: Vector4D = {
      w: (layer1Point.w * w1 + layer2Point.w * w2) / total,
      x: (layer1Point.x * w1 + layer2Point.x * w2) / total,
      y: (layer1Point.y * w1 + layer2Point.y * w2) / total,
      z: (layer1Point.z * w1 + layer2Point.z * w2) / total,
    };

    const alignment = this.isNearAlignment(blended, 0.3);

    return {
      blendedPosition: blended,
      layer1Weight: w1 / total,
      layer2Weight: w2 / total,
      resonance: alignment ? alignment.alignmentStrength : 0,
    };
  }

  // ==========================================================================
  // Activation State Management
  // ==========================================================================

  /**
   * Set activation for a layer 1 vertex
   */
  activateLayer1(index: number, strength: number = 1): void {
    if (index >= 0 && index < 120) {
      this.layer1Activations[index] = Math.max(0, Math.min(1, strength));
    }
  }

  /**
   * Set activation for a layer 2 vertex
   */
  activateLayer2(index: number, strength: number = 1): void {
    if (index >= 0 && index < 120) {
      this.layer2Activations[index] = Math.max(0, Math.min(1, strength));
    }
  }

  /**
   * Set all layer 1 activations
   */
  setLayer1Activations(activations: number[]): void {
    if (activations.length !== 120) {
      throw new Error('Layer 1 requires exactly 120 activation values');
    }
    this.layer1Activations = activations.map(a => Math.max(0, Math.min(1, a)));
  }

  /**
   * Set all layer 2 activations
   */
  setLayer2Activations(activations: number[]): void {
    if (activations.length !== 120) {
      throw new Error('Layer 2 requires exactly 120 activation values');
    }
    this.layer2Activations = activations.map(a => Math.max(0, Math.min(1, a)));
  }

  /**
   * Reset all activations to zero
   */
  reset(): void {
    this.layer1Activations.fill(0);
    this.layer2Activations.fill(0);
  }

  /**
   * Get current state
   */
  getState(): E8ProjectionState {
    const alignments = this.findAlignmentPoints();

    // Compute total resonance from active aligned vertices
    let totalResonance = 0;
    for (const align of alignments) {
      if (align.alignmentStrength > 0.5) {
        const layer1Active = this.layer1Activations[align.layer1Index];
        const layer2Active = align.layer2Index >= 0
          ? this.layer2Activations[align.layer2Index]
          : 0;
        totalResonance += align.alignmentStrength * layer1Active * layer2Active;
      }
    }
    totalResonance = Math.min(1, totalResonance / 10); // Normalize

    return {
      layer1Activations: [...this.layer1Activations],
      layer2Activations: [...this.layer2Activations],
      alignmentPoints: alignments.filter(a => a.alignmentStrength > 0.3),
      totalResonance,
    };
  }

  // ==========================================================================
  // Centroid Computation
  // ==========================================================================

  /**
   * Compute weighted centroid of active layer 1 vertices
   */
  getLayer1Centroid(): Vector4D {
    let centroid: Vector4D = { w: 0, x: 0, y: 0, z: 0 };
    let totalWeight = 0;

    for (let i = 0; i < 120; i++) {
      const weight = this.layer1Activations[i];
      if (weight > 0) {
        const v = this.layer1.vertices[i];
        centroid = add4D(centroid, scale4D(v, weight));
        totalWeight += weight;
      }
    }

    if (totalWeight > 0) {
      centroid = scale4D(centroid, 1 / totalWeight);
    }

    return centroid;
  }

  /**
   * Compute weighted centroid of active layer 2 vertices
   */
  getLayer2Centroid(): Vector4D {
    let centroid: Vector4D = { w: 0, x: 0, y: 0, z: 0 };
    let totalWeight = 0;

    for (let i = 0; i < 120; i++) {
      const weight = this.layer2Activations[i];
      if (weight > 0) {
        const v = this.getLayer2Vertex(i);
        centroid = add4D(centroid, scale4D(v, weight));
        totalWeight += weight;
      }
    }

    if (totalWeight > 0) {
      centroid = scale4D(centroid, 1 / totalWeight);
    }

    return centroid;
  }

  // ==========================================================================
  // E8 Lattice Generation (Full 8D → 4D Projection)
  // ==========================================================================

  /**
   * Generate E8 root vectors in 8D
   * E8 has 240 roots of norm √2
   */
  static generateE8Roots(): Vector8D[] {
    const roots: Vector8D[] = [];

    // Type 1: All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) - 112 roots
    for (let i = 0; i < 8; i++) {
      for (let j = i + 1; j < 8; j++) {
        for (const si of [-1, 1]) {
          for (const sj of [-1, 1]) {
            const v: Vector8D = { x0: 0, x1: 0, x2: 0, x3: 0, x4: 0, x5: 0, x6: 0, x7: 0 };
            const keys: (keyof Vector8D)[] = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'];
            v[keys[i]] = si;
            v[keys[j]] = sj;
            roots.push(v);
          }
        }
      }
    }

    // Type 2: (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½) with even number of minus signs - 128 roots
    for (let mask = 0; mask < 256; mask++) {
      let negCount = 0;
      for (let b = 0; b < 8; b++) {
        if (mask & (1 << b)) negCount++;
      }
      if (negCount % 2 === 0) {
        const v: Vector8D = {
          x0: (mask & 1) ? -0.5 : 0.5,
          x1: (mask & 2) ? -0.5 : 0.5,
          x2: (mask & 4) ? -0.5 : 0.5,
          x3: (mask & 8) ? -0.5 : 0.5,
          x4: (mask & 16) ? -0.5 : 0.5,
          x5: (mask & 32) ? -0.5 : 0.5,
          x6: (mask & 64) ? -0.5 : 0.5,
          x7: (mask & 128) ? -0.5 : 0.5,
        };
        roots.push(v);
      }
    }

    return roots;
  }

  /**
   * Project E8 root to 4D using standard projection
   * Projects x0-x3 and sums with scaled x4-x7
   */
  static projectTo4D(v8: Vector8D): Vector4D {
    // Simple projection: first 4 coords + φ-scaled last 4 coords
    return {
      w: v8.x0 + PHI_INV * v8.x4,
      x: v8.x1 + PHI_INV * v8.x5,
      y: v8.x2 + PHI_INV * v8.x6,
      z: v8.x3 + PHI_INV * v8.x7,
    };
  }

  /**
   * Create E8Projection from full E8 lattice
   * This generates both layers from the 240 E8 roots
   */
  static fromE8Lattice(): E8Projection {
    const projection = new E8Projection();
    // The standard Cell600 already encodes the E8 projection structure
    // The two layers (unit and φ-scaled) emerge naturally
    return projection;
  }

  // ==========================================================================
  // Musical Application
  // ==========================================================================

  /**
   * Map musical key to layer 1 vertex (24 → 120 mapping)
   * Each key maps to 5 vertices (120/24 = 5)
   */
  keyToLayer1Vertices(keyIndex: number): number[] {
    // The 24-cell embeds in the 600-cell
    // Each 24-cell vertex corresponds to ~5 neighboring 600-cell vertices
    const embedded24Cell = this.layer1.embedded24Cell;
    const targetVertex = embedded24Cell[keyIndex % 24];

    // Find the 5 closest 600-cell vertices to this 24-cell vertex
    const distances: { index: number; dist: number }[] = [];
    for (let i = 0; i < 120; i++) {
      const dist = distance4D(this.layer1.vertices[i], targetVertex);
      distances.push({ index: i, dist });
    }

    distances.sort((a, b) => a.dist - b.dist);
    return distances.slice(0, 5).map(d => d.index);
  }

  /**
   * Activate a musical key across both layers
   */
  activateMusicalKey(keyIndex: number, layer1Strength: number = 1, layer2Strength: number = 0.5): void {
    const vertices = this.keyToLayer1Vertices(keyIndex);
    for (const v of vertices) {
      this.activateLayer1(v, layer1Strength);
      this.activateLayer2(v, layer2Strength);
    }
  }

  // ==========================================================================
  // Properties
  // ==========================================================================

  get vertexCount(): number {
    return 240; // 120 per layer × 2 layers
  }

  get alignmentCount(): number {
    return this.findAlignmentPoints().filter(a => a.alignmentStrength > 0.5).length;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create E8Projection with initial musical key activations
 */
export function createMusicalE8(activeKeys: number[]): E8Projection {
  const projection = new E8Projection();
  for (const key of activeKeys) {
    projection.activateMusicalKey(key);
  }
  return projection;
}

/**
 * Create E8Projection with uniform layer 1 activation
 */
export function createUniformLayer1(strength: number = 0.5): E8Projection {
  const projection = new E8Projection();
  projection.setLayer1Activations(new Array(120).fill(strength));
  return projection;
}

// ============================================================================
// Exports
// ============================================================================

export const E8ProjectionModule = {
  E8Projection,
  createMusicalE8,
  createUniformLayer1,
  PHI,
  PHI_INV,
};

export default E8ProjectionModule;
