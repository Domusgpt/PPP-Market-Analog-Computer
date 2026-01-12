/**
 * Trinity Decomposition of the 24-Cell
 *
 * The 24-cell uniquely decomposes into THREE orthogonal 16-cells:
 * - Alpha (α) - Thesis
 * - Beta (β) - Antithesis
 * - Gamma (γ) - Synthesis
 *
 * This provides a NATIVE geometric structure for dialectic reasoning:
 * - Set Alpha state (thesis)
 * - Set Beta state (antithesis)
 * - Compute Gamma state (synthesis through overlap)
 *
 * MATHEMATICAL BASIS:
 * The 24 vertices of the 24-cell partition into 3 groups of 8:
 * - Each group forms a 16-cell (cross-polytope)
 * - The three 16-cells are mutually orthogonal
 * - Any two 16-cells determine the third
 *
 * VERTEX PARTITION:
 * α (Alpha): vertices 0-7 (from inner 16-cell)
 * β (Beta): vertices 8-15 (first half of outer 8-cell)
 * γ (Gamma): vertices 16-23 (second half of outer 8-cell)
 *
 * COGNITIVE APPLICATION:
 * - Alpha: Initial concept/belief (thesis)
 * - Beta: Opposing concept/challenge (antithesis)
 * - Gamma: Emergent resolution (synthesis)
 *
 * The synthesis isn't computed symbolically - it EMERGES from the
 * geometric overlap of Alpha and Beta projections.
 *
 * REFERENCES:
 * - https://en.wikipedia.org/wiki/24-cell#Three_16-cells
 */

import type { Vector4D } from '../music/music-geometry-domain';
import { Cell24, Cell16, distance4D, scale4D, add4D } from '../music/polytopes';

// ============================================================================
// Types
// ============================================================================

export type TrinityRole = 'alpha' | 'beta' | 'gamma';

export interface TrinityVertex {
  index: number;
  coords: Vector4D;
  role: TrinityRole;
  activation: number; // 0-1, how "active" this vertex is
}

export interface TrinityState {
  alpha: TrinityVertex[];
  beta: TrinityVertex[];
  gamma: TrinityVertex[];
}

export interface DialecticResult {
  thesis: Vector4D;      // Centroid of active Alpha vertices
  antithesis: Vector4D;  // Centroid of active Beta vertices
  synthesis: Vector4D;   // Emergent Gamma centroid
  tension: number;       // 0-1, how opposed are thesis/antithesis
  resolution: number;    // 0-1, how well does synthesis resolve
  convergence: boolean;  // Did Alpha and Beta converge to Gamma?
}

export interface OverlapResult {
  overlapVertices: number[];  // Gamma vertices activated by overlap
  overlapStrength: number;    // 0-1, strength of overlap
  pattern: 'convergent' | 'divergent' | 'orthogonal' | 'parallel';
}

// ============================================================================
// Trinity24Cell Class
// ============================================================================

export class Trinity24Cell {
  private cell24: Cell24;
  private vertices: TrinityVertex[];

  // The three 16-cell components
  readonly alpha: TrinityVertex[];
  readonly beta: TrinityVertex[];
  readonly gamma: TrinityVertex[];

  constructor() {
    this.cell24 = new Cell24();
    this.vertices = this.partitionVertices();

    this.alpha = this.vertices.filter(v => v.role === 'alpha');
    this.beta = this.vertices.filter(v => v.role === 'beta');
    this.gamma = this.vertices.filter(v => v.role === 'gamma');
  }

  /**
   * Partition 24 vertices into three orthogonal 16-cells
   */
  private partitionVertices(): TrinityVertex[] {
    const vertices: TrinityVertex[] = [];

    for (let i = 0; i < 24; i++) {
      const coords = this.cell24.vertices[i];
      let role: TrinityRole;

      // Partition based on vertex type and position
      // First 8 (from 16-cell): Alpha
      // Next 8: Beta
      // Last 8: Gamma
      if (i < 8) {
        role = 'alpha';
      } else if (i < 16) {
        role = 'beta';
      } else {
        role = 'gamma';
      }

      vertices.push({
        index: i,
        coords,
        role,
        activation: 0, // Start inactive
      });
    }

    return vertices;
  }

  // ==========================================================================
  // State Management
  // ==========================================================================

  /**
   * Set the activation state for Alpha (thesis)
   */
  setThesis(activations: number[]): void {
    if (activations.length !== 8) {
      throw new Error('Alpha requires exactly 8 activation values');
    }
    for (let i = 0; i < 8; i++) {
      this.alpha[i].activation = Math.max(0, Math.min(1, activations[i]));
    }
  }

  /**
   * Set the activation state for Beta (antithesis)
   */
  setAntithesis(activations: number[]): void {
    if (activations.length !== 8) {
      throw new Error('Beta requires exactly 8 activation values');
    }
    for (let i = 0; i < 8; i++) {
      this.beta[i].activation = Math.max(0, Math.min(1, activations[i]));
    }
  }

  /**
   * Activate a single vertex by index
   */
  activateVertex(index: number, strength: number = 1): void {
    const vertex = this.vertices.find(v => v.index === index);
    if (vertex) {
      vertex.activation = Math.max(0, Math.min(1, strength));
    }
  }

  /**
   * Reset all activations to zero
   */
  reset(): void {
    for (const v of this.vertices) {
      v.activation = 0;
    }
  }

  // ==========================================================================
  // Dialectic Computation
  // ==========================================================================

  /**
   * Compute the dialectic result from current Alpha/Beta states
   */
  computeDialectic(): DialecticResult {
    // Compute weighted centroids
    const thesis = this.computeCentroid(this.alpha);
    const antithesis = this.computeCentroid(this.beta);

    // Compute tension (distance between thesis and antithesis)
    const tensionDist = distance4D(thesis, antithesis);
    const maxDist = 2; // Maximum possible distance in unit 24-cell
    const tension = Math.min(1, tensionDist / maxDist);

    // Compute synthesis by "projecting" the midpoint onto Gamma
    const midpoint = this.midpoint4D(thesis, antithesis);
    const synthesis = this.projectOntoGamma(midpoint);

    // Compute resolution (how close is synthesis to both thesis and antithesis)
    const distToThesis = distance4D(synthesis, thesis);
    const distToAntithesis = distance4D(synthesis, antithesis);
    const avgDist = (distToThesis + distToAntithesis) / 2;
    const resolution = Math.max(0, 1 - avgDist / maxDist);

    // Check convergence (did we find a good synthesis?)
    const convergence = resolution > 0.5 && tension > 0.3;

    return {
      thesis,
      antithesis,
      synthesis,
      tension,
      resolution,
      convergence,
    };
  }

  /**
   * Compute overlap between Alpha and Beta projections
   */
  computeOverlap(): OverlapResult {
    const activeAlpha = this.alpha.filter(v => v.activation > 0.5);
    const activeBeta = this.beta.filter(v => v.activation > 0.5);

    if (activeAlpha.length === 0 || activeBeta.length === 0) {
      return {
        overlapVertices: [],
        overlapStrength: 0,
        pattern: 'orthogonal',
      };
    }

    // Find Gamma vertices that are "between" active Alpha and Beta vertices
    const overlapVertices: number[] = [];
    let totalStrength = 0;

    for (const gamma of this.gamma) {
      let minDistToAlpha = Infinity;
      let minDistToBeta = Infinity;

      for (const alpha of activeAlpha) {
        const dist = distance4D(gamma.coords, alpha.coords);
        if (dist < minDistToAlpha) minDistToAlpha = dist;
      }

      for (const beta of activeBeta) {
        const dist = distance4D(gamma.coords, beta.coords);
        if (dist < minDistToBeta) minDistToBeta = dist;
      }

      // Gamma vertex is in "overlap" if close to both Alpha and Beta
      const threshold = 1.5;
      if (minDistToAlpha < threshold && minDistToBeta < threshold) {
        overlapVertices.push(gamma.index);
        // Activate this gamma vertex
        gamma.activation = Math.max(gamma.activation,
          1 - (minDistToAlpha + minDistToBeta) / (2 * threshold));
        totalStrength += gamma.activation;
      }
    }

    const overlapStrength = overlapVertices.length > 0
      ? totalStrength / overlapVertices.length
      : 0;

    // Determine pattern
    let pattern: OverlapResult['pattern'];
    if (overlapVertices.length === 0) {
      pattern = 'orthogonal';
    } else if (overlapStrength > 0.7) {
      pattern = 'convergent';
    } else if (overlapStrength < 0.3) {
      pattern = 'divergent';
    } else {
      pattern = 'parallel';
    }

    return {
      overlapVertices,
      overlapStrength,
      pattern,
    };
  }

  /**
   * Perform one dialectic step: synthesis becomes new thesis
   */
  dialecticStep(): void {
    const result = this.computeDialectic();

    // Save current beta activations as new alpha
    const newAlphaActivations = this.beta.map(v => v.activation);

    // Gamma becomes new beta
    const newBetaActivations = this.gamma.map(v => v.activation);

    // Apply
    this.setThesis(newAlphaActivations);
    this.setAntithesis(newBetaActivations);

    // Reset gamma for next synthesis
    for (const v of this.gamma) {
      v.activation = 0;
    }
  }

  // ==========================================================================
  // Helper Methods
  // ==========================================================================

  private computeCentroid(vertices: TrinityVertex[]): Vector4D {
    let totalWeight = 0;
    let centroid: Vector4D = { w: 0, x: 0, y: 0, z: 0 };

    for (const v of vertices) {
      if (v.activation > 0) {
        centroid = add4D(centroid, scale4D(v.coords, v.activation));
        totalWeight += v.activation;
      }
    }

    if (totalWeight > 0) {
      centroid = scale4D(centroid, 1 / totalWeight);
    }

    return centroid;
  }

  private midpoint4D(a: Vector4D, b: Vector4D): Vector4D {
    return {
      w: (a.w + b.w) / 2,
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      z: (a.z + b.z) / 2,
    };
  }

  private projectOntoGamma(point: Vector4D): Vector4D {
    // Find the closest Gamma vertex to the point
    let minDist = Infinity;
    let closest: Vector4D = this.gamma[0].coords;

    for (const v of this.gamma) {
      const dist = distance4D(point, v.coords);
      if (dist < minDist) {
        minDist = dist;
        closest = v.coords;
      }
    }

    return closest;
  }

  // ==========================================================================
  // State Inspection
  // ==========================================================================

  /**
   * Get current state of all three components
   */
  getState(): TrinityState {
    return {
      alpha: [...this.alpha],
      beta: [...this.beta],
      gamma: [...this.gamma],
    };
  }

  /**
   * Get active vertices (activation > threshold)
   */
  getActiveVertices(threshold: number = 0.5): TrinityVertex[] {
    return this.vertices.filter(v => v.activation > threshold);
  }

  /**
   * Check if the three 16-cells are properly orthogonal
   */
  verifyOrthogonality(): {
    alphaBetaOrthogonal: boolean;
    betaGammaOrthogonal: boolean;
    gammaAlphaOrthogonal: boolean;
  } {
    // Two 16-cells are "orthogonal" if their centroids form 90° angle
    const alphaCentroid = this.computeCentroid(
      this.alpha.map(v => ({ ...v, activation: 1 }))
    );
    const betaCentroid = this.computeCentroid(
      this.beta.map(v => ({ ...v, activation: 1 }))
    );
    const gammaCentroid = this.computeCentroid(
      this.gamma.map(v => ({ ...v, activation: 1 }))
    );

    const dot = (a: Vector4D, b: Vector4D) =>
      a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;

    // Check if dot products are near zero (orthogonal)
    const threshold = 0.1;
    return {
      alphaBetaOrthogonal: Math.abs(dot(alphaCentroid, betaCentroid)) < threshold,
      betaGammaOrthogonal: Math.abs(dot(betaCentroid, gammaCentroid)) < threshold,
      gammaAlphaOrthogonal: Math.abs(dot(gammaCentroid, alphaCentroid)) < threshold,
    };
  }

  // Properties
  get vertexCount(): number { return 24; }
  get alphaCount(): number { return 8; }
  get betaCount(): number { return 8; }
  get gammaCount(): number { return 8; }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a Trinity24Cell with preset dialectic states
 */
export function createDialecticPair(
  thesisActivations: number[],
  antithesisActivations: number[]
): Trinity24Cell {
  const trinity = new Trinity24Cell();
  trinity.setThesis(thesisActivations);
  trinity.setAntithesis(antithesisActivations);
  return trinity;
}

/**
 * Create opposing thesis/antithesis (maximum tension)
 */
export function createOpposition(): Trinity24Cell {
  const trinity = new Trinity24Cell();
  // Activate opposite vertices
  trinity.setThesis([1, 0, 0, 0, 0, 0, 0, 1]);
  trinity.setAntithesis([0, 1, 1, 0, 0, 1, 1, 0]);
  return trinity;
}

/**
 * Create convergent thesis/antithesis (minimum tension)
 */
export function createConvergence(): Trinity24Cell {
  const trinity = new Trinity24Cell();
  // Activate similar vertices
  trinity.setThesis([1, 1, 0, 0, 0, 0, 0, 0]);
  trinity.setAntithesis([1, 1, 0, 0, 0, 0, 0, 0]);
  return trinity;
}

// ============================================================================
// Exports
// ============================================================================

export const TrinityModule = {
  Trinity24Cell,
  createDialecticPair,
  createOpposition,
  createConvergence,
};

export default TrinityModule;
