/**
 * E8 Lattice Renderer
 *
 * @package hemac
 * @module visualization/E8Renderer
 *
 * Generates and renders the E8 root system (240 roots) using Moxness folding.
 * Projects 8D E8 lattice to 4D using the φ-coupled projection matrix.
 *
 * Mathematical Foundation:
 * - E8 has 240 roots (non-zero vectors of squared length 2)
 * - Moxness matrix projects 8D→4D preserving golden ratio structure
 * - Result: Two interlocking 600-cells at φ:1 ratio
 *
 * Verified from: e8codec-hyperrendering/utils/e8.ts, ppp-info-site/lib/topology/E8H4Folding.ts
 */

import type { Vector4D, Vector8D, Point4D } from '../types/index.js';
import { PHI, PHI_INV, E8_ROOT_COUNT } from '../types/index.js';

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * Moxness 8D→4D Projection Matrix
 *
 * This matrix projects E8 roots into 4D such that the result is
 * two interlocking H4 600-cells (each with 120 vertices) at φ:1 ratio.
 *
 * Source: Verified in e8codec-hyperrendering/utils/e8.ts
 */
export const MOXNESS_MATRIX: number[][] = [
  [1, 1, 0, 0, PHI, 0, 0, PHI_INV],
  [0, 1, 1, 0, 0, PHI, PHI_INV, 0],
  [0, 0, 1, 1, PHI_INV, 0, PHI, 0],
  [1, 0, 0, 1, 0, PHI_INV, 0, PHI],
];

/** Normalization factor for Moxness projection */
const MOXNESS_SCALE = 1 / Math.sqrt(2 + PHI);

// =============================================================================
// E8 ROOT GENERATION
// =============================================================================

/**
 * Generate all 240 roots of the E8 lattice.
 *
 * E8 roots come in three families:
 * 1. ±eᵢ ± eⱼ (112 roots) - permutations with two ±1s
 * 2. (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½) with even number of minus signs (128 roots)
 */
export function generateE8Roots(): Vector8D[] {
  const roots: Vector8D[] = [];

  // Type 1: ±eᵢ ± eⱼ (all pairs i < j, all sign combinations)
  // 8 choose 2 = 28 pairs × 4 sign combos = 112 roots
  for (let i = 0; i < 8; i++) {
    for (let j = i + 1; j < 8; j++) {
      for (const si of [-1, 1]) {
        for (const sj of [-1, 1]) {
          const root: Vector8D = [0, 0, 0, 0, 0, 0, 0, 0];
          root[i] = si;
          root[j] = sj;
          roots.push(root);
        }
      }
    }
  }

  // Type 2: Half-integer vectors with even parity
  // All (±½)⁸ with even number of minus signs = 128 roots
  for (let bits = 0; bits < 256; bits++) {
    // Count number of set bits (minus signs)
    let count = 0;
    for (let i = 0; i < 8; i++) {
      if (bits & (1 << i)) count++;
    }

    // Only even parity (even number of minus signs)
    if (count % 2 === 0) {
      const root: Vector8D = [0, 0, 0, 0, 0, 0, 0, 0];
      for (let i = 0; i < 8; i++) {
        root[i] = (bits & (1 << i)) ? -0.5 : 0.5;
      }
      roots.push(root);
    }
  }

  return roots;
}

/**
 * Project an 8D vector to 4D using the Moxness matrix.
 */
export function projectMoxness(v8d: Vector8D): Vector4D {
  const result: Vector4D = { x: 0, y: 0, z: 0, w: 0 };

  for (let i = 0; i < 4; i++) {
    let sum = 0;
    for (let j = 0; j < 8; j++) {
      sum += MOXNESS_MATRIX[i][j] * v8d[j];
    }
    if (i === 0) result.x = sum * MOXNESS_SCALE;
    else if (i === 1) result.y = sum * MOXNESS_SCALE;
    else if (i === 2) result.z = sum * MOXNESS_SCALE;
    else result.w = sum * MOXNESS_SCALE;
  }

  return result;
}

/**
 * Generate the full E8 lattice data with 4D projections.
 */
export function generateE8LatticeData(): {
  roots8D: Vector8D[];
  roots4D: Point4D[];
  diagnostics: {
    rootCount: number;
    rootCountValid: boolean;
    shellInner: number;
    shellOuter: number;
    shellRatio: number;
    scale: number;
  };
} {
  const roots8D = generateE8Roots();
  const roots4D: Point4D[] = [];

  let minRadius = Infinity;
  let maxRadius = 0;

  for (let i = 0; i < roots8D.length; i++) {
    const v4d = projectMoxness(roots8D[i]);
    const radius = Math.sqrt(v4d.x ** 2 + v4d.y ** 2 + v4d.z ** 2 + v4d.w ** 2);

    minRadius = Math.min(minRadius, radius);
    maxRadius = Math.max(maxRadius, radius);

    // Assign triadic color group based on index
    const colorGroup = assignTriadGroup(i, roots8D[i]);

    roots4D.push({
      ...v4d,
      id: i,
      colorGroup
    });
  }

  return {
    roots8D,
    roots4D,
    diagnostics: {
      rootCount: roots8D.length,
      rootCountValid: roots8D.length === E8_ROOT_COUNT,
      shellInner: minRadius,
      shellOuter: maxRadius,
      shellRatio: maxRadius / minRadius,
      scale: MOXNESS_SCALE
    }
  };
}

// =============================================================================
// TRIADIC COLOR ASSIGNMENT
// =============================================================================

/**
 * Assign a triadic color group to an E8 root.
 *
 * Groups correspond to the three "chiralities" in the E8→H4 decomposition:
 * - Group A: Inner 600-cell vertices
 * - Group B: Outer 600-cell vertices
 * - Group C: Shared/boundary vertices
 *
 * Source: Verified in e8codec-hyperrendering/utils/e8.ts assignTriadGroup()
 */
export function assignTriadGroup(index: number, root: Vector8D): 'A' | 'B' | 'C' {
  // Sum of coordinates determines chirality
  const sum = root.reduce((a, b) => a + b, 0);

  if (Math.abs(sum) < 0.1) {
    // Zero sum → boundary (C)
    return 'C';
  } else if (sum > 0) {
    // Positive sum → inner (A)
    return 'A';
  } else {
    // Negative sum → outer (B)
    return 'B';
  }
}

// =============================================================================
// STEREOGRAPHIC PROJECTION (4D → 3D)
// =============================================================================

/**
 * Stereographic projection from 4D to 3D.
 * Projects from the point (0, 0, 0, R) onto the w=0 hyperplane.
 *
 * This is conformal (preserves angles) and maps the 4D structure
 * into 3D space for visualization.
 */
export function stereographicProject4Dto3D(
  p: Vector4D,
  projectionRadius: number = 2
): { x: number; y: number; z: number; scale: number } {
  const R = projectionRadius;
  const denom = R - p.w;

  if (Math.abs(denom) < 1e-10) {
    // Point at projection pole
    return { x: 1e6, y: 1e6, z: 1e6, scale: 1e6 };
  }

  const scale = R / denom;
  return {
    x: p.x * scale,
    y: p.y * scale,
    z: p.z * scale,
    scale
  };
}

// =============================================================================
// E8 RENDERER CLASS
// =============================================================================

/** Renderer configuration */
export interface E8RendererConfig {
  projectionRadius: number;
  pointSize: number;
  edgeThreshold: number;  // Max distance for edge connections
  showEdges: boolean;
  colorByGroup: boolean;
  opacity: number;
}

export const DEFAULT_E8_RENDERER_CONFIG: E8RendererConfig = {
  projectionRadius: 2.5,
  pointSize: 3,
  edgeThreshold: 0.5,
  showEdges: true,
  colorByGroup: true,
  opacity: 0.8
};

/** Color palette for triadic groups */
export const TRIAD_COLORS: Record<'A' | 'B' | 'C', { r: number; g: number; b: number }> = {
  A: { r: 255, g: 100, b: 100 },  // Red-ish (thesis)
  B: { r: 100, g: 100, b: 255 },  // Blue-ish (antithesis)
  C: { r: 100, g: 255, b: 100 }   // Green-ish (synthesis)
};

/**
 * E8Renderer - Renders the E8 lattice in 3D space.
 */
export class E8Renderer {
  private config: E8RendererConfig;
  private latticeData: ReturnType<typeof generateE8LatticeData>;
  private projected3D: Array<{ x: number; y: number; z: number; scale: number; colorGroup: 'A' | 'B' | 'C' }>;
  private edges: Array<[number, number]>;

  constructor(config: Partial<E8RendererConfig> = {}) {
    this.config = { ...DEFAULT_E8_RENDERER_CONFIG, ...config };
    this.latticeData = generateE8LatticeData();
    this.projected3D = [];
    this.edges = [];

    this.updateProjection();
    if (this.config.showEdges) {
      this.computeEdges();
    }
  }

  /**
   * Update 3D projections from 4D points.
   */
  private updateProjection(): void {
    this.projected3D = this.latticeData.roots4D.map(p4d => {
      const p3d = stereographicProject4Dto3D(p4d, this.config.projectionRadius);
      return {
        ...p3d,
        colorGroup: p4d.colorGroup!
      };
    });
  }

  /**
   * Compute edges between nearby vertices.
   */
  private computeEdges(): void {
    this.edges = [];
    const threshold = this.config.edgeThreshold;
    const thresholdSq = threshold * threshold;

    for (let i = 0; i < this.projected3D.length; i++) {
      for (let j = i + 1; j < this.projected3D.length; j++) {
        const pi = this.projected3D[i];
        const pj = this.projected3D[j];

        const dx = pi.x - pj.x;
        const dy = pi.y - pj.y;
        const dz = pi.z - pj.z;
        const distSq = dx * dx + dy * dy + dz * dz;

        if (distSq < thresholdSq) {
          this.edges.push([i, j]);
        }
      }
    }
  }

  /**
   * Apply a 4D rotation to the lattice before projection.
   */
  rotateInPlane(plane: 'XY' | 'XZ' | 'XW' | 'YZ' | 'YW' | 'ZW', angle: number): void {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    for (const p4d of this.latticeData.roots4D) {
      let a: number, b: number;

      switch (plane) {
        case 'XY': a = p4d.x; b = p4d.y; p4d.x = cos * a - sin * b; p4d.y = sin * a + cos * b; break;
        case 'XZ': a = p4d.x; b = p4d.z; p4d.x = cos * a - sin * b; p4d.z = sin * a + cos * b; break;
        case 'XW': a = p4d.x; b = p4d.w; p4d.x = cos * a - sin * b; p4d.w = sin * a + cos * b; break;
        case 'YZ': a = p4d.y; b = p4d.z; p4d.y = cos * a - sin * b; p4d.z = sin * a + cos * b; break;
        case 'YW': a = p4d.y; b = p4d.w; p4d.y = cos * a - sin * b; p4d.w = sin * a + cos * b; break;
        case 'ZW': a = p4d.z; b = p4d.w; p4d.z = cos * a - sin * b; p4d.w = sin * a + cos * b; break;
      }
    }

    this.updateProjection();
  }

  /**
   * Get vertices as flat array for WebGL rendering.
   * Format: [x, y, z, r, g, b, a, ...]
   */
  getVertexBuffer(): Float32Array {
    const buffer = new Float32Array(this.projected3D.length * 7);

    for (let i = 0; i < this.projected3D.length; i++) {
      const p = this.projected3D[i];
      const color = this.config.colorByGroup ? TRIAD_COLORS[p.colorGroup] : { r: 200, g: 200, b: 200 };

      buffer[i * 7 + 0] = p.x;
      buffer[i * 7 + 1] = p.y;
      buffer[i * 7 + 2] = p.z;
      buffer[i * 7 + 3] = color.r / 255;
      buffer[i * 7 + 4] = color.g / 255;
      buffer[i * 7 + 5] = color.b / 255;
      buffer[i * 7 + 6] = this.config.opacity;
    }

    return buffer;
  }

  /**
   * Get edge indices for line rendering.
   */
  getEdgeBuffer(): Uint32Array {
    const buffer = new Uint32Array(this.edges.length * 2);

    for (let i = 0; i < this.edges.length; i++) {
      buffer[i * 2 + 0] = this.edges[i][0];
      buffer[i * 2 + 1] = this.edges[i][1];
    }

    return buffer;
  }

  /**
   * Get diagnostics about the lattice.
   */
  getDiagnostics(): typeof this.latticeData.diagnostics & { edgeCount: number } {
    return {
      ...this.latticeData.diagnostics,
      edgeCount: this.edges.length
    };
  }

  /**
   * Update configuration.
   */
  configure(config: Partial<E8RendererConfig>): void {
    const shouldRecomputeEdges = config.edgeThreshold !== undefined ||
                                  config.showEdges !== undefined;

    this.config = { ...this.config, ...config };

    if (config.projectionRadius !== undefined) {
      this.updateProjection();
    }

    if (shouldRecomputeEdges && this.config.showEdges) {
      this.computeEdges();
    }
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export { generateE8LatticeData as generateLattice };
