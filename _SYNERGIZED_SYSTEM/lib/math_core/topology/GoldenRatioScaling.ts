/**
 * Golden Ratio Scaling System
 *
 * Implements φ-nested polytope structures where polytopes are scaled by
 * powers of the golden ratio φ = (1 + √5) / 2.
 *
 * This creates the "moiré layer" effect where interference patterns emerge
 * between nested structures at different φ-scales.
 *
 * Key properties:
 * - φ² = φ + 1
 * - φ^n = F(n)φ + F(n-1) where F(n) is the n-th Fibonacci number
 * - 1/φ = φ - 1
 * - φ' = -1/φ (Galois conjugate)
 */

import type { Vector4D } from '../geometric_algebra/types.js';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Golden ratio φ = (1 + √5) / 2 */
export const PHI = (1 + Math.sqrt(5)) / 2;

/** Conjugate golden ratio φ' = (1 - √5) / 2 = -1/φ */
export const PHI_CONJUGATE = (1 - Math.sqrt(5)) / 2;

/** φ² = φ + 1 */
export const PHI_SQUARED = PHI * PHI;

/** 1/φ = φ - 1 */
export const INV_PHI = PHI - 1;

/** First 20 Fibonacci numbers */
export const FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181];

/** First 20 Lucas numbers */
export const LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364, 2207, 3571, 5778, 9349];

// =============================================================================
// TYPES
// =============================================================================

/** A nested structure with multiple φ-scaled layers */
export interface PhiNestedStructure<T> {
  /** Array of layers, indexed by scale level */
  readonly layers: PhiLayer<T>[];
  /** Center point of the nested structure */
  readonly center: Vector4D;
  /** Base scale (for layer 0) */
  readonly baseScale: number;
}

/** A single layer in a φ-nested structure */
export interface PhiLayer<T> {
  /** Scale level (0 = base, positive = outer, negative = inner) */
  readonly level: number;
  /** Absolute scale factor (φ^level) */
  readonly scale: number;
  /** Content at this scale */
  readonly content: T;
  /** Opacity for rendering (decreases with |level|) */
  readonly opacity: number;
}

/** Configuration for moiré pattern detection */
export interface MoireConfig {
  /** Number of layers to consider */
  layerCount: number;
  /** Threshold for interference detection */
  interferenceThreshold: number;
  /** Whether to use Galois conjugates */
  includeConjugates: boolean;
}

/** Moiré interference pattern */
export interface MoirePattern {
  /** Positions where interference occurs */
  readonly interferences: MoireInterference[];
  /** Overall pattern intensity */
  readonly intensity: number;
  /** Dominant interference frequency */
  readonly dominantFrequency: number;
}

/** A single moiré interference point */
export interface MoireInterference {
  /** 4D position */
  readonly position: Vector4D;
  /** Constructive (positive) or destructive (negative) */
  readonly type: 'constructive' | 'destructive';
  /** Intensity at this point */
  readonly intensity: number;
  /** Contributing layers */
  readonly layers: [number, number];
}

// =============================================================================
// GOLDEN RATIO FUNCTIONS
// =============================================================================

/**
 * Compute φ^n using Binet's formula.
 */
export function phiPower(n: number): number {
  if (n === 0) return 1;
  if (n === 1) return PHI;
  if (n === -1) return INV_PHI;

  // Use recursive squaring for efficiency
  if (n < 0) {
    return 1 / phiPower(-n);
  }

  // φ^n can also be computed as F(n)φ + F(n-1)
  if (n < FIBONACCI.length) {
    return FIBONACCI[n] * PHI + FIBONACCI[n - 1];
  }

  // Fall back to direct computation
  return Math.pow(PHI, n);
}

/**
 * Compute φ^n using the conjugate formula.
 * Returns both φ^n and φ'^n.
 */
export function phiPowerPair(n: number): { phi: number; phiConj: number } {
  return {
    phi: phiPower(n),
    phiConj: Math.pow(PHI_CONJUGATE, n)
  };
}

/**
 * Find the nearest Fibonacci number to a value.
 */
export function nearestFibonacci(value: number): { index: number; value: number } {
  let bestIdx = 0;
  let bestDist = Math.abs(value - FIBONACCI[0]);

  for (let i = 1; i < FIBONACCI.length; i++) {
    const dist = Math.abs(value - FIBONACCI[i]);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
    if (FIBONACCI[i] > value * 2) break;
  }

  return { index: bestIdx, value: FIBONACCI[bestIdx] };
}

/**
 * Check if a ratio is approximately φ.
 */
export function isGoldenRatio(a: number, b: number, tolerance: number = 0.001): boolean {
  if (b === 0) return false;
  const ratio = a / b;
  return Math.abs(ratio - PHI) < tolerance || Math.abs(ratio - INV_PHI) < tolerance;
}

// =============================================================================
// SCALING FUNCTIONS
// =============================================================================

/**
 * Scale a 4D vector by φ^n.
 */
export function scaleByPhi(v: Vector4D, n: number): Vector4D {
  const scale = phiPower(n);
  return [v[0] * scale, v[1] * scale, v[2] * scale, v[3] * scale];
}

/**
 * Scale a set of 4D vertices by φ^n.
 */
export function scaleVerticesByPhi(vertices: Vector4D[], n: number): Vector4D[] {
  const scale = phiPower(n);
  return vertices.map(v => [v[0] * scale, v[1] * scale, v[2] * scale, v[3] * scale]);
}

/**
 * Create a nested structure with multiple φ-scaled layers.
 */
export function createPhiNestedStructure<T>(
  baseContent: T,
  contentScaler: (content: T, scale: number) => T,
  minLevel: number = -2,
  maxLevel: number = 2,
  center: Vector4D = [0, 0, 0, 0],
  baseScale: number = 1
): PhiNestedStructure<T> {
  const layers: PhiLayer<T>[] = [];

  for (let level = minLevel; level <= maxLevel; level++) {
    const scale = phiPower(level) * baseScale;
    const opacity = 1 / (1 + Math.abs(level) * 0.3); // Fade with distance from base

    layers.push({
      level,
      scale,
      content: contentScaler(baseContent, scale),
      opacity
    });
  }

  return { layers, center, baseScale };
}

/**
 * Create nested 4D vertices at multiple φ-scales.
 */
export function createNestedVertices(
  baseVertices: Vector4D[],
  minLevel: number = -2,
  maxLevel: number = 2
): PhiNestedStructure<Vector4D[]> {
  return createPhiNestedStructure(
    baseVertices,
    (verts, scale) => verts.map(v => [v[0] * scale, v[1] * scale, v[2] * scale, v[3] * scale]),
    minLevel,
    maxLevel
  );
}

// =============================================================================
// MOIRÉ PATTERN DETECTION
// =============================================================================

/**
 * Detect moiré interference patterns between nested layers.
 */
export function detectMoirePatterns(
  nested: PhiNestedStructure<Vector4D[]>,
  config: MoireConfig = { layerCount: 5, interferenceThreshold: 0.1, includeConjugates: false }
): MoirePattern {
  const interferences: MoireInterference[] = [];

  // Compare each pair of layers
  for (let i = 0; i < nested.layers.length - 1; i++) {
    for (let j = i + 1; j < nested.layers.length; j++) {
      const layer1 = nested.layers[i];
      const layer2 = nested.layers[j];

      // Check for near-coincidences between vertices
      for (const v1 of layer1.content) {
        for (const v2 of layer2.content) {
          const dist = distance4D(v1, v2);

          if (dist < config.interferenceThreshold) {
            // Near-coincidence = constructive interference
            interferences.push({
              position: midpoint4D(v1, v2),
              type: 'constructive',
              intensity: 1 - dist / config.interferenceThreshold,
              layers: [layer1.level, layer2.level]
            });
          } else if (dist < config.interferenceThreshold * 2) {
            // Partial overlap = potential destructive interference
            interferences.push({
              position: midpoint4D(v1, v2),
              type: 'destructive',
              intensity: (config.interferenceThreshold * 2 - dist) / config.interferenceThreshold,
              layers: [layer1.level, layer2.level]
            });
          }
        }
      }
    }
  }

  // Calculate overall pattern properties
  const totalIntensity = interferences.reduce((sum, i) => sum + i.intensity, 0);
  const avgIntensity = interferences.length > 0 ? totalIntensity / interferences.length : 0;

  // Dominant frequency is based on the most common layer pair difference
  const layerDiffs = interferences.map(i => Math.abs(i.layers[1] - i.layers[0]));
  const dominantFrequency = layerDiffs.length > 0
    ? layerDiffs.reduce((a, b) => a + b, 0) / layerDiffs.length
    : 0;

  return {
    interferences,
    intensity: avgIntensity,
    dominantFrequency
  };
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Euclidean distance in 4D.
 */
function distance4D(a: Vector4D, b: Vector4D): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  const dw = a[3] - b[3];
  return Math.sqrt(dx*dx + dy*dy + dz*dz + dw*dw);
}

/**
 * Midpoint of two 4D vectors.
 */
function midpoint4D(a: Vector4D, b: Vector4D): Vector4D {
  return [
    (a[0] + b[0]) / 2,
    (a[1] + b[1]) / 2,
    (a[2] + b[2]) / 2,
    (a[3] + b[3]) / 2
  ];
}

// =============================================================================
// PHI-SPIRAL GENERATION
// =============================================================================

/**
 * Generate points along a 4D φ-spiral (golden spiral in 4D).
 *
 * The spiral follows the golden angle in multiple rotation planes.
 */
export function generatePhiSpiral(
  numPoints: number,
  startRadius: number = 0.1,
  radiusGrowth: number = PHI - 1
): Vector4D[] {
  const points: Vector4D[] = [];
  const goldenAngle = 2 * Math.PI / (PHI * PHI); // ~137.5°

  for (let i = 0; i < numPoints; i++) {
    const angle1 = i * goldenAngle;
    const angle2 = i * goldenAngle * INV_PHI;
    const radius = startRadius * Math.pow(PHI, i * radiusGrowth / numPoints);

    // Create 4D point using two rotation angles
    points.push([
      radius * Math.cos(angle1) * Math.cos(angle2),
      radius * Math.sin(angle1) * Math.cos(angle2),
      radius * Math.cos(angle1) * Math.sin(angle2),
      radius * Math.sin(angle1) * Math.sin(angle2)
    ]);
  }

  return points;
}

/**
 * Generate a φ-lattice (Penrose-like tiling in 4D).
 */
export function generatePhiLattice(
  gridSize: number,
  scale: number = 1
): Vector4D[] {
  const points: Vector4D[] = [];

  // Generate lattice points using φ-based spacing
  for (let i = -gridSize; i <= gridSize; i++) {
    for (let j = -gridSize; j <= gridSize; j++) {
      for (let k = -gridSize; k <= gridSize; k++) {
        for (let l = -gridSize; l <= gridSize; l++) {
          // Use Fibonacci-weighted coordinates
          const x = (i + j * INV_PHI) * scale;
          const y = (j + k * INV_PHI) * scale;
          const z = (k + l * INV_PHI) * scale;
          const w = (l + i * INV_PHI) * scale;

          points.push([x, y, z, w]);
        }
      }
    }
  }

  return points;
}

// =============================================================================
// GOLDEN RATIO SCALING CLASS
// =============================================================================

/**
 * Golden Ratio Scaling Manager.
 *
 * Manages φ-nested polytope structures and moiré pattern detection.
 */
export class GoldenRatioScaler {
  private _baseScale: number;
  private _levels: number[];

  constructor(baseScale: number = 1, minLevel: number = -3, maxLevel: number = 3) {
    this._baseScale = baseScale;
    this._levels = [];
    for (let i = minLevel; i <= maxLevel; i++) {
      this._levels.push(i);
    }
  }

  /** Get all scale levels */
  get levels(): number[] {
    return [...this._levels];
  }

  /** Get scale factor for a level */
  getScale(level: number): number {
    return phiPower(level) * this._baseScale;
  }

  /** Scale vertices to a specific level */
  scaleToLevel(vertices: Vector4D[], level: number): Vector4D[] {
    return scaleVerticesByPhi(vertices, level);
  }

  /** Create full nested structure */
  createNested(baseVertices: Vector4D[]): PhiNestedStructure<Vector4D[]> {
    return createNestedVertices(
      baseVertices,
      Math.min(...this._levels),
      Math.max(...this._levels)
    );
  }

  /** Detect moiré patterns in nested structure */
  detectMoire(nested: PhiNestedStructure<Vector4D[]>): MoirePattern {
    return detectMoirePatterns(nested);
  }

  /** Get the Fibonacci approximation of a ratio */
  fibonacciApprox(value: number): { num: number; den: number; error: number } {
    // Find best Fibonacci approximation
    let bestNum = 1, bestDen = 1, bestError = Infinity;

    for (let i = 1; i < FIBONACCI.length - 1; i++) {
      const num = FIBONACCI[i + 1];
      const den = FIBONACCI[i];
      const approx = num / den;
      const error = Math.abs(value - approx);

      if (error < bestError) {
        bestError = error;
        bestNum = num;
        bestDen = den;
      }
    }

    return { num: bestNum, den: bestDen, error: bestError };
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export default GoldenRatioScaler;
