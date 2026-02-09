/**
 * Ghost Frequency Detector
 *
 * Identifies implied but absent harmonic content via topological voids (β₂ features).
 * Based on the principle that persistent voids in the activation pattern suggest
 * "missing" vertices that would complete a harmonic structure.
 *
 * From Phillips GIT: "Ghost frequencies" are detected through persistent β₂ features
 * (voids/cavities) in the homology of the activated vertex set.
 */

import type {
  Vector4D,
  TrinityAxis,
  BettiProfile,
  PersistencePair
} from '../types/index';

import { Lattice24, type LatticeVertex } from '../topology/Lattice24';
import { normalize4D, dot4D, subtract4D, add4D, scale4D } from '../math/GeometricAlgebra';

// =============================================================================
// TYPES
// =============================================================================

export interface GhostVertex {
  /** Index of the ghost (missing) vertex in the 24-cell */
  readonly vertexIndex: number;
  /** 4D coordinates of the ghost vertex */
  readonly position: Vector4D;
  /** Trinity axis of the ghost vertex */
  readonly axis: TrinityAxis;
  /** Confidence score (0-1) based on void persistence */
  readonly confidence: number;
  /** The void that implies this ghost */
  readonly sourceVoid: VoidFeature;
  /** Musical interpretation (if applicable) */
  readonly musicalKey?: string;
}

export interface VoidFeature {
  /** Center of the void in 4D space */
  readonly center: Vector4D;
  /** Approximate radius of the void */
  readonly radius: number;
  /** Birth time in filtration */
  readonly birth: number;
  /** Death time in filtration (Infinity if still alive) */
  readonly death: number;
  /** Persistence = death - birth */
  readonly persistence: number;
  /** Vertices that form the boundary of this void */
  readonly boundaryVertices: number[];
}

export interface GhostAnalysis {
  /** Detected ghost vertices */
  readonly ghosts: GhostVertex[];
  /** All detected voids */
  readonly voids: VoidFeature[];
  /** Overall "incompleteness" score (0-1) */
  readonly incompleteness: number;
  /** Suggested resolution actions */
  readonly suggestions: ResolutionSuggestion[];
  /** Trinity axis most affected by ghosts */
  readonly dominantGhostAxis: TrinityAxis | null;
}

export interface ResolutionSuggestion {
  /** Type of resolution */
  readonly type: 'activate' | 'modulate' | 'transition';
  /** Target vertex or key */
  readonly target: string;
  /** Expected tension reduction */
  readonly tensionReduction: number;
  /** Description */
  readonly description: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const VOID_PERSISTENCE_THRESHOLD = 0.15; // Minimum persistence for significant void
const GHOST_CONFIDENCE_THRESHOLD = 0.3; // Minimum confidence to report ghost
const MAX_GHOSTS_TO_REPORT = 5;

// Key names for musical interpretation
const KEY_NAMES = [
  'C major', 'G major', 'D major', 'A major', 'E major', 'B major',
  'F# major', 'Db major', 'Ab major', 'Eb major', 'Bb major', 'F major',
  'A minor', 'E minor', 'B minor', 'F# minor', 'C# minor', 'G# minor',
  'D# minor', 'Bb minor', 'F minor', 'C minor', 'G minor', 'D minor'
];

// =============================================================================
// GHOST FREQUENCY DETECTOR
// =============================================================================

export class GhostFrequencyDetector {
  private _lattice: Lattice24;

  constructor() {
    this._lattice = new Lattice24();
  }

  /**
   * Analyze active vertices for ghost frequencies (missing harmonics).
   *
   * @param activeVertexIndices - Indices of currently active vertices
   * @param persistencePairs - Optional pre-computed persistence pairs
   * @returns Ghost analysis with detected missing vertices
   */
  analyzeGhosts(
    activeVertexIndices: number[],
    persistencePairs?: PersistencePair[]
  ): GhostAnalysis {
    const activeSet = new Set(activeVertexIndices);
    const inactiveIndices = Array.from({ length: 24 }, (_, i) => i)
      .filter(i => !activeSet.has(i));

    // Get active vertex positions
    const activePositions = activeVertexIndices.map(i =>
      this._lattice.getVertex(i).coordinates
    );

    // Detect voids in the active vertex pattern
    const voids = this._detectVoids(activePositions, activeVertexIndices);

    // Find ghost vertices (inactive vertices near void centers)
    const ghosts = this._findGhostVertices(voids, inactiveIndices);

    // Calculate overall incompleteness
    const incompleteness = this._calculateIncompleteness(
      activeVertexIndices.length,
      ghosts
    );

    // Generate resolution suggestions
    const suggestions = this._generateSuggestions(ghosts);

    // Determine dominant ghost axis
    const dominantGhostAxis = this._findDominantGhostAxis(ghosts);

    return {
      ghosts: ghosts.slice(0, MAX_GHOSTS_TO_REPORT),
      voids,
      incompleteness,
      suggestions,
      dominantGhostAxis
    };
  }

  /**
   * Detect voids (β₂ features) in the active vertex pattern.
   */
  private _detectVoids(
    activePositions: Vector4D[],
    activeIndices: number[]
  ): VoidFeature[] {
    if (activePositions.length < 4) {
      return []; // Need at least 4 points to form a void in 4D
    }

    const voids: VoidFeature[] = [];

    // Compute centroid of active vertices
    const centroid = this._computeCentroid(activePositions);

    // Check for voids by looking at gaps in angular coverage
    // A void exists where there's a large angular gap from the centroid

    // Sample directions and check coverage
    const directions = this._sampleDirections();

    for (const dir of directions) {
      // Find vertices in this direction
      const verticesInDirection = activePositions
        .map((pos, idx) => ({
          pos,
          idx: activeIndices[idx],
          alignment: dot4D(normalize4D(subtract4D(pos, centroid)), dir)
        }))
        .filter(v => v.alignment > 0.5);

      // If no vertices in this direction, potential void
      if (verticesInDirection.length === 0) {
        const voidCenter = add4D(centroid, scale4D(dir, 0.5));

        // Check if this void is significant
        const nearestDist = this._findNearestDistance(voidCenter, activePositions);

        if (nearestDist > 0.3) { // Significant gap
          // Find boundary vertices
          const boundaryVertices = activePositions
            .map((pos, idx) => ({
              idx: activeIndices[idx],
              dist: this._distance4D(pos, voidCenter)
            }))
            .sort((a, b) => a.dist - b.dist)
            .slice(0, 4)
            .map(v => v.idx);

          voids.push({
            center: voidCenter,
            radius: nearestDist,
            birth: 0,
            death: nearestDist,
            persistence: nearestDist,
            boundaryVertices
          });
        }
      }
    }

    // Sort by persistence (most persistent first)
    return voids
      .filter(v => v.persistence >= VOID_PERSISTENCE_THRESHOLD)
      .sort((a, b) => b.persistence - a.persistence);
  }

  /**
   * Find ghost vertices near void centers.
   */
  private _findGhostVertices(
    voids: VoidFeature[],
    inactiveIndices: number[]
  ): GhostVertex[] {
    const ghosts: GhostVertex[] = [];
    const usedIndices = new Set<number>();

    for (const voidFeature of voids) {
      // Find inactive vertex closest to void center
      let bestIndex = -1;
      let bestDistance = Infinity;

      for (const idx of inactiveIndices) {
        if (usedIndices.has(idx)) continue;

        const vertex = this._lattice.getVertex(idx);
        const dist = this._distance4D(vertex.coordinates, voidFeature.center);

        if (dist < bestDistance) {
          bestDistance = dist;
          bestIndex = idx;
        }
      }

      if (bestIndex >= 0 && bestDistance < voidFeature.radius * 1.5) {
        const vertex = this._lattice.getVertex(bestIndex);

        // Confidence based on how well vertex fills the void
        const confidence = Math.max(0, 1 - (bestDistance / voidFeature.radius));

        if (confidence >= GHOST_CONFIDENCE_THRESHOLD) {
          ghosts.push({
            vertexIndex: bestIndex,
            position: vertex.coordinates,
            axis: vertex.trinityAxis,
            confidence,
            sourceVoid: voidFeature,
            musicalKey: KEY_NAMES[bestIndex] || undefined
          });

          usedIndices.add(bestIndex);
        }
      }
    }

    // Sort by confidence
    return ghosts.sort((a, b) => b.confidence - a.confidence);
  }

  /**
   * Calculate overall incompleteness score.
   */
  private _calculateIncompleteness(
    activeCount: number,
    ghosts: GhostVertex[]
  ): number {
    if (activeCount >= 24) return 0;
    if (activeCount === 0) return 1;

    // Base incompleteness from coverage
    const coverageScore = 1 - (activeCount / 24);

    // Weighted by ghost confidence
    const ghostWeight = ghosts.reduce((sum, g) => sum + g.confidence, 0) / 24;

    return Math.min(1, coverageScore * 0.6 + ghostWeight * 0.4);
  }

  /**
   * Generate resolution suggestions for detected ghosts.
   */
  private _generateSuggestions(ghosts: GhostVertex[]): ResolutionSuggestion[] {
    const suggestions: ResolutionSuggestion[] = [];

    for (const ghost of ghosts.slice(0, 3)) {
      // Primary suggestion: activate the ghost vertex
      suggestions.push({
        type: 'activate',
        target: ghost.musicalKey || `Vertex ${ghost.vertexIndex}`,
        tensionReduction: ghost.confidence * 0.5,
        description: `Activate ${ghost.musicalKey || `vertex ${ghost.vertexIndex}`} to fill harmonic void`
      });

      // If ghost is on different axis, suggest transition
      suggestions.push({
        type: 'transition',
        target: ghost.axis,
        tensionReduction: ghost.confidence * 0.3,
        description: `Transition to ${ghost.axis} axis to approach ghost frequency`
      });
    }

    return suggestions;
  }

  /**
   * Find the Trinity axis most affected by ghosts.
   */
  private _findDominantGhostAxis(ghosts: GhostVertex[]): TrinityAxis | null {
    if (ghosts.length === 0) return null;

    const axisCounts: Record<TrinityAxis, number> = {
      'alpha': 0,
      'beta': 0,
      'gamma': 0
    };

    for (const ghost of ghosts) {
      axisCounts[ghost.axis] += ghost.confidence;
    }

    const maxAxis = Object.entries(axisCounts)
      .sort(([, a], [, b]) => b - a)[0];

    return maxAxis[1] > 0 ? maxAxis[0] as TrinityAxis : null;
  }

  /**
   * Compute centroid of positions.
   */
  private _computeCentroid(positions: Vector4D[]): Vector4D {
    const sum: Vector4D = [0, 0, 0, 0];
    for (const pos of positions) {
      sum[0] += pos[0];
      sum[1] += pos[1];
      sum[2] += pos[2];
      sum[3] += pos[3];
    }
    const n = positions.length;
    return [sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n];
  }

  /**
   * Sample directions in 4D for void detection.
   */
  private _sampleDirections(): Vector4D[] {
    // Use vertices of 24-cell as sample directions
    const directions: Vector4D[] = [];

    // Permutations of (±1, ±1, 0, 0)
    const signs = [-1, 1];
    for (const s1 of signs) {
      for (const s2 of signs) {
        directions.push([s1, s2, 0, 0]);
        directions.push([s1, 0, s2, 0]);
        directions.push([s1, 0, 0, s2]);
        directions.push([0, s1, s2, 0]);
        directions.push([0, s1, 0, s2]);
        directions.push([0, 0, s1, s2]);
      }
    }

    return directions.map(d => normalize4D(d));
  }

  /**
   * Find nearest distance from point to set of positions.
   */
  private _findNearestDistance(point: Vector4D, positions: Vector4D[]): number {
    let minDist = Infinity;
    for (const pos of positions) {
      const dist = this._distance4D(point, pos);
      if (dist < minDist) minDist = dist;
    }
    return minDist;
  }

  /**
   * Euclidean distance in 4D.
   */
  private _distance4D(a: Vector4D, b: Vector4D): number {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    const dw = a[3] - b[3];
    return Math.sqrt(dx*dx + dy*dy + dz*dz + dw*dw);
  }

  /**
   * Get suggested key to play to resolve highest-confidence ghost.
   */
  getSuggestedKey(activeVertexIndices: number[]): string | null {
    const analysis = this.analyzeGhosts(activeVertexIndices);
    if (analysis.ghosts.length === 0) return null;
    return analysis.ghosts[0].musicalKey || null;
  }

  /**
   * Check if playing a specific vertex would collapse a void.
   */
  wouldCollapseVoid(
    vertexIndex: number,
    activeVertexIndices: number[]
  ): { wouldCollapse: boolean; tensionReduction: number } {
    const analysis = this.analyzeGhosts(activeVertexIndices);

    for (const ghost of analysis.ghosts) {
      if (ghost.vertexIndex === vertexIndex) {
        return {
          wouldCollapse: true,
          tensionReduction: ghost.confidence * 0.5
        };
      }
    }

    return { wouldCollapse: false, tensionReduction: 0 };
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export default GhostFrequencyDetector;
