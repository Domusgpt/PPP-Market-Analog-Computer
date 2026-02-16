/**
 * Music-Geometry Domain
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 *
 * Maps musical structures to 24-cell geometry based on:
 * - Pythagorean harmony principles
 * - Neo-Riemannian transformations (PLR)
 * - Circle of fifths as 4D rotation
 * - Octatonic collections to Trinity axes
 *
 * The 24 vertices of the 24-cell map to:
 * - 12 major keys (C, G, D, A, E, B, F#, Db, Ab, Eb, Bb, F)
 * - 12 minor keys (c, g, d, a, e, b, f#, c#, g#, eb, bb, f)
 *
 * Trinity-Octatonic Mapping:
 * - Alpha → OCT₀,₁ = {C, C#, Eb, E, F#, G, A, Bb}
 * - Beta  → OCT₁,₂ = {C#, D, E, F, G, Ab, Bb, B}
 * - Gamma → OCT₀,₂ = {C, D, Eb, F, F#, Ab, A, B}
 *
 * Ported from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * with import paths adapted for PPP math_core layout.
 */

import {
  Vector4D,
  TrinityAxis,
  MusicalKey,
  GeometricChord,
  OctatonicCollection,
  MATH_CONSTANTS
} from '../geometric_algebra/types.js';

import { Lattice24, getDefaultLattice } from '../geometric_algebra/Lattice24.js';
import { dot, magnitude, normalize, centroid as computeCentroid } from '../geometric_algebra/GeometricAlgebra.js';

// =============================================================================
// MUSICAL CONSTANTS
// =============================================================================

/** Pitch class names */
const PITCH_CLASSES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

/** Circle of fifths order (by pitch class) */
const CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5];

/** Octatonic collections (diminished scales) */
const OCTATONIC_COLLECTIONS: Record<string, number[]> = {
  'OCT_0_1': [0, 1, 3, 4, 6, 7, 9, 10],   // C, C#, Eb, E, F#, G, A, Bb
  'OCT_1_2': [1, 2, 4, 5, 7, 8, 10, 11],  // C#, D, E, F, G, Ab, Bb, B
  'OCT_0_2': [0, 2, 3, 5, 6, 8, 9, 11]    // C, D, Eb, F, F#, Ab, A, B
};

/** Map Trinity axis to Octatonic collection */
const TRINITY_OCTATONIC_MAP: Record<TrinityAxis, string> = {
  'alpha': 'OCT_0_1',
  'beta': 'OCT_1_2',
  'gamma': 'OCT_0_2'
};

// =============================================================================
// KEY-VERTEX MAPPING
// =============================================================================

/**
 * Generate the mapping from musical keys to 24-cell vertices.
 *
 * Strategy:
 * - Major keys map to vertices with positive coordinate sums
 * - Minor keys map to vertices with negative coordinate sums
 * - Circle of fifths progression corresponds to rotation in 4D
 */
function generateKeyVertexMapping(lattice: Lattice24): Map<string, number> {
  const mapping = new Map<string, number>();

  // Get vertices sorted by their position on the lattice
  const vertices = lattice.vertices;

  // Separate vertices by Trinity axis and sign pattern
  const axisVertices: Record<TrinityAxis, number[]> = {
    alpha: [],
    beta: [],
    gamma: []
  };

  for (let i = 0; i < vertices.length; i++) {
    const axis = vertices[i].trinityAxis;
    if (axis) {
      axisVertices[axis].push(i);
    }
  }

  // Map keys to vertices within each axis
  // Use coordinate patterns to distribute major/minor

  let vertexIndex = 0;

  // Map major keys (first 12 vertices based on positive w or positive sum)
  for (let i = 0; i < 12; i++) {
    const pitchClass = CIRCLE_OF_FIFTHS[i];
    const keyName = `${PITCH_CLASSES[pitchClass]}`;
    mapping.set(keyName, vertexIndex % 24);
    vertexIndex++;
  }

  // Map minor keys (remaining 12 vertices)
  for (let i = 0; i < 12; i++) {
    const pitchClass = CIRCLE_OF_FIFTHS[i];
    const keyName = `${PITCH_CLASSES[pitchClass].toLowerCase()}m`;
    mapping.set(keyName, vertexIndex % 24);
    vertexIndex++;
  }

  return mapping;
}

/**
 * Generate reverse mapping from vertex to keys.
 */
function generateVertexKeyMapping(keyVertexMap: Map<string, number>): Map<number, string[]> {
  const vertexKeyMap = new Map<number, string[]>();

  for (const [key, vertex] of keyVertexMap) {
    if (!vertexKeyMap.has(vertex)) {
      vertexKeyMap.set(vertex, []);
    }
    vertexKeyMap.get(vertex)!.push(key);
  }

  return vertexKeyMap;
}

// =============================================================================
// CHORD GEOMETRY
// =============================================================================

/**
 * Map a chord to activated vertices.
 */
function chordToVertices(
  pitchClasses: number[],
  keyVertexMap: Map<string, number>
): number[] {
  const vertices: number[] = [];

  // Find keys that contain these pitch classes
  // This is a simplified mapping - a full implementation would use
  // harmonic analysis to determine the most likely key context

  for (const pc of pitchClasses) {
    const majorKey = PITCH_CLASSES[pc];
    const minorKey = `${PITCH_CLASSES[pc].toLowerCase()}m`;

    const majorVertex = keyVertexMap.get(majorKey);
    const minorVertex = keyVertexMap.get(minorKey);

    if (majorVertex !== undefined) vertices.push(majorVertex);
    if (minorVertex !== undefined) vertices.push(minorVertex);
  }

  return [...new Set(vertices)]; // Remove duplicates
}

/**
 * Compute Trinity weights for a chord.
 */
function computeChordTrinityWeights(
  activeVertices: number[],
  lattice: Lattice24
): [number, number, number] {
  let alpha = 0, beta = 0, gamma = 0;

  for (const vId of activeVertices) {
    const vertex = lattice.getVertex(vId);
    if (vertex?.trinityAxis === 'alpha') alpha++;
    else if (vertex?.trinityAxis === 'beta') beta++;
    else if (vertex?.trinityAxis === 'gamma') gamma++;
  }

  const total = activeVertices.length || 1;
  return [alpha / total, beta / total, gamma / total];
}

// =============================================================================
// NEO-RIEMANNIAN TRANSFORMATIONS
// =============================================================================

/**
 * Neo-Riemannian P (Parallel) transformation.
 * Transforms between major and relative minor (same root).
 * C major ↔ C minor
 */
function parallelTransform(pitchClass: number, isMajor: boolean): { pc: number; major: boolean } {
  return { pc: pitchClass, major: !isMajor };
}

/**
 * Neo-Riemannian L (Leading-tone) transformation.
 * C major ↔ E minor, C minor ↔ Ab major
 */
function leadingToneTransform(pitchClass: number, isMajor: boolean): { pc: number; major: boolean } {
  if (isMajor) {
    return { pc: (pitchClass + 4) % 12, major: false }; // Up major third, to minor
  } else {
    return { pc: (pitchClass + 8) % 12, major: true };  // Down major third, to major
  }
}

/**
 * Neo-Riemannian R (Relative) transformation.
 * C major ↔ A minor
 */
function relativeTransform(pitchClass: number, isMajor: boolean): { pc: number; major: boolean } {
  if (isMajor) {
    return { pc: (pitchClass + 9) % 12, major: false }; // Down minor third, to minor
  } else {
    return { pc: (pitchClass + 3) % 12, major: true };  // Up minor third, to major
  }
}

// =============================================================================
// MUSIC GEOMETRY DOMAIN CLASS
// =============================================================================

/**
 * MusicGeometryDomain provides musical-geometric mapping.
 *
 * Usage:
 * ```typescript
 * const domain = new MusicGeometryDomain();
 * const vertex = domain.keyToVertex('C'); // Get vertex for C major
 * const chord = domain.encodeChord([0, 4, 7]); // Encode C major triad
 * ```
 */
export class MusicGeometryDomain {
  private _lattice: Lattice24;
  private _keyVertexMap: Map<string, number>;
  private _vertexKeyMap: Map<number, string[]>;
  private _octatonicCollections: Map<TrinityAxis, OctatonicCollection>;

  constructor(lattice?: Lattice24) {
    this._lattice = lattice ?? getDefaultLattice();
    this._keyVertexMap = generateKeyVertexMapping(this._lattice);
    this._vertexKeyMap = generateVertexKeyMapping(this._keyVertexMap);
    this._octatonicCollections = this._buildOctatonicCollections();
  }

  private _buildOctatonicCollections(): Map<TrinityAxis, OctatonicCollection> {
    const collections = new Map<TrinityAxis, OctatonicCollection>();

    const axes: TrinityAxis[] = ['alpha', 'beta', 'gamma'];
    const names = ['OCT₀,₁', 'OCT₁,₂', 'OCT₀,₂'];
    const octKeys = ['OCT_0_1', 'OCT_1_2', 'OCT_0_2'];

    for (let i = 0; i < 3; i++) {
      collections.set(axes[i], {
        axis: axes[i],
        pitchClasses: OCTATONIC_COLLECTIONS[octKeys[i]],
        name: names[i]
      });
    }

    return collections;
  }

  // =========================================================================
  // KEY OPERATIONS
  // =========================================================================

  /**
   * Get vertex ID for a key name.
   */
  keyToVertex(keyName: string): number | undefined {
    return this._keyVertexMap.get(keyName);
  }

  /**
   * Get key names for a vertex ID.
   */
  vertexToKeys(vertexId: number): string[] {
    return this._vertexKeyMap.get(vertexId) ?? [];
  }

  /**
   * Get all key names.
   */
  getAllKeys(): string[] {
    return [...this._keyVertexMap.keys()];
  }

  /**
   * Get MusicalKey object for a key name.
   */
  getKey(keyName: string): MusicalKey | undefined {
    const vertex = this._keyVertexMap.get(keyName);
    if (vertex === undefined) return undefined;

    const isMajor = !keyName.includes('m');
    const rootName = keyName.replace('m', '').toUpperCase();
    const root = PITCH_CLASSES.indexOf(rootName);

    return {
      root: root >= 0 ? root : 0,
      mode: isMajor ? 'major' : 'minor',
      vertexId: vertex
    };
  }

  // =========================================================================
  // CHORD OPERATIONS
  // =========================================================================

  /**
   * Encode a chord as a GeometricChord.
   */
  encodeChord(pitchClasses: number[]): GeometricChord {
    const activeVertices = chordToVertices(pitchClasses, this._keyVertexMap);

    // Compute centroid
    const vertexCoords = activeVertices.map(id =>
      this._lattice.getVertex(id)?.coordinates ?? [0, 0, 0, 0] as Vector4D
    );
    const chordCentroid = computeCentroid(vertexCoords);

    // Compute Trinity weights
    const trinityWeights = computeChordTrinityWeights(activeVertices, this._lattice);

    return {
      pitchClasses,
      activeVertices,
      centroid: chordCentroid,
      trinityWeights
    };
  }

  /**
   * Encode a chord from note names.
   */
  encodeChordFromNotes(notes: string[]): GeometricChord {
    const pitchClasses = notes.map(note => {
      const pc = PITCH_CLASSES.indexOf(note.toUpperCase());
      return pc >= 0 ? pc : 0;
    });
    return this.encodeChord(pitchClasses);
  }

  /**
   * Get the dominant Trinity axis for a chord.
   */
  getChordAxis(chord: GeometricChord): TrinityAxis {
    const [alpha, beta, gamma] = chord.trinityWeights;
    if (alpha >= beta && alpha >= gamma) return 'alpha';
    if (beta >= gamma) return 'beta';
    return 'gamma';
  }

  // =========================================================================
  // OCTATONIC OPERATIONS
  // =========================================================================

  /**
   * Get the octatonic collection for a Trinity axis.
   */
  getOctatonicCollection(axis: TrinityAxis): OctatonicCollection {
    return this._octatonicCollections.get(axis)!;
  }

  /**
   * Check if a pitch class belongs to an axis's octatonic collection.
   */
  isInOctatonic(pitchClass: number, axis: TrinityAxis): boolean {
    const collection = this._octatonicCollections.get(axis);
    return collection?.pitchClasses.includes(pitchClass) ?? false;
  }

  /**
   * Get the primary axis for a pitch class.
   */
  getPitchClassAxis(pitchClass: number): TrinityAxis | null {
    for (const [axis, collection] of this._octatonicCollections) {
      if (collection.pitchClasses.includes(pitchClass)) {
        return axis;
      }
    }
    return null;
  }

  // =========================================================================
  // CIRCLE OF FIFTHS
  // =========================================================================

  /**
   * Get the next key in the circle of fifths.
   */
  nextInCircleOfFifths(keyName: string): string {
    const key = this.getKey(keyName);
    if (!key) return keyName;

    const nextRoot = (key.root + 7) % 12;
    const nextKey = PITCH_CLASSES[nextRoot] + (key.mode === 'minor' ? 'm' : '');
    return nextKey;
  }

  /**
   * Get the previous key in the circle of fifths (circle of fourths).
   */
  prevInCircleOfFifths(keyName: string): string {
    const key = this.getKey(keyName);
    if (!key) return keyName;

    const prevRoot = (key.root + 5) % 12;
    const prevKey = PITCH_CLASSES[prevRoot] + (key.mode === 'minor' ? 'm' : '');
    return prevKey;
  }

  /**
   * Get distance in circle of fifths between two keys.
   */
  circleOfFifthsDistance(key1: string, key2: string): number {
    const k1 = this.getKey(key1);
    const k2 = this.getKey(key2);
    if (!k1 || !k2) return 0;

    const idx1 = CIRCLE_OF_FIFTHS.indexOf(k1.root);
    const idx2 = CIRCLE_OF_FIFTHS.indexOf(k2.root);

    const dist = Math.abs(idx2 - idx1);
    return Math.min(dist, 12 - dist);
  }

  // =========================================================================
  // NEO-RIEMANNIAN TRANSFORMATIONS
  // =========================================================================

  /**
   * Apply P (Parallel) transformation to a key.
   */
  parallelTransform(keyName: string): string {
    const key = this.getKey(keyName);
    if (!key) return keyName;

    const result = parallelTransform(key.root, key.mode === 'major');
    return PITCH_CLASSES[result.pc] + (result.major ? '' : 'm');
  }

  /**
   * Apply L (Leading-tone) transformation to a key.
   */
  leadingToneTransform(keyName: string): string {
    const key = this.getKey(keyName);
    if (!key) return keyName;

    const result = leadingToneTransform(key.root, key.mode === 'major');
    return PITCH_CLASSES[result.pc] + (result.major ? '' : 'm');
  }

  /**
   * Apply R (Relative) transformation to a key.
   */
  relativeTransform(keyName: string): string {
    const key = this.getKey(keyName);
    if (!key) return keyName;

    const result = relativeTransform(key.root, key.mode === 'major');
    return PITCH_CLASSES[result.pc] + (result.major ? '' : 'm');
  }

  /**
   * Apply a sequence of PLR transformations.
   */
  applyTransformations(keyName: string, sequence: string): string {
    let currentKey = keyName;

    for (const t of sequence.toUpperCase()) {
      switch (t) {
        case 'P':
          currentKey = this.parallelTransform(currentKey);
          break;
        case 'L':
          currentKey = this.leadingToneTransform(currentKey);
          break;
        case 'R':
          currentKey = this.relativeTransform(currentKey);
          break;
      }
    }

    return currentKey;
  }

  // =========================================================================
  // KEY DETECTION
  // =========================================================================

  /**
   * Detect the most likely key from a set of pitch classes.
   */
  detectKey(pitchClasses: number[]): { key: string; confidence: number }[] {
    const results: { key: string; score: number }[] = [];
    const pcSet = new Set(pitchClasses);

    // Score each key based on how many pitch classes fit
    for (const keyName of this._keyVertexMap.keys()) {
      const key = this.getKey(keyName);
      if (!key) continue;

      // Get scale degrees for this key
      const scale = key.mode === 'major'
        ? [0, 2, 4, 5, 7, 9, 11] // Major scale intervals
        : [0, 2, 3, 5, 7, 8, 10]; // Natural minor scale intervals

      const scaleNotes = scale.map(interval => (key.root + interval) % 12);

      // Count matches
      let matches = 0;
      for (const pc of pitchClasses) {
        if (scaleNotes.includes(pc)) matches++;
      }

      const score = matches / pitchClasses.length;
      if (score > 0.5) {
        results.push({ key: keyName, score });
      }
    }

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);

    return results.slice(0, 5).map(r => ({
      key: r.key,
      confidence: r.score
    }));
  }

  // =========================================================================
  // POSITION TO KEY
  // =========================================================================

  /**
   * Get the nearest key to a 4D position.
   */
  positionToKey(position: Vector4D): string {
    const nearestVertex = this._lattice.findNearest(position);
    const keys = this._vertexKeyMap.get(nearestVertex);
    return keys?.[0] ?? 'C';
  }

  /**
   * Get key weights for a 4D position.
   */
  positionToKeyWeights(position: Vector4D, k: number = 4): Map<string, number> {
    const nearestVertices = this._lattice.findKNearest(position, k);
    const weights = new Map<string, number>();

    // Calculate distance-weighted contribution from each vertex
    let totalWeight = 0;
    const vertexWeights: { vertex: number; weight: number }[] = [];

    for (const vId of nearestVertices) {
      const v = this._lattice.getVertex(vId);
      if (!v) continue;

      const dist = magnitude([
        position[0] - v.coordinates[0],
        position[1] - v.coordinates[1],
        position[2] - v.coordinates[2],
        position[3] - v.coordinates[3]
      ]);

      const weight = 1 / (dist + 0.001);
      totalWeight += weight;
      vertexWeights.push({ vertex: vId, weight });
    }

    // Normalize and assign to keys
    for (const { vertex, weight } of vertexWeights) {
      const normalizedWeight = weight / totalWeight;
      const keys = this._vertexKeyMap.get(vertex) ?? [];

      for (const key of keys) {
        const current = weights.get(key) ?? 0;
        weights.set(key, current + normalizedWeight / keys.length);
      }
    }

    return weights;
  }

  // =========================================================================
  // STATISTICS
  // =========================================================================

  getStats(): Record<string, unknown> {
    return {
      totalKeys: this._keyVertexMap.size,
      majorKeys: [...this._keyVertexMap.keys()].filter(k => !k.includes('m')).length,
      minorKeys: [...this._keyVertexMap.keys()].filter(k => k.includes('m')).length,
      octatonicCollections: Object.fromEntries(
        [...this._octatonicCollections.entries()].map(([axis, coll]) => [
          axis,
          { name: coll.name, pitchClasses: coll.pitchClasses }
        ])
      )
    };
  }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

export function createMusicGeometryDomain(): MusicGeometryDomain {
  return new MusicGeometryDomain();
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  PITCH_CLASSES,
  CIRCLE_OF_FIFTHS,
  OCTATONIC_COLLECTIONS,
  TRINITY_OCTATONIC_MAP
};
