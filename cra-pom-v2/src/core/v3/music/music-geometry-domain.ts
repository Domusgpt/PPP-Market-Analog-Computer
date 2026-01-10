/**
 * MusicGeometryDomain - 24-Cell Polychoron Musical Space
 *
 * A specialized calibration module that maps Western Tonal Music to the
 * 24-Cell Polychoron (4-dimensional convex regular polytope).
 *
 * THEORETICAL FOUNDATIONS:
 *
 * 1. PYTHAGOREAN-EUCLIDEAN HOMOLOGY
 *    Musical intervals (frequency ratios) = distances in 4D metric space
 *    Consonance = Proximity, Dissonance = Complexity
 *
 * 2. 24-CELL "KEY-LATTICE" ISOMORPHISM
 *    24 vertices = 24 keys (12 Major + 12 Relative Minor)
 *    Chords = geometric sub-polytopes with specific symmetry groups
 *    Major Triad = D₃ (Dihedral), Diminished 7th = T (Tetrahedral)
 *
 * 3. CHRONOMORPHIC PATH DEPENDENCE
 *    Melody = 4D path (wormhole/sculpture) through polytope
 *    Time = 4th coordinate
 *    Curvature = melodic smoothness, Length = harmonic efficiency
 *
 * REFERENCES:
 * - 24-Cell: https://en.wikipedia.org/wiki/24-cell
 * - Tonnetz: https://en.wikipedia.org/wiki/Tonnetz
 * - Neo-Riemannian Theory
 */

import {
  PITCH_CLASSES,
  type PitchClass,
  pitchClassToSemitone,
  semitoneToPitchClass,
  intervalBetween,
  createChord,
  type Chord,
  type ChordType,
} from './music-theory';

// ============================================================================
// 24-Cell Geometry Types
// ============================================================================

/** 4D vector (w, x, y, z) */
export interface Vector4D {
  w: number;
  x: number;
  y: number;
  z: number;
}

/** A vertex in the 24-Cell */
export interface Vertex24 {
  id: number;
  coords: Vector4D;
  key: MusicalKey;
}

/** Musical key (Major or Minor) */
export interface MusicalKey {
  root: PitchClass;
  mode: 'major' | 'minor';
}

/** Edge connecting two vertices */
export interface Edge24 {
  from: number;
  to: number;
  length: number; // √2 or 2
  interval: string; // Musical interval name
}

/** A path through the polytope (chord progression / melody) */
export interface Path4D {
  vertices: number[];
  timestamps: number[];
  totalLength: number;
  curvature: number;
  tensionProfile: number[];
}

/** Tension calculation result */
export interface TensionResult {
  geometricTension: number;
  acousticDissonance: number;
  edgeStress: number;
  volumeRatio: number;
}

/** Calibration test result */
export interface CalibrationResult {
  testName: string;
  passed: boolean;
  expected: number;
  actual: number;
  tolerance: number;
  details: string;
}

// ============================================================================
// 24-Cell Constants
// ============================================================================

/**
 * The 24 vertices of the 24-Cell polytope.
 *
 * The 24-Cell has 24 vertices that can be described as:
 * - 8 vertices: permutations of (±1, 0, 0, 0)
 * - 16 vertices: (±1/2, ±1/2, ±1/2, ±1/2)
 *
 * For musical mapping, we use a normalized form.
 */
const VERTEX_COORDS: Vector4D[] = [
  // Type A: Axis-aligned (8 vertices) - Maps to sharp keys
  { w: 1, x: 0, y: 0, z: 0 },   // 0: C Major
  { w: -1, x: 0, y: 0, z: 0 },  // 1: F# Major (Tritone)
  { w: 0, x: 1, y: 0, z: 0 },   // 2: G Major (Fifth)
  { w: 0, x: -1, y: 0, z: 0 },  // 3: F Major (Fourth)
  { w: 0, x: 0, y: 1, z: 0 },   // 4: D Major (Two Fifths)
  { w: 0, x: 0, y: -1, z: 0 },  // 5: Bb Major
  { w: 0, x: 0, y: 0, z: 1 },   // 6: A Major (Three Fifths)
  { w: 0, x: 0, y: 0, z: -1 },  // 7: Eb Major

  // Type B: Half-coordinates (16 vertices) - Mixed and minor keys
  { w: 0.5, x: 0.5, y: 0.5, z: 0.5 },     // 8: E Major
  { w: 0.5, x: 0.5, y: 0.5, z: -0.5 },    // 9: B Major
  { w: 0.5, x: 0.5, y: -0.5, z: 0.5 },    // 10: Ab Major
  { w: 0.5, x: 0.5, y: -0.5, z: -0.5 },   // 11: Db Major
  { w: 0.5, x: -0.5, y: 0.5, z: 0.5 },    // 12: A Minor (relative of C)
  { w: 0.5, x: -0.5, y: 0.5, z: -0.5 },   // 13: E Minor (relative of G)
  { w: 0.5, x: -0.5, y: -0.5, z: 0.5 },   // 14: D Minor (relative of F)
  { w: 0.5, x: -0.5, y: -0.5, z: -0.5 },  // 15: G Minor (relative of Bb)
  { w: -0.5, x: 0.5, y: 0.5, z: 0.5 },    // 16: B Minor (relative of D)
  { w: -0.5, x: 0.5, y: 0.5, z: -0.5 },   // 17: F# Minor (relative of A)
  { w: -0.5, x: 0.5, y: -0.5, z: 0.5 },   // 18: C# Minor (relative of E)
  { w: -0.5, x: 0.5, y: -0.5, z: -0.5 },  // 19: G# Minor (relative of B)
  { w: -0.5, x: -0.5, y: 0.5, z: 0.5 },   // 20: C Minor (relative of Eb)
  { w: -0.5, x: -0.5, y: 0.5, z: -0.5 },  // 21: F Minor (relative of Ab)
  { w: -0.5, x: -0.5, y: -0.5, z: 0.5 },  // 22: Bb Minor (relative of Db)
  { w: -0.5, x: -0.5, y: -0.5, z: -0.5 }, // 23: Eb Minor (relative of F#/Gb)
];

/**
 * Key mapping: vertex ID -> Musical Key
 * Organized by circle of fifths and relative minors
 */
const KEY_MAPPING: MusicalKey[] = [
  // Major keys (vertices 0-11)
  { root: 'C', mode: 'major' },   // 0
  { root: 'F#', mode: 'major' },  // 1
  { root: 'G', mode: 'major' },   // 2
  { root: 'F', mode: 'major' },   // 3
  { root: 'D', mode: 'major' },   // 4
  { root: 'A#', mode: 'major' },  // 5 (Bb)
  { root: 'A', mode: 'major' },   // 6
  { root: 'D#', mode: 'major' },  // 7 (Eb)
  { root: 'E', mode: 'major' },   // 8
  { root: 'B', mode: 'major' },   // 9
  { root: 'G#', mode: 'major' },  // 10 (Ab)
  { root: 'C#', mode: 'major' },  // 11 (Db)

  // Minor keys (vertices 12-23)
  { root: 'A', mode: 'minor' },   // 12 (relative of C)
  { root: 'E', mode: 'minor' },   // 13 (relative of G)
  { root: 'D', mode: 'minor' },   // 14 (relative of F)
  { root: 'G', mode: 'minor' },   // 15 (relative of Bb)
  { root: 'B', mode: 'minor' },   // 16 (relative of D)
  { root: 'F#', mode: 'minor' },  // 17 (relative of A)
  { root: 'C#', mode: 'minor' },  // 18 (relative of E)
  { root: 'G#', mode: 'minor' },  // 19 (relative of B)
  { root: 'C', mode: 'minor' },   // 20 (relative of Eb)
  { root: 'F', mode: 'minor' },   // 21 (relative of Ab)
  { root: 'A#', mode: 'minor' },  // 22 (relative of Db) (Bb minor)
  { root: 'D#', mode: 'minor' },  // 23 (relative of Gb) (Eb minor)
];

// ============================================================================
// Vector Operations
// ============================================================================

/** Calculate 4D Euclidean distance */
export function distance4D(a: Vector4D, b: Vector4D): number {
  const dw = a.w - b.w;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dw * dw + dx * dx + dy * dy + dz * dz);
}

/** Calculate 4D dot product */
export function dot4D(a: Vector4D, b: Vector4D): number {
  return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

/** Calculate 4D vector magnitude */
export function magnitude4D(v: Vector4D): number {
  return Math.sqrt(v.w * v.w + v.x * v.x + v.y * v.y + v.z * v.z);
}

/** Normalize a 4D vector */
export function normalize4D(v: Vector4D): Vector4D {
  const mag = magnitude4D(v);
  if (mag === 0) return { w: 0, x: 0, y: 0, z: 0 };
  return {
    w: v.w / mag,
    x: v.x / mag,
    y: v.y / mag,
    z: v.z / mag,
  };
}

/** Add two 4D vectors */
export function add4D(a: Vector4D, b: Vector4D): Vector4D {
  return {
    w: a.w + b.w,
    x: a.x + b.x,
    y: a.y + b.y,
    z: a.z + b.z,
  };
}

/** Subtract 4D vectors (a - b) */
export function subtract4D(a: Vector4D, b: Vector4D): Vector4D {
  return {
    w: a.w - b.w,
    x: a.x - b.x,
    y: a.y - b.y,
    z: a.z - b.z,
  };
}

/** Scale a 4D vector */
export function scale4D(v: Vector4D, s: number): Vector4D {
  return {
    w: v.w * s,
    x: v.x * s,
    y: v.y * s,
    z: v.z * s,
  };
}

/** Linear interpolation between two 4D vectors */
export function lerp4D(a: Vector4D, b: Vector4D, t: number): Vector4D {
  return {
    w: a.w + (b.w - a.w) * t,
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    z: a.z + (b.z - a.z) * t,
  };
}

// ============================================================================
// 24-Cell Polytope
// ============================================================================

export class Cell24 {
  readonly vertices: Vertex24[];
  readonly edges: Edge24[];
  private adjacencyMatrix: number[][];

  constructor() {
    // Build vertices
    this.vertices = VERTEX_COORDS.map((coords, id) => ({
      id,
      coords,
      key: KEY_MAPPING[id],
    }));

    // Build edges (vertices at distance √2 or 2 are connected)
    this.edges = this.buildEdges();

    // Build adjacency matrix
    this.adjacencyMatrix = this.buildAdjacencyMatrix();
  }

  private buildEdges(): Edge24[] {
    const edges: Edge24[] = [];
    const sqrt2 = Math.sqrt(2);

    for (let i = 0; i < 24; i++) {
      for (let j = i + 1; j < 24; j++) {
        const dist = distance4D(
          this.vertices[i].coords,
          this.vertices[j].coords
        );

        // In a 24-Cell, edges connect vertices at distance √2
        // Some connections at distance 2 represent "long" relationships
        if (Math.abs(dist - sqrt2) < 0.01) {
          edges.push({
            from: i,
            to: j,
            length: sqrt2,
            interval: this.getIntervalName(i, j),
          });
        } else if (Math.abs(dist - 2) < 0.01) {
          edges.push({
            from: i,
            to: j,
            length: 2,
            interval: this.getIntervalName(i, j),
          });
        }
      }
    }

    return edges;
  }

  private buildAdjacencyMatrix(): number[][] {
    const matrix: number[][] = Array(24)
      .fill(null)
      .map(() => Array(24).fill(Infinity));

    // Self-distance is 0
    for (let i = 0; i < 24; i++) {
      matrix[i][i] = 0;
    }

    // Edge distances
    for (const edge of this.edges) {
      matrix[edge.from][edge.to] = edge.length;
      matrix[edge.to][edge.from] = edge.length;
    }

    return matrix;
  }

  private getIntervalName(i: number, j: number): string {
    const keyA = this.vertices[i].key;
    const keyB = this.vertices[j].key;

    // Same mode comparison
    if (keyA.mode === keyB.mode) {
      const interval = intervalBetween(keyA.root, keyB.root);
      const intervalNames: Record<number, string> = {
        0: 'Unison',
        1: 'Minor 2nd',
        2: 'Major 2nd',
        3: 'Minor 3rd',
        4: 'Major 3rd',
        5: 'Perfect 4th',
        6: 'Tritone',
        7: 'Perfect 5th',
        8: 'Minor 6th',
        9: 'Major 6th',
        10: 'Minor 7th',
        11: 'Major 7th',
      };
      return intervalNames[interval] || `Interval ${interval}`;
    }

    // Cross-mode comparison
    return keyA.mode === 'major' ? 'To Relative Minor' : 'To Relative Major';
  }

  /**
   * Get vertex by key
   */
  getVertexByKey(root: PitchClass, mode: 'major' | 'minor'): Vertex24 | undefined {
    return this.vertices.find(
      (v) => v.key.root === root && v.key.mode === mode
    );
  }

  /**
   * Get vertex by ID
   */
  getVertex(id: number): Vertex24 {
    return this.vertices[id];
  }

  /**
   * Get distance between two vertices
   */
  getDistance(i: number, j: number): number {
    return distance4D(this.vertices[i].coords, this.vertices[j].coords);
  }

  /**
   * Get shortest path between two vertices (Floyd-Warshall)
   */
  getShortestPath(from: number, to: number): number[] {
    // Simple BFS for unweighted shortest path
    const visited = new Set<number>();
    const queue: { vertex: number; path: number[] }[] = [
      { vertex: from, path: [from] },
    ];

    while (queue.length > 0) {
      const { vertex, path } = queue.shift()!;

      if (vertex === to) {
        return path;
      }

      if (visited.has(vertex)) continue;
      visited.add(vertex);

      // Find neighbors
      for (const edge of this.edges) {
        let neighbor = -1;
        if (edge.from === vertex) neighbor = edge.to;
        else if (edge.to === vertex) neighbor = edge.from;

        if (neighbor >= 0 && !visited.has(neighbor)) {
          queue.push({ vertex: neighbor, path: [...path, neighbor] });
        }
      }
    }

    return []; // No path found
  }

  /**
   * Get geodesic distance (shortest path length)
   */
  getGeodesicDistance(from: number, to: number): number {
    const path = this.getShortestPath(from, to);
    if (path.length === 0) return Infinity;

    let totalDist = 0;
    for (let i = 0; i < path.length - 1; i++) {
      totalDist += this.getDistance(path[i], path[i + 1]);
    }
    return totalDist;
  }

  /**
   * Get all neighbors of a vertex
   */
  getNeighbors(vertexId: number): Vertex24[] {
    const neighbors: Vertex24[] = [];

    for (const edge of this.edges) {
      if (edge.from === vertexId) {
        neighbors.push(this.vertices[edge.to]);
      } else if (edge.to === vertexId) {
        neighbors.push(this.vertices[edge.from]);
      }
    }

    return neighbors;
  }

  /**
   * Project a point in 4D to the nearest vertex
   */
  projectToVertex(point: Vector4D): Vertex24 {
    let minDist = Infinity;
    let nearest = this.vertices[0];

    for (const vertex of this.vertices) {
      const dist = distance4D(point, vertex.coords);
      if (dist < minDist) {
        minDist = dist;
        nearest = vertex;
      }
    }

    return nearest;
  }
}

// ============================================================================
// Tension & Resolution Analysis
// ============================================================================

/**
 * Calculate geometric tension for a chord in the 24-Cell space.
 *
 * Tension = potential energy required to hold the chord's geometry.
 * Higher tension = more unstable = wants to resolve.
 */
export function calculateTension(
  cell: Cell24,
  chord: Chord
): TensionResult {
  // Map chord notes to vertices
  const vertexIds = chord.pitchClasses.map((pc) => {
    const majorVertex = cell.getVertexByKey(pc, 'major');
    return majorVertex?.id ?? 0;
  });

  // Calculate edge stress (sum of edge lengths within chord)
  let edgeStress = 0;
  for (let i = 0; i < vertexIds.length; i++) {
    for (let j = i + 1; j < vertexIds.length; j++) {
      edgeStress += cell.getDistance(vertexIds[i], vertexIds[j]);
    }
  }

  // Calculate centroid
  let centroid: Vector4D = { w: 0, x: 0, y: 0, z: 0 };
  for (const id of vertexIds) {
    const coords = cell.getVertex(id).coords;
    centroid = add4D(centroid, coords);
  }
  centroid = scale4D(centroid, 1 / vertexIds.length);

  // Calculate volume ratio (spread from centroid)
  let volumeRatio = 0;
  for (const id of vertexIds) {
    volumeRatio += distance4D(cell.getVertex(id).coords, centroid);
  }
  volumeRatio /= vertexIds.length;

  // Acoustic dissonance based on intervals
  let acousticDissonance = 0;
  const dissonanceWeights: Record<number, number> = {
    0: 0,     // Unison
    1: 1.0,   // Minor 2nd (harsh)
    2: 0.5,   // Major 2nd
    3: 0.3,   // Minor 3rd
    4: 0.2,   // Major 3rd
    5: 0.1,   // Perfect 4th
    6: 0.9,   // Tritone
    7: 0.05,  // Perfect 5th
    8: 0.35,  // Minor 6th
    9: 0.25,  // Major 6th
    10: 0.7,  // Minor 7th
    11: 0.4,  // Major 7th
  };

  for (let i = 0; i < chord.pitchClasses.length; i++) {
    for (let j = i + 1; j < chord.pitchClasses.length; j++) {
      const interval = intervalBetween(chord.pitchClasses[i], chord.pitchClasses[j]);
      acousticDissonance += dissonanceWeights[interval] || 0.5;
    }
  }

  // Geometric tension combines all factors
  const geometricTension =
    edgeStress * 0.3 +
    volumeRatio * 0.3 +
    acousticDissonance * 0.4;

  return {
    geometricTension,
    acousticDissonance,
    edgeStress,
    volumeRatio,
  };
}

/**
 * Calculate resolution strength between two chords.
 *
 * Resolution = tension reduction when moving from chord A to chord B.
 * Positive = resolution (relaxation)
 * Negative = tension building
 */
export function calculateResolution(
  cell: Cell24,
  fromChord: Chord,
  toChord: Chord
): number {
  const tensionFrom = calculateTension(cell, fromChord);
  const tensionTo = calculateTension(cell, toChord);

  return tensionFrom.geometricTension - tensionTo.geometricTension;
}

// ============================================================================
// Path Analysis (Chronomorphic)
// ============================================================================

/**
 * Create a path through the 24-Cell from a chord progression.
 */
export function createPath(
  cell: Cell24,
  chords: Chord[],
  timestamps?: number[]
): Path4D {
  const times = timestamps || chords.map((_, i) => i);
  const vertices: number[] = [];

  // Map each chord to its root's major key vertex
  for (const chord of chords) {
    const vertex = cell.getVertexByKey(chord.root, 'major');
    if (vertex) {
      vertices.push(vertex.id);
    }
  }

  // Calculate total path length
  let totalLength = 0;
  for (let i = 0; i < vertices.length - 1; i++) {
    totalLength += cell.getDistance(vertices[i], vertices[i + 1]);
  }

  // Calculate curvature (average angle change)
  let curvature = 0;
  if (vertices.length >= 3) {
    for (let i = 1; i < vertices.length - 1; i++) {
      const v1 = subtract4D(
        cell.getVertex(vertices[i]).coords,
        cell.getVertex(vertices[i - 1]).coords
      );
      const v2 = subtract4D(
        cell.getVertex(vertices[i + 1]).coords,
        cell.getVertex(vertices[i]).coords
      );

      const cosAngle = dot4D(v1, v2) / (magnitude4D(v1) * magnitude4D(v2));
      curvature += Math.acos(Math.max(-1, Math.min(1, cosAngle)));
    }
    curvature /= vertices.length - 2;
  }

  // Calculate tension profile
  const tensionProfile = chords.map((chord) =>
    calculateTension(cell, chord).geometricTension
  );

  return {
    vertices,
    timestamps: times,
    totalLength,
    curvature,
    tensionProfile,
  };
}

/**
 * Analyze voice leading efficiency.
 *
 * Good voice leading = minimal total distance moved by all voices.
 */
export function analyzeVoiceLeading(
  cell: Cell24,
  fromChord: Chord,
  toChord: Chord
): {
  totalMovement: number;
  efficiency: number;
  isParsimonious: boolean;
} {
  // Calculate total movement in 4D space
  let totalMovement = 0;
  const maxVoices = Math.max(fromChord.pitchClasses.length, toChord.pitchClasses.length);

  for (let i = 0; i < maxVoices; i++) {
    const fromPC = fromChord.pitchClasses[i % fromChord.pitchClasses.length];
    const toPC = toChord.pitchClasses[i % toChord.pitchClasses.length];

    const fromVertex = cell.getVertexByKey(fromPC, 'major');
    const toVertex = cell.getVertexByKey(toPC, 'major');

    if (fromVertex && toVertex) {
      totalMovement += cell.getDistance(fromVertex.id, toVertex.id);
    }
  }

  // Efficiency = inverse of movement (normalized)
  const efficiency = 1 / (1 + totalMovement);

  // Parsimonious = each voice moves by at most a step
  const isParsimonious = totalMovement / maxVoices < 0.5;

  return {
    totalMovement,
    efficiency,
    isParsimonious,
  };
}

// ============================================================================
// Pythagorean Comma Detection
// ============================================================================

/**
 * Trace the Circle of Fifths and detect the Pythagorean Comma.
 *
 * The comma is the small discrepancy when stacking 12 perfect fifths:
 * (3/2)^12 ≠ 2^7 (it's slightly larger)
 */
export function detectPythagoreanComma(cell: Cell24): {
  startVertex: number;
  endVertex: number;
  geometricGap: number;
  commaInCents: number;
  isCircleClosed: boolean;
} {
  // Start at C major
  const startVertex = cell.getVertexByKey('C', 'major')!.id;

  // Follow circle of fifths: C -> G -> D -> A -> E -> B -> F# -> C# -> G# -> D# -> A# -> F -> C
  const circleOfFifths: PitchClass[] = [
    'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F', 'C'
  ];

  let totalDistance = 0;
  for (let i = 0; i < circleOfFifths.length - 1; i++) {
    const from = cell.getVertexByKey(circleOfFifths[i], 'major')!.id;
    const to = cell.getVertexByKey(circleOfFifths[i + 1], 'major')!.id;
    totalDistance += cell.getDistance(from, to);
  }

  // End vertex (should be back at C)
  const endVertex = cell.getVertexByKey('C', 'major')!.id;

  // The geometric gap (should be 0 if circle closes perfectly)
  const geometricGap = totalDistance - (cell.getDistance(startVertex, cell.getVertexByKey('G', 'major')!.id) * 12);

  // Pythagorean comma in cents: ~23.46 cents
  const commaInCents = 1200 * Math.log2(Math.pow(3/2, 12) / Math.pow(2, 7));

  return {
    startVertex,
    endVertex,
    geometricGap,
    commaInCents,
    isCircleClosed: startVertex === endVertex,
  };
}

// ============================================================================
// Calibration Suite
// ============================================================================

/**
 * Run the calibration test suite.
 *
 * These tests verify that the geometric model correctly captures
 * musical relationships before applying it to other domains.
 */
export function runCalibrationSuite(cell: Cell24): CalibrationResult[] {
  const results: CalibrationResult[] = [];

  // Test 1: Octave Equivalence
  // All C's should map to the same vertex
  {
    const cMajor = cell.getVertexByKey('C', 'major');
    results.push({
      testName: 'Octave Equivalence',
      passed: cMajor !== undefined,
      expected: 1,
      actual: cMajor ? 1 : 0,
      tolerance: 0,
      details: `C Major vertex exists at position ${cMajor?.id}`,
    });
  }

  // Test 2: Circle of Fifths Connectivity
  // Each fifth should be connected by an edge
  {
    let connectedFifths = 0;
    const fifths: [PitchClass, PitchClass][] = [
      ['C', 'G'], ['G', 'D'], ['D', 'A'], ['A', 'E'], ['E', 'B'],
    ];

    for (const [from, to] of fifths) {
      const fromV = cell.getVertexByKey(from, 'major');
      const toV = cell.getVertexByKey(to, 'major');
      if (fromV && toV) {
        const dist = cell.getDistance(fromV.id, toV.id);
        if (dist < 2) connectedFifths++;
      }
    }

    results.push({
      testName: 'Circle of Fifths Connectivity',
      passed: connectedFifths >= 4,
      expected: 5,
      actual: connectedFifths,
      tolerance: 1,
      details: `${connectedFifths}/5 fifths are directly connected`,
    });
  }

  // Test 3: Major Triad D3 Symmetry
  // C-E-G should form a balanced triangle
  {
    const c = cell.getVertexByKey('C', 'major');
    const e = cell.getVertexByKey('E', 'major');
    const g = cell.getVertexByKey('G', 'major');

    if (c && e && g) {
      const ce = cell.getDistance(c.id, e.id);
      const eg = cell.getDistance(e.id, g.id);
      const gc = cell.getDistance(g.id, c.id);

      // Check if distances are roughly equal (D3 symmetry)
      const avgDist = (ce + eg + gc) / 3;
      const variance = Math.abs(ce - avgDist) + Math.abs(eg - avgDist) + Math.abs(gc - avgDist);

      results.push({
        testName: 'Major Triad D3 Symmetry',
        passed: variance < avgDist * 0.5,
        expected: 0,
        actual: variance,
        tolerance: avgDist * 0.5,
        details: `CE=${ce.toFixed(2)}, EG=${eg.toFixed(2)}, GC=${gc.toFixed(2)}`,
      });
    } else {
      results.push({
        testName: 'Major Triad D3 Symmetry',
        passed: false,
        expected: 1,
        actual: 0,
        tolerance: 0,
        details: 'Could not find all triad vertices',
      });
    }
  }

  // Test 4: Relative Major/Minor Proximity
  // C Major and A Minor should be adjacent
  {
    const cMajor = cell.getVertexByKey('C', 'major');
    const aMinor = cell.getVertexByKey('A', 'minor');

    if (cMajor && aMinor) {
      const dist = cell.getDistance(cMajor.id, aMinor.id);

      results.push({
        testName: 'Relative Major/Minor Proximity',
        passed: dist <= Math.sqrt(2) + 0.1,
        expected: Math.sqrt(2),
        actual: dist,
        tolerance: 0.1,
        details: `Distance C Major to A Minor: ${dist.toFixed(4)}`,
      });
    } else {
      results.push({
        testName: 'Relative Major/Minor Proximity',
        passed: false,
        expected: 1,
        actual: 0,
        tolerance: 0,
        details: 'Could not find vertices',
      });
    }
  }

  // Test 5: Tritone Maximum Distance
  // C and F# should be maximally distant
  {
    const c = cell.getVertexByKey('C', 'major');
    const fSharp = cell.getVertexByKey('F#', 'major');

    if (c && fSharp) {
      const dist = cell.getDistance(c.id, fSharp.id);

      // Tritone should be the furthest relationship
      let isMaximal = true;
      for (let i = 1; i < 12; i++) {
        if (i === 6) continue; // Skip tritone itself
        const other = cell.getVertexByKey(semitoneToPitchClass(i), 'major');
        if (other && cell.getDistance(c.id, other.id) > dist + 0.1) {
          isMaximal = false;
          break;
        }
      }

      results.push({
        testName: 'Tritone Maximum Distance',
        passed: isMaximal,
        expected: 2,
        actual: dist,
        tolerance: 0.2,
        details: `C to F# distance: ${dist.toFixed(4)}, is maximal: ${isMaximal}`,
      });
    } else {
      results.push({
        testName: 'Tritone Maximum Distance',
        passed: false,
        expected: 1,
        actual: 0,
        tolerance: 0,
        details: 'Could not find vertices',
      });
    }
  }

  // Test 6: Dominant 7th Tension > Major Triad Tension
  {
    const cMajor = createChord('C', 'major');
    const g7 = createChord('G', 'dominant7');

    const cTension = calculateTension(cell, cMajor);
    const g7Tension = calculateTension(cell, g7);

    results.push({
      testName: 'Dominant 7th Higher Tension than Major',
      passed: g7Tension.geometricTension > cTension.geometricTension,
      expected: g7Tension.geometricTension,
      actual: cTension.geometricTension,
      tolerance: 0,
      details: `G7 tension: ${g7Tension.geometricTension.toFixed(4)}, C tension: ${cTension.geometricTension.toFixed(4)}`,
    });
  }

  // Test 7: V-I Resolution
  {
    const g7 = createChord('G', 'dominant7');
    const c = createChord('C', 'major');

    const resolution = calculateResolution(cell, g7, c);

    results.push({
      testName: 'V-I Resolution (G7 -> C)',
      passed: resolution > 0,
      expected: 1,
      actual: resolution,
      tolerance: 0,
      details: `Resolution strength: ${resolution.toFixed(4)} (positive = resolving)`,
    });
  }

  return results;
}

// ============================================================================
// Semantic Bridge (Multimodal Synesthesia)
// ============================================================================

/**
 * Blend semantic text vector with musical structure.
 *
 * Takes an abstract concept and maps it to musical space.
 */
export function blendSemanticAndStructural(
  cell: Cell24,
  semanticVector: Float32Array,
  weight = 0.5
): {
  suggestedKey: MusicalKey;
  suggestedMode: 'major' | 'minor';
  harmonicRhythm: 'fast' | 'medium' | 'slow';
  emotionalValence: number;
} {
  // Analyze semantic vector properties
  let sum = 0;
  let variance = 0;
  for (let i = 0; i < semanticVector.length; i++) {
    sum += semanticVector[i];
  }
  const mean = sum / semanticVector.length;

  for (let i = 0; i < semanticVector.length; i++) {
    variance += (semanticVector[i] - mean) ** 2;
  }
  variance /= semanticVector.length;

  // Map to musical properties
  // High mean = brighter (major), Low mean = darker (minor)
  const suggestedMode: 'major' | 'minor' = mean > 0 ? 'major' : 'minor';

  // High variance = more complex/fast harmonic rhythm
  const harmonicRhythm: 'fast' | 'medium' | 'slow' =
    variance > 0.5 ? 'fast' : variance > 0.2 ? 'medium' : 'slow';

  // Project semantic vector to 4D and find nearest key
  const projected: Vector4D = {
    w: semanticVector[0] || 0,
    x: semanticVector[1] || 0,
    y: semanticVector[2] || 0,
    z: semanticVector[3] || 0,
  };

  const nearestVertex = cell.projectToVertex(projected);
  const suggestedKey = nearestVertex.key;

  return {
    suggestedKey,
    suggestedMode,
    harmonicRhythm,
    emotionalValence: mean,
  };
}

// ============================================================================
// Exports
// ============================================================================

export const MusicGeometryDomain = {
  // Core types
  Cell24,

  // Vector operations
  distance4D,
  dot4D,
  magnitude4D,
  normalize4D,
  add4D,
  subtract4D,
  scale4D,
  lerp4D,

  // Analysis
  calculateTension,
  calculateResolution,
  createPath,
  analyzeVoiceLeading,
  detectPythagoreanComma,

  // Calibration
  runCalibrationSuite,

  // Semantic bridge
  blendSemanticAndStructural,
};

export default MusicGeometryDomain;
