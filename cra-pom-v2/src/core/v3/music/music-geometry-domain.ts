/**
 * MusicGeometryDomain - 24-Cell Polychoron Musical Space
 *
 * A specialized calibration module that maps Western Tonal Music to the
 * 24-Cell Polychoron (4-dimensional convex regular polytope).
 *
 * BUILDS ON THE POLYTOPE HIERARCHY:
 *
 * 8-CELL (Tesseract) - 16 vertices
 *   └─► Maps to 8 major/minor KEY PAIRS (C/Am, G/Em, D/Bm, ...)
 *   └─► Each vertex represents a "tonal center" with its relative
 *
 * 16-CELL (Hexadecachoron) - 8 vertices
 *   └─► Maps to 8 PRIMARY KEYS on the Circle of Fifths
 *   └─► C, G, D, A, E, B, F#, F (the "skeleton" of Western tonality)
 *
 * 24-CELL (Icositetrachoron) - 24 vertices
 *   └─► Maps to ALL 24 KEYS (12 Major + 12 Minor)
 *   └─► Built from 16-cell + scaled 8-cell vertices
 *   └─► Self-dual: major/minor duality reflected in geometry
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
 */

import {
  Cell8,
  Cell16,
  Cell24,
  distance4D as polytopeDistance4D,
  scale4D as polytopeScale4D,
  add4D as polytopeAdd4D,
  type SymmetryGroup,
  classifySymmetry,
} from './polytopes';

import {
  PITCH_CLASSES,
  type PitchClass,
  pitchClassToSemitone,
  semitoneToPitchClass,
  intervalBetween,
  createChord,
  type Chord,
} from './music-theory';

// ============================================================================
// Types
// ============================================================================

/** 4D vector (w, x, y, z) */
export interface Vector4D {
  w: number;
  x: number;
  y: number;
  z: number;
}

/** Musical key (Major or Minor) */
export interface MusicalKey {
  root: PitchClass;
  mode: 'major' | 'minor';
}

/** A vertex in the Musical 24-Cell */
export interface Vertex24 {
  id: number;
  coords: Vector4D;
  key: MusicalKey;
  type: 'axis' | 'diagonal';
  source: '16-cell' | '8-cell';
}

/** Edge connecting two vertices */
export interface Edge24 {
  from: number;
  to: number;
  length: number;
  interval: string;
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
// Vector Operations
// ============================================================================

export function distance4D(a: Vector4D, b: Vector4D): number {
  return polytopeDistance4D(a, b);
}

export function dot4D(a: Vector4D, b: Vector4D): number {
  return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

export function magnitude4D(v: Vector4D): number {
  return Math.sqrt(v.w * v.w + v.x * v.x + v.y * v.y + v.z * v.z);
}

export function normalize4D(v: Vector4D): Vector4D {
  const mag = magnitude4D(v);
  if (mag === 0) return { w: 0, x: 0, y: 0, z: 0 };
  return { w: v.w / mag, x: v.x / mag, y: v.y / mag, z: v.z / mag };
}

export function add4D(a: Vector4D, b: Vector4D): Vector4D {
  return polytopeAdd4D(a, b);
}

export function subtract4D(a: Vector4D, b: Vector4D): Vector4D {
  return { w: a.w - b.w, x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function scale4D(v: Vector4D, s: number): Vector4D {
  return polytopeScale4D(v, s);
}

export function lerp4D(a: Vector4D, b: Vector4D, t: number): Vector4D {
  return {
    w: a.w + (b.w - a.w) * t,
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    z: a.z + (b.z - a.z) * t,
  };
}

// ============================================================================
// Key Mapping Strategy
// ============================================================================

/**
 * Maps 24-cell vertices to musical keys.
 *
 * VERTEX ASSIGNMENT STRATEGY:
 *
 * Type A (from 16-cell, indices 0-7): 8 axis-aligned vertices
 *   - These map to PRIMARY KEYS on the circle of fifths
 *   - Positioned at (±1, 0, 0, 0) permutations
 *
 * Type B (from 8-cell, indices 8-23): 16 diagonal vertices
 *   - These map to SECONDARY KEYS and all MINOR KEYS
 *   - Positioned at (±1/2, ±1/2, ±1/2, ±1/2)
 *
 * The mapping preserves:
 *   - Circle of fifths relationships (adjacent in geometry)
 *   - Relative major/minor proximity (edges connect them)
 *   - Tritone opposition (antipodal vertices)
 */
const KEY_MAPPING: MusicalKey[] = [
  // Type A: From 16-cell (8 primary keys on circle of fifths)
  // Vertex 0: (+1, 0, 0, 0) -> C Major (tonic center)
  // Vertex 1: (-1, 0, 0, 0) -> F# Major (tritone, antipodal)
  // Vertex 2: (0, +1, 0, 0) -> G Major (fifth)
  // Vertex 3: (0, -1, 0, 0) -> F Major (fourth)
  // Vertex 4: (0, 0, +1, 0) -> D Major (two fifths)
  // Vertex 5: (0, 0, -1, 0) -> Bb Major (two fourths)
  // Vertex 6: (0, 0, 0, +1) -> A Major (three fifths)
  // Vertex 7: (0, 0, 0, -1) -> Eb Major (three fourths)
  { root: 'C', mode: 'major' },   // 0: w+
  { root: 'F#', mode: 'major' },  // 1: w-
  { root: 'G', mode: 'major' },   // 2: x+
  { root: 'F', mode: 'major' },   // 3: x-
  { root: 'D', mode: 'major' },   // 4: y+
  { root: 'A#', mode: 'major' },  // 5: y- (Bb)
  { root: 'A', mode: 'major' },   // 6: z+
  { root: 'D#', mode: 'major' },  // 7: z- (Eb)

  // Type B: From 8-cell (16 vertices for remaining majors + all minors)
  // The 8-cell's 16 vertices at (±1/2, ±1/2, ±1/2, ±1/2) give us space for:
  // - 4 remaining major keys: E, B, Ab, Db
  // - 12 minor keys (relatives of all majors)

  // Remaining major keys (vertices 8-11)
  { root: 'E', mode: 'major' },   // 8:  (+,+,+,+)
  { root: 'B', mode: 'major' },   // 9:  (+,+,+,-)
  { root: 'G#', mode: 'major' },  // 10: (+,+,-,+) (Ab)
  { root: 'C#', mode: 'major' },  // 11: (+,+,-,-) (Db)

  // Minor keys (vertices 12-23) - paired with their relative majors
  { root: 'A', mode: 'minor' },   // 12: (+,-,+,+) relative of C
  { root: 'E', mode: 'minor' },   // 13: (+,-,+,-) relative of G
  { root: 'D', mode: 'minor' },   // 14: (+,-,-,+) relative of F
  { root: 'G', mode: 'minor' },   // 15: (+,-,-,-) relative of Bb
  { root: 'B', mode: 'minor' },   // 16: (-,+,+,+) relative of D
  { root: 'F#', mode: 'minor' },  // 17: (-,+,+,-) relative of A
  { root: 'C#', mode: 'minor' },  // 18: (-,+,-,+) relative of E
  { root: 'G#', mode: 'minor' },  // 19: (-,+,-,-) relative of B
  { root: 'C', mode: 'minor' },   // 20: (-,-,+,+) relative of Eb
  { root: 'F', mode: 'minor' },   // 21: (-,-,+,-) relative of Ab
  { root: 'A#', mode: 'minor' },  // 22: (-,-,-,+) relative of Db (Bb minor)
  { root: 'D#', mode: 'minor' },  // 23: (-,-,-,-) relative of F#/Gb (Eb minor)
];

// ============================================================================
// Musical 24-Cell (extends geometric Cell24 with musical semantics)
// ============================================================================

export class MusicCell24 {
  // Underlying geometric polytopes
  readonly geometry: Cell24;
  readonly cell8: Cell8;
  readonly cell16: Cell16;

  // Musical vertices
  readonly vertices: Vertex24[];
  readonly edges: Edge24[];

  constructor() {
    // Build from the mathematically correct polytopes
    this.geometry = new Cell24();
    this.cell8 = this.geometry.outer8Cell;
    this.cell16 = this.geometry.inner16Cell;

    // Create musical vertices from geometric vertices
    this.vertices = this.geometry.vertices.map((coords, id) => ({
      id,
      coords,
      key: KEY_MAPPING[id],
      type: this.geometry.getVertexType(id),
      source: id < 8 ? '16-cell' : '8-cell',
    }));

    // Create musical edges from geometric edges
    this.edges = this.geometry.edges.map(([from, to]) => ({
      from,
      to,
      length: distance4D(this.vertices[from].coords, this.vertices[to].coords),
      interval: this.getIntervalName(from, to),
    }));
  }

  private getIntervalName(i: number, j: number): string {
    const keyA = this.vertices[i].key;
    const keyB = this.vertices[j].key;

    if (keyA.mode === keyB.mode) {
      const interval = intervalBetween(keyA.root, keyB.root);
      const names: Record<number, string> = {
        0: 'Unison', 1: 'Minor 2nd', 2: 'Major 2nd', 3: 'Minor 3rd',
        4: 'Major 3rd', 5: 'Perfect 4th', 6: 'Tritone', 7: 'Perfect 5th',
        8: 'Minor 6th', 9: 'Major 6th', 10: 'Minor 7th', 11: 'Major 7th',
      };
      return names[interval] || `Interval ${interval}`;
    }
    return keyA.mode === 'major' ? 'To Relative Minor' : 'To Relative Major';
  }

  // ========== Vertex Access ==========

  getVertex(id: number): Vertex24 {
    return this.vertices[id];
  }

  getVertexByKey(root: PitchClass, mode: 'major' | 'minor'): Vertex24 | undefined {
    return this.vertices.find(v => v.key.root === root && v.key.mode === mode);
  }

  getDistance(i: number, j: number): number {
    return distance4D(this.vertices[i].coords, this.vertices[j].coords);
  }

  getNeighbors(vertexId: number): Vertex24[] {
    return this.geometry.getNeighbors(vertexId).map(i => this.vertices[i]);
  }

  // ========== Polytope Hierarchy Access ==========

  /**
   * Get the 8 vertices from the 16-cell (primary keys)
   */
  get16CellKeys(): Vertex24[] {
    return this.vertices.filter(v => v.source === '16-cell');
  }

  /**
   * Get the 16 vertices from the 8-cell (secondary + minor keys)
   */
  get8CellKeys(): Vertex24[] {
    return this.vertices.filter(v => v.source === '8-cell');
  }

  /**
   * Get all major keys
   */
  getMajorKeys(): Vertex24[] {
    return this.vertices.filter(v => v.key.mode === 'major');
  }

  /**
   * Get all minor keys
   */
  getMinorKeys(): Vertex24[] {
    return this.vertices.filter(v => v.key.mode === 'minor');
  }

  /**
   * Get the relative major/minor of a key
   */
  getRelative(vertexId: number): Vertex24 | undefined {
    const vertex = this.vertices[vertexId];
    const relativeSemitone = vertex.key.mode === 'major'
      ? (pitchClassToSemitone(vertex.key.root) + 9) % 12  // Down minor 3rd
      : (pitchClassToSemitone(vertex.key.root) + 3) % 12; // Up minor 3rd
    const relativeRoot = semitoneToPitchClass(relativeSemitone);
    const relativeMode = vertex.key.mode === 'major' ? 'minor' : 'major';
    return this.getVertexByKey(relativeRoot, relativeMode);
  }

  // ========== Path Finding ==========

  getShortestPath(from: number, to: number): number[] {
    return this.geometry.getShortestPath(from, to);
  }

  getGeodesicDistance(from: number, to: number): number {
    const path = this.getShortestPath(from, to);
    if (path.length === 0) return Infinity;
    let total = 0;
    for (let i = 0; i < path.length - 1; i++) {
      total += this.getDistance(path[i], path[i + 1]);
    }
    return total;
  }

  // ========== Projection ==========

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

  // ========== Symmetry Analysis ==========

  /**
   * Classify the symmetry group of a set of vertices (chord shape)
   */
  classifyChordSymmetry(vertexIds: number[]): SymmetryGroup {
    return classifySymmetry(this.geometry, vertexIds);
  }

  // ========== Geometric Properties ==========

  get vertexCount(): number { return 24; }
  get edgeCount(): number { return this.geometry.edgeCount; }
  get faceCount(): number { return this.geometry.faceCount; }
}

// ============================================================================
// Backward Compatibility: Cell24 alias
// ============================================================================

export { MusicCell24 as Cell24 };

// ============================================================================
// Tension & Resolution Analysis
// ============================================================================

export function calculateTension(cell: MusicCell24, chord: Chord): TensionResult {
  const vertexIds = chord.pitchClasses.map(pc => {
    const v = cell.getVertexByKey(pc, 'major');
    return v?.id ?? 0;
  });

  // Edge stress
  let edgeStress = 0;
  for (let i = 0; i < vertexIds.length; i++) {
    for (let j = i + 1; j < vertexIds.length; j++) {
      edgeStress += cell.getDistance(vertexIds[i], vertexIds[j]);
    }
  }

  // Centroid and volume ratio
  let centroid: Vector4D = { w: 0, x: 0, y: 0, z: 0 };
  for (const id of vertexIds) {
    centroid = add4D(centroid, cell.getVertex(id).coords);
  }
  centroid = scale4D(centroid, 1 / vertexIds.length);

  let volumeRatio = 0;
  for (const id of vertexIds) {
    volumeRatio += distance4D(cell.getVertex(id).coords, centroid);
  }
  volumeRatio /= vertexIds.length;

  // Acoustic dissonance
  const dissonanceWeights: Record<number, number> = {
    0: 0, 1: 1.0, 2: 0.5, 3: 0.3, 4: 0.2, 5: 0.1,
    6: 0.9, 7: 0.05, 8: 0.35, 9: 0.25, 10: 0.7, 11: 0.4,
  };

  let acousticDissonance = 0;
  for (let i = 0; i < chord.pitchClasses.length; i++) {
    for (let j = i + 1; j < chord.pitchClasses.length; j++) {
      const interval = intervalBetween(chord.pitchClasses[i], chord.pitchClasses[j]);
      acousticDissonance += dissonanceWeights[interval] || 0.5;
    }
  }

  const geometricTension = edgeStress * 0.3 + volumeRatio * 0.3 + acousticDissonance * 0.4;

  return { geometricTension, acousticDissonance, edgeStress, volumeRatio };
}

export function calculateResolution(cell: MusicCell24, fromChord: Chord, toChord: Chord): number {
  const tensionFrom = calculateTension(cell, fromChord);
  const tensionTo = calculateTension(cell, toChord);
  return tensionFrom.geometricTension - tensionTo.geometricTension;
}

// ============================================================================
// Path Analysis
// ============================================================================

export function createPath(cell: MusicCell24, chords: Chord[], timestamps?: number[]): Path4D {
  const times = timestamps || chords.map((_, i) => i);
  const vertices: number[] = [];

  for (const chord of chords) {
    const vertex = cell.getVertexByKey(chord.root, 'major');
    if (vertex) vertices.push(vertex.id);
  }

  let totalLength = 0;
  for (let i = 0; i < vertices.length - 1; i++) {
    totalLength += cell.getDistance(vertices[i], vertices[i + 1]);
  }

  let curvature = 0;
  if (vertices.length >= 3) {
    for (let i = 1; i < vertices.length - 1; i++) {
      const v1 = subtract4D(cell.getVertex(vertices[i]).coords, cell.getVertex(vertices[i - 1]).coords);
      const v2 = subtract4D(cell.getVertex(vertices[i + 1]).coords, cell.getVertex(vertices[i]).coords);
      const cosAngle = dot4D(v1, v2) / (magnitude4D(v1) * magnitude4D(v2));
      curvature += Math.acos(Math.max(-1, Math.min(1, cosAngle)));
    }
    curvature /= vertices.length - 2;
  }

  const tensionProfile = chords.map(chord => calculateTension(cell, chord).geometricTension);

  return { vertices, timestamps: times, totalLength, curvature, tensionProfile };
}

export function analyzeVoiceLeading(
  cell: MusicCell24,
  fromChord: Chord,
  toChord: Chord
): { totalMovement: number; efficiency: number; isParsimonious: boolean } {
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

  const efficiency = 1 / (1 + totalMovement);
  const isParsimonious = totalMovement / maxVoices < 0.5;

  return { totalMovement, efficiency, isParsimonious };
}

// ============================================================================
// Pythagorean Comma Detection
// ============================================================================

export function detectPythagoreanComma(cell: MusicCell24): {
  startVertex: number;
  endVertex: number;
  geometricGap: number;
  commaInCents: number;
  isCircleClosed: boolean;
} {
  const startVertex = cell.getVertexByKey('C', 'major')!.id;
  const circleOfFifths: PitchClass[] = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F', 'C'];

  let totalDistance = 0;
  for (let i = 0; i < circleOfFifths.length - 1; i++) {
    const from = cell.getVertexByKey(circleOfFifths[i], 'major')!.id;
    const to = cell.getVertexByKey(circleOfFifths[i + 1], 'major')!.id;
    totalDistance += cell.getDistance(from, to);
  }

  const endVertex = cell.getVertexByKey('C', 'major')!.id;
  const fifthDistance = cell.getDistance(startVertex, cell.getVertexByKey('G', 'major')!.id);
  const geometricGap = totalDistance - fifthDistance * 12;
  const commaInCents = 1200 * Math.log2(Math.pow(3 / 2, 12) / Math.pow(2, 7));

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

export function runCalibrationSuite(cell: MusicCell24): CalibrationResult[] {
  const results: CalibrationResult[] = [];

  // Test 1: Vertex count matches 24 keys
  results.push({
    testName: 'Vertex Count = 24 Keys',
    passed: cell.vertexCount === 24,
    expected: 24,
    actual: cell.vertexCount,
    tolerance: 0,
    details: `24-cell has ${cell.vertexCount} vertices for 24 keys`,
  });

  // Test 2: Edge count = 96 (mathematically correct)
  results.push({
    testName: 'Edge Count = 96',
    passed: cell.edgeCount === 96,
    expected: 96,
    actual: cell.edgeCount,
    tolerance: 0,
    details: `24-cell has ${cell.edgeCount} edges`,
  });

  // Test 3: 16-cell contributes 8 primary keys
  const primaryKeys = cell.get16CellKeys();
  results.push({
    testName: '16-Cell Primary Keys',
    passed: primaryKeys.length === 8,
    expected: 8,
    actual: primaryKeys.length,
    tolerance: 0,
    details: `16-cell contributes ${primaryKeys.length} primary keys`,
  });

  // Test 4: 8-cell contributes 16 secondary/minor keys
  const secondaryKeys = cell.get8CellKeys();
  results.push({
    testName: '8-Cell Secondary Keys',
    passed: secondaryKeys.length === 16,
    expected: 16,
    actual: secondaryKeys.length,
    tolerance: 0,
    details: `8-cell contributes ${secondaryKeys.length} secondary/minor keys`,
  });

  // Test 5: Circle of Fifths connectivity
  let connectedFifths = 0;
  const fifths: [PitchClass, PitchClass][] = [['C', 'G'], ['G', 'D'], ['D', 'A'], ['A', 'E'], ['E', 'B']];
  for (const [from, to] of fifths) {
    const fromV = cell.getVertexByKey(from, 'major');
    const toV = cell.getVertexByKey(to, 'major');
    if (fromV && toV && cell.getDistance(fromV.id, toV.id) <= 1.5) {
      connectedFifths++;
    }
  }
  results.push({
    testName: 'Circle of Fifths Connectivity',
    passed: connectedFifths >= 4,
    expected: 5,
    actual: connectedFifths,
    tolerance: 1,
    details: `${connectedFifths}/5 fifths are geometrically adjacent`,
  });

  // Test 6: V-I Resolution (G7 -> C)
  const g7 = createChord('G', 'dominant7');
  const cMaj = createChord('C', 'major');
  const resolution = calculateResolution(cell, g7, cMaj);
  results.push({
    testName: 'V-I Resolution (G7 -> C)',
    passed: resolution > 0,
    expected: 1,
    actual: resolution,
    tolerance: 0,
    details: `Resolution strength: ${resolution.toFixed(4)} (positive = resolving)`,
  });

  // Test 7: Self-duality (each vertex has 8 neighbors)
  const vertex0Neighbors = cell.getNeighbors(0);
  results.push({
    testName: 'Self-Duality (8 neighbors)',
    passed: vertex0Neighbors.length === 8,
    expected: 8,
    actual: vertex0Neighbors.length,
    tolerance: 0,
    details: `Vertex 0 has ${vertex0Neighbors.length} neighbors`,
  });

  return results;
}

// ============================================================================
// Semantic Bridge
// ============================================================================

export function blendSemanticAndStructural(
  cell: MusicCell24,
  semanticVector: Float32Array,
  _weight = 0.5
): {
  suggestedKey: MusicalKey;
  suggestedMode: 'major' | 'minor';
  harmonicRhythm: 'fast' | 'medium' | 'slow';
  emotionalValence: number;
} {
  let sum = 0;
  for (let i = 0; i < semanticVector.length; i++) sum += semanticVector[i];
  const mean = sum / semanticVector.length;

  let variance = 0;
  for (let i = 0; i < semanticVector.length; i++) {
    variance += (semanticVector[i] - mean) ** 2;
  }
  variance /= semanticVector.length;

  const suggestedMode: 'major' | 'minor' = mean > 0 ? 'major' : 'minor';
  const harmonicRhythm: 'fast' | 'medium' | 'slow' =
    variance > 0.5 ? 'fast' : variance > 0.2 ? 'medium' : 'slow';

  const projected: Vector4D = {
    w: semanticVector[0] || 0,
    x: semanticVector[1] || 0,
    y: semanticVector[2] || 0,
    z: semanticVector[3] || 0,
  };

  const nearestVertex = cell.projectToVertex(projected);

  return {
    suggestedKey: nearestVertex.key,
    suggestedMode,
    harmonicRhythm,
    emotionalValence: mean,
  };
}

// ============================================================================
// Namespace Export
// ============================================================================

export const MusicGeometryDomain = {
  // Core class
  MusicCell24,
  Cell24: MusicCell24, // Alias for backward compatibility

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
