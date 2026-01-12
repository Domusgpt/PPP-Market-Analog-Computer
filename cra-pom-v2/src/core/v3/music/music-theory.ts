/**
 * PPP v3 Music Theory Embedding System
 *
 * A mathematically-grounded embedding space for music based on:
 * 1. Tonnetz (Euler 1739) - Lattice representation of tonal relationships
 * 2. Pitch Class Space - Cyclic group R/12Z (octave equivalence)
 * 3. Voice Leading Geometry (Tymoczko) - Chords as points in orbifolds
 * 4. Fourier Phase Space - Continuous harmonic geometry via DFT
 *
 * WHY MUSIC FOR TESTING EMBEDDINGS:
 * - Mathematical ground truth: C and G are exactly a perfect fifth (3:2 ratio)
 * - Verifiable analogies: C:E :: D:F# (both major thirds)
 * - Hierarchical structure: note → interval → chord → progression
 * - Audio grounding: embeddings can be validated against actual sound
 *
 * REFERENCES:
 * - https://en.wikipedia.org/wiki/Tonnetz
 * - https://dmitri.mycpanel.princeton.edu/geometry-of-music.html
 * - https://www.ams.org/publicoutreach/math-and-music
 */

// ============================================================================
// Musical Constants
// ============================================================================

/** The 12 pitch classes in Western music */
export const PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;
export type PitchClass = typeof PITCH_CLASSES[number];

/** Semitone distances for intervals */
export const INTERVALS = {
  unison: 0,
  minorSecond: 1,
  majorSecond: 2,
  minorThird: 3,
  majorThird: 4,
  perfectFourth: 5,
  tritone: 6,
  perfectFifth: 7,
  minorSixth: 8,
  majorSixth: 9,
  minorSeventh: 10,
  majorSeventh: 11,
  octave: 12,
} as const;

/** Frequency ratios for just intonation */
export const JUST_RATIOS: Record<keyof typeof INTERVALS, [number, number]> = {
  unison: [1, 1],
  minorSecond: [16, 15],
  majorSecond: [9, 8],
  minorThird: [6, 5],
  majorThird: [5, 4],
  perfectFourth: [4, 3],
  tritone: [45, 32],
  perfectFifth: [3, 2],
  minorSixth: [8, 5],
  majorSixth: [5, 3],
  minorSeventh: [9, 5],
  majorSeventh: [15, 8],
  octave: [2, 1],
};

/** A4 = 440 Hz standard tuning */
export const A4_FREQUENCY = 440;
export const A4_MIDI = 69;

// ============================================================================
// Types
// ============================================================================

export interface Note {
  pitchClass: PitchClass;
  octave: number;
  midi: number;
  frequency: number;
}

export interface Chord {
  root: PitchClass;
  type: ChordType;
  pitchClasses: PitchClass[];
  intervals: number[];
}

export type ChordType =
  | 'major'
  | 'minor'
  | 'diminished'
  | 'augmented'
  | 'major7'
  | 'minor7'
  | 'dominant7'
  | 'diminished7'
  | 'halfDiminished7';

export interface TonnetzCoordinate {
  /** Position on circle of fifths (x-axis) */
  fifth: number;
  /** Position on major thirds axis (y-axis) */
  majorThird: number;
}

export interface MusicEmbedding {
  /** The embedded musical object */
  object: Note | Chord | PitchClass;
  /** Embedding vector */
  vector: Float32Array;
  /** Tonnetz coordinates (for pitch classes and chords) */
  tonnetz?: TonnetzCoordinate;
  /** Fourier phase components */
  fourierPhase?: { third: number; fifth: number };
  /** Source of embedding */
  source: 'geometric' | 'learned' | 'hybrid';
}

// ============================================================================
// Pitch and Frequency Calculations
// ============================================================================

/**
 * Convert pitch class to semitone number (C = 0)
 */
export function pitchClassToSemitone(pc: PitchClass): number {
  return PITCH_CLASSES.indexOf(pc);
}

/**
 * Convert semitone number to pitch class
 */
export function semitoneToPitchClass(semitone: number): PitchClass {
  return PITCH_CLASSES[((semitone % 12) + 12) % 12];
}

/**
 * Calculate MIDI note number
 */
export function toMidi(pitchClass: PitchClass, octave: number): number {
  return pitchClassToSemitone(pitchClass) + (octave + 1) * 12;
}

/**
 * Calculate frequency from MIDI note number (equal temperament)
 */
export function midiToFrequency(midi: number): number {
  return A4_FREQUENCY * Math.pow(2, (midi - A4_MIDI) / 12);
}

/**
 * Create a Note object
 */
export function createNote(pitchClass: PitchClass, octave: number): Note {
  const midi = toMidi(pitchClass, octave);
  return {
    pitchClass,
    octave,
    midi,
    frequency: midiToFrequency(midi),
  };
}

/**
 * Calculate interval between two pitch classes (in semitones)
 */
export function intervalBetween(pc1: PitchClass, pc2: PitchClass): number {
  const s1 = pitchClassToSemitone(pc1);
  const s2 = pitchClassToSemitone(pc2);
  return ((s2 - s1) % 12 + 12) % 12;
}

// ============================================================================
// Tonnetz Geometry
// ============================================================================

/**
 * Map a pitch class to Tonnetz coordinates.
 *
 * The Tonnetz is a 2D lattice where:
 * - Horizontal axis: Circle of fifths (each step = +7 semitones)
 * - Vertical axis: Major thirds (each step = +4 semitones)
 *
 * This creates a toroidal topology where:
 * - Moving 12 steps horizontally returns to start (cycle of fifths)
 * - Moving 3 steps vertically returns to start (augmented chord)
 */
export function pitchClassToTonnetz(pc: PitchClass): TonnetzCoordinate {
  const semitone = pitchClassToSemitone(pc);

  // Find position on circle of fifths
  // C=0, G=1, D=2, A=3, E=4, B=5, F#=6, C#=7, G#=8, D#=9, A#=10, F=11
  const fifthPosition = (semitone * 7) % 12;

  // Find position on major third axis
  // C=0, E=1, G#=2, then repeats
  const majorThirdPosition = (semitone * 4) % 12;

  return {
    fifth: fifthPosition,
    majorThird: majorThirdPosition / 4, // Normalize to 0-3 range
  };
}

/**
 * Calculate Tonnetz distance between two pitch classes.
 * Uses taxicab (Manhattan) distance on the Tonnetz grid.
 */
export function tonnetzDistance(pc1: PitchClass, pc2: PitchClass): number {
  const t1 = pitchClassToTonnetz(pc1);
  const t2 = pitchClassToTonnetz(pc2);

  // Handle wraparound for circle of fifths
  const fifthDist = Math.min(
    Math.abs(t1.fifth - t2.fifth),
    12 - Math.abs(t1.fifth - t2.fifth)
  );

  // Handle wraparound for major thirds (3 positions)
  const thirdDist = Math.min(
    Math.abs(t1.majorThird - t2.majorThird),
    3 - Math.abs(t1.majorThird - t2.majorThird)
  );

  return fifthDist + thirdDist * 4; // Weight thirds more heavily
}

// ============================================================================
// Chord Construction
// ============================================================================

/** Chord formulas as intervals from root */
const CHORD_FORMULAS: Record<ChordType, number[]> = {
  major: [0, 4, 7],
  minor: [0, 3, 7],
  diminished: [0, 3, 6],
  augmented: [0, 4, 8],
  major7: [0, 4, 7, 11],
  minor7: [0, 3, 7, 10],
  dominant7: [0, 4, 7, 10],
  diminished7: [0, 3, 6, 9],
  halfDiminished7: [0, 3, 6, 10],
};

/**
 * Create a chord from root and type
 */
export function createChord(root: PitchClass, type: ChordType): Chord {
  const intervals = CHORD_FORMULAS[type];
  const rootSemitone = pitchClassToSemitone(root);

  const pitchClasses = intervals.map((interval) =>
    semitoneToPitchClass(rootSemitone + interval)
  );

  return {
    root,
    type,
    pitchClasses,
    intervals,
  };
}

/**
 * Get chord name string
 */
export function chordName(chord: Chord): string {
  const suffixes: Record<ChordType, string> = {
    major: '',
    minor: 'm',
    diminished: 'dim',
    augmented: 'aug',
    major7: 'maj7',
    minor7: 'm7',
    dominant7: '7',
    diminished7: 'dim7',
    halfDiminished7: 'm7b5',
  };
  return `${chord.root}${suffixes[chord.type]}`;
}

// ============================================================================
// Fourier Phase Space (Continuous Tonnetz)
// ============================================================================

/**
 * Calculate Fourier phase components for a pitch class set.
 *
 * This implements the DFT approach from Quinn/Lewin that creates
 * a continuous generalization of the Tonnetz.
 *
 * The third and fifth Fourier components (indices 3 and 5 in the DFT)
 * correspond to the major third and perfect fifth relationships.
 */
export function fourierPhase(pitchClasses: PitchClass[]): { third: number; fifth: number } {
  const n = 12; // 12-TET

  // Convert to pitch class set (binary indicator)
  const pcSet = new Array(12).fill(0);
  for (const pc of pitchClasses) {
    pcSet[pitchClassToSemitone(pc)] = 1;
  }

  // Calculate DFT components 3 and 5
  const calcComponent = (k: number): { magnitude: number; phase: number } => {
    let real = 0;
    let imag = 0;

    for (let j = 0; j < n; j++) {
      const angle = (2 * Math.PI * k * j) / n;
      real += pcSet[j] * Math.cos(angle);
      imag -= pcSet[j] * Math.sin(angle);
    }

    return {
      magnitude: Math.sqrt(real * real + imag * imag),
      phase: Math.atan2(imag, real),
    };
  };

  const third = calcComponent(3);
  const fifth = calcComponent(5);

  return {
    third: third.phase,
    fifth: fifth.phase,
  };
}

// ============================================================================
// Geometric Embedding
// ============================================================================

/**
 * Create a geometric embedding for a pitch class.
 *
 * The embedding combines:
 * 1. Tonnetz coordinates (2D)
 * 2. Fourier phase (2D)
 * 3. Circle of fifths position (1D, periodic)
 * 4. Chromatic position (1D)
 *
 * Total: 6-dimensional embedding with geometric meaning
 */
export function embedPitchClass(pc: PitchClass): MusicEmbedding {
  const tonnetz = pitchClassToTonnetz(pc);
  const phase = fourierPhase([pc]);

  // Create 6D embedding
  const vector = new Float32Array(6);

  // Tonnetz position (normalized to unit circle for each axis)
  vector[0] = Math.cos((2 * Math.PI * tonnetz.fifth) / 12);
  vector[1] = Math.sin((2 * Math.PI * tonnetz.fifth) / 12);
  vector[2] = Math.cos((2 * Math.PI * tonnetz.majorThird) / 3);
  vector[3] = Math.sin((2 * Math.PI * tonnetz.majorThird) / 3);

  // Fourier phase components
  vector[4] = phase.third / Math.PI; // Normalize to [-1, 1]
  vector[5] = phase.fifth / Math.PI;

  return {
    object: pc,
    vector,
    tonnetz,
    fourierPhase: phase,
    source: 'geometric',
  };
}

/**
 * Create a geometric embedding for a chord.
 *
 * Chord embeddings are computed as:
 * 1. Centroid of constituent pitch class embeddings
 * 2. Additional dimensions for chord quality
 */
export function embedChord(chord: Chord): MusicEmbedding {
  // Get embeddings for all pitch classes
  const pcEmbeddings = chord.pitchClasses.map(embedPitchClass);

  // Compute centroid
  const dim = pcEmbeddings[0].vector.length;
  const centroid = new Float32Array(dim + 4); // +4 for chord quality

  for (const emb of pcEmbeddings) {
    for (let i = 0; i < dim; i++) {
      centroid[i] += emb.vector[i] / pcEmbeddings.length;
    }
  }

  // Add chord quality encoding
  // [major/minor, has 7th, is diminished, is augmented]
  centroid[dim] = chord.type.includes('minor') ? -1 : 1;
  centroid[dim + 1] = chord.type.includes('7') ? 1 : 0;
  centroid[dim + 2] = chord.type.includes('dim') ? 1 : 0;
  centroid[dim + 3] = chord.type.includes('aug') ? 1 : 0;

  // Calculate aggregate Tonnetz (root position)
  const rootTonnetz = pitchClassToTonnetz(chord.root);
  const rootPhase = fourierPhase(chord.pitchClasses);

  return {
    object: chord,
    vector: centroid,
    tonnetz: rootTonnetz,
    fourierPhase: rootPhase,
    source: 'geometric',
  };
}

// ============================================================================
// Similarity and Distance
// ============================================================================

/**
 * Cosine similarity between two embeddings
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('Vector dimension mismatch');
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * Euclidean distance between two embeddings
 */
export function euclideanDistance(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('Vector dimension mismatch');
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

// ============================================================================
// Analogy Testing
// ============================================================================

/**
 * Test musical analogies using vector arithmetic.
 *
 * Example: C:E :: D:? should give F# (both are major thirds)
 *
 * This is the "king - man + woman = queen" test for music.
 */
export function analogy(
  a: PitchClass,
  b: PitchClass,
  c: PitchClass
): { predicted: PitchClass; similarity: number; expected: PitchClass } {
  const embA = embedPitchClass(a);
  const embB = embedPitchClass(b);
  const embC = embedPitchClass(c);

  // Compute b - a + c
  const result = new Float32Array(embA.vector.length);
  for (let i = 0; i < result.length; i++) {
    result[i] = embB.vector[i] - embA.vector[i] + embC.vector[i];
  }

  // Find nearest pitch class
  let bestSim = -Infinity;
  let bestPC: PitchClass = 'C';

  for (const pc of PITCH_CLASSES) {
    const emb = embedPitchClass(pc);
    const sim = cosineSimilarity(result, emb.vector);
    if (sim > bestSim) {
      bestSim = sim;
      bestPC = pc;
    }
  }

  // Calculate expected based on interval arithmetic
  const intervalAB = intervalBetween(a, b);
  const expectedSemitone = (pitchClassToSemitone(c) + intervalAB) % 12;
  const expected = semitoneToPitchClass(expectedSemitone);

  return {
    predicted: bestPC,
    similarity: bestSim,
    expected,
  };
}

// ============================================================================
// Music Theory Verification Tests
// ============================================================================

/**
 * Verify that the embedding space preserves musical relationships.
 * Returns test results for CI/CD verification.
 */
export function verifyMusicEmbeddings(): {
  passed: boolean;
  tests: Array<{ name: string; passed: boolean; details: string }>;
} {
  const tests: Array<{ name: string; passed: boolean; details: string }> = [];

  // Test 1: Perfect fifths should be close
  {
    const c = embedPitchClass('C');
    const g = embedPitchClass('G');
    const fSharp = embedPitchClass('F#');

    const cgSim = cosineSimilarity(c.vector, g.vector);
    const cfSharpSim = cosineSimilarity(c.vector, fSharp.vector);

    // C-G (fifth) should be more similar than C-F# (tritone)
    const passed = cgSim > cfSharpSim;
    tests.push({
      name: 'Perfect fifth closer than tritone',
      passed,
      details: `C-G: ${cgSim.toFixed(4)}, C-F#: ${cfSharpSim.toFixed(4)}`,
    });
  }

  // Test 2: Major thirds analogy
  {
    const result = analogy('C', 'E', 'D');
    const passed = result.predicted === result.expected;
    tests.push({
      name: 'Major third analogy (C:E :: D:?)',
      passed,
      details: `Expected: ${result.expected}, Got: ${result.predicted} (sim: ${result.similarity.toFixed(4)})`,
    });
  }

  // Test 3: Perfect fifth analogy
  {
    const result = analogy('C', 'G', 'D');
    const passed = result.predicted === result.expected;
    tests.push({
      name: 'Perfect fifth analogy (C:G :: D:?)',
      passed,
      details: `Expected: ${result.expected}, Got: ${result.predicted}`,
    });
  }

  // Test 4: Relative major/minor (C major and A minor share notes)
  {
    const cMaj = createChord('C', 'major');
    const aMin = createChord('A', 'minor');
    const fSharpDim = createChord('F#', 'diminished');

    const embCMaj = embedChord(cMaj);
    const embAMin = embedChord(aMin);
    const embFSharpDim = embedChord(fSharpDim);

    const relSim = cosineSimilarity(embCMaj.vector, embAMin.vector);
    const unrelSim = cosineSimilarity(embCMaj.vector, embFSharpDim.vector);

    const passed = relSim > unrelSim;
    tests.push({
      name: 'Relative major/minor closer than unrelated',
      passed,
      details: `C-Am: ${relSim.toFixed(4)}, C-F#dim: ${unrelSim.toFixed(4)}`,
    });
  }

  // Test 5: Circle of fifths ordering
  {
    const c = embedPitchClass('C');
    const g = embedPitchClass('G');
    const d = embedPitchClass('D');
    const fSharp = embedPitchClass('F#');

    // C should be closer to G than to D (1 fifth vs 2 fifths)
    const cgDist = euclideanDistance(c.vector, g.vector);
    const cdDist = euclideanDistance(c.vector, d.vector);
    const cfSharpDist = euclideanDistance(c.vector, fSharp.vector);

    const passed = cgDist < cdDist && cdDist < cfSharpDist;
    tests.push({
      name: 'Circle of fifths distance ordering',
      passed,
      details: `C-G: ${cgDist.toFixed(4)}, C-D: ${cdDist.toFixed(4)}, C-F#: ${cfSharpDist.toFixed(4)}`,
    });
  }

  return {
    passed: tests.every((t) => t.passed),
    tests,
  };
}

// ============================================================================
// Exports for Integration
// ============================================================================

export const MusicTheory = {
  // Constants
  PITCH_CLASSES,
  INTERVALS,
  JUST_RATIOS,

  // Note operations
  createNote,
  toMidi,
  midiToFrequency,

  // Pitch class operations
  pitchClassToSemitone,
  semitoneToPitchClass,
  intervalBetween,

  // Tonnetz
  pitchClassToTonnetz,
  tonnetzDistance,

  // Chords
  createChord,
  chordName,

  // Fourier analysis
  fourierPhase,

  // Embeddings
  embedPitchClass,
  embedChord,
  cosineSimilarity,
  euclideanDistance,

  // Analogies
  analogy,

  // Verification
  verifyMusicEmbeddings,
};

export default MusicTheory;
