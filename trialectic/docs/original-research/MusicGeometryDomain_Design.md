# MusicGeometryDomain: A Calibration Framework for the Chronomorphic Polytopal Engine

**Version:** 1.0.0
**Date:** 2026-01-10
**Status:** Design & Implementation

---

## Executive Summary

This document describes the design and implementation of `MusicGeometryDomain`, a specialized domain module that maps musical structures to the 24-Cell polytope geometry. Music serves as an ideal calibration domain because:

1. **Measurable** - Frequencies, intervals, and rhythms are precisely quantifiable
2. **Perceptible** - Results can be heard and validated by human perception
3. **Mathematical** - Deep connections to Pythagorean and Euclidean foundations
4. **Temporal** - Natural 4th dimension through rhythm and time
5. **Emotional** - Bridges abstract geometry to felt human experience

---

## Table of Contents

1. [Theoretical Foundations](#1-theoretical-foundations)
2. [The 24-Cell Musical Mapping](#2-the-24-cell-musical-mapping)
3. [Implementation Architecture](#3-implementation-architecture)
4. [Experimental Parameters](#4-experimental-parameters)
5. [Calibration Methodology](#5-calibration-methodology)
6. [API Reference](#6-api-reference)
7. [Research Extensions](#7-research-extensions)

---

## 1. Theoretical Foundations

### 1.1 Pythagorean Music Theory

Pythagoras discovered that musical harmony corresponds to simple integer ratios:

| Interval | Ratio | Cents | Geometric Interpretation |
|----------|-------|-------|--------------------------|
| Unison | 1:1 | 0 | Identity (origin) |
| Octave | 2:1 | 1200 | Doubling (scaling) |
| Perfect Fifth | 3:2 | 702 | Primary rotation |
| Perfect Fourth | 4:3 | 498 | Complementary rotation |
| Major Third | 5:4 | 386 | Secondary axis |
| Minor Third | 6:5 | 316 | Tertiary axis |

**The Tetractys (1+2+3+4=10):**
```
    •
   • •
  • • •
 • • • •
```
Contains all consonant intervals within the first four integers. This is the Pythagorean "source of all things."

### 1.2 Euclidean Geometry Parallels

Both Pythagorean music and Euclidean geometry share foundational principles:

| Principle | In Music | In Geometry | In 24-Cell |
|-----------|----------|-------------|------------|
| **Ratio** | Interval = freq ratio | Proportion = length ratio | Vertex distances |
| **Symmetry** | Inversion, retrograde | Reflection, rotation | 1152 symmetries |
| **Commensurability** | Consonance | Rational proportion | Vertex alignment |
| **The Golden Ratio φ** | Minor 6th ≈ φ | Pentagon diagonal | Edge relationships |

### 1.3 Why the 24-Cell?

The 24-Cell is uniquely suited for musical mapping:

1. **24 Vertices** = 24 major/minor keys (12 major + 12 minor)
2. **24 Octahedral Cells** = 24 diatonic modes
3. **96 Edges** = Interval relationships
4. **Self-Dual** = Major/minor duality
5. **4 Dimensions** = Pitch, Time, Timbre, Dynamics

---

## 2. The 24-Cell Musical Mapping

### 2.1 Vertex-to-Key Assignment

The 24 vertices of the 24-Cell map to the 24 major and minor keys:

```typescript
// 24-Cell vertices (±1, ±1, 0, 0) and permutations
const VERTICES_24CELL = [
  // Type A: (±1, ±1, 0, 0) - 8 vertices
  [1, 1, 0, 0], [1, -1, 0, 0], [-1, 1, 0, 0], [-1, -1, 0, 0],
  [1, 0, 1, 0], [1, 0, -1, 0], [-1, 0, 1, 0], [-1, 0, -1, 0],
  [1, 0, 0, 1], [1, 0, 0, -1], [-1, 0, 0, 1], [-1, 0, 0, -1],
  // Type B: (0, 0, ±1, ±1) - 8 vertices
  [0, 1, 1, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, -1, -1, 0],
  [0, 1, 0, 1], [0, 1, 0, -1], [0, -1, 0, 1], [0, -1, 0, -1],
  [0, 0, 1, 1], [0, 0, 1, -1], [0, 0, -1, 1], [0, 0, -1, -1],
];

// Key mapping following circle of fifths
const KEY_MAPPING = {
  // Major keys (vertices 0-11)
  'C_major':  0,  'G_major':  1,  'D_major':  2,  'A_major':  3,
  'E_major':  4,  'B_major':  5,  'F#_major': 6,  'Db_major': 7,
  'Ab_major': 8,  'Eb_major': 9,  'Bb_major': 10, 'F_major':  11,

  // Minor keys (vertices 12-23) - relative minors
  'A_minor':  12, 'E_minor':  13, 'B_minor':  14, 'F#_minor': 15,
  'C#_minor': 16, 'G#_minor': 17, 'D#_minor': 18, 'Bb_minor': 19,
  'F_minor':  20, 'C_minor':  21, 'G_minor':  22, 'D_minor':  23,
};
```

### 2.2 Circle of Fifths as Rotation

The circle of fifths corresponds to rotation in the 24-Cell:

```
        C
    F       G
  Bb          D
 Eb            A
  Ab          E
    Db      B
       F#/Gb

Movement by fifth = 72° rotation in x-y plane (360°/5 = 72°)
Movement by fourth = -72° rotation (inverse)
```

**Mathematical formulation:**
```typescript
// Moving by a fifth is a rotation matrix in 4D
function rotateByFifth(point: number[]): number[] {
  const angle = Math.PI * 7 / 6; // ~210° in 4D
  // Rotation in the x-y plane
  return [
    point[0] * Math.cos(angle) - point[1] * Math.sin(angle),
    point[0] * Math.sin(angle) + point[1] * Math.cos(angle),
    point[2],
    point[3]
  ];
}
```

### 2.3 Interval-to-Edge Mapping

Edges of the 24-Cell represent musical intervals:

| Edge Type | Distance | Musical Interval | Consonance |
|-----------|----------|------------------|------------|
| Short edge | √2 | Minor 2nd / Major 7th | Dissonant |
| Medium edge | √3 | Minor 3rd / Major 6th | Imperfect consonance |
| Long edge | 2 | Perfect 4th / 5th | Perfect consonance |
| Diagonal | √4=2 | Tritone | Maximum tension |

### 2.4 Chord Geometry

Chords form sub-polytopes within the 24-Cell:

```typescript
interface ChordGeometry {
  root: number[];           // 4D position of root note
  vertices: number[][];     // All notes in the chord
  centroid: number[];       // Center of mass
  volume: number;           // 4D hypervolume (chord "size")
  symmetryGroup: string;    // e.g., "D3" for major triad
}

// Major triad forms an equilateral triangle in 3D projection
const C_MAJOR_TRIAD = {
  root: VERTICES_24CELL[KEY_MAPPING['C_major']],
  vertices: [
    noteToVertex('C'),  // Root
    noteToVertex('E'),  // Major 3rd
    noteToVertex('G'),  // Perfect 5th
  ],
  symmetryGroup: 'D3'  // 6 symmetries (3 rotations, 3 reflections)
};

// Minor triad - same symmetry, different orientation
const A_MINOR_TRIAD = {
  root: VERTICES_24CELL[KEY_MAPPING['A_minor']],
  vertices: [
    noteToVertex('A'),  // Root
    noteToVertex('C'),  // Minor 3rd
    noteToVertex('E'),  // Perfect 5th
  ],
  symmetryGroup: 'D3'
};

// Diminished 7th - forms a regular tetrahedron!
const DIM7_CHORD = {
  vertices: [
    noteToVertex('B'),
    noteToVertex('D'),
    noteToVertex('F'),
    noteToVertex('Ab'),
  ],
  symmetryGroup: 'Td'  // 24 symmetries - tetrahedral
};
```

### 2.5 The 4th Dimension: Time

The 4th coordinate represents temporal position:

```typescript
interface TemporalNote {
  pitch: [number, number, number];  // First 3 coords (pitch space)
  time: number;                      // 4th coord (beat position)
  duration: number;                  // Note length
  velocity: number;                  // Dynamics (mapped to distance from origin)
}

// A melody traces a path through 4D space
type Melody = TemporalNote[];

// Chord progression = sequence of polytope configurations
type Progression = ChordGeometry[];
```

---

## 3. Implementation Architecture

### 3.1 Module Structure

```
lib/
├── domains/
│   └── MusicGeometryDomain.ts    # Main domain module
├── music/
│   ├── PitchSpace.ts             # Note/frequency mapping
│   ├── IntervalGeometry.ts       # Interval calculations
│   ├── ChordGeometry.ts          # Chord polytope structures
│   ├── ProgressionPath.ts        # Chord progression as 4D path
│   └── RhythmGeometry.ts         # Temporal/rhythmic mapping
└── calibration/
    └── MusicCalibration.ts       # Validation and tuning
```

### 3.2 Core Classes

```typescript
// lib/domains/MusicGeometryDomain.ts

export interface MusicGeometryConfig {
  // Tuning system
  tuningSystem: 'pythagorean' | 'equal_temperament' | 'just_intonation';
  referenceFrequency: number;  // A4 = 440 Hz default

  // Mapping parameters (EXPERIMENTAL - can be tuned)
  pitchToXY: 'circle_of_fifths' | 'chromatic' | 'tonnetz';
  timeScale: number;           // How time maps to 4th dimension
  dynamicsRadius: number;      // How velocity affects distance from origin

  // Integration
  useEmbeddings: boolean;      // Bridge to HDCEncoder for semantic analysis
  embeddingWeight: number;     // How much semantic content influences position
}

export class MusicGeometryDomain {
  private config: MusicGeometryConfig;
  private vertices24Cell: number[][];
  private keyMapping: Map<string, number>;

  constructor(config: Partial<MusicGeometryConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.initializeGeometry();
  }

  // Core transformations
  noteToCoordinate(note: Note): Vector4D;
  coordinateToNote(coord: Vector4D): Note;
  chordToPolytope(chord: Chord): ChordGeometry;
  progressionToPath(progression: Progression): Path4D;

  // Analysis
  measureConsonance(interval: Interval): number;
  measureTension(chord: Chord): number;
  measureMotion(from: Chord, to: Chord): MotionVector;

  // Calibration
  validateMapping(testCases: TestCase[]): ValidationResult;
  optimizeParameters(trainingData: TrainingData): OptimizedConfig;
}
```

### 3.3 Integration with CPE Components

```typescript
// Integration flow

// 1. HDCEncoder provides semantic context
const semanticVector = await hdcEncoder.textToForceAsync(
  "melancholic piano melody in A minor"
);

// 2. MusicGeometryDomain maps musical elements
const musicDomain = new MusicGeometryDomain();
const chordGeometry = musicDomain.chordToPolytope(['A', 'C', 'E']);

// 3. Combined vector influences polytope state
const combinedForce = blendVectors(semanticVector, chordGeometry.centroid, {
  semanticWeight: 0.3,
  musicalWeight: 0.7
});

// 4. CausalReasoningEngine analyzes harmonic causation
const causalPath = causalEngine.findPath(
  chordGeometry,
  nextChordGeometry
);

// 5. Renderer visualizes the 4D structure
renderer.updatePolytope(combinedForce, {
  highlightVertices: chordGeometry.vertices,
  animatePath: causalPath
});
```

---

## 4. Experimental Parameters

These parameters can be modified for research and calibration:

### 4.1 Mapping Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `tuningSystem` | 'equal_temperament' | enum | Changes interval ratios |
| `referenceFrequency` | 440 | 415-466 | Historical tuning variations |
| `pitchToXY` | 'circle_of_fifths' | enum | How pitch maps to x-y plane |
| `timeScale` | 1.0 | 0.1-10.0 | Temporal compression/expansion |
| `dynamicsRadius` | 1.0 | 0.5-2.0 | How loudness affects distance |

### 4.2 Geometric Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `vertexAssignment` | 'fifths_spiral' | enum | How keys map to vertices |
| `edgeWeighting` | 'consonance' | enum | Edge weight calculation |
| `chordCentroidMethod` | 'geometric' | enum | How chord center is computed |
| `projectionMethod` | 'stereographic' | enum | 4D to 3D projection |

### 4.3 Semantic Integration Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `embeddingWeight` | 0.3 | 0.0-1.0 | Semantic vs. structural balance |
| `archetypeMapping` | 'emotion' | enum | Which archetypes map to music |
| `semanticSmoothing` | 0.5 | 0.0-1.0 | Temporal smoothing of embeddings |

### 4.4 Tuning Rationale

**Why these defaults?**

- `equal_temperament`: Most common modern tuning, allows all keys
- `circle_of_fifths`: Respects the fundamental 3:2 ratio relationship
- `timeScale: 1.0`: 1 beat = 1 unit in 4th dimension
- `embeddingWeight: 0.3`: Music structure dominates, semantics modulate

---

## 5. Calibration Methodology

### 5.1 Ground Truth Data

We establish ground truth using:

1. **Acoustic measurements**
   - Frequency ratios (exact)
   - Beating frequencies for dissonance

2. **Music theory consensus**
   - Circle of fifths relationships
   - Chord function (tonic, dominant, subdominant)

3. **Perceptual studies**
   - Consonance/dissonance ratings
   - Emotional associations

### 5.2 Calibration Test Cases

```typescript
interface CalibrationTestCase {
  input: MusicalInput;
  expectedOutput: GeometricOutput;
  tolerance: number;
  weight: number;  // Importance in optimization
}

const CALIBRATION_SUITE: CalibrationTestCase[] = [
  // Test 1: Octave should map to same position (modulo)
  {
    input: { note: 'C4' },
    expectedOutput: { vertex: 0, octaveEquivalent: true },
    tolerance: 0.001,
    weight: 1.0
  },

  // Test 2: Fifth should be adjacent vertex
  {
    input: { interval: ['C', 'G'] },
    expectedOutput: { edgeType: 'perfect_consonance', distance: 2.0 },
    tolerance: 0.01,
    weight: 1.0
  },

  // Test 3: Major triad symmetry
  {
    input: { chord: ['C', 'E', 'G'] },
    expectedOutput: { symmetryGroup: 'D3', regularity: 0.95 },
    tolerance: 0.05,
    weight: 0.8
  },

  // Test 4: Diminished 7th forms tetrahedron
  {
    input: { chord: ['B', 'D', 'F', 'Ab'] },
    expectedOutput: { symmetryGroup: 'Td', vertices: 4 },
    tolerance: 0.01,
    weight: 0.9
  },

  // Test 5: ii-V-I progression traces expected path
  {
    input: { progression: ['Dm', 'G7', 'C'] },
    expectedOutput: { pathType: 'cadential', tension_curve: 'rise_fall' },
    tolerance: 0.1,
    weight: 0.7
  }
];
```

### 5.3 Validation Metrics

```typescript
interface ValidationMetrics {
  // Geometric accuracy
  vertexMappingError: number;      // How far notes land from expected vertices
  edgePreservation: number;        // Do intervals map to correct edges?
  symmetryPreservation: number;    // Do chord symmetries survive mapping?

  // Musical validity
  consonanceCorrelation: number;   // Does geometric distance correlate with consonance?
  progressionCoherence: number;    // Do progressions form smooth paths?

  // Perceptual alignment
  emotionalMapping: number;        // Do embeddings match expected emotions?
  tensionCurveMatch: number;       // Does geometric tension match perceived tension?
}
```

### 5.4 Optimization Loop

```typescript
async function calibrate(
  testSuite: CalibrationTestCase[],
  initialConfig: MusicGeometryConfig
): Promise<OptimizedConfig> {

  let bestConfig = initialConfig;
  let bestScore = evaluate(testSuite, bestConfig);

  // Grid search over parameter space
  for (const tuning of TUNING_SYSTEMS) {
    for (const mapping of MAPPING_METHODS) {
      for (const timeScale of [0.5, 1.0, 2.0]) {
        const config = { ...initialConfig, tuning, mapping, timeScale };
        const score = evaluate(testSuite, config);

        if (score > bestScore) {
          bestScore = score;
          bestConfig = config;
        }
      }
    }
  }

  return { config: bestConfig, score: bestScore, metrics: getMetrics(bestConfig) };
}
```

---

## 6. API Reference

### 6.1 Core Functions

```typescript
// Note conversion
noteToCoordinate(note: string | Note): Vector4D
coordinateToNote(coord: Vector4D): Note
frequencyToCoordinate(hz: number): Vector4D

// Interval analysis
intervalToDistance(interval: Interval): number
distanceToInterval(distance: number): Interval
measureConsonance(a: Note, b: Note): number  // 0 = dissonant, 1 = consonant

// Chord geometry
chordToPolytope(notes: Note[]): ChordGeometry
polytopeToChord(geometry: ChordGeometry): Note[]
chordTension(chord: Chord): number  // 0 = stable, 1 = unstable

// Progressions
progressionToPath(chords: Chord[]): Path4D
pathToProgression(path: Path4D): Chord[]
analyzeCadence(progression: Chord[]): CadenceType

// Temporal
melodyToPath(notes: TemporalNote[]): Path4D
rhythmToPattern(durations: number[]): RhythmGeometry

// Semantic bridge
embedMeaning(text: string): Promise<Vector4D>  // Uses HDCEncoder
blendSemanticAndStructural(semantic: Vector4D, structural: Vector4D): Vector4D
```

### 6.2 Types

```typescript
type Vector4D = [number, number, number, number];
type Note = { pitch: string; octave: number; duration?: number; velocity?: number };
type Chord = Note[] | string[];  // e.g., ['C', 'E', 'G'] or [{pitch: 'C', octave: 4}, ...]
type Interval = { semitones: number; ratio?: [number, number] };
type Progression = Chord[];

interface ChordGeometry {
  root: Vector4D;
  vertices: Vector4D[];
  centroid: Vector4D;
  edges: [number, number][];
  volume: number;
  symmetryGroup: string;
  tension: number;
}

interface Path4D {
  points: Vector4D[];
  tangents: Vector4D[];
  curvature: number[];
  length: number;
}
```

---

## 7. Research Extensions

### 7.1 Future Experiments

1. **Microtonal mapping**
   - Extend beyond 12-TET to 19-TET, 31-TET, etc.
   - Map to higher-dimensional polytopes

2. **Timbre as dimension**
   - Map spectral content to additional coordinates
   - Different instruments = different regions of space

3. **Cultural tuning systems**
   - Arabic maqam system
   - Indian raga system
   - Map to different polytope structures

4. **Machine learning calibration**
   - Train on human perceptual data
   - Learn optimal parameter mappings

### 7.2 Open Questions

1. Is the 24-Cell the optimal polytope, or would 120-Cell or 600-Cell reveal more structure?
2. How does the mapping perform for atonal/12-tone music?
3. Can we derive Pythagorean ratios from the polytope geometry, or must we assume them?
4. What is the relationship between geometric path length and perceived musical distance?

### 7.3 Validation Against Music Theory

The mapping should be validated against established music theory:

- Schenker's hierarchical analysis
- Riemann's harmonic function theory
- Neo-Riemannian transformations (PLR operations)
- Lerdahl's tonal pitch space

---

## Appendix A: Mathematical Derivations

### A.1 24-Cell Vertex Coordinates

The 24-Cell has vertices at all permutations of (±1, ±1, 0, 0):

```
8 vertices: (±1, ±1, 0, 0)
8 vertices: (±1, 0, ±1, 0)
8 vertices: (±1, 0, 0, ±1)
```

Wait, that's only 24 total... let me recalculate:
- (±1, ±1, 0, 0): 4 sign choices = 4 vertices
- (±1, 0, ±1, 0): 4 vertices
- (±1, 0, 0, ±1): 4 vertices
- (0, ±1, ±1, 0): 4 vertices
- (0, ±1, 0, ±1): 4 vertices
- (0, 0, ±1, ±1): 4 vertices

Total: 24 vertices ✓

### A.2 Pythagorean Comma and Geometric Closure

The Pythagorean comma (312 / 219 ≈ 1.0136) represents the failure of 12 perfect fifths to equal 7 octaves. Geometrically, this appears as a small gap when traversing the circle of fifths - the spiral doesn't quite close.

In equal temperament, we distribute this comma equally, which corresponds to a slight rotation adjustment at each fifth.

---

## Appendix B: Quick Start Example

```typescript
import { MusicGeometryDomain } from './lib/domains/MusicGeometryDomain';

// Create domain with default config
const music = new MusicGeometryDomain();

// Map a note to 4D
const cCoord = music.noteToCoordinate('C4');
console.log('C4 position:', cCoord);  // [1, 1, 0, 0]

// Map a chord
const cMajor = music.chordToPolytope(['C', 'E', 'G']);
console.log('C major centroid:', cMajor.centroid);
console.log('C major tension:', cMajor.tension);  // Low (stable)

// Analyze a progression
const progression = music.progressionToPath(['Dm', 'G7', 'C']);
console.log('ii-V-I path length:', progression.length);
console.log('Cadential motion:', music.analyzeCadence(['Dm', 'G7', 'C']));

// Bridge to semantics
const emotionalContext = await music.embedMeaning('sad, nostalgic feeling');
const combined = music.blendSemanticAndStructural(emotionalContext, cMajor.centroid);
```

---

*This document is version-controlled and should be updated as the implementation evolves.*
