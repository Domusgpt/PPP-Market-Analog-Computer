/**
 * MusicGeometryDomain.ts
 *
 * Maps musical structures to 24-Cell polytope geometry.
 * Serves as the calibration domain for the Chronomorphic Polytopal Engine.
 *
 * @version 1.0.0
 * @author CPE Research
 *
 * EXPERIMENTAL PARAMETERS are marked with [TUNABLE] comments.
 * These can be modified for research and calibration.
 */

// =============================================================================
// TYPES
// =============================================================================

export type Vector4D = [number, number, number, number];

export interface Note {
    pitch: string;          // e.g., 'C', 'F#', 'Bb'
    octave: number;         // e.g., 4 for middle C
    duration?: number;      // in beats
    velocity?: number;      // 0-127 MIDI velocity
    time?: number;          // beat position
}

export interface Interval {
    semitones: number;
    ratio?: [number, number];  // Pythagorean ratio, e.g., [3, 2] for perfect fifth
    name?: string;
}

export type Chord = Note[] | string[];

export interface ChordGeometry {
    root: Vector4D;
    vertices: Vector4D[];
    centroid: Vector4D;
    edges: [number, number][];
    volume: number;
    symmetryGroup: string;
    tension: number;
}

export interface Path4D {
    points: Vector4D[];
    tangents: Vector4D[];
    curvature: number[];
    length: number;
}

export interface TemporalNote extends Note {
    time: number;
}

export type Progression = Chord[];
export type Melody = TemporalNote[];

// =============================================================================
// CONFIGURATION
// =============================================================================

export type TuningSystem = 'pythagorean' | 'equal_temperament' | 'just_intonation';
export type PitchMapping = 'circle_of_fifths' | 'chromatic' | 'tonnetz';
export type VertexAssignment = 'fifths_spiral' | 'chromatic_order' | 'tonnetz_grid';

export interface MusicGeometryConfig {
    // [TUNABLE] Tuning system affects interval ratios
    tuningSystem: TuningSystem;

    // [TUNABLE] Reference frequency for A4 (historical: 415-466 Hz)
    referenceFrequency: number;

    // [TUNABLE] How pitch maps to x-y plane
    pitchToXY: PitchMapping;

    // [TUNABLE] How time maps to 4th dimension (beats per unit)
    timeScale: number;

    // [TUNABLE] How velocity/dynamics affects distance from origin
    dynamicsRadius: number;

    // [TUNABLE] Which vertex assignment scheme to use
    vertexAssignment: VertexAssignment;

    // [TUNABLE] Weight for semantic embeddings (0 = pure structure, 1 = pure semantic)
    embeddingWeight: number;

    // [TUNABLE] Smoothing factor for temporal semantic changes
    semanticSmoothing: number;
}

export const DEFAULT_CONFIG: MusicGeometryConfig = {
    tuningSystem: 'equal_temperament',
    referenceFrequency: 440,
    pitchToXY: 'circle_of_fifths',
    timeScale: 1.0,
    dynamicsRadius: 1.0,
    vertexAssignment: 'fifths_spiral',
    embeddingWeight: 0.3,
    semanticSmoothing: 0.5,
};

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * 24-Cell vertices at all permutations of (±1, ±1, 0, 0)
 * Normalized to unit distance from origin
 */
export const VERTICES_24CELL: Vector4D[] = [
    // (±1, ±1, 0, 0) permutations - 24 total
    [1, 1, 0, 0], [1, -1, 0, 0], [-1, 1, 0, 0], [-1, -1, 0, 0],
    [1, 0, 1, 0], [1, 0, -1, 0], [-1, 0, 1, 0], [-1, 0, -1, 0],
    [1, 0, 0, 1], [1, 0, 0, -1], [-1, 0, 0, 1], [-1, 0, 0, -1],
    [0, 1, 1, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, -1, -1, 0],
    [0, 1, 0, 1], [0, 1, 0, -1], [0, -1, 0, 1], [0, -1, 0, -1],
    [0, 0, 1, 1], [0, 0, 1, -1], [0, 0, -1, 1], [0, 0, -1, -1],
];

/**
 * Normalize vertices to unit length
 */
const SQRT2_INV = 1 / Math.sqrt(2);
export const VERTICES_24CELL_NORMALIZED: Vector4D[] = VERTICES_24CELL.map(
    v => v.map(x => x * SQRT2_INV) as Vector4D
);

/**
 * Circle of fifths order (for pitch mapping)
 */
export const CIRCLE_OF_FIFTHS = [
    'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F'
];

/**
 * Chromatic scale
 */
export const CHROMATIC_SCALE = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
];

/**
 * Enharmonic equivalents
 */
export const ENHARMONIC: Record<string, string> = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb',
};

/**
 * Pythagorean interval ratios
 */
export const PYTHAGOREAN_RATIOS: Record<number, [number, number]> = {
    0: [1, 1],      // Unison
    1: [256, 243],  // Minor 2nd
    2: [9, 8],      // Major 2nd
    3: [32, 27],    // Minor 3rd
    4: [81, 64],    // Major 3rd
    5: [4, 3],      // Perfect 4th
    6: [729, 512],  // Tritone
    7: [3, 2],      // Perfect 5th
    8: [128, 81],   // Minor 6th
    9: [27, 16],    // Major 6th
    10: [16, 9],    // Minor 7th
    11: [243, 128], // Major 7th
    12: [2, 1],     // Octave
};

/**
 * Just intonation ratios
 */
export const JUST_RATIOS: Record<number, [number, number]> = {
    0: [1, 1],    // Unison
    1: [16, 15],  // Minor 2nd
    2: [9, 8],    // Major 2nd
    3: [6, 5],    // Minor 3rd
    4: [5, 4],    // Major 3rd
    5: [4, 3],    // Perfect 4th
    6: [45, 32],  // Tritone
    7: [3, 2],    // Perfect 5th
    8: [8, 5],    // Minor 6th
    9: [5, 3],    // Major 6th
    10: [9, 5],   // Minor 7th
    11: [15, 8],  // Major 7th
    12: [2, 1],   // Octave
};

/**
 * Consonance values (0 = dissonant, 1 = consonant)
 * Based on sensory dissonance research
 */
export const CONSONANCE_VALUES: Record<number, number> = {
    0: 1.0,    // Unison - perfect
    1: 0.1,    // Minor 2nd - very dissonant
    2: 0.3,    // Major 2nd - dissonant
    3: 0.7,    // Minor 3rd - consonant
    4: 0.8,    // Major 3rd - consonant
    5: 0.9,    // Perfect 4th - consonant
    6: 0.2,    // Tritone - dissonant
    7: 0.95,   // Perfect 5th - very consonant
    8: 0.75,   // Minor 6th - consonant
    9: 0.8,    // Major 6th - consonant
    10: 0.4,   // Minor 7th - somewhat dissonant
    11: 0.15,  // Major 7th - dissonant
    12: 1.0,   // Octave - perfect
};

// =============================================================================
// MAIN CLASS
// =============================================================================

export class MusicGeometryDomain {
    private config: MusicGeometryConfig;
    private keyToVertex: Map<string, number>;
    private vertexToKey: Map<number, string>;

    constructor(config: Partial<MusicGeometryConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.keyToVertex = new Map();
        this.vertexToKey = new Map();
        this.initializeMapping();
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    private initializeMapping(): void {
        // Map keys to vertices based on assignment scheme
        switch (this.config.vertexAssignment) {
            case 'fifths_spiral':
                this.initializeFifthsSpiral();
                break;
            case 'chromatic_order':
                this.initializeChromaticOrder();
                break;
            case 'tonnetz_grid':
                this.initializeTonnetzGrid();
                break;
        }
    }

    /**
     * [TUNABLE] Fifths spiral: vertices follow circle of fifths
     * Major keys: vertices 0-11, Minor keys: vertices 12-23
     */
    private initializeFifthsSpiral(): void {
        // Major keys
        CIRCLE_OF_FIFTHS.forEach((note, i) => {
            this.keyToVertex.set(`${note}_major`, i);
            this.vertexToKey.set(i, `${note}_major`);
        });

        // Relative minor keys (offset by 12)
        const relativeMinors = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'Bb', 'F', 'C', 'G', 'D'];
        relativeMinors.forEach((note, i) => {
            this.keyToVertex.set(`${note}_minor`, i + 12);
            this.vertexToKey.set(i + 12, `${note}_minor`);
        });
    }

    private initializeChromaticOrder(): void {
        // Major keys follow chromatic order
        CHROMATIC_SCALE.forEach((note, i) => {
            this.keyToVertex.set(`${note}_major`, i);
            this.vertexToKey.set(i, `${note}_major`);
        });

        // Minor keys
        CHROMATIC_SCALE.forEach((note, i) => {
            this.keyToVertex.set(`${note}_minor`, i + 12);
            this.vertexToKey.set(i + 12, `${note}_minor`);
        });
    }

    private initializeTonnetzGrid(): void {
        // Tonnetz-inspired mapping (major thirds and perfect fifths)
        // This creates a different geometric relationship
        const tonnetzOrder = [
            'C', 'E', 'G#', 'G', 'B', 'D#', 'D', 'F#', 'A#', 'A', 'C#', 'F'
        ];
        tonnetzOrder.forEach((note, i) => {
            this.keyToVertex.set(`${note}_major`, i);
            this.vertexToKey.set(i, `${note}_major`);
        });

        // Minor follows same pattern
        tonnetzOrder.forEach((note, i) => {
            this.keyToVertex.set(`${note}_minor`, i + 12);
            this.vertexToKey.set(i + 12, `${note}_minor`);
        });
    }

    // =========================================================================
    // NOTE CONVERSION
    // =========================================================================

    /**
     * Convert a note to 4D coordinates
     *
     * @param note - Note as string ('C4') or Note object
     * @returns 4D coordinate vector
     */
    noteToCoordinate(note: string | Note): Vector4D {
        const parsed = this.parseNote(note);
        const pitchClass = this.getPitchClass(parsed.pitch);
        const octave = parsed.octave;

        // Get base position from pitch mapping
        let basePosition: Vector4D;

        switch (this.config.pitchToXY) {
            case 'circle_of_fifths':
                basePosition = this.pitchToCircleOfFifths(pitchClass);
                break;
            case 'chromatic':
                basePosition = this.pitchToChromatic(pitchClass);
                break;
            case 'tonnetz':
                basePosition = this.pitchToTonnetz(pitchClass);
                break;
            default:
                basePosition = this.pitchToCircleOfFifths(pitchClass);
        }

        // Octave affects the z-coordinate (3rd dimension)
        // [TUNABLE] Octave scaling factor
        const octaveOffset = (octave - 4) * 0.5;  // Middle C (C4) is at z=0
        basePosition[2] += octaveOffset;

        // Time position (4th dimension)
        if (parsed.time !== undefined) {
            basePosition[3] = parsed.time * this.config.timeScale;
        }

        // Velocity affects distance from origin
        if (parsed.velocity !== undefined) {
            const velocityScale = (parsed.velocity / 127) * this.config.dynamicsRadius;
            basePosition = this.scaleVector(basePosition, velocityScale);
        }

        return basePosition;
    }

    /**
     * Circle of fifths mapping: notes spiral around origin
     */
    private pitchToCircleOfFifths(pitchClass: number): Vector4D {
        // Map pitch class to angle (7 semitones per step in circle of fifths)
        const fifthsPosition = (pitchClass * 7) % 12;
        const angle = (fifthsPosition / 12) * 2 * Math.PI;

        // [TUNABLE] Radius of the circle
        const radius = 1.0;

        return [
            radius * Math.cos(angle),
            radius * Math.sin(angle),
            0,
            0
        ];
    }

    /**
     * Chromatic mapping: notes on a line/helix
     */
    private pitchToChromatic(pitchClass: number): Vector4D {
        const angle = (pitchClass / 12) * 2 * Math.PI;
        const radius = 1.0;

        return [
            radius * Math.cos(angle),
            radius * Math.sin(angle),
            0,
            0
        ];
    }

    /**
     * Tonnetz mapping: hexagonal grid (major thirds and fifths)
     */
    private pitchToTonnetz(pitchClass: number): Vector4D {
        // Tonnetz is a hexagonal grid where:
        // - Horizontal movement = perfect fifth (7 semitones)
        // - Diagonal movement = major third (4 semitones)

        // Calculate position in tonnetz grid
        const fifthsX = Math.floor(pitchClass / 7);
        const thirdsY = Math.floor(pitchClass / 4);

        // [TUNABLE] Grid spacing
        const spacing = 0.5;

        return [
            fifthsX * spacing,
            thirdsY * spacing * Math.sqrt(3) / 2,  // Hexagonal offset
            0,
            0
        ];
    }

    /**
     * Convert 4D coordinate back to nearest note
     */
    coordinateToNote(coord: Vector4D): Note {
        // Find nearest pitch class based on mapping
        let nearestPitchClass = 0;
        let minDistance = Infinity;

        for (let pc = 0; pc < 12; pc++) {
            const noteCoord = this.pitchToCircleOfFifths(pc);
            const dist = this.distance(coord, noteCoord);
            if (dist < minDistance) {
                minDistance = dist;
                nearestPitchClass = pc;
            }
        }

        // Determine octave from z-coordinate
        const octave = Math.round(coord[2] / 0.5) + 4;

        return {
            pitch: CHROMATIC_SCALE[nearestPitchClass],
            octave: Math.max(0, Math.min(9, octave)),
            time: coord[3] / this.config.timeScale
        };
    }

    /**
     * Convert frequency to 4D coordinate
     */
    frequencyToCoordinate(hz: number): Vector4D {
        // Calculate pitch class and octave from frequency
        const a4 = this.config.referenceFrequency;
        const semitonesFromA4 = 12 * Math.log2(hz / a4);
        const noteNumber = Math.round(semitonesFromA4) + 69;  // MIDI note number

        const pitchClass = noteNumber % 12;
        const octave = Math.floor(noteNumber / 12) - 1;

        return this.noteToCoordinate({
            pitch: CHROMATIC_SCALE[pitchClass],
            octave
        });
    }

    // =========================================================================
    // INTERVAL ANALYSIS
    // =========================================================================

    /**
     * Calculate interval between two notes
     */
    getInterval(a: string | Note, b: string | Note): Interval {
        const noteA = this.parseNote(a);
        const noteB = this.parseNote(b);

        const pcA = this.getPitchClass(noteA.pitch);
        const pcB = this.getPitchClass(noteB.pitch);

        let semitones = (pcB - pcA + 12) % 12;

        // Add octave difference
        semitones += (noteB.octave - noteA.octave) * 12;

        return {
            semitones: semitones % 12,
            ratio: this.getIntervalRatio(semitones % 12),
            name: this.getIntervalName(semitones % 12)
        };
    }

    /**
     * Get interval ratio based on tuning system
     */
    private getIntervalRatio(semitones: number): [number, number] {
        switch (this.config.tuningSystem) {
            case 'pythagorean':
                return PYTHAGOREAN_RATIOS[semitones] || [1, 1];
            case 'just_intonation':
                return JUST_RATIOS[semitones] || [1, 1];
            case 'equal_temperament':
            default:
                // Equal temperament ratio as decimal approximation
                const ratio = Math.pow(2, semitones / 12);
                return [Math.round(ratio * 1000), 1000];
        }
    }

    private getIntervalName(semitones: number): string {
        const names = [
            'Unison', 'Minor 2nd', 'Major 2nd', 'Minor 3rd',
            'Major 3rd', 'Perfect 4th', 'Tritone', 'Perfect 5th',
            'Minor 6th', 'Major 6th', 'Minor 7th', 'Major 7th'
        ];
        return names[semitones % 12];
    }

    /**
     * Calculate interval as geometric distance
     */
    intervalToDistance(interval: Interval): number {
        // [TUNABLE] Distance mapping
        // Consonant intervals = shorter distance
        const consonance = CONSONANCE_VALUES[interval.semitones] || 0.5;
        return 2 * (1 - consonance);  // 0 (consonant) to 2 (dissonant)
    }

    /**
     * Measure consonance between two notes (0 = dissonant, 1 = consonant)
     */
    measureConsonance(a: string | Note, b: string | Note): number {
        const interval = this.getInterval(a, b);
        return CONSONANCE_VALUES[interval.semitones] || 0.5;
    }

    // =========================================================================
    // CHORD GEOMETRY
    // =========================================================================

    /**
     * Convert a chord to its polytope geometry
     */
    chordToPolytope(chord: Chord): ChordGeometry {
        // Parse all notes
        const notes = chord.map(n => this.parseNote(n));
        const vertices = notes.map(n => this.noteToCoordinate(n));

        // Calculate centroid
        const centroid = this.calculateCentroid(vertices);

        // Calculate edges (all pairs)
        const edges: [number, number][] = [];
        for (let i = 0; i < vertices.length; i++) {
            for (let j = i + 1; j < vertices.length; j++) {
                edges.push([i, j]);
            }
        }

        // Calculate volume (hypervolume in 4D)
        const volume = this.calculateHypervolume(vertices);

        // Determine symmetry group
        const symmetryGroup = this.determineSymmetryGroup(chord);

        // Calculate tension
        const tension = this.calculateChordTension(chord);

        return {
            root: vertices[0],
            vertices,
            centroid,
            edges,
            volume,
            symmetryGroup,
            tension
        };
    }

    /**
     * Calculate centroid of vertices
     */
    private calculateCentroid(vertices: Vector4D[]): Vector4D {
        const sum: Vector4D = [0, 0, 0, 0];
        for (const v of vertices) {
            sum[0] += v[0];
            sum[1] += v[1];
            sum[2] += v[2];
            sum[3] += v[3];
        }
        return sum.map(x => x / vertices.length) as Vector4D;
    }

    /**
     * Calculate hypervolume (simplified as product of spans)
     */
    private calculateHypervolume(vertices: Vector4D[]): number {
        if (vertices.length < 2) return 0;

        // Calculate span in each dimension
        const spans = [0, 1, 2, 3].map(dim => {
            const values = vertices.map(v => v[dim]);
            return Math.max(...values) - Math.min(...values);
        });

        // Hypervolume approximation
        return spans.reduce((a, b) => a * Math.max(b, 0.001), 1);
    }

    /**
     * Determine symmetry group of chord
     */
    private determineSymmetryGroup(chord: Chord): string {
        const notes = chord.map(n => this.parseNote(n));
        const intervals: number[] = [];

        // Calculate all intervals
        for (let i = 1; i < notes.length; i++) {
            const interval = this.getInterval(notes[0], notes[i]);
            intervals.push(interval.semitones);
        }

        // Check for common chord types
        const sorted = [...intervals].sort((a, b) => a - b);

        // Major triad: 4, 7 semitones
        if (this.arraysEqual(sorted, [4, 7])) return 'D3';  // Dihedral group order 3

        // Minor triad: 3, 7 semitones
        if (this.arraysEqual(sorted, [3, 7])) return 'D3';

        // Diminished triad: 3, 6 semitones
        if (this.arraysEqual(sorted, [3, 6])) return 'D3';

        // Augmented triad: 4, 8 semitones (maximally symmetric)
        if (this.arraysEqual(sorted, [4, 8])) return 'D3';  // Actually C3, but...

        // Diminished 7th: 3, 6, 9 (forms tetrahedron!)
        if (this.arraysEqual(sorted, [3, 6, 9])) return 'Td';  // Tetrahedral

        // Major 7th: 4, 7, 11
        if (this.arraysEqual(sorted, [4, 7, 11])) return 'C1';  // No symmetry

        // Dominant 7th: 4, 7, 10
        if (this.arraysEqual(sorted, [4, 7, 10])) return 'C1';

        return 'C1';  // Default: no symmetry
    }

    /**
     * Calculate chord tension (0 = stable, 1 = unstable)
     */
    calculateChordTension(chord: Chord): number {
        const notes = chord.map(n => this.parseNote(n));

        if (notes.length < 2) return 0;

        // Average dissonance of all intervals
        let totalDissonance = 0;
        let count = 0;

        for (let i = 0; i < notes.length; i++) {
            for (let j = i + 1; j < notes.length; j++) {
                const consonance = this.measureConsonance(notes[i], notes[j]);
                totalDissonance += (1 - consonance);
                count++;
            }
        }

        return count > 0 ? totalDissonance / count : 0;
    }

    // =========================================================================
    // PROGRESSIONS AND PATHS
    // =========================================================================

    /**
     * Convert chord progression to 4D path
     */
    progressionToPath(progression: Progression): Path4D {
        const geometries = progression.map(chord => this.chordToPolytope(chord));
        const points = geometries.map(g => g.centroid);

        // Calculate tangents (direction of motion)
        const tangents: Vector4D[] = [];
        for (let i = 0; i < points.length - 1; i++) {
            const tangent = this.subtractVectors(points[i + 1], points[i]);
            tangents.push(this.normalizeVector(tangent));
        }
        if (tangents.length > 0) {
            tangents.push(tangents[tangents.length - 1]);  // Duplicate last
        }

        // Calculate curvature
        const curvature: number[] = [];
        for (let i = 1; i < tangents.length; i++) {
            const angleDiff = this.angleBetween(tangents[i - 1], tangents[i]);
            curvature.push(angleDiff);
        }
        curvature.unshift(0);  // First point has no curvature

        // Calculate total path length
        let length = 0;
        for (let i = 0; i < points.length - 1; i++) {
            length += this.distance(points[i], points[i + 1]);
        }

        return { points, tangents, curvature, length };
    }

    /**
     * Analyze cadence type
     */
    analyzeCadence(progression: Chord[]): string {
        if (progression.length < 2) return 'none';

        const last = this.parseChordName(progression[progression.length - 1]);
        const secondLast = this.parseChordName(progression[progression.length - 2]);

        // V-I (Authentic)
        if (this.isFifthAbove(secondLast.root, last.root) && last.quality === 'major') {
            return 'authentic';
        }

        // IV-I (Plagal)
        if (this.isFourthAbove(secondLast.root, last.root) && last.quality === 'major') {
            return 'plagal';
        }

        // V-vi (Deceptive)
        if (this.isFifthAbove(secondLast.root, last.root) && last.quality === 'minor') {
            return 'deceptive';
        }

        // I-V (Half)
        if (secondLast.quality === 'major' && this.isFifthAbove(secondLast.root, last.root)) {
            return 'half';
        }

        return 'other';
    }

    /**
     * Measure motion between two chords
     */
    measureMotion(from: Chord, to: Chord): { distance: number; direction: Vector4D; tensionChange: number } {
        const fromGeom = this.chordToPolytope(from);
        const toGeom = this.chordToPolytope(to);

        const direction = this.subtractVectors(toGeom.centroid, fromGeom.centroid);
        const distance = this.magnitude(direction);
        const tensionChange = toGeom.tension - fromGeom.tension;

        return {
            distance,
            direction: this.normalizeVector(direction),
            tensionChange
        };
    }

    // =========================================================================
    // MELODY
    // =========================================================================

    /**
     * Convert melody to 4D path
     */
    melodyToPath(notes: TemporalNote[]): Path4D {
        const points = notes.map(n => this.noteToCoordinate(n));

        // Calculate tangents
        const tangents: Vector4D[] = [];
        for (let i = 0; i < points.length - 1; i++) {
            const tangent = this.subtractVectors(points[i + 1], points[i]);
            tangents.push(this.normalizeVector(tangent));
        }
        if (tangents.length > 0) {
            tangents.push(tangents[tangents.length - 1]);
        }

        // Calculate curvature
        const curvature: number[] = [0];
        for (let i = 1; i < tangents.length; i++) {
            curvature.push(this.angleBetween(tangents[i - 1], tangents[i]));
        }

        // Calculate length
        let length = 0;
        for (let i = 0; i < points.length - 1; i++) {
            length += this.distance(points[i], points[i + 1]);
        }

        return { points, tangents, curvature, length };
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    private parseNote(input: string | Note): Note {
        if (typeof input === 'object') return input;

        // Parse string like 'C4', 'F#5', 'Bb3'
        const match = input.match(/^([A-Ga-g][#b]?)(\d)?$/);
        if (!match) {
            return { pitch: 'C', octave: 4 };
        }

        return {
            pitch: match[1].toUpperCase(),
            octave: match[2] ? parseInt(match[2]) : 4
        };
    }

    private parseChordName(chord: Chord | string): { root: string; quality: string } {
        if (Array.isArray(chord)) {
            // Infer from notes
            const notes = chord.map(n => this.parseNote(n));
            const root = notes[0].pitch;
            const intervals = notes.slice(1).map(n => this.getInterval(notes[0], n).semitones);

            if (intervals.includes(4)) return { root, quality: 'major' };
            if (intervals.includes(3)) return { root, quality: 'minor' };
            return { root, quality: 'unknown' };
        }

        // Parse string like 'Cmaj', 'Am', 'G7'
        const match = (chord as string).match(/^([A-G][#b]?)(m|maj|min|dim|aug|7)?/i);
        if (!match) return { root: 'C', quality: 'major' };

        const root = match[1];
        const suffix = match[2]?.toLowerCase() || '';

        if (suffix === 'm' || suffix === 'min') return { root, quality: 'minor' };
        if (suffix === 'dim') return { root, quality: 'diminished' };
        if (suffix === 'aug') return { root, quality: 'augmented' };
        return { root, quality: 'major' };
    }

    private getPitchClass(pitch: string): number {
        const normalized = pitch.replace('b', '#');
        let pc = CHROMATIC_SCALE.indexOf(pitch);
        if (pc === -1) {
            pc = CHROMATIC_SCALE.indexOf(ENHARMONIC[pitch] || 'C');
        }
        return pc;
    }

    private isFifthAbove(from: string, to: string): boolean {
        const pcFrom = this.getPitchClass(from);
        const pcTo = this.getPitchClass(to);
        return (pcTo - pcFrom + 12) % 12 === 7;
    }

    private isFourthAbove(from: string, to: string): boolean {
        const pcFrom = this.getPitchClass(from);
        const pcTo = this.getPitchClass(to);
        return (pcTo - pcFrom + 12) % 12 === 5;
    }

    // Vector operations
    private distance(a: Vector4D, b: Vector4D): number {
        return Math.sqrt(
            Math.pow(b[0] - a[0], 2) +
            Math.pow(b[1] - a[1], 2) +
            Math.pow(b[2] - a[2], 2) +
            Math.pow(b[3] - a[3], 2)
        );
    }

    private magnitude(v: Vector4D): number {
        return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
    }

    private normalizeVector(v: Vector4D): Vector4D {
        const mag = this.magnitude(v);
        if (mag === 0) return [0, 0, 0, 0];
        return [v[0] / mag, v[1] / mag, v[2] / mag, v[3] / mag];
    }

    private scaleVector(v: Vector4D, s: number): Vector4D {
        return [v[0] * s, v[1] * s, v[2] * s, v[3] * s];
    }

    private subtractVectors(a: Vector4D, b: Vector4D): Vector4D {
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    }

    private dotProduct(a: Vector4D, b: Vector4D): number {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    }

    private angleBetween(a: Vector4D, b: Vector4D): number {
        const dot = this.dotProduct(a, b);
        const magA = this.magnitude(a);
        const magB = this.magnitude(b);
        if (magA === 0 || magB === 0) return 0;
        return Math.acos(Math.max(-1, Math.min(1, dot / (magA * magB))));
    }

    private arraysEqual(a: number[], b: number[]): boolean {
        if (a.length !== b.length) return false;
        return a.every((v, i) => v === b[i]);
    }

    // =========================================================================
    // PUBLIC CONFIG ACCESS
    // =========================================================================

    getConfig(): MusicGeometryConfig {
        return { ...this.config };
    }

    updateConfig(updates: Partial<MusicGeometryConfig>): void {
        this.config = { ...this.config, ...updates };
        this.initializeMapping();
    }

    getVertices(): Vector4D[] {
        return [...VERTICES_24CELL_NORMALIZED];
    }

    getKeyMapping(): Map<string, number> {
        return new Map(this.keyToVertex);
    }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/**
 * Create domain with Pythagorean tuning (for research)
 */
export function createPythagoreanDomain(): MusicGeometryDomain {
    return new MusicGeometryDomain({
        tuningSystem: 'pythagorean',
        pitchToXY: 'circle_of_fifths'
    });
}

/**
 * Create domain with Just Intonation (for pure ratios)
 */
export function createJustDomain(): MusicGeometryDomain {
    return new MusicGeometryDomain({
        tuningSystem: 'just_intonation',
        pitchToXY: 'tonnetz'
    });
}

/**
 * Create domain optimized for temporal analysis
 */
export function createTemporalDomain(beatsPerUnit: number = 4): MusicGeometryDomain {
    return new MusicGeometryDomain({
        timeScale: 1 / beatsPerUnit,
        dynamicsRadius: 1.5
    });
}

// =============================================================================
// EXPORTS
// =============================================================================

export default MusicGeometryDomain;
