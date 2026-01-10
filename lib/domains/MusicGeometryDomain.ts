/**
 * MusicGeometryDomain - Prototype mapping between musical structures and 4D geometry.
 *
 * This module provides deterministic mappings from notes, chords, and progressions
 * into 4D vectors, plus key-to-24-cell vertex bindings for calibration workflows.
 */

import { Vector4D } from '../../types/index.js';
import { getDefaultLattice } from '../topology/Lattice24.js';
import { MusicEmbeddingBridge } from './MusicEmbeddingBridge.js';

const NOTE_SEQUENCE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

const FLAT_ALIASES: Record<string, string> = {
    Db: 'C#',
    Eb: 'D#',
    Gb: 'F#',
    Ab: 'G#',
    Bb: 'A#'
};

const MAJOR_KEYS = [
    'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F'
];

const MINOR_KEYS = [
    'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F', 'C', 'G', 'D'
];

const CONSONANCE_WEIGHTS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35];

export interface NoteAnalysis {
    readonly note: string;
    readonly frequency: number;
    readonly pitchClass: number;
    readonly octave: number;
    readonly vector: Vector4D;
}

export interface ChordConfiguration {
    readonly notes: string[];
    readonly vectors: Vector4D[];
    readonly centroid: Vector4D;
    readonly spread: number;
}

export interface ProgressionTrajectory {
    readonly stages: ChordConfiguration[];
    readonly path: Vector4D[];
}

export interface TimbreProfile {
    readonly brightness: number;
    readonly warmth: number;
    readonly roughness: number;
}

export class MusicGeometryDomain {
    private readonly _lattice = getDefaultLattice();
    private readonly _embeddingBridge = new MusicEmbeddingBridge();

    noteToFrequency(note: string, referenceA4 = 440): number {
        const { pitchClass, octave } = this._parseNote(note);
        const semitoneOffset = (octave - 4) * 12 + (pitchClass - 9);
        return referenceA4 * Math.pow(2, semitoneOffset / 12);
    }

    noteToVector4D(note: string): Vector4D {
        const { pitchClass, octave } = this._parseNote(note);
        const circleIndex = (pitchClass * 7) % 12;
        const angle = (circleIndex / 12) * Math.PI * 2;
        const fifthX = Math.cos(angle);
        const fifthY = Math.sin(angle);
        const octaveOffset = octave - 4;
        const intervalClass = Math.min(pitchClass, 12 - pitchClass);
        const consonance = CONSONANCE_WEIGHTS[intervalClass] ?? 0.3;

        return [fifthX, fifthY, octaveOffset, consonance];
    }

    analyzeNote(note: string): NoteAnalysis {
        return {
            note,
            frequency: this.noteToFrequency(note),
            pitchClass: this._parseNote(note).pitchClass,
            octave: this._parseNote(note).octave,
            vector: this.noteToVector4D(note)
        };
    }

    chordToConfiguration(notes: string[]): ChordConfiguration {
        const vectors = notes.map(note => this.noteToVector4D(note));
        const centroid: Vector4D = [0, 0, 0, 0];
        for (const vec of vectors) {
            centroid[0] += vec[0];
            centroid[1] += vec[1];
            centroid[2] += vec[2];
            centroid[3] += vec[3];
        }
        if (vectors.length > 0) {
            centroid[0] /= vectors.length;
            centroid[1] /= vectors.length;
            centroid[2] /= vectors.length;
            centroid[3] /= vectors.length;
        }

        const spread = vectors.reduce((sum, vec) => {
            const dx = vec[0] - centroid[0];
            const dy = vec[1] - centroid[1];
            const dz = vec[2] - centroid[2];
            const dw = vec[3] - centroid[3];
            return sum + Math.sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
        }, 0) / Math.max(1, vectors.length);

        return {
            notes,
            vectors,
            centroid,
            spread
        };
    }

    progressionToTrajectory(chords: string[][]): ProgressionTrajectory {
        const stages = chords.map(chord => this.chordToConfiguration(chord));
        const path = stages.map(stage => stage.centroid);
        return { stages, path };
    }

    keyToVertex(key: string, mode: 'major' | 'minor' = 'major'): Vector4D {
        const normalizedKey = this._normalizeNoteName(key);
        const keyList = mode === 'major' ? MAJOR_KEYS : MINOR_KEYS;
        const index = keyList.indexOf(normalizedKey);
        const vertexIndex = index >= 0 ? index : 0;
        return this._lattice.vertices[vertexIndex].coordinates;
    }

    timbreToArchetypeWeights(profile: TimbreProfile): number[] {
        const clamp = (value: number) => Math.max(0, Math.min(1, value));
        const brightness = clamp(profile.brightness);
        const warmth = clamp(profile.warmth);
        const roughness = clamp(profile.roughness);

        const center = Math.round(brightness * 11) + Math.round(warmth * 11);
        const weights = Array.from({ length: 24 }, (_, i) => {
            const distance = Math.abs(i - center);
            const base = Math.exp(-distance / 6);
            return base * (1 - roughness * 0.3);
        });

        const total = weights.reduce((sum, value) => sum + value, 0);
        return total > 0 ? weights.map(value => value / total) : weights;
    }

    embeddingToVector4D(embedding: Float32Array | number[]): Vector4D {
        return this._embeddingBridge.embeddingToVector4D(embedding);
    }

    async textToVector4DWithVoyage(
        text: string,
        apiKey: string,
        model?: string
    ): Promise<Vector4D> {
        return this._embeddingBridge.textToVector4DWithVoyage(text, apiKey, model);
    }

    private _parseNote(note: string): { pitchClass: number; octave: number } {
        const trimmed = note.trim();
        const match = trimmed.match(/^([A-Ga-g])([#b]?)(-?\\d+)$/);
        if (!match) {
            return { pitchClass: 0, octave: 4 };
        }
        const name = match[1].toUpperCase();
        const accidental = match[2];
        const octave = Number.parseInt(match[3], 10);
        const normalized = this._normalizeNoteName(`${name}${accidental}`);
        const pitchClass = NOTE_SEQUENCE.indexOf(normalized as (typeof NOTE_SEQUENCE)[number]);
        return { pitchClass: pitchClass >= 0 ? pitchClass : 0, octave };
    }

    private _normalizeNoteName(note: string): string {
        const normalized = note.replace(/\s+/g, '');
        return FLAT_ALIASES[normalized] ?? normalized;
    }
}

export const musicGeometryDomain = new MusicGeometryDomain();
