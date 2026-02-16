/**
 * Music Generation Mode
 *
 * Application mode that uses the Chronomorphic Engine for algorithmic music
 * generation. Features:
 *
 * - FFT analysis mapped to 24-cell vertices
 * - Key detection via vertex proximity
 * - Ghost frequency resolution as note generation
 * - Trinity axis modulation for harmonic progression
 * - Real-time MIDI output
 *
 * Ported from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * with import paths adapted for PPP math_core layout.
 */

import type { Vector4D, TrinityAxis } from '../geometric_algebra/types';
import { MusicGeometryDomain } from '../domains/MusicGeometryDomain';
import { GhostFrequencyDetector, type GhostAnalysis } from '../tda/GhostFrequencyDetector';

// =============================================================================
// TYPES
// =============================================================================

/** Pitch class (0-11) */
export type PitchClass = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11;

/** Musical note */
export interface Note {
  readonly pitchClass: PitchClass;
  readonly octave: number;
  readonly velocity: number; // 0-127 MIDI velocity
  readonly duration: number; // Duration in ms
  readonly channel: number;  // MIDI channel (0-15)
}

/** Chord structure */
export interface Chord {
  readonly notes: Note[];
  readonly name: string;
  readonly rootPitchClass: PitchClass;
  readonly quality: 'major' | 'minor' | 'diminished' | 'augmented' | 'sus2' | 'sus4' | 'dom7' | 'custom';
}

/** Music event for timeline */
export interface MusicEvent {
  readonly type: 'note_on' | 'note_off' | 'chord' | 'modulation' | 'tempo_change';
  readonly time: number;    // Timestamp in ms
  readonly data: Note | Chord | ModulationEvent | { bpm: number };
}

/** FFT result from audio analysis */
export interface FFTResult {
  readonly magnitudes: Float32Array;
  readonly frequencies: Float32Array;
  readonly sampleRate: number;
}

/** Chroma vector (pitch class distribution) */
export type ChromaVector = [number, number, number, number, number, number,
                             number, number, number, number, number, number];

/** Key detection result */
export interface KeyDetection {
  readonly key: string;
  readonly confidence: number;
  readonly chromaVector: ChromaVector;
  readonly alternativeKeys: { key: string; confidence: number }[];
}

/** Music generation configuration */
export interface MusicGenerationConfig {
  /** Base tempo (BPM) */
  tempo: number;
  /** Default velocity */
  velocity: number;
  /** Default note duration (ms) */
  noteDuration: number;
  /** Enable ghost frequency resolution */
  resolveGhostFrequencies: boolean;
  /** Maximum simultaneous notes */
  maxPolyphony: number;
  /** MIDI channel */
  channel: number;
  /** Modulation sensitivity (0-1) */
  modulationSensitivity: number;
  /** Minimum confidence for key detection */
  keyDetectionThreshold: number;
}

/** Generation output */
export interface GenerationOutput {
  readonly events: MusicEvent[];
  readonly detectedKey: KeyDetection;
  readonly currentAxis: TrinityAxis;
  readonly ghostsResolved: number;
}

/** Modulation event */
export interface ModulationEvent {
  readonly fromKey: string;
  readonly toKey: string;
  readonly type: 'parallel' | 'relative' | 'leading_tone' | 'circle_of_fifths' | 'enharmonic';
  readonly confidence: number;
}

// =============================================================================
// DEFAULTS
// =============================================================================

export const DEFAULT_MUSIC_CONFIG: MusicGenerationConfig = {
  tempo: 120,
  velocity: 80,
  noteDuration: 500,
  resolveGhostFrequencies: true,
  maxPolyphony: 4,
  channel: 0,
  modulationSensitivity: 0.5,
  keyDetectionThreshold: 0.6
};

// =============================================================================
// PITCH CLASS NAMES
// =============================================================================

const PC_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

// =============================================================================
// MUSIC GENERATION MODE
// =============================================================================

/**
 * Music Generation Mode.
 *
 * Uses the Chronomorphic Engine's geometric state to drive
 * algorithmic music generation.
 */
export class MusicGenerationMode {
  private _config: MusicGenerationConfig;
  private _musicDomain: MusicGeometryDomain;
  private _ghostDetector: GhostFrequencyDetector;
  private _currentKey: string;
  private _currentAxis: TrinityAxis;
  private _eventHistory: MusicEvent[];
  private _lastModulationTime: number;
  private _timeCounter: number;

  constructor(config: Partial<MusicGenerationConfig> = {}) {
    this._config = { ...DEFAULT_MUSIC_CONFIG, ...config };
    this._musicDomain = new MusicGeometryDomain();
    this._ghostDetector = new GhostFrequencyDetector();
    this._currentKey = 'C';
    this._currentAxis = 'alpha';
    this._eventHistory = [];
    this._lastModulationTime = 0;
    this._timeCounter = 0;
  }

  // =========================================================================
  // FFT PROCESSING
  // =========================================================================

  /**
   * Convert FFT result to chroma vector.
   */
  fftToChroma(fft: FFTResult): ChromaVector {
    const chroma: ChromaVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    for (let i = 0; i < fft.magnitudes.length; i++) {
      const freq = fft.frequencies[i];
      if (freq < 20 || freq > 4186) continue; // Piano range

      // Convert frequency to pitch class
      const midi = 69 + 12 * Math.log2(freq / 440);
      const pc = Math.round(midi) % 12;
      if (pc >= 0 && pc < 12) {
        chroma[pc] += fft.magnitudes[i];
      }
    }

    // Normalize
    const max = Math.max(...chroma, 0.001);
    for (let i = 0; i < 12; i++) {
      chroma[i] /= max;
    }

    return chroma;
  }

  /**
   * Detect key from chroma vector.
   */
  detectKey(chroma: ChromaVector): KeyDetection {
    // Get pitch classes with significant energy
    const activePCs: number[] = [];
    for (let i = 0; i < 12; i++) {
      if (chroma[i] > 0.3) {
        activePCs.push(i);
      }
    }

    // Use music domain for key detection
    const results = this._musicDomain.detectKey(activePCs);

    const primaryResult = results[0] ?? { key: this._currentKey, confidence: 0 };

    return {
      key: primaryResult.key,
      confidence: primaryResult.confidence,
      chromaVector: chroma,
      alternativeKeys: results.slice(1, 4)
    };
  }

  // =========================================================================
  // GENERATION
  // =========================================================================

  /**
   * Generate music events from engine state.
   */
  generate(
    position: Vector4D,
    trinityWeights: [number, number, number],
    tension: number,
    ghostAnalysis?: GhostAnalysis
  ): GenerationOutput {
    const events: MusicEvent[] = [];
    const beatDuration = 60000 / this._config.tempo;

    // Determine current key from position
    const positionKey = this._musicDomain.positionToKey(position);
    const detectedKey = {
      key: positionKey,
      confidence: 0.8,
      chromaVector: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] as ChromaVector,
      alternativeKeys: [] as { key: string; confidence: number }[]
    };

    // Determine active axis
    const [alpha, beta, gamma] = trinityWeights;
    if (alpha >= beta && alpha >= gamma) this._currentAxis = 'alpha';
    else if (beta >= gamma) this._currentAxis = 'beta';
    else this._currentAxis = 'gamma';

    // Check for modulation
    if (positionKey !== this._currentKey) {
      const now = this._timeCounter;
      if (now - this._lastModulationTime > beatDuration * 4) {
        // Determine modulation type
        const distance = this._musicDomain.circleOfFifthsDistance(
          this._currentKey, positionKey
        );

        let modType: ModulationEvent['type'] = 'circle_of_fifths';
        if (distance <= 1) modType = 'circle_of_fifths';
        else if (this._musicDomain.parallelTransform(this._currentKey) === positionKey) {
          modType = 'parallel';
        } else if (this._musicDomain.relativeTransform(this._currentKey) === positionKey) {
          modType = 'relative';
        }

        events.push({
          type: 'modulation',
          time: this._timeCounter,
          data: {
            fromKey: this._currentKey,
            toKey: positionKey,
            type: modType,
            confidence: 0.8
          }
        });

        this._currentKey = positionKey;
        this._lastModulationTime = now;
      }
    }

    // Generate notes based on position
    const key = this._musicDomain.getKey(this._currentKey);
    if (key) {
      // Root note
      const rootNote: Note = {
        pitchClass: key.root as PitchClass,
        octave: 4,
        velocity: Math.round(this._config.velocity * (0.7 + tension * 0.3)),
        duration: this._config.noteDuration,
        channel: this._config.channel
      };

      events.push({
        type: 'note_on',
        time: this._timeCounter,
        data: rootNote
      });

      // Add harmonic notes based on Trinity weights
      const intervals = key.mode === 'major'
        ? [4, 7, 11] // Major 3rd, P5, Major 7th
        : [3, 7, 10]; // Minor 3rd, P5, Minor 7th

      for (let i = 0; i < intervals.length && i < this._config.maxPolyphony - 1; i++) {
        const weight = trinityWeights[i] ?? 0.5;
        if (weight > 0.3) {
          const note: Note = {
            pitchClass: ((key.root + intervals[i]) % 12) as PitchClass,
            octave: 4,
            velocity: Math.round(this._config.velocity * weight),
            duration: this._config.noteDuration * (0.5 + weight * 0.5),
            channel: this._config.channel
          };

          events.push({
            type: 'note_on',
            time: this._timeCounter + i * (beatDuration / 4),
            data: note
          });
        }
      }
    }

    // Resolve ghost frequencies as additional notes
    let ghostsResolved = 0;
    if (this._config.resolveGhostFrequencies && ghostAnalysis) {
      for (const ghost of ghostAnalysis.ghosts ?? []) {
        if (ghost.resolution && ghostsResolved < 2) {
          // Map ghost vertex to a note
          const ghostKeys = this._musicDomain.vertexToKeys(ghost.vertexId ?? 0);
          if (ghostKeys.length > 0) {
            const ghostKey = this._musicDomain.getKey(ghostKeys[0]);
            if (ghostKey) {
              events.push({
                type: 'note_on',
                time: this._timeCounter + beatDuration / 2,
                data: {
                  pitchClass: ghostKey.root as PitchClass,
                  octave: 5,
                  velocity: Math.round(this._config.velocity * 0.5),
                  duration: this._config.noteDuration / 2,
                  channel: this._config.channel
                }
              });
              ghostsResolved++;
            }
          }
        }
      }
    }

    this._timeCounter += beatDuration;

    // Store in history
    this._eventHistory.push(...events);
    if (this._eventHistory.length > 1000) {
      this._eventHistory = this._eventHistory.slice(-500);
    }

    return {
      events,
      detectedKey: detectedKey,
      currentAxis: this._currentAxis,
      ghostsResolved
    };
  }

  // =========================================================================
  // API
  // =========================================================================

  getCurrentKey(): string {
    return this._currentKey;
  }

  getCurrentAxis(): TrinityAxis {
    return this._currentAxis;
  }

  getEventHistory(): MusicEvent[] {
    return [...this._eventHistory];
  }

  setTempo(bpm: number): void {
    this._config.tempo = Math.max(20, Math.min(300, bpm));
  }

  setKey(keyName: string): void {
    this._currentKey = keyName;
  }

  reset(): void {
    this._eventHistory = [];
    this._timeCounter = 0;
    this._lastModulationTime = 0;
    this._currentKey = 'C';
    this._currentAxis = 'alpha';
  }
}

export default MusicGenerationMode;
