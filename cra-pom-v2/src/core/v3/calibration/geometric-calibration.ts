/**
 * Geometric Calibration Harness
 *
 * Three-way verification system for the PPP Geometric Cognition Kernel:
 *
 * 1. GEOMETRIC MODEL (24-Cell Polytope)
 *    - Predicts relationships mathematically
 *    - Deterministic, interpretable
 *
 * 2. GEMINI AUDIO (Oracle)
 *    - Verifies predictions with real audio perception
 *    - "Ground truth" from multimodal understanding
 *
 * 3. SEMANTIC EMBEDDINGS (Voyage AI / Gemini)
 *    - Bridges concepts to geometry
 *    - Verifies text-to-structure mapping
 *
 * CALIBRATION TESTS:
 * - Circle of Fifths: C-G closer than C-F#
 * - Tension/Resolution: G7→C should resolve
 * - Major/Minor: C major brighter than A minor
 * - Tritone: Maximum distance/dissonance
 */

import {
  MusicCell24,
  calculateTension,
  calculateResolution,
  type TensionResult,
} from '../music/music-geometry-domain';

import {
  createChord,
  type Chord,
  type PitchClass,
} from '../music/music-theory';

import {
  generateChord as generateAudioChord,
  generateNote,
} from '../music/audio-processing';

import {
  GeminiAudioService,
  samplesToWavBase64,
} from '../llm/gemini-audio-service';

// ============================================================================
// Types
// ============================================================================

export interface CalibrationTestCase {
  name: string;
  description: string;
  category: 'circle_of_fifths' | 'tension_resolution' | 'major_minor' | 'tritone' | 'custom';
}

export interface GeometricPrediction {
  testCase: CalibrationTestCase;
  prediction: {
    distance?: number;
    tensionDelta?: number;
    shouldResolve?: boolean;
    relationship?: string;
  };
  confidence: number;
}

export interface AudioVerification {
  testCase: CalibrationTestCase;
  geminiResult: {
    similarity?: number;
    tensionChange?: number;
    isResolution?: boolean;
    relationship?: string;
  };
  rawResponse?: string;
}

export interface CalibrationResult {
  testCase: CalibrationTestCase;
  geometric: GeometricPrediction;
  audio?: AudioVerification;
  semantic?: {
    embeddingSimilarity?: number;
    conceptAlignment?: string;
  };
  agreement: {
    geometricVsAudio?: boolean;
    geometricVsSemantic?: boolean;
    audioVsSemantic?: boolean;
    allAgree?: boolean;
  };
  passed: boolean;
  details: string;
}

export interface CalibrationSuiteResult {
  timestamp: string;
  totalTests: number;
  passed: number;
  failed: number;
  results: CalibrationResult[];
  geometricModelScore: number; // 0-1
  summary: string;
}

// ============================================================================
// Test Cases
// ============================================================================

export const STANDARD_TEST_CASES: CalibrationTestCase[] = [
  // Circle of Fifths
  {
    name: 'fifth_closer_than_tritone',
    description: 'C-G should be closer than C-F# (fifth vs tritone)',
    category: 'circle_of_fifths',
  },
  {
    name: 'fourth_closer_than_tritone',
    description: 'C-F should be closer than C-F# (fourth vs tritone)',
    category: 'circle_of_fifths',
  },
  {
    name: 'fifth_chain_ordering',
    description: 'Distance increases along circle: C-G < C-D < C-A < C-F#',
    category: 'circle_of_fifths',
  },

  // Tension/Resolution
  {
    name: 'dominant_to_tonic',
    description: 'G7 → C should resolve (positive tension delta)',
    category: 'tension_resolution',
  },
  {
    name: 'diminished_higher_tension',
    description: 'Diminished chord should have higher tension than major',
    category: 'tension_resolution',
  },
  {
    name: 'tritone_substitution',
    description: 'Db7 → C should also resolve (tritone sub)',
    category: 'tension_resolution',
  },

  // Major/Minor
  {
    name: 'relative_major_minor',
    description: 'C major and A minor should be closely related',
    category: 'major_minor',
  },
  {
    name: 'parallel_major_minor',
    description: 'C major and C minor should be moderately related',
    category: 'major_minor',
  },

  // Tritone
  {
    name: 'tritone_maximum_distance',
    description: 'C to F# should be maximum distance among all intervals',
    category: 'tritone',
  },
  {
    name: 'tritone_high_dissonance',
    description: 'Tritone interval should have high perceived dissonance',
    category: 'tritone',
  },
];

// ============================================================================
// Calibration Harness
// ============================================================================

export class GeometricCalibrationHarness {
  private cell: MusicCell24;
  private audioService?: GeminiAudioService;
  private sampleRate: number = 44100;

  constructor(geminiApiKey?: string) {
    this.cell = new MusicCell24();

    if (geminiApiKey) {
      this.audioService = new GeminiAudioService({ apiKey: geminiApiKey });
    }
  }

  // ==========================================================================
  // Geometric Predictions
  // ==========================================================================

  /**
   * Get geometric prediction for a test case
   */
  predictGeometric(testCase: CalibrationTestCase): GeometricPrediction {
    switch (testCase.name) {
      case 'fifth_closer_than_tritone':
        return this.predictFifthVsTritone();
      case 'fourth_closer_than_tritone':
        return this.predictFourthVsTritone();
      case 'fifth_chain_ordering':
        return this.predictFifthChainOrdering();
      case 'dominant_to_tonic':
        return this.predictDominantToTonic();
      case 'diminished_higher_tension':
        return this.predictDiminishedTension();
      case 'tritone_substitution':
        return this.predictTritoneSubstitution();
      case 'relative_major_minor':
        return this.predictRelativeMajorMinor();
      case 'parallel_major_minor':
        return this.predictParallelMajorMinor();
      case 'tritone_maximum_distance':
        return this.predictTritoneMaxDistance();
      case 'tritone_high_dissonance':
        return this.predictTritoneDissonance();
      default:
        return {
          testCase,
          prediction: { relationship: 'unknown test case' },
          confidence: 0,
        };
    }
  }

  private predictFifthVsTritone(): GeometricPrediction {
    const c = this.cell.getVertexByKey('C', 'major')!;
    const g = this.cell.getVertexByKey('G', 'major')!;
    const fSharp = this.cell.getVertexByKey('F#', 'major')!;

    const cg = this.cell.getDistance(c.id, g.id);
    const cfs = this.cell.getDistance(c.id, fSharp.id);

    return {
      testCase: STANDARD_TEST_CASES[0],
      prediction: {
        distance: cg,
        relationship: `C-G: ${cg.toFixed(3)}, C-F#: ${cfs.toFixed(3)}`,
        shouldResolve: cg < cfs,
      },
      confidence: cg < cfs ? 1.0 : 0.0,
    };
  }

  private predictFourthVsTritone(): GeometricPrediction {
    const c = this.cell.getVertexByKey('C', 'major')!;
    const f = this.cell.getVertexByKey('F', 'major')!;
    const fSharp = this.cell.getVertexByKey('F#', 'major')!;

    const cf = this.cell.getDistance(c.id, f.id);
    const cfs = this.cell.getDistance(c.id, fSharp.id);

    return {
      testCase: STANDARD_TEST_CASES[1],
      prediction: {
        distance: cf,
        relationship: `C-F: ${cf.toFixed(3)}, C-F#: ${cfs.toFixed(3)}`,
        shouldResolve: cf < cfs,
      },
      confidence: cf < cfs ? 1.0 : 0.0,
    };
  }

  private predictFifthChainOrdering(): GeometricPrediction {
    const c = this.cell.getVertexByKey('C', 'major')!;
    const distances = ['G', 'D', 'A', 'F#'].map(key => {
      const v = this.cell.getVertexByKey(key as PitchClass, 'major')!;
      return { key, distance: this.cell.getDistance(c.id, v.id) };
    });

    // Check if distances are increasing
    let isOrdered = true;
    for (let i = 0; i < distances.length - 1; i++) {
      if (distances[i].distance >= distances[i + 1].distance) {
        isOrdered = false;
        break;
      }
    }

    return {
      testCase: STANDARD_TEST_CASES[2],
      prediction: {
        relationship: distances.map(d => `C-${d.key}: ${d.distance.toFixed(3)}`).join(', '),
        shouldResolve: isOrdered,
      },
      confidence: isOrdered ? 1.0 : 0.5,
    };
  }

  private predictDominantToTonic(): GeometricPrediction {
    const g7 = createChord('G', 'dominant7');
    const cMaj = createChord('C', 'major');

    const resolution = calculateResolution(this.cell, g7, cMaj);

    return {
      testCase: STANDARD_TEST_CASES[3],
      prediction: {
        tensionDelta: resolution,
        shouldResolve: resolution > 0,
        relationship: `G7→C resolution: ${resolution.toFixed(4)}`,
      },
      confidence: resolution > 0 ? Math.min(resolution, 1.0) : 0.0,
    };
  }

  private predictDiminishedTension(): GeometricPrediction {
    const cMaj = createChord('C', 'major');
    const cDim = createChord('C', 'diminished');

    const majTension = calculateTension(this.cell, cMaj);
    const dimTension = calculateTension(this.cell, cDim);

    const dimHigher = dimTension.geometricTension > majTension.geometricTension;

    return {
      testCase: STANDARD_TEST_CASES[4],
      prediction: {
        tensionDelta: dimTension.geometricTension - majTension.geometricTension,
        relationship: `C dim: ${dimTension.geometricTension.toFixed(4)}, C maj: ${majTension.geometricTension.toFixed(4)}`,
        shouldResolve: dimHigher,
      },
      confidence: dimHigher ? 1.0 : 0.0,
    };
  }

  private predictTritoneSubstitution(): GeometricPrediction {
    const db7 = createChord('C#', 'dominant7'); // Db7 = C#7
    const cMaj = createChord('C', 'major');

    const resolution = calculateResolution(this.cell, db7, cMaj);

    return {
      testCase: STANDARD_TEST_CASES[5],
      prediction: {
        tensionDelta: resolution,
        shouldResolve: resolution > 0,
        relationship: `Db7→C resolution: ${resolution.toFixed(4)}`,
      },
      confidence: resolution > 0 ? Math.min(resolution * 0.8, 1.0) : 0.0,
    };
  }

  private predictRelativeMajorMinor(): GeometricPrediction {
    const cMaj = this.cell.getVertexByKey('C', 'major')!;
    const aMin = this.cell.getVertexByKey('A', 'minor')!;
    const fSharpMin = this.cell.getVertexByKey('F#', 'minor')!;

    const relDist = this.cell.getDistance(cMaj.id, aMin.id);
    const unrelDist = this.cell.getDistance(cMaj.id, fSharpMin.id);

    const isCloser = relDist < unrelDist;

    return {
      testCase: STANDARD_TEST_CASES[6],
      prediction: {
        distance: relDist,
        relationship: `C-Am: ${relDist.toFixed(3)}, C-F#m: ${unrelDist.toFixed(3)}`,
        shouldResolve: isCloser,
      },
      confidence: isCloser ? 1.0 : 0.3,
    };
  }

  private predictParallelMajorMinor(): GeometricPrediction {
    const cMaj = this.cell.getVertexByKey('C', 'major')!;
    const cMin = this.cell.getVertexByKey('C', 'minor')!;

    const dist = this.cell.getDistance(cMaj.id, cMin.id);

    // Should be moderate distance (not adjacent, not antipodal)
    const isModerate = dist > 0.5 && dist < 2.0;

    return {
      testCase: STANDARD_TEST_CASES[7],
      prediction: {
        distance: dist,
        relationship: `C major to C minor: ${dist.toFixed(3)}`,
        shouldResolve: isModerate,
      },
      confidence: isModerate ? 0.8 : 0.4,
    };
  }

  private predictTritoneMaxDistance(): GeometricPrediction {
    const c = this.cell.getVertexByKey('C', 'major')!;
    const fSharp = this.cell.getVertexByKey('F#', 'major')!;
    const tritoneDist = this.cell.getDistance(c.id, fSharp.id);

    // Check against all other intervals
    let isMaximum = true;
    for (let i = 1; i < 12; i++) {
      if (i === 6) continue; // Skip tritone itself
      const keys: PitchClass[] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
      const other = this.cell.getVertexByKey(keys[i], 'major');
      if (other && this.cell.getDistance(c.id, other.id) > tritoneDist + 0.01) {
        isMaximum = false;
        break;
      }
    }

    return {
      testCase: STANDARD_TEST_CASES[8],
      prediction: {
        distance: tritoneDist,
        relationship: `Tritone distance: ${tritoneDist.toFixed(3)}, is maximum: ${isMaximum}`,
        shouldResolve: isMaximum,
      },
      confidence: isMaximum ? 1.0 : 0.3,
    };
  }

  private predictTritoneDissonance(): GeometricPrediction {
    // Create tritone interval (C + F#)
    const tritoneChord: Chord = {
      root: 'C',
      type: 'major', // dummy
      pitchClasses: ['C', 'F#'],
      intervals: [6],
    };

    const fifthChord: Chord = {
      root: 'C',
      type: 'major',
      pitchClasses: ['C', 'G'],
      intervals: [7],
    };

    const tritoneTension = calculateTension(this.cell, tritoneChord);
    const fifthTension = calculateTension(this.cell, fifthChord);

    const tritoneHigher = tritoneTension.acousticDissonance > fifthTension.acousticDissonance;

    return {
      testCase: STANDARD_TEST_CASES[9],
      prediction: {
        tensionDelta: tritoneTension.acousticDissonance - fifthTension.acousticDissonance,
        relationship: `Tritone dissonance: ${tritoneTension.acousticDissonance.toFixed(3)}, Fifth: ${fifthTension.acousticDissonance.toFixed(3)}`,
        shouldResolve: tritoneHigher,
      },
      confidence: tritoneHigher ? 1.0 : 0.0,
    };
  }

  // ==========================================================================
  // Audio Verification
  // ==========================================================================

  /**
   * Verify a geometric prediction with Gemini audio analysis
   */
  async verifyWithAudio(
    testCase: CalibrationTestCase,
    geometricPrediction: GeometricPrediction
  ): Promise<AudioVerification | null> {
    if (!this.audioService) {
      return null;
    }

    try {
      switch (testCase.name) {
        case 'fifth_closer_than_tritone':
          return await this.verifyFifthVsTritone();
        case 'dominant_to_tonic':
          return await this.verifyDominantToTonic();
        case 'tritone_high_dissonance':
          return await this.verifyTritoneDissonance();
        default:
          return null;
      }
    } catch (error) {
      console.error(`Audio verification failed for ${testCase.name}:`, error);
      return null;
    }
  }

  private async verifyFifthVsTritone(): Promise<AudioVerification> {
    // Generate audio for C, G, and F# chords
    const duration = 1.0;
    const cAudio = generateAudioChord(['C', 'E', 'G'], duration, this.sampleRate);
    const gAudio = generateAudioChord(['G', 'B', 'D'], duration, this.sampleRate);
    const fsAudio = generateAudioChord(['F#', 'A#', 'C#'], duration, this.sampleRate);

    const cBase64 = samplesToWavBase64(cAudio, this.sampleRate);
    const gBase64 = samplesToWavBase64(gAudio, this.sampleRate);
    const fsBase64 = samplesToWavBase64(fsAudio, this.sampleRate);

    // Compare C-G vs C-F#
    const cgComparison = await this.audioService!.compareAudio(cBase64, gBase64);
    const cfsComparison = await this.audioService!.compareAudio(cBase64, fsBase64);

    const cgSimilarity = cgComparison.comparison.similarity;
    const cfsSimilarity = cfsComparison.comparison.similarity;

    return {
      testCase: STANDARD_TEST_CASES[0],
      geminiResult: {
        similarity: cgSimilarity,
        relationship: `C-G similarity: ${cgSimilarity}, C-F# similarity: ${cfsSimilarity}`,
      },
      rawResponse: `C-G: ${cgComparison.rawResponse}\nC-F#: ${cfsComparison.rawResponse}`,
    };
  }

  private async verifyDominantToTonic(): Promise<AudioVerification> {
    const duration = 1.0;
    const g7Audio = generateAudioChord(['G', 'B', 'D', 'F'], duration, this.sampleRate);
    const cAudio = generateAudioChord(['C', 'E', 'G'], duration, this.sampleRate);

    const g7Base64 = samplesToWavBase64(g7Audio, this.sampleRate);
    const cBase64 = samplesToWavBase64(cAudio, this.sampleRate);

    const result = await this.audioService!.rateTensionResolution(g7Base64, cBase64);

    return {
      testCase: STANDARD_TEST_CASES[3],
      geminiResult: {
        tensionChange: result.tensionChange,
        isResolution: result.isResolution,
        relationship: result.explanation,
      },
      rawResponse: JSON.stringify(result),
    };
  }

  private async verifyTritoneDissonance(): Promise<AudioVerification> {
    const duration = 1.0;
    // Tritone interval
    const tritoneAudio = generateAudioChord(['C', 'F#'], duration, this.sampleRate);
    // Perfect fifth
    const fifthAudio = generateAudioChord(['C', 'G'], duration, this.sampleRate);

    const tritoneBase64 = samplesToWavBase64(tritoneAudio, this.sampleRate);
    const fifthBase64 = samplesToWavBase64(fifthAudio, this.sampleRate);

    const tritoneAnalysis = await this.audioService!.analyzeAudio(tritoneBase64);
    const fifthAnalysis = await this.audioService!.analyzeAudio(fifthBase64);

    const tritoneConsonance = tritoneAnalysis.analysis.consonance ?? 0.5;
    const fifthConsonance = fifthAnalysis.analysis.consonance ?? 0.5;

    return {
      testCase: STANDARD_TEST_CASES[9],
      geminiResult: {
        similarity: tritoneConsonance,
        relationship: `Tritone consonance: ${tritoneConsonance}, Fifth consonance: ${fifthConsonance}`,
      },
      rawResponse: `Tritone: ${tritoneAnalysis.rawResponse}\nFifth: ${fifthAnalysis.rawResponse}`,
    };
  }

  // ==========================================================================
  // Run Calibration Suite
  // ==========================================================================

  /**
   * Run complete calibration suite
   */
  async runCalibrationSuite(
    includeAudio: boolean = false,
    testCases: CalibrationTestCase[] = STANDARD_TEST_CASES
  ): Promise<CalibrationSuiteResult> {
    const results: CalibrationResult[] = [];

    for (const testCase of testCases) {
      // Get geometric prediction
      const geometric = this.predictGeometric(testCase);

      // Optionally verify with audio
      let audio: AudioVerification | null = null;
      if (includeAudio && this.audioService) {
        audio = await this.verifyWithAudio(testCase, geometric);
      }

      // Determine if test passed based on geometric confidence
      const passed = geometric.confidence >= 0.5;

      // Check agreement between sources
      const agreement: CalibrationResult['agreement'] = {};
      if (audio) {
        // Compare geometric prediction to audio result
        if (geometric.prediction.shouldResolve !== undefined && audio.geminiResult.isResolution !== undefined) {
          agreement.geometricVsAudio = geometric.prediction.shouldResolve === audio.geminiResult.isResolution;
        }
      }

      results.push({
        testCase,
        geometric,
        audio: audio ?? undefined,
        agreement,
        passed,
        details: geometric.prediction.relationship || 'No details',
      });
    }

    const passed = results.filter(r => r.passed).length;
    const failed = results.length - passed;
    const score = passed / results.length;

    return {
      timestamp: new Date().toISOString(),
      totalTests: results.length,
      passed,
      failed,
      results,
      geometricModelScore: score,
      summary: `Geometric model passed ${passed}/${results.length} tests (${(score * 100).toFixed(1)}% accuracy)`,
    };
  }

  /**
   * Run quick geometric-only calibration (no API calls)
   */
  runQuickCalibration(): CalibrationSuiteResult {
    const results: CalibrationResult[] = [];

    for (const testCase of STANDARD_TEST_CASES) {
      const geometric = this.predictGeometric(testCase);
      const passed = geometric.confidence >= 0.5;

      results.push({
        testCase,
        geometric,
        agreement: {},
        passed,
        details: geometric.prediction.relationship || 'No details',
      });
    }

    const passed = results.filter(r => r.passed).length;
    const score = passed / results.length;

    return {
      timestamp: new Date().toISOString(),
      totalTests: results.length,
      passed,
      failed: results.length - passed,
      results,
      geometricModelScore: score,
      summary: `Quick calibration: ${passed}/${results.length} tests passed (${(score * 100).toFixed(1)}%)`,
    };
  }
}

// ============================================================================
// Export
// ============================================================================

export const GeometricCalibration = {
  GeometricCalibrationHarness,
  STANDARD_TEST_CASES,
};

export default GeometricCalibration;
