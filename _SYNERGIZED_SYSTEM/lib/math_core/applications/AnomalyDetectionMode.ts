/**
 * Anomaly Detection Mode
 *
 * Application mode that uses the Chronomorphic Engine for anomaly detection
 * in multivariate time series data. Features:
 *
 * - Polytope-based normality modeling
 * - Topological anomaly detection via persistent homology
 * - Trinity axis correlation analysis
 * - Real-time streaming anomaly scoring
 *
 * Ported from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * with import paths adapted for PPP math_core layout.
 */

import type { Vector4D, TrinityAxis, BettiProfile } from '../geometric_algebra/types';
import { StateClassifier, type StateClassification, type StateCategory } from '../metacognition/StateClassifier';
import { HarmonicTopologist } from '../tda/PersistentHomology';
import { Lattice24 } from '../geometric_algebra/Lattice24';

// =============================================================================
// TYPES
// =============================================================================

/** Raw data point */
export interface DataPoint {
  /** Feature values */
  readonly features: number[];
  /** Timestamp */
  readonly timestamp: number;
  /** Optional label */
  readonly label?: string;
  /** Optional metadata */
  readonly metadata?: Record<string, unknown>;
}

/** Anomaly detection result */
export interface AnomalyResult {
  /** Anomaly score (0 = normal, 1 = highly anomalous) */
  readonly score: number;
  /** Whether this is classified as anomalous */
  readonly isAnomaly: boolean;
  /** Anomaly type(s) */
  readonly types: AnomalyType[];
  /** Contributing factors */
  readonly factors: AnomalyFactor[];
  /** Engine state classification */
  readonly stateClassification: StateClassification;
  /** 4D position used */
  readonly position: Vector4D;
  /** Timestamp */
  readonly timestamp: number;
}

/** Types of anomalies */
export type AnomalyType =
  | 'point'        // Single point anomaly
  | 'contextual'   // Anomalous in context but not globally
  | 'collective'   // Group of points anomalous together
  | 'topological'  // Anomalous topological structure
  | 'drift'        // Gradual concept drift
  | 'shift';       // Sudden distribution shift

/** Factor contributing to anomaly score */
export interface AnomalyFactor {
  /** Factor name */
  readonly name: string;
  /** Contribution to score (0-1) */
  readonly contribution: number;
  /** Human-readable explanation */
  readonly explanation: string;
}

/** Baseline profile for normal behavior */
export interface BaselineProfile {
  /** Mean feature values */
  readonly mean: number[];
  /** Standard deviation */
  readonly std: number[];
  /** Expected Betti numbers */
  readonly expectedBetti: BettiProfile;
  /** Number of samples in baseline */
  readonly sampleCount: number;
  /** Baseline time range */
  readonly timeRange: [number, number];
}

/** Configuration */
export interface AnomalyDetectionConfig {
  /** Anomaly threshold (0-1) */
  threshold: number;
  /** Window size for contextual detection */
  windowSize: number;
  /** Minimum samples for baseline */
  minBaselineSamples: number;
  /** Feature dimensions */
  featureDimensions: number;
  /** Enable topological analysis */
  enableTopological: boolean;
  /** Enable drift detection */
  enableDriftDetection: boolean;
  /** Exponential moving average alpha */
  emaAlpha: number;
}

// =============================================================================
// DEFAULTS
// =============================================================================

export const DEFAULT_ANOMALY_CONFIG: AnomalyDetectionConfig = {
  threshold: 0.7,
  windowSize: 50,
  minBaselineSamples: 100,
  featureDimensions: 4,
  enableTopological: true,
  enableDriftDetection: true,
  emaAlpha: 0.1
};

// =============================================================================
// ANOMALY DETECTION MODE
// =============================================================================

/**
 * Anomaly Detection Mode.
 *
 * Maps multivariate data to 4D polytope space and detects anomalies
 * via geometric and topological analysis.
 */
export class AnomalyDetectionMode {
  private _config: AnomalyDetectionConfig;
  private _classifier: StateClassifier;
  private _baseline: BaselineProfile | null;
  private _window: DataPoint[];
  private _resultHistory: AnomalyResult[];
  private _emaScore: number;
  private _driftAccumulator: number;

  constructor(config: Partial<AnomalyDetectionConfig> = {}) {
    this._config = { ...DEFAULT_ANOMALY_CONFIG, ...config };
    this._classifier = new StateClassifier();
    this._baseline = null;
    this._window = [];
    this._resultHistory = [];
    this._emaScore = 0;
    this._driftAccumulator = 0;
  }

  // =========================================================================
  // DATA PROCESSING
  // =========================================================================

  /**
   * Process a data point and return anomaly result.
   */
  process(point: DataPoint): AnomalyResult {
    // Add to window
    this._window.push(point);
    if (this._window.length > this._config.windowSize) {
      this._window.shift();
    }

    // Map features to 4D position
    const position = this._featuresToPosition(point.features);

    // Compute scores from different detectors
    const factors: AnomalyFactor[] = [];

    // 1. Distance-based score
    const distanceScore = this._computeDistanceScore(position, point.features);
    factors.push({
      name: 'distance',
      contribution: distanceScore,
      explanation: `Distance from baseline center: ${distanceScore.toFixed(3)}`
    });

    // 2. Statistical score
    const statScore = this._computeStatisticalScore(point.features);
    factors.push({
      name: 'statistical',
      contribution: statScore,
      explanation: `Z-score deviation: ${statScore.toFixed(3)}`
    });

    // 3. Contextual score
    const contextScore = this._computeContextualScore(point);
    factors.push({
      name: 'contextual',
      contribution: contextScore,
      explanation: `Contextual deviation within window: ${contextScore.toFixed(3)}`
    });

    // 4. Topological score (if enabled)
    let topoScore = 0;
    if (this._config.enableTopological && this._window.length >= 10) {
      topoScore = this._computeTopologicalScore();
      factors.push({
        name: 'topological',
        contribution: topoScore,
        explanation: `Topological anomaly: ${topoScore.toFixed(3)}`
      });
    }

    // Combine scores (weighted average)
    const weights = [0.3, 0.25, 0.25, 0.2];
    const scores = [distanceScore, statScore, contextScore, topoScore];
    const totalWeight = this._config.enableTopological
      ? weights.reduce((a, b) => a + b, 0)
      : weights.slice(0, 3).reduce((a, b) => a + b, 0);

    let combinedScore = 0;
    for (let i = 0; i < (this._config.enableTopological ? 4 : 3); i++) {
      combinedScore += scores[i] * weights[i];
    }
    combinedScore /= totalWeight;

    // Update EMA
    this._emaScore = this._config.emaAlpha * combinedScore +
                     (1 - this._config.emaAlpha) * this._emaScore;

    // Drift detection
    if (this._config.enableDriftDetection) {
      const driftScore = this._computeDriftScore(combinedScore);
      if (driftScore > 0.1) {
        factors.push({
          name: 'drift',
          contribution: driftScore,
          explanation: `Concept drift detected: ${driftScore.toFixed(3)}`
        });
      }
    }

    // Determine anomaly types
    const types = this._classifyAnomalyTypes(factors, combinedScore);

    // Classify engine state
    const stateClassification = this._classifier.classify(
      position,
      combinedScore,
      { activeAxis: 'alpha', weights: [0.33, 0.33, 0.33], tension: combinedScore, phaseProgress: 0 },
      null,
      1 - combinedScore,
      true,
      distanceScore
    );

    const result: AnomalyResult = {
      score: combinedScore,
      isAnomaly: combinedScore > this._config.threshold,
      types,
      factors,
      stateClassification,
      position,
      timestamp: point.timestamp
    };

    // Store result
    this._resultHistory.push(result);
    if (this._resultHistory.length > 1000) {
      this._resultHistory = this._resultHistory.slice(-500);
    }

    return result;
  }

  // =========================================================================
  // BASELINE
  // =========================================================================

  /**
   * Build baseline from data points.
   */
  buildBaseline(points: DataPoint[]): void {
    if (points.length < this._config.minBaselineSamples) {
      return;
    }

    const dim = this._config.featureDimensions;
    const mean = new Array(dim).fill(0);
    const std = new Array(dim).fill(0);

    // Compute mean
    for (const point of points) {
      for (let i = 0; i < dim; i++) {
        mean[i] += (point.features[i] ?? 0) / points.length;
      }
    }

    // Compute std
    for (const point of points) {
      for (let i = 0; i < dim; i++) {
        std[i] += ((point.features[i] ?? 0) - mean[i]) ** 2;
      }
    }
    for (let i = 0; i < dim; i++) {
      std[i] = Math.sqrt(std[i] / points.length);
    }

    this._baseline = {
      mean,
      std,
      expectedBetti: { beta0: 1, beta1: 0, beta2: 0 },
      sampleCount: points.length,
      timeRange: [points[0].timestamp, points[points.length - 1].timestamp]
    };
  }

  // =========================================================================
  // SCORING METHODS
  // =========================================================================

  private _featuresToPosition(features: number[]): Vector4D {
    // Map first 4 features to 4D position
    return [
      features[0] ?? 0,
      features[1] ?? 0,
      features[2] ?? 0,
      features[3] ?? 0
    ];
  }

  private _computeDistanceScore(position: Vector4D, features: number[]): number {
    if (!this._baseline) return 0;

    let sumSq = 0;
    for (let i = 0; i < features.length && i < this._baseline.mean.length; i++) {
      const diff = features[i] - this._baseline.mean[i];
      const std = this._baseline.std[i] || 1;
      sumSq += (diff / std) ** 2;
    }

    const mahalanobis = Math.sqrt(sumSq / features.length);
    return Math.min(1, mahalanobis / 3); // Normalize: 3 sigma = 1.0
  }

  private _computeStatisticalScore(features: number[]): number {
    if (!this._baseline) return 0;

    let maxZ = 0;
    for (let i = 0; i < features.length && i < this._baseline.mean.length; i++) {
      const z = Math.abs((features[i] - this._baseline.mean[i]) / (this._baseline.std[i] || 1));
      maxZ = Math.max(maxZ, z);
    }

    return Math.min(1, maxZ / 4); // Normalize: 4 sigma = 1.0
  }

  private _computeContextualScore(point: DataPoint): number {
    if (this._window.length < 5) return 0;

    // Compute local mean
    const dim = point.features.length;
    const localMean = new Array(dim).fill(0);

    for (const w of this._window) {
      for (let i = 0; i < dim; i++) {
        localMean[i] += (w.features[i] ?? 0) / this._window.length;
      }
    }

    // Compute deviation from local mean
    let maxDev = 0;
    for (let i = 0; i < dim; i++) {
      const dev = Math.abs((point.features[i] ?? 0) - localMean[i]);
      maxDev = Math.max(maxDev, dev);
    }

    return Math.min(1, maxDev / 3);
  }

  private _computeTopologicalScore(): number {
    // Simplified topological anomaly detection
    // In a full implementation, this would use PersistentHomology
    // to compare current Betti numbers with baseline
    if (!this._baseline) return 0;

    // Check if window has unexpected topological structure
    const windowSize = this._window.length;
    if (windowSize < 10) return 0;

    // Simple heuristic: large variance in recent data suggests topological change
    const recentFeatures = this._window.slice(-10).map(p => p.features[0] ?? 0);
    const mean = recentFeatures.reduce((a, b) => a + b, 0) / recentFeatures.length;
    const variance = recentFeatures.reduce((a, b) => a + (b - mean) ** 2, 0) / recentFeatures.length;

    return Math.min(1, variance / 10);
  }

  private _computeDriftScore(currentScore: number): number {
    // Detect gradual drift via CUSUM-like accumulator
    const expected = 0.3; // Expected baseline score
    const slack = 0.1;

    this._driftAccumulator = Math.max(0,
      this._driftAccumulator + (currentScore - expected - slack)
    );

    return Math.min(1, this._driftAccumulator / 5);
  }

  private _classifyAnomalyTypes(
    factors: AnomalyFactor[],
    combinedScore: number
  ): AnomalyType[] {
    const types: AnomalyType[] = [];

    if (combinedScore < this._config.threshold) return types;

    // Point anomaly: high statistical score
    const statFactor = factors.find(f => f.name === 'statistical');
    if (statFactor && statFactor.contribution > 0.6) {
      types.push('point');
    }

    // Contextual anomaly: high contextual, low statistical
    const contextFactor = factors.find(f => f.name === 'contextual');
    if (contextFactor && contextFactor.contribution > 0.5 &&
        (!statFactor || statFactor.contribution < 0.4)) {
      types.push('contextual');
    }

    // Topological anomaly
    const topoFactor = factors.find(f => f.name === 'topological');
    if (topoFactor && topoFactor.contribution > 0.5) {
      types.push('topological');
    }

    // Drift
    const driftFactor = factors.find(f => f.name === 'drift');
    if (driftFactor && driftFactor.contribution > 0.3) {
      types.push('drift');
    }

    // Default to point if nothing else matches
    if (types.length === 0) {
      types.push('point');
    }

    return types;
  }

  // =========================================================================
  // API
  // =========================================================================

  getBaseline(): BaselineProfile | null {
    return this._baseline;
  }

  getResultHistory(): AnomalyResult[] {
    return [...this._resultHistory];
  }

  getAnomalyRate(): number {
    if (this._resultHistory.length === 0) return 0;
    const anomalies = this._resultHistory.filter(r => r.isAnomaly).length;
    return anomalies / this._resultHistory.length;
  }

  setThreshold(threshold: number): void {
    this._config.threshold = Math.max(0, Math.min(1, threshold));
  }

  reset(): void {
    this._window = [];
    this._resultHistory = [];
    this._emaScore = 0;
    this._driftAccumulator = 0;
    this._classifier.reset();
  }
}

export default AnomalyDetectionMode;
