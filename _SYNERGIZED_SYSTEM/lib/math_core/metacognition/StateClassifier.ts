/**
 * State Classifier
 *
 * Rule-based state classification for the Chronomorphic Engine.
 * Classifies engine states into categories for metacognitive feedback.
 *
 * State Categories:
 * - COHERENT: High coherence, stable axis, normal operation
 * - TRANSITIONING: Phase shift in progress
 * - AMBIGUOUS: High β₂ (voids detected), uncertain state
 * - POLYTONAL: Multiple axes active simultaneously
 * - STUCK: Low velocity, oscillating near same position
 * - INVALID: Outside convex hull, non-physical state
 *
 * Ported from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * with import paths adapted for PPP math_core layout.
 */

import type {
  Vector4D,
  TrinityAxis,
  TrinityState,
  BettiProfile
} from '../geometric_algebra/types';

// =============================================================================
// TYPES
// =============================================================================

/** Possible state categories */
export type StateCategory =
  | 'COHERENT'
  | 'TRANSITIONING'
  | 'AMBIGUOUS'
  | 'POLYTONAL'
  | 'STUCK'
  | 'INVALID'
  | 'UNKNOWN';

/** Confidence level for classification */
export type ConfidenceLevel = 'high' | 'medium' | 'low';

/** State classification result */
export interface StateClassification {
  /** Primary category */
  readonly category: StateCategory;
  /** Confidence in classification */
  readonly confidence: ConfidenceLevel;
  /** Confidence score (0-1) */
  readonly score: number;
  /** Secondary possible categories */
  readonly alternatives: StateCategory[];
  /** Human-readable description */
  readonly description: string;
  /** Suggested actions */
  readonly suggestions: StateSuggestion[];
  /** Feature values used for classification */
  readonly features: StateFeatures;
}

/** Suggestion for state improvement */
export interface StateSuggestion {
  /** Action type */
  readonly action: 'navigate' | 'apply_force' | 'wait' | 'reset' | 'modulate';
  /** Target (key, vertex, axis, etc.) */
  readonly target?: string;
  /** Priority (1 = highest) */
  readonly priority: number;
  /** Expected improvement */
  readonly expectedImprovement: string;
}

/** Extracted features for classification */
export interface StateFeatures {
  // Motion features
  readonly velocity: number;
  readonly acceleration: number;
  readonly isAtRest: boolean;

  // Position features
  readonly distanceFromOrigin: number;
  readonly nearestVertexDistance: number;
  readonly isInsideHull: boolean;

  // Trinity features
  readonly dominantAxisWeight: number;
  readonly axisBalance: number; // 0 = one dominant, 1 = perfectly balanced
  readonly tension: number;
  readonly phaseShiftActive: boolean;

  // Topological features
  readonly coherence: number;
  readonly ambiguity: number;
  readonly bettiSum: number;
  readonly voidCount: number;

  // Temporal features
  readonly timeSinceLastTransition: number;
  readonly oscillationFrequency: number;
}

/** Configuration for the classifier */
export interface ClassifierConfig {
  // Thresholds
  readonly coherenceThreshold: number;
  readonly ambiguityThreshold: number;
  readonly stuckVelocityThreshold: number;
  readonly stuckTimeThreshold: number;
  readonly polytonalBalanceThreshold: number;
  readonly transitionTensionThreshold: number;

  // Weights for scoring
  readonly featureWeights: Partial<Record<keyof StateFeatures, number>>;
}

// =============================================================================
// DEFAULT CONFIGURATION
// =============================================================================

export const DEFAULT_CLASSIFIER_CONFIG: ClassifierConfig = {
  coherenceThreshold: 0.7,
  ambiguityThreshold: 0.4,
  stuckVelocityThreshold: 0.01,
  stuckTimeThreshold: 2000, // ms
  polytonalBalanceThreshold: 0.6,
  transitionTensionThreshold: 0.35,

  featureWeights: {
    coherence: 1.0,
    tension: 0.8,
    velocity: 0.6,
    ambiguity: 0.9,
    axisBalance: 0.7
  }
};

// =============================================================================
// STATE CLASSIFIER
// =============================================================================

/**
 * State Classifier for Chronomorphic Engine.
 *
 * Classifies engine states into categories based on extracted features.
 */
export class StateClassifier {
  private _config: ClassifierConfig;
  private _history: StateClassification[];
  private _lastTransitionTime: number;
  private _positionHistory: Vector4D[];

  constructor(config: Partial<ClassifierConfig> = {}) {
    this._config = { ...DEFAULT_CLASSIFIER_CONFIG, ...config };
    this._history = [];
    this._lastTransitionTime = Date.now();
    this._positionHistory = [];
  }

  /**
   * Classify the current engine state.
   */
  classify(
    position: Vector4D,
    velocity: number,
    trinityState: TrinityState,
    betti: BettiProfile | null,
    coherence: number,
    isInsideHull: boolean,
    nearestVertexDistance: number
  ): StateClassification {
    // Extract features
    const features = this._extractFeatures(
      position,
      velocity,
      trinityState,
      betti,
      coherence,
      isInsideHull,
      nearestVertexDistance
    );

    // Run classification rules
    const scores = this._computeCategoryScores(features);

    // Find best category
    const sortedCategories = Object.entries(scores)
      .sort(([, a], [, b]) => b - a) as [StateCategory, number][];

    const [bestCategory, bestScore] = sortedCategories[0];
    const alternatives = sortedCategories.slice(1, 3).map(([cat]) => cat);

    // Determine confidence
    const confidence = this._determineConfidence(bestScore, sortedCategories);

    // Generate description and suggestions
    const description = this._generateDescription(bestCategory, features);
    const suggestions = this._generateSuggestions(bestCategory, features);

    const classification: StateClassification = {
      category: bestCategory,
      confidence,
      score: bestScore,
      alternatives,
      description,
      suggestions,
      features
    };

    // Update history
    this._history.push(classification);
    if (this._history.length > 100) {
      this._history.shift();
    }

    // Track transitions
    if (this._history.length > 1) {
      const prev = this._history[this._history.length - 2];
      if (prev.category !== classification.category) {
        this._lastTransitionTime = Date.now();
      }
    }

    return classification;
  }

  /**
   * Extract features from state data.
   */
  private _extractFeatures(
    position: Vector4D,
    velocity: number,
    trinityState: TrinityState,
    betti: BettiProfile | null,
    coherence: number,
    isInsideHull: boolean,
    nearestVertexDistance: number
  ): StateFeatures {
    // Update position history
    this._positionHistory.push(position);
    if (this._positionHistory.length > 60) {
      this._positionHistory.shift();
    }

    // Calculate derived features
    const distanceFromOrigin = Math.sqrt(
      position[0]**2 + position[1]**2 + position[2]**2 + position[3]**2
    );

    // Axis balance: 0 = one dominant, 1 = perfectly equal
    const [w1, w2, w3] = trinityState.weights;
    const maxWeight = Math.max(w1, w2, w3);
    const minWeight = Math.min(w1, w2, w3);
    const axisBalance = 1 - (maxWeight - minWeight);

    // Acceleration estimate
    let acceleration = 0;
    if (this._history.length > 1) {
      const prevVel = this._history[this._history.length - 1].features.velocity;
      acceleration = velocity - prevVel;
    }

    // Oscillation frequency (simplified)
    const oscillationFrequency = this._estimateOscillation();

    // Betti-derived features
    const bettiSum = betti ? betti.beta0 + betti.beta1 + betti.beta2 : 0;
    const voidCount = betti ? betti.beta2 : 0;
    const ambiguity = betti ? Math.min(1, betti.beta2 * 0.3 + betti.beta1 * 0.1) : 0;

    return {
      velocity,
      acceleration,
      isAtRest: velocity < this._config.stuckVelocityThreshold,
      distanceFromOrigin,
      nearestVertexDistance,
      isInsideHull,
      dominantAxisWeight: maxWeight,
      axisBalance,
      tension: trinityState.tension,
      phaseShiftActive: trinityState.phaseProgress > 0 && trinityState.phaseProgress < 1,
      coherence,
      ambiguity,
      bettiSum,
      voidCount,
      timeSinceLastTransition: Date.now() - this._lastTransitionTime,
      oscillationFrequency
    };
  }

  /**
   * Compute scores for each category.
   */
  private _computeCategoryScores(features: StateFeatures): Record<StateCategory, number> {
    const cfg = this._config;

    // COHERENT: High coherence, stable, inside hull
    const coherentScore = features.isInsideHull
      ? features.coherence * 0.4 +
        (1 - features.tension) * 0.3 +
        features.dominantAxisWeight * 0.2 +
        (features.velocity > 0.01 ? 0.1 : 0)
      : 0;

    // TRANSITIONING: Phase shift active, high tension
    const transitioningScore = features.phaseShiftActive
      ? 0.6 + features.tension * 0.3 + (1 - features.axisBalance) * 0.1
      : features.tension > cfg.transitionTensionThreshold
        ? 0.3 + features.tension * 0.4
        : 0;

    // AMBIGUOUS: High beta2, voids present
    const ambiguousScore = features.ambiguity > cfg.ambiguityThreshold
      ? features.ambiguity * 0.5 + (features.voidCount > 0 ? 0.3 : 0) + (1 - features.coherence) * 0.2
      : features.ambiguity * 0.3;

    // POLYTONAL: Balanced axes, multiple active
    const polytonalScore = features.axisBalance > cfg.polytonalBalanceThreshold
      ? features.axisBalance * 0.5 + (features.tension > 0.2 ? 0.3 : 0.1) + features.coherence * 0.2
      : 0;

    // STUCK: Low velocity for extended time
    const stuckScore = features.isAtRest &&
                       features.timeSinceLastTransition > cfg.stuckTimeThreshold
      ? 0.5 + (1 - features.velocity / cfg.stuckVelocityThreshold) * 0.3 +
        features.oscillationFrequency * 0.2
      : 0;

    // INVALID: Outside convex hull
    const invalidScore = !features.isInsideHull
      ? 0.8 + features.distanceFromOrigin * 0.1
      : 0;

    return {
      'COHERENT': Math.min(1, coherentScore),
      'TRANSITIONING': Math.min(1, transitioningScore),
      'AMBIGUOUS': Math.min(1, ambiguousScore),
      'POLYTONAL': Math.min(1, polytonalScore),
      'STUCK': Math.min(1, stuckScore),
      'INVALID': Math.min(1, invalidScore),
      'UNKNOWN': 0.1 // Base score for unknown
    };
  }

  /**
   * Determine confidence level from scores.
   */
  private _determineConfidence(
    bestScore: number,
    sorted: [StateCategory, number][]
  ): ConfidenceLevel {
    if (bestScore > 0.7 && (sorted.length < 2 || sorted[1][1] < bestScore - 0.3)) {
      return 'high';
    }
    if (bestScore > 0.4 && (sorted.length < 2 || sorted[1][1] < bestScore - 0.15)) {
      return 'medium';
    }
    return 'low';
  }

  /**
   * Generate human-readable description.
   */
  private _generateDescription(category: StateCategory, features: StateFeatures): string {
    switch (category) {
      case 'COHERENT':
        return `Stable state on ${features.dominantAxisWeight > 0.6 ? 'dominant' : 'balanced'} axis with ${(features.coherence * 100).toFixed(0)}% coherence`;
      case 'TRANSITIONING':
        return `Phase transition in progress (tension: ${(features.tension * 100).toFixed(0)}%)`;
      case 'AMBIGUOUS':
        return `Ambiguous state with ${features.voidCount} topological void(s) detected`;
      case 'POLYTONAL':
        return `Polytonal superposition across multiple axes (balance: ${(features.axisBalance * 100).toFixed(0)}%)`;
      case 'STUCK':
        return `Motion stalled for ${(features.timeSinceLastTransition / 1000).toFixed(1)}s`;
      case 'INVALID':
        return `Invalid state: outside convex hull at distance ${features.distanceFromOrigin.toFixed(3)}`;
      default:
        return 'Unknown state';
    }
  }

  /**
   * Generate action suggestions.
   */
  private _generateSuggestions(category: StateCategory, features: StateFeatures): StateSuggestion[] {
    const suggestions: StateSuggestion[] = [];

    switch (category) {
      case 'COHERENT':
        break;
      case 'TRANSITIONING':
        suggestions.push({
          action: 'wait',
          priority: 1,
          expectedImprovement: 'Allow transition to complete naturally'
        });
        break;
      case 'AMBIGUOUS':
        suggestions.push({
          action: 'navigate',
          target: 'nearest_vertex',
          priority: 1,
          expectedImprovement: 'Collapse void by activating ghost frequency'
        });
        suggestions.push({
          action: 'apply_force',
          target: 'toward_center',
          priority: 2,
          expectedImprovement: 'Increase coherence by moving toward lattice center'
        });
        break;
      case 'POLYTONAL':
        suggestions.push({
          action: 'modulate',
          target: 'dominant_axis',
          priority: 1,
          expectedImprovement: 'Resolve to single axis for clarity'
        });
        break;
      case 'STUCK':
        suggestions.push({
          action: 'apply_force',
          target: 'random',
          priority: 1,
          expectedImprovement: 'Break oscillation with perturbation'
        });
        suggestions.push({
          action: 'navigate',
          target: 'adjacent_key',
          priority: 2,
          expectedImprovement: 'Move to neighboring harmonic region'
        });
        break;
      case 'INVALID':
        suggestions.push({
          action: 'reset',
          priority: 1,
          expectedImprovement: 'Return to valid state space'
        });
        suggestions.push({
          action: 'navigate',
          target: 'nearest_vertex',
          priority: 2,
          expectedImprovement: 'Project back to nearest valid lattice point'
        });
        break;
    }

    return suggestions;
  }

  /**
   * Estimate oscillation frequency from position history.
   */
  private _estimateOscillation(): number {
    if (this._positionHistory.length < 10) return 0;

    let directionChanges = 0;
    for (let i = 2; i < this._positionHistory.length; i++) {
      const prev = this._positionHistory[i - 2];
      const curr = this._positionHistory[i - 1];
      const next = this._positionHistory[i];

      const delta1 = [
        curr[0] - prev[0], curr[1] - prev[1],
        curr[2] - prev[2], curr[3] - prev[3]
      ];
      const delta2 = [
        next[0] - curr[0], next[1] - curr[1],
        next[2] - curr[2], next[3] - curr[3]
      ];

      const dot = delta1[0]*delta2[0] + delta1[1]*delta2[1] +
                  delta1[2]*delta2[2] + delta1[3]*delta2[3];

      if (dot < 0) directionChanges++;
    }

    return directionChanges / (this._positionHistory.length - 2);
  }

  getHistory(): StateClassification[] {
    return [...this._history];
  }

  getTransitionStats(): Record<string, number> {
    const transitions: Record<string, number> = {};

    for (let i = 1; i < this._history.length; i++) {
      const from = this._history[i - 1].category;
      const to = this._history[i].category;
      if (from !== to) {
        const key = `${from}->${to}`;
        transitions[key] = (transitions[key] || 0) + 1;
      }
    }

    return transitions;
  }

  reset(): void {
    this._history = [];
    this._positionHistory = [];
    this._lastTransitionTime = Date.now();
  }
}

export default StateClassifier;
