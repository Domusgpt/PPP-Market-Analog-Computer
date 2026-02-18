/**
 * Trinity Engine: Multi-State Superposition Model
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module implements the Trinity Architecture for the Chronomorphic Polytopal Engine.
 * It manages the multi-state superposition across the three 16-cells (Alpha/Beta/Gamma)
 * and detects phase shifts (analogous to musical modulation).
 *
 * Core Concepts:
 * - State Vector Ψ = [w_α, w_β, w_γ] representing superposition of all three axes
 * - Phase shifts detected when dominant axis changes
 * - Tension measured as deviation from single-axis dominance
 *
 * Musical Mapping:
 * - Alpha → OCT₀,₁ = {C, C♯, E♭, E, F♯, G, A, B♭}
 * - Beta  → OCT₁,₂
 * - Gamma → OCT₀,₂
 *
 * Mathematical Foundation:
 * - W(D₄) ⊂ W(F₄) with index 3 (D₄ triality)
 * - Each 16-cell is self-dual and regular
 * - Phase shifts correspond to rotations between axis pairs
 */

import {
  Vector4D,
  TrinityAxis,
  TrinityState,
  PhaseShiftInfo,
  TelemetryEvent,
  TelemetryEventType,
  TelemetrySubscriber,
  MATH_CONSTANTS
} from '../geometric_algebra/types.js';

import { Lattice24, getDefaultLattice } from '../topology/CPE_Lattice24.js';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Minimum weight change to trigger phase shift detection */
const PHASE_SHIFT_THRESHOLD = 0.15;

/** Tension threshold above which polytonality is detected */
const POLYTONAL_THRESHOLD = 0.4;

/** Decay rate for tension when staying in same axis */
const TENSION_DECAY_RATE = 0.95;

/** Rate at which phase shift progress updates */
const PHASE_PROGRESS_RATE = 0.1;

// =============================================================================
// TRINITY ENGINE CLASS
// =============================================================================

/**
 * TrinityEngine manages the multi-state superposition across the three 16-cells.
 *
 * Usage:
 * ```typescript
 * const trinity = new TrinityEngine();
 * const result = trinity.updatePosition([0.5, 0.5, 0.3, 0.2]);
 * console.log(`Dominant axis: ${result.state.activeAxis}`);
 * console.log(`Tension: ${result.state.tension}`);
 * ```
 */
export class TrinityEngine {
  private _state: TrinityState;
  private _lattice: Lattice24;
  private _previousVertex: number;
  private _phaseShiftInProgress: boolean;
  private _phaseShiftInfo: PhaseShiftInfo | null;
  private _subscribers: Set<TelemetrySubscriber>;
  private _updateCount: number;

  constructor(lattice?: Lattice24) {
    this._lattice = lattice ?? getDefaultLattice();
    this._state = this._createInitialState();
    this._previousVertex = 0;
    this._phaseShiftInProgress = false;
    this._phaseShiftInfo = null;
    this._subscribers = new Set();
    this._updateCount = 0;
  }

  // =========================================================================
  // INITIALIZATION
  // =========================================================================

  private _createInitialState(): TrinityState {
    return {
      activeAxis: 'alpha',
      weights: [1, 0, 0],
      tension: 0,
      phaseProgress: 0
    };
  }

  // =========================================================================
  // ACCESSORS
  // =========================================================================

  get state(): TrinityState {
    return this._state;
  }

  get lattice(): Lattice24 {
    return this._lattice;
  }

  get isPhaseShiftInProgress(): boolean {
    return this._phaseShiftInProgress;
  }

  get currentPhaseShift(): PhaseShiftInfo | null {
    return this._phaseShiftInfo;
  }

  get updateCount(): number {
    return this._updateCount;
  }

  // =========================================================================
  // STATE UPDATE
  // =========================================================================

  /**
   * Update the Trinity state based on a new position in 4D space.
   *
   * @param position - Current position in the Orthocognitum
   * @param k - Number of nearest vertices to consider (default 4)
   * @returns Updated state and any phase shift information
   */
  updatePosition(position: Vector4D, k: number = 4): {
    state: TrinityState;
    phaseShift: PhaseShiftInfo | null;
    transitionComplete: boolean;
  } {
    this._updateCount++;

    // Get current Trinity weights
    const weights = this._lattice.getTrinityWeights(position, k);

    // Find the dominant axis
    const dominantAxis = this._determineDominantAxis(weights);

    // Check for phase shift
    let phaseShift: PhaseShiftInfo | null = null;
    let transitionComplete = false;

    if (dominantAxis !== this._state.activeAxis) {
      // Potential phase shift detected
      if (!this._phaseShiftInProgress) {
        // Start new phase shift
        phaseShift = this._startPhaseShift(this._state.activeAxis, dominantAxis);
        this._phaseShiftInProgress = true;
        this._phaseShiftInfo = phaseShift;
      }
    }

    // Update phase progress
    let phaseProgress = this._state.phaseProgress;
    if (this._phaseShiftInProgress) {
      phaseProgress = Math.min(1, phaseProgress + PHASE_PROGRESS_RATE);

      if (phaseProgress >= 1) {
        // Phase shift complete
        transitionComplete = true;
        this._phaseShiftInProgress = false;
        phaseShift = this._phaseShiftInfo;
        this._phaseShiftInfo = null;
        phaseProgress = 0;

        this._emitEvent(TelemetryEventType.PHASE_SHIFT_COMPLETE, {
          from: this._state.activeAxis,
          to: dominantAxis,
          position
        });
      }
    }

    // Calculate tension (deviation from pure state)
    const tension = this._calculateTension(weights);

    // Update state
    const newState: TrinityState = {
      activeAxis: transitionComplete ? dominantAxis : this._state.activeAxis,
      weights: weights as [number, number, number],
      tension,
      phaseProgress
    };

    // Apply tension decay if staying in same axis
    if (!this._phaseShiftInProgress && tension < POLYTONAL_THRESHOLD) {
      newState.weights[0] = this._state.weights[0] + (weights[0] - this._state.weights[0]) * 0.1;
      newState.weights[1] = this._state.weights[1] + (weights[1] - this._state.weights[1]) * 0.1;
      newState.weights[2] = this._state.weights[2] + (weights[2] - this._state.weights[2]) * 0.1;
    }

    this._state = newState;

    // Update previous vertex
    this._previousVertex = this._lattice.findNearest(position);

    return {
      state: newState,
      phaseShift,
      transitionComplete
    };
  }

  /**
   * Update based on vertex transition (for discrete navigation).
   */
  updateVertex(newVertex: number): {
    state: TrinityState;
    phaseShift: PhaseShiftInfo | null;
  } {
    const vertex = this._lattice.getVertex(newVertex);
    if (!vertex) {
      return { state: this._state, phaseShift: null };
    }

    return this.updatePosition(vertex.coordinates);
  }

  // =========================================================================
  // PHASE SHIFT DETECTION
  // =========================================================================

  private _determineDominantAxis(weights: [number, number, number]): TrinityAxis {
    const [alpha, beta, gamma] = weights;

    if (alpha >= beta && alpha >= gamma) return 'alpha';
    if (beta >= gamma) return 'beta';
    return 'gamma';
  }

  private _startPhaseShift(from: TrinityAxis, to: TrinityAxis): PhaseShiftInfo {
    this._emitEvent(TelemetryEventType.PHASE_SHIFT_START, {
      from,
      to
    });

    // Use lattice to get phase shift info
    // Find a representative vertex transition
    const fromVertices = this._lattice.getAxisVertices(from);
    const toVertices = this._lattice.getAxisVertices(to);

    // Get first vertices as representatives
    const fromVertex = fromVertices[0];
    const toVertex = toVertices[0];

    return this._lattice.detectPhaseShift(fromVertex, toVertex) ?? {
      from,
      to,
      crossAxisVertices: [],
      rotationPlane: 0,
      direction: 1
    };
  }

  private _calculateTension(weights: [number, number, number]): number {
    // Tension is highest when weights are evenly distributed
    // Tension is lowest when one weight dominates
    const maxWeight = Math.max(...weights);
    const entropy = -weights.reduce((sum, w) => {
      if (w > MATH_CONSTANTS.EPSILON) {
        return sum + w * Math.log(w);
      }
      return sum;
    }, 0);

    // Normalize: max entropy is ln(3) when all equal
    const maxEntropy = Math.log(3);
    const normalizedEntropy = entropy / maxEntropy;

    // Also factor in max weight (lower max weight = higher tension)
    const dominanceDeficit = 1 - maxWeight;

    return (normalizedEntropy + dominanceDeficit) / 2;
  }

  // =========================================================================
  // QUERY METHODS
  // =========================================================================

  /**
   * Check if current state is polytonal (significant weights in multiple axes).
   */
  isPolytonal(): boolean {
    const sortedWeights = [...this._state.weights].sort((a, b) => b - a);
    // Polytonal if second-largest weight is significant
    return sortedWeights[1] >= POLYTONAL_THRESHOLD;
  }

  /**
   * Get the secondary axis (second-highest weight).
   */
  getSecondaryAxis(): TrinityAxis | null {
    const weights = this._state.weights;
    const maxIdx = weights.indexOf(Math.max(...weights));

    // Find second max
    let secondMax = -1;
    let secondIdx = -1;
    for (let i = 0; i < 3; i++) {
      if (i !== maxIdx && weights[i] > secondMax) {
        secondMax = weights[i];
        secondIdx = i;
      }
    }

    if (secondMax < PHASE_SHIFT_THRESHOLD) {
      return null;
    }

    const axes: TrinityAxis[] = ['alpha', 'beta', 'gamma'];
    return axes[secondIdx];
  }

  /**
   * Predict the most likely next phase shift.
   */
  predictNextPhaseShift(): { axis: TrinityAxis; probability: number } | null {
    if (this._state.tension < 0.2) {
      return null; // Low tension, unlikely to shift
    }

    const secondary = this.getSecondaryAxis();
    if (!secondary) {
      return null;
    }

    // Probability based on tension and secondary weight
    const secondaryWeight = this._state.weights[
      secondary === 'alpha' ? 0 : secondary === 'beta' ? 1 : 2
    ];

    const probability = this._state.tension * secondaryWeight;

    return { axis: secondary, probability };
  }

  /**
   * Get recommended pivot vertices for transitioning to a target axis.
   */
  getTransitionPivots(targetAxis: TrinityAxis): number[] {
    if (targetAxis === this._state.activeAxis) {
      return [];
    }

    // Get vertices that bridge the current and target axes
    const currentVertices = this._lattice.getAxisVertices(this._state.activeAxis);
    const pivots: number[] = [];

    for (const vertexId of currentVertices) {
      const crossAxisNeighbors = this._lattice.getCrossAxisNeighbors(vertexId);

      for (const neighborId of crossAxisNeighbors) {
        const neighborAxis = this._lattice.getVertexAxis(neighborId);
        if (neighborAxis === targetAxis) {
          pivots.push(neighborId);
        }
      }
    }

    return [...new Set(pivots)]; // Remove duplicates
  }

  // =========================================================================
  // STATE MANIPULATION
  // =========================================================================

  /**
   * Force a transition to a specific axis.
   */
  forceAxisTransition(targetAxis: TrinityAxis): void {
    if (targetAxis === this._state.activeAxis) {
      return;
    }

    const phaseShift = this._startPhaseShift(this._state.activeAxis, targetAxis);

    this._state = {
      activeAxis: targetAxis,
      weights: [
        targetAxis === 'alpha' ? 1 : 0,
        targetAxis === 'beta' ? 1 : 0,
        targetAxis === 'gamma' ? 1 : 0
      ],
      tension: 0,
      phaseProgress: 0
    };

    this._phaseShiftInProgress = false;
    this._phaseShiftInfo = null;

    this._emitEvent(TelemetryEventType.PHASE_SHIFT_COMPLETE, {
      from: phaseShift.from,
      to: phaseShift.to,
      forced: true
    });
  }

  /**
   * Reset to initial state.
   */
  reset(): void {
    this._state = this._createInitialState();
    this._previousVertex = 0;
    this._phaseShiftInProgress = false;
    this._phaseShiftInfo = null;
    this._updateCount = 0;
  }

  // =========================================================================
  // TELEMETRY
  // =========================================================================

  subscribe(callback: TelemetrySubscriber): () => void {
    this._subscribers.add(callback);
    return () => this._subscribers.delete(callback);
  }

  unsubscribe(callback: TelemetrySubscriber): void {
    this._subscribers.delete(callback);
  }

  private _emitEvent(eventType: TelemetryEventType, payload: Record<string, unknown>): void {
    const event: TelemetryEvent = {
      timestamp: Date.now(),
      eventType,
      payload: {
        ...payload,
        updateCount: this._updateCount
      }
    };

    for (const subscriber of this._subscribers) {
      try {
        subscriber(event);
      } catch (error) {
        console.error('Trinity telemetry subscriber error:', error);
      }
    }
  }

  // =========================================================================
  // STATISTICS
  // =========================================================================

  getStats(): Record<string, unknown> {
    return {
      updateCount: this._updateCount,
      activeAxis: this._state.activeAxis,
      weights: [...this._state.weights],
      tension: this._state.tension,
      phaseProgress: this._state.phaseProgress,
      isPolytonal: this.isPolytonal(),
      phaseShiftInProgress: this._phaseShiftInProgress,
      previousVertex: this._previousVertex,
      subscriberCount: this._subscribers.size
    };
  }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/**
 * Create a new TrinityEngine with default lattice.
 */
export function createTrinityEngine(): TrinityEngine {
  return new TrinityEngine();
}

/**
 * Create a TrinityEngine starting at a specific position.
 */
export function createTrinityEngineAt(position: Vector4D): TrinityEngine {
  const engine = new TrinityEngine();
  engine.updatePosition(position);
  return engine;
}

/**
 * Create a TrinityEngine starting at a specific axis.
 */
export function createTrinityEngineWithAxis(axis: TrinityAxis): TrinityEngine {
  const engine = new TrinityEngine();
  engine.forceAxisTransition(axis);
  return engine;
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  PHASE_SHIFT_THRESHOLD,
  POLYTONAL_THRESHOLD,
  TENSION_DECAY_RATE
};
