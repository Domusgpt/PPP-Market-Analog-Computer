/**
 * Chronomorphic Polytopal Engine - Main Unified Engine
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 *
 * The ChronomorphicEngine unifies all subsystems:
 * - CausalReasoningEngine (force/torque physics)
 * - TrinityEngine (phase shift detection)
 * - MusicGeometryDomain (musical mapping)
 * - HDCEncoder (hyperdimensional computing)
 * - HarmonicTopologist (TDA/persistent homology)
 * - Lattice24 (24-cell topology)
 * - Cell600 (600-cell container)
 *
 * This is the main entry point for applications using the CPE.
 *
 * Theoretical Foundation:
 * - P1: Information is ontologically primary
 * - P2: Dual entropic drives (EPO-D, EPO-I)
 * - P3: Universe is self-contained
 * - D1: Convexity required for equilibrium
 * - D3: 24-cell is maximal integration structure
 * - D4: Visual rendering constitutes computation
 */

import {
  Vector4D,
  Bivector4D,
  Rotor,
  TrinityAxis,
  TrinityState,
  EngineState,
  EngineConfig,
  Force,
  UpdateResult,
  ConvexityResult,
  BettiProfile,
  TopologicalVoid,
  TelemetryEvent,
  TelemetryEventType,
  TelemetrySubscriber,
  GeometricChord,
  DEFAULT_ENGINE_CONFIG,
  MATH_CONSTANTS
} from '../types/index.js';

import { Lattice24, getDefaultLattice } from '../topology/Lattice24.js';
import { Cell600, getDefaultCell600 } from '../topology/Cell600.js';
import { CausalReasoningEngine } from './CausalReasoningEngine.js';
import { TrinityEngine } from './TrinityEngine.js';
import { MusicGeometryDomain } from '../domains/MusicGeometryDomain.js';
import { HDCEncoder, Hypervector } from '../encoding/HDCEncoder.js';
import { HarmonicTopologist } from '../tda/PersistentHomology.js';
import { magnitude, normalize } from '../math/GeometricAlgebra.js';

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * ChronomorphicEngine configuration.
 */
export interface ChronomorphicConfig {
  engine: Partial<EngineConfig>;
  hdc: {
    dimensions: number;
    bipolar: boolean;
  };
  tda: {
    maxDimension: number;
    maxRadius: number;
  };
  enableMusic: boolean;
  enableTDA: boolean;
  enableHDC: boolean;
  enableCell600: boolean;
}

const DEFAULT_CHRONOMORPHIC_CONFIG: ChronomorphicConfig = {
  engine: DEFAULT_ENGINE_CONFIG,
  hdc: {
    dimensions: 4096,
    bipolar: true
  },
  tda: {
    maxDimension: 2,
    maxRadius: 3.0
  },
  enableMusic: true,
  enableTDA: true,
  enableHDC: true,
  enableCell600: true
};

// =============================================================================
// UNIFIED STATE
// =============================================================================

/**
 * Complete state of the ChronomorphicEngine.
 */
export interface ChronomorphicState {
  // Core physics state
  position: Vector4D;
  orientation: Rotor;
  velocity: Vector4D;
  angularVelocity: Bivector4D;

  // Trinity state
  trinity: TrinityState;

  // Topological state
  coherence: number;
  nearestVertex: number;
  activeVertices: number[];

  // TDA state
  betti?: BettiProfile;
  voids?: TopologicalVoid[];
  ambiguity: number;

  // Musical state
  currentKey?: string;
  keyWeights?: Map<string, number>;

  // Metadata
  timestamp: number;
  updateCount: number;
}

// =============================================================================
// CHRONOMORPHIC ENGINE CLASS
// =============================================================================

/**
 * ChronomorphicEngine is the main unified engine.
 *
 * Usage:
 * ```typescript
 * const engine = new ChronomorphicEngine();
 *
 * // Apply force and update
 * engine.applyForce([1, 0, 0, 0], 0.5);
 * const state = engine.update(1/60);
 *
 * // Get musical interpretation
 * const key = engine.getCurrentKey();
 * const chord = engine.encodeChord([0, 4, 7]);
 *
 * // Get topological analysis
 * const topology = engine.analyzeTopology();
 * ```
 */
export class ChronomorphicEngine {
  // Subsystems
  private _causalEngine: CausalReasoningEngine;
  private _trinityEngine: TrinityEngine;
  private _musicDomain: MusicGeometryDomain | null;
  private _hdcEncoder: HDCEncoder | null;
  private _topologist: HarmonicTopologist | null;
  private _lattice: Lattice24;
  private _cell600: Cell600 | null;

  // Configuration
  private _config: ChronomorphicConfig;

  // State tracking
  private _updateCount: number;
  private _lastTDAUpdate: number;
  private _cachedBetti: BettiProfile | null;
  private _cachedVoids: TopologicalVoid[];

  // Telemetry
  private _subscribers: Set<TelemetrySubscriber>;

  constructor(config: Partial<ChronomorphicConfig> = {}) {
    this._config = { ...DEFAULT_CHRONOMORPHIC_CONFIG, ...config };

    // Initialize core components
    this._lattice = getDefaultLattice();
    this._causalEngine = new CausalReasoningEngine(this._config.engine, this._lattice);
    this._trinityEngine = new TrinityEngine(this._lattice);

    // Initialize optional components
    this._musicDomain = this._config.enableMusic
      ? new MusicGeometryDomain(this._lattice)
      : null;

    this._hdcEncoder = this._config.enableHDC
      ? new HDCEncoder(this._config.hdc, this._lattice)
      : null;

    this._topologist = this._config.enableTDA
      ? new HarmonicTopologist(this._lattice, this._config.tda.maxDimension, this._config.tda.maxRadius)
      : null;

    this._cell600 = this._config.enableCell600
      ? getDefaultCell600()
      : null;

    // Initialize state tracking
    this._updateCount = 0;
    this._lastTDAUpdate = 0;
    this._cachedBetti = null;
    this._cachedVoids = [];

    // Initialize telemetry
    this._subscribers = new Set();

    // Wire up internal telemetry
    this._causalEngine.subscribe(event => this._handleSubsystemEvent('causal', event));
    this._trinityEngine.subscribe(event => this._handleSubsystemEvent('trinity', event));
  }

  // =========================================================================
  // ACCESSORS
  // =========================================================================

  get position(): Vector4D { return this._causalEngine.state.position; }
  get orientation(): Rotor { return this._causalEngine.state.orientation; }
  get velocity(): Vector4D { return this._causalEngine.state.velocity; }
  get trinityState(): TrinityState { return this._trinityEngine.state; }
  get lattice(): Lattice24 { return this._lattice; }
  get cell600(): Cell600 | null { return this._cell600; }
  get updateCount(): number { return this._updateCount; }

  // =========================================================================
  // CORE UPDATE
  // =========================================================================

  /**
   * Main update loop.
   */
  update(deltaTime: number): ChronomorphicState {
    this._updateCount++;

    // Update causal physics
    const physicsResult = this._causalEngine.update(deltaTime);

    // Update Trinity state
    const trinityResult = this._trinityEngine.updatePosition(
      physicsResult.state.position
    );

    // Update TDA periodically (expensive)
    if (this._topologist && this._updateCount - this._lastTDAUpdate > 30) {
      this._updateTDA();
      this._lastTDAUpdate = this._updateCount;
    }

    // Build unified state
    const state = this._buildState(physicsResult, trinityResult);

    // Emit update event
    this._emitEvent(TelemetryEventType.STATE_UPDATE, {
      position: [...state.position],
      coherence: state.coherence,
      trinityAxis: state.trinity.activeAxis,
      ambiguity: state.ambiguity
    });

    return state;
  }

  /**
   * Build the unified state object.
   */
  private _buildState(
    physicsResult: UpdateResult,
    trinityResult: { state: TrinityState }
  ): ChronomorphicState {
    const position = physicsResult.state.position;

    // Get musical state if enabled
    let currentKey: string | undefined;
    let keyWeights: Map<string, number> | undefined;

    if (this._musicDomain) {
      currentKey = this._musicDomain.positionToKey(position);
      keyWeights = this._musicDomain.positionToKeyWeights(position);
    }

    return {
      // Core physics
      position,
      orientation: physicsResult.state.orientation,
      velocity: physicsResult.state.velocity,
      angularVelocity: physicsResult.state.angularVelocity,

      // Trinity
      trinity: trinityResult.state,

      // Topology
      coherence: physicsResult.convexity.coherence,
      nearestVertex: physicsResult.convexity.nearestVertex,
      activeVertices: physicsResult.convexity.activeVertices,

      // TDA
      betti: this._cachedBetti ?? undefined,
      voids: this._cachedVoids.length > 0 ? this._cachedVoids : undefined,
      ambiguity: this._cachedBetti?.beta2 ?? 0,

      // Music
      currentKey,
      keyWeights,

      // Metadata
      timestamp: Date.now(),
      updateCount: this._updateCount
    };
  }

  /**
   * Update TDA analysis.
   */
  private _updateTDA(): void {
    if (!this._topologist) return;

    // Get active vertex positions
    const activeVertices = this._lattice.findKNearest(this.position, 8);
    const points = activeVertices.map(id =>
      this._lattice.getVertex(id)?.coordinates ?? [0, 0, 0, 0] as Vector4D
    );

    // Compute persistent homology
    this._cachedBetti = this._topologist.analyze(points);
    this._cachedVoids = this._topologist.detectVoids(points);
  }

  // =========================================================================
  // FORCE APPLICATION
  // =========================================================================

  /**
   * Apply a linear force.
   */
  applyForce(direction: Vector4D, magnitude: number, source: string = 'external'): void {
    this._causalEngine.applyLinearForce(direction, magnitude, source);
  }

  /**
   * Apply a rotational force.
   */
  applyRotation(bivector: Bivector4D, magnitude: number, source: string = 'external'): void {
    this._causalEngine.applyRotationalForce(bivector, magnitude, source);
  }

  /**
   * Apply a combined force.
   */
  applyFullForce(force: Force): void {
    this._causalEngine.applyForce(force);
  }

  /**
   * Navigate towards a position.
   */
  navigateTo(target: Vector4D, strength: number = 1): void {
    this._causalEngine.navigateTowards(target, strength);
  }

  /**
   * Navigate to a vertex.
   */
  navigateToVertex(vertexId: number, strength: number = 1): void {
    this._causalEngine.navigateToVertex(vertexId, strength);
  }

  /**
   * Navigate to a key (musical).
   */
  navigateToKey(keyName: string, strength: number = 1): void {
    if (!this._musicDomain) return;

    const vertexId = this._musicDomain.keyToVertex(keyName);
    if (vertexId !== undefined) {
      this.navigateToVertex(vertexId, strength);
    }
  }

  // =========================================================================
  // STATE MANIPULATION
  // =========================================================================

  /**
   * Set position directly.
   */
  setPosition(position: Vector4D): void {
    this._causalEngine.setPosition(position);
    this._trinityEngine.updatePosition(position);
  }

  /**
   * Reset to initial state.
   */
  reset(): void {
    this._causalEngine.reset();
    this._trinityEngine.reset();
    this._updateCount = 0;
    this._lastTDAUpdate = 0;
    this._cachedBetti = null;
    this._cachedVoids = [];

    this._emitEvent(TelemetryEventType.ENGINE_RESET, {});
  }

  /**
   * Reset to a specific vertex.
   */
  resetToVertex(vertexId: number): void {
    this._causalEngine.resetToVertex(vertexId);
    this._trinityEngine.updateVertex(vertexId);
  }

  /**
   * Reset to a specific key.
   */
  resetToKey(keyName: string): void {
    if (!this._musicDomain) return;

    const vertexId = this._musicDomain.keyToVertex(keyName);
    if (vertexId !== undefined) {
      this.resetToVertex(vertexId);
    }
  }

  // =========================================================================
  // TRINITY OPERATIONS
  // =========================================================================

  /**
   * Get current dominant Trinity axis.
   */
  getDominantAxis(): TrinityAxis {
    return this._trinityEngine.state.activeAxis;
  }

  /**
   * Get Trinity weights.
   */
  getTrinityWeights(): [number, number, number] {
    return [...this._trinityEngine.state.weights] as [number, number, number];
  }

  /**
   * Check if polytonal.
   */
  isPolytonal(): boolean {
    return this._trinityEngine.isPolytonal();
  }

  /**
   * Get phase shift info if in progress.
   */
  getPhaseShiftInfo(): { from: TrinityAxis; to: TrinityAxis } | null {
    const info = this._trinityEngine.currentPhaseShift;
    if (info) {
      return { from: info.from, to: info.to };
    }
    return null;
  }

  /**
   * Force transition to a Trinity axis.
   */
  forceAxisTransition(axis: TrinityAxis): void {
    this._trinityEngine.forceAxisTransition(axis);
  }

  // =========================================================================
  // MUSIC OPERATIONS
  // =========================================================================

  /**
   * Get current key.
   */
  getCurrentKey(): string | null {
    if (!this._musicDomain) return null;
    return this._musicDomain.positionToKey(this.position);
  }

  /**
   * Encode a chord.
   */
  encodeChord(pitchClasses: number[]): GeometricChord | null {
    if (!this._musicDomain) return null;
    return this._musicDomain.encodeChord(pitchClasses);
  }

  /**
   * Detect key from pitch classes.
   */
  detectKey(pitchClasses: number[]): { key: string; confidence: number }[] {
    if (!this._musicDomain) return [];
    return this._musicDomain.detectKey(pitchClasses);
  }

  /**
   * Apply Neo-Riemannian transformation.
   */
  applyTransformation(transformation: string): string | null {
    if (!this._musicDomain) return null;
    const currentKey = this.getCurrentKey();
    if (!currentKey) return null;
    return this._musicDomain.applyTransformations(currentKey, transformation);
  }

  /**
   * Get circle of fifths distance between current key and target.
   */
  getKeyDistance(targetKey: string): number {
    if (!this._musicDomain) return 0;
    const currentKey = this.getCurrentKey();
    if (!currentKey) return 0;
    return this._musicDomain.circleOfFifthsDistance(currentKey, targetKey);
  }

  // =========================================================================
  // HDC OPERATIONS
  // =========================================================================

  /**
   * Encode a symbol to hypervector.
   */
  encodeSymbol(symbol: string): Hypervector | null {
    if (!this._hdcEncoder) return null;
    return this._hdcEncoder.encode(symbol);
  }

  /**
   * Encode current position to hypervector.
   */
  encodeCurrentPosition(): Hypervector | null {
    if (!this._hdcEncoder) return null;
    return this._hdcEncoder.encodePosition(this.position);
  }

  /**
   * Find nearest vertex from hypervector.
   */
  hypervectorToVertex(hv: Hypervector): number | null {
    if (!this._hdcEncoder) return null;
    return this._hdcEncoder.findNearestVertex(hv).vertexId;
  }

  /**
   * Query HDC memory.
   */
  queryMemory(hv: Hypervector, topK: number = 5): { symbol: string; similarity: number }[] {
    if (!this._hdcEncoder) return [];
    return this._hdcEncoder.query(hv, topK);
  }

  // =========================================================================
  // TDA OPERATIONS
  // =========================================================================

  /**
   * Analyze topology of current state.
   */
  analyzeTopology(): BettiProfile | null {
    if (!this._topologist) return null;

    const activeVertices = this._lattice.findKNearest(this.position, 8);
    const points = activeVertices.map(id =>
      this._lattice.getVertex(id)?.coordinates ?? [0, 0, 0, 0] as Vector4D
    );

    return this._topologist.analyze(points);
  }

  /**
   * Get ghost vertices (void centers).
   */
  getGhostVertices(): number[] {
    if (!this._topologist) return [];

    const activeVertices = this._lattice.findKNearest(this.position, 8);
    const points = activeVertices.map(id =>
      this._lattice.getVertex(id)?.coordinates ?? [0, 0, 0, 0] as Vector4D
    );

    return this._topologist.getGhostVertices(points);
  }

  /**
   * Get ambiguity score.
   */
  getAmbiguity(): number {
    return this._cachedBetti?.beta2 ?? 0;
  }

  /**
   * Get cohesion score.
   */
  getCohesion(): number {
    if (!this._cachedBetti) return 1;
    return 1 / Math.max(1, this._cachedBetti.beta0);
  }

  // =========================================================================
  // 600-CELL OPERATIONS
  // =========================================================================

  /**
   * Find nearest 600-cell vertex.
   */
  findNearest600CellVertex(): number | null {
    if (!this._cell600) return null;
    return this._cell600.findNearest(this.position);
  }

  /**
   * Get inscribed 24-cells containing nearest vertex.
   */
  getContaining24Cells(): number[] {
    if (!this._cell600) return [];
    const nearest = this._cell600.findNearest(this.position);
    return this._cell600.getContaining24Cells(nearest);
  }

  /**
   * Project E8 vector to H4.
   */
  projectE8(e8Vector: number[]): { outer: Vector4D; inner: Vector4D } | null {
    if (!this._cell600) return null;
    return this._cell600.projectE8(e8Vector);
  }

  // =========================================================================
  // QUERIES
  // =========================================================================

  /**
   * Get coherence score.
   */
  getCoherence(): number {
    return this._causalEngine.getCoherence();
  }

  /**
   * Get nearest vertex.
   */
  getNearestVertex(): number {
    return this._causalEngine.getNearestVertex();
  }

  /**
   * Get current convexity result.
   */
  checkConvexity(): ConvexityResult & { trinityWeights: [number, number, number] } {
    return this._lattice.checkConvexity(this.position);
  }

  /**
   * Check if at rest.
   */
  isAtRest(): boolean {
    return this._causalEngine.isAtRest();
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

  private _handleSubsystemEvent(source: string, event: TelemetryEvent): void {
    // Forward with source tag
    this._emitEvent(event.eventType, {
      ...event.payload,
      source
    });
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
        console.error('ChronomorphicEngine telemetry error:', error);
      }
    }
  }

  // =========================================================================
  // STATISTICS
  // =========================================================================

  getStats(): Record<string, unknown> {
    return {
      updateCount: this._updateCount,
      position: [...this.position],
      velocity: magnitude(this.velocity),
      coherence: this.getCoherence(),
      trinityAxis: this.getDominantAxis(),
      trinityWeights: this.getTrinityWeights(),
      isPolytonal: this.isPolytonal(),
      currentKey: this.getCurrentKey(),
      ambiguity: this.getAmbiguity(),
      cohesion: this.getCohesion(),
      nearestVertex: this.getNearestVertex(),
      isAtRest: this.isAtRest(),
      subsystems: {
        musicEnabled: !!this._musicDomain,
        hdcEnabled: !!this._hdcEncoder,
        tdaEnabled: !!this._topologist,
        cell600Enabled: !!this._cell600
      }
    };
  }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

export function createChronomorphicEngine(
  config?: Partial<ChronomorphicConfig>
): ChronomorphicEngine {
  return new ChronomorphicEngine(config);
}

export function createMinimalEngine(): ChronomorphicEngine {
  return new ChronomorphicEngine({
    enableMusic: false,
    enableTDA: false,
    enableHDC: false,
    enableCell600: false
  });
}

export function createFullEngine(): ChronomorphicEngine {
  return new ChronomorphicEngine({
    enableMusic: true,
    enableTDA: true,
    enableHDC: true,
    enableCell600: true
  });
}
