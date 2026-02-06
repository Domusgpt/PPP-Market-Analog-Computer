/**
 * HemocPythonBridge.ts - The HEMOC-SGF ↔ PPP Adapter Bridge
 *
 * Translates Python physics engine telemetry (HEMOC-SGF) into the PPP
 * Telemetry Signal/Manifold schema so the frontend UI can render
 * real-time Moiré interference, kirigami deformation, and reservoir
 * state from the backend engine.
 *
 * Communication: WebSocket (JSON payloads)
 *
 * Data flow:
 *   1. HEMOC-SGF (Python) calculates a frame of Moiré interference
 *   2. Python telemetry extracts metrics: petal_rotation, lattice_stress, reservoir_entropy
 *   3. This Bridge receives the JSON packet via WebSocket
 *   4. Bridge normalizes exotic physics outputs (quaternion rotations, etc.)
 *      into standard PPP Signals (0.0–1.0)
 *   5. Bridge routes data to the specific Manifold/Signal channel in the PPP UI
 *
 * Source repos bridged:
 *   - Backend:  HEMOC-SGF (hemoc-stain-glass-src) → Physics Core
 *   - Frontend: PPP Market Analog Computer → Telemetry & UI Spine
 */

import type { PPPAdapter } from '../contracts/AdapterContracts';
import type { RawApiTick } from '../fusion/StereoscopicFeed';

// ============================================================================
// HEMOC-SGF Payload Types (Python → JSON)
// ============================================================================

/**
 * Raw telemetry payload emitted by the HEMOC-SGF Python engine per frame.
 * Maps to the output of backend/engine/telemetry/metrics.py
 */
export interface HemocPhysicsPayload {
  /** Frame sequence number from the engine */
  frame_id: number;

  /** Unix timestamp (ms) when the frame was computed */
  timestamp: number;

  /** Moiré interference metrics */
  moire: {
    /** Dominant moiré period (micrometers) */
    period: number;
    /** Fringe contrast (0.0–1.0) */
    contrast: number;
    /** Dominant spatial frequency (cycles/mm) */
    dominant_frequency: number;
    /** Intensity field statistics */
    mean_intensity: number;
    max_intensity: number;
    min_intensity: number;
  };

  /** Kirigami lattice state */
  kirigami: {
    /** Per-petal rotation angles (radians), one per petal in the trilatic flower */
    petal_rotations: number[];
    /** Lattice stress tensor (flattened 3×3) */
    lattice_stress: number[];
    /** Cell states: fraction in each tristable mode [flat, half, full] */
    cell_distribution: [number, number, number];
    /** Current operating angle (degrees) from commensurate set */
    operating_angle: number;
  };

  /** Reservoir computing state */
  reservoir: {
    /** Shannon entropy of the reservoir state (bits) */
    entropy: number;
    /** Lyapunov exponent estimate (edge-of-chaos metric) */
    lyapunov: number;
    /** Reservoir memory capacity (0.0–1.0) */
    memory_capacity: number;
    /** Fading memory kernel values */
    kernel_weights: number[];
  };

  /** Talbot resonator state */
  talbot: {
    /** Current gap mode: 'integer' | 'half_integer' */
    gap_mode: 'integer' | 'half_integer';
    /** Logic polarity: 'positive' (AND/OR) | 'negative' (NAND/XOR) */
    logic_polarity: 'positive' | 'negative';
    /** Talbot gap distance (micrometers) */
    gap_distance: number;
  };

  /** Tripole actuator state (per layer) */
  actuators: {
    tip: number;
    tilt: number;
    piston: number;
  }[];

  /** Optional: feature vector for downstream ML */
  feature_vector?: number[];
}

// ============================================================================
// Bridge Configuration
// ============================================================================

export interface HemocPythonBridgeConfig {
  /** WebSocket URL for the HEMOC-SGF engine (default: ws://localhost:8765) */
  wsUrl?: string;

  /** Tick callback for PPPAdapter integration */
  onTick?: (tick: RawApiTick) => void;

  /** Source identifier for metrics tracking */
  source?: string;

  /** Reconnect on disconnect (default: true) */
  autoReconnect?: boolean;

  /** Max reconnect attempts (default: 10) */
  maxReconnectAttempts?: number;

  /** Reconnect backoff base (ms, default: 1000) */
  reconnectBackoff?: number;

  /** Optional callback for raw physics payloads (for visualization) */
  onPhysicsFrame?: (payload: HemocPhysicsPayload) => void;
}

// ============================================================================
// Normalized Telemetry Channels
// ============================================================================

/**
 * Maps HEMOC physics metrics to PPP channel indices.
 * These indices correspond to RawApiTick.channels[]
 *
 * Channel layout mirrors the PPP telemetry schema:
 *   [0]  moire_contrast         - Fringe visibility (0–1)
 *   [1]  moire_frequency        - Dominant spatial frequency (normalized)
 *   [2]  lattice_stress         - Frobenius norm of stress tensor (normalized)
 *   [3]  reservoir_entropy      - Shannon entropy (normalized to max bits)
 *   [4]  reservoir_lyapunov     - Lyapunov exponent (clamped 0–1)
 *   [5]  talbot_gap_normalized  - Gap distance (normalized)
 *   [6]  petal_rotation_mean    - Mean petal rotation angle (normalized to 2π)
 *   [7]  cell_flat_fraction     - Fraction of cells in flat state
 *   [8]  cell_half_fraction     - Fraction of cells in half-fold state
 *   [9]  cell_full_fraction     - Fraction of cells in full-fold state
 *   [10] memory_capacity        - Reservoir memory (0–1)
 *   [11] logic_polarity         - 0 = positive, 1 = negative
 */
export const HEMOC_CHANNEL_MAP = {
  MOIRE_CONTRAST: 0,
  MOIRE_FREQUENCY: 1,
  LATTICE_STRESS: 2,
  RESERVOIR_ENTROPY: 3,
  RESERVOIR_LYAPUNOV: 4,
  TALBOT_GAP: 5,
  PETAL_ROTATION_MEAN: 6,
  CELL_FLAT: 7,
  CELL_HALF: 8,
  CELL_FULL: 9,
  MEMORY_CAPACITY: 10,
  LOGIC_POLARITY: 11,
} as const;

// ============================================================================
// HemocPythonBridge
// ============================================================================

export class HemocPythonBridge implements PPPAdapter {
  private ws: WebSocket | null = null;
  private onTickCallback?: (tick: RawApiTick) => void;
  private onPhysicsFrameCallback?: (payload: HemocPhysicsPayload) => void;

  private readonly wsUrl: string;
  private readonly source: string;
  private readonly autoReconnect: boolean;
  private readonly maxReconnectAttempts: number;
  private readonly reconnectBackoff: number;

  // Metrics
  private ticks = 0;
  private lastTimestamp: number | null = null;
  private reconnectAttempts = 0;
  private connected = false;
  private drops = 0;
  private parseErrors = 0;

  // Normalization constants (derived from physics constraints)
  private static readonly MAX_MOIRE_FREQUENCY = 100; // cycles/mm
  private static readonly MAX_STRESS_FROBENIUS = 10;  // normalized stress units
  private static readonly MAX_ENTROPY_BITS = 8;       // log2(256) for reservoir states
  private static readonly TWO_PI = 2 * Math.PI;

  constructor(config: HemocPythonBridgeConfig = {}) {
    this.wsUrl = config.wsUrl ?? 'ws://localhost:8765';
    this.onTickCallback = config.onTick;
    this.onPhysicsFrameCallback = config.onPhysicsFrame;
    this.source = config.source ?? 'hemoc-sgf';
    this.autoReconnect = config.autoReconnect ?? true;
    this.maxReconnectAttempts = config.maxReconnectAttempts ?? 10;
    this.reconnectBackoff = config.reconnectBackoff ?? 1000;
  }

  // --------------------------------------------------------------------------
  // PPPAdapter interface
  // --------------------------------------------------------------------------

  connect(): void {
    this.reconnectAttempts = 0;
    this.openWebSocket();
  }

  disconnect(): void {
    this.autoReconnect && (this.reconnectAttempts = this.maxReconnectAttempts);
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
  }

  onTick(callback: (tick: RawApiTick) => void): () => void {
    this.onTickCallback = callback;
    return () => {
      if (this.onTickCallback === callback) {
        this.onTickCallback = undefined;
      }
    };
  }

  metrics(): Record<string, number | string | boolean> {
    return {
      adapter: 'HemocPythonBridge',
      source: this.source,
      connected: this.connected,
      ticks: this.ticks,
      drops: this.drops,
      parseErrors: this.parseErrors,
      reconnectAttempts: this.reconnectAttempts,
      lastTimestamp: this.lastTimestamp ?? 'n/a',
    };
  }

  resetMetrics(): void {
    this.ticks = 0;
    this.drops = 0;
    this.parseErrors = 0;
    this.lastTimestamp = null;
  }

  // --------------------------------------------------------------------------
  // WebSocket Management
  // --------------------------------------------------------------------------

  private openWebSocket(): void {
    try {
      this.ws = new WebSocket(this.wsUrl);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.connected = true;
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event: MessageEvent) => {
      this.handleMessage(event.data);
    };

    this.ws.onerror = () => {
      this.drops += 1;
    };

    this.ws.onclose = () => {
      this.connected = false;
      if (this.autoReconnect) {
        this.scheduleReconnect();
      }
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    const delay = this.reconnectBackoff * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts += 1;
    setTimeout(() => this.openWebSocket(), delay);
  }

  // --------------------------------------------------------------------------
  // Message Handling & Normalization
  // --------------------------------------------------------------------------

  private handleMessage(data: string | ArrayBuffer): void {
    let payload: HemocPhysicsPayload;
    try {
      const text = typeof data === 'string' ? data : new TextDecoder().decode(data);
      payload = JSON.parse(text) as HemocPhysicsPayload;
    } catch {
      this.parseErrors += 1;
      return;
    }

    // Fire raw physics callback for visualization layers
    this.onPhysicsFrameCallback?.(payload);

    // Normalize to PPP RawApiTick
    const tick = this.normalizeToTick(payload);
    this.ticks += 1;
    this.lastTimestamp = tick.timestamp ?? null;
    this.onTickCallback?.(tick);
  }

  /**
   * Core normalization: HEMOC-SGF physics payload → PPP RawApiTick
   *
   * Maps exotic physics quantities to the [0, 1] normalized channel
   * layout consumed by the PPP telemetry spine.
   */
  private normalizeToTick(payload: HemocPhysicsPayload): RawApiTick {
    const { moire, kirigami, reservoir, talbot } = payload;

    // Normalize moiré frequency to 0–1 range
    const normalizedFrequency = clamp(
      moire.dominant_frequency / HemocPythonBridge.MAX_MOIRE_FREQUENCY,
      0, 1
    );

    // Compute Frobenius norm of stress tensor, normalize
    const stressFrobenius = frobenius(kirigami.lattice_stress);
    const normalizedStress = clamp(
      stressFrobenius / HemocPythonBridge.MAX_STRESS_FROBENIUS,
      0, 1
    );

    // Normalize entropy
    const normalizedEntropy = clamp(
      reservoir.entropy / HemocPythonBridge.MAX_ENTROPY_BITS,
      0, 1
    );

    // Normalize Lyapunov (positive = chaotic, clamp to [0,1])
    const normalizedLyapunov = clamp(reservoir.lyapunov, 0, 1);

    // Normalize gap distance (assume max ~100 μm)
    const normalizedGap = clamp(talbot.gap_distance / 100, 0, 1);

    // Mean petal rotation normalized to [0, 1] over full rotation
    const meanRotation = kirigami.petal_rotations.length > 0
      ? kirigami.petal_rotations.reduce((s, r) => s + r, 0) / kirigami.petal_rotations.length
      : 0;
    const normalizedRotation = (meanRotation % HemocPythonBridge.TWO_PI) / HemocPythonBridge.TWO_PI;

    // Build the channel array matching HEMOC_CHANNEL_MAP
    const channels: number[] = [
      moire.contrast,                            // [0] moire_contrast (already 0–1)
      normalizedFrequency,                       // [1] moire_frequency
      normalizedStress,                          // [2] lattice_stress
      normalizedEntropy,                         // [3] reservoir_entropy
      normalizedLyapunov,                        // [4] reservoir_lyapunov
      normalizedGap,                             // [5] talbot_gap
      normalizedRotation,                        // [6] petal_rotation_mean
      kirigami.cell_distribution[0],             // [7] cell_flat
      kirigami.cell_distribution[1],             // [8] cell_half
      kirigami.cell_distribution[2],             // [9] cell_full
      reservoir.memory_capacity,                 // [10] memory_capacity (already 0–1)
      talbot.logic_polarity === 'positive' ? 0 : 1, // [11] logic_polarity
    ];

    // Map physics to price-like quantities for the PPP Stereoscopic Feed:
    //   price  = moiré contrast (primary "value" signal)
    //   volume = reservoir entropy (activity measure)
    //   bid    = lattice stress (lower bound / tension)
    //   ask    = petal rotation mean (upper bound / angular state)
    return {
      symbol: this.source,
      price: moire.contrast,
      volume: normalizedEntropy,
      bid: normalizedStress,
      ask: normalizedRotation,
      timestamp: payload.timestamp,
      sequence: payload.frame_id,
      channels,
    };
  }

  // --------------------------------------------------------------------------
  // Manual Ingestion (for testing / non-WebSocket scenarios)
  // --------------------------------------------------------------------------

  /**
   * Manually ingest a physics payload (useful for file playback or testing).
   */
  ingestPayload(payload: HemocPhysicsPayload): RawApiTick {
    this.onPhysicsFrameCallback?.(payload);
    const tick = this.normalizeToTick(payload);
    this.ticks += 1;
    this.lastTimestamp = tick.timestamp ?? null;
    this.onTickCallback?.(tick);
    return tick;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Frobenius norm of a flattened matrix */
function frobenius(values: number[]): number {
  let sum = 0;
  for (const v of values) {
    sum += v * v;
  }
  return Math.sqrt(sum);
}
