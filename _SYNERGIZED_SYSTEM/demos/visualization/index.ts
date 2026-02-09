/**
 * HEMAC Visualization Module
 *
 * @package hemac
 * @module visualization
 *
 * Provides visualization capabilities for the HEMAC system:
 * - E8 lattice rendering with Moxness 8D→4D projection
 * - Moiré interference patterns for analog visual computation
 * - Triadic color mapping based on Trinity decomposition
 * - Stereographic 4D→3D projection
 *
 * Visual Analog Computation:
 * The visualization is not just display - it's computation.
 * Moiré patterns emerge from the interference of periodic structures,
 * encoding high-dimensional information in a form the human visual
 * system can process intuitively.
 *
 * Based on: e8codec-hyperrendering, OptoOrthoKinetiscope
 */

// Re-export E8 Renderer
export {
  E8Renderer,
  type E8RendererConfig,
  DEFAULT_E8_RENDERER_CONFIG,
  MOXNESS_MATRIX,
  TRIAD_COLORS,
  generateE8Roots,
  generateE8LatticeData,
  generateLattice,
  projectMoxness,
  assignTriadGroup,
  stereographicProject4Dto3D
} from './E8Renderer.js';

// Re-export Moiré Overlay
export {
  MoireOverlay,
  type MoireConfig,
  type MoireRenderParams,
  DEFAULT_MOIRE_CONFIG,
  generateStandingWave,
  combinePatterns
} from './MoireOverlay.js';

// =============================================================================
// INTEGRATED VISUALIZATION STAGE
// =============================================================================

import { E8Renderer, E8RendererConfig } from './E8Renderer.js';
import { MoireOverlay, MoireConfig } from './MoireOverlay.js';

/** Combined visualization configuration */
export interface VisualizationConfig {
  e8: Partial<E8RendererConfig>;
  moire: Partial<MoireConfig>;
  showE8: boolean;
  showMoire: boolean;
  autoRotate: boolean;
  rotationSpeeds: {
    xy: number;
    xz: number;
    xw: number;
    yz: number;
    yw: number;
    zw: number;
  };
}

/** Default visualization configuration */
export const DEFAULT_VISUALIZATION_CONFIG: VisualizationConfig = {
  e8: {},
  moire: {},
  showE8: true,
  showMoire: true,
  autoRotate: true,
  rotationSpeeds: {
    xy: 0.001,
    xz: 0.0007,
    xw: 0.0005,
    yz: 0.0003,
    yw: 0.0002,
    zw: 0.0001
  }
};

/**
 * HolographicStage - Unified visualization manager.
 *
 * Combines E8 lattice rendering with Moiré overlays to create
 * the "holographic" visual computation interface.
 *
 * Usage:
 * ```typescript
 * const stage = new HolographicStage(canvas);
 *
 * // Connect to physics engine
 * engine.subscribe((event) => {
 *   if (event.eventType === 'STATE_UPDATE') {
 *     stage.setPhysicsState({
 *       tension: event.payload.coherence,
 *       phase: event.payload.angularMagnitude,
 *       coherence: event.payload.coherence
 *     });
 *   }
 * });
 *
 * // Animation loop
 * function animate() {
 *   stage.update(1/60);
 *   stage.render();
 *   requestAnimationFrame(animate);
 * }
 * animate();
 * ```
 */
export class HolographicStage {
  private config: VisualizationConfig;
  private e8Renderer: E8Renderer;
  private moireOverlay: MoireOverlay;
  private canvas: HTMLCanvasElement | null;
  private time: number;

  constructor(canvas?: HTMLCanvasElement, config: Partial<VisualizationConfig> = {}) {
    this.config = { ...DEFAULT_VISUALIZATION_CONFIG, ...config };
    this.canvas = canvas || null;
    this.e8Renderer = new E8Renderer(this.config.e8);
    this.moireOverlay = new MoireOverlay(canvas, this.config.moire);
    this.time = 0;
  }

  /**
   * Update visualization state.
   */
  update(deltaTime: number): void {
    this.time += deltaTime;

    // Auto-rotate E8 if enabled
    if (this.config.autoRotate && this.config.showE8) {
      const speeds = this.config.rotationSpeeds;
      this.e8Renderer.rotateInPlane('XY', speeds.xy);
      this.e8Renderer.rotateInPlane('XZ', speeds.xz);
      this.e8Renderer.rotateInPlane('XW', speeds.xw);
      this.e8Renderer.rotateInPlane('YZ', speeds.yz);
      this.e8Renderer.rotateInPlane('YW', speeds.yw);
      this.e8Renderer.rotateInPlane('ZW', speeds.zw);
    }

    // Update moiré animation
    if (this.config.showMoire) {
      this.moireOverlay.update(deltaTime);
    }
  }

  /**
   * Set state from physics engine.
   */
  setPhysicsState(state: { tension: number; phase: number; coherence: number }): void {
    this.moireOverlay.setFromPhysicsState(state);
  }

  /**
   * Get E8 vertex data for WebGL rendering.
   */
  getE8VertexBuffer(): Float32Array {
    return this.e8Renderer.getVertexBuffer();
  }

  /**
   * Get E8 edge data for WebGL rendering.
   */
  getE8EdgeBuffer(): Uint32Array {
    return this.e8Renderer.getEdgeBuffer();
  }

  /**
   * Get Moiré render parameters for shader.
   */
  getMoireParams(): ReturnType<MoireOverlay['getRenderParams']> {
    return this.moireOverlay.getRenderParams();
  }

  /**
   * Render Moiré to canvas (fallback mode).
   */
  renderMoire(): void {
    if (this.config.showMoire) {
      this.moireOverlay.renderFast();
    }
  }

  /**
   * Get diagnostics.
   */
  getDiagnostics(): {
    e8: ReturnType<E8Renderer['getDiagnostics']>;
    time: number;
  } {
    return {
      e8: this.e8Renderer.getDiagnostics(),
      time: this.time
    };
  }

  /**
   * Configure visualization.
   */
  configure(config: Partial<VisualizationConfig>): void {
    this.config = { ...this.config, ...config };

    if (config.e8) {
      this.e8Renderer.configure(config.e8);
    }
    if (config.moire) {
      this.moireOverlay.configure(config.moire);
    }
  }
}

/**
 * Create a visualization stage.
 */
export function createVisualization(
  canvas?: HTMLCanvasElement,
  config?: Partial<VisualizationConfig>
): HolographicStage {
  return new HolographicStage(canvas, config);
}
