/**
 * Moiré Pattern Overlay
 *
 * @package hemac
 * @module visualization/MoireOverlay
 *
 * Generates Moiré interference patterns as a visual analog computation layer.
 * The patterns encode information about the system state through visual interference.
 *
 * Theoretical Foundation:
 * Moiré patterns emerge from the interference of two periodic structures.
 * In HEMAC, we use them to visualize:
 * - Tension between Trinity axes (pattern frequency)
 * - Phase/coherence (pattern alignment)
 * - Market regime (pattern color/intensity)
 *
 * The patterns serve as a "holographic" display - encoding high-dimensional
 * information in a 2D visual form that the human visual system can process.
 *
 * Based on: OptoOrthoKinetiscope MoireEngine.ts
 */

// =============================================================================
// TYPES
// =============================================================================

/** Moiré pattern configuration */
export interface MoireConfig {
  frequency: number;      // Base pattern frequency (lines per unit)
  phase: number;          // Phase offset (radians)
  amplitude: number;      // Pattern intensity (0-1)
  direction: number;      // Pattern direction (radians)
  tension: number;        // Inter-pattern tension (affects beat frequency)
  enabled: boolean;
}

/** Rendering parameters passed to shader/canvas */
export interface MoireRenderParams {
  frequency1: number;
  frequency2: number;
  angle1: number;
  angle2: number;
  phase1: number;
  phase2: number;
  amplitude: number;
  blendMode: 'multiply' | 'screen' | 'difference';
  colorA: { r: number; g: number; b: number };
  colorB: { r: number; g: number; b: number };
}

/** Default moiré configuration */
export const DEFAULT_MOIRE_CONFIG: MoireConfig = {
  frequency: 50,
  phase: 0,
  amplitude: 0.5,
  direction: 0,
  tension: 0,
  enabled: true
};

// =============================================================================
// MOIRE PATTERN GENERATOR
// =============================================================================

/**
 * MoireOverlay - Generates and renders Moiré interference patterns.
 *
 * Usage with Canvas:
 * ```typescript
 * const moire = new MoireOverlay(canvas);
 * moire.configure({ frequency: 60, tension: 0.3 });
 * moire.render();
 *
 * // Animate
 * function animate() {
 *   moire.update(1/60);
 *   moire.render();
 *   requestAnimationFrame(animate);
 * }
 * ```
 *
 * Usage as data source (for WebGL shader):
 * ```typescript
 * const moire = new MoireOverlay();
 * const params = moire.getRenderParams();
 * shader.setUniforms(params);
 * ```
 */
export class MoireOverlay {
  private config: MoireConfig;
  private canvas: HTMLCanvasElement | null;
  private ctx: CanvasRenderingContext2D | null;
  private time: number;
  private animationSpeed: number;

  constructor(canvas?: HTMLCanvasElement, config: Partial<MoireConfig> = {}) {
    this.config = { ...DEFAULT_MOIRE_CONFIG, ...config };
    this.canvas = canvas || null;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.time = 0;
    this.animationSpeed = 1;
  }

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------

  /**
   * Update configuration.
   */
  configure(config: Partial<MoireConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Set animation speed.
   */
  setAnimationSpeed(speed: number): void {
    this.animationSpeed = speed;
  }

  /**
   * Enable/disable the overlay.
   */
  setEnabled(enabled: boolean): void {
    this.config.enabled = enabled;
  }

  // ---------------------------------------------------------------------------
  // Update
  // ---------------------------------------------------------------------------

  /**
   * Update internal time (for animation).
   */
  update(deltaTime: number): void {
    this.time += deltaTime * this.animationSpeed;
  }

  /**
   * Set state from physics engine output.
   */
  setFromPhysicsState(state: {
    tension: number;
    coherence: number;
    phase: number;
  }): void {
    this.config.tension = state.tension;
    this.config.amplitude = state.coherence;
    this.config.phase = state.phase;
  }

  // ---------------------------------------------------------------------------
  // Render Parameters
  // ---------------------------------------------------------------------------

  /**
   * Get render parameters for shader use.
   */
  getRenderParams(): MoireRenderParams {
    const baseFreq = this.config.frequency;
    const tensionOffset = this.config.tension * 10; // Tension affects beat frequency

    return {
      frequency1: baseFreq,
      frequency2: baseFreq + tensionOffset,
      angle1: this.config.direction,
      angle2: this.config.direction + Math.PI / 6 + this.config.tension * Math.PI / 12,
      phase1: this.config.phase + this.time,
      phase2: this.config.phase - this.time * 0.7,
      amplitude: this.config.amplitude,
      blendMode: 'multiply',
      colorA: this.tensionToColor(this.config.tension, 'A'),
      colorB: this.tensionToColor(this.config.tension, 'B')
    };
  }

  /**
   * Map tension to color.
   */
  private tensionToColor(
    tension: number,
    layer: 'A' | 'B'
  ): { r: number; g: number; b: number } {
    // Low tension = cool colors, high tension = warm colors
    const t = Math.max(0, Math.min(1, tension));

    if (layer === 'A') {
      return {
        r: Math.floor(100 + 155 * t),
        g: Math.floor(200 - 100 * t),
        b: Math.floor(255 - 155 * t)
      };
    } else {
      return {
        r: Math.floor(255 - 100 * t),
        g: Math.floor(100 + 100 * t),
        b: Math.floor(100 + 50 * t)
      };
    }
  }

  // ---------------------------------------------------------------------------
  // Canvas Rendering
  // ---------------------------------------------------------------------------

  /**
   * Render to canvas (CPU fallback when WebGL not available).
   */
  render(): void {
    if (!this.canvas || !this.ctx || !this.config.enabled) return;

    const { width, height } = this.canvas;
    const imageData = this.ctx.createImageData(width, height);
    const data = imageData.data;

    const params = this.getRenderParams();

    // Generate interference pattern
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;

        // Normalized coordinates
        const nx = (x / width - 0.5) * 2;
        const ny = (y / height - 0.5) * 2;

        // Pattern 1
        const d1 = nx * Math.cos(params.angle1) + ny * Math.sin(params.angle1);
        const v1 = Math.sin(d1 * params.frequency1 + params.phase1);

        // Pattern 2
        const d2 = nx * Math.cos(params.angle2) + ny * Math.sin(params.angle2);
        const v2 = Math.sin(d2 * params.frequency2 + params.phase2);

        // Interference (multiply blend)
        const interference = (v1 * v2 + 1) / 2;
        const intensity = interference * params.amplitude;

        // Color mixing
        const r = params.colorA.r * (1 - intensity) + params.colorB.r * intensity;
        const g = params.colorA.g * (1 - intensity) + params.colorB.g * intensity;
        const b = params.colorA.b * (1 - intensity) + params.colorB.b * intensity;

        data[idx + 0] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = Math.floor(255 * params.amplitude);
      }
    }

    this.ctx.putImageData(imageData, 0, 0);
  }

  /**
   * Render with simplified parameters (faster).
   */
  renderFast(): void {
    if (!this.canvas || !this.ctx || !this.config.enabled) return;

    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    const params = this.getRenderParams();

    // Use gradients for faster rendering
    this.ctx.globalAlpha = params.amplitude;

    // Pattern 1 - radial gradient
    const grad1 = this.ctx.createRadialGradient(
      width / 2, height / 2, 0,
      width / 2, height / 2, Math.max(width, height)
    );
    grad1.addColorStop(0, `rgba(${params.colorA.r}, ${params.colorA.g}, ${params.colorA.b}, 0.5)`);
    grad1.addColorStop(0.5, `rgba(${params.colorB.r}, ${params.colorB.g}, ${params.colorB.b}, 0.5)`);
    grad1.addColorStop(1, `rgba(${params.colorA.r}, ${params.colorA.g}, ${params.colorA.b}, 0.5)`);

    this.ctx.fillStyle = grad1;
    this.ctx.fillRect(0, 0, width, height);

    // Pattern 2 - linear gradient with rotation
    this.ctx.save();
    this.ctx.translate(width / 2, height / 2);
    this.ctx.rotate(params.angle2);

    const grad2 = this.ctx.createLinearGradient(-width, 0, width, 0);
    const numStripes = Math.floor(params.frequency2 / 5);
    for (let i = 0; i <= numStripes; i++) {
      const pos = i / numStripes;
      const color = i % 2 === 0 ? params.colorA : params.colorB;
      grad2.addColorStop(pos, `rgba(${color.r}, ${color.g}, ${color.b}, 0.3)`);
    }

    this.ctx.globalCompositeOperation = 'multiply';
    this.ctx.fillStyle = grad2;
    this.ctx.fillRect(-width, -height, width * 2, height * 2);

    this.ctx.restore();
    this.ctx.globalAlpha = 1;
    this.ctx.globalCompositeOperation = 'source-over';
  }

  // ---------------------------------------------------------------------------
  // WebGL Shader Code
  // ---------------------------------------------------------------------------

  /**
   * Get GLSL fragment shader code for GPU rendering.
   */
  static getFragmentShaderCode(): string {
    return `
      precision mediump float;

      uniform float u_frequency1;
      uniform float u_frequency2;
      uniform float u_angle1;
      uniform float u_angle2;
      uniform float u_phase1;
      uniform float u_phase2;
      uniform float u_amplitude;
      uniform vec3 u_colorA;
      uniform vec3 u_colorB;
      uniform vec2 u_resolution;

      void main() {
        vec2 uv = (gl_FragCoord.xy / u_resolution.xy - 0.5) * 2.0;

        // Pattern 1
        float d1 = uv.x * cos(u_angle1) + uv.y * sin(u_angle1);
        float v1 = sin(d1 * u_frequency1 + u_phase1);

        // Pattern 2
        float d2 = uv.x * cos(u_angle2) + uv.y * sin(u_angle2);
        float v2 = sin(d2 * u_frequency2 + u_phase2);

        // Interference
        float interference = (v1 * v2 + 1.0) / 2.0;
        float intensity = interference * u_amplitude;

        // Color mixing
        vec3 color = mix(u_colorA, u_colorB, intensity);

        gl_FragColor = vec4(color, u_amplitude);
      }
    `;
  }

  /**
   * Get GLSL vertex shader code.
   */
  static getVertexShaderCode(): string {
    return `
      attribute vec2 a_position;

      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;
  }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Generate a standing wave pattern (for circular moiré).
 */
export function generateStandingWave(
  width: number,
  height: number,
  frequency: number,
  phase: number
): Float32Array {
  const data = new Float32Array(width * height);
  const cx = width / 2;
  const cy = height / 2;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy);
      data[y * width + x] = Math.sin(r * frequency * 0.1 + phase);
    }
  }

  return data;
}

/**
 * Combine two patterns with interference.
 */
export function combinePatterns(
  pattern1: Float32Array,
  pattern2: Float32Array,
  mode: 'add' | 'multiply' | 'difference'
): Float32Array {
  const result = new Float32Array(pattern1.length);

  for (let i = 0; i < pattern1.length; i++) {
    switch (mode) {
      case 'add':
        result[i] = (pattern1[i] + pattern2[i]) / 2;
        break;
      case 'multiply':
        result[i] = pattern1[i] * pattern2[i];
        break;
      case 'difference':
        result[i] = Math.abs(pattern1[i] - pattern2[i]);
        break;
    }
  }

  return result;
}
