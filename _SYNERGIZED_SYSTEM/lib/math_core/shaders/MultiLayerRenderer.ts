/**
 * Multi-Layer Renderer
 *
 * Provides advanced rendering controls for nested polytope structures:
 * - Individual layer visibility toggle
 * - Alpha blending by 4D depth (w-coordinate)
 * - Cross-section slicing through 3D hyperplanes
 * - Moiré layer overlay effects
 *
 * Based on the visualization requirements from Phase 5 of the implementation plan.
 *
 * Ported from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * with import paths adapted for PPP math_core layout.
 */

import type { Vector4D, TrinityAxis } from '../geometric_algebra/types';
import { PHI, phiPower } from '../topology/GoldenRatioScaling';

// =============================================================================
// TYPES
// =============================================================================

export interface LayerConfig {
  /** Layer index/level */
  readonly id: number;
  /** Whether layer is visible */
  visible: boolean;
  /** Opacity (0-1) */
  opacity: number;
  /** Color tint [r, g, b] (0-1) */
  colorTint: [number, number, number];
  /** Scale factor (typically φ^level) */
  scale: number;
  /** Whether to highlight this layer */
  highlighted: boolean;
}

export interface SliceConfig {
  /** Whether slicing is enabled */
  enabled: boolean;
  /** Slice plane normal (4D) */
  normal: Vector4D;
  /** Slice plane offset from origin */
  offset: number;
  /** Slice thickness (for soft cutoff) */
  thickness: number;
  /** Whether to show both sides of slice */
  showBothSides: boolean;
}

export interface RenderConfig {
  /** Layer configurations */
  layers: LayerConfig[];
  /** Slice configuration */
  slice: SliceConfig;
  /** Global opacity multiplier */
  globalOpacity: number;
  /** Whether to use w-depth sorting */
  wDepthSort: boolean;
  /** Background color */
  backgroundColor: [number, number, number];
  /** Whether to show edges */
  showEdges: boolean;
  /** Whether to show vertices */
  showVertices: boolean;
  /** Whether to show faces */
  showFaces: boolean;
  /** Edge thickness */
  edgeThickness: number;
  /** Vertex size */
  vertexSize: number;
}

// =============================================================================
// DEFAULT CONFIGURATIONS
// =============================================================================

export const DEFAULT_LAYER_CONFIG: LayerConfig = {
  id: 0,
  visible: true,
  opacity: 1.0,
  colorTint: [1, 1, 1],
  scale: 1.0,
  highlighted: false
};

export const DEFAULT_SLICE_CONFIG: SliceConfig = {
  enabled: false,
  normal: [0, 0, 0, 1], // Slice along W axis
  offset: 0,
  thickness: 0.1,
  showBothSides: false
};

export const DEFAULT_RENDER_CONFIG: RenderConfig = {
  layers: [{ ...DEFAULT_LAYER_CONFIG }],
  slice: { ...DEFAULT_SLICE_CONFIG },
  globalOpacity: 1.0,
  wDepthSort: true,
  backgroundColor: [0.05, 0.05, 0.1],
  showEdges: true,
  showVertices: true,
  showFaces: false,
  edgeThickness: 0.02,
  vertexSize: 0.05
};

// =============================================================================
// MULTI-LAYER SHADER CODE
// =============================================================================

/**
 * GLSL code for multi-layer vertex shader.
 */
export const MULTI_LAYER_VERTEX_SHADER = `#version 300 es
precision highp float;

// Attributes
in vec4 a_position;      // 4D position
in vec3 a_color;         // Vertex color
in float a_layerId;      // Layer index
in float a_trinityAxis;  // Trinity axis (0=alpha, 1=beta, 2=gamma)

// Uniforms
uniform mat4 u_modelView;
uniform mat4 u_projection;
uniform mat4 u_rotation4D[2];  // Two 4x4 matrices for 4D rotation
uniform float u_wProjection;    // W-coordinate for stereographic projection

// Layer uniforms
uniform float u_layerOpacity[8];
uniform vec3 u_layerTint[8];
uniform float u_layerScale[8];
uniform int u_layerVisible[8];

// Slice uniforms
uniform bool u_sliceEnabled;
uniform vec4 u_sliceNormal;
uniform float u_sliceOffset;
uniform float u_sliceThickness;

// Outputs
out vec3 v_color;
out float v_opacity;
out float v_sliceDistance;
out float v_wDepth;

// Apply 4D rotation
vec4 rotate4D(vec4 p) {
    vec4 temp = u_rotation4D[0] * p;
    return u_rotation4D[1] * temp;
}

// Stereographic projection from 4D to 3D
vec3 stereographicProject(vec4 p4) {
    float w = p4.w;
    float denom = u_wProjection - w;
    if (abs(denom) < 0.001) denom = 0.001;
    return p4.xyz / denom;
}

void main() {
    int layerIdx = int(a_layerId);

    // Check layer visibility
    if (u_layerVisible[layerIdx] == 0) {
        gl_Position = vec4(0.0, 0.0, -1000.0, 1.0); // Cull
        v_opacity = 0.0;
        return;
    }

    // Scale by layer
    vec4 scaledPos = a_position * u_layerScale[layerIdx];

    // Apply 4D rotation
    vec4 rotated = rotate4D(scaledPos);

    // Calculate slice distance
    v_sliceDistance = dot(rotated, u_sliceNormal) - u_sliceOffset;

    // If slicing enabled and outside slice, cull
    if (u_sliceEnabled && abs(v_sliceDistance) > u_sliceThickness) {
        gl_Position = vec4(0.0, 0.0, -1000.0, 1.0);
        v_opacity = 0.0;
        return;
    }

    // Store W depth for sorting
    v_wDepth = rotated.w;

    // Project to 3D
    vec3 pos3D = stereographicProject(rotated);

    // Apply model-view and projection
    gl_Position = u_projection * u_modelView * vec4(pos3D, 1.0);

    // Calculate color with layer tint
    v_color = a_color * u_layerTint[layerIdx];

    // Calculate opacity with layer opacity and slice fade
    float sliceFade = u_sliceEnabled
        ? 1.0 - smoothstep(0.0, u_sliceThickness, abs(v_sliceDistance))
        : 1.0;

    v_opacity = u_layerOpacity[layerIdx] * sliceFade;
}
`;

/**
 * GLSL code for multi-layer fragment shader.
 */
export const MULTI_LAYER_FRAGMENT_SHADER = `#version 300 es
precision highp float;

// Inputs
in vec3 v_color;
in float v_opacity;
in float v_sliceDistance;
in float v_wDepth;

// Uniforms
uniform float u_globalOpacity;
uniform bool u_wDepthSort;
uniform vec3 u_backgroundColor;

// Output
out vec4 fragColor;

void main() {
    if (v_opacity < 0.01) {
        discard;
    }

    // W-depth based alpha modification
    float depthAlpha = u_wDepthSort
        ? clamp(1.0 - abs(v_wDepth) * 0.3, 0.3, 1.0)
        : 1.0;

    float finalOpacity = v_opacity * u_globalOpacity * depthAlpha;

    // Output color with opacity
    fragColor = vec4(v_color, finalOpacity);
}
`;

// =============================================================================
// CROSS-SECTION SLICER
// =============================================================================

/**
 * Cross-section slicer for 4D polytopes.
 *
 * Creates 3D cross-sections by intersecting the 4D polytope
 * with a 3D hyperplane.
 */
export class CrossSectionSlicer {
  private _config: SliceConfig;

  constructor(config: Partial<SliceConfig> = {}) {
    this._config = { ...DEFAULT_SLICE_CONFIG, ...config };
  }

  /** Get current configuration */
  get config(): SliceConfig {
    return { ...this._config };
  }

  /** Enable slicing */
  enable(): void {
    this._config.enabled = true;
  }

  /** Disable slicing */
  disable(): void {
    this._config.enabled = false;
  }

  /** Set slice plane normal */
  setNormal(normal: Vector4D): void {
    // Normalize
    const len = Math.sqrt(
      normal[0]*normal[0] + normal[1]*normal[1] +
      normal[2]*normal[2] + normal[3]*normal[3]
    );
    this._config.normal = [
      normal[0]/len, normal[1]/len,
      normal[2]/len, normal[3]/len
    ];
  }

  /** Set slice offset */
  setOffset(offset: number): void {
    this._config.offset = offset;
  }

  /** Set slice thickness */
  setThickness(thickness: number): void {
    this._config.thickness = Math.max(0.001, thickness);
  }

  /** Animate slice through polytope */
  animateSlice(
    fromOffset: number,
    toOffset: number,
    duration: number,
    onUpdate: (offset: number) => void
  ): void {
    const startTime = performance.now();
    const range = toOffset - fromOffset;

    const animate = () => {
      const elapsed = performance.now() - startTime;
      const t = Math.min(1, elapsed / duration);

      // Smooth easing
      const eased = t * t * (3 - 2 * t);
      const offset = fromOffset + range * eased;

      this._config.offset = offset;
      onUpdate(offset);

      if (t < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }

  /**
   * Compute signed distance from point to slice plane.
   */
  signedDistance(point: Vector4D): number {
    return (
      point[0] * this._config.normal[0] +
      point[1] * this._config.normal[1] +
      point[2] * this._config.normal[2] +
      point[3] * this._config.normal[3]
    ) - this._config.offset;
  }

  /**
   * Check if a point is within the slice.
   */
  isInSlice(point: Vector4D): boolean {
    if (!this._config.enabled) return true;
    return Math.abs(this.signedDistance(point)) <= this._config.thickness;
  }

  /**
   * Filter vertices to only those in slice.
   */
  filterVertices(vertices: Vector4D[]): Vector4D[] {
    if (!this._config.enabled) return vertices;
    return vertices.filter(v => this.isInSlice(v));
  }

  /**
   * Compute intersection of edge with slice plane.
   */
  intersectEdge(v1: Vector4D, v2: Vector4D): Vector4D | null {
    if (!this._config.enabled) return null;

    const d1 = this.signedDistance(v1);
    const d2 = this.signedDistance(v2);

    // Check if edge crosses plane
    if (d1 * d2 >= 0) return null;

    // Compute intersection parameter
    const t = d1 / (d1 - d2);

    return [
      v1[0] + t * (v2[0] - v1[0]),
      v1[1] + t * (v2[1] - v1[1]),
      v1[2] + t * (v2[2] - v1[2]),
      v1[3] + t * (v2[3] - v1[3])
    ];
  }

  /**
   * Get preset slice planes.
   */
  static getPresetPlanes(): Record<string, Vector4D> {
    return {
      'XYZ (W=const)': [0, 0, 0, 1],
      'XYW (Z=const)': [0, 0, 1, 0],
      'XZW (Y=const)': [0, 1, 0, 0],
      'YZW (X=const)': [1, 0, 0, 0],
      'Diagonal': [0.5, 0.5, 0.5, 0.5],
      'Trinity Alpha': [1, 1, 0, 0],
      'Trinity Beta': [1, 0, 1, 0],
      'Trinity Gamma': [1, 0, 0, 1]
    };
  }
}

// =============================================================================
// MULTI-LAYER MANAGER
// =============================================================================

/**
 * Multi-Layer Render Manager.
 *
 * Manages multiple φ-scaled layers with visibility and opacity controls.
 */
export class MultiLayerManager {
  private _config: RenderConfig;
  private _slicer: CrossSectionSlicer;

  constructor(numLayers: number = 5, config: Partial<RenderConfig> = {}) {
    // Create layer configs
    const layers: LayerConfig[] = [];
    const centerLevel = Math.floor(numLayers / 2);

    for (let i = 0; i < numLayers; i++) {
      const level = i - centerLevel;
      layers.push({
        id: i,
        visible: true,
        opacity: 1 / (1 + Math.abs(level) * 0.3),
        colorTint: this._getDefaultTint(level),
        scale: phiPower(level),
        highlighted: level === 0
      });
    }

    this._config = {
      ...DEFAULT_RENDER_CONFIG,
      layers,
      ...config
    };

    this._slicer = new CrossSectionSlicer(this._config.slice);
  }

  /** Get current configuration */
  get config(): RenderConfig {
    return this._config;
  }

  /** Get slicer */
  get slicer(): CrossSectionSlicer {
    return this._slicer;
  }

  /** Get number of layers */
  get layerCount(): number {
    return this._config.layers.length;
  }

  /**
   * Set layer visibility.
   */
  setLayerVisible(layerId: number, visible: boolean): void {
    if (layerId >= 0 && layerId < this._config.layers.length) {
      this._config.layers[layerId].visible = visible;
    }
  }

  /**
   * Set layer opacity.
   */
  setLayerOpacity(layerId: number, opacity: number): void {
    if (layerId >= 0 && layerId < this._config.layers.length) {
      this._config.layers[layerId].opacity = Math.max(0, Math.min(1, opacity));
    }
  }

  /**
   * Set layer color tint.
   */
  setLayerTint(layerId: number, tint: [number, number, number]): void {
    if (layerId >= 0 && layerId < this._config.layers.length) {
      this._config.layers[layerId].colorTint = tint;
    }
  }

  /**
   * Toggle all layers.
   */
  toggleAllLayers(visible: boolean): void {
    for (const layer of this._config.layers) {
      layer.visible = visible;
    }
  }

  /**
   * Show only specific layer.
   */
  showOnlyLayer(layerId: number): void {
    for (const layer of this._config.layers) {
      layer.visible = layer.id === layerId;
    }
  }

  /**
   * Get shader uniforms for current configuration.
   */
  getShaderUniforms(): Record<string, number | number[] | boolean> {
    const layerOpacity: number[] = [];
    const layerTint: number[] = [];
    const layerScale: number[] = [];
    const layerVisible: number[] = [];

    for (const layer of this._config.layers) {
      layerOpacity.push(layer.opacity);
      layerTint.push(...layer.colorTint);
      layerScale.push(layer.scale);
      layerVisible.push(layer.visible ? 1 : 0);
    }

    // Pad to 8 layers (shader array size)
    while (layerOpacity.length < 8) {
      layerOpacity.push(0);
      layerTint.push(1, 1, 1);
      layerScale.push(1);
      layerVisible.push(0);
    }

    const slice = this._slicer.config;

    return {
      u_layerOpacity: layerOpacity,
      u_layerTint: layerTint,
      u_layerScale: layerScale,
      u_layerVisible: layerVisible,
      u_globalOpacity: this._config.globalOpacity,
      u_wDepthSort: this._config.wDepthSort,
      u_sliceEnabled: slice.enabled,
      u_sliceNormal: [...slice.normal],
      u_sliceOffset: slice.offset,
      u_sliceThickness: slice.thickness
    };
  }

  /**
   * Get default color tint for layer level.
   */
  private _getDefaultTint(level: number): [number, number, number] {
    if (level < 0) {
      // Inner layers: blue tint
      return [0.7, 0.8, 1.0];
    } else if (level > 0) {
      // Outer layers: gold tint
      return [1.0, 0.9, 0.7];
    }
    // Base layer: white
    return [1.0, 1.0, 1.0];
  }

  /**
   * Create preset configurations.
   */
  static createPreset(name: string): MultiLayerManager {
    switch (name) {
      case 'trinity': {
        const trinity = new MultiLayerManager(3);
        trinity.setLayerTint(0, [1.0, 0.3, 0.3]); // Alpha - red
        trinity.setLayerTint(1, [0.3, 1.0, 0.3]); // Beta - green
        trinity.setLayerTint(2, [0.3, 0.3, 1.0]); // Gamma - blue
        return trinity;
      }

      case 'nested':
        return new MultiLayerManager(5);

      case 'e8': {
        const e8 = new MultiLayerManager(2);
        e8.setLayerTint(0, [1.0, 0.8, 0.5]); // Outer - gold
        e8.setLayerTint(1, [0.5, 0.8, 1.0]); // Inner - silver
        return e8;
      }

      default:
        return new MultiLayerManager();
    }
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  CrossSectionSlicer,
  MultiLayerManager
};

export default MultiLayerManager;
