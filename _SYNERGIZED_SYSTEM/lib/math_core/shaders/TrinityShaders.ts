/**
 * Trinity-Aware WebGL Shaders
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 *
 * Provides vertex and fragment shaders for rendering the 24-cell
 * with Trinity axis coloring and phase shift visualization.
 *
 * Color Scheme:
 * - Alpha axis: Red (#FF3366)
 * - Beta axis: Green (#33FF66)
 * - Gamma axis: Blue (#3366FF)
 * - Cross-axis edges: Yellow (#FFFF33)
 *
 * Features:
 * - 6D rotation (all 6 planes in 4D)
 * - Stereographic projection 4D → 3D
 * - Trinity axis coloring
 * - Phase shift glow effects
 * - Depth-based transparency
 *
 * Ported from CPE repo (Domusgpt/-Chronomorphic-Polytopal-Engine)
 * No import path changes needed (pure GLSL strings + utility functions).
 */

// =============================================================================
// VERTEX SHADER
// =============================================================================

export const TRINITY_VERTEX_SHADER = `#version 300 es
precision highp float;

// Attributes
in vec4 a_position;      // 4D vertex position
in float a_vertexId;     // Vertex index (0-23)
in float a_trinityAxis;  // 0=alpha, 1=beta, 2=gamma

// Uniforms - 6D rotation angles
uniform float u_rot4dXY;
uniform float u_rot4dXZ;
uniform float u_rot4dXW;
uniform float u_rot4dYZ;
uniform float u_rot4dYW;
uniform float u_rot4dZW;

// Projection parameters
uniform float u_projectionDistance;  // Stereographic projection distance
uniform mat4 u_modelViewProjection;  // 3D MVP matrix
uniform float u_time;

// Trinity state
uniform vec3 u_trinityWeights;    // [alpha, beta, gamma] weights
uniform float u_phaseProgress;    // Phase shift progress (0-1)
uniform float u_tension;          // Inter-axis tension (0-1)

// Outputs to fragment shader
out vec3 v_color;
out float v_depth4D;       // W coordinate for depth effects
out float v_intensity;     // Vertex intensity
out float v_trinityAxis;   // Which axis this vertex belongs to

// 4D rotation matrices for each plane
mat4 rotateXY(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat4(
    c, -s, 0, 0,
    s,  c, 0, 0,
    0,  0, 1, 0,
    0,  0, 0, 1
  );
}

mat4 rotateXZ(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat4(
    c, 0, -s, 0,
    0, 1,  0, 0,
    s, 0,  c, 0,
    0, 0,  0, 1
  );
}

mat4 rotateXW(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat4(
    c, 0, 0, -s,
    0, 1, 0,  0,
    0, 0, 1,  0,
    s, 0, 0,  c
  );
}

mat4 rotateYZ(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat4(
    1, 0,  0, 0,
    0, c, -s, 0,
    0, s,  c, 0,
    0, 0,  0, 1
  );
}

mat4 rotateYW(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat4(
    1, 0, 0,  0,
    0, c, 0, -s,
    0, 0, 1,  0,
    0, s, 0,  c
  );
}

mat4 rotateZW(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat4(
    1, 0, 0,  0,
    0, 1, 0,  0,
    0, 0, c, -s,
    0, 0, s,  c
  );
}

// Apply all 6 rotations
vec4 rotate4D(vec4 p) {
  mat4 R = rotateXY(u_rot4dXY);
  R = rotateXZ(u_rot4dXZ) * R;
  R = rotateXW(u_rot4dXW) * R;
  R = rotateYZ(u_rot4dYZ) * R;
  R = rotateYW(u_rot4dYW) * R;
  R = rotateZW(u_rot4dZW) * R;
  return R * p;
}

// Stereographic projection from 4D to 3D
vec3 stereographicProject(vec4 p4d) {
  float d = u_projectionDistance;
  float denom = d - p4d.w;
  if (abs(denom) < 0.001) {
    denom = 0.001 * sign(denom + 0.0001);
  }
  float scale = d / denom;
  return p4d.xyz * scale;
}

// Trinity axis colors
vec3 getTrinityColor(float axis, vec3 weights, float tension) {
  vec3 alphaColor = vec3(1.0, 0.2, 0.4);   // Red-pink
  vec3 betaColor = vec3(0.2, 1.0, 0.4);    // Green
  vec3 gammaColor = vec3(0.2, 0.4, 1.0);   // Blue

  // Base color from axis
  vec3 baseColor;
  if (axis < 0.5) {
    baseColor = alphaColor;
  } else if (axis < 1.5) {
    baseColor = betaColor;
  } else {
    baseColor = gammaColor;
  }

  // Blend with weights for superposition effect
  vec3 blendedColor = alphaColor * weights.x + betaColor * weights.y + gammaColor * weights.z;

  // Mix based on tension (high tension = more blended)
  return mix(baseColor, blendedColor, tension * 0.5);
}

void main() {
  // Apply 4D rotation
  vec4 rotated = rotate4D(a_position);

  // Store W for depth effects
  v_depth4D = rotated.w;

  // Stereographic projection
  vec3 projected = stereographicProject(rotated);

  // Apply 3D MVP transformation
  gl_Position = u_modelViewProjection * vec4(projected, 1.0);

  // Set point size based on W depth
  float depthFactor = 1.0 / (2.0 - rotated.w);
  gl_PointSize = 10.0 * depthFactor;

  // Compute color
  v_color = getTrinityColor(a_trinityAxis, u_trinityWeights, u_tension);

  // Compute intensity (brighter for vertices closer in W)
  v_intensity = 0.5 + 0.5 * (1.0 + rotated.w) / 2.0;

  // Phase shift glow
  if (u_phaseProgress > 0.0) {
    float glow = sin(u_phaseProgress * 3.14159) * 0.3;
    v_intensity += glow;
  }

  v_trinityAxis = a_trinityAxis;
}
`;

// =============================================================================
// FRAGMENT SHADER
// =============================================================================

export const TRINITY_FRAGMENT_SHADER = `#version 300 es
precision highp float;

// Inputs from vertex shader
in vec3 v_color;
in float v_depth4D;
in float v_intensity;
in float v_trinityAxis;

// Uniforms
uniform float u_opacity;
uniform float u_tension;
uniform float u_phaseProgress;
uniform float u_time;

// Output
out vec4 fragColor;

void main() {
  // Base color with intensity
  vec3 color = v_color * v_intensity;

  // Depth-based alpha (farther in W = more transparent)
  float depthAlpha = 0.3 + 0.7 * (1.0 + v_depth4D) / 2.0;

  // Phase shift pulsing
  float pulse = 1.0;
  if (u_phaseProgress > 0.0) {
    pulse = 0.8 + 0.4 * sin(u_time * 10.0 + v_trinityAxis * 2.094);
  }

  // Tension glow (yellow-ish overlay)
  if (u_tension > 0.3) {
    vec3 tensionColor = vec3(1.0, 1.0, 0.3);
    color = mix(color, tensionColor, (u_tension - 0.3) * 0.5);
  }

  // Final alpha
  float alpha = u_opacity * depthAlpha * pulse;

  fragColor = vec4(color, alpha);
}
`;

// =============================================================================
// EDGE SHADERS (for rendering edges between vertices)
// =============================================================================

export const EDGE_VERTEX_SHADER = `#version 300 es
precision highp float;

// Attributes
in vec4 a_position;       // 4D vertex position
in float a_trinityAxis;   // Trinity axis of this endpoint
in float a_otherAxis;     // Trinity axis of other endpoint

// Uniforms - same as vertex shader
uniform float u_rot4dXY;
uniform float u_rot4dXZ;
uniform float u_rot4dXW;
uniform float u_rot4dYZ;
uniform float u_rot4dYW;
uniform float u_rot4dZW;
uniform float u_projectionDistance;
uniform mat4 u_modelViewProjection;
uniform vec3 u_trinityWeights;
uniform float u_tension;

// Outputs
out vec3 v_color;
out float v_depth4D;
out float v_isCrossAxis;  // 1.0 if edge crosses axes

// Rotation functions (same as vertex shader)
mat4 rotateXY(float angle) {
  float c = cos(angle), s = sin(angle);
  return mat4(c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1);
}
mat4 rotateXZ(float angle) {
  float c = cos(angle), s = sin(angle);
  return mat4(c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1);
}
mat4 rotateXW(float angle) {
  float c = cos(angle), s = sin(angle);
  return mat4(c,0,0,-s, 0,1,0,0, 0,0,1,0, s,0,0,c);
}
mat4 rotateYZ(float angle) {
  float c = cos(angle), s = sin(angle);
  return mat4(1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1);
}
mat4 rotateYW(float angle) {
  float c = cos(angle), s = sin(angle);
  return mat4(1,0,0,0, 0,c,0,-s, 0,0,1,0, 0,s,0,c);
}
mat4 rotateZW(float angle) {
  float c = cos(angle), s = sin(angle);
  return mat4(1,0,0,0, 0,1,0,0, 0,0,c,-s, 0,0,s,c);
}

vec4 rotate4D(vec4 p) {
  mat4 R = rotateXY(u_rot4dXY);
  R = rotateXZ(u_rot4dXZ) * R;
  R = rotateXW(u_rot4dXW) * R;
  R = rotateYZ(u_rot4dYZ) * R;
  R = rotateYW(u_rot4dYW) * R;
  R = rotateZW(u_rot4dZW) * R;
  return R * p;
}

vec3 stereographicProject(vec4 p4d) {
  float d = u_projectionDistance;
  float denom = max(abs(d - p4d.w), 0.001) * sign(d - p4d.w + 0.0001);
  return p4d.xyz * (d / denom);
}

void main() {
  vec4 rotated = rotate4D(a_position);
  v_depth4D = rotated.w;

  vec3 projected = stereographicProject(rotated);
  gl_Position = u_modelViewProjection * vec4(projected, 1.0);

  // Check if cross-axis edge
  v_isCrossAxis = abs(a_trinityAxis - a_otherAxis) > 0.5 ? 1.0 : 0.0;

  // Edge color
  if (v_isCrossAxis > 0.5) {
    // Cross-axis edges are yellow
    v_color = vec3(1.0, 1.0, 0.3);
  } else {
    // Same-axis edges use axis color
    vec3 alphaColor = vec3(1.0, 0.2, 0.4);
    vec3 betaColor = vec3(0.2, 1.0, 0.4);
    vec3 gammaColor = vec3(0.2, 0.4, 1.0);

    if (a_trinityAxis < 0.5) {
      v_color = alphaColor;
    } else if (a_trinityAxis < 1.5) {
      v_color = betaColor;
    } else {
      v_color = gammaColor;
    }
  }
}
`;

export const EDGE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec3 v_color;
in float v_depth4D;
in float v_isCrossAxis;

uniform float u_opacity;
uniform float u_tension;

out vec4 fragColor;

void main() {
  float depthAlpha = 0.3 + 0.7 * (1.0 + v_depth4D) / 2.0;

  // Cross-axis edges glow more with tension
  float crossGlow = v_isCrossAxis * u_tension * 0.5;

  vec3 color = v_color * (1.0 + crossGlow);
  float alpha = u_opacity * depthAlpha * (0.7 + crossGlow);

  fragColor = vec4(color, alpha);
}
`;

// =============================================================================
// SHADER UTILITIES
// =============================================================================

/**
 * Compile a shader from source.
 */
export function compileShader(
  gl: WebGL2RenderingContext,
  source: string,
  type: number
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

/**
 * Create a shader program from vertex and fragment shaders.
 */
export function createShaderProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string
): WebGLProgram | null {
  const vertexShader = compileShader(gl, vertexSource, gl.VERTEX_SHADER);
  const fragmentShader = compileShader(gl, fragmentSource, gl.FRAGMENT_SHADER);

  if (!vertexShader || !fragmentShader) return null;

  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program linking error:', gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

/**
 * Create the Trinity vertex program.
 */
export function createTrinityVertexProgram(
  gl: WebGL2RenderingContext
): WebGLProgram | null {
  return createShaderProgram(gl, TRINITY_VERTEX_SHADER, TRINITY_FRAGMENT_SHADER);
}

/**
 * Create the Trinity edge program.
 */
export function createTrinityEdgeProgram(
  gl: WebGL2RenderingContext
): WebGLProgram | null {
  return createShaderProgram(gl, EDGE_VERTEX_SHADER, EDGE_FRAGMENT_SHADER);
}

// =============================================================================
// SHADER EXPORT (GLSL/HLSL)
// =============================================================================

/**
 * Export shaders for use in other environments.
 */
export const ShaderExports = {
  GLSL: {
    vertex: TRINITY_VERTEX_SHADER,
    fragment: TRINITY_FRAGMENT_SHADER,
    edgeVertex: EDGE_VERTEX_SHADER,
    edgeFragment: EDGE_FRAGMENT_SHADER
  },
};
