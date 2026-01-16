/**
 * E8 Visualization Renderer
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * WebGL renderer extension for E8→H4 polytopal visualization.
 * Integrates with the existing HypercubeRenderer to display:
 *
 * 1. Four chiral 600-cells (H4L, φH4L, H4R, φH4R)
 * 2. Trinity decomposition (3×16-cell color coding)
 * 3. Three-body trajectory on 600-cell surface
 * 4. Standard Model particle positions
 *
 * Visualization Modes:
 * - E8_PROJECTION: Full E8→4D projection (240 vertices)
 * - CHIRAL_600: Four overlapping 600-cells with chirality coloring
 * - TRINITY_24: 24-cell with RGB color-coded 16-cell subsets
 * - THREE_BODY: 600-cell with body trajectories as geodesics
 * - STANDARD_MODEL: Particle positions on 24-cell (Ali mapping)
 */

import { DATA_CHANNEL_COUNT } from './constants.js';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Golden ratio for E8→H4 relationships */
const PHI = (1 + Math.sqrt(5)) / 2;

/** Inverse golden ratio */
const PHI_INV = PHI - 1;

/** Visualization modes */
export const E8_VIS_MODES = {
    E8_PROJECTION: 'e8_projection',
    CHIRAL_600: 'chiral_600',
    TRINITY_24: 'trinity_24',
    THREE_BODY: 'three_body',
    STANDARD_MODEL: 'standard_model'
};

/** Default configuration */
const DEFAULT_E8_CONFIG = {
    mode: E8_VIS_MODES.E8_PROJECTION,
    showVertices: true,
    showEdges: true,
    showCells: false,
    vertexSize: 0.04,
    edgeThickness: 0.01,
    chirality: 'both', // 'L', 'R', 'both'
    trinityOpacity: [0.8, 0.8, 0.8], // α, β, γ
    bodyColors: ['#FF4444', '#44FF44', '#4444FF'],
    animateProjection: true,
    projectionSpeed: 0.001
};

// =============================================================================
// E8 VERTEX GENERATION (SHADER-COMPATIBLE)
// =============================================================================

/**
 * Generate E8 roots as Float32Array for GPU upload.
 * Returns 240 vertices × 8 components = 1920 floats.
 */
export function generateE8VertexBuffer() {
    const roots = [];

    // Type 1: 112 roots from (±1, ±1, 0, 0, 0, 0, 0, 0) permutations
    for (let i = 0; i < 8; i++) {
        for (let j = i + 1; j < 8; j++) {
            for (const si of [-1, 1]) {
                for (const sj of [-1, 1]) {
                    const root = [0, 0, 0, 0, 0, 0, 0, 0];
                    root[i] = si;
                    root[j] = sj;
                    roots.push(...root);
                }
            }
        }
    }

    // Type 2: 128 roots from (±½)^8 with even parity
    const half = 0.5;
    for (let mask = 0; mask < 256; mask++) {
        let popcount = 0;
        let m = mask;
        while (m) {
            popcount += m & 1;
            m >>= 1;
        }

        if (popcount % 2 === 0) {
            roots.push(
                (mask & 1) ? -half : half,
                (mask & 2) ? -half : half,
                (mask & 4) ? -half : half,
                (mask & 8) ? -half : half,
                (mask & 16) ? -half : half,
                (mask & 32) ? -half : half,
                (mask & 64) ? -half : half,
                (mask & 128) ? -half : half
            );
        }
    }

    return new Float32Array(roots);
}

/**
 * Generate 24-cell vertices as Float32Array.
 * Returns 24 vertices × 4 components = 96 floats.
 */
export function generate24CellVertexBuffer() {
    const vertices = [];

    for (let i = 0; i < 4; i++) {
        for (let j = i + 1; j < 4; j++) {
            for (const si of [-1, 1]) {
                for (const sj of [-1, 1]) {
                    const v = [0, 0, 0, 0];
                    v[i] = si;
                    v[j] = sj;
                    vertices.push(...v);
                }
            }
        }
    }

    return new Float32Array(vertices);
}

/**
 * Generate 600-cell vertices as Float32Array.
 * Returns 120 vertices × 4 components = 480 floats.
 */
export function generate600CellVertexBuffer() {
    const vertices = [];

    // Type A: 8 vertices (±1, 0, 0, 0)
    for (let axis = 0; axis < 4; axis++) {
        for (const sign of [-1, 1]) {
            const v = [0, 0, 0, 0];
            v[axis] = sign;
            vertices.push(...v);
        }
    }

    // Type B: 16 vertices (±½, ±½, ±½, ±½)
    for (let mask = 0; mask < 16; mask++) {
        vertices.push(
            (mask & 1) ? 0.5 : -0.5,
            (mask & 2) ? 0.5 : -0.5,
            (mask & 4) ? 0.5 : -0.5,
            (mask & 8) ? 0.5 : -0.5
        );
    }

    // Type C: 96 vertices from golden ratio combinations
    // Even permutations of (0, 1/(2φ), 1/2, φ/2)
    const baseCoords = [0, 0.5 * PHI_INV, 0.5, 0.5 * PHI];
    const evenPerms = generateEvenPermutations([0, 1, 2, 3]);

    for (const perm of evenPerms) {
        const permuted = perm.map(i => baseCoords[i]);

        for (let signMask = 0; signMask < 16; signMask++) {
            const v = [0, 0, 0, 0];
            let valid = true;

            for (let i = 0; i < 4; i++) {
                if (permuted[i] === 0) {
                    v[i] = 0;
                    if (signMask & (1 << i)) {
                        valid = false;
                        break;
                    }
                } else {
                    v[i] = (signMask & (1 << i)) ? -permuted[i] : permuted[i];
                }
            }

            if (valid && !isDuplicate(vertices, v)) {
                vertices.push(...v);
            }
        }
    }

    return new Float32Array(vertices);
}

function generateEvenPermutations(arr) {
    const result = [];

    function permute(current, remaining, isEven) {
        if (remaining.length === 0) {
            if (isEven) result.push([...current]);
            return;
        }

        for (let i = 0; i < remaining.length; i++) {
            const next = [...current, remaining[i]];
            const newRemaining = remaining.filter((_, j) => j !== i);
            const newIsEven = i % 2 === 0 ? isEven : !isEven;
            permute(next, newRemaining, newIsEven);
        }
    }

    permute([], arr, true);
    return result;
}

function isDuplicate(vertices, v) {
    const epsilon = 0.0001;
    for (let i = 0; i < vertices.length; i += 4) {
        if (
            Math.abs(vertices[i] - v[0]) < epsilon &&
            Math.abs(vertices[i + 1] - v[1]) < epsilon &&
            Math.abs(vertices[i + 2] - v[2]) < epsilon &&
            Math.abs(vertices[i + 3] - v[3]) < epsilon
        ) {
            return true;
        }
    }
    return false;
}

// =============================================================================
// MOXNESS MATRIX (8×8)
// =============================================================================

/**
 * Create the Moxness 8×8 folding matrix as Float32Array.
 */
export function createMoxnessMatrixBuffer() {
    const a = 0.5;
    const b = 0.5 * PHI_INV;
    const c = 0.5 * PHI;

    // Row-major 8×8 matrix
    return new Float32Array([
        a, a, a, a, b, b, -b, -b,      // Row 0
        a, a, -a, -a, b, -b, b, -b,    // Row 1
        a, -a, a, -a, b, -b, -b, b,    // Row 2
        a, -a, -a, a, b, b, -b, -b,    // Row 3
        c, c, c, c, -a, -a, a, a,      // Row 4
        c, c, -c, -c, -a, a, -a, a,    // Row 5
        c, -c, c, -c, -a, a, a, -a,    // Row 6
        c, -c, -c, c, -a, -a, a, a     // Row 7
    ]);
}

// =============================================================================
// E8 VISUALIZATION RENDERER CLASS
// =============================================================================

/**
 * E8 Visualization Renderer
 *
 * Extends the WebGL rendering pipeline to support E8→H4 polytopal visualization.
 * Manages vertex buffers, projection matrices, and visualization modes.
 */
export class E8VisualizationRenderer {
    constructor(gl, config = {}) {
        this.gl = gl;
        this.config = { ...DEFAULT_E8_CONFIG, ...config };

        // Vertex buffers
        this.e8Buffer = null;
        this.h4Buffer = null;
        this.cell24Buffer = null;
        this.cell600Buffer = null;

        // Transformation state
        this.moxnessMatrix = createMoxnessMatrixBuffer();
        this.projectionAngle = 0;

        // Three-body state
        this.bodyPositions = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ];
        this.trajectoryHistory = [];
        this.maxTrajectoryLength = 1000;

        // Initialize buffers
        this._initBuffers();
    }

    /**
     * Initialize GPU buffers for all polytope vertex data.
     */
    _initBuffers() {
        const gl = this.gl;

        // E8 vertices (240 × 8D)
        this.e8Buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.e8Buffer);
        gl.bufferData(gl.ARRAY_BUFFER, generateE8VertexBuffer(), gl.STATIC_DRAW);

        // 24-cell vertices (24 × 4D)
        this.cell24Buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.cell24Buffer);
        gl.bufferData(gl.ARRAY_BUFFER, generate24CellVertexBuffer(), gl.STATIC_DRAW);

        // 600-cell vertices (120 × 4D)
        this.cell600Buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.cell600Buffer);
        gl.bufferData(gl.ARRAY_BUFFER, generate600CellVertexBuffer(), gl.STATIC_DRAW);

        console.log('[E8VisualizationRenderer] Initialized vertex buffers');
    }

    /**
     * Set the visualization mode.
     */
    setMode(mode) {
        if (Object.values(E8_VIS_MODES).includes(mode)) {
            this.config.mode = mode;
        }
    }

    /**
     * Update three-body positions from physics simulation.
     */
    updateThreeBodyState(body1, body2, body3) {
        this.bodyPositions[0] = body1;
        this.bodyPositions[1] = body2;
        this.bodyPositions[2] = body3;

        // Record trajectory
        if (this.trajectoryHistory.length < this.maxTrajectoryLength) {
            this.trajectoryHistory.push([...body1, ...body2, ...body3]);
        } else {
            this.trajectoryHistory.shift();
            this.trajectoryHistory.push([...body1, ...body2, ...body3]);
        }
    }

    /**
     * Get projection uniforms for shader.
     */
    getProjectionUniforms() {
        // Animate projection angle if enabled
        if (this.config.animateProjection) {
            this.projectionAngle += this.config.projectionSpeed;
        }

        return {
            u_e8Mode: Object.values(E8_VIS_MODES).indexOf(this.config.mode),
            u_projectionAngle: this.projectionAngle,
            u_phi: PHI,
            u_phiInv: PHI_INV,
            u_vertexSize: this.config.vertexSize,
            u_edgeThickness: this.config.edgeThickness,
            u_trinityOpacity: this.config.trinityOpacity
        };
    }

    /**
     * Render the E8 visualization layer.
     * This method should be called from the main render loop.
     */
    render(time = 0) {
        // Get uniforms for shader
        const uniforms = this.getProjectionUniforms();

        // Return render data for integration with HypercubeRenderer
        return {
            mode: this.config.mode,
            uniforms,
            bodyPositions: this.bodyPositions,
            trajectoryLength: this.trajectoryHistory.length,
            vertexCounts: {
                e8: 240,
                cell600: 120,
                cell24: 24
            }
        };
    }

    /**
     * Clean up GPU resources.
     */
    dispose() {
        const gl = this.gl;

        if (this.e8Buffer) {
            gl.deleteBuffer(this.e8Buffer);
        }
        if (this.cell24Buffer) {
            gl.deleteBuffer(this.cell24Buffer);
        }
        if (this.cell600Buffer) {
            gl.deleteBuffer(this.cell600Buffer);
        }

        console.log('[E8VisualizationRenderer] Disposed GPU resources');
    }
}

// =============================================================================
// TRINITY COLORING HELPER
// =============================================================================

/**
 * Get RGB color for Trinity decomposition based on vertex index.
 */
export function getTrinityColor(vertexIndex) {
    // Map vertex to α (Red), β (Green), or γ (Blue) subset
    const subset = vertexIndex % 3;

    switch (subset) {
        case 0: return [1.0, 0.2, 0.2]; // Red - α
        case 1: return [0.2, 1.0, 0.2]; // Green - β
        case 2: return [0.2, 0.2, 1.0]; // Blue - γ
        default: return [1.0, 1.0, 1.0];
    }
}

/**
 * Get particle color for Standard Model mapping.
 */
export function getParticleColor(particleType) {
    switch (particleType) {
        case 'quark': return [1.0, 0.5, 0.0];    // Orange
        case 'lepton': return [0.0, 0.8, 1.0];   // Cyan
        case 'gluon': return [1.0, 0.0, 1.0];    // Magenta
        case 'boson': return [1.0, 1.0, 0.0];    // Yellow
        case 'higgs': return [1.0, 1.0, 1.0];    // White
        default: return [0.5, 0.5, 0.5];
    }
}

// =============================================================================
// SHADER EXTENSION
// =============================================================================

/**
 * E8-specific shader functions for injection into fragment shader.
 */
export const e8ShaderFunctions = /* glsl */ `
    // Golden ratio constants
    #define PHI 1.6180339887
    #define PHI_INV 0.6180339887

    // Apply Moxness folding to project 8D to 4D
    vec4 moxnessFold(vec4 v8_lo, vec4 v8_hi) {
        // Simplified 8D→4D projection using first 4 rows of Moxness matrix
        float a = 0.5;
        float b = 0.5 * PHI_INV;

        return vec4(
            a*(v8_lo.x + v8_lo.y + v8_lo.z + v8_lo.w) + b*(v8_hi.x + v8_hi.y - v8_hi.z - v8_hi.w),
            a*(v8_lo.x + v8_lo.y - v8_lo.z - v8_lo.w) + b*(v8_hi.x - v8_hi.y + v8_hi.z - v8_hi.w),
            a*(v8_lo.x - v8_lo.y + v8_lo.z - v8_lo.w) + b*(v8_hi.x - v8_hi.y - v8_hi.z + v8_hi.w),
            a*(v8_lo.x - v8_lo.y - v8_lo.z + v8_lo.w) + b*(v8_hi.x + v8_hi.y - v8_hi.z - v8_hi.w)
        );
    }

    // Trinity color mapping for 24-cell vertices
    vec3 trinityColor(float vertexIndex) {
        float subset = mod(vertexIndex, 3.0);
        if (subset < 0.5) return vec3(1.0, 0.2, 0.2); // Red - α
        if (subset < 1.5) return vec3(0.2, 1.0, 0.2); // Green - β
        return vec3(0.2, 0.2, 1.0); // Blue - γ
    }

    // Distance field for 24-cell surface
    float cell24SDF(vec4 p) {
        // 24-cell is intersection of 24 half-spaces
        // Distance to boundary is max over half-space distances
        float d = 0.0;

        // The 24-cell has 24 octahedral cells, each with normal pointing to a vertex
        // Simplified: use |x|+|y|+|z|+|w| = 1 approximation
        d = abs(p.x) + abs(p.y) + abs(p.z) + abs(p.w);

        return d - sqrt(2.0); // Circumradius of 24-cell
    }

    // Distance field for 600-cell surface
    float cell600SDF(vec4 p) {
        // 600-cell has 120 vertices on unit 3-sphere
        // Simplified: use spherical approximation
        return length(p) - 1.0;
    }
`;

// =============================================================================
// INTEGRATION HELPERS
// =============================================================================

/**
 * Create an E8VisualizationRenderer and wire it to existing PPP infrastructure.
 */
export function initializeE8Visualization(gl, pppConfig = {}) {
    const renderer = new E8VisualizationRenderer(gl, pppConfig.e8 || {});

    // Expose on global PPP object if available
    if (typeof window !== 'undefined' && window.PPP) {
        window.PPP.e8 = {
            renderer,
            setMode: (mode) => renderer.setMode(mode),
            getModes: () => E8_VIS_MODES,
            updateBodies: (b1, b2, b3) => renderer.updateThreeBodyState(b1, b2, b3),
            getState: () => renderer.render()
        };
        console.log('[E8Visualization] Exposed PPP.e8 API');
    }

    return renderer;
}

// =============================================================================
// EXPORTS
// =============================================================================

export default E8VisualizationRenderer;
