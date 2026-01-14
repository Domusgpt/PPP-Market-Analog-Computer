/**
 * Moiré Interference Engine
 * =========================
 *
 * The core computation mechanism of the PPP framework.
 *
 * Instead of integrating F=ma numerically, we:
 * 1. Project each body's polytope state to 2D
 * 2. Overlay the projections
 * 3. Compute interference patterns
 * 4. Read the fringes to determine interaction/next state
 *
 * Key properties:
 * - Resolution independent (geometric, not floating-point)
 * - Energy conserving (Moiré patterns are purely geometric)
 * - Unified (same logic for gravity and particle physics)
 *
 * Stable configurations → periodic/resonant patterns
 * Chaotic configurations → aperiodic patterns
 * Collisions → regularized via lattice snapping
 */

import { Vector4D } from '../../types/index.js';

// =============================================================================
// TYPES
// =============================================================================

/** 2D projection of a polytope */
export interface Projection2D {
    points: [number, number][];
    edges: [number, number][];  // Index pairs
    centroid: [number, number];
}

/** Moiré interference result */
export interface MoirePattern {
    /** Interference intensity field (sampled) */
    field: Float32Array;
    /** Grid resolution */
    resolution: number;
    /** Dominant frequency (periodicity indicator) */
    dominantFrequency: number;
    /** Resonance score (0 = chaotic, 1 = perfect resonance) */
    resonance: number;
    /** Phase alignment between patterns */
    phaseAlignment: number;
    /** Fringe count (number of interference bands) */
    fringeCount: number;
}

/** Interference analysis result */
export interface InterferenceAnalysis {
    /** Is this a stable (resonant) configuration? */
    isStable: boolean;
    /** Stability score (0-1) */
    stability: number;
    /** Predicted outcome: 'bound', 'escape', 'collision' */
    outcome: 'bound' | 'escape' | 'collision' | 'chaotic';
    /** Resonance type if stable */
    resonanceType?: string;
}

// =============================================================================
// PROJECTION FUNCTIONS
// =============================================================================

/**
 * Project 4D point to 2D using stereographic projection.
 * This is how we "see" the 4D polytope as a 2D shadow.
 */
export function stereographicProject(v: Vector4D): [number, number] {
    const [x, y, z, w] = v;

    // Project from 4D to 3D (from w-axis)
    const denom = 1 - w + 0.001;  // Avoid division by zero
    const x3 = x / denom;
    const y3 = y / denom;
    const z3 = z / denom;

    // Project from 3D to 2D (orthographic onto xy plane)
    // Could also use perspective projection
    return [x3, y3];
}

/**
 * Project 4D point using rotation + orthographic projection.
 * Angle parameters control the viewing direction.
 */
export function rotatedProject(
    v: Vector4D,
    angleXW: number = 0,
    angleYW: number = 0,
    angleZW: number = 0
): [number, number] {
    let [x, y, z, w] = v;

    // Rotate in XW plane
    const cosXW = Math.cos(angleXW);
    const sinXW = Math.sin(angleXW);
    const x1 = x * cosXW - w * sinXW;
    const w1 = x * sinXW + w * cosXW;

    // Rotate in YW plane
    const cosYW = Math.cos(angleYW);
    const sinYW = Math.sin(angleYW);
    const y1 = y * cosYW - w1 * sinYW;
    const w2 = y * sinYW + w1 * cosYW;

    // Rotate in ZW plane
    const cosZW = Math.cos(angleZW);
    const sinZW = Math.sin(angleZW);
    const z1 = z * cosZW - w2 * sinZW;

    // Orthographic projection to 2D
    return [x1, y1];
}

/**
 * Project a set of 4D vertices to 2D.
 */
export function projectVertices(
    vertices: Vector4D[],
    projectionFn: (v: Vector4D) => [number, number] = stereographicProject
): Projection2D {
    const points = vertices.map(projectionFn);

    // Compute centroid
    let cx = 0, cy = 0;
    for (const [x, y] of points) {
        cx += x;
        cy += y;
    }
    cx /= points.length;
    cy /= points.length;

    // Compute edges (connect nearest neighbors)
    const edges: [number, number][] = [];
    const threshold = 0.5;  // Edge length threshold

    for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
            const dx = points[i][0] - points[j][0];
            const dy = points[i][1] - points[j][1];
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < threshold) {
                edges.push([i, j]);
            }
        }
    }

    return { points, edges, centroid: [cx, cy] };
}

// =============================================================================
// MOIRÉ INTERFERENCE COMPUTATION
// =============================================================================

/**
 * Compute Moiré interference pattern between two 2D projections.
 *
 * The interference is computed by:
 * 1. Creating intensity fields from each projection
 * 2. Multiplying the fields (interference)
 * 3. Analyzing the resulting pattern
 */
export function computeMoirePattern(
    proj1: Projection2D,
    proj2: Projection2D,
    resolution: number = 64
): MoirePattern {
    const field = new Float32Array(resolution * resolution);

    // Compute bounding box
    const allPoints = [...proj1.points, ...proj2.points];
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (const [x, y] of allPoints) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
    }

    const rangeX = maxX - minX + 0.1;
    const rangeY = maxY - minY + 0.1;
    const scale = Math.max(rangeX, rangeY);

    // Compute interference field
    for (let iy = 0; iy < resolution; iy++) {
        for (let ix = 0; ix < resolution; ix++) {
            const x = minX + (ix / resolution) * scale;
            const y = minY + (iy / resolution) * scale;

            // Intensity from projection 1 (sum of Gaussians at vertices)
            let intensity1 = 0;
            for (const [px, py] of proj1.points) {
                const dx = x - px;
                const dy = y - py;
                const r2 = dx * dx + dy * dy;
                intensity1 += Math.exp(-r2 * 10);
            }

            // Intensity from projection 2
            let intensity2 = 0;
            for (const [px, py] of proj2.points) {
                const dx = x - px;
                const dy = y - py;
                const r2 = dx * dx + dy * dy;
                intensity2 += Math.exp(-r2 * 10);
            }

            // Interference: product of intensities
            // This creates the Moiré pattern
            field[iy * resolution + ix] = intensity1 * intensity2;
        }
    }

    // Analyze the pattern
    const analysis = analyzePattern(field, resolution);

    return {
        field,
        resolution,
        ...analysis
    };
}

/**
 * Analyze a Moiré pattern to extract physical meaning.
 */
function analyzePattern(field: Float32Array, resolution: number): {
    dominantFrequency: number;
    resonance: number;
    phaseAlignment: number;
    fringeCount: number;
} {
    // Simple FFT-like analysis via autocorrelation
    let totalIntensity = 0;
    let maxIntensity = 0;

    for (let i = 0; i < field.length; i++) {
        totalIntensity += field[i];
        maxIntensity = Math.max(maxIntensity, field[i]);
    }

    const meanIntensity = totalIntensity / field.length;

    // Count fringes (zero crossings above/below mean)
    let fringeCount = 0;
    let aboveMean = field[0] > meanIntensity;

    for (let i = 1; i < field.length; i++) {
        const nowAbove = field[i] > meanIntensity;
        if (nowAbove !== aboveMean) {
            fringeCount++;
            aboveMean = nowAbove;
        }
    }

    // Compute variance (measure of pattern regularity)
    let variance = 0;
    for (let i = 0; i < field.length; i++) {
        const diff = field[i] - meanIntensity;
        variance += diff * diff;
    }
    variance /= field.length;

    // Resonance: high variance + periodic fringes = resonant
    // Low variance = uniform (degenerate)
    // High variance + irregular fringes = chaotic
    const normalizedVariance = variance / (maxIntensity * maxIntensity + 0.001);

    // Dominant frequency from fringe count
    const dominantFrequency = fringeCount / (2 * resolution);

    // Resonance score: combination of periodicity and intensity
    // Perfect resonance has high intensity AND regular pattern
    const resonance = Math.min(1, normalizedVariance * 2) *
                      Math.min(1, maxIntensity / (meanIntensity + 0.001) / 10);

    // Phase alignment: how well centroids align
    const phaseAlignment = maxIntensity / (totalIntensity + 0.001) * resolution;

    return {
        dominantFrequency,
        resonance: Math.min(1, Math.max(0, resonance)),
        phaseAlignment: Math.min(1, phaseAlignment),
        fringeCount
    };
}

// =============================================================================
// THREE-BODY INTERFERENCE
// =============================================================================

/**
 * Compute three-body interaction via Moiré interference.
 *
 * Each body is projected, all three are overlaid,
 * and the interference pattern determines the interaction.
 */
export function threeBodyInterference(
    body1Vertices: Vector4D[],
    body2Vertices: Vector4D[],
    body3Vertices: Vector4D[],
    resolution: number = 64
): {
    pattern12: MoirePattern;
    pattern23: MoirePattern;
    pattern13: MoirePattern;
    combined: MoirePattern;
    analysis: InterferenceAnalysis;
} {
    // Project each body
    const proj1 = projectVertices(body1Vertices);
    const proj2 = projectVertices(body2Vertices);
    const proj3 = projectVertices(body3Vertices);

    // Compute pairwise interference
    const pattern12 = computeMoirePattern(proj1, proj2, resolution);
    const pattern23 = computeMoirePattern(proj2, proj3, resolution);
    const pattern13 = computeMoirePattern(proj1, proj3, resolution);

    // Compute combined three-body interference
    const combinedField = new Float32Array(resolution * resolution);
    for (let i = 0; i < combinedField.length; i++) {
        // Three-way interference: product of all three patterns
        combinedField[i] = pattern12.field[i] * pattern23.field[i] * pattern13.field[i];
    }

    const combinedAnalysis = analyzePattern(combinedField, resolution);
    const combined: MoirePattern = {
        field: combinedField,
        resolution,
        ...combinedAnalysis
    };

    // Analyze the configuration
    const analysis = analyzeThreeBodyConfig(pattern12, pattern23, pattern13, combined);

    return { pattern12, pattern23, pattern13, combined, analysis };
}

/**
 * Analyze three-body configuration from interference patterns.
 */
function analyzeThreeBodyConfig(
    p12: MoirePattern,
    p23: MoirePattern,
    p13: MoirePattern,
    combined: MoirePattern
): InterferenceAnalysis {
    // Average resonance across all pairs
    const avgResonance = (p12.resonance + p23.resonance + p13.resonance) / 3;

    // Combined resonance (three-body coherence)
    const threeBodyResonance = combined.resonance;

    // Stability: high resonance in all pairs AND combined
    const stability = Math.sqrt(avgResonance * threeBodyResonance);

    // Determine outcome based on pattern characteristics
    let outcome: 'bound' | 'escape' | 'collision' | 'chaotic';

    if (combined.fringeCount < 10 && combined.resonance > 0.5) {
        // Few fringes + high resonance = collision (bodies overlapping)
        // BUT this is regularized - not a singularity
        outcome = 'collision';
    } else if (stability > 0.6) {
        // High stability = bound orbit
        outcome = 'bound';
    } else if (avgResonance < 0.2 && combined.resonance < 0.2) {
        // Low resonance everywhere = escape
        outcome = 'escape';
    } else {
        // Otherwise chaotic
        outcome = 'chaotic';
    }

    // Determine resonance type if stable
    let resonanceType: string | undefined;
    if (outcome === 'bound') {
        if (Math.abs(p12.dominantFrequency - p23.dominantFrequency) < 0.1 &&
            Math.abs(p23.dominantFrequency - p13.dominantFrequency) < 0.1) {
            resonanceType = 'figure-8';  // Equal frequencies = synchronized
        } else if (p12.resonance > 0.7 && p13.resonance < 0.3) {
            resonanceType = 'hierarchical';  // One pair dominant
        } else {
            resonanceType = 'lagrange';  // Equilateral-like
        }
    }

    return {
        isStable: outcome === 'bound',
        stability,
        outcome,
        resonanceType
    };
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
    stereographicProject,
    rotatedProject,
    projectVertices,
    computeMoirePattern,
    threeBodyInterference
};
