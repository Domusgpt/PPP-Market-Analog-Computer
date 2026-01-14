/**
 * Trinity Decomposition - Standard Model Particle Mapping
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module implements Ahmed Farag Ali's "Quantum Spacetime Imprints" framework,
 * mapping Standard Model particles to the vertices of the 24-cell polytope.
 *
 * The 24-cell naturally decomposes into:
 * - 1 × 16-cell (8 vertices): Maps to 8 gluons (strong force carriers)
 * - 1 × Tesseract (16 vertices): Maps to fermions + EW bosons + Higgs
 *
 * Alternatively, the 24-cell decomposes into 3 × 16-cells (Trinity):
 * - Set α (8 vertices): Red color charge / Generation 1
 * - Set β (8 vertices): Green color charge / Generation 2
 * - Set γ (8 vertices): Blue color charge / Generation 3
 *
 * The Phillips Synthesis emerges: combining any two 16-cell projections
 * geometrically reveals the third, encoding QCD's color confinement.
 *
 * References:
 * - Ali, A.F. "Quantum Spacetime Imprints" EPJC (2025)
 * - Ali, A.F. "The Mass Gap of the Space-time and its Shape" arXiv:2403.02655 (2024)
 * - Phillips, P.R. "Trinity Dialectic Logic" PPP Framework
 */

import { Vector4D, MATH_CONSTANTS } from '../../types/index.js';
// Note: Lattice24 import removed - using local 24-cell generation

// =============================================================================
// TYPES
// =============================================================================

/** Particle classification */
export type ParticleType = 'fermion' | 'boson' | 'gauge' | 'higgs';

/** Color charge for quarks/gluons */
export type ColorCharge = 'red' | 'green' | 'blue' | 'none' | 'mixed';

/** Particle generation (1, 2, or 3) */
export type Generation = 1 | 2 | 3;

/** Standard Model particle definition */
export interface SMParticle {
    readonly name: string;
    readonly symbol: string;
    readonly type: ParticleType;
    readonly charge: number;        // Electric charge in units of e
    readonly spin: number;          // Spin quantum number
    readonly mass: number;          // Mass in MeV/c²
    readonly color: ColorCharge;
    readonly generation: Generation | null;
    readonly vertexId: number;      // 24-cell vertex mapping
    readonly coordinates: Vector4D; // 4D position in 24-cell
    readonly isAntiparticle: boolean;
}

/** 16-cell subset representing a color/generation */
export interface Cell16Subset {
    readonly label: string;
    readonly color: ColorCharge;
    readonly generation: Generation;
    readonly vertexIds: number[];
    readonly vertices: Vector4D[];
    readonly particles: SMParticle[];
}

/** The Trinity decomposition of a 24-cell */
export interface TrinityDecomposition {
    readonly alpha: Cell16Subset;  // Red / Gen 1
    readonly beta: Cell16Subset;   // Green / Gen 2
    readonly gamma: Cell16Subset;  // Blue / Gen 3
}

/** Ali's primary decomposition: 16-cell + 8-cell */
export interface AliDecomposition {
    readonly gluonCell: {
        readonly vertexIds: number[];
        readonly vertices: Vector4D[];
        readonly particles: SMParticle[];
    };
    readonly matterCell: {
        readonly vertexIds: number[];
        readonly vertices: Vector4D[];
        readonly particles: SMParticle[];
    };
}

// =============================================================================
// STANDARD MODEL PARTICLE DEFINITIONS
// =============================================================================

/**
 * The 8 gluons - mapped to 16-cell (orthoplex) vertices.
 * Gluons carry color-anticolor pairs.
 */
const GLUONS: Omit<SMParticle, 'vertexId' | 'coordinates'>[] = [
    { name: 'Gluon RḠ', symbol: 'g₁', type: 'gauge', charge: 0, spin: 1, mass: 0, color: 'mixed', generation: null, isAntiparticle: false },
    { name: 'Gluon RB̄', symbol: 'g₂', type: 'gauge', charge: 0, spin: 1, mass: 0, color: 'mixed', generation: null, isAntiparticle: false },
    { name: 'Gluon GR̄', symbol: 'g₃', type: 'gauge', charge: 0, spin: 1, mass: 0, color: 'mixed', generation: null, isAntiparticle: false },
    { name: 'Gluon GB̄', symbol: 'g₄', type: 'gauge', charge: 0, spin: 1, mass: 0, color: 'mixed', generation: null, isAntiparticle: false },
    { name: 'Gluon BR̄', symbol: 'g₅', type: 'gauge', charge: 0, spin: 1, mass: 0, color: 'mixed', generation: null, isAntiparticle: false },
    { name: 'Gluon BḠ', symbol: 'g₆', type: 'gauge', charge: 0, spin: 1, mass: 0, color: 'mixed', generation: null, isAntiparticle: false },
    { name: 'Gluon (RR̄-GḠ)/√2', symbol: 'g₇', type: 'gauge', charge: 0, spin: 1, mass: 0, color: 'none', generation: null, isAntiparticle: false },
    { name: 'Gluon (RR̄+GḠ-2BB̄)/√6', symbol: 'g₈', type: 'gauge', charge: 0, spin: 1, mass: 0, color: 'none', generation: null, isAntiparticle: false }
];

/**
 * The 12 elementary fermions (6 quarks + 6 leptons) across 3 generations.
 */
const FERMIONS: Omit<SMParticle, 'vertexId' | 'coordinates'>[] = [
    // Generation 1
    { name: 'Up quark', symbol: 'u', type: 'fermion', charge: 2/3, spin: 0.5, mass: 2.2, color: 'red', generation: 1, isAntiparticle: false },
    { name: 'Down quark', symbol: 'd', type: 'fermion', charge: -1/3, spin: 0.5, mass: 4.7, color: 'red', generation: 1, isAntiparticle: false },
    { name: 'Electron', symbol: 'e⁻', type: 'fermion', charge: -1, spin: 0.5, mass: 0.511, color: 'none', generation: 1, isAntiparticle: false },
    { name: 'Electron neutrino', symbol: 'νₑ', type: 'fermion', charge: 0, spin: 0.5, mass: 0.0000022, color: 'none', generation: 1, isAntiparticle: false },

    // Generation 2
    { name: 'Charm quark', symbol: 'c', type: 'fermion', charge: 2/3, spin: 0.5, mass: 1280, color: 'green', generation: 2, isAntiparticle: false },
    { name: 'Strange quark', symbol: 's', type: 'fermion', charge: -1/3, spin: 0.5, mass: 96, color: 'green', generation: 2, isAntiparticle: false },
    { name: 'Muon', symbol: 'μ⁻', type: 'fermion', charge: -1, spin: 0.5, mass: 105.7, color: 'none', generation: 2, isAntiparticle: false },
    { name: 'Muon neutrino', symbol: 'νμ', type: 'fermion', charge: 0, spin: 0.5, mass: 0.17, color: 'none', generation: 2, isAntiparticle: false },

    // Generation 3
    { name: 'Top quark', symbol: 't', type: 'fermion', charge: 2/3, spin: 0.5, mass: 173100, color: 'blue', generation: 3, isAntiparticle: false },
    { name: 'Bottom quark', symbol: 'b', type: 'fermion', charge: -1/3, spin: 0.5, mass: 4180, color: 'blue', generation: 3, isAntiparticle: false },
    { name: 'Tau', symbol: 'τ⁻', type: 'fermion', charge: -1, spin: 0.5, mass: 1776.8, color: 'none', generation: 3, isAntiparticle: false },
    { name: 'Tau neutrino', symbol: 'ντ', type: 'fermion', charge: 0, spin: 0.5, mass: 15.5, color: 'none', generation: 3, isAntiparticle: false }
];

/**
 * The 4 electroweak bosons + Higgs.
 */
const BOSONS: Omit<SMParticle, 'vertexId' | 'coordinates'>[] = [
    { name: 'Photon', symbol: 'γ', type: 'gauge', charge: 0, spin: 1, mass: 0, color: 'none', generation: null, isAntiparticle: false },
    { name: 'W+ boson', symbol: 'W⁺', type: 'gauge', charge: 1, spin: 1, mass: 80379, color: 'none', generation: null, isAntiparticle: false },
    { name: 'W- boson', symbol: 'W⁻', type: 'gauge', charge: -1, spin: 1, mass: 80379, color: 'none', generation: null, isAntiparticle: true },
    { name: 'Z boson', symbol: 'Z⁰', type: 'gauge', charge: 0, spin: 1, mass: 91188, color: 'none', generation: null, isAntiparticle: false },
    { name: 'Higgs boson', symbol: 'H⁰', type: 'higgs', charge: 0, spin: 0, mass: 125100, color: 'none', generation: null, isAntiparticle: false }
];

// =============================================================================
// 24-CELL VERTEX GENERATION
// =============================================================================

/**
 * Generate the 24 vertices of the 24-cell.
 * Using the standard permutations of (±1, ±1, 0, 0).
 */
function generate24CellVertices(): Vector4D[] {
    const vertices: Vector4D[] = [];

    for (let i = 0; i < 4; i++) {
        for (let j = i + 1; j < 4; j++) {
            for (const si of [-1, 1]) {
                for (const sj of [-1, 1]) {
                    const v: Vector4D = [0, 0, 0, 0];
                    v[i] = si;
                    v[j] = sj;
                    vertices.push(v);
                }
            }
        }
    }

    return vertices;
}

/**
 * Generate the 8 vertices of the 16-cell (cross-polytope).
 * Permutations of (±1, 0, 0, 0).
 */
function generate16CellVertices(): Vector4D[] {
    const vertices: Vector4D[] = [];

    for (let axis = 0; axis < 4; axis++) {
        for (const sign of [-1, 1]) {
            const v: Vector4D = [0, 0, 0, 0];
            v[axis] = sign;
            vertices.push(v);
        }
    }

    return vertices;
}

/**
 * Generate the 16 vertices of the 8-cell (tesseract).
 * Permutations of (±0.5, ±0.5, ±0.5, ±0.5).
 */
function generate8CellVertices(): Vector4D[] {
    const vertices: Vector4D[] = [];
    const h = 0.5;

    for (let mask = 0; mask < 16; mask++) {
        vertices.push([
            (mask & 1) ? h : -h,
            (mask & 2) ? h : -h,
            (mask & 4) ? h : -h,
            (mask & 8) ? h : -h
        ]);
    }

    return vertices;
}

// =============================================================================
// TRINITY DECOMPOSITION (3 × 16-CELL)
// =============================================================================

/**
 * Decompose the 24-cell into three disjoint 16-cells.
 *
 * The 24 vertices split into three sets of 8, where each set forms a 16-cell.
 * This corresponds to:
 * - α (Red): Vertices where sum of non-zero coordinates has specific pattern
 * - β (Green): Second orthogonal subset
 * - γ (Blue): Third orthogonal subset
 *
 * The three 16-cells are related by 120° rotations in 4D.
 */
export function computeTrinityDecomposition(): TrinityDecomposition {
    const vertices = generate24CellVertices();

    // The three 16-cells are defined by specific vertex index patterns
    // Based on coordinate structure in 24-cell

    // Set α: Vertices with coordinates in {X,Y} or {Z,W} planes
    const alphaIds: number[] = [];
    const betaIds: number[] = [];
    const gammaIds: number[] = [];

    for (let i = 0; i < vertices.length; i++) {
        const v = vertices[i];

        // Determine which 16-cell based on which coordinates are non-zero
        const nonZeroIndices: number[] = [];
        for (let j = 0; j < 4; j++) {
            if (Math.abs(v[j]) > 0.5) {
                nonZeroIndices.push(j);
            }
        }

        // Classify into α, β, γ based on index pairs
        // {0,1} or {2,3} → α
        // {0,2} or {1,3} → β
        // {0,3} or {1,2} → γ
        if (nonZeroIndices.length === 2) {
            const [a, b] = nonZeroIndices;
            if ((a === 0 && b === 1) || (a === 2 && b === 3)) {
                alphaIds.push(i);
            } else if ((a === 0 && b === 2) || (a === 1 && b === 3)) {
                betaIds.push(i);
            } else {
                gammaIds.push(i);
            }
        }
    }

    // Create particle mappings for each subset
    const createSubset = (
        ids: number[],
        label: string,
        color: ColorCharge,
        gen: Generation
    ): Cell16Subset => ({
        label,
        color,
        generation: gen,
        vertexIds: ids,
        vertices: ids.map(i => vertices[i]),
        particles: [] // Will be populated below
    });

    return {
        alpha: createSubset(alphaIds, 'α (Red/Gen1)', 'red', 1),
        beta: createSubset(betaIds, 'β (Green/Gen2)', 'green', 2),
        gamma: createSubset(gammaIds, 'γ (Blue/Gen3)', 'blue', 3)
    };
}

// =============================================================================
// ALI DECOMPOSITION (16-CELL + 8-CELL)
// =============================================================================

/**
 * Compute Ali's primary decomposition: 16-cell (gluons) + 8-cell (matter).
 *
 * This mapping assigns:
 * - 8 gluons to the 8 vertices of an inscribed 16-cell
 * - 16 particles (12 fermions + 4 EW bosons) to the 16 vertices of an inscribed 8-cell
 *
 * The geometric containment (16-cell inscribed in 24-cell) provides
 * a natural explanation for color confinement.
 */
export function computeAliDecomposition(): AliDecomposition {
    const vertices16 = generate16CellVertices();
    const vertices8 = generate8CellVertices();

    // Map gluons to 16-cell vertices
    const gluonParticles: SMParticle[] = GLUONS.map((g, i) => ({
        ...g,
        vertexId: i,
        coordinates: vertices16[i]
    }));

    // Map fermions and bosons to 8-cell vertices
    // First 12 vertices: fermions
    // Last 4 vertices: EW bosons
    // (Higgs occupies a special position, often at center)
    const allMatter = [...FERMIONS, ...BOSONS.slice(0, 4)];
    const matterParticles: SMParticle[] = allMatter.map((p, i) => ({
        ...p,
        vertexId: i + 8, // Offset from gluon vertices
        coordinates: vertices8[i % 16]
    }));

    return {
        gluonCell: {
            vertexIds: vertices16.map((_, i) => i),
            vertices: vertices16,
            particles: gluonParticles
        },
        matterCell: {
            vertexIds: vertices8.map((_, i) => i + 8),
            vertices: vertices8,
            particles: matterParticles
        }
    };
}

// =============================================================================
// PARTICLE ASSIGNMENT
// =============================================================================

/**
 * Assign all Standard Model particles to 24-cell vertices.
 * Returns a complete mapping with particle metadata.
 */
export function assignParticlesToVertices(): SMParticle[] {
    const vertices = generate24CellVertices();
    const particles: SMParticle[] = [];

    // Use Ali decomposition for primary assignment
    const ali = computeAliDecomposition();

    // Add all particles from Ali decomposition
    particles.push(...ali.gluonCell.particles);
    particles.push(...ali.matterCell.particles);

    return particles;
}

/**
 * Get particle at a specific 24-cell vertex.
 */
export function getParticleAtVertex(vertexId: number): SMParticle | null {
    const particles = assignParticlesToVertices();
    return particles.find(p => p.vertexId === vertexId) || null;
}

/**
 * Find particles by type.
 */
export function getParticlesByType(type: ParticleType): SMParticle[] {
    return assignParticlesToVertices().filter(p => p.type === type);
}

/**
 * Find particles by generation.
 */
export function getParticlesByGeneration(gen: Generation): SMParticle[] {
    return assignParticlesToVertices().filter(p => p.generation === gen);
}

/**
 * Find particles by color charge.
 */
export function getParticlesByColor(color: ColorCharge): SMParticle[] {
    return assignParticlesToVertices().filter(p => p.color === color);
}

// =============================================================================
// PHILLIPS SYNTHESIS
// =============================================================================

/**
 * The Phillips Synthesis: combining two 16-cell projections reveals the third.
 *
 * Given particles in α and β subsets, compute the interference pattern
 * that encodes the γ subset. This is the geometric mechanism for
 * color confinement - two quarks (thesis + antithesis) require a third
 * color (synthesis) to form a color-neutral hadron.
 *
 * @param alphaState - State in the α (Red) 16-cell
 * @param betaState - State in the β (Green) 16-cell
 * @returns The synthesized γ (Blue) state
 */
export function phillipsSynthesis(
    alphaState: Vector4D,
    betaState: Vector4D
): Vector4D {
    // The synthesis is computed as the geometric complement
    // In the 24-cell, if α and β are known, γ is determined by orthogonality

    // Compute the cross product in 4D (using wedge product properties)
    // The result lies in the γ subspace

    // Simplified version: Find the vertex in γ that balances α and β
    const trinity = computeTrinityDecomposition();

    // Find nearest γ vertex that would create a balanced triad
    let bestGamma: Vector4D = trinity.gamma.vertices[0];
    let bestBalance = Infinity;

    for (const gammaVertex of trinity.gamma.vertices) {
        // Compute the "balance" - how well this creates a color-neutral state
        const centroid: Vector4D = [
            (alphaState[0] + betaState[0] + gammaVertex[0]) / 3,
            (alphaState[1] + betaState[1] + gammaVertex[1]) / 3,
            (alphaState[2] + betaState[2] + gammaVertex[2]) / 3,
            (alphaState[3] + betaState[3] + gammaVertex[3]) / 3
        ];

        const distFromOrigin = Math.sqrt(
            centroid[0]**2 + centroid[1]**2 + centroid[2]**2 + centroid[3]**2
        );

        if (distFromOrigin < bestBalance) {
            bestBalance = distFromOrigin;
            bestGamma = gammaVertex;
        }
    }

    return bestGamma;
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
    GLUONS,
    FERMIONS,
    BOSONS,
    generate24CellVertices,
    generate16CellVertices,
    generate8CellVertices
};
