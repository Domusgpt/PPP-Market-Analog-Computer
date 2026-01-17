/**
 * Trinity Decomposition - Dual Mapping System
 *
 * @package @clear-seas/cpe
 * @version 2.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module implements TWO convergent derivations of the 24-cell decomposition:
 *
 * 1. MUSICAL (Phillips, 2025-2026) - Original derivation for harmonic ambiguity
 *    - α = Thesis / Home / Diatonic (Octatonic Collection I)
 *    - β = Antithesis / Tension / Chromatic (Octatonic Collection II)
 *    - γ = Synthesis / Resolution (Octatonic Collection III)
 *
 * 2. PHYSICS (Ali, 2025) - Convergent validation for particle physics
 *    - α = Red color charge / Generation 1
 *    - β = Green color charge / Generation 2
 *    - γ = Blue color charge / Generation 3
 *
 * The Trinity Index: |F₄|/|B₄| = 1152/384 = 3
 * This index matches: 3 Octatonic collections, 3 fermion generations, 3 color charges
 *
 * The Phillips Synthesis: Given α and β, γ is geometrically determined.
 * - Music: Resolution from tension
 * - Physics: Color confinement (RGB → white)
 * - Logic: Synthesis from thesis/antithesis
 *
 * References:
 * - Phillips, P. "The Trinity Refactor" CPE Documentation (2026)
 * - Ali, A.F. "Quantum Spacetime Imprints" EPJC (2025)
 */

import { Vector4D, MATH_CONSTANTS } from '../../types/index.js';
import { getDefaultLattice, Lattice24 } from './Lattice24.js';

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

/**
 * 8-vertex subset representing a color/generation.
 *
 * NOTE: These are NOT 16-cells (cross-polytopes). Each subset is a
 * duoprism (□ × □) consisting of two orthogonal squares in ℝ⁴.
 * The subsets partition the 24-cell based on coordinate plane pairing:
 * - α: (xy, zw) planes
 * - β: (xz, yw) planes
 * - γ: (xw, yz) planes
 *
 * Historical note: The "Cell16Subset" name is retained for backward
 * compatibility but the geometric structure is a duoprism, not a 16-cell.
 */
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
// MUSICAL TRIALECTIC TYPES (Phillips Original Derivation)
// =============================================================================

/** Musical function in the trialectic system */
export type MusicalFunction = 'thesis' | 'antithesis' | 'synthesis';

/** Harmonic phase state */
export type HarmonicPhase = 'diatonic' | 'chromatic' | 'resolution';

/** Octatonic collection (only 3 exist in 12-tone system) */
export type OctatonicCollection = 1 | 2 | 3;

/**
 * Trinity State Vector - Ψ = [wα, wβ, wγ]
 * Represents the weighted activation of each 16-cell.
 *
 * Interpretation:
 * - [1, 0, 0] = Pure α (stable, diatonic, thesis)
 * - [0.5, 0.5, 0] = Ambiguous (pivoting between α and β)
 * - [0.33, 0.33, 0.33] = Maximum entropy (chromatic chaos, full superposition)
 */
export interface TrinityStateVector {
    readonly alpha: number;  // Weight of α 16-cell
    readonly beta: number;   // Weight of β 16-cell
    readonly gamma: number;  // Weight of γ 16-cell
}

/** Musical 16-cell with dialectic semantics */
export interface Musical16Cell {
    readonly axisPairs: [number, number][];  // Coordinate axis pairs
    readonly dialectic: MusicalFunction;      // Thesis/Antithesis/Synthesis
    readonly harmonic: HarmonicPhase;         // Diatonic/Chromatic/Resolution
    readonly octatonic: OctatonicCollection;  // Octatonic collection I/II/III
    readonly vertices: Vector4D[];
    readonly vertexIds: number[];
}

/** Full musical trialectic decomposition */
export interface MusicalTrialecticDecomposition {
    readonly alpha: Musical16Cell;  // Thesis / Diatonic / OCT-I
    readonly beta: Musical16Cell;   // Antithesis / Chromatic / OCT-II
    readonly gamma: Musical16Cell;  // Synthesis / Resolution / OCT-III
}

/** Phase shift event when crossing between 16-cells */
export interface PhaseShift {
    readonly from: MusicalFunction;
    readonly to: MusicalFunction;
    readonly tension: number;  // Energy barrier (interstice volume)
    readonly type: 'local' | 'modulation' | 'grand_cycle';
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
// TRINITY DECOMPOSITION (3 × 8-VERTEX DUOPRISMS)
// =============================================================================

/**
 * Decompose the 24-cell into three disjoint 8-vertex subsets.
 *
 * MATHEMATICAL NOTE: These subsets are NOT 16-cells (cross-polytopes).
 * Each subset is a duoprism (□ × □) consisting of two orthogonal squares.
 *
 * The decomposition is based on coordinate plane pairing:
 * - α: Vertices in (xy) or (zw) planes → indices (0,1) or (2,3)
 * - β: Vertices in (xz) or (yw) planes → indices (0,2) or (1,3)
 * - γ: Vertices in (xw) or (yz) planes → indices (0,3) or (1,2)
 *
 * This corresponds to the 3 ways to partition {0,1,2,3} into two pairs:
 * - (01|23), (02|13), (03|12)
 *
 * Physical/Musical interpretation:
 * - α (Red): Color charge red / Generation 1 / Octatonic I
 * - β (Green): Color charge green / Generation 2 / Octatonic II
 * - γ (Blue): Color charge blue / Generation 3 / Octatonic III
 *
 * The three subsets are related by coordinate permutations, not 120° rotations.
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
// MUSICAL TRIALECTIC DECOMPOSITION
// =============================================================================

/**
 * Compute the musical trialectic decomposition.
 * This is the Phillips original derivation from music theory.
 *
 * The axis pairs correspond to orthogonal planes in 4D:
 * - α: {xy, zw} planes → Thesis / Diatonic / Octatonic I
 * - β: {xz, yw} planes → Antithesis / Chromatic / Octatonic II
 * - γ: {xw, yz} planes → Synthesis / Resolution / Octatonic III
 */
export function computeMusicalTrialectic(): MusicalTrialecticDecomposition {
    const trinity = computeTrinityDecomposition();

    return {
        alpha: {
            axisPairs: [[0, 1], [2, 3]],
            dialectic: 'thesis',
            harmonic: 'diatonic',
            octatonic: 1,
            vertices: trinity.alpha.vertices,
            vertexIds: trinity.alpha.vertexIds
        },
        beta: {
            axisPairs: [[0, 2], [1, 3]],
            dialectic: 'antithesis',
            harmonic: 'chromatic',
            octatonic: 2,
            vertices: trinity.beta.vertices,
            vertexIds: trinity.beta.vertexIds
        },
        gamma: {
            axisPairs: [[0, 3], [1, 2]],
            dialectic: 'synthesis',
            harmonic: 'resolution',
            octatonic: 3,
            vertices: trinity.gamma.vertices,
            vertexIds: trinity.gamma.vertexIds
        }
    };
}

/**
 * Compute the Trinity State Vector Ψ = [wα, wβ, wγ] for a 4D point.
 *
 * The weights are computed based on distance to each 16-cell.
 * Closer to a 16-cell = higher weight for that component.
 *
 * @param point - 4D position in the 24-cell space
 * @returns Normalized Trinity State Vector (components sum to 1)
 */
export function computeTrinityStateVector(point: Vector4D): TrinityStateVector {
    const trinity = computeTrinityDecomposition();

    // Compute minimum distance to each 16-cell
    const distToCell = (cell: Cell16Subset): number => {
        let minDist = Infinity;
        for (const vertex of cell.vertices) {
            const dist = Math.sqrt(
                (point[0] - vertex[0]) ** 2 +
                (point[1] - vertex[1]) ** 2 +
                (point[2] - vertex[2]) ** 2 +
                (point[3] - vertex[3]) ** 2
            );
            if (dist < minDist) minDist = dist;
        }
        return minDist;
    };

    const dAlpha = distToCell(trinity.alpha);
    const dBeta = distToCell(trinity.beta);
    const dGamma = distToCell(trinity.gamma);

    // Convert distances to weights (inverse relationship)
    // Using softmax-like transformation for smooth weights
    const epsilon = 0.001;
    const wAlpha = 1 / (dAlpha + epsilon);
    const wBeta = 1 / (dBeta + epsilon);
    const wGamma = 1 / (dGamma + epsilon);

    // Normalize to sum to 1
    const total = wAlpha + wBeta + wGamma;

    return {
        alpha: wAlpha / total,
        beta: wBeta / total,
        gamma: wGamma / total
    };
}

/**
 * Classify the Trinity state based on the state vector.
 *
 * @param state - Trinity State Vector
 * @returns Classification of the harmonic state
 */
export function classifyTrinityState(state: TrinityStateVector): {
    dominant: MusicalFunction | 'ambiguous' | 'superposition';
    entropy: number;
    description: string;
} {
    const { alpha, beta, gamma } = state;

    // Compute entropy (Shannon) - higher = more ambiguous
    const entropy = -[alpha, beta, gamma]
        .filter(w => w > 0)
        .reduce((sum, w) => sum + w * Math.log2(w), 0);

    // Maximum entropy for 3 components is log2(3) ≈ 1.585
    const maxEntropy = Math.log2(3);
    const normalizedEntropy = entropy / maxEntropy;

    // Determine dominant state
    const threshold = 0.5;
    let dominant: MusicalFunction | 'ambiguous' | 'superposition';
    let description: string;

    if (alpha > threshold) {
        dominant = 'thesis';
        description = 'Stable diatonic state (Home key)';
    } else if (beta > threshold) {
        dominant = 'antithesis';
        description = 'Chromatic tension (Modulation in progress)';
    } else if (gamma > threshold) {
        dominant = 'synthesis';
        description = 'Resolution achieved';
    } else if (normalizedEntropy > 0.9) {
        dominant = 'superposition';
        description = 'Maximum ambiguity (Full chromatic)';
    } else {
        dominant = 'ambiguous';
        description = 'Harmonic pivot point';
    }

    return { dominant, entropy: normalizedEntropy, description };
}

/**
 * Detect phase shift between two Trinity states.
 *
 * @param from - Previous state
 * @param to - Current state
 * @returns Phase shift information or null if no shift
 */
export function detectPhaseShift(
    from: TrinityStateVector,
    to: TrinityStateVector
): PhaseShift | null {
    const fromClass = classifyTrinityState(from);
    const toClass = classifyTrinityState(to);

    // No shift if same dominant state
    if (fromClass.dominant === toClass.dominant) return null;

    // Can't have shift from/to ambiguous states (not a discrete phase)
    if (fromClass.dominant === 'ambiguous' ||
        fromClass.dominant === 'superposition' ||
        toClass.dominant === 'ambiguous' ||
        toClass.dominant === 'superposition') {
        return null;
    }

    // Compute tension as the entropy change
    const tension = Math.abs(toClass.entropy - fromClass.entropy);

    // Classify the shift type
    let type: 'local' | 'modulation' | 'grand_cycle';

    // Grand cycle: α → β → γ → α (or reverse)
    const isGrandCycle =
        (fromClass.dominant === 'thesis' && toClass.dominant === 'antithesis') ||
        (fromClass.dominant === 'antithesis' && toClass.dominant === 'synthesis') ||
        (fromClass.dominant === 'synthesis' && toClass.dominant === 'thesis');

    if (isGrandCycle && tension > 0.3) {
        type = 'grand_cycle';
    } else if (tension > 0.2) {
        type = 'modulation';
    } else {
        type = 'local';
    }

    return {
        from: fromClass.dominant as MusicalFunction,
        to: toClass.dominant as MusicalFunction,
        tension,
        type
    };
}

/**
 * Map three bodies to the Trinity 16-cells.
 * Each body is assigned to one 16-cell based on its phase space position.
 *
 * This creates the fundamental correspondence:
 * - Body 1 → α (Thesis)
 * - Body 2 → β (Antithesis)
 * - Body 3 → γ (Synthesis)
 *
 * When all three bodies are in their designated cells, the system is "locked"
 * (stable periodic orbit). When bodies cross into other cells, the system
 * exhibits chaotic behavior.
 *
 * @param body1Pos - 4D position of body 1
 * @param body2Pos - 4D position of body 2
 * @param body3Pos - 4D position of body 3
 * @returns Trinity state vectors for each body and system stability
 */
export function mapThreeBodiesToTrinity(
    body1Pos: Vector4D,
    body2Pos: Vector4D,
    body3Pos: Vector4D
): {
    body1State: TrinityStateVector;
    body2State: TrinityStateVector;
    body3State: TrinityStateVector;
    isLocked: boolean;
    lockingScore: number;
} {
    const body1State = computeTrinityStateVector(body1Pos);
    const body2State = computeTrinityStateVector(body2Pos);
    const body3State = computeTrinityStateVector(body3Pos);

    // System is "locked" when each body dominates its assigned cell
    // Body1 → α, Body2 → β, Body3 → γ
    const lockingScore = (
        body1State.alpha +
        body2State.beta +
        body3State.gamma
    ) / 3;

    const isLocked = lockingScore > 0.6;

    return {
        body1State,
        body2State,
        body3State,
        isLocked,
        lockingScore
    };
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
