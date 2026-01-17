/**
 * @clear-seas/cpe Topology Module
 *
 * Re-exports topological structures for the Chronomorphic Polytopal Engine.
 */

// Original PPP topology (named exports to avoid computeCoherence conflict with Epistaorthognition)
export {
    Lattice24,
    getDefaultLattice,
    createLattice,
    isInsideConvexHull,
    findNearestVertex,
    findKNearestVertices,
    checkConvexity,
    projectToConvexHull,
    CIRCUMRADIUS,
    EDGE_LENGTH,
    INRADIUS,
    NUM_VERTICES,
    NUM_EDGES
} from './Lattice24.js';
export * from './Lattice24Provider.js';
export * from './Simplex5.js';
export * from './Hypercube8.js';
export * from './TopologyController.js';

// 3-Body E8/H4 extension (named exports to avoid conflicts)
export {
    // E8H4Folding
    type Vector8D,
    type Matrix8x8,
    type FoldingMatrix4x8,
    type Chirality,
    type GoldenScale,
    type H4Copy,
    type FoldingResult,
    createMoxnessMatrix,
    createFoldingMatrix,
    generateE8Roots,
    applyMoxnessMatrix,
    extractH4Left,
    extractH4Right,
    foldE8toH4,
    projectE8to4D,
    projectAllE8to4D,
    norm8D,
    dot8D,
    normalize8D,
    verifyUnimodularity
} from './E8H4Folding.js';

export {
    // Lattice600
    Lattice600,
    type Lattice600Vertex,
    type Cell24Subset,
    type Lattice600Structure,
    type Edge,
    generate600CellVerticesIcosian,
    getDefaultLattice600,
    createLattice600
} from './Lattice600.js';

export {
    // ThreeBodyPhaseSpace
    type Vector3D,
    type BodyState,
    type ThreeBodyState,
    type JacobiCoordinates,
    type ShapeSpherePoint,
    type ReducedPhasePoint,
    type Cell600Mapping,
    toJacobiCoordinates,
    toShapeSphere,
    encodeToPhaseSpace,
    decodeFromPhaseSpace,
    mapTo600Cell,
    createFigure8Orbit,
    createLagrangeOrbit,
    createEulerOrbit,
    computeEnergy,
    computeAngularMomentum,
    computeGravitationalForce
} from './ThreeBodyPhaseSpace.js';

export {
    // TrinityDecomposition - Physics types (Ali)
    type ParticleType,
    type ColorCharge,
    type Generation,
    type SMParticle,
    type Cell16Subset,
    type TrinityDecomposition,
    type AliDecomposition,

    // TrinityDecomposition - Musical types (Phillips)
    type MusicalFunction,
    type HarmonicPhase,
    type OctatonicCollection,
    type TrinityStateVector,
    type Musical16Cell,
    type MusicalTrialecticDecomposition,
    type PhaseShift,

    // Physics functions (Ali)
    computeTrinityDecomposition,
    computeAliDecomposition,
    phillipsSynthesis,
    assignParticlesToVertices,
    getParticleAtVertex,
    getParticlesByType,
    getParticlesByGeneration,
    getParticlesByColor,

    // Musical functions (Phillips)
    computeMusicalTrialectic,
    computeTrinityStateVector,
    classifyTrinityState,
    detectPhaseShift,
    mapThreeBodiesToTrinity,

    // Vertex generation
    generate24CellVertices,
    generate16CellVertices,
    generate8CellVertices,
    GLUONS,
    FERMIONS,
    BOSONS
} from './TrinityDecomposition.js';
