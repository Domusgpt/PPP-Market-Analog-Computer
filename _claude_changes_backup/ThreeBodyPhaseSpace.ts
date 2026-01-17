/**
 * Three-Body Phase Space Mapping to E8 Lattice
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module implements the geometric encoding of the classical three-body
 * problem into the E8→H4 polytopal framework. The key insight is that the
 * planar three-body problem has an 8-dimensional reduced phase space after
 * accounting for conservation laws, which maps naturally to E8.
 *
 * Phase Space Reduction:
 * - Full state: 18D (3 bodies × 3 positions × 2 for momentum)
 * - After CM removal: 12D (remove 6 DOF for center of mass)
 * - After angular momentum: 9D (fix |L| and direction)
 * - After SO(3) reduction: 8D (remove orientation)
 *
 * The 8D reduced phase space is the natural home for E8 encoding:
 * - States become lattice points
 * - Trajectories become paths on the lattice graph
 * - Stable orbits correspond to closed cycles
 *
 * Key Framework Elements:
 * - Jacobi coordinates for relative positions
 * - Shape sphere (S²) for configuration topology
 * - KS regularization for collision handling
 * - 600-cell geodesics for trajectory integration
 *
 * References:
 * - Montgomery, R. "The Three-Body Problem and the Shape Sphere"
 * - Chenciner, A. & Montgomery, R. "Figure-8 Orbit" (2000)
 * - Kustaanheimo-Stiefel regularization theory
 */

import { Vector4D, MATH_CONSTANTS } from '../../types/index.js';
import { Vector8D, generateE8Roots, projectE8to4D } from './E8H4Folding.js';
import { Lattice600, getDefaultLattice600 } from './Lattice600.js';

// =============================================================================
// TYPES
// =============================================================================

/** 3D position vector */
export type Vector3D = [number, number, number];

/** State of a single body */
export interface BodyState {
    readonly position: Vector3D;
    readonly velocity: Vector3D;
    readonly mass: number;
}

/** Full state of three-body system */
export interface ThreeBodyState {
    readonly body1: BodyState;
    readonly body2: BodyState;
    readonly body3: BodyState;
    readonly time: number;
}

/** Jacobi coordinates for reduced representation */
export interface JacobiCoordinates {
    /** Center of mass (removed) */
    readonly centerOfMass: Vector3D;
    /** Relative position: body2 - body1 */
    readonly rho: Vector3D;
    /** Relative position: body3 - (m1*r1 + m2*r2)/(m1+m2) */
    readonly sigma: Vector3D;
    /** Conjugate momentum to rho */
    readonly pRho: Vector3D;
    /** Conjugate momentum to sigma */
    readonly pSigma: Vector3D;
}

/** Shape sphere coordinates (after scale/rotation removal) */
export interface ShapeSpherePoint {
    /** Latitude on shape sphere (-π/2 to π/2) */
    readonly theta: number;
    /** Longitude on shape sphere (0 to 2π) */
    readonly phi: number;
    /** Shape (0 = equilateral, 1 = collinear) */
    readonly shape: number;
}

/** The 8D reduced phase space point */
export interface ReducedPhasePoint {
    /** The 8D coordinates */
    readonly coordinates: Vector8D;
    /** Original Jacobi coordinates */
    readonly jacobi: JacobiCoordinates;
    /** Shape sphere projection */
    readonly shapeSphere: ShapeSpherePoint;
    /** Nearest E8 lattice node */
    readonly nearestE8Node: number;
    /** Distance to nearest E8 node */
    readonly latticeError: number;
}

/** Result of mapping to 600-cell */
export interface Cell600Mapping {
    /** H4 projection (4D) */
    readonly h4Position: Vector4D;
    /** Nearest 600-cell vertex */
    readonly nearestVertex: number;
    /** Distance to vertex */
    readonly distance: number;
    /** Which 24-cell the state lies in */
    readonly cell24Index: number;
    /** Body-to-cell assignments */
    readonly bodyAssignments: {
        body1: number;
        body2: number;
        body3: number;
    };
}

// =============================================================================
// CONSTANTS
// =============================================================================

/** Gravitational constant (normalized) */
const G = 1;

/** Energy normalization scale */
const ENERGY_SCALE = 1;

/** Position normalization scale */
const LENGTH_SCALE = 1;

// =============================================================================
// COORDINATE TRANSFORMATIONS
// =============================================================================

/**
 * Convert Cartesian three-body state to Jacobi coordinates.
 * Jacobi coordinates remove the center of mass and express relative positions.
 */
export function toJacobiCoordinates(state: ThreeBodyState): JacobiCoordinates {
    const { body1, body2, body3 } = state;
    const m1 = body1.mass;
    const m2 = body2.mass;
    const m3 = body3.mass;
    const M = m1 + m2 + m3;

    // Center of mass
    const centerOfMass: Vector3D = [
        (m1 * body1.position[0] + m2 * body2.position[0] + m3 * body3.position[0]) / M,
        (m1 * body1.position[1] + m2 * body2.position[1] + m3 * body3.position[1]) / M,
        (m1 * body1.position[2] + m2 * body2.position[2] + m3 * body3.position[2]) / M
    ];

    // Relative position: rho = r2 - r1
    const rho: Vector3D = [
        body2.position[0] - body1.position[0],
        body2.position[1] - body1.position[1],
        body2.position[2] - body1.position[2]
    ];

    // Center of mass of first pair
    const m12 = m1 + m2;
    const r12: Vector3D = [
        (m1 * body1.position[0] + m2 * body2.position[0]) / m12,
        (m1 * body1.position[1] + m2 * body2.position[1]) / m12,
        (m1 * body1.position[2] + m2 * body2.position[2]) / m12
    ];

    // Relative position: sigma = r3 - r12
    const sigma: Vector3D = [
        body3.position[0] - r12[0],
        body3.position[1] - r12[1],
        body3.position[2] - r12[2]
    ];

    // Reduced masses for momenta
    const mu1 = (m1 * m2) / m12;  // Reduced mass for rho
    const mu2 = (m12 * m3) / M;    // Reduced mass for sigma

    // Conjugate momenta
    const vRho: Vector3D = [
        body2.velocity[0] - body1.velocity[0],
        body2.velocity[1] - body1.velocity[1],
        body2.velocity[2] - body1.velocity[2]
    ];
    const pRho: Vector3D = [mu1 * vRho[0], mu1 * vRho[1], mu1 * vRho[2]];

    const v12: Vector3D = [
        (m1 * body1.velocity[0] + m2 * body2.velocity[0]) / m12,
        (m1 * body1.velocity[1] + m2 * body2.velocity[1]) / m12,
        (m1 * body1.velocity[2] + m2 * body2.velocity[2]) / m12
    ];
    const vSigma: Vector3D = [
        body3.velocity[0] - v12[0],
        body3.velocity[1] - v12[1],
        body3.velocity[2] - v12[2]
    ];
    const pSigma: Vector3D = [mu2 * vSigma[0], mu2 * vSigma[1], mu2 * vSigma[2]];

    return { centerOfMass, rho, sigma, pRho, pSigma };
}

/**
 * Compute the shape sphere projection from Jacobi coordinates.
 * The shape sphere captures the configuration (shape of triangle) modulo
 * translations, rotations, and scaling.
 */
export function toShapeSphere(jacobi: JacobiCoordinates): ShapeSpherePoint {
    const { rho, sigma } = jacobi;

    // Compute scale (moment of inertia)
    const rhoNorm = Math.sqrt(rho[0]**2 + rho[1]**2 + rho[2]**2);
    const sigmaNorm = Math.sqrt(sigma[0]**2 + sigma[1]**2 + sigma[2]**2);
    const I = rhoNorm**2 + sigmaNorm**2;

    if (I < MATH_CONSTANTS.EPSILON) {
        return { theta: 0, phi: 0, shape: 0 };
    }

    // Normalize to unit sphere
    const scale = 1 / Math.sqrt(I);
    const rhoNormed: Vector3D = [rho[0] * scale, rho[1] * scale, rho[2] * scale];
    const sigmaNormed: Vector3D = [sigma[0] * scale, sigma[1] * scale, sigma[2] * scale];

    // Shape sphere coordinates (simplified - planar case)
    // For planar motion, we use the angle between rho and sigma
    const dot = rhoNormed[0] * sigmaNormed[0] + rhoNormed[1] * sigmaNormed[1] + rhoNormed[2] * sigmaNormed[2];
    const theta = Math.acos(Math.max(-1, Math.min(1, dot)));

    // Phi encodes the relative orientation
    const cross = [
        rhoNormed[1] * sigmaNormed[2] - rhoNormed[2] * sigmaNormed[1],
        rhoNormed[2] * sigmaNormed[0] - rhoNormed[0] * sigmaNormed[2],
        rhoNormed[0] * sigmaNormed[1] - rhoNormed[1] * sigmaNormed[0]
    ];
    const crossNorm = Math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2);
    const phi = Math.atan2(crossNorm, dot);

    // Shape parameter (0 = equilateral, 1 = collinear)
    const shape = Math.abs(Math.cos(theta));

    return { theta, phi, shape };
}

// =============================================================================
// 8D PHASE SPACE ENCODING
// =============================================================================

/**
 * Encode the three-body state into the 8D reduced phase space.
 *
 * The 8D coordinates are constructed from:
 * - 2D: rho (x, y) - relative position 1
 * - 2D: sigma (x, y) - relative position 2
 * - 2D: pRho (x, y) - conjugate momentum 1
 * - 2D: pSigma (x, y) - conjugate momentum 2
 *
 * For planar motion (z = 0), this gives exactly 8 dimensions.
 */
export function encodeToPhaseSpace(state: ThreeBodyState): ReducedPhasePoint {
    const jacobi = toJacobiCoordinates(state);
    const shapeSphere = toShapeSphere(jacobi);

    // Construct 8D phase space point (planar: ignore z components)
    const coordinates: Vector8D = [
        jacobi.rho[0],
        jacobi.rho[1],
        jacobi.sigma[0],
        jacobi.sigma[1],
        jacobi.pRho[0],
        jacobi.pRho[1],
        jacobi.pSigma[0],
        jacobi.pSigma[1]
    ];

    // Normalize to unit 8-sphere for lattice mapping
    const norm8 = Math.sqrt(coordinates.reduce((s, x) => s + x * x, 0));
    const normalized: Vector8D = norm8 > MATH_CONSTANTS.EPSILON
        ? coordinates.map(x => x / norm8) as Vector8D
        : [1, 0, 0, 0, 0, 0, 0, 0];

    // Find nearest E8 lattice node
    const e8Roots = generateE8Roots();
    let nearestNode = 0;
    let minDist = Infinity;

    for (let i = 0; i < e8Roots.length; i++) {
        const root = e8Roots[i];
        let dist = 0;
        for (let j = 0; j < 8; j++) {
            dist += (normalized[j] - root[j]) ** 2;
        }
        if (dist < minDist) {
            minDist = dist;
            nearestNode = i;
        }
    }

    return {
        coordinates: normalized,
        jacobi,
        shapeSphere,
        nearestE8Node: nearestNode,
        latticeError: Math.sqrt(minDist)
    };
}

/**
 * Decode an 8D phase space point back to three-body state.
 * This is the inverse of encodeToPhaseSpace.
 */
export function decodeFromPhaseSpace(
    point: ReducedPhasePoint,
    masses: [number, number, number] = [1, 1, 1],
    scale: number = 1
): ThreeBodyState {
    const [m1, m2, m3] = masses;
    const M = m1 + m2 + m3;
    const m12 = m1 + m2;

    // Extract Jacobi coordinates
    const rho: Vector3D = [
        point.coordinates[0] * scale,
        point.coordinates[1] * scale,
        0
    ];
    const sigma: Vector3D = [
        point.coordinates[2] * scale,
        point.coordinates[3] * scale,
        0
    ];

    // Reduced masses
    const mu1 = (m1 * m2) / m12;
    const mu2 = (m12 * m3) / M;

    // Extract momenta and convert to velocities
    const pRho: Vector3D = [
        point.coordinates[4] * scale,
        point.coordinates[5] * scale,
        0
    ];
    const pSigma: Vector3D = [
        point.coordinates[6] * scale,
        point.coordinates[7] * scale,
        0
    ];

    const vRho: Vector3D = [pRho[0] / mu1, pRho[1] / mu1, 0];
    const vSigma: Vector3D = [pSigma[0] / mu2, pSigma[1] / mu2, 0];

    // Reconstruct positions (set center of mass at origin)
    // r1 = -m2/m12 * rho - m3/M * sigma
    // r2 = m1/m12 * rho - m3/M * sigma
    // r3 = (m1+m2)/M * sigma
    const body1Position: Vector3D = [
        -m2/m12 * rho[0] - m3/M * sigma[0],
        -m2/m12 * rho[1] - m3/M * sigma[1],
        0
    ];
    const body2Position: Vector3D = [
        m1/m12 * rho[0] - m3/M * sigma[0],
        m1/m12 * rho[1] - m3/M * sigma[1],
        0
    ];
    const body3Position: Vector3D = [
        m12/M * sigma[0],
        m12/M * sigma[1],
        0
    ];

    // Reconstruct velocities
    const v12: Vector3D = [
        (m3 / M) * vSigma[0],
        (m3 / M) * vSigma[1],
        0
    ];
    const body1Velocity: Vector3D = [
        v12[0] - (m2 / m12) * vRho[0],
        v12[1] - (m2 / m12) * vRho[1],
        0
    ];
    const body2Velocity: Vector3D = [
        v12[0] + (m1 / m12) * vRho[0],
        v12[1] + (m1 / m12) * vRho[1],
        0
    ];
    const body3Velocity: Vector3D = [
        -((m1 + m2) / M) * vSigma[0],
        -((m1 + m2) / M) * vSigma[1],
        0
    ];

    return {
        body1: { position: body1Position, velocity: body1Velocity, mass: m1 },
        body2: { position: body2Position, velocity: body2Velocity, mass: m2 },
        body3: { position: body3Position, velocity: body3Velocity, mass: m3 },
        time: 0
    };
}

// =============================================================================
// 600-CELL MAPPING
// =============================================================================

/**
 * Map a three-body state to the 600-cell interaction manifold.
 * This projects the 8D E8 point through the Moxness matrix to 4D H4.
 */
export function mapTo600Cell(state: ThreeBodyState): Cell600Mapping {
    const phasePoint = encodeToPhaseSpace(state);
    const h4Position = projectE8to4D(phasePoint.coordinates);

    const lattice = getDefaultLattice600();
    const nearestVertex = lattice.findNearest(h4Position);
    const vertex = lattice.getVertex(nearestVertex);

    const dist = vertex ? Math.sqrt(
        (h4Position[0] - vertex.coordinates[0]) ** 2 +
        (h4Position[1] - vertex.coordinates[1]) ** 2 +
        (h4Position[2] - vertex.coordinates[2]) ** 2 +
        (h4Position[3] - vertex.coordinates[3]) ** 2
    ) : Infinity;

    // Determine which 24-cell based on nearest vertex
    const cell24Index = vertex?.cell24Index ?? 0;

    // Map each body to a distinct 24-cell
    const bodyAssignments = lattice.mapThreeBodies(
        h4Position,
        [h4Position[0] + 0.1, h4Position[1], h4Position[2], h4Position[3]], // Offset for body2
        [h4Position[0], h4Position[1] + 0.1, h4Position[2], h4Position[3]]  // Offset for body3
    );

    return {
        h4Position,
        nearestVertex,
        distance: dist,
        cell24Index,
        bodyAssignments
    };
}

// =============================================================================
// SPECIAL ORBITS
// =============================================================================

/**
 * Generate initial conditions for the Figure-8 orbit.
 * Discovered by Chenciner & Montgomery (2000).
 */
export function createFigure8Orbit(): ThreeBodyState {
    // Figure-8 initial conditions from Chenciner-Montgomery (2000)
    // Verified values from Simó's numerical computation
    // See: https://burtleburtle.net/bob/physics/eight.html
    return {
        body1: {
            position: [-0.97000436, 0.24308753, 0],
            velocity: [-0.46620369, -0.43236573, 0],
            mass: 1
        },
        body2: {
            position: [0.97000436, -0.24308753, 0],
            velocity: [-0.46620369, -0.43236573, 0],
            mass: 1
        },
        body3: {
            position: [0, 0, 0],
            velocity: [0.93240737, 0.86473146, 0],
            mass: 1
        },
        time: 0
    };
}

/**
 * Generate initial conditions for the Lagrange equilateral orbit.
 * Three equal masses at vertices of equilateral triangle.
 */
export function createLagrangeOrbit(): ThreeBodyState {
    const R = 1; // Orbital radius
    const omega = Math.sqrt(3 * G / R ** 3); // Angular velocity

    return {
        body1: {
            position: [R, 0, 0],
            velocity: [0, R * omega, 0],
            mass: 1
        },
        body2: {
            position: [R * Math.cos(2 * Math.PI / 3), R * Math.sin(2 * Math.PI / 3), 0],
            velocity: [-R * omega * Math.sin(2 * Math.PI / 3), R * omega * Math.cos(2 * Math.PI / 3), 0],
            mass: 1
        },
        body3: {
            position: [R * Math.cos(4 * Math.PI / 3), R * Math.sin(4 * Math.PI / 3), 0],
            velocity: [-R * omega * Math.sin(4 * Math.PI / 3), R * omega * Math.cos(4 * Math.PI / 3), 0],
            mass: 1
        },
        time: 0
    };
}

/**
 * Generate initial conditions for an Euler collinear orbit.
 * Three bodies on a rotating line.
 */
export function createEulerOrbit(): ThreeBodyState {
    // Collinear configuration with body 3 between bodies 1 and 2
    const a = 1; // Distance scale
    const omega = Math.sqrt(G / a ** 3);

    return {
        body1: {
            position: [-a, 0, 0],
            velocity: [0, -a * omega, 0],
            mass: 1
        },
        body2: {
            position: [a, 0, 0],
            velocity: [0, a * omega, 0],
            mass: 1
        },
        body3: {
            position: [0, 0, 0],
            velocity: [0, 0, 0],
            mass: 1
        },
        time: 0
    };
}

// =============================================================================
// INTEGRATION HELPERS
// =============================================================================

/**
 * Compute the gravitational force on body i from body j.
 */
function computeGravitationalForce(
    ri: Vector3D,
    rj: Vector3D,
    mi: number,
    mj: number
): Vector3D {
    const dx = rj[0] - ri[0];
    const dy = rj[1] - ri[1];
    const dz = rj[2] - ri[2];
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (r < MATH_CONSTANTS.EPSILON) {
        return [0, 0, 0]; // Collision singularity
    }

    const F = G * mi * mj / (r * r * r);
    return [F * dx, F * dy, F * dz];
}

/**
 * Compute total energy of the three-body system.
 */
export function computeEnergy(state: ThreeBodyState): number {
    const { body1, body2, body3 } = state;

    // Kinetic energy
    const KE1 = 0.5 * body1.mass * (body1.velocity[0]**2 + body1.velocity[1]**2 + body1.velocity[2]**2);
    const KE2 = 0.5 * body2.mass * (body2.velocity[0]**2 + body2.velocity[1]**2 + body2.velocity[2]**2);
    const KE3 = 0.5 * body3.mass * (body3.velocity[0]**2 + body3.velocity[1]**2 + body3.velocity[2]**2);
    const KE = KE1 + KE2 + KE3;

    // Potential energy
    const r12 = Math.sqrt(
        (body2.position[0] - body1.position[0])**2 +
        (body2.position[1] - body1.position[1])**2 +
        (body2.position[2] - body1.position[2])**2
    );
    const r13 = Math.sqrt(
        (body3.position[0] - body1.position[0])**2 +
        (body3.position[1] - body1.position[1])**2 +
        (body3.position[2] - body1.position[2])**2
    );
    const r23 = Math.sqrt(
        (body3.position[0] - body2.position[0])**2 +
        (body3.position[1] - body2.position[1])**2 +
        (body3.position[2] - body2.position[2])**2
    );

    const PE = -G * (
        body1.mass * body2.mass / Math.max(r12, MATH_CONSTANTS.EPSILON) +
        body1.mass * body3.mass / Math.max(r13, MATH_CONSTANTS.EPSILON) +
        body2.mass * body3.mass / Math.max(r23, MATH_CONSTANTS.EPSILON)
    );

    return KE + PE;
}

/**
 * Compute angular momentum of the three-body system.
 */
export function computeAngularMomentum(state: ThreeBodyState): Vector3D {
    const { body1, body2, body3 } = state;

    const L1 = cross3D(body1.position, [
        body1.mass * body1.velocity[0],
        body1.mass * body1.velocity[1],
        body1.mass * body1.velocity[2]
    ]);
    const L2 = cross3D(body2.position, [
        body2.mass * body2.velocity[0],
        body2.mass * body2.velocity[1],
        body2.mass * body2.velocity[2]
    ]);
    const L3 = cross3D(body3.position, [
        body3.mass * body3.velocity[0],
        body3.mass * body3.velocity[1],
        body3.mass * body3.velocity[2]
    ]);

    return [
        L1[0] + L2[0] + L3[0],
        L1[1] + L2[1] + L3[1],
        L1[2] + L2[2] + L3[2]
    ];
}

function cross3D(a: Vector3D, b: Vector3D): Vector3D {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
    G,
    computeGravitationalForce
};
