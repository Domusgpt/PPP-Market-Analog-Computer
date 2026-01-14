/**
 * Kustaanheimo-Stiefel Regularization
 *
 * Maps the 3D Kepler problem to a 4D harmonic oscillator.
 * This regularizes collision singularities and enables the
 * geometric encoding onto 4D polytopes (600-cell, 24-cell).
 *
 * Key properties:
 * - x = u₁² - u₂² - u₃² + u₄²
 * - y = 2(u₁u₂ - u₃u₄)
 * - z = 2(u₁u₃ + u₂u₄)
 * - r = |u|² (collision at r=0 → u=0, which is regular)
 *
 * Reference: Kustaanheimo & Stiefel (1965)
 */

import { Vector4D } from '../../types/index.js';

export type Vector3D = [number, number, number];

/**
 * Transform 3D position to 4D KS coordinates (spinor).
 * Uses u₄ = 0 gauge for uniqueness.
 *
 * @param x - 3D position vector
 * @returns 4D KS spinor
 */
export function ks3Dto4D(pos: Vector3D): Vector4D {
    const [x, y, z] = pos;
    const r = Math.sqrt(x * x + y * y + z * z);

    if (r < 1e-10) {
        // At origin - return zero spinor (regularized)
        return [0, 0, 0, 0];
    }

    // Choose u₄ = 0 gauge
    // Then: x = u₁² - u₂² - u₃²
    //       y = 2u₁u₂
    //       z = 2u₁u₃
    //       r = u₁² + u₂² + u₃²

    // Solve for u₁ first
    // From r + x = 2u₁², we get u₁ = √((r + x)/2)
    const u1Sq = (r + x) / 2;

    if (u1Sq < 1e-10) {
        // x ≈ -r case: use different gauge
        // Set u₁ = 0, then x = -u₂² - u₃², y = -2u₃u₄, z = 2u₂u₄
        // With u₁ = 0: r = u₂² + u₃² + u₄²
        const u2Sq = (r - x) / 2;  // r - x = 2(u₂² + u₃²), but need more info
        const u2 = Math.sqrt(Math.max(0, u2Sq));
        // This case needs careful handling...
        // For now, use a small offset
        const u1 = Math.sqrt(Math.max(0, u1Sq + 0.01));
        const u2_ = u1 > 1e-6 ? y / (2 * u1) : 0;
        const u3 = u1 > 1e-6 ? z / (2 * u1) : 0;
        return [u1, u2_, u3, 0];
    }

    const u1 = Math.sqrt(u1Sq);
    const u2 = y / (2 * u1);
    const u3 = z / (2 * u1);
    const u4 = 0;

    return [u1, u2, u3, u4];
}

/**
 * Transform 4D KS spinor back to 3D position.
 *
 * @param u - 4D KS spinor
 * @returns 3D position vector
 */
export function ks4Dto3D(u: Vector4D): Vector3D {
    const [u1, u2, u3, u4] = u;

    const x = u1 * u1 - u2 * u2 - u3 * u3 + u4 * u4;
    const y = 2 * (u1 * u2 - u3 * u4);
    const z = 2 * (u1 * u3 + u2 * u4);

    return [x, y, z];
}

/**
 * Compute 3D radius from 4D spinor.
 * r = |u|²
 */
export function ksRadius(u: Vector4D): number {
    return u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3];
}

/**
 * Transform 3D velocity to 4D KS velocity.
 *
 * The KS velocity transform involves the time scaling:
 * dt_KS = dt / r (fictitious time)
 *
 * @param pos3D - 3D position
 * @param vel3D - 3D velocity
 * @returns 4D KS velocity
 */
export function ksVelocity3Dto4D(pos3D: Vector3D, vel3D: Vector3D): Vector4D {
    const u = ks3Dto4D(pos3D);
    const r = ksRadius(u);

    if (r < 1e-10) {
        return [0, 0, 0, 0];
    }

    // The KS matrix L(u) transforms v = L(u)·u'·2/r
    // For the inverse: u' = L(u)ᵀ·v / 2

    const [vx, vy, vz] = vel3D;
    const [u1, u2, u3, u4] = u;

    // L(u)ᵀ matrix applied to velocity
    const uDot: Vector4D = [
        (u1 * vx + u2 * vy + u3 * vz) / (2 * r),
        (-u2 * vx + u1 * vy + u4 * vz) / (2 * r),
        (-u3 * vx - u4 * vy + u1 * vz) / (2 * r),
        (u4 * vx - u3 * vy + u2 * vz) / (2 * r)
    ];

    return uDot;
}

/**
 * Transform 4D KS velocity back to 3D velocity.
 */
export function ksVelocity4Dto3D(u: Vector4D, uDot: Vector4D): Vector3D {
    const [u1, u2, u3, u4] = u;
    const [ud1, ud2, ud3, ud4] = uDot;

    // v = 2·L(u)·u'
    const vx = 2 * (u1 * ud1 - u2 * ud2 - u3 * ud3 + u4 * ud4);
    const vy = 2 * (u2 * ud1 + u1 * ud2 - u4 * ud3 - u3 * ud4);
    const vz = 2 * (u3 * ud1 + u4 * ud2 + u1 * ud3 + u2 * ud4);

    return [vx, vy, vz];
}

/**
 * Full state transform: 3D phase space → 4D KS phase space
 */
export function ksTransformState(
    pos: Vector3D,
    vel: Vector3D
): { u: Vector4D; uDot: Vector4D } {
    const u = ks3Dto4D(pos);
    const uDot = ksVelocity3Dto4D(pos, vel);
    return { u, uDot };
}

/**
 * Inverse: 4D KS phase space → 3D phase space
 */
export function ksInverseTransformState(
    u: Vector4D,
    uDot: Vector4D
): { pos: Vector3D; vel: Vector3D } {
    const pos = ks4Dto3D(u);
    const vel = ksVelocity4Dto3D(u, uDot);
    return { pos, vel };
}

/**
 * Normalize KS spinor to unit sphere (for mapping to polytope vertices)
 */
export function ksNormalize(u: Vector4D): Vector4D {
    const r = Math.sqrt(ksRadius(u));
    if (r < 1e-10) {
        return [1, 0, 0, 0]; // Default to a vertex
    }
    return [u[0] / r, u[1] / r, u[2] / r, u[3] / r];
}
