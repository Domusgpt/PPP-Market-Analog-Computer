/**
 * PhysicsGeometryDomain - Map real physical data into 4D geometry
 *
 * Like MusicGeometryDomain maps notes → 4D vectors, this maps:
 * - Galaxy rotation curves (r, v, M, a) → 4D vectors
 * - Particle properties (m, Q, spin, generation) → 4D vectors
 * - Three-body configurations (masses, L, E, stability) → 4D vectors
 */

import { Vector4D } from '../../types/index.js';
import { getDefaultLattice } from '../topology/Lattice24.js';
import { createEngine } from '../engine/CausalReasoningEngine.js';

// Physical constants
const PHI = (1 + Math.sqrt(5)) / 2;
const c = 299792458;  // m/s
const G = 6.674e-11;  // m³/kg/s²
const m_e = 0.51099895;  // MeV electron mass

// =============================================================================
// GALAXY ROTATION CURVE DATA MAPPING
// =============================================================================

export interface RotationCurvePoint {
    radius_kpc: number;      // Galactocentric radius in kpc
    velocity_kms: number;    // Circular velocity in km/s
    velocity_err?: number;   // Uncertainty
    baryonic_mass?: number;  // Enclosed baryonic mass (if known)
}

export interface GalaxyData {
    name: string;
    distance_Mpc?: number;
    total_mass_solar?: number;
    rotation_curve: RotationCurvePoint[];
}

export class PhysicsGeometryDomain {
    private readonly _lattice = getDefaultLattice();

    // -------------------------------------------------------------------------
    // GALAXY ROTATION CURVES → 4D
    // -------------------------------------------------------------------------

    /**
     * Map a single rotation curve point to 4D vector
     *
     * Dimensions:
     *   x: log(radius) normalized - spatial scale
     *   y: v/v_flat - velocity relative to flat curve
     *   z: log(a/a₀) - acceleration relative to MOND scale
     *   w: Newtonian deviation - how much it differs from Keplerian
     */
    rotationPointTo4D(point: RotationCurvePoint, v_flat: number): Vector4D {
        // MOND acceleration scale
        const a0 = 1.2e-10;  // m/s²

        // Convert to SI
        const r_m = point.radius_kpc * 3.086e19;  // kpc to meters
        const v_ms = point.velocity_kms * 1000;   // km/s to m/s

        // Compute centripetal acceleration
        const a = v_ms * v_ms / r_m;

        // Compute what Newtonian would predict (v ∝ 1/√r)
        // Normalized so that deviation = 0 means perfectly flat
        const newtonianRatio = Math.sqrt(point.radius_kpc);  // Should decrease

        // Map to 4D
        const x = Math.log10(point.radius_kpc + 1) / 2;  // 0 to ~1 for 1-100 kpc
        const y = point.velocity_kms / v_flat;           // ~1 for flat curve
        const z = Math.log10(a / a0);                    // Should be near 0 at a₀
        const w = y * newtonianRatio - 1;                // Deviation from Newtonian

        return this._normalize([x, y, z, w]);
    }

    /**
     * Map entire galaxy rotation curve to trajectory in 4D
     */
    galaxyTo4DTrajectory(galaxy: GalaxyData): Vector4D[] {
        // Find flat velocity (use outer points)
        const outerVelocities = galaxy.rotation_curve
            .filter(p => p.radius_kpc > 5)
            .map(p => p.velocity_kms);
        const v_flat = outerVelocities.length > 0
            ? outerVelocities.reduce((a, b) => a + b, 0) / outerVelocities.length
            : 200;  // Default flat velocity

        return galaxy.rotation_curve.map(p => this.rotationPointTo4D(p, v_flat));
    }

    /**
     * Compute MOND regime indicator for a rotation curve
     * Returns coherence-like metric: high = MOND regime, low = Newtonian
     */
    computeMONDIndicator(galaxy: GalaxyData): number {
        const a0 = 1.2e-10;
        let mondCount = 0;
        let total = 0;

        for (const point of galaxy.rotation_curve) {
            const r_m = point.radius_kpc * 3.086e19;
            const v_ms = point.velocity_kms * 1000;
            const a = v_ms * v_ms / r_m;

            if (a < 10 * a0) mondCount++;  // In or near MOND regime
            total++;
        }

        return mondCount / total;
    }

    // -------------------------------------------------------------------------
    // PARTICLE MASSES → 4D
    // -------------------------------------------------------------------------

    /**
     * Map particle properties to 4D vector
     *
     * Dimensions:
     *   x: log_φ(m/m_e) - mass on golden ratio scale
     *   y: charge (fractional for quarks)
     *   z: spin
     *   w: generation (1, 2, 3)
     */
    particleTo4D(mass_MeV: number, charge: number, spin: number, generation: number): Vector4D {
        const x = Math.log(mass_MeV / m_e) / Math.log(PHI);  // log_φ of mass ratio
        const y = charge;
        const z = spin;
        const w = generation / 3;  // Normalize to [0,1]

        return this._normalize([x / 30, y, z * 2, w]);  // Scale x to comparable range
    }

    /**
     * Predict mass from 4D position using φ^n hypothesis
     */
    positionToMass(position: Vector4D): number {
        // Extract log_φ(mass) from x component
        const logPhiMass = position[0] * 30;  // Reverse the scaling
        return m_e * Math.pow(PHI, logPhiMass);
    }

    /**
     * Find nearest E8 Coxeter exponent for a mass
     */
    massToNearestCoxeter(mass_MeV: number): { n: number; predicted: number; error: number } {
        const COXETER = [1, 7, 11, 13, 17, 19, 23, 29];
        const logPhiMass = Math.log(mass_MeV / m_e) / Math.log(PHI);

        let bestN = 1;
        let bestError = Infinity;

        for (const n of COXETER) {
            const error = Math.abs(n - logPhiMass);
            if (error < bestError) {
                bestError = error;
                bestN = n;
            }
        }

        const predicted = m_e * Math.pow(PHI, bestN);
        const relativeError = Math.abs(predicted - mass_MeV) / mass_MeV;

        return { n: bestN, predicted, error: relativeError };
    }

    // -------------------------------------------------------------------------
    // THREE-BODY CONFIGURATIONS → 4D
    // -------------------------------------------------------------------------

    /**
     * Map three-body orbital parameters to 4D
     *
     * Dimensions:
     *   x: mass entropy (how equal the masses are)
     *   y: angular momentum (normalized)
     *   z: energy (negative = bound)
     *   w: hierarchy parameter (separation of scales)
     */
    threeBodyTo4D(
        m1: number, m2: number, m3: number,  // Masses (relative)
        L: number,                            // Angular momentum (normalized)
        E: number                             // Energy (normalized, neg = bound)
    ): Vector4D {
        const total = m1 + m2 + m3;
        const p1 = m1 / total, p2 = m2 / total, p3 = m3 / total;

        // Mass entropy (max = ln(3) for equal masses)
        const entropy = -(p1 * Math.log(p1) + p2 * Math.log(p2) + p3 * Math.log(p3)) / Math.log(3);

        // Hierarchy: how different are the mass ratios
        const masses = [p1, p2, p3].sort((a, b) => b - a);
        const hierarchy = masses[0] / masses[2];  // Ratio of largest to smallest

        const x = entropy;
        const y = Math.tanh(L);  // Bounded angular momentum
        const z = Math.tanh(E);  // Bounded energy
        const w = Math.tanh(Math.log(hierarchy));  // Log hierarchy

        return this._normalize([x, y, z, w]);
    }

    /**
     * Process three-body config through engine and get stability prediction
     */
    predictStability(m1: number, m2: number, m3: number, L: number, E: number): {
        stable: boolean;
        coherence: number;
        nearestVertex: number;
    } {
        const initialState = this.threeBodyTo4D(m1, m2, m3, L, E);
        const engine = createEngine({ damping: 0.1, inertia: 0.2 });
        engine.state.position = [...initialState];

        // Evolve with small perturbations
        const NUM_STEPS = 50;
        for (let i = 0; i < NUM_STEPS; i++) {
            const perturbation: Vector4D = [
                (Math.random() - 0.5) * 0.05,
                (Math.random() - 0.5) * 0.05,
                (Math.random() - 0.5) * 0.05,
                (Math.random() - 0.5) * 0.05
            ];
            engine.applyLinearForce(perturbation, 0.1);
            engine.update(0.05);
        }

        const finalResult = engine.checkConvexity();

        return {
            stable: finalResult.coherence > 0.7,
            coherence: finalResult.coherence,
            nearestVertex: finalResult.nearestVertex
        };
    }

    // -------------------------------------------------------------------------
    // UTILITIES
    // -------------------------------------------------------------------------

    private _normalize(v: Vector4D): Vector4D {
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        if (norm < 1e-10) return [0, 0, 0, 0];
        return v.map(x => x / norm * 0.8) as Vector4D;  // Scale to stay in 24-cell
    }
}

export const physicsGeometryDomain = new PhysicsGeometryDomain();
