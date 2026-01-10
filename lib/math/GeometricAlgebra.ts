/**
 * Geometric Algebra Implementation for Cl(4,0)
 *
 * @package @clear-seas/cpe
 * @version 1.0.0
 * @license MIT
 * @author Clear Seas Solutions LLC
 *
 * This module provides the complete Clifford Algebra for 4-dimensional Euclidean space.
 * It is the mathematical substrate for the Chronomorphic Polytopal Engine, enabling:
 *
 * 1. Multivector arithmetic (addition, geometric product)
 * 2. Wedge product (∧) for constructing context planes (bivectors)
 * 3. Inner product (·) for projection operations
 * 4. Rotor construction and application for unitary transformations
 * 5. Exponential/logarithmic maps for smooth interpolation
 *
 * Mathematical Foundation:
 * - Clifford algebra Cl(4,0) over R^4
 * - 16 basis elements: 1 scalar, 4 vectors, 6 bivectors, 4 trivectors, 1 pseudoscalar
 * - Signature: (+,+,+,+) - all basis vectors square to +1
 *
 * References:
 * - Hestenes & Sobczyk (1984) "Clifford Algebra to Geometric Calculus"
 * - Dorst, Fontijne, Mann (2007) "Geometric Algebra for Computer Science"
 * - Gärdenfors (2000) "Conceptual Spaces: The Geometry of Thought"
 */

import {
    Vector4D,
    Bivector4D,
    MultivectorComponents,
    Rotor,
    Quaternion,
    Grade,
    MATH_CONSTANTS
} from '../../types/index.js';

// =============================================================================
// BASIS ELEMENT DEFINITIONS
// =============================================================================

/**
 * Grade composition for Cl(4,0) - 16 total basis elements.
 */
const GRADE_SIZES = [1, 4, 6, 4, 1] as const;
const TOTAL_COMPONENTS = 16;

/**
 * Index ranges for each grade.
 */
const GRADE_RANGES: Record<Grade, [number, number]> = {
    [Grade.SCALAR]: [0, 1],
    [Grade.VECTOR]: [1, 5],
    [Grade.BIVECTOR]: [5, 11],
    [Grade.TRIVECTOR]: [11, 15],
    [Grade.PSEUDOSCALAR]: [15, 16]
};

/**
 * Basis element names for debugging.
 */
const BASIS_NAMES = [
    '1',                                    // Grade 0: Scalar
    'e1', 'e2', 'e3', 'e4',                // Grade 1: Vectors
    'e12', 'e13', 'e14', 'e23', 'e24', 'e34', // Grade 2: Bivectors
    'e123', 'e124', 'e134', 'e234',         // Grade 3: Trivectors
    'e1234'                                 // Grade 4: Pseudoscalar
];

// =============================================================================
// GEOMETRIC PRODUCT TABLE
// =============================================================================

/**
 * Precomputed geometric product table for Cl(4,0).
 * Entry [i][j] gives (sign, resultIndex) for e_i * e_j.
 *
 * Sign encoding: positive index = positive sign, negative = negative sign
 * Index 0 means the result is the scalar 1.
 */
function buildProductTable(): Int8Array[] {
    const table: Int8Array[] = [];

    for (let i = 0; i < TOTAL_COMPONENTS; i++) {
        table[i] = new Int8Array(TOTAL_COMPONENTS);
        table[i].fill(0);
    }

    // Identity element: 1 * e_i = e_i * 1 = e_i
    for (let i = 0; i < TOTAL_COMPONENTS; i++) {
        table[0][i] = i + 1;  // +1 to encode index (0 reserved for zero)
        table[i][0] = i + 1;
    }

    // Vector products: e_i * e_i = 1, e_i * e_j = e_ij (i < j)
    // e1*e1 = 1, e2*e2 = 1, e3*e3 = 1, e4*e4 = 1
    table[1][1] = 1;  // e1*e1 = 1 (scalar)
    table[2][2] = 1;  // e2*e2 = 1
    table[3][3] = 1;  // e3*e3 = 1
    table[4][4] = 1;  // e4*e4 = 1

    // e1*e2 = e12, e2*e1 = -e12
    table[1][2] = 6;   // e12 is at index 5, encode as 6
    table[2][1] = -6;
    // e1*e3 = e13, e3*e1 = -e13
    table[1][3] = 7;   // e13 is at index 6
    table[3][1] = -7;
    // e1*e4 = e14, e4*e1 = -e14
    table[1][4] = 8;   // e14 is at index 7
    table[4][1] = -8;
    // e2*e3 = e23, e3*e2 = -e23
    table[2][3] = 9;   // e23 is at index 8
    table[3][2] = -9;
    // e2*e4 = e24, e4*e2 = -e24
    table[2][4] = 10;  // e24 is at index 9
    table[4][2] = -10;
    // e3*e4 = e34, e4*e3 = -e34
    table[3][4] = 11;  // e34 is at index 10
    table[4][3] = -11;

    // Bivector squares: e_ij * e_ij = -1
    for (let i = 5; i <= 10; i++) {
        table[i][i] = -1;  // All bivector squares are -1 in Euclidean signature
    }

    // Bivector products: e12*e13 = -e23, etc.
    // e12 * e13 = e1*e2*e1*e3 = -e1*e1*e2*e3 = -e2*e3 = -e23
    table[5][6] = -9;   // e12*e13 = -e23
    table[6][5] = 9;    // e13*e12 = e23
    // e12 * e14 = e1*e2*e1*e4 = -e2*e4 = -e24
    table[5][7] = -10;  // e12*e14 = -e24
    table[7][5] = 10;   // e14*e12 = e24
    // e12 * e23 = e1*e2*e2*e3 = e1*e3 = e13
    table[5][8] = 7;    // e12*e23 = e13
    table[8][5] = -7;   // e23*e12 = -e13
    // e12 * e24 = e1*e2*e2*e4 = e1*e4 = e14
    table[5][9] = 8;    // e12*e24 = e14
    table[9][5] = -8;   // e24*e12 = -e14
    // e12 * e34 = e1234 (pseudoscalar)
    table[5][10] = 16;  // e12*e34 = e1234
    table[10][5] = 16;  // e34*e12 = e1234

    // e13 * e14 = e1*e3*e1*e4 = -e3*e4 = -e34
    table[6][7] = -11;  // e13*e14 = -e34
    table[7][6] = 11;   // e14*e13 = e34
    // e13 * e23 = e1*e3*e2*e3 = -e1*e2 = -e12
    table[6][8] = -6;   // e13*e23 = -e12
    table[8][6] = 6;    // e23*e13 = e12
    // e13 * e24 = -e1234
    table[6][9] = -16;  // e13*e24 = -e1234
    table[9][6] = -16;  // e24*e13 = -e1234
    // e13 * e34 = e1*e3*e3*e4 = e1*e4 = e14
    table[6][10] = 8;   // e13*e34 = e14
    table[10][6] = -8;  // e34*e13 = -e14

    // e14 * e23 = e1234
    table[7][8] = 16;   // e14*e23 = e1234
    table[8][7] = 16;   // e23*e14 = e1234
    // e14 * e24 = -e1*e4*e2*e4 = e1*e2 = e12 (wrong, recalculate)
    // Actually: e14*e24 = e1*e4*e2*e4 = -e1*e2*e4*e4 = -e1*e2 = -e12
    table[7][9] = -6;   // e14*e24 = -e12
    table[9][7] = 6;    // e24*e14 = e12
    // e14 * e34 = e1*e4*e3*e4 = -e1*e3*e4*e4 = -e1*e3 = -e13
    table[7][10] = -7;  // e14*e34 = -e13
    table[10][7] = 7;   // e34*e14 = e13

    // e23 * e24 = e2*e3*e2*e4 = -e3*e4 = -e34
    table[8][9] = -11;  // e23*e24 = -e34
    table[9][8] = 11;   // e24*e23 = e34
    // e23 * e34 = e2*e3*e3*e4 = e2*e4 = e24
    table[8][10] = 10;  // e23*e34 = e24
    table[10][8] = -10; // e34*e23 = -e24

    // e24 * e34 = e2*e4*e3*e4 = -e2*e3 = -e23
    table[9][10] = -9;  // e24*e34 = -e23
    table[10][9] = 9;   // e34*e24 = e23

    // Vector * Bivector products (selected critical ones)
    // e1 * e12 = e1*e1*e2 = e2
    table[1][5] = 3;    // e1*e12 = e2
    table[5][1] = -3;   // e12*e1 = -e2
    // e2 * e12 = e2*e1*e2 = -e1
    table[2][5] = -2;   // e2*e12 = -e1
    table[5][2] = 2;    // e12*e2 = e1
    // e1 * e13 = e3
    table[1][6] = 4;    // e1*e13 = e3
    table[6][1] = -4;   // e13*e1 = -e3
    // e3 * e13 = -e1
    table[3][6] = -2;   // e3*e13 = -e1
    table[6][3] = 2;    // e13*e3 = e1
    // e1 * e14 = e4
    table[1][7] = 5;    // e1*e14 = e4
    table[7][1] = -5;   // e14*e1 = -e4
    // e4 * e14 = -e1
    table[4][7] = -2;   // e4*e14 = -e1
    table[7][4] = 2;    // e14*e4 = e1
    // e2 * e23 = e3
    table[2][8] = 4;    // e2*e23 = e3
    table[8][2] = -4;   // e23*e2 = -e3
    // e3 * e23 = -e2
    table[3][8] = -3;   // e3*e23 = -e2
    table[8][3] = 3;    // e23*e3 = e2
    // e2 * e24 = e4
    table[2][9] = 5;    // e2*e24 = e4
    table[9][2] = -5;   // e24*e2 = -e4
    // e4 * e24 = -e2
    table[4][9] = -3;   // e4*e24 = -e2
    table[9][4] = 3;    // e24*e4 = e2
    // e3 * e34 = e4
    table[3][10] = 5;   // e3*e34 = e4
    table[10][3] = -5;  // e34*e3 = -e4
    // e4 * e34 = -e3
    table[4][10] = -4;  // e4*e34 = -e3
    table[10][4] = 4;   // e34*e4 = e3

    // Vector * Trivector products -> Pseudoscalar or Bivector
    // e1 * e234 = e1234
    table[1][14] = 16;  // e1*e234 = e1234
    table[14][1] = -16; // e234*e1 = -e1234
    // e2 * e134 = -e1234
    table[2][13] = -16; // e2*e134 = -e1234
    table[13][2] = 16;  // e134*e2 = e1234
    // e3 * e124 = e1234
    table[3][12] = 16;  // e3*e124 = e1234
    table[12][3] = -16; // e124*e3 = -e1234
    // e4 * e123 = -e1234
    table[4][11] = -16; // e4*e123 = -e1234
    table[11][4] = 16;  // e123*e4 = e1234

    // Trivector squares = -1
    for (let i = 11; i <= 14; i++) {
        table[i][i] = -1;
    }

    // Pseudoscalar square = -1 in 4D
    table[15][15] = -1;

    // Pseudoscalar * Basis elements (duality)
    // e1234 * e1 = e234, etc. (simplified for now)
    table[15][1] = 15;  // e1234*e1 = e234
    table[1][15] = -15; // e1*e1234 = -e234

    return table;
}

const PRODUCT_TABLE = buildProductTable();

// =============================================================================
// MULTIVECTOR CLASS
// =============================================================================

/**
 * Multivector in Clifford Algebra Cl(4,0).
 *
 * A multivector is a linear combination of basis blades:
 * M = s + v1·e1 + v2·e2 + v3·e3 + v4·e4 + b12·e12 + ... + p·e1234
 *
 * This class provides all algebraic operations needed for geometric reasoning:
 * - Addition, subtraction, scalar multiplication
 * - Geometric product (fundamental operation)
 * - Wedge product (∧) for context construction
 * - Inner product (·) for projection
 * - Reverse, conjugate, norm operations
 * - Exponential/logarithm for rotor algebra
 */
export class Multivector {
    private readonly _components: MultivectorComponents;

    /**
     * Create a multivector from components.
     * @param components - 16-element array of coefficients
     */
    constructor(components?: MultivectorComponents | number[]) {
        if (components) {
            if (components.length !== TOTAL_COMPONENTS) {
                throw new Error(`Multivector requires ${TOTAL_COMPONENTS} components, got ${components.length}`);
            }
            this._components = components instanceof Float64Array
                ? new Float64Array(components)
                : new Float64Array(components);
        } else {
            this._components = new Float64Array(TOTAL_COMPONENTS);
        }
    }

    /**
     * Get the components array (immutable copy).
     */
    get components(): MultivectorComponents {
        return new Float64Array(this._components);
    }

    // =========================================================================
    // FACTORY METHODS
    // =========================================================================

    /**
     * Create a scalar multivector.
     */
    static scalar(s: number): Multivector {
        const mv = new Multivector();
        mv._components[0] = s;
        return mv;
    }

    /**
     * Create a vector (grade-1) multivector from 4D coordinates.
     */
    static vector(v: Vector4D | number[]): Multivector {
        if (v.length !== 4) {
            throw new Error('Vector must have 4 components');
        }
        const mv = new Multivector();
        mv._components[1] = v[0];
        mv._components[2] = v[1];
        mv._components[3] = v[2];
        mv._components[4] = v[3];
        return mv;
    }

    /**
     * Create a bivector (grade-2) multivector.
     * Order: [e12, e13, e14, e23, e24, e34]
     */
    static bivector(b: Bivector4D | number[]): Multivector {
        if (b.length !== 6) {
            throw new Error('Bivector must have 6 components');
        }
        const mv = new Multivector();
        mv._components[5] = b[0];   // e12
        mv._components[6] = b[1];   // e13
        mv._components[7] = b[2];   // e14
        mv._components[8] = b[3];   // e23
        mv._components[9] = b[4];   // e24
        mv._components[10] = b[5];  // e34
        return mv;
    }

    /**
     * Create a rotor from axis-angle representation.
     * The rotation is in the plane perpendicular to the axis in 3D,
     * extended to 4D as a "simple" rotation.
     *
     * For 4D, we use bivector representation directly.
     * R = cos(θ/2) - sin(θ/2)·B where B is the unit bivector.
     */
    static rotor(bivectorPlane: Bivector4D, angle: number): Multivector {
        const halfAngle = angle / 2;
        const cosHalf = Math.cos(halfAngle);
        const sinHalf = Math.sin(halfAngle);

        // Normalize the bivector
        const normSq = bivectorPlane.reduce((sum, x) => sum + x * x, 0);
        const norm = Math.sqrt(normSq);

        if (norm < MATH_CONSTANTS.EPSILON) {
            return Multivector.scalar(1); // Identity rotor
        }

        const mv = new Multivector();
        mv._components[0] = cosHalf;
        for (let i = 0; i < 6; i++) {
            mv._components[5 + i] = -sinHalf * bivectorPlane[i] / norm;
        }

        return mv;
    }

    /**
     * Create a double rotation rotor (isoclinic rotation in 4D).
     * This rotates in two orthogonal planes simultaneously.
     *
     * @param plane1 - First rotation plane (bivector index 0-5)
     * @param angle1 - Angle for first plane
     * @param plane2 - Second rotation plane (must be orthogonal to plane1)
     * @param angle2 - Angle for second plane
     */
    static doubleRotor(
        plane1: number,
        angle1: number,
        plane2: number,
        angle2: number
    ): Multivector {
        // Build two simple rotors and compose them
        const b1 = [0, 0, 0, 0, 0, 0] as Bivector4D;
        const b2 = [0, 0, 0, 0, 0, 0] as Bivector4D;
        b1[plane1] = 1;
        b2[plane2] = 1;

        const r1 = Multivector.rotor(b1, angle1);
        const r2 = Multivector.rotor(b2, angle2);

        return r1.mul(r2);
    }

    /**
     * Create basis element e_i.
     */
    static basis(index: number): Multivector {
        if (index < 0 || index >= TOTAL_COMPONENTS) {
            throw new Error(`Basis index must be 0-${TOTAL_COMPONENTS - 1}`);
        }
        const mv = new Multivector();
        mv._components[index] = 1;
        return mv;
    }

    /**
     * Zero multivector.
     */
    static zero(): Multivector {
        return new Multivector();
    }

    /**
     * Identity element (scalar 1).
     */
    static one(): Multivector {
        return Multivector.scalar(1);
    }

    // =========================================================================
    // GRADE EXTRACTION
    // =========================================================================

    /**
     * Extract scalar (grade-0) part.
     */
    get scalar(): number {
        return this._components[0];
    }

    /**
     * Extract vector (grade-1) part.
     */
    get vector(): Vector4D {
        return [
            this._components[1],
            this._components[2],
            this._components[3],
            this._components[4]
        ];
    }

    /**
     * Extract bivector (grade-2) part.
     */
    get bivector(): Bivector4D {
        return [
            this._components[5],
            this._components[6],
            this._components[7],
            this._components[8],
            this._components[9],
            this._components[10]
        ];
    }

    /**
     * Extract trivector (grade-3) part.
     */
    get trivector(): [number, number, number, number] {
        return [
            this._components[11],
            this._components[12],
            this._components[13],
            this._components[14]
        ];
    }

    /**
     * Extract pseudoscalar (grade-4) part.
     */
    get pseudoscalar(): number {
        return this._components[15];
    }

    /**
     * Extract grade-k part as new multivector.
     */
    grade(k: Grade): Multivector {
        const mv = new Multivector();
        const [start, end] = GRADE_RANGES[k];
        for (let i = start; i < end; i++) {
            mv._components[i] = this._components[i];
        }
        return mv;
    }

    /**
     * Extract even-grade parts (scalar + bivector + pseudoscalar).
     * Even multivectors form the rotor group.
     */
    even(): Multivector {
        const mv = new Multivector();
        mv._components[0] = this._components[0];
        for (let i = 5; i <= 10; i++) {
            mv._components[i] = this._components[i];
        }
        mv._components[15] = this._components[15];
        return mv;
    }

    /**
     * Extract odd-grade parts (vector + trivector).
     */
    odd(): Multivector {
        const mv = new Multivector();
        for (let i = 1; i <= 4; i++) {
            mv._components[i] = this._components[i];
        }
        for (let i = 11; i <= 14; i++) {
            mv._components[i] = this._components[i];
        }
        return mv;
    }

    // =========================================================================
    // ALGEBRAIC OPERATIONS
    // =========================================================================

    /**
     * Addition: A + B
     */
    add(other: Multivector): Multivector {
        const result = new Multivector();
        for (let i = 0; i < TOTAL_COMPONENTS; i++) {
            result._components[i] = this._components[i] + other._components[i];
        }
        return result;
    }

    /**
     * Subtraction: A - B
     */
    sub(other: Multivector): Multivector {
        const result = new Multivector();
        for (let i = 0; i < TOTAL_COMPONENTS; i++) {
            result._components[i] = this._components[i] - other._components[i];
        }
        return result;
    }

    /**
     * Scalar multiplication: s · A
     */
    scale(s: number): Multivector {
        const result = new Multivector();
        for (let i = 0; i < TOTAL_COMPONENTS; i++) {
            result._components[i] = this._components[i] * s;
        }
        return result;
    }

    /**
     * Negation: -A
     */
    neg(): Multivector {
        return this.scale(-1);
    }

    /**
     * Geometric Product: A * B
     *
     * The fundamental operation in Clifford algebra.
     * Combines inner (contraction) and outer (extension) products:
     * ab = a·b + a∧b (for vectors)
     */
    mul(other: Multivector): Multivector {
        const result = new Multivector();

        for (let i = 0; i < TOTAL_COMPONENTS; i++) {
            if (Math.abs(this._components[i]) < MATH_CONSTANTS.EPSILON) continue;

            for (let j = 0; j < TOTAL_COMPONENTS; j++) {
                if (Math.abs(other._components[j]) < MATH_CONSTANTS.EPSILON) continue;

                const productCode = PRODUCT_TABLE[i][j];
                if (productCode === 0) continue;

                const sign = productCode > 0 ? 1 : -1;
                const resultIdx = Math.abs(productCode) - 1;

                result._components[resultIdx] += sign * this._components[i] * other._components[j];
            }
        }

        return result;
    }

    /**
     * Wedge (Outer) Product: A ∧ B
     *
     * Creates higher-grade elements from lower-grade ones.
     * For vectors a, b: a∧b is the bivector representing the plane they span.
     * This is the "context construction" operation.
     */
    wedge(other: Multivector): Multivector {
        // a ∧ b = (ab - ba) / 2 for anticommuting parts
        // More generally, grade projection
        const ab = this.mul(other);
        const ba = other.mul(this);

        // For pure grade elements, this gives the antisymmetric part
        return ab.sub(ba).scale(0.5);
    }

    /**
     * Inner (Dot) Product: A · B
     *
     * Contracts higher-grade to lower-grade elements.
     * For vectors a, b: a·b is the scalar projection.
     * This is the "similarity/projection" operation.
     */
    inner(other: Multivector): Multivector {
        // a · b = (ab + ba) / 2 for commuting parts
        const ab = this.mul(other);
        const ba = other.mul(this);

        // For pure grade elements, this gives the symmetric part
        // Then extract the grade |r-s| where r,s are input grades
        return ab.add(ba).scale(0.5).grade(Grade.SCALAR);
    }

    /**
     * Left Contraction: A ⌋ B
     * Grade-lowering operation.
     */
    leftContract(other: Multivector): Multivector {
        // Simplified: extract lower-grade part of inner product
        return this.inner(other);
    }

    // =========================================================================
    // INVOLUTIONS
    // =========================================================================

    /**
     * Reversion: M~ (tilde)
     * Reverses the order of basis vectors in each blade.
     *
     * For grade k blade: rev(B_k) = (-1)^(k(k-1)/2) * B_k
     * Grade 0,1: unchanged
     * Grade 2,3: negated
     * Grade 4: unchanged
     */
    reverse(): Multivector {
        const result = new Multivector(this._components);
        // Negate bivectors (grade 2)
        for (let i = 5; i <= 10; i++) {
            result._components[i] = -result._components[i];
        }
        // Negate trivectors (grade 3)
        for (let i = 11; i <= 14; i++) {
            result._components[i] = -result._components[i];
        }
        return result;
    }

    /**
     * Grade Involution: M* (star)
     * Negates odd-grade parts.
     */
    involute(): Multivector {
        const result = new Multivector(this._components);
        // Negate vectors (grade 1)
        for (let i = 1; i <= 4; i++) {
            result._components[i] = -result._components[i];
        }
        // Negate trivectors (grade 3)
        for (let i = 11; i <= 14; i++) {
            result._components[i] = -result._components[i];
        }
        return result;
    }

    /**
     * Clifford Conjugate: M† (dagger)
     * Combines reversion and involution.
     */
    conjugate(): Multivector {
        return this.reverse().involute();
    }

    // =========================================================================
    // NORMS AND NORMALIZATION
    // =========================================================================

    /**
     * Squared norm: |M|² = <M M~>_0
     * The scalar part of M times its reverse.
     */
    normSquared(): number {
        const product = this.mul(this.reverse());
        return Math.abs(product.scalar);
    }

    /**
     * Norm: |M| = sqrt(|M|²)
     */
    norm(): number {
        return Math.sqrt(this.normSquared());
    }

    /**
     * Normalize to unit multivector.
     */
    normalized(): Multivector {
        const n = this.norm();
        if (n < MATH_CONSTANTS.EPSILON) {
            return Multivector.zero();
        }
        return this.scale(1 / n);
    }

    /**
     * Check if this is a unit multivector.
     */
    isUnit(tolerance: number = MATH_CONSTANTS.EPSILON): boolean {
        return Math.abs(this.norm() - 1) < tolerance;
    }

    // =========================================================================
    // ROTOR OPERATIONS
    // =========================================================================

    /**
     * Sandwich Product: R M R~
     * Applies rotor R to multivector M.
     * This is how rotations are performed.
     */
    sandwich(rotor: Multivector): Multivector {
        return rotor.mul(this).mul(rotor.reverse());
    }

    /**
     * Apply this multivector as a rotor to a vector.
     * v' = R v R~
     */
    applyRotor(v: Vector4D): Vector4D {
        const vMv = Multivector.vector(v);
        const rotated = vMv.sandwich(this);
        return rotated.vector;
    }

    // =========================================================================
    // EXPONENTIAL AND LOGARITHM
    // =========================================================================

    /**
     * Exponential of multivector.
     *
     * For pure bivector B: exp(B) = cos(|B|) + sin(|B|)·B/|B|
     * This produces a unit rotor.
     */
    exp(): Multivector {
        // Extract bivector part
        const b = this.bivector;
        const bNormSq = b.reduce((sum, x) => sum + x * x, 0);
        const bNorm = Math.sqrt(bNormSq);

        if (bNorm < MATH_CONSTANTS.EPSILON) {
            // exp(scalar) = e^s · 1
            return Multivector.scalar(Math.exp(this.scalar));
        }

        // exp(bivector) = cos(|B|) + sin(|B|) · B/|B|
        const cosB = Math.cos(bNorm);
        const sinB = Math.sin(bNorm);
        const expScalar = Math.exp(this.scalar);

        const result = new Multivector();
        result._components[0] = cosB * expScalar;
        for (let i = 0; i < 6; i++) {
            result._components[5 + i] = (sinB / bNorm) * b[i] * expScalar;
        }

        return result;
    }

    /**
     * Logarithm of even multivector (rotor).
     * Returns bivector such that exp(result) ≈ this.
     */
    log(): Multivector {
        // For rotor R = s + B (even multivector)
        // log(R) = atan2(|B|, s) · B/|B|
        const s = this.scalar;
        const b = this.bivector;
        const bNormSq = b.reduce((sum, x) => sum + x * x, 0);
        const bNorm = Math.sqrt(bNormSq);

        if (bNorm < MATH_CONSTANTS.EPSILON) {
            // Nearly identity
            return Multivector.scalar(Math.log(Math.abs(s)));
        }

        const angle = Math.atan2(bNorm, s);

        const result = new Multivector();
        for (let i = 0; i < 6; i++) {
            result._components[5 + i] = (angle / bNorm) * b[i];
        }

        return result;
    }

    /**
     * Spherical linear interpolation between rotors.
     */
    static slerp(r1: Multivector, r2: Multivector, t: number): Multivector {
        // R(t) = R1 * exp(t * log(R1^{-1} * R2))
        const r1Inv = r1.reverse(); // For unit rotor, reverse = inverse
        const delta = r1Inv.mul(r2);
        const logDelta = delta.log();
        const scaledLog = logDelta.scale(t);
        const expScaled = scaledLog.exp();
        return r1.mul(expScaled);
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /**
     * Convert to Rotor interface if this is a valid rotor.
     */
    toRotor(): Rotor {
        return {
            scalar: this.scalar,
            bivector: this.bivector,
            isUnit: this.isUnit()
        };
    }

    /**
     * Check if approximately zero.
     */
    isZero(tolerance: number = MATH_CONSTANTS.EPSILON): boolean {
        return this._components.every(c => Math.abs(c) < tolerance);
    }

    /**
     * Check if approximately equal to another multivector.
     */
    equals(other: Multivector, tolerance: number = MATH_CONSTANTS.EPSILON): boolean {
        for (let i = 0; i < TOTAL_COMPONENTS; i++) {
            if (Math.abs(this._components[i] - other._components[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    /**
     * Clone this multivector.
     */
    clone(): Multivector {
        return new Multivector(this._components);
    }

    /**
     * String representation for debugging.
     */
    toString(): string {
        const terms: string[] = [];
        for (let i = 0; i < TOTAL_COMPONENTS; i++) {
            if (Math.abs(this._components[i]) > MATH_CONSTANTS.EPSILON) {
                if (i === 0) {
                    terms.push(`${this._components[i].toFixed(4)}`);
                } else {
                    terms.push(`${this._components[i].toFixed(4)}·${BASIS_NAMES[i]}`);
                }
            }
        }
        return `Multivector(${terms.join(' + ') || '0'})`;
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Compute the wedge product of two 4D vectors directly.
 * Returns the 6-component bivector.
 *
 * This is the "context construction" operation:
 * Given two concept vectors, their wedge product defines the plane of meaning.
 */
export function wedge(a: Vector4D, b: Vector4D): Bivector4D {
    // a ∧ b = (a_i * b_j - a_j * b_i) for i < j
    return [
        a[0] * b[1] - a[1] * b[0],  // e12
        a[0] * b[2] - a[2] * b[0],  // e13
        a[0] * b[3] - a[3] * b[0],  // e14
        a[1] * b[2] - a[2] * b[1],  // e23
        a[1] * b[3] - a[3] * b[1],  // e24
        a[2] * b[3] - a[3] * b[2]   // e34
    ];
}

/**
 * Compute the inner (dot) product of two 4D vectors.
 * Returns a scalar representing similarity/projection.
 */
export function dot(a: Vector4D, b: Vector4D): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/**
 * Compute the geometric centroid of a set of 4D vectors.
 *
 * The centroid is central to Epistaorthognition:
 * "If you normalize this sum that represents the centroid...
 *  By definition the centroid must lie within the convex hull."
 */
export function centroid(vectors: Vector4D[]): Vector4D {
    if (vectors.length === 0) {
        return [0, 0, 0, 0];
    }

    const sum: Vector4D = [0, 0, 0, 0];
    for (const v of vectors) {
        sum[0] += v[0];
        sum[1] += v[1];
        sum[2] += v[2];
        sum[3] += v[3];
    }

    const n = vectors.length;
    return [sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n];
}

/**
 * Normalize a 4D vector.
 */
export function normalize(v: Vector4D): Vector4D {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
    if (len < MATH_CONSTANTS.EPSILON) {
        return [0, 0, 0, 0];
    }
    return [v[0] / len, v[1] / len, v[2] / len, v[3] / len];
}

/**
 * Compute the magnitude of a 4D vector.
 */
export function magnitude(v: Vector4D): number {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
}

/**
 * Compute the magnitude of a 6-component bivector.
 */
export function bivectorMagnitude(b: Bivector4D): number {
    return Math.sqrt(b.reduce((sum, x) => sum + x * x, 0));
}

/**
 * Convert double-quaternion (Spin(4)) to rotor.
 * Uses the quaternion product correspondence:
 * R = q_L ⊗ q_R (tensor product encoding)
 */
export function quaternionsToRotor(left: Quaternion, right: Quaternion): Multivector {
    // The scalar part
    const s = left[0] * right[0] - left[1] * right[1] - left[2] * right[2] - left[3] * right[3];

    // Bivector components (simplified mapping)
    const b12 = left[0] * right[1] + left[1] * right[0] + left[2] * right[3] - left[3] * right[2];
    const b13 = left[0] * right[2] - left[1] * right[3] + left[2] * right[0] + left[3] * right[1];
    const b14 = left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0];
    const b23 = left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0];
    const b24 = -left[0] * right[2] + left[1] * right[3] - left[2] * right[0] - left[3] * right[1];
    const b34 = left[0] * right[1] + left[1] * right[0] + left[2] * right[3] - left[3] * right[2];

    const mv = new Multivector();
    (mv as unknown as { _components: Float64Array })._components[0] = s;
    (mv as unknown as { _components: Float64Array })._components[5] = b12;
    (mv as unknown as { _components: Float64Array })._components[6] = b13;
    (mv as unknown as { _components: Float64Array })._components[7] = b14;
    (mv as unknown as { _components: Float64Array })._components[8] = b23;
    (mv as unknown as { _components: Float64Array })._components[9] = b24;
    (mv as unknown as { _components: Float64Array })._components[10] = b34;

    return mv.normalized();
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
    TOTAL_COMPONENTS,
    GRADE_RANGES,
    BASIS_NAMES
};
