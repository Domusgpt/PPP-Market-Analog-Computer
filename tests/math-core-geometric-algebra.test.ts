/**
 * Tests for GeometricAlgebra.ts
 *
 * Verifies fundamental invariants of 4D Clifford algebra operations:
 * - Vector operations (dot, magnitude, normalize)
 * - Bivector operations (wedge product)
 * - Rotor operations (sandwich product, norm preservation)
 * - Matrix operations (rotation matrices, projection)
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  dot,
  magnitude,
  normalize,
  scale,
  add,
  subtract,
  centroid,
  distance,
  wedge,
  rotorInPlane,
  applyRotorToVector,
  stereographicProject,
  stereographicUnproject,
  combinedRotationMatrix,
  matrixVectorMultiply
} from '../_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/GeometricAlgebra.js';

import { RotationPlane } from '../_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/types.js';
import type { Vector4D, Bivector4D } from '../_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/types.js';

const EPSILON = 1e-10;

function approxEqual(a: number, b: number, eps = EPSILON) {
  assert.ok(Math.abs(a - b) < eps, `Expected ${a} ≈ ${b} (diff: ${Math.abs(a - b)})`);
}

describe('Vector Operations', () => {
  it('dot product is commutative', () => {
    const a: Vector4D = [1, 2, 3, 4];
    const b: Vector4D = [5, 6, 7, 8];
    approxEqual(dot(a, b), dot(b, a));
  });

  it('dot product with self equals magnitude squared', () => {
    const v: Vector4D = [3, 4, 0, 0];
    approxEqual(dot(v, v), magnitude(v) ** 2);
  });

  it('magnitude of unit vector is 1', () => {
    const v: Vector4D = [1, 2, 3, 4];
    const n = normalize(v);
    approxEqual(magnitude(n), 1.0);
  });

  it('normalize preserves direction', () => {
    const v: Vector4D = [3, 0, 0, 0];
    const n = normalize(v);
    approxEqual(n[0], 1);
    approxEqual(n[1], 0);
    approxEqual(n[2], 0);
    approxEqual(n[3], 0);
  });

  it('scale multiplies all components', () => {
    const v: Vector4D = [1, 2, 3, 4];
    const s = scale(v, 2);
    assert.deepStrictEqual(s, [2, 4, 6, 8]);
  });

  it('add is commutative', () => {
    const a: Vector4D = [1, 2, 3, 4];
    const b: Vector4D = [5, 6, 7, 8];
    const ab = add(a, b);
    const ba = add(b, a);
    for (let i = 0; i < 4; i++) approxEqual(ab[i], ba[i]);
  });

  it('subtract gives zero for identical vectors', () => {
    const v: Vector4D = [1, 2, 3, 4];
    const diff = subtract(v, v);
    approxEqual(magnitude(diff), 0);
  });

  it('centroid of identical points is the point', () => {
    const v: Vector4D = [1, 2, 3, 4];
    const c = centroid([v, v, v]);
    for (let i = 0; i < 4; i++) approxEqual(c[i], v[i]);
  });

  it('distance is symmetric', () => {
    const a: Vector4D = [1, 0, 0, 0];
    const b: Vector4D = [0, 1, 0, 0];
    approxEqual(distance(a, b), distance(b, a));
  });

  it('distance from point to itself is 0', () => {
    const v: Vector4D = [1, 2, 3, 4];
    approxEqual(distance(v, v), 0);
  });
});

describe('Bivector Operations', () => {
  it('wedge product is anti-commutative', () => {
    const a: Vector4D = [1, 0, 0, 0];
    const b: Vector4D = [0, 1, 0, 0];
    const ab = wedge(a, b);
    const ba = wedge(b, a);
    for (let i = 0; i < 6; i++) {
      approxEqual(ab[i], -ba[i]);
    }
  });

  it('wedge of parallel vectors is zero', () => {
    const a: Vector4D = [1, 2, 3, 4];
    const b: Vector4D = [2, 4, 6, 8]; // 2*a
    const w = wedge(a, b);
    for (let i = 0; i < 6; i++) {
      approxEqual(w[i], 0);
    }
  });

  it('wedge of orthonormal basis vectors has unit magnitude', () => {
    const e1: Vector4D = [1, 0, 0, 0];
    const e2: Vector4D = [0, 1, 0, 0];
    const w = wedge(e1, e2);
    // Should be [1, 0, 0, 0, 0, 0] (XY component)
    approxEqual(w[0], 1);
    for (let i = 1; i < 6; i++) approxEqual(w[i], 0);
  });
});

describe('Rotor Operations', () => {
  it('rotation preserves vector magnitude', () => {
    const v: Vector4D = [1, 2, 3, 4];
    const rotor = rotorInPlane(RotationPlane.XY, Math.PI / 4);
    const rotated = applyRotorToVector(rotor, v);

    approxEqual(magnitude(rotated), magnitude(v), 1e-8);
  });

  it('rotation by 0 is identity', () => {
    const v: Vector4D = [1, 2, 3, 4];
    const rotor = rotorInPlane(RotationPlane.XY, 0);
    const rotated = applyRotorToVector(rotor, v);

    for (let i = 0; i < 4; i++) {
      approxEqual(rotated[i], v[i], 1e-8);
    }
  });

  it('rotation by 2π returns to original', () => {
    const v: Vector4D = [1, 2, 3, 4];
    const rotor = rotorInPlane(RotationPlane.XY, 2 * Math.PI);
    const rotated = applyRotorToVector(rotor, v);

    for (let i = 0; i < 4; i++) {
      approxEqual(rotated[i], v[i], 1e-6);
    }
  });

  it('rotation by π/2 in XY plane swaps x and y', () => {
    const v: Vector4D = [1, 0, 0, 0];
    const rotor = rotorInPlane(RotationPlane.XY, Math.PI / 2);
    const rotated = applyRotorToVector(rotor, v);

    approxEqual(rotated[0], 0, 1e-8);
    approxEqual(Math.abs(rotated[1]), 1, 1e-8);
    approxEqual(rotated[2], 0, 1e-8);
    approxEqual(rotated[3], 0, 1e-8);
  });
});

describe('Matrix Operations', () => {
  it('combined rotation matrix preserves magnitude', () => {
    const matrix = combinedRotationMatrix(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
    const v: Vector4D = [1, 2, 3, 4];
    const rotated = matrixVectorMultiply(matrix, v);

    approxEqual(magnitude(rotated), magnitude(v), 1e-6);
  });

  it('zero rotation matrix is identity', () => {
    const matrix = combinedRotationMatrix(0, 0, 0, 0, 0, 0);
    const v: Vector4D = [1, 2, 3, 4];
    const rotated = matrixVectorMultiply(matrix, v);

    for (let i = 0; i < 4; i++) {
      approxEqual(rotated[i], v[i], 1e-10);
    }
  });
});

describe('Stereographic Projection', () => {
  it('project produces 3D output', () => {
    const v: Vector4D = [0.5, 0.3, 0.2, 0.1];
    const projected = stereographicProject(v, 2.0);
    assert.equal(projected.length, 3);
    for (const val of projected) {
      assert.ok(Number.isFinite(val), `Projected value not finite: ${val}`);
    }
  });

  it('unproject produces 4D output on unit sphere', () => {
    const p: [number, number, number] = [1, 0.5, 0.3];
    const unprojected = stereographicUnproject(p, 2.0);
    assert.equal(unprojected.length, 4);
    // Unprojected point should be on unit sphere
    const norm = magnitude(unprojected);
    approxEqual(norm, 1.0, 0.01);
  });

  it('origin projects to origin', () => {
    const v: Vector4D = [0, 0, 0, 0];
    const projected = stereographicProject(v, 2.0);
    approxEqual(projected[0], 0);
    approxEqual(projected[1], 0);
    approxEqual(projected[2], 0);
  });
});
