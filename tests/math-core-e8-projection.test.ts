/**
 * Tests for E8Projection.ts
 *
 * Verifies E8 → H4 projection pipeline:
 * - 240 E8 roots generated
 * - Projection matrix dimensions
 * - Projected points have correct dimensionality
 * - Golden ratio properties preserved
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  E8ProjectionPipeline,
  generateE8Roots,
  projectE8Root,
  generate600CellVertices,
  PHI as PHI_E8,
  PHI_CONJUGATE
} from '../_SYNERGIZED_SYSTEM/lib/math_core/topology/E8Projection.js';

const PHI = (1 + Math.sqrt(5)) / 2;
const EPSILON = 1e-8;

function approxEqual(a: number, b: number, eps = EPSILON) {
  assert.ok(Math.abs(a - b) < eps, `Expected ${a} ≈ ${b} (diff: ${Math.abs(a - b)})`);
}

describe('E8 Constants', () => {
  it('PHI_E8 equals golden ratio', () => {
    approxEqual(PHI_E8, PHI);
  });

  it('PHI_CONJUGATE equals 1 - φ', () => {
    approxEqual(PHI_CONJUGATE, 1 - PHI, 1e-6);
  });

  it('PHI * PHI_CONJUGATE = -1 (quadratic property)', () => {
    // φ * (1-φ) = φ - φ² = φ - (φ+1) = -1
    approxEqual(PHI_E8 * PHI_CONJUGATE, -1);
  });
});

describe('E8 Root System', () => {
  it('generates 240 E8 roots', () => {
    const roots = generateE8Roots();
    assert.equal(roots.length, 240, `Expected 240 E8 roots, got ${roots.length}`);
  });

  it('all roots are 8-dimensional (E8Root objects)', () => {
    const roots = generateE8Roots();
    for (const root of roots) {
      // E8Root objects have a .coordinates property which is a Vector8D (8-tuple)
      assert.ok(root.coordinates, 'Root should have coordinates property');
      assert.equal(root.coordinates.length, 8, `Root has ${root.coordinates.length} dimensions, expected 8`);
    }
  });

  it('all roots have the same norm', () => {
    const roots = generateE8Roots();
    const norms = roots.map(r =>
      Math.sqrt(r.coordinates.reduce((s: number, v: number) => s + v * v, 0))
    );

    const firstNorm = norms[0];
    for (let i = 1; i < norms.length; i++) {
      approxEqual(norms[i], firstNorm, 1e-6);
    }
  });

  it('E8 roots come in pairs (r and -r)', () => {
    const roots = generateE8Roots();

    // For each root, its negation should also be a root
    let pairsFound = 0;
    for (const root of roots) {
      const negCoords = root.coordinates.map((v: number) => -v);
      const hasNeg = roots.some(r =>
        r.coordinates.every((v: number, i: number) => Math.abs(v - negCoords[i]) < 1e-8)
      );
      if (hasNeg) pairsFound++;
    }

    // All 240 roots should have their negation
    assert.equal(pairsFound, 240, `Expected 240 paired roots, found ${pairsFound}`);
  });
});

describe('E8 to H4 Projection', () => {
  it('projectE8Root produces outer/inner 4D vectors', () => {
    const roots = generateE8Roots();
    const projected = projectE8Root(roots[0]);

    // ProjectedPoint has outer and inner Vector4D properties
    assert.ok(projected.outer, 'Should have outer property');
    assert.ok(projected.inner, 'Should have inner property');
    assert.equal(projected.outer.length, 4, `Outer has ${projected.outer.length} dimensions`);
    assert.equal(projected.inner.length, 4, `Inner has ${projected.inner.length} dimensions`);
  });

  it('all projected points are finite', () => {
    const roots = generateE8Roots();
    for (const root of roots) {
      const projected = projectE8Root(root);
      for (const v of projected.outer) {
        assert.ok(Number.isFinite(v), `Outer projected value is not finite: ${v}`);
      }
      for (const v of projected.inner) {
        assert.ok(Number.isFinite(v), `Inner projected value is not finite: ${v}`);
      }
    }
  });
});

describe('600-Cell Vertices', () => {
  it('generates 120 vertices', () => {
    const vertices = generate600CellVertices();
    assert.equal(vertices.length, 120, `Expected 120 vertices, got ${vertices.length}`);
  });

  it('all vertices are 4D', () => {
    const vertices = generate600CellVertices();
    for (const v of vertices) {
      assert.equal(v.length, 4);
    }
  });

  it('all vertices have the same norm', () => {
    const vertices = generate600CellVertices();
    const norms = vertices.map(v =>
      Math.sqrt(v.reduce((s: number, x: number) => s + x * x, 0))
    );

    const firstNorm = norms[0];
    for (let i = 1; i < norms.length; i++) {
      approxEqual(norms[i], firstNorm, 1e-4);
    }
  });
});

describe('E8ProjectionPipeline', () => {
  it('creates and initializes', () => {
    const pipeline = new E8ProjectionPipeline();
    assert.ok(pipeline instanceof E8ProjectionPipeline);
  });

  it('provides access to outer and inner vertices', () => {
    const pipeline = new E8ProjectionPipeline();
    const outer = pipeline.outerVertices;
    const inner = pipeline.innerVertices;

    assert.ok(outer.length > 0, 'Should have outer vertices');
    assert.ok(inner.length > 0, 'Should have inner vertices');

    // All outer vertices should be 4D
    for (const v of outer) {
      assert.equal(v.length, 4);
    }
  });
});
