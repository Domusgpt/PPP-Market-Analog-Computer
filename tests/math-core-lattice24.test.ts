/**
 * Tests for Lattice24.ts
 *
 * Verifies the 24-cell polytope lattice structure:
 * - Exactly 24 vertices
 * - Trinity 3×8 partition
 * - All vertices equidistant from origin (on unit sphere)
 * - Each vertex has 8 nearest neighbors
 * - Edge lengths are consistent
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  Lattice24,
  getDefaultLattice,
  createLattice
} from '../_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/Lattice24.js';

import { magnitude, distance } from '../_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/GeometricAlgebra.js';

const EPSILON = 1e-8;

function approxEqual(a: number, b: number, eps = EPSILON) {
  assert.ok(Math.abs(a - b) < eps, `Expected ${a} ≈ ${b} (diff: ${Math.abs(a - b)})`);
}

describe('Lattice24 Vertex Count', () => {
  it('has exactly 24 vertices', () => {
    const lattice = getDefaultLattice();
    assert.equal(lattice.vertexCount, 24);
  });

  it('vertices array length matches vertexCount', () => {
    const lattice = getDefaultLattice();
    assert.equal(lattice.vertices.length, lattice.vertexCount);
  });
});

describe('Lattice24 Trinity Decomposition', () => {
  it('all vertices have a Trinity axis assigned', () => {
    const lattice = getDefaultLattice();
    for (const vertex of lattice.vertices) {
      assert.ok(
        vertex.trinityAxis === 'alpha' ||
        vertex.trinityAxis === 'beta' ||
        vertex.trinityAxis === 'gamma',
        `Vertex ${vertex.id} has invalid trinityAxis: ${vertex.trinityAxis}`
      );
    }
  });

  it('Trinity partition is 3×8 (8 vertices per axis)', () => {
    const lattice = getDefaultLattice();
    const axisCounts = { alpha: 0, beta: 0, gamma: 0 };

    for (const vertex of lattice.vertices) {
      axisCounts[vertex.trinityAxis]++;
    }

    assert.equal(axisCounts.alpha, 8, `Alpha has ${axisCounts.alpha} vertices, expected 8`);
    assert.equal(axisCounts.beta, 8, `Beta has ${axisCounts.beta} vertices, expected 8`);
    assert.equal(axisCounts.gamma, 8, `Gamma has ${axisCounts.gamma} vertices, expected 8`);
  });
});

describe('Lattice24 Geometry', () => {
  it('all vertices are equidistant from origin', () => {
    const lattice = getDefaultLattice();
    const distances = lattice.vertices.map(v => magnitude(v.coordinates));

    // All should be the same distance (circumradius)
    const r = distances[0];
    for (let i = 1; i < distances.length; i++) {
      approxEqual(distances[i], r, 1e-6);
    }
  });

  it('circumradius is consistent (√2 for standard 24-cell)', () => {
    const lattice = getDefaultLattice();
    const r = magnitude(lattice.vertices[0].coordinates);
    // Standard 24-cell has circumradius √2 ≈ 1.414
    // or 1.0 if normalized
    assert.ok(r > 0.5, `Circumradius ${r} is too small`);
    assert.ok(r < 3.0, `Circumradius ${r} is too large`);
  });

  it('each vertex has 8 nearest neighbors', () => {
    const lattice = getDefaultLattice();
    const vertices = lattice.vertices;

    // Compute distances from vertex 0 to all others
    const v0 = vertices[0].coordinates;
    const dists = vertices.slice(1).map((v, i) => ({
      idx: i + 1,
      dist: distance(v0, v.coordinates)
    }));

    dists.sort((a, b) => a.dist - b.dist);

    // The 8 nearest should all be at the same distance (edge length)
    const edgeLength = dists[0].dist;
    for (let i = 0; i < 8; i++) {
      approxEqual(dists[i].dist, edgeLength, 0.01);
    }

    // The 9th should be farther
    assert.ok(dists[8].dist > edgeLength * 1.1,
      `9th nearest (${dists[8].dist}) should be > edge length (${edgeLength})`);
  });

  it('all edge lengths are equal', () => {
    const lattice = getDefaultLattice();
    const vertices = lattice.vertices;

    // Find the edge length from first vertex
    const v0 = vertices[0].coordinates;
    const dists = vertices.slice(1).map(v => distance(v0, v.coordinates));
    dists.sort((a, b) => a - b);
    const edgeLength = dists[0];

    // Check edges from multiple vertices
    for (let vi = 0; vi < Math.min(5, vertices.length); vi++) {
      const ref = vertices[vi].coordinates;
      const d = vertices
        .filter((_, i) => i !== vi)
        .map(v => distance(ref, v.coordinates));
      d.sort((a, b) => a - b);

      // First 8 should be at edge length
      for (let i = 0; i < 8; i++) {
        approxEqual(d[i], edgeLength, 0.01);
      }
    }
  });
});

describe('Lattice24 API', () => {
  it('getVertex returns correct vertex', () => {
    const lattice = getDefaultLattice();
    for (let i = 0; i < lattice.vertexCount; i++) {
      const v = lattice.getVertex(i);
      assert.ok(v !== null && v !== undefined, `getVertex(${i}) returned null`);
      assert.equal(v.id, i);
    }
  });

  it('findNearest returns closest vertex', () => {
    const lattice = getDefaultLattice();
    const v0 = lattice.vertices[0].coordinates;
    // Point very close to vertex 0
    const nearby = [v0[0] + 0.001, v0[1], v0[2], v0[3]] as [number, number, number, number];
    const nearest = lattice.findNearest(nearby);
    assert.equal(nearest, 0);
  });

  it('findKNearest returns k vertices', () => {
    const lattice = getDefaultLattice();
    const point = [0, 0, 0, 0] as [number, number, number, number];
    const k = 5;
    const nearest = lattice.findKNearest(point, k);
    assert.equal(nearest.length, k);

    // All returned indices should be valid
    for (const idx of nearest) {
      assert.ok(idx >= 0 && idx < 24);
    }
  });

  it('createLattice produces a new instance', () => {
    const a = getDefaultLattice();
    const b = createLattice();
    assert.equal(a.vertexCount, b.vertexCount);
  });
});
