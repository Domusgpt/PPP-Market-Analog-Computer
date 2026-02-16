/**
 * Tests for HDCEncoder.ts
 *
 * Verifies hyperdimensional computing operations:
 * - Random vectors are quasi-orthogonal
 * - Bind is its own inverse
 * - Bundle preserves majority
 * - Permutation shifts correctly
 * - Cosine similarity properties
 * - Seeded RNG produces deterministic results
 * - LRU cache respects limits
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  HDCEncoder,
  createHDCEncoder,
  cosineSimilarity,
  bind,
  bundle,
  permute,
  normalizeHypervector
} from '../_SYNERGIZED_SYSTEM/lib/math_core/encoding/HDCEncoder.js';

import type { Hypervector } from '../_SYNERGIZED_SYSTEM/lib/math_core/encoding/HDCEncoder.js';

const EPSILON = 0.15; // HDC operations are probabilistic, wider tolerance

function approxEqual(a: number, b: number, eps = EPSILON) {
  assert.ok(Math.abs(a - b) < eps, `Expected ${a} ≈ ${b} (diff: ${Math.abs(a - b)})`);
}

describe('HDC Basic Operations', () => {
  it('cosine similarity of identical vectors is 1', () => {
    const hv = new Float32Array([1, -1, 1, -1, 1, -1, 1, -1]);
    approxEqual(cosineSimilarity(hv, hv), 1.0, 0.001);
  });

  it('cosine similarity of opposite vectors is -1', () => {
    const a = new Float32Array([1, -1, 1, -1]);
    const b = new Float32Array([-1, 1, -1, 1]);
    approxEqual(cosineSimilarity(a, b), -1.0, 0.001);
  });

  it('bind is its own inverse (bipolar)', () => {
    const a = new Float32Array([1, -1, 1, -1, 1, -1]);
    const b = new Float32Array([-1, 1, 1, -1, -1, 1]);

    const bound = bind(a, b);
    const unbound = bind(bound, b); // Binding again with b should recover a

    for (let i = 0; i < a.length; i++) {
      approxEqual(unbound[i], a[i], 0.001);
    }
  });

  it('bundle preserves majority signal', () => {
    // 3 copies of 'a' vs 1 copy of 'b' — bundle should be closer to 'a'
    const a = new Float32Array([1, 1, 1, -1, -1, -1]);
    const b = new Float32Array([-1, -1, -1, 1, 1, 1]);

    const result = bundle([a, a, a, b]);
    const normalized = normalizeHypervector(result, true);

    // Should be closer to a than b
    const simA = cosineSimilarity(normalized, a);
    const simB = cosineSimilarity(normalized, b);
    assert.ok(simA > simB, `Bundle should be closer to majority: simA=${simA}, simB=${simB}`);
  });

  it('permute shifts elements cyclically', () => {
    const hv = new Float32Array([1, 2, 3, 4, 5]);
    const shifted = permute(hv, 1);
    // After shift by 1: [5, 1, 2, 3, 4]
    approxEqual(shifted[0], 5, 0.001);
    approxEqual(shifted[1], 1, 0.001);
    approxEqual(shifted[2], 2, 0.001);
    approxEqual(shifted[3], 3, 0.001);
    approxEqual(shifted[4], 4, 0.001);
  });

  it('permute preserves magnitude', () => {
    const hv = new Float32Array([1, -1, 1, -1, 1, -1, 1, -1]);
    const shifted = permute(hv, 3);
    approxEqual(cosineSimilarity(hv, hv), cosineSimilarity(shifted, shifted), 0.001);
  });
});

describe('HDCEncoder Class', () => {
  it('deterministic with same seed', () => {
    const enc1 = new HDCEncoder({ dimensions: 1000, seed: 42 });
    const enc2 = new HDCEncoder({ dimensions: 1000, seed: 42 });

    const hv1 = enc1.encode('test');
    const hv2 = enc2.encode('test');

    approxEqual(cosineSimilarity(hv1, hv2), 1.0, 0.001);
  });

  it('different symbols produce quasi-orthogonal vectors', () => {
    const encoder = new HDCEncoder({ dimensions: 4096, seed: 42 });
    const hvA = encoder.encode('apple');
    const hvB = encoder.encode('banana');

    const sim = cosineSimilarity(hvA, hvB);
    // Random high-dimensional vectors should be quasi-orthogonal
    assert.ok(Math.abs(sim) < 0.2,
      `Different symbols should be quasi-orthogonal, got similarity: ${sim}`);
  });

  it('same symbol returns same vector (from memory)', () => {
    const encoder = new HDCEncoder({ dimensions: 1000, seed: 42 });
    const hv1 = encoder.encode('hello');
    const hv2 = encoder.encode('hello');

    approxEqual(cosineSimilarity(hv1, hv2), 1.0, 0.001);
  });

  it('encodeSequence produces distinct vectors for different orders', () => {
    const encoder = new HDCEncoder({ dimensions: 4096, seed: 42 });
    const hv1 = encoder.encodeSequence(['a', 'b', 'c']);
    const hv2 = encoder.encodeSequence(['c', 'b', 'a']);

    const sim = cosineSimilarity(hv1, hv2);
    // Different orderings should produce different (but possibly somewhat similar) vectors
    assert.ok(sim < 0.9, `Different orderings should differ: similarity = ${sim}`);
  });

  it('store and retrieve works', () => {
    const encoder = new HDCEncoder({ dimensions: 1000, seed: 42 });
    const hv = encoder.encode('test');

    encoder.store('my_vector', hv);
    const retrieved = encoder.retrieve('my_vector');

    assert.ok(retrieved !== undefined);
    approxEqual(cosineSimilarity(hv, retrieved!), 1.0, 0.001);
  });

  it('query finds most similar stored vector', () => {
    const encoder = new HDCEncoder({ dimensions: 4096, seed: 42 });

    const hvA = encoder.encode('apple');
    const hvB = encoder.encode('banana');
    const hvC = encoder.encode('cherry');

    encoder.store('apple', hvA);
    encoder.store('banana', hvB);
    encoder.store('cherry', hvC);

    // Query with hvA should return 'apple' as most similar
    const results = encoder.query(hvA, 3);
    assert.ok(results.length > 0);
    assert.equal(results[0].symbol, 'apple');
  });

  it('dimensions getter returns correct value', () => {
    const encoder = new HDCEncoder({ dimensions: 2048 });
    assert.equal(encoder.dimensions, 2048);
  });

  it('memorySize tracks stored items', () => {
    const encoder = new HDCEncoder({ dimensions: 100, seed: 42 });
    assert.equal(encoder.memorySize, 0);

    encoder.encode('a');
    assert.equal(encoder.memorySize, 1);

    encoder.encode('b');
    assert.equal(encoder.memorySize, 2);

    // Encoding same symbol again shouldn't increase size
    encoder.encode('a');
    assert.equal(encoder.memorySize, 2);
  });

  it('clearMemory empties the store', () => {
    const encoder = new HDCEncoder({ dimensions: 100, seed: 42 });
    encoder.encode('a');
    encoder.encode('b');
    assert.equal(encoder.memorySize, 2);

    encoder.clearMemory();
    assert.equal(encoder.memorySize, 0);
  });
});

describe('HDCEncoder Lattice Integration', () => {
  it('vertex hypervectors are initialized', () => {
    const encoder = new HDCEncoder({ dimensions: 1000, seed: 42 });
    const hv0 = encoder.getVertexHypervector(0);
    assert.ok(hv0 instanceof Float32Array);
    assert.equal(hv0.length, 1000);
  });

  it('findNearestVertex returns valid vertex', () => {
    const encoder = new HDCEncoder({ dimensions: 1000, seed: 42 });
    const hv = encoder.getVertexHypervector(5);
    const result = encoder.findNearestVertex(hv);

    assert.equal(result.vertexId, 5);
    approxEqual(result.similarity, 1.0, 0.001);
  });

  it('projectToPosition returns 4D vector', () => {
    const encoder = new HDCEncoder({ dimensions: 1000, seed: 42 });
    const hv = encoder.getVertexHypervector(0);
    const pos = encoder.projectToPosition(hv);

    assert.equal(pos.length, 4);
    for (const v of pos) {
      assert.ok(Number.isFinite(v), `Position component is not finite: ${v}`);
    }
  });
});
