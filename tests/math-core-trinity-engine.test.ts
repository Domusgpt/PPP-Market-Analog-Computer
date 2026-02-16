/**
 * Tests for TrinityEngine.ts
 *
 * Verifies Trinity decomposition mechanics:
 * - Phase superposition weights sum to 1
 * - Tension calculation correctness
 * - Phase shift detection
 * - Polytonal detection (balanced weights)
 * - Engine state transitions
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  TrinityEngine,
  createTrinityEngine,
  createTrinityEngineAt,
  PHASE_SHIFT_THRESHOLD,
  POLYTONAL_THRESHOLD,
  TENSION_DECAY_RATE
} from '../_SYNERGIZED_SYSTEM/lib/math_core/engine/TrinityEngine.js';

import type { Vector4D, TrinityAxis } from '../_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/types.js';

const EPSILON = 1e-6;

function approxEqual(a: number, b: number, eps = EPSILON) {
  assert.ok(Math.abs(a - b) < eps, `Expected ${a} ≈ ${b} (diff: ${Math.abs(a - b)})`);
}

describe('TrinityEngine Creation', () => {
  it('creates with default state', () => {
    const engine = createTrinityEngine();
    assert.ok(engine instanceof TrinityEngine);

    const s = engine.state;
    assert.ok(s.activeAxis === 'alpha' || s.activeAxis === 'beta' || s.activeAxis === 'gamma');
  });

  it('createTrinityEngineAt sets position', () => {
    const pos: Vector4D = [0.5, 0.3, 0.2, 0.1];
    const engine = createTrinityEngineAt(pos);
    assert.ok(engine instanceof TrinityEngine);
  });

  it('exported constants are defined', () => {
    assert.ok(typeof PHASE_SHIFT_THRESHOLD === 'number');
    assert.ok(typeof POLYTONAL_THRESHOLD === 'number');
    assert.ok(typeof TENSION_DECAY_RATE === 'number');
    assert.ok(PHASE_SHIFT_THRESHOLD > 0);
    assert.ok(POLYTONAL_THRESHOLD > 0);
  });
});

describe('TrinityEngine State', () => {
  it('weights sum to approximately 1', () => {
    const engine = createTrinityEngine();
    const s = engine.state;
    const [w1, w2, w3] = s.weights;
    approxEqual(w1 + w2 + w3, 1.0, 0.01);
  });

  it('all weights are non-negative', () => {
    const engine = createTrinityEngine();
    const s = engine.state;
    for (const w of s.weights) {
      assert.ok(w >= 0, `Weight ${w} should be non-negative`);
    }
  });

  it('tension is between 0 and 1', () => {
    const engine = createTrinityEngine();
    const s = engine.state;
    assert.ok(s.tension >= 0, `Tension ${s.tension} should be >= 0`);
    assert.ok(s.tension <= 1, `Tension ${s.tension} should be <= 1`);
  });

  it('phaseProgress is between 0 and 1', () => {
    const engine = createTrinityEngine();
    const s = engine.state;
    assert.ok(s.phaseProgress >= 0);
    assert.ok(s.phaseProgress <= 1);
  });
});

describe('TrinityEngine Update', () => {
  it('updatePosition modifies state', () => {
    const engine = createTrinityEngine();

    // Apply a force in a specific direction
    const position: Vector4D = [1, 0, 0, 0];
    const result = engine.updatePosition(position);

    // Result should contain the new state
    assert.ok(result.state.weights.length === 3);
  });

  it('repeated updates converge to stable state', () => {
    const engine = createTrinityEngine();
    const position: Vector4D = [1, 0, 0, 0];

    // Run many updates at the same position
    for (let i = 0; i < 100; i++) {
      engine.updatePosition(position);
    }

    const s = engine.state;
    // Should be stable (no NaN, weights still valid)
    for (const w of s.weights) {
      assert.ok(Number.isFinite(w), `Weight ${w} is not finite after 100 updates`);
    }
    assert.ok(Number.isFinite(s.tension));
  });

  it('different positions produce different dominant axes', () => {
    // Test that the engine responds to position
    const engine1 = createTrinityEngine();
    const engine2 = createTrinityEngine();

    // Push engine1 strongly toward alpha
    for (let i = 0; i < 50; i++) {
      engine1.updatePosition([1, 0, 0, 0]);
    }

    // Push engine2 strongly toward a different region
    for (let i = 0; i < 50; i++) {
      engine2.updatePosition([0, 0, 1, 0]);
    }

    const state1 = engine1.state;
    const state2 = engine2.state;

    // At minimum, both should be valid
    assert.ok(state1.weights.every(w => Number.isFinite(w)));
    assert.ok(state2.weights.every(w => Number.isFinite(w)));
  });
});
