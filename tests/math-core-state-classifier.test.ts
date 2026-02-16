/**
 * Tests for StateClassifier.ts
 *
 * Verifies metacognitive state classification:
 * - Classification returns valid categories
 * - Confidence levels are meaningful
 * - Suggestions are generated
 * - History tracking works
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  StateClassifier,
  DEFAULT_CLASSIFIER_CONFIG
} from '../_SYNERGIZED_SYSTEM/lib/math_core/metacognition/StateClassifier.js';

import type {
  StateCategory,
  StateClassification,
  TrinityState,
  BettiProfile
} from '../_SYNERGIZED_SYSTEM/lib/math_core/metacognition/StateClassifier.js';

import type { Vector4D } from '../_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/types.js';

const VALID_CATEGORIES: StateCategory[] = [
  'COHERENT', 'TRANSITIONING', 'AMBIGUOUS', 'POLYTONAL', 'STUCK', 'INVALID', 'UNKNOWN'
];

describe('StateClassifier Construction', () => {
  it('creates with defaults', () => {
    const classifier = new StateClassifier();
    assert.ok(classifier instanceof StateClassifier);
  });

  it('DEFAULT_CLASSIFIER_CONFIG has required fields', () => {
    assert.ok(typeof DEFAULT_CLASSIFIER_CONFIG.coherenceThreshold === 'number');
    assert.ok(typeof DEFAULT_CLASSIFIER_CONFIG.ambiguityThreshold === 'number');
    assert.ok(typeof DEFAULT_CLASSIFIER_CONFIG.stuckVelocityThreshold === 'number');
  });
});

describe('StateClassifier Classification', () => {
  it('returns valid category', () => {
    const classifier = new StateClassifier();

    const result = classifier.classify(
      [0.5, 0.3, 0.2, 0.1] as Vector4D,   // position
      0.1,                                    // velocity
      { activeAxis: 'alpha' as const, weights: [0.7, 0.2, 0.1] as [number, number, number], tension: 0.1, phaseProgress: 0 },
      { beta0: 1, beta1: 0, beta2: 0 },     // betti
      0.9,                                    // coherence
      true,                                   // isInsideHull
      0.1                                     // nearestVertexDistance
    );

    assert.ok(VALID_CATEGORIES.includes(result.category),
      `Invalid category: ${result.category}`);
  });

  it('score is between 0 and 1', () => {
    const classifier = new StateClassifier();

    const result = classifier.classify(
      [0, 0, 0, 0] as Vector4D,
      0.05,
      { activeAxis: 'alpha' as const, weights: [0.5, 0.3, 0.2] as [number, number, number], tension: 0.2, phaseProgress: 0 },
      null,
      0.8,
      true,
      0.2
    );

    assert.ok(result.score >= 0, `Score ${result.score} < 0`);
    assert.ok(result.score <= 1, `Score ${result.score} > 1`);
  });

  it('confidence is high/medium/low', () => {
    const classifier = new StateClassifier();

    const result = classifier.classify(
      [0.5, 0.3, 0.2, 0.1] as Vector4D,
      0.1,
      { activeAxis: 'alpha' as const, weights: [0.8, 0.1, 0.1] as [number, number, number], tension: 0.1, phaseProgress: 0 },
      { beta0: 1, beta1: 0, beta2: 0 },
      0.95,
      true,
      0.05
    );

    assert.ok(
      result.confidence === 'high' ||
      result.confidence === 'medium' ||
      result.confidence === 'low',
      `Invalid confidence: ${result.confidence}`
    );
  });

  it('returns alternatives list', () => {
    const classifier = new StateClassifier();

    const result = classifier.classify(
      [0, 0, 0, 0] as Vector4D,
      0.05,
      { activeAxis: 'beta' as const, weights: [0.3, 0.5, 0.2] as [number, number, number], tension: 0.3, phaseProgress: 0 },
      null,
      0.7,
      true,
      0.2
    );

    assert.ok(Array.isArray(result.alternatives));
    for (const alt of result.alternatives) {
      assert.ok(VALID_CATEGORIES.includes(alt));
    }
  });

  it('generates description string', () => {
    const classifier = new StateClassifier();

    const result = classifier.classify(
      [0, 0, 0, 0] as Vector4D,
      0.05,
      { activeAxis: 'alpha' as const, weights: [0.6, 0.2, 0.2] as [number, number, number], tension: 0.1, phaseProgress: 0 },
      null,
      0.8,
      true,
      0.1
    );

    assert.ok(typeof result.description === 'string' && result.description.length > 0);
  });

  it('features object has all required fields', () => {
    const classifier = new StateClassifier();

    const result = classifier.classify(
      [0.5, 0.3, 0.2, 0.1] as Vector4D,
      0.1,
      { activeAxis: 'alpha' as const, weights: [0.7, 0.2, 0.1] as [number, number, number], tension: 0.15, phaseProgress: 0 },
      { beta0: 1, beta1: 0, beta2: 0 },
      0.85,
      true,
      0.1
    );

    assert.ok('velocity' in result.features);
    assert.ok('coherence' in result.features);
    assert.ok('tension' in result.features);
    assert.ok('dominantAxisWeight' in result.features);
    assert.ok('axisBalance' in result.features);
    assert.ok('isInsideHull' in result.features);
  });
});

describe('StateClassifier Specific States', () => {
  it('classifies INVALID when outside hull', () => {
    const classifier = new StateClassifier();

    const result = classifier.classify(
      [10, 10, 10, 10] as Vector4D,  // Far from any vertex
      0.1,
      { activeAxis: 'alpha' as const, weights: [0.4, 0.3, 0.3] as [number, number, number], tension: 0.5, phaseProgress: 0 },
      null,
      0.2,
      false,    // OUTSIDE hull
      5.0
    );

    assert.equal(result.category, 'INVALID',
      `Expected INVALID for outside hull, got ${result.category}`);
  });

  it('classifies TRANSITIONING when phase shift active', () => {
    const classifier = new StateClassifier();

    const result = classifier.classify(
      [0.5, 0.3, 0.2, 0.1] as Vector4D,
      0.3,
      { activeAxis: 'alpha' as const, weights: [0.4, 0.4, 0.2] as [number, number, number], tension: 0.6, phaseProgress: 0.5 },
      null,
      0.5,
      true,
      0.3
    );

    assert.equal(result.category, 'TRANSITIONING',
      `Expected TRANSITIONING with phaseProgress=0.5, got ${result.category}`);
  });
});

describe('StateClassifier History', () => {
  it('tracks history', () => {
    const classifier = new StateClassifier();

    // Classify 3 times
    for (let i = 0; i < 3; i++) {
      classifier.classify(
        [0, 0, 0, 0] as Vector4D,
        0.05,
        { activeAxis: 'alpha' as const, weights: [0.5, 0.3, 0.2] as [number, number, number], tension: 0.1, phaseProgress: 0 },
        null,
        0.8,
        true,
        0.1
      );
    }

    const history = classifier.getHistory();
    assert.equal(history.length, 3);
  });

  it('reset clears history', () => {
    const classifier = new StateClassifier();

    classifier.classify(
      [0, 0, 0, 0] as Vector4D, 0.05,
      { activeAxis: 'alpha' as const, weights: [0.5, 0.3, 0.2] as [number, number, number], tension: 0.1, phaseProgress: 0 },
      null, 0.8, true, 0.1
    );

    classifier.reset();
    assert.equal(classifier.getHistory().length, 0);
  });
});
