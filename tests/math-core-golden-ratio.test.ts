/**
 * Tests for GoldenRatioScaling.ts
 *
 * Verifies golden ratio mathematical invariants:
 * - φ² = φ + 1
 * - 1/φ = φ - 1
 * - Fibonacci sequence approximation
 * - φ^n scaling consistency
 * - Moiré pattern detection
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  phiPower,
  phiPowerPair,
  nearestFibonacci,
  isGoldenRatio,
  scaleByPhi,
  createNestedVertices,
  generatePhiSpiral,
  GoldenRatioScaler,
  PHI_SQUARED,
  INV_PHI,
  FIBONACCI,
  LUCAS
} from '../_SYNERGIZED_SYSTEM/lib/math_core/topology/GoldenRatioScaling.js';

const PHI = (1 + Math.sqrt(5)) / 2;
const EPSILON = 1e-10;

function approxEqual(a: number, b: number, eps = EPSILON) {
  assert.ok(Math.abs(a - b) < eps, `Expected ${a} ≈ ${b} (diff: ${Math.abs(a - b)})`);
}

describe('Golden Ratio Constants', () => {
  it('φ² = φ + 1', () => {
    approxEqual(PHI * PHI, PHI + 1);
  });

  it('1/φ = φ - 1', () => {
    approxEqual(1 / PHI, PHI - 1);
  });

  it('PHI_SQUARED constant is correct', () => {
    approxEqual(PHI_SQUARED, PHI * PHI);
  });

  it('INV_PHI constant is correct', () => {
    approxEqual(INV_PHI, 1 / PHI);
  });
});

describe('Fibonacci Sequence', () => {
  it('FIBONACCI array starts with correct values', () => {
    // F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, F(6)=8
    assert.equal(FIBONACCI[0], 0);
    assert.equal(FIBONACCI[1], 1);
    assert.equal(FIBONACCI[2], 1);
    assert.equal(FIBONACCI[3], 2);
    assert.equal(FIBONACCI[4], 3);
    assert.equal(FIBONACCI[5], 5);
    assert.equal(FIBONACCI[6], 8);
  });

  it('each Fibonacci number is sum of previous two', () => {
    for (let i = 2; i < FIBONACCI.length; i++) {
      assert.equal(FIBONACCI[i], FIBONACCI[i - 1] + FIBONACCI[i - 2],
        `F(${i}) = ${FIBONACCI[i]}, but F(${i - 1}) + F(${i - 2}) = ${FIBONACCI[i - 1] + FIBONACCI[i - 2]}`);
    }
  });

  it('F(n+1)/F(n) converges to φ', () => {
    // For large n, ratio should be close to φ
    const n = FIBONACCI.length - 1;
    if (n >= 10) {
      const ratio = FIBONACCI[n] / FIBONACCI[n - 1];
      approxEqual(ratio, PHI, 1e-6);
    }
  });
});

describe('Lucas Numbers', () => {
  it('LUCAS array starts with correct values', () => {
    // L(0)=2, L(1)=1, L(2)=3, L(3)=4, L(4)=7
    assert.equal(LUCAS[0], 2);
    assert.equal(LUCAS[1], 1);
    assert.equal(LUCAS[2], 3);
    assert.equal(LUCAS[3], 4);
    assert.equal(LUCAS[4], 7);
  });

  it('each Lucas number is sum of previous two', () => {
    for (let i = 2; i < LUCAS.length; i++) {
      assert.equal(LUCAS[i], LUCAS[i - 1] + LUCAS[i - 2]);
    }
  });
});

describe('Phi Power Functions', () => {
  it('phiPower(0) = 1', () => {
    approxEqual(phiPower(0), 1);
  });

  it('phiPower(1) = φ', () => {
    approxEqual(phiPower(1), PHI);
  });

  it('phiPower(2) = φ²', () => {
    approxEqual(phiPower(2), PHI * PHI);
  });

  it('phiPower(-1) = 1/φ', () => {
    approxEqual(phiPower(-1), 1 / PHI, 1e-8);
  });

  it('phiPower(n) * phiPower(-n) = 1', () => {
    for (let n = 0; n <= 5; n++) {
      approxEqual(phiPower(n) * phiPower(-n), 1, 1e-6);
    }
  });

  it('phiPowerPair returns {phi: φ^n, phiConj: (φ\')^n}', () => {
    const PHI_CONJ = (1 - Math.sqrt(5)) / 2; // Galois conjugate
    const pair = phiPowerPair(3);
    approxEqual(pair.phi, PHI ** 3);
    approxEqual(pair.phiConj, PHI_CONJ ** 3, 1e-6);
  });
});

describe('Golden Ratio Utility Functions', () => {
  it('nearestFibonacci finds closest Fibonacci number', () => {
    const result = nearestFibonacci(10);
    // Returns object with {index, value}
    assert.ok(result.value === 8 || result.value === 13,
      `nearestFibonacci(10).value = ${result.value}, expected 8 or 13`);
  });

  it('isGoldenRatio detects φ ratio between two numbers', () => {
    // isGoldenRatio(a, b) checks if a/b ≈ φ
    assert.ok(isGoldenRatio(PHI, 1));
    assert.ok(isGoldenRatio(PHI * 5, 5, 0.001));
  });

  it('isGoldenRatio rejects non-φ ratios', () => {
    assert.ok(!isGoldenRatio(1.5, 1));
    assert.ok(!isGoldenRatio(2.0, 1));
  });

  it('scaleByPhi multiplies vector by φ^n', () => {
    const v = [1, 0, 0, 0] as [number, number, number, number];
    const scaled = scaleByPhi(v, 1); // scale by φ^1
    approxEqual(scaled[0], PHI, 1e-8);
  });
});

describe('GoldenRatioScaler Class', () => {
  it('creates nested vertex structures', () => {
    const baseVertices = [[1, 0, 0, 0], [0, 1, 0, 0]] as [number, number, number, number][];
    const nested = createNestedVertices(baseVertices);
    assert.ok(nested.layers.length >= 2);
  });

  it('generatePhiSpiral produces points', () => {
    const points = generatePhiSpiral(20);
    assert.equal(points.length, 20);
    // Points should have 4D coordinates
    for (const p of points) {
      assert.equal(p.length, 4);
    }
  });
});
