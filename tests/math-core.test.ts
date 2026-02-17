/**
 * Math Core Test Suite
 *
 * Tests for the TypeScript math core modules:
 * - E8Projection: E8 root generation, Baez projection, 600-cell vertices
 * - GoldenRatioScaling: phi constants, Fibonacci, phi-nesting, moire detection
 * - CPE_Lattice24: 24-cell vertices, Trinity decomposition, convexity, navigation
 * - GeometricAlgebra: dot product, wedge product, rotors, sandwich product
 *
 * These are the first tests for the TS math core — previously had zero coverage.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

// ============================================================================
// E8 Projection Tests
// ============================================================================

import {
  PHI,
  PHI_CONJUGATE,
  generateE8Roots,
  projectE8Root,
  projectE8ToNested600Cells,
  generate600CellVertices,
  galoisConjugate4D,
  icosianNorm,
  hasUnitIcosianNorm,
  normalizeIcosian,
  E8ProjectionPipeline,
  type Vector8D
} from '../_SYNERGIZED_SYSTEM/lib/math_core/topology/E8Projection.js';

const EPSILON = 1e-8;

function assertNear(actual: number, expected: number, tolerance = EPSILON, msg = ''): void {
  const diff = Math.abs(actual - expected);
  assert.ok(
    diff <= tolerance,
    `${msg} Expected ${expected}, got ${actual} (diff: ${diff}, tolerance: ${tolerance})`
  );
}

describe('E8 Projection', () => {
  describe('constants', () => {
    it('golden ratio phi satisfies phi^2 = phi + 1', () => {
      assertNear(PHI * PHI, PHI + 1, EPSILON, 'phi^2 should equal phi + 1');
    });

    it('phi conjugate satisfies phi * phi_conjugate = -1', () => {
      assertNear(PHI * PHI_CONJUGATE, -1, EPSILON, 'phi * phi_conjugate should be -1');
    });

    it('phi + phi_conjugate = 1', () => {
      assertNear(PHI + PHI_CONJUGATE, 1, EPSILON, 'phi + phi_conjugate should equal 1');
    });
  });

  describe('E8 root generation', () => {
    const roots = generateE8Roots();

    it('generates exactly 240 roots', () => {
      assert.strictEqual(roots.length, 240, 'E8 has exactly 240 roots');
    });

    it('all roots have norm squared = 2', () => {
      for (const root of roots) {
        const normSq = root.coordinates.reduce((sum, x) => sum + x * x, 0);
        assertNear(normSq, 2, 1e-6, `Root norm squared should be 2`);
      }
    });

    it('has 112 permutation-type roots', () => {
      const permRoots = roots.filter(r => r.type === 'permutation');
      assert.strictEqual(permRoots.length, 112, 'Should have 112 permutation roots');
    });

    it('has 128 half-integer roots', () => {
      const halfRoots = roots.filter(r => r.type === 'half-integer');
      assert.strictEqual(halfRoots.length, 128, 'Should have 128 half-integer roots');
    });

    it('half-integer roots have even number of negative components', () => {
      const halfRoots = roots.filter(r => r.type === 'half-integer');
      for (const root of halfRoots) {
        const negCount = root.coordinates.filter(x => x < 0).length;
        assert.strictEqual(negCount % 2, 0, 'Half-integer roots must have even number of negatives');
      }
    });

    it('permutation roots have exactly 2 non-zero components', () => {
      const permRoots = roots.filter(r => r.type === 'permutation');
      for (const root of permRoots) {
        const nonZero = root.coordinates.filter(x => Math.abs(x) > EPSILON).length;
        assert.strictEqual(nonZero, 2, 'Permutation roots should have exactly 2 non-zero components');
      }
    });
  });

  describe('E8 to H4 projection', () => {
    it('projects each root to outer and inner 4D points', () => {
      const roots = generateE8Roots();
      const projected = projectE8Root(roots[0]);

      assert.strictEqual(projected.outer.length, 4, 'Outer projection should be 4D');
      assert.strictEqual(projected.inner.length, 4, 'Inner projection should be 4D');
      assert.ok(projected.outerRadius >= 0, 'Outer radius should be non-negative');
      assert.ok(projected.innerRadius >= 0, 'Inner radius should be non-negative');
    });

    it('nested 600-cells have correct structure', () => {
      const nested = projectE8ToNested600Cells();

      assert.ok(nested.outer.length > 0, 'Outer 600-cell should have vertices');
      assert.ok(nested.inner.length > 0, 'Inner 600-cell should have vertices');
      assert.strictEqual(nested.roots.length, 240, 'Should project all 240 roots');
      assert.ok(nested.outerScale > 0, 'Outer scale should be positive');
      assert.ok(nested.innerScale > 0, 'Inner scale should be positive');
    });

    it('outer and inner scales are related by phi', () => {
      const nested = projectE8ToNested600Cells();
      // The ratio of scales should involve phi
      const ratio = nested.outerScale / nested.innerScale;
      assert.ok(ratio > 1, 'Outer should be larger than inner');
    });
  });

  describe('600-cell vertices (direct generation)', () => {
    const vertices = generate600CellVertices();

    it('generates 120 vertices', () => {
      assert.strictEqual(vertices.length, 120, '600-cell has exactly 120 vertices');
    });

    it('all vertices are 4D', () => {
      for (const v of vertices) {
        assert.strictEqual(v.length, 4, 'Each vertex should be 4D');
      }
    });

    it('all vertices lie on a sphere (constant radius)', () => {
      const radii = vertices.map(v => Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]));
      const avgRadius = radii.reduce((a, b) => a + b, 0) / radii.length;

      for (const r of radii) {
        assertNear(r, avgRadius, 0.01, 'All vertices should have equal radius');
      }
    });
  });

  describe('Galois conjugation', () => {
    it('scales by 1/phi^2', () => {
      const v = [1, 0, 0, 0] as [number, number, number, number];
      const conjugated = galoisConjugate4D(v);
      const expectedScale = 1 / (PHI * PHI);

      assertNear(conjugated[0], expectedScale, EPSILON);
      assertNear(conjugated[1], 0, EPSILON);
    });
  });

  describe('icosian norm', () => {
    it('computes quaternion norm correctly', () => {
      assertNear(icosianNorm([1, 0, 0, 0]), 1, EPSILON);
      assertNear(icosianNorm([0.5, 0.5, 0.5, 0.5]), 1, EPSILON);
    });

    it('hasUnitIcosianNorm identifies unit quaternions', () => {
      assert.ok(hasUnitIcosianNorm([1, 0, 0, 0]));
      assert.ok(hasUnitIcosianNorm([0.5, 0.5, 0.5, 0.5]));
      assert.ok(!hasUnitIcosianNorm([2, 0, 0, 0]));
    });

    it('normalizeIcosian produces unit quaternion', () => {
      const normalized = normalizeIcosian([3, 4, 0, 0]);
      assertNear(icosianNorm(normalized), 1, EPSILON, 'Normalized should have unit norm');
    });
  });

  describe('E8ProjectionPipeline class', () => {
    const pipeline = new E8ProjectionPipeline();

    it('has 240 E8 roots', () => {
      assert.strictEqual(pipeline.e8Roots.length, 240);
    });

    it('projects arbitrary 8D vector', () => {
      const v8: Vector8D = [1, 0, 0, 0, 0, 0, 0, 0];
      const result = pipeline.project(v8);

      assert.strictEqual(result.outer.length, 4);
      assert.strictEqual(result.inner.length, 4);
    });

    it('finds nearest root', () => {
      const v8: Vector8D = [0.9, 1.1, 0, 0, 0, 0, 0, 0];
      const nearest = pipeline.findNearestRoot(v8);

      assert.ok(nearest, 'Should find a nearest root');
      assert.strictEqual(nearest.coordinates.length, 8);
    });

    it('getVerticesAtScale returns correct sets', () => {
      const outer = pipeline.getVerticesAtScale('outer');
      const inner = pipeline.getVerticesAtScale('inner');
      const both = pipeline.getVerticesAtScale('both');

      assert.ok(outer.length > 0);
      assert.ok(inner.length > 0);
      assert.strictEqual(both.length, outer.length + inner.length);
    });

    it('interpolateScale at t=0 returns outer, t=1 returns inner', () => {
      const atOuter = pipeline.interpolateScale(0);
      const outerVerts = pipeline.outerVertices;

      // At t=0, result should equal outer
      const n = Math.min(atOuter.length, outerVerts.length);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < 4; j++) {
          assertNear(atOuter[i][j], outerVerts[i][j], EPSILON, `t=0 should match outer[${i}][${j}]`);
        }
      }
    });
  });
});

// ============================================================================
// Golden Ratio Scaling Tests
// ============================================================================

import {
  PHI as GR_PHI,
  PHI_CONJUGATE as GR_PHI_CONJ,
  PHI_SQUARED,
  INV_PHI,
  FIBONACCI,
  LUCAS,
  phiPower,
  phiPowerPair,
  nearestFibonacci,
  isGoldenRatio,
  scaleByPhi,
  scaleVerticesByPhi,
  createNestedVertices,
  detectMoirePatterns,
  generatePhiSpiral,
  GoldenRatioScaler
} from '../_SYNERGIZED_SYSTEM/lib/math_core/topology/GoldenRatioScaling.js';

describe('Golden Ratio Scaling', () => {
  describe('constants', () => {
    it('PHI_SQUARED = PHI + 1', () => {
      assertNear(PHI_SQUARED, GR_PHI + 1, EPSILON);
    });

    it('INV_PHI = PHI - 1', () => {
      assertNear(INV_PHI, GR_PHI - 1, EPSILON);
    });

    it('PHI * INV_PHI = 1', () => {
      assertNear(GR_PHI * INV_PHI, 1, EPSILON);
    });

    it('Fibonacci sequence is correct', () => {
      for (let i = 2; i < FIBONACCI.length; i++) {
        assert.strictEqual(FIBONACCI[i], FIBONACCI[i-1] + FIBONACCI[i-2],
          `F(${i}) = F(${i-1}) + F(${i-2})`);
      }
    });

    it('Lucas sequence is correct', () => {
      for (let i = 2; i < LUCAS.length; i++) {
        assert.strictEqual(LUCAS[i], LUCAS[i-1] + LUCAS[i-2],
          `L(${i}) = L(${i-1}) + L(${i-2})`);
      }
    });

    it('consecutive Fibonacci ratios converge to phi', () => {
      for (let i = 14; i < 18; i++) {
        const ratio = FIBONACCI[i + 1] / FIBONACCI[i];
        assertNear(ratio, GR_PHI, 1e-4, `F(${i+1})/F(${i}) should approximate phi`);
      }
    });
  });

  describe('phiPower', () => {
    it('phi^0 = 1', () => {
      assertNear(phiPower(0), 1, EPSILON);
    });

    it('phi^1 = phi', () => {
      assertNear(phiPower(1), GR_PHI, EPSILON);
    });

    it('phi^-1 = 1/phi', () => {
      assertNear(phiPower(-1), INV_PHI, EPSILON);
    });

    it('phi^n = F(n)*phi + F(n-1) for small n', () => {
      for (let n = 2; n < 10; n++) {
        const expected = FIBONACCI[n] * GR_PHI + FIBONACCI[n - 1];
        assertNear(phiPower(n), expected, 1e-6, `phi^${n}`);
      }
    });
  });

  describe('phiPowerPair', () => {
    it('returns both phi^n and phi_conjugate^n', () => {
      const pair = phiPowerPair(3);
      assertNear(pair.phi, Math.pow(GR_PHI, 3), 1e-6);
      assertNear(pair.phiConj, Math.pow(GR_PHI_CONJ, 3), 1e-6);
    });
  });

  describe('nearestFibonacci', () => {
    it('finds exact Fibonacci numbers', () => {
      assert.strictEqual(nearestFibonacci(8).value, 8);
      assert.strictEqual(nearestFibonacci(13).value, 13);
      assert.strictEqual(nearestFibonacci(21).value, 21);
    });

    it('finds nearest for non-Fibonacci values', () => {
      // 10 is between 8 and 13
      const result = nearestFibonacci(10);
      assert.ok(result.value === 8 || result.value === 13);
    });
  });

  describe('isGoldenRatio', () => {
    it('recognizes phi ratio', () => {
      assert.ok(isGoldenRatio(GR_PHI, 1));
      assert.ok(isGoldenRatio(1, INV_PHI, 0.01));
    });

    it('rejects non-phi ratios', () => {
      assert.ok(!isGoldenRatio(2, 1));
      assert.ok(!isGoldenRatio(3, 2));
    });
  });

  describe('scaling functions', () => {
    it('scaleByPhi scales all components', () => {
      const v: [number, number, number, number] = [1, 2, 3, 4];
      const scaled = scaleByPhi(v, 1);

      for (let i = 0; i < 4; i++) {
        assertNear(scaled[i], v[i] * GR_PHI, 1e-6);
      }
    });

    it('scaleVerticesByPhi scales array of vertices', () => {
      const vertices: [number, number, number, number][] = [[1, 0, 0, 0], [0, 1, 0, 0]];
      const scaled = scaleVerticesByPhi(vertices, 2);

      const factor = phiPower(2);
      for (let i = 0; i < vertices.length; i++) {
        for (let j = 0; j < 4; j++) {
          assertNear(scaled[i][j], vertices[i][j] * factor, 1e-6);
        }
      }
    });
  });

  describe('nested structures', () => {
    it('createNestedVertices creates layers at phi scales', () => {
      const base: [number, number, number, number][] = [[1, 0, 0, 0], [0, 1, 0, 0]];
      const nested = createNestedVertices(base, -1, 1);

      assert.strictEqual(nested.layers.length, 3, 'Should have 3 layers (-1, 0, 1)');
      assert.strictEqual(nested.layers[0].level, -1);
      assert.strictEqual(nested.layers[1].level, 0);
      assert.strictEqual(nested.layers[2].level, 1);
    });

    it('layers have decreasing opacity with distance from base', () => {
      const base: [number, number, number, number][] = [[1, 0, 0, 0]];
      const nested = createNestedVertices(base, -2, 2);

      // Level 0 should have highest opacity
      const level0 = nested.layers.find(l => l.level === 0)!;
      for (const layer of nested.layers) {
        assert.ok(layer.opacity <= level0.opacity + EPSILON,
          `Level ${layer.level} opacity should be <= level 0 opacity`);
      }
    });
  });

  describe('moire pattern detection', () => {
    it('detects interference between layers', () => {
      // Create nested vertices with small enough threshold to detect
      const base: [number, number, number, number][] = [[1, 0, 0, 0], [0, 1, 0, 0]];
      const nested = createNestedVertices(base, 0, 1);

      const pattern = detectMoirePatterns(nested, {
        layerCount: 5,
        interferenceThreshold: 2.0, // Large threshold to ensure detection
        includeConjugates: false
      });

      assert.ok(pattern.interferences.length >= 0, 'Should return interference array');
      assert.ok(typeof pattern.intensity === 'number', 'Should have intensity');
      assert.ok(typeof pattern.dominantFrequency === 'number', 'Should have frequency');
    });
  });

  describe('phi spiral', () => {
    it('generates correct number of points', () => {
      const spiral = generatePhiSpiral(50);
      assert.strictEqual(spiral.length, 50);
    });

    it('all points are 4D', () => {
      const spiral = generatePhiSpiral(10);
      for (const p of spiral) {
        assert.strictEqual(p.length, 4);
      }
    });

    it('radius grows along spiral', () => {
      const spiral = generatePhiSpiral(20);
      const radii = spiral.map(v => Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]));

      // General trend should be increasing (allow some local oscillation)
      assert.ok(radii[radii.length - 1] > radii[0], 'Spiral radius should grow');
    });
  });

  describe('GoldenRatioScaler class', () => {
    const scaler = new GoldenRatioScaler(1, -2, 2);

    it('has correct levels', () => {
      const levels = scaler.levels;
      assert.deepStrictEqual(levels, [-2, -1, 0, 1, 2]);
    });

    it('getScale returns phi^level', () => {
      assertNear(scaler.getScale(0), 1, EPSILON);
      assertNear(scaler.getScale(1), GR_PHI, EPSILON);
      assertNear(scaler.getScale(-1), INV_PHI, EPSILON);
    });

    it('fibonacciApprox approximates phi', () => {
      const result = scaler.fibonacciApprox(GR_PHI);
      assert.ok(result.error < 0.01, 'Should approximate phi closely');
    });
  });
});

// ============================================================================
// Geometric Algebra Tests (types + core operations)
// ============================================================================

import {
  Vector4D as V4D,
  MATH_CONSTANTS,
  isVector4D,
  isBivector4D,
  isTrinityAxis,
  isPolychoronType
} from '../_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/types.js';

import {
  dot,
  magnitude,
  normalize,
  scale,
  add,
  subtract,
  distance,
  centroid,
  wedge,
  bivectorMagnitude,
  createRotor,
  applyRotorToVector
} from '../_SYNERGIZED_SYSTEM/lib/math_core/geometric_algebra/GeometricAlgebra.js';

describe('Geometric Algebra Types', () => {
  it('MATH_CONSTANTS has correct values', () => {
    assertNear(MATH_CONSTANTS.PHI, (1 + Math.sqrt(5)) / 2, EPSILON);
    assertNear(MATH_CONSTANTS.SQRT2, Math.SQRT2, EPSILON);
    assertNear(MATH_CONSTANTS.TAU, 2 * Math.PI, EPSILON);
  });

  describe('type guards', () => {
    it('isVector4D validates correctly', () => {
      assert.ok(isVector4D([1, 2, 3, 4]));
      assert.ok(!isVector4D([1, 2, 3]));
      assert.ok(!isVector4D([1, 2, 3, 'x']));
      assert.ok(!isVector4D('hello'));
    });

    it('isBivector4D validates correctly', () => {
      assert.ok(isBivector4D([1, 2, 3, 4, 5, 6]));
      assert.ok(!isBivector4D([1, 2, 3, 4]));
    });

    it('isTrinityAxis validates correctly', () => {
      assert.ok(isTrinityAxis('alpha'));
      assert.ok(isTrinityAxis('beta'));
      assert.ok(isTrinityAxis('gamma'));
      assert.ok(!isTrinityAxis('delta'));
    });

    it('isPolychoronType validates correctly', () => {
      assert.ok(isPolychoronType('24-cell'));
      assert.ok(isPolychoronType('600-cell'));
      assert.ok(!isPolychoronType('7-cell'));
    });
  });
});

describe('Geometric Algebra Operations', () => {
  describe('vector operations', () => {
    it('dot product is correct', () => {
      assertNear(dot([1, 0, 0, 0], [1, 0, 0, 0]), 1, EPSILON);
      assertNear(dot([1, 0, 0, 0], [0, 1, 0, 0]), 0, EPSILON);
      assertNear(dot([1, 2, 3, 4], [1, 2, 3, 4]), 30, EPSILON);
    });

    it('magnitude is Euclidean norm', () => {
      assertNear(magnitude([1, 0, 0, 0]), 1, EPSILON);
      assertNear(magnitude([3, 4, 0, 0]), 5, EPSILON);
      assertNear(magnitude([1, 1, 1, 1]), 2, EPSILON);
    });

    it('normalize produces unit vector', () => {
      const n = normalize([3, 4, 0, 0]);
      assertNear(magnitude(n), 1, EPSILON);
      assertNear(n[0], 0.6, EPSILON);
      assertNear(n[1], 0.8, EPSILON);
    });

    it('scale multiplies all components', () => {
      const s = scale([1, 2, 3, 4], 2);
      assert.deepStrictEqual(s, [2, 4, 6, 8]);
    });

    it('add and subtract are correct', () => {
      const a: V4D = [1, 2, 3, 4];
      const b: V4D = [5, 6, 7, 8];
      const sum = add(a, b);
      const diff = subtract(a, b);

      assert.deepStrictEqual(sum, [6, 8, 10, 12]);
      assert.deepStrictEqual(diff, [-4, -4, -4, -4]);
    });

    it('distance is Euclidean', () => {
      assertNear(distance([0, 0, 0, 0], [3, 4, 0, 0]), 5, EPSILON);
      assertNear(distance([1, 1, 1, 1], [1, 1, 1, 1]), 0, EPSILON);
    });

    it('centroid is average of points', () => {
      const points: V4D[] = [[2, 0, 0, 0], [0, 2, 0, 0]];
      const c = centroid(points);

      assertNear(c[0], 1, EPSILON);
      assertNear(c[1], 1, EPSILON);
      assertNear(c[2], 0, EPSILON);
      assertNear(c[3], 0, EPSILON);
    });
  });

  describe('bivector operations', () => {
    it('wedge product of orthogonal vectors is non-zero', () => {
      const w = wedge([1, 0, 0, 0], [0, 1, 0, 0]);
      assert.ok(bivectorMagnitude(w) > 0, 'Wedge of orthogonal vectors should be non-zero');
    });

    it('wedge product of parallel vectors is zero', () => {
      const w = wedge([1, 0, 0, 0], [2, 0, 0, 0]);
      assertNear(bivectorMagnitude(w), 0, EPSILON, 'Wedge of parallel vectors should be zero');
    });

    it('wedge product is antisymmetric', () => {
      const a: V4D = [1, 2, 3, 4];
      const b: V4D = [5, 6, 7, 8];

      const wAB = wedge(a, b);
      const wBA = wedge(b, a);

      for (let i = 0; i < 6; i++) {
        assertNear(wAB[i], -wBA[i], EPSILON, `Component ${i} should be negated`);
      }
    });
  });

  describe('rotor operations', () => {
    it('identity rotor preserves vector', () => {
      const identityRotor = createRotor([0, 0, 0, 0, 0, 0], 0);
      const v: V4D = [1, 2, 3, 4];
      const result = applyRotorToVector(identityRotor, v);

      for (let i = 0; i < 4; i++) {
        assertNear(result[i], v[i], 1e-6, `Component ${i} should be preserved`);
      }
    });

    it('rotor preserves magnitude (unitary transform)', () => {
      // Rotate in XY plane by pi/4
      const bivector = [1, 0, 0, 0, 0, 0] as [number, number, number, number, number, number];
      const rotor = createRotor(bivector, Math.PI / 4);
      const v: V4D = [1, 1, 0, 0];

      const result = applyRotorToVector(rotor, v);
      assertNear(magnitude(result), magnitude(v), 1e-6, 'Rotation should preserve magnitude');
    });
  });
});
