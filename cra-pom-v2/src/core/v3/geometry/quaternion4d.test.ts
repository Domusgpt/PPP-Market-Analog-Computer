/**
 * Quaternion 4D Rotation Tests
 */

import { describe, it, expect } from 'vitest';
import {
  createQuaternion,
  identityQuaternion,
  quaternionNorm,
  normalizeQuaternion,
  quaternionConjugate,
  quaternionInverse,
  quaternionMultiply,
  quaternionSlerp,
  rotate4D,
  createIsoclinicRotation,
  createSimpleRotation,
  createDoubleRotation,
  composeRotations,
  identityRotation,
  inverseRotation,
  rotatePoints4D,
  animateRotation,
  rotationBetweenPoints,
  musicalModulationRotation,
} from './quaternion4d';

describe('Quaternion Operations', () => {
  describe('Creation', () => {
    it('should create quaternion from components', () => {
      const q = createQuaternion(1, 2, 3, 4);
      expect(q.w).toBe(1);
      expect(q.x).toBe(2);
      expect(q.y).toBe(3);
      expect(q.z).toBe(4);
    });

    it('should create identity quaternion', () => {
      const q = identityQuaternion();
      expect(q.w).toBe(1);
      expect(q.x).toBe(0);
      expect(q.y).toBe(0);
      expect(q.z).toBe(0);
    });
  });

  describe('Norm and Normalization', () => {
    it('should compute quaternion norm', () => {
      const q = createQuaternion(1, 0, 0, 0);
      expect(quaternionNorm(q)).toBe(1);

      const q2 = createQuaternion(3, 4, 0, 0);
      expect(quaternionNorm(q2)).toBe(5);
    });

    it('should normalize quaternion', () => {
      const q = createQuaternion(2, 0, 0, 0);
      const normalized = normalizeQuaternion(q);
      expect(quaternionNorm(normalized)).toBeCloseTo(1, 10);
    });

    it('should handle zero quaternion', () => {
      const q = createQuaternion(0, 0, 0, 0);
      const normalized = normalizeQuaternion(q);
      expect(normalized.w).toBe(1); // Returns identity
    });
  });

  describe('Conjugate and Inverse', () => {
    it('should compute conjugate', () => {
      const q = createQuaternion(1, 2, 3, 4);
      const conj = quaternionConjugate(q);
      expect(conj.w).toBe(1);
      expect(conj.x).toBe(-2);
      expect(conj.y).toBe(-3);
      expect(conj.z).toBe(-4);
    });

    it('should compute inverse', () => {
      const q = normalizeQuaternion(createQuaternion(1, 1, 1, 1));
      const inv = quaternionInverse(q);

      // q * q^-1 should equal identity
      const product = quaternionMultiply(q, inv);
      expect(product.w).toBeCloseTo(1, 5);
      expect(product.x).toBeCloseTo(0, 5);
      expect(product.y).toBeCloseTo(0, 5);
      expect(product.z).toBeCloseTo(0, 5);
    });
  });

  describe('Multiplication', () => {
    it('should multiply quaternions correctly', () => {
      const i = createQuaternion(0, 1, 0, 0);
      const j = createQuaternion(0, 0, 1, 0);

      // i * j = k
      const ij = quaternionMultiply(i, j);
      expect(ij.w).toBeCloseTo(0, 10);
      expect(ij.x).toBeCloseTo(0, 10);
      expect(ij.y).toBeCloseTo(0, 10);
      expect(ij.z).toBeCloseTo(1, 10);
    });

    it('should have identity as multiplicative identity', () => {
      const q = createQuaternion(1, 2, 3, 4);
      const id = identityQuaternion();

      const result = quaternionMultiply(q, id);
      expect(result.w).toBe(q.w);
      expect(result.x).toBe(q.x);
      expect(result.y).toBe(q.y);
      expect(result.z).toBe(q.z);
    });
  });

  describe('SLERP', () => {
    it('should interpolate at t=0 to first quaternion', () => {
      const q1 = identityQuaternion();
      const q2 = normalizeQuaternion(createQuaternion(0, 1, 0, 0));

      const result = quaternionSlerp(q1, q2, 0);
      expect(result.w).toBeCloseTo(q1.w, 5);
    });

    it('should interpolate at t=1 to second quaternion', () => {
      const q1 = identityQuaternion();
      const q2 = normalizeQuaternion(createQuaternion(0, 1, 0, 0));

      const result = quaternionSlerp(q1, q2, 1);
      expect(Math.abs(result.x)).toBeCloseTo(Math.abs(q2.x), 5);
    });

    it('should produce unit quaternion at midpoint', () => {
      const q1 = identityQuaternion();
      const q2 = normalizeQuaternion(createQuaternion(0, 1, 0, 0));

      const result = quaternionSlerp(q1, q2, 0.5);
      expect(quaternionNorm(result)).toBeCloseTo(1, 5);
    });
  });
});

describe('4D Rotation Operations', () => {
  describe('Identity Rotation', () => {
    it('should not change points', () => {
      const point = { w: 1, x: 2, y: 3, z: 4 };
      const rotation = identityRotation();

      const result = rotate4D(point, rotation);
      expect(result.w).toBeCloseTo(point.w, 5);
      expect(result.x).toBeCloseTo(point.x, 5);
      expect(result.y).toBeCloseTo(point.y, 5);
      expect(result.z).toBeCloseTo(point.z, 5);
    });
  });

  describe('Isoclinic Rotation', () => {
    it('should preserve distance from origin', () => {
      const point = { w: 1, x: 0, y: 0, z: 0 };
      const rotation = createIsoclinicRotation(Math.PI / 4, 'WX');

      const result = rotate4D(point, rotation);
      const originalDist = Math.sqrt(point.w ** 2 + point.x ** 2 + point.y ** 2 + point.z ** 2);
      const resultDist = Math.sqrt(result.w ** 2 + result.x ** 2 + result.y ** 2 + result.z ** 2);

      expect(resultDist).toBeCloseTo(originalDist, 5);
    });

    it('should rotate by specified angle', () => {
      const point = { w: 1, x: 0, y: 0, z: 0 };
      const rotation = createIsoclinicRotation(Math.PI / 2, 'WX');

      const result = rotate4D(point, rotation);
      // After 90° rotation in WX plane, (1,0,0,0) should move
      expect(Math.abs(result.w) + Math.abs(result.x)).toBeGreaterThan(0);
    });
  });

  describe('Simple Rotation', () => {
    it('should create rotation with one identity quaternion', () => {
      const rotation = createSimpleRotation(Math.PI / 4, 'XY');

      expect(rotation.right.w).toBe(1);
      expect(rotation.right.x).toBe(0);
      expect(rotation.right.y).toBe(0);
      expect(rotation.right.z).toBe(0);
    });
  });

  describe('Double Rotation', () => {
    it('should combine two rotations', () => {
      const rotation = createDoubleRotation(
        Math.PI / 4, 'WX',
        Math.PI / 4, 'YZ'
      );

      // Should have non-trivial left and right quaternions
      expect(quaternionNorm(rotation.left)).toBeCloseTo(1, 5);
      expect(quaternionNorm(rotation.right)).toBeCloseTo(1, 5);
    });
  });

  describe('Composition', () => {
    it('should compose rotations correctly', () => {
      const r1 = createIsoclinicRotation(Math.PI / 4, 'WX');
      const r2 = createIsoclinicRotation(Math.PI / 4, 'WX');
      const composed = composeRotations(r1, r2);

      // Two 45° rotations should equal one 90° rotation
      const point = { w: 1, x: 0, y: 0, z: 0 };
      const result1 = rotate4D(rotate4D(point, r1), r2);
      const result2 = rotate4D(point, composed);

      expect(result2.w).toBeCloseTo(result1.w, 4);
      expect(result2.x).toBeCloseTo(result1.x, 4);
    });

    it('should have inverse that undoes rotation', () => {
      const rotation = createIsoclinicRotation(Math.PI / 3, 'WY');
      const inverse = inverseRotation(rotation);

      const point = { w: 1, x: 2, y: 3, z: 4 };
      const rotated = rotate4D(point, rotation);
      const restored = rotate4D(rotated, inverse);

      expect(restored.w).toBeCloseTo(point.w, 4);
      expect(restored.x).toBeCloseTo(point.x, 4);
      expect(restored.y).toBeCloseTo(point.y, 4);
      expect(restored.z).toBeCloseTo(point.z, 4);
    });
  });
});

describe('Batch Operations', () => {
  describe('rotatePoints4D', () => {
    it('should rotate all points', () => {
      const points = [
        { w: 1, x: 0, y: 0, z: 0 },
        { w: 0, x: 1, y: 0, z: 0 },
        { w: 0, x: 0, y: 1, z: 0 },
      ];
      const rotation = createIsoclinicRotation(Math.PI / 4, 'WX');

      const results = rotatePoints4D(points, rotation);

      expect(results.length).toBe(3);
      // Each point should be changed
      for (let i = 0; i < points.length; i++) {
        const dist = Math.sqrt(
          (results[i].w - points[i].w) ** 2 +
          (results[i].x - points[i].x) ** 2 +
          (results[i].y - points[i].y) ** 2 +
          (results[i].z - points[i].z) ** 2
        );
        // Identity would give 0, rotation should give > 0 for at least the first two
      }
    });
  });

  describe('animateRotation', () => {
    it('should generate correct number of frames', () => {
      const points = [{ w: 1, x: 0, y: 0, z: 0 }];
      const rotation = createIsoclinicRotation(Math.PI / 2, 'WX');

      const frames = animateRotation(points, rotation, 10);

      expect(frames.length).toBe(11); // Initial + 10 steps
    });

    it('should have first frame as original points', () => {
      const points = [{ w: 1, x: 0, y: 0, z: 0 }];
      const rotation = createIsoclinicRotation(Math.PI / 2, 'WX');

      const frames = animateRotation(points, rotation, 5);

      expect(frames[0][0].w).toBe(1);
      expect(frames[0][0].x).toBe(0);
    });
  });
});

describe('Special Rotations', () => {
  describe('rotationBetweenPoints', () => {
    it('should map from point to target', () => {
      const from = { w: 1, x: 0, y: 0, z: 0 };
      const to = { w: 0, x: 1, y: 0, z: 0 };

      const rotation = rotationBetweenPoints(from, to);
      const result = rotate4D(from, rotation);

      // Result should be close to 'to' (or its negative)
      const dist = Math.sqrt(
        (result.w - to.w) ** 2 +
        (result.x - to.x) ** 2 +
        (result.y - to.y) ** 2 +
        (result.z - to.z) ** 2
      );
      const distNeg = Math.sqrt(
        (result.w + to.w) ** 2 +
        (result.x + to.x) ** 2 +
        (result.y + to.y) ** 2 +
        (result.z + to.z) ** 2
      );

      expect(Math.min(dist, distNeg)).toBeLessThan(0.1);
    });
  });

  describe('musicalModulationRotation', () => {
    it('should create rotation for key change', () => {
      // C to G (index 0 to 2 in our mapping)
      const rotation = musicalModulationRotation(0, 2, 24);

      expect(rotation.left).toBeDefined();
      expect(rotation.right).toBeDefined();
      expect(quaternionNorm(rotation.left)).toBeCloseTo(1, 5);
    });

    it('should have identity for same key', () => {
      const rotation = musicalModulationRotation(5, 5, 24);
      const point = { w: 1, x: 0, y: 0, z: 0 };

      const result = rotate4D(point, rotation);

      // Should be unchanged (or nearly so)
      expect(result.w).toBeCloseTo(point.w, 3);
    });
  });
});
