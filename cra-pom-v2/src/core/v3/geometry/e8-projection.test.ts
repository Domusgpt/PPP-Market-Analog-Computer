/**
 * E8 Projection and φ-Scaling Tests
 *
 * Verifies the two-layer golden ratio system:
 * - Layer 1 at unit scale (abstract/conceptual)
 * - Layer 2 at φ-scale (concrete/physical)
 * - Alignment detection between layers
 * - E8 lattice generation and projection
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  E8Projection,
  createMusicalE8,
  createUniformLayer1,
} from './e8-projection';
import { PHI, PHI_INV } from './cell600';
import { distance4D } from '../music/polytopes';

describe('E8Projection', () => {
  let projection: E8Projection;

  beforeEach(() => {
    projection = new E8Projection();
  });

  describe('Structure', () => {
    it('should have 240 total vertices (120 per layer)', () => {
      expect(projection.vertexCount).toBe(240);
    });

    it('should have 120 layer 1 vertices at unit scale', () => {
      const layer1 = projection.getLayer1Vertices();
      expect(layer1.length).toBe(120);

      // Check unit scale (vertices at distance ~1 from origin)
      for (const v of layer1) {
        const dist = Math.sqrt(v.w ** 2 + v.x ** 2 + v.y ** 2 + v.z ** 2);
        expect(dist).toBeCloseTo(1, 4);
      }
    });

    it('should have 120 layer 2 vertices at φ scale', () => {
      const layer2 = projection.getLayer2Vertices();
      expect(layer2.length).toBe(120);

      // Check φ scale (vertices at distance ~φ from origin)
      for (const v of layer2) {
        const dist = Math.sqrt(v.w ** 2 + v.x ** 2 + v.y ** 2 + v.z ** 2);
        expect(dist).toBeCloseTo(PHI, 4);
      }
    });

    it('should scale layer 2 vertices by golden ratio', () => {
      const v1 = projection.getLayer1Vertex(0);
      const v2 = projection.getLayer2Vertex(0);

      expect(v2.w).toBeCloseTo(v1.w * PHI, 10);
      expect(v2.x).toBeCloseTo(v1.x * PHI, 10);
      expect(v2.y).toBeCloseTo(v1.y * PHI, 10);
      expect(v2.z).toBeCloseTo(v1.z * PHI, 10);
    });
  });

  describe('Golden Ratio Properties', () => {
    it('should use correct golden ratio value', () => {
      expect(PHI).toBeCloseTo(1.618033988749895, 10);
    });

    it('should use correct golden ratio inverse', () => {
      expect(PHI_INV).toBeCloseTo(0.618033988749895, 10);
    });

    it('should satisfy φ = 1 + 1/φ', () => {
      expect(PHI).toBeCloseTo(1 + PHI_INV, 10);
    });

    it('should satisfy φ² = φ + 1', () => {
      expect(PHI * PHI).toBeCloseTo(PHI + 1, 10);
    });
  });

  describe('Alignment Detection', () => {
    it('should find alignment points between layers', () => {
      const alignments = projection.findAlignmentPoints();
      expect(alignments.length).toBe(120);
    });

    it('should classify alignment types correctly', () => {
      const alignments = projection.findAlignmentPoints();
      const types = new Set(alignments.map(a => a.type));
      expect(types.size).toBeGreaterThan(0);
    });

    it('should have alignment strength between 0 and 1', () => {
      const alignments = projection.findAlignmentPoints();
      for (const align of alignments) {
        expect(align.alignmentStrength).toBeGreaterThanOrEqual(0);
        expect(align.alignmentStrength).toBeLessThanOrEqual(1);
      }
    });

    it('should find some strong alignments', () => {
      const strong = projection.getStrongAlignments(0.3);
      expect(strong.length).toBeGreaterThan(0);
    });

    it('should detect alignment near a point', () => {
      const v = projection.getLayer1Vertex(0);
      const alignment = projection.isNearAlignment(v, 0.5);
      // May or may not find alignment depending on structure
      if (alignment) {
        expect(alignment.alignmentStrength).toBeGreaterThan(0);
      }
    });
  });

  describe('Layer Blending', () => {
    it('should blend layers with factor 0 returning layer 1', () => {
      const v1 = projection.getLayer1Vertex(5);
      const result = projection.blendLayers(v1, 0);

      expect(result.layer1Weight).toBe(1);
      expect(result.layer2Weight).toBe(0);
      expect(result.blendedPosition.w).toBeCloseTo(v1.w, 10);
      expect(result.blendedPosition.x).toBeCloseTo(v1.x, 10);
      expect(result.blendedPosition.y).toBeCloseTo(v1.y, 10);
      expect(result.blendedPosition.z).toBeCloseTo(v1.z, 10);
    });

    it('should blend layers with factor 1 returning layer 2', () => {
      const v1 = projection.getLayer1Vertex(5);
      const v2 = projection.getLayer2Vertex(5);
      const result = projection.blendLayers(v1, 1);

      expect(result.layer1Weight).toBe(0);
      expect(result.layer2Weight).toBe(1);
      expect(result.blendedPosition.w).toBeCloseTo(v2.w, 10);
      expect(result.blendedPosition.x).toBeCloseTo(v2.x, 10);
      expect(result.blendedPosition.y).toBeCloseTo(v2.y, 10);
      expect(result.blendedPosition.z).toBeCloseTo(v2.z, 10);
    });

    it('should blend layers at midpoint with factor 0.5', () => {
      const v1 = projection.getLayer1Vertex(10);
      const v2 = projection.getLayer2Vertex(10);
      const result = projection.blendLayers(v1, 0.5);

      expect(result.layer1Weight).toBe(0.5);
      expect(result.layer2Weight).toBe(0.5);
      expect(result.blendedPosition.w).toBeCloseTo((v1.w + v2.w) / 2, 10);
      expect(result.blendedPosition.x).toBeCloseTo((v1.x + v2.x) / 2, 10);
    });

    it('should clamp blend factor to [0, 1]', () => {
      const v1 = projection.getLayer1Vertex(0);
      const result1 = projection.blendLayers(v1, -0.5);
      const result2 = projection.blendLayers(v1, 1.5);

      expect(result1.layer1Weight).toBe(1);
      expect(result1.layer2Weight).toBe(0);
      expect(result2.layer1Weight).toBe(0);
      expect(result2.layer2Weight).toBe(1);
    });

    it('should compute resonance in blend result', () => {
      const v1 = projection.getLayer1Vertex(0);
      const result = projection.blendLayers(v1, 0.5);

      expect(result.resonance).toBeGreaterThanOrEqual(0);
      expect(result.resonance).toBeLessThanOrEqual(1);
    });
  });

  describe('Golden Blend', () => {
    it('should use golden ratio weights', () => {
      const v1 = projection.getLayer1Vertex(0);
      const result = projection.goldenBlend(v1);

      // Weights should be proportional to φ⁻¹ and φ⁻²
      const expectedW1 = PHI_INV / (PHI_INV + PHI_INV * PHI_INV);
      const expectedW2 = (PHI_INV * PHI_INV) / (PHI_INV + PHI_INV * PHI_INV);

      expect(result.layer1Weight).toBeCloseTo(expectedW1, 10);
      expect(result.layer2Weight).toBeCloseTo(expectedW2, 10);
    });

    it('should have weights summing to 1', () => {
      const v1 = projection.getLayer1Vertex(0);
      const result = projection.goldenBlend(v1);

      expect(result.layer1Weight + result.layer2Weight).toBeCloseTo(1, 10);
    });
  });

  describe('Activation State', () => {
    it('should initialize with zero activations', () => {
      const state = projection.getState();
      expect(state.layer1Activations.every(a => a === 0)).toBe(true);
      expect(state.layer2Activations.every(a => a === 0)).toBe(true);
    });

    it('should activate layer 1 vertex', () => {
      projection.activateLayer1(50, 0.8);
      const state = projection.getState();

      expect(state.layer1Activations[50]).toBe(0.8);
      expect(state.layer1Activations[49]).toBe(0);
    });

    it('should activate layer 2 vertex', () => {
      projection.activateLayer2(30, 0.6);
      const state = projection.getState();

      expect(state.layer2Activations[30]).toBe(0.6);
    });

    it('should clamp activations to [0, 1]', () => {
      projection.activateLayer1(0, 2.5);
      projection.activateLayer2(0, -1);
      const state = projection.getState();

      expect(state.layer1Activations[0]).toBe(1);
      expect(state.layer2Activations[0]).toBe(0);
    });

    it('should set all layer activations', () => {
      const activations = new Array(120).fill(0.5);
      activations[0] = 1;
      activations[119] = 0.1;

      projection.setLayer1Activations(activations);
      const state = projection.getState();

      expect(state.layer1Activations[0]).toBe(1);
      expect(state.layer1Activations[50]).toBe(0.5);
      expect(state.layer1Activations[119]).toBe(0.1);
    });

    it('should throw for wrong activation array length', () => {
      expect(() => projection.setLayer1Activations([1, 2, 3])).toThrow();
      expect(() => projection.setLayer2Activations([])).toThrow();
    });

    it('should reset all activations', () => {
      projection.activateLayer1(0, 1);
      projection.activateLayer2(0, 1);
      projection.reset();

      const state = projection.getState();
      expect(state.layer1Activations.every(a => a === 0)).toBe(true);
      expect(state.layer2Activations.every(a => a === 0)).toBe(true);
    });
  });

  describe('Centroid Computation', () => {
    it('should compute zero centroid when no activations', () => {
      const centroid = projection.getLayer1Centroid();
      expect(centroid.w).toBe(0);
      expect(centroid.x).toBe(0);
      expect(centroid.y).toBe(0);
      expect(centroid.z).toBe(0);
    });

    it('should compute centroid of single active vertex', () => {
      projection.activateLayer1(0, 1);
      const centroid = projection.getLayer1Centroid();
      const v = projection.getLayer1Vertex(0);

      expect(centroid.w).toBeCloseTo(v.w, 10);
      expect(centroid.x).toBeCloseTo(v.x, 10);
      expect(centroid.y).toBeCloseTo(v.y, 10);
      expect(centroid.z).toBeCloseTo(v.z, 10);
    });

    it('should compute weighted centroid', () => {
      projection.activateLayer1(0, 0.75);
      projection.activateLayer1(1, 0.25);

      const v0 = projection.getLayer1Vertex(0);
      const v1 = projection.getLayer1Vertex(1);
      const centroid = projection.getLayer1Centroid();

      const expectedW = (v0.w * 0.75 + v1.w * 0.25);
      expect(centroid.w).toBeCloseTo(expectedW, 10);
    });
  });

  describe('E8 Lattice', () => {
    it('should generate 240 E8 roots', () => {
      const roots = E8Projection.generateE8Roots();
      expect(roots.length).toBe(240);
    });

    it('should have roots of norm √2', () => {
      const roots = E8Projection.generateE8Roots();

      for (const root of roots) {
        const normSq = root.x0 ** 2 + root.x1 ** 2 + root.x2 ** 2 + root.x3 ** 2 +
          root.x4 ** 2 + root.x5 ** 2 + root.x6 ** 2 + root.x7 ** 2;
        expect(normSq).toBeCloseTo(2, 10);
      }
    });

    it('should project E8 roots to 4D', () => {
      const roots = E8Projection.generateE8Roots();
      const projected = roots.map(r => E8Projection.projectTo4D(r));

      expect(projected.length).toBe(240);

      // Check all projections are valid 4D points
      for (const p of projected) {
        expect(typeof p.w).toBe('number');
        expect(typeof p.x).toBe('number');
        expect(typeof p.y).toBe('number');
        expect(typeof p.z).toBe('number');
        expect(Number.isFinite(p.w)).toBe(true);
      }
    });

    it('should create E8Projection from E8 lattice', () => {
      const fromLattice = E8Projection.fromE8Lattice();
      expect(fromLattice.vertexCount).toBe(240);
    });
  });

  describe('Musical Application', () => {
    it('should map musical key to 5 layer 1 vertices', () => {
      const vertices = projection.keyToLayer1Vertices(0);
      expect(vertices.length).toBe(5);

      // All indices should be valid
      for (const v of vertices) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(120);
      }
    });

    it('should map different keys to different vertices', () => {
      const key0 = projection.keyToLayer1Vertices(0);
      const key1 = projection.keyToLayer1Vertices(1);
      const key12 = projection.keyToLayer1Vertices(12);

      // Should have different sets
      const set0 = new Set(key0);
      const set1 = new Set(key1);

      let overlap = 0;
      for (const v of key1) {
        if (set0.has(v)) overlap++;
      }

      // Some overlap possible, but not complete
      expect(overlap).toBeLessThan(5);
    });

    it('should activate musical key across both layers', () => {
      projection.activateMusicalKey(5, 0.9, 0.4);
      const state = projection.getState();

      // Should have some activations in both layers
      const activeL1 = state.layer1Activations.filter(a => a > 0).length;
      const activeL2 = state.layer2Activations.filter(a => a > 0).length;

      expect(activeL1).toBeGreaterThan(0);
      expect(activeL2).toBeGreaterThan(0);
    });
  });

  describe('Total Resonance', () => {
    it('should have zero resonance when no activations', () => {
      const state = projection.getState();
      expect(state.totalResonance).toBe(0);
    });

    it('should compute resonance from aligned active vertices', () => {
      // Activate vertices in both layers
      for (let i = 0; i < 10; i++) {
        projection.activateLayer1(i, 1);
        projection.activateLayer2(i, 1);
      }

      const state = projection.getState();
      // Resonance depends on alignment, may be > 0
      expect(state.totalResonance).toBeGreaterThanOrEqual(0);
      expect(state.totalResonance).toBeLessThanOrEqual(1);
    });
  });
});

describe('Factory Functions', () => {
  it('should create musical E8 with active keys', () => {
    const projection = createMusicalE8([0, 7, 12]); // C, G, Am
    const state = projection.getState();

    const activeCount = state.layer1Activations.filter(a => a > 0).length;
    expect(activeCount).toBeGreaterThan(0);
  });

  it('should create uniform layer 1 activation', () => {
    const projection = createUniformLayer1(0.3);
    const state = projection.getState();

    expect(state.layer1Activations.every(a => a === 0.3)).toBe(true);
    expect(state.layer2Activations.every(a => a === 0)).toBe(true);
  });
});
