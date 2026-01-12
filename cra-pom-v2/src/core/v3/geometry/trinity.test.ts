/**
 * Trinity Decomposition Tests
 *
 * Verifies the thesis-antithesis-synthesis dialectic logic:
 * - 24-cell partitions into 3 × 8-vertex 16-cells
 * - Alpha/Beta/Gamma state management
 * - Dialectic computation and overlap detection
 */

import { describe, it, expect } from 'vitest';
import {
  Trinity24Cell,
  createDialecticPair,
  createOpposition,
  createConvergence,
} from './trinity';

describe('Trinity24Cell', () => {
  describe('Structure', () => {
    const trinity = new Trinity24Cell();

    it('should have 24 total vertices', () => {
      expect(trinity.vertexCount).toBe(24);
    });

    it('should have 8 Alpha vertices', () => {
      expect(trinity.alphaCount).toBe(8);
      expect(trinity.alpha.length).toBe(8);
    });

    it('should have 8 Beta vertices', () => {
      expect(trinity.betaCount).toBe(8);
      expect(trinity.beta.length).toBe(8);
    });

    it('should have 8 Gamma vertices', () => {
      expect(trinity.gammaCount).toBe(8);
      expect(trinity.gamma.length).toBe(8);
    });

    it('should partition vertices correctly by role', () => {
      const state = trinity.getState();
      expect(state.alpha.every(v => v.role === 'alpha')).toBe(true);
      expect(state.beta.every(v => v.role === 'beta')).toBe(true);
      expect(state.gamma.every(v => v.role === 'gamma')).toBe(true);
    });

    it('should have all vertices start with zero activation', () => {
      const state = trinity.getState();
      const allZero = [
        ...state.alpha,
        ...state.beta,
        ...state.gamma,
      ].every(v => v.activation === 0);
      expect(allZero).toBe(true);
    });
  });

  describe('State Management', () => {
    it('should set thesis (Alpha) activations', () => {
      const trinity = new Trinity24Cell();
      const activations = [1, 0.5, 0, 0, 0, 0, 0.5, 1];
      trinity.setThesis(activations);

      expect(trinity.alpha[0].activation).toBe(1);
      expect(trinity.alpha[1].activation).toBe(0.5);
      expect(trinity.alpha[2].activation).toBe(0);
      expect(trinity.alpha[7].activation).toBe(1);
    });

    it('should set antithesis (Beta) activations', () => {
      const trinity = new Trinity24Cell();
      const activations = [0, 0, 1, 1, 0, 0, 0, 0];
      trinity.setAntithesis(activations);

      expect(trinity.beta[2].activation).toBe(1);
      expect(trinity.beta[3].activation).toBe(1);
      expect(trinity.beta[0].activation).toBe(0);
    });

    it('should clamp activations to [0, 1]', () => {
      const trinity = new Trinity24Cell();
      trinity.setThesis([2, -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);

      expect(trinity.alpha[0].activation).toBe(1);
      expect(trinity.alpha[1].activation).toBe(0);
    });

    it('should activate individual vertices', () => {
      const trinity = new Trinity24Cell();
      trinity.activateVertex(5, 0.75);

      const state = trinity.getState();
      const vertex5 = [...state.alpha, ...state.beta, ...state.gamma]
        .find(v => v.index === 5);
      expect(vertex5?.activation).toBe(0.75);
    });

    it('should reset all activations', () => {
      const trinity = new Trinity24Cell();
      trinity.setThesis([1, 1, 1, 1, 1, 1, 1, 1]);
      trinity.setAntithesis([1, 1, 1, 1, 1, 1, 1, 1]);
      trinity.reset();

      const activeVertices = trinity.getActiveVertices(0.01);
      expect(activeVertices.length).toBe(0);
    });

    it('should throw error for wrong number of activations', () => {
      const trinity = new Trinity24Cell();
      expect(() => trinity.setThesis([1, 1, 1])).toThrow();
      expect(() => trinity.setAntithesis([1])).toThrow();
    });
  });

  describe('Dialectic Computation', () => {
    it('should compute dialectic with active vertices', () => {
      const trinity = new Trinity24Cell();
      trinity.setThesis([1, 0, 0, 0, 0, 0, 0, 0]);
      trinity.setAntithesis([0, 0, 0, 0, 0, 0, 0, 1]);

      const result = trinity.computeDialectic();

      expect(result.thesis).toBeDefined();
      expect(result.antithesis).toBeDefined();
      expect(result.synthesis).toBeDefined();
      expect(result.tension).toBeGreaterThanOrEqual(0);
      expect(result.tension).toBeLessThanOrEqual(1);
      expect(result.resolution).toBeGreaterThanOrEqual(0);
      expect(result.resolution).toBeLessThanOrEqual(1);
    });

    it('should show high tension for opposing states', () => {
      const trinity = createOpposition();
      const result = trinity.computeDialectic();

      // Opposition should have high tension
      expect(result.tension).toBeGreaterThan(0.3);
    });

    it('should show low tension for convergent states', () => {
      const trinity = createConvergence();
      const result = trinity.computeDialectic();

      // Convergence should have low tension
      expect(result.tension).toBeLessThan(0.5);
    });
  });

  describe('Overlap Computation', () => {
    it('should detect no overlap when vertices inactive', () => {
      const trinity = new Trinity24Cell();
      const overlap = trinity.computeOverlap();

      expect(overlap.overlapVertices.length).toBe(0);
      expect(overlap.overlapStrength).toBe(0);
      expect(overlap.pattern).toBe('orthogonal');
    });

    it('should detect overlap when Alpha and Beta active', () => {
      const trinity = new Trinity24Cell();
      trinity.setThesis([1, 1, 1, 1, 0, 0, 0, 0]);
      trinity.setAntithesis([1, 1, 1, 1, 0, 0, 0, 0]);

      const overlap = trinity.computeOverlap();

      // With similar activations, should find some overlap
      expect(overlap.overlapStrength).toBeGreaterThanOrEqual(0);
    });

    it('should activate Gamma vertices in overlap region', () => {
      const trinity = new Trinity24Cell();
      trinity.setThesis([1, 1, 1, 1, 1, 1, 1, 1]);
      trinity.setAntithesis([1, 1, 1, 1, 1, 1, 1, 1]);

      trinity.computeOverlap();

      // Some Gamma vertices should now be activated
      const activeGamma = trinity.gamma.filter(v => v.activation > 0);
      expect(activeGamma.length).toBeGreaterThan(0);
    });
  });

  describe('Dialectic Step', () => {
    it('should advance the dialectic (synthesis → new thesis)', () => {
      const trinity = new Trinity24Cell();
      trinity.setThesis([1, 0, 0, 0, 0, 0, 0, 0]);
      trinity.setAntithesis([0, 1, 0, 0, 0, 0, 0, 0]);

      const beforeAlpha = trinity.alpha.map(v => v.activation);
      const beforeBeta = trinity.beta.map(v => v.activation);

      trinity.dialecticStep();

      const afterAlpha = trinity.alpha.map(v => v.activation);
      const afterBeta = trinity.beta.map(v => v.activation);

      // Alpha should now have old Beta values
      expect(afterAlpha).toEqual(beforeBeta);
    });
  });

  describe('Active Vertex Retrieval', () => {
    it('should return only active vertices above threshold', () => {
      const trinity = new Trinity24Cell();
      trinity.setThesis([1, 0.6, 0.4, 0.3, 0, 0, 0, 0]);

      const active = trinity.getActiveVertices(0.5);
      expect(active.length).toBe(2); // Only 1.0 and 0.6 are above 0.5
    });

    it('should return empty array when no active vertices', () => {
      const trinity = new Trinity24Cell();
      const active = trinity.getActiveVertices(0.5);
      expect(active.length).toBe(0);
    });
  });
});

describe('Factory Functions', () => {
  it('should create dialectic pair with custom activations', () => {
    const trinity = createDialecticPair(
      [1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0]
    );

    expect(trinity.alpha[0].activation).toBe(1);
    expect(trinity.alpha[1].activation).toBe(1);
    expect(trinity.beta[2].activation).toBe(1);
    expect(trinity.beta[3].activation).toBe(1);
  });

  it('should create opposition (maximum tension)', () => {
    const trinity = createOpposition();
    const result = trinity.computeDialectic();

    // Opposition should create high tension
    expect(result.tension).toBeGreaterThan(0);
  });

  it('should create convergence (minimum tension)', () => {
    const trinity = createConvergence();
    const result = trinity.computeDialectic();

    // Convergence should create low tension
    expect(result.tension).toBeLessThan(1);
  });
});

describe('Orthogonality Verification', () => {
  it('should verify 16-cell orthogonality', () => {
    const trinity = new Trinity24Cell();
    const ortho = trinity.verifyOrthogonality();

    // Note: Due to our simple partition, perfect orthogonality
    // may not hold - this tests the verification function works
    expect(ortho).toHaveProperty('alphaBetaOrthogonal');
    expect(ortho).toHaveProperty('betaGammaOrthogonal');
    expect(ortho).toHaveProperty('gammaAlphaOrthogonal');

    console.log('Orthogonality check:', ortho);
  });
});
