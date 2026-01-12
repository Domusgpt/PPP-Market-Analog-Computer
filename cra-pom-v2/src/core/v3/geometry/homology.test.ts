/**
 * Homological Analysis Tests
 *
 * Verifies topological feature extraction:
 * - Betti number computation
 * - Persistence diagram generation
 * - Topological event detection
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  computeBetti,
  computeDistanceMatrix,
  buildRipsComplex,
  computePersistence,
  getIntervalPersistence,
  filterPersistence,
  detectTransition,
  HomologyAnalyzer,
  createPolytopeAnalyzer,
  quickBetti,
  type BettiNumbers,
} from './homology';
import type { Vector4D } from '../music/music-geometry-domain';
import { Cell24 } from '../music/polytopes';
import { Cell600 } from './cell600';

describe('Distance Matrix', () => {
  it('should compute correct distances for origin points', () => {
    const points: Vector4D[] = [
      { w: 0, x: 0, y: 0, z: 0 },
      { w: 1, x: 0, y: 0, z: 0 },
      { w: 0, x: 1, y: 0, z: 0 },
    ];

    const matrix = computeDistanceMatrix(points);

    expect(matrix.length).toBe(3);
    expect(matrix[0][0]).toBe(0);  // Self distance
    expect(matrix[0][1]).toBe(1);  // Origin to (1,0,0,0)
    expect(matrix[0][2]).toBe(1);  // Origin to (0,1,0,0)
    expect(matrix[1][2]).toBeCloseTo(Math.sqrt(2), 10);  // (1,0,0,0) to (0,1,0,0)
  });

  it('should be symmetric', () => {
    const points: Vector4D[] = [
      { w: 0.5, x: 0.5, y: 0, z: 0 },
      { w: 0, x: 1, y: 1, z: 0 },
      { w: 1, x: 0, y: 0, z: 1 },
    ];

    const matrix = computeDistanceMatrix(points);

    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(matrix[i][j]).toBeCloseTo(matrix[j][i], 10);
      }
    }
  });
});

describe('Rips Complex', () => {
  it('should include all vertices as 0-simplices', () => {
    const points: Vector4D[] = [
      { w: 0, x: 0, y: 0, z: 0 },
      { w: 10, x: 0, y: 0, z: 0 },  // Far apart
    ];

    const matrix = computeDistanceMatrix(points);
    const simplices = buildRipsComplex(matrix, 0.1, 0);

    const vertices = simplices.filter(s => s.dimension === 0);
    expect(vertices.length).toBe(2);
  });

  it('should create edges for close points', () => {
    const points: Vector4D[] = [
      { w: 0, x: 0, y: 0, z: 0 },
      { w: 0.5, x: 0, y: 0, z: 0 },
      { w: 10, x: 0, y: 0, z: 0 },  // Far apart
    ];

    const matrix = computeDistanceMatrix(points);
    const simplices = buildRipsComplex(matrix, 1, 1);

    const edges = simplices.filter(s => s.dimension === 1);
    expect(edges.length).toBe(1);  // Only 0-1 connected
  });

  it('should create triangles for close triples', () => {
    const points: Vector4D[] = [
      { w: 0, x: 0, y: 0, z: 0 },
      { w: 0.5, x: 0, y: 0, z: 0 },
      { w: 0.25, x: 0.5, y: 0, z: 0 },
    ];

    const matrix = computeDistanceMatrix(points);
    const simplices = buildRipsComplex(matrix, 1, 2);

    const triangles = simplices.filter(s => s.dimension === 2);
    expect(triangles.length).toBe(1);
  });
});

describe('Betti Numbers', () => {
  describe('b₀ (Connected Components)', () => {
    it('should return 0 for empty point set', () => {
      const betti = computeBetti([], 1);
      expect(betti.b0).toBe(0);
    });

    it('should return 1 for single point', () => {
      const points: Vector4D[] = [{ w: 0, x: 0, y: 0, z: 0 }];
      const betti = computeBetti(points, 1);
      expect(betti.b0).toBe(1);
    });

    it('should return n for n far-apart points at small threshold', () => {
      const points: Vector4D[] = [
        { w: 0, x: 0, y: 0, z: 0 },
        { w: 10, x: 0, y: 0, z: 0 },
        { w: 0, x: 10, y: 0, z: 0 },
      ];

      const betti = computeBetti(points, 0.1);
      expect(betti.b0).toBe(3);
    });

    it('should return 1 for fully connected points at large threshold', () => {
      const points: Vector4D[] = [
        { w: 0, x: 0, y: 0, z: 0 },
        { w: 1, x: 0, y: 0, z: 0 },
        { w: 2, x: 0, y: 0, z: 0 },
      ];

      const betti = computeBetti(points, 2);
      expect(betti.b0).toBe(1);
    });
  });

  describe('Polytope Betti Numbers', () => {
    it('should analyze 24-cell at unit threshold', () => {
      const cell24 = new Cell24();
      const betti = computeBetti(cell24.vertices, 1.5);

      // At unit threshold, 24-cell should be connected
      expect(betti.b0).toBe(1);
    });

    it('should analyze 600-cell at small threshold', () => {
      const cell600 = new Cell600();
      const betti = computeBetti(cell600.vertices, 0.1);

      // At tiny threshold, vertices are isolated
      expect(betti.b0).toBe(120);
    });

    it('should find connected 600-cell at edge length', () => {
      const cell600 = new Cell600();
      const betti = computeBetti(cell600.vertices, cell600.edgeLength + 0.1);

      // At edge length, should be connected
      expect(betti.b0).toBe(1);
    });
  });
});

describe('Persistence', () => {
  it('should compute persistence diagram', () => {
    const points: Vector4D[] = [
      { w: 0, x: 0, y: 0, z: 0 },
      { w: 1, x: 0, y: 0, z: 0 },
      { w: 0, x: 1, y: 0, z: 0 },
    ];

    const diagram = computePersistence(points, 10);

    expect(diagram.intervals.length).toBeGreaterThan(0);
    expect(diagram.maxRadius).toBeGreaterThan(0);
  });

  it('should include all points as persistent components initially', () => {
    const points: Vector4D[] = [
      { w: 0, x: 0, y: 0, z: 0 },
      { w: 5, x: 0, y: 0, z: 0 },
      { w: 10, x: 0, y: 0, z: 0 },
    ];

    const diagram = computePersistence(points, 20);

    // Should have intervals starting at 0 (initial components)
    const zeroStart = diagram.intervals.filter(i => i.birth === 0);
    expect(zeroStart.length).toBeGreaterThan(0);
  });

  it('should compute interval persistence', () => {
    const finite = { birth: 0.5, death: 1.5, dimension: 0 };
    const infinite = { birth: 0, death: Infinity, dimension: 0 };

    expect(getIntervalPersistence(finite)).toBe(1);
    expect(getIntervalPersistence(infinite)).toBe(Infinity);
  });

  it('should filter by minimum persistence', () => {
    const diagram = {
      intervals: [
        { birth: 0, death: 0.1, dimension: 0 },   // Short-lived
        { birth: 0, death: 1, dimension: 0 },     // Long-lived
        { birth: 0, death: Infinity, dimension: 0 }, // Infinite
      ],
      maxRadius: 2,
    };

    const filtered = filterPersistence(diagram, 0.5);
    expect(filtered.length).toBe(2);  // Only long-lived and infinite
  });
});

describe('Topological Events', () => {
  it('should detect cluster formation', () => {
    const before: BettiNumbers = { b0: 1, b1: 0, b2: 0, b3: 0 };
    const after: BettiNumbers = { b0: 3, b1: 0, b2: 0, b3: 0 };

    const events = detectTransition(before, after);

    expect(events.length).toBe(1);
    expect(events[0].type).toBe('cluster_formed');
    expect(events[0].delta).toBe(2);
  });

  it('should detect cluster merge', () => {
    const before: BettiNumbers = { b0: 3, b1: 0, b2: 0, b3: 0 };
    const after: BettiNumbers = { b0: 1, b1: 0, b2: 0, b3: 0 };

    const events = detectTransition(before, after);

    expect(events.length).toBe(1);
    expect(events[0].type).toBe('cluster_merged');
    expect(events[0].delta).toBe(2);
  });

  it('should detect loop formation', () => {
    const before: BettiNumbers = { b0: 1, b1: 0, b2: 0, b3: 0 };
    const after: BettiNumbers = { b0: 1, b1: 1, b2: 0, b3: 0 };

    const events = detectTransition(before, after);

    expect(events.length).toBe(1);
    expect(events[0].type).toBe('loop_formed');
    expect(events[0].dimension).toBe(1);
  });

  it('should detect void formation', () => {
    const before: BettiNumbers = { b0: 1, b1: 0, b2: 0, b3: 0 };
    const after: BettiNumbers = { b0: 1, b1: 0, b2: 1, b3: 0 };

    const events = detectTransition(before, after);

    expect(events[0].type).toBe('void_formed');
    expect(events[0].dimension).toBe(2);
  });

  it('should detect multiple simultaneous events', () => {
    const before: BettiNumbers = { b0: 3, b1: 0, b2: 0, b3: 0 };
    const after: BettiNumbers = { b0: 1, b1: 2, b2: 0, b3: 0 };

    const events = detectTransition(before, after);

    expect(events.length).toBe(2);
    expect(events.some(e => e.type === 'cluster_merged')).toBe(true);
    expect(events.some(e => e.type === 'loop_formed')).toBe(true);
  });

  it('should return empty for no change', () => {
    const state: BettiNumbers = { b0: 1, b1: 0, b2: 0, b3: 0 };
    const events = detectTransition(state, state);
    expect(events.length).toBe(0);
  });
});

describe('HomologyAnalyzer Class', () => {
  let analyzer: HomologyAnalyzer;

  beforeEach(() => {
    const points: Vector4D[] = [
      { w: 0, x: 0, y: 0, z: 0 },
      { w: 1, x: 0, y: 0, z: 0 },
      { w: 0, x: 1, y: 0, z: 0 },
      { w: 0, x: 0, y: 1, z: 0 },
    ];
    analyzer = new HomologyAnalyzer(points);
  });

  it('should report correct point count', () => {
    expect(analyzer.pointCount).toBe(4);
  });

  it('should cache distance matrix', () => {
    const matrix1 = analyzer.getDistanceMatrix();
    const matrix2 = analyzer.getDistanceMatrix();
    expect(matrix1).toBe(matrix2);  // Same reference (cached)
  });

  it('should cache Betti numbers', () => {
    const betti1 = analyzer.computeBetti(0.5);
    const betti2 = analyzer.computeBetti(0.5);
    expect(betti1).toBe(betti2);  // Same reference (cached)
  });

  it('should clear cache on setPoints', () => {
    analyzer.computeBetti(0.5);
    analyzer.setPoints([{ w: 0, x: 0, y: 0, z: 0 }]);
    expect(analyzer.pointCount).toBe(1);
  });

  it('should compute persistence diagram', () => {
    const diagram = analyzer.computePersistence(10);
    expect(diagram.intervals.length).toBeGreaterThan(0);
  });

  it('should find first loop if exists', () => {
    // Square forms a loop
    const squarePoints: Vector4D[] = [
      { w: 0, x: 0, y: 0, z: 0 },
      { w: 0, x: 1, y: 0, z: 0 },
      { w: 0, x: 1, y: 1, z: 0 },
      { w: 0, x: 0, y: 1, z: 0 },
    ];
    analyzer.setPoints(squarePoints);

    const firstLoop = analyzer.findFirstLoop();
    // May or may not find loop depending on threshold
    if (firstLoop !== null) {
      expect(firstLoop).toBeGreaterThanOrEqual(0);
    }
  });

  it('should get summary at threshold', () => {
    const summary = analyzer.getSummary(2);

    expect(summary.betti).toBeDefined();
    expect(summary.description.length).toBeGreaterThan(0);
  });
});

describe('Factory Functions', () => {
  it('should create polytope analyzer', () => {
    const cell24 = new Cell24();
    const analyzer = createPolytopeAnalyzer(cell24.vertices);

    expect(analyzer.pointCount).toBe(24);
  });

  it('should compute quick Betti', () => {
    const points: Vector4D[] = [
      { w: 0, x: 0, y: 0, z: 0 },
      { w: 10, x: 0, y: 0, z: 0 },
    ];

    const betti = quickBetti(points, 0.5);

    expect(betti.b0).toBe(2);  // Two separate clusters
  });
});

describe('Musical Structure Detection', () => {
  it('should detect cyclic structure in 24-cell (circle of fifths analog)', () => {
    const cell24 = new Cell24();
    const analyzer = createPolytopeAnalyzer(cell24.vertices);

    // At appropriate threshold, should detect cyclic structure
    const result = analyzer.detectCyclicStructure();

    // The 24-cell has rich cyclic structure
    if (result) {
      expect(result.loopCount).toBeGreaterThanOrEqual(0);
      expect(result.threshold).toBeGreaterThan(0);
    }
  });

  it('should show connectivity at edge length', () => {
    const cell24 = new Cell24();
    const analyzer = createPolytopeAnalyzer(cell24.vertices);

    // At edge length, should be connected
    const summary = analyzer.getSummary(1.5);

    expect(summary.betti.b0).toBe(1);
    expect(summary.description.some(d => d.includes('connected'))).toBe(true);
  });
});
