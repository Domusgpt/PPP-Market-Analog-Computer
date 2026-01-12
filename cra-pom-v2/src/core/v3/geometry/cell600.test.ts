/**
 * 600-Cell Tests
 *
 * Verifies mathematical correctness of the 600-cell construction:
 * - 120 vertices (binary icosahedral group)
 * - 720 edges
 * - Edge length = 1/φ (for unit 600-cell)
 * - 12 neighbors per vertex
 * - Contains embedded 24-cell
 */

import { describe, it, expect } from 'vitest';
import { Cell600, verify600Cell, PHI, PHI_INV } from './cell600';

describe('600-Cell (Hexacosichoron)', () => {
  const cell = new Cell600();

  describe('Vertex Count', () => {
    it('should have exactly 120 vertices', () => {
      expect(cell.vertexCount).toBe(120);
      expect(cell.vertices.length).toBe(120);
    });

    it('should have all vertices at unit distance from origin', () => {
      for (const v of cell.vertices) {
        const mag = Math.sqrt(v.w ** 2 + v.x ** 2 + v.y ** 2 + v.z ** 2);
        expect(mag).toBeCloseTo(1, 4);
      }
    });

    it('should have no duplicate vertices', () => {
      const seen = new Set<string>();
      for (const v of cell.vertices) {
        const key = `${v.w.toFixed(4)},${v.x.toFixed(4)},${v.y.toFixed(4)},${v.z.toFixed(4)}`;
        expect(seen.has(key)).toBe(false);
        seen.add(key);
      }
    });
  });

  describe('Edge Properties', () => {
    it('should have 720 edges', () => {
      expect(cell.edgeCount).toBe(720);
      expect(cell.edges.length).toBe(720);
    });

    it('should have edge length equal to 1/φ', () => {
      expect(cell.edgeLength).toBeCloseTo(PHI_INV, 5);
    });

    it('should have all edges at correct length', () => {
      // Check first 20 edges
      for (const [i, j] of cell.edges.slice(0, 20)) {
        const a = cell.vertices[i];
        const b = cell.vertices[j];
        const dist = Math.sqrt(
          (a.w - b.w) ** 2 + (a.x - b.x) ** 2 +
          (a.y - b.y) ** 2 + (a.z - b.z) ** 2
        );
        expect(dist).toBeCloseTo(PHI_INV, 3);
      }
    });
  });

  describe('Vertex Connectivity', () => {
    it('should have 12 neighbors per vertex', () => {
      // Check first 10 vertices
      for (let i = 0; i < 10; i++) {
        const neighbors = cell.getNeighbors(i);
        expect(neighbors.length).toBe(12);
      }
    });

    it('should have symmetric connectivity', () => {
      // If A is neighbor of B, then B is neighbor of A
      for (const [i, j] of cell.edges.slice(0, 50)) {
        const neighborsOfI = cell.getNeighbors(i);
        const neighborsOfJ = cell.getNeighbors(j);
        expect(neighborsOfI).toContain(j);
        expect(neighborsOfJ).toContain(i);
      }
    });
  });

  describe('Golden Ratio Properties', () => {
    it('should use golden ratio correctly', () => {
      expect(PHI).toBeCloseTo(1.618033988749895, 10);
      expect(PHI_INV).toBeCloseTo(0.618033988749895, 10);
      expect(PHI * PHI_INV).toBeCloseTo(1, 10);
      expect(PHI - PHI_INV).toBeCloseTo(1, 10);
    });

    it('should have vertices at golden ratio positions', () => {
      // Some vertices should have φ/2 or 1/(2φ) coordinates
      let hasGoldenCoords = false;
      const halfPhi = PHI / 2;
      const halfPhiInv = PHI_INV / 2;

      for (const v of cell.vertices) {
        const coords = [Math.abs(v.w), Math.abs(v.x), Math.abs(v.y), Math.abs(v.z)];
        if (coords.some(c => Math.abs(c - halfPhi) < 0.001 || Math.abs(c - halfPhiInv) < 0.001)) {
          hasGoldenCoords = true;
          break;
        }
      }

      expect(hasGoldenCoords).toBe(true);
    });
  });

  describe('Embedded 24-Cell', () => {
    it('should contain a 24-cell as first 24 vertices', () => {
      expect(cell.embedded24Cell.length).toBe(24);
    });

    it('should have embedded 24-cell vertices on unit sphere', () => {
      for (const v of cell.embedded24Cell) {
        const mag = Math.sqrt(v.w ** 2 + v.x ** 2 + v.y ** 2 + v.z ** 2);
        expect(mag).toBeCloseTo(1, 4);
      }
    });

    it('should return 24-cell subset for index 0', () => {
      const subset = cell.get24CellSubset(0);
      expect(subset.length).toBe(24);
    });
  });

  describe('Point Operations', () => {
    it('should find closest vertex to a point', () => {
      const point = { w: 1, x: 0, y: 0, z: 0 };
      const { index, distance } = cell.findClosestVertex(point);

      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThan(120);
      expect(distance).toBeCloseTo(0, 3); // Should be very close to a vertex
    });

    it('should correctly identify points on polytope', () => {
      // First vertex should be on the polytope
      expect(cell.isOnPolytope(cell.vertices[0])).toBe(true);

      // Origin should not be on the polytope
      expect(cell.isOnPolytope({ w: 0, x: 0, y: 0, z: 0 })).toBe(false);
    });

    it('should interpolate between vertices', () => {
      const midpoint = cell.interpolateEdge(0, 1, 0.5);
      expect(midpoint.w).toBeDefined();
      expect(midpoint.x).toBeDefined();
      expect(midpoint.y).toBeDefined();
      expect(midpoint.z).toBeDefined();
    });
  });

  describe('Verification Function', () => {
    it('should verify construction correctly', () => {
      const result = verify600Cell();

      expect(result.vertexCount).toBe(120);
      expect(result.edgeCount).toBe(720);
      expect(result.neighborsPerVertex).toBe(12);
      expect(result.contains24Cell).toBe(true);
      expect(result.allUnitLength).toBe(true);

      console.log('600-cell verification:');
      result.details.forEach(d => console.log(`  ${d}`));
    });
  });

  describe('Geometric Properties', () => {
    it('should have correct face and cell counts', () => {
      expect(cell.faceCount).toBe(1200);
      expect(cell.cellCount).toBe(600);
    });

    it('should be 5x larger than 24-cell in vertex count', () => {
      expect(cell.vertexCount / 24).toBe(5);
    });
  });
});

describe('600-Cell vs 24-Cell Relationship', () => {
  it('should have 600-cell containing 24-cell vertices', () => {
    const cell600 = new Cell600();

    // The first 24 vertices should include axis-aligned points
    const axisPoints = cell600.vertices.filter(v => {
      const coords = [v.w, v.x, v.y, v.z];
      const nonZero = coords.filter(c => Math.abs(c) > 0.001);
      return nonZero.length === 1 && Math.abs(Math.abs(nonZero[0]) - 1) < 0.001;
    });

    // Should have 8 axis points (like 16-cell)
    expect(axisPoints.length).toBe(8);
  });

  it('should have 600-cell with more vertices for finer granularity', () => {
    const cell600 = new Cell600();
    // 120 vertices vs 24 = 5x finer resolution
    expect(cell600.vertexCount).toBe(120);
  });
});
